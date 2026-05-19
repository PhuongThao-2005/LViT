# test_roi_based.py — Phase 2C: Coarse-to-Fine ROI Inference (fixed)
#
# Fixes so với version cũ:
# [F1] Double sigmoid bug — UNet n_classes==1 đã sigmoid bên trong, KHÔNG sigmoid thêm
# [F2] fine_model dispatch — UNet không nhận text, LViT mới nhận
# [F3] text_t shape guard — đảm bảo [1, 10, 768] trước khi feed LViT
# [F4] Missing closing parenthesis trong main() — result(), save_visualization()
# [F5] UNet coarse nên dùng n_classes=1 (binary detector), không dùng config.n_labels

import os
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import jaccard_score

import Config as config
from nets.LViT import LViT
from nets.UNet import UNet
from Load_Dataset import HFTextEmbedder
from utils import read_text
from roi_dataset import align_mask_to_image

# ─── Cấu hình ────────────────────────────────────────────────────────────────
COARSE_SIZE       = 512
COARSE_THRESHOLD  = 0.3

ROI_PADDING       = getattr(config, 'roi_padding', 80)
ROI_MIN_SIZE      = 224
ROI_MIN_AREA_PCT  = 0.001
ROI_MAX_FRAC      = 0.25
MAX_ROIS          = 10

WIN_SIZE          = 224
STRIDE            = 112
FINE_THRESHOLD    = 0.5
USE_GAUSSIAN      = True
# ─────────────────────────────────────────────────────────────────────────────

def _is_lvit(model: torch.nn.Module) -> bool:
    return isinstance(model, LViT)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — COARSE UNET
# ═══════════════════════════════════════════════════════════════════════════════

def load_coarse_model(device: torch.device) -> torch.nn.Module:
    coarse_session = getattr(config, 'coarse_session', None)
    if coarse_session is None:
        raise ValueError("Config.py thiếu coarse_session.")

    ckpt_dir  = os.path.join(getattr(config, 'base_save_dir', '.'),
                              config.task_name, 'UNet', coarse_session, 'models')
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pth.tar')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, 'latest.pth.tar')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Coarse UNet checkpoint không tìm thấy: {ckpt_path}")

    # [F5] Dùng n_classes=1 cho coarse UNet (binary detector)
    model = UNet(n_channels=config.n_channels, n_classes=1)
    ck    = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck.get('model_state_dict', ck.get('state_dict')), strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[Coarse] Loaded: {ckpt_path}")
    return model

def coarse_predict(coarse_model: torch.nn.Module,
                   image_bgr: np.ndarray,
                   device: torch.device) -> np.ndarray:
    H, W   = image_bgr.shape[:2]
    scale  = COARSE_SIZE / max(H, W)
    new_h, new_w = int(H * scale), int(W * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h))

    pad_h  = COARSE_SIZE - new_h
    pad_w  = COARSE_SIZE - new_w
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w,
                                cv2.BORDER_CONSTANT, value=0)

    rgb  = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std  = np.array([0.229, 0.224, 0.225], np.float32)
    t    = torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1)) \
               .unsqueeze(0).float().to(device)

    with torch.no_grad():
        out  = coarse_model(t)
        # [F1] UNet n_classes==1 đã sigmoid bên trong → KHÔNG sigmoid thêm
        prob = out.squeeze().cpu().numpy()   # đã là 0~1

    prob_unpad = prob[:new_h, :new_w]
    prob_orig  = cv2.resize(prob_unpad, (W, H), interpolation=cv2.INTER_LINEAR)
    return prob_orig

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — CROP ROI
# ═══════════════════════════════════════════════════════════════════════════════

def _expand_coarse_box(x, y, w, h, H, W, padding_px):
    x1 = max(0, x - padding_px)
    y1 = max(0, y - padding_px)
    x2 = min(W, x + w + padding_px)
    y2 = min(H, y + h + padding_px)

    for axis, (a1, a2, Max) in enumerate([(x1, x2, int(W * 0.7)), (y1, y2, int(H * 0.7))]):
        if a2 - a1 > Max:
            cx = (a1 + a2) // 2
            a1 = max(0, cx - Max // 2)
            a2 = min([W, H][axis], a1 + Max)
            if axis == 0:
                x1, x2 = a1, a2
            else:
                y1, y2 = a1, a2

    if x2 - x1 < ROI_MIN_SIZE:
        diff = ROI_MIN_SIZE - (x2 - x1)
        x1 = max(0, x1 - diff // 2)
        x2 = min(W, x1 + ROI_MIN_SIZE)
    if y2 - y1 < ROI_MIN_SIZE:
        diff = ROI_MIN_SIZE - (y2 - y1)
        y1 = max(0, y1 - diff // 2)
        y2 = min(H, y1 + ROI_MIN_SIZE)

    return (x1, y1, x2, y2)

def get_rois_from_coarse(coarse_prob: np.ndarray,
                         image_shape: tuple,
                         padding_px: int = ROI_PADDING):
    """Trả list bbox — mỗi coarse component một ROI (multi-tumor)."""
    H, W       = image_shape[:2]
    coarse_bin = (coarse_prob > COARSE_THRESHOLD).astype(np.uint8)

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    coarse_bin = cv2.morphologyEx(coarse_bin, cv2.MORPH_OPEN,  k_open,  iterations=1)
    coarse_bin = cv2.morphologyEx(coarse_bin, cv2.MORPH_CLOSE, k_close, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(coarse_bin, connectivity=8)

    candidates = []
    for i in range(1, num_labels):
        x    = stats[i, cv2.CC_STAT_LEFT]
        y    = stats[i, cv2.CC_STAT_TOP]
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < H * W * ROI_MIN_AREA_PCT:
            continue
        if area / (H * W) > ROI_MAX_FRAC:
            continue

        border_touch = (x <= 5 or y <= 5 or x + w >= W - 5 or y + h >= H - 5)
        mask         = labels == i
        mean_conf    = coarse_prob[mask].mean()
        max_conf     = coarse_prob[mask].max()
        fill_ratio   = area / (w * h + 1e-6)

        score = mean_conf * 0.5 + max_conf * 0.3 + fill_ratio * 0.2
        if border_touch:
            score *= 0.5

        candidates.append((score, area, x, y, w, h))

    if not candidates:
        return None

    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [
        _expand_coarse_box(x, y, w, h, H, W, padding_px)
        for _, _, x, y, w, h in candidates[:MAX_ROIS]
    ]

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — FINE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_fine_model(device: torch.device) -> torch.nn.Module:
    _mn = str(config.model_name).lower().replace('-', '_')
    if _mn in ('lvit', 'lvit_pretrain'):
        model = LViT(config.get_CTranS_config(),
                     n_channels=config.n_channels, n_classes=config.n_labels)
    elif _mn == 'unet':
        model = UNet(n_channels=config.n_channels, n_classes=config.n_labels)
    else:
        raise ValueError(f"Unknown model_name: {config.model_name}")

    test_session = getattr(config, 'test_session', None)
    if test_session is None:
        raise ValueError("Config.py thiếu test_session.")

    ckpt_dir  = os.path.join(getattr(config, 'base_save_dir', '.'),
                              config.task_name, config.model_name, test_session, 'models', 'roi')
    ckpt_path = None
    for name in (
        'best_model.pth.tar',
        f'best_model-{config.model_name}_roi.pth.tar',
    ):
        cand = os.path.join(ckpt_dir, name)
        if os.path.exists(cand):
            ckpt_path = cand
            break
    if ckpt_path is None:
        ckpt_path = os.path.normpath(
            os.path.join(ckpt_dir, '..', f'best_model-{config.model_name}.pth.tar')
        )
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(ckpt_dir), 'best_model.pth.tar')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Fine model checkpoint không tìm thấy: {ckpt_dir}")

    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck.get('model_state_dict', ck.get('state_dict')), strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[Fine] Loaded: {ckpt_path}")
    return model

def make_gaussian_weight(win_size: int, sigma_ratio: float = 0.25) -> np.ndarray:
    sigma = win_size * sigma_ratio
    cx = cy = win_size // 2
    y, x = np.ogrid[:win_size, :win_size]
    return np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2)).astype(np.float32)

MEAN_NP = np.array([0.485, 0.456, 0.406], np.float32)
STD_NP  = np.array([0.229, 0.224, 0.225], np.float32)

def _preprocess_patch(patch_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN_NP) / STD_NP).transpose(2, 0, 1)) \
               .unsqueeze(0).float().to(device)

def _forward_fine(fine_model: torch.nn.Module,
                  patch_t: torch.Tensor,
                  text_t: torch.Tensor) -> np.ndarray:
    """
    [F2] Dispatch đúng theo model type:
      - LViT → forward(img, text)
      - UNet → forward(img)  [không có text]
    Trả về prob map (H,W) numpy float32 đã là 0~1.
    """
    if _is_lvit(fine_model):
        # [F3] Đảm bảo text shape = [B, 10, 768]
        t = text_t
        if t.dim() == 2:                        # [10, 768] → [1, 10, 768]
            t = t.unsqueeze(0)
        if t.shape[0] != patch_t.shape[0]:
            t = t.expand(patch_t.shape[0], -1, -1)
        out = fine_model(patch_t, t)
    else:
        # UNet — không nhận text
        out = fine_model(patch_t)

    # UNet n_classes==1 đã sigmoid bên trong → prob đã là 0~1
    # LViT: kiểm tra last_activation — nếu chưa sigmoid thì uncomment dòng dưới
    # out = torch.sigmoid(out)
    return out.squeeze().cpu().numpy()

def _forward_tile(fine_model, roi_crop, py, px, text_t, device):
    """Extract tile, resize to WIN_SIZE for LViT (196 pos-embed), return prob at tile size."""
    roi_h, roi_w = roi_crop.shape[:2]
    y2 = min(py + WIN_SIZE, roi_h)
    x2 = min(px + WIN_SIZE, roi_w)
    tile = roi_crop[py:y2, px:x2]
    th, tw = tile.shape[:2]

    tile_224 = tile if (th == WIN_SIZE and tw == WIN_SIZE) \
        else cv2.resize(tile, (WIN_SIZE, WIN_SIZE), interpolation=cv2.INTER_LINEAR)

    prob = _forward_fine(fine_model, _preprocess_patch(tile_224, device), text_t)

    if th != WIN_SIZE or tw != WIN_SIZE:
        prob = cv2.resize(prob, (tw, th), interpolation=cv2.INTER_LINEAR)
    return prob, th, tw

def fine_predict_roi(fine_model: torch.nn.Module,
                     image_bgr: np.ndarray,
                     roi_box: tuple,
                     text_t: torch.Tensor,
                     device: torch.device) -> np.ndarray:
    x1, y1, x2, y2 = roi_box
    roi_crop        = image_bgr[y1:y2, x1:x2]
    roi_h, roi_w    = roi_crop.shape[:2]

    weight_map = make_gaussian_weight(WIN_SIZE) if USE_GAUSSIAN \
                 else np.ones((WIN_SIZE, WIN_SIZE), np.float32)

    # ROI vừa một tile 224×224
    if roi_h <= WIN_SIZE and roi_w <= WIN_SIZE:
        patch = cv2.resize(roi_crop, (WIN_SIZE, WIN_SIZE))
        with torch.no_grad():
            prob_224 = _forward_fine(fine_model, _preprocess_patch(patch, device), text_t)
        return cv2.resize(prob_224, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

    # ROI lớn: tiling — mỗi tile luôn feed LViT 224×224
    pred_map  = np.zeros((roi_h, roi_w), np.float32)
    count_map = np.zeros((roi_h, roi_w), np.float32)

    ys = list(range(0, max(1, roi_h - WIN_SIZE), STRIDE))
    if not ys or ys[-1] + WIN_SIZE < roi_h:
        ys.append(max(0, roi_h - WIN_SIZE))
    xs = list(range(0, max(1, roi_w - WIN_SIZE), STRIDE))
    if not xs or xs[-1] + WIN_SIZE < roi_w:
        xs.append(max(0, roi_w - WIN_SIZE))

    with torch.no_grad():
        for py in ys:
            for px in xs:
                prob, th, tw = _forward_tile(
                    fine_model, roi_crop, py, px, text_t, device
                )
                wgt = weight_map[:th, :tw]
                pred_map [py:py + th, px:px + tw] += prob * wgt
                count_map[py:py + th, px:px + tw] += wgt

    return pred_map / np.maximum(count_map, 1e-6)

def coarse_to_fine_predict(coarse_model, fine_model,
                            image_bgr: np.ndarray,
                            text_tensor: torch.Tensor,
                            device: torch.device) -> dict:
    H, W   = image_bgr.shape[:2]
    text_t = text_tensor.to(device)

    coarse_prob = coarse_predict(coarse_model, image_bgr, device)
    roi_boxes   = get_rois_from_coarse(coarse_prob, (H, W))

    if roi_boxes is None:
        resized = cv2.resize(image_bgr, (WIN_SIZE, WIN_SIZE))
        with torch.no_grad():
            prob_224 = _forward_fine(fine_model,
                                     _preprocess_patch(resized, device), text_t)
        fine_prob = cv2.resize(prob_224, (W, H), interpolation=cv2.INTER_LINEAR)
        return {'fine_prob': fine_prob, 'coarse_prob': coarse_prob,
                'roi_boxes': None, 'used_fallback': True}

    fine_prob = np.zeros((H, W), np.float32)
    for roi_box in roi_boxes:
        fine_prob_roi = fine_predict_roi(fine_model, image_bgr, roi_box, text_t, device)
        x1, y1, x2, y2 = roi_box
        roi_h, roi_w = y2 - y1, x2 - x1
        fine_prob_roi = cv2.resize(fine_prob_roi, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
        sl = fine_prob[y1:y2, x1:x2]
        fine_prob[y1:y2, x1:x2] = np.maximum(sl, fine_prob_roi)

    return {'fine_prob': fine_prob, 'coarse_prob': coarse_prob,
            'roi_boxes': roi_boxes, 'used_fallback': False}

# ═══════════════════════════════════════════════════════════════════════════════
# METRICS & VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dice(pred_bin, gt_bin):
    p, g = pred_bin.reshape(-1).astype(np.float32), gt_bin.reshape(-1).astype(np.float32)
    return float(2 * np.sum(p*g) / (np.sum(p) + np.sum(g) + 1e-5))

def compute_iou(pred_bin, gt_bin):
    return float(jaccard_score(gt_bin.reshape(-1), pred_bin.reshape(-1), zero_division=0))

def save_visualization(image_bgr, coarse_prob, fine_prob, gt_bin,
                       roi_boxes, save_path, dice, iou):
    H, W = image_bgr.shape[:2]

    p1 = image_bgr.copy()
    boxes = roi_boxes if isinstance(roi_boxes, list) else ([roi_boxes] if roi_boxes else [])
    for box in boxes:
        cv2.rectangle(p1, (box[0], box[1]), (box[2], box[3]),
                      (0, 255, 0), max(2, H // 300))

    coarse_u8 = (coarse_prob * 255).clip(0, 255).astype(np.uint8)
    p2        = cv2.applyColorMap(coarse_u8, cv2.COLORMAP_JET)
    p3        = cv2.cvtColor((gt_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    fine_bin = (fine_prob > FINE_THRESHOLD).astype(np.uint8)
    p4       = image_bgr.copy()
    ov       = np.zeros_like(image_bgr); ov[fine_bin == 1] = (0, 0, 200)
    p4       = cv2.addWeighted(p4, 0.6, ov, 0.4, 0)
    cv2.putText(p4, f"Dice:{dice:.3f} IoU:{iou:.3f}",
                (10, max(30, H//30)), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.6, H/2000), (255, 255, 0), 2)

    target_h = 480
    def _r(p):
        s = target_h / p.shape[0]
        return cv2.resize(p, (int(p.shape[1]*s), target_h))

    panels = [_r(p) for p in [p1, p2, p3, p4]]
    for panel, label in zip(panels, ['Original+ROI', 'Coarse Mask', 'GT Mask', 'Fine Pred']):
        cv2.putText(panel, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    cv2.imwrite(save_path, np.hstack(panels))

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")
    print(f"Fine model: {config.model_name}")
    print()

    coarse_model = load_coarse_model(device)
    fine_model   = load_fine_model(device)

    # Text
    test_text_path = os.path.join(config.test_dataset, 'Test_text.xlsx')
    if os.path.exists(test_text_path):
        test_text = read_text(test_text_path)
    else:
        label_dir = os.path.join(config.test_dataset, 'labelcol')
        test_text = {f: 'bone tumor xray lesion segmentation EOF XXX EOF XXX EOF XXX EOF XXX'
                     for f in os.listdir(label_dir)}
        print("Test_text.xlsx not found — using default text.")

    embedder  = HFTextEmbedder(model_name="bert-base-uncased", max_tokens=10)
    img_dir   = os.path.join(config.test_dataset, 'img')
    mask_dir  = os.path.join(config.test_dataset, 'labelcol')
    img_files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Test samples: {len(img_files)}\n")

    out_dir  = os.path.join(config.save_path, 'c2f_test')
    vis_dir  = os.path.join(out_dir, 'vis')
    pred_dir = os.path.join(out_dir, 'preds')
    os.makedirs(vis_dir,  exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    dice_list, iou_list, fallback_list = [], [], []
    processed_files = []

    with tqdm(img_files, desc='Coarse→Fine', ncols=90) as pbar:
        for img_fn in pbar:
            stem      = os.path.splitext(img_fn)[0]
            image_bgr = cv2.imread(os.path.join(img_dir, img_fn))
            if image_bgr is None:
                continue
            H_orig, W_orig = image_bgr.shape[:2]

            mask_fn = None
            for ext in (img_fn, stem + '.png', stem + '.jpg'):
                cand = os.path.join(mask_dir, ext)
                if os.path.exists(cand):
                    mask_fn = cand
                    break
            if mask_fn is None:
                continue

            mask_orig = cv2.imread(mask_fn, 0)
            mask_orig = align_mask_to_image(mask_orig, H_orig, W_orig)
            gt_bin    = (mask_orig > 0).astype(np.uint8)

            text_str = test_text.get(os.path.basename(mask_fn),
                           test_text.get(img_fn,
                               'bone tumor xray lesion segmentation EOF XXX EOF XXX EOF XXX EOF XXX'))
            text_emb = embedder.encode(text_str)                          # (10, 768)
            text_t   = torch.from_numpy(text_emb).unsqueeze(0).float()   # [1, 10, 768]

            # [F4] Đóng ngoặc đúng
            result = coarse_to_fine_predict(
                coarse_model, fine_model, image_bgr, text_t, device
            )

            if result['used_fallback']:
                fallback_list.append(img_fn)

            pred_bin = (result['fine_prob'] > FINE_THRESHOLD).astype(np.uint8)
            dice     = compute_dice(pred_bin, gt_bin)
            iou      = compute_iou (pred_bin, gt_bin)
            dice_list.append(dice)
            iou_list.append(iou)
            processed_files.append(img_fn)

            cv2.imwrite(os.path.join(pred_dir, stem + '_c2f_pred.png'), pred_bin * 255)

            save_visualization(
                image_bgr, result['coarse_prob'], result['fine_prob'],
                gt_bin, result['roi_boxes'],
                os.path.join(vis_dir, stem + '_vis.jpg'),
                dice, iou
            )

            pbar.set_postfix(dice=f'{dice:.4f}', iou=f'{iou:.4f}',
                             fb='Y' if result['used_fallback'] else 'N')

    mean_dice = np.mean(dice_list)
    mean_iou  = np.mean(iou_list)

    print(f"\n{'='*60}")
    print(f"  Phase 2C — Coarse-to-Fine Results")
    print(f"  Samples   : {len(dice_list)}")
    print(f"  Mean Dice : {mean_dice:.4f} ({mean_dice*100:.2f}%)")
    print(f"  Mean IoU  : {mean_iou:.4f}  ({mean_iou*100:.2f}%)")
    print(f"  Fallback  : {len(fallback_list)} images")
    print(f"{'='*60}")

    result_txt = os.path.join(out_dir, 'results.txt')
    with open(result_txt, 'w') as f:
        f.write(f"Mean Dice: {mean_dice:.4f}\nMean IoU: {mean_iou:.4f}\n")
        f.write(f"Fallback : {len(fallback_list)}\n")
        f.write("\nPer-sample:\n")
        for fn, d, i in zip(processed_files, dice_list, iou_list):
            fb = '*' if fn in fallback_list else ' '
            f.write(f"  {fb}{fn}: Dice={d:.4f}, IoU={i:.4f}\n")

    print(f"\n  Results: {result_txt}")
    print(f"  Vis    : {vis_dir}")


if __name__ == '__main__':
    main()
