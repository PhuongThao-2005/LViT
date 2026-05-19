# -*- coding: utf-8 -*-
"""
inference_roi.py — Coarse-to-Fine Inference Pipeline

Flow:
  full image → Coarse UNet (512) → bbox ROI
             → crop ROI → LViT fine segment (224×224 patch)
             → paste prediction back → full-res binary mask

Cách dùng:
  python inference_roi.py

Config.py cần có:
  coarse_session = '...'   # UNet coarse checkpoint session
  test_session   = '...'   # LViT fine checkpoint session (ROI-trained)
  model_name     = 'LViT'

Kết quả lưu vào: config.save_path / roi_inference / {preds, vis, results.txt}
"""

import os
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import jaccard_score

import Config as config
from nets.LViT import LViT
from nets.UNet import UNet
from roi_dataset import align_mask_to_image

# ─── Hyper-params ─────────────────────────────────────────────────────────────
COARSE_INPUT_SIZE = 512     # UNet coarse input size
COARSE_THRESHOLD  = 0.3     # thấp để không bỏ sót lesion nhỏ
ROI_PADDING       = getattr(config, 'roi_padding', 80)
ROI_MIN_AREA_PCT  = 0.001   # bỏ coarse mask nếu diện tích < 0.1%
ROI_MAX_FRAC      = 0.25    # bỏ blob quá lớn (nhiễu nền)
ROI_FALLBACK      = True    # fallback toàn ảnh nếu coarse không detect được
MAX_ROIS          = 10      # tối đa ROI / ảnh (multi-tumor)

FINE_INPUT_SIZE   = 224     # LViT patch size
FINE_STRIDE       = 112     # tiling khi ROI lớn (ảnh ~2000px)
USE_GAUSSIAN      = True
THRESHOLD         = 0.5     # threshold cho final prediction

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_image(image_bgr: np.ndarray, target_size: int) -> torch.Tensor:
    """BGR → normalized tensor (1, 3, H, W) — pad-aware resize giữ aspect ratio."""
    H, W   = image_bgr.shape[:2]
    scale  = target_size / max(H, W)
    new_h  = int(H * scale)
    new_w  = int(W * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h))

    pad_b  = target_size - new_h
    pad_r  = target_size - new_w
    padded = cv2.copyMakeBorder(resized, 0, pad_b, 0, pad_r,
                                cv2.BORDER_CONSTANT, value=0)

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()

def preprocess_patch(patch_bgr: np.ndarray) -> torch.Tensor:
    """Crop patch → tensor (1, 3, H, W)."""
    rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()

# ─── Model loaders ────────────────────────────────────────────────────────────

def load_coarse_unet(device: torch.device) -> torch.nn.Module:
    """Load Coarse UNet từ config.coarse_session."""
    coarse_session = getattr(config, 'coarse_session', None)
    if coarse_session is None:
        raise ValueError(
            "Config.py thiếu coarse_session.\n"
            "Thêm: coarse_session = '<session_name>'"
        )

    ckpt_dir  = os.path.join(config.base_save_dir, config.task_name,
                              'UNet', coarse_session, 'models')
    ckpt_path = os.path.join(ckpt_dir, 'best_model-UNet.pth.tar')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, 'best_model.pth.tar')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Coarse UNet checkpoint không tìm thấy: {ckpt_dir}")

    # [FIX-I1] UNet coarse dùng n_classes=1 (binary lesion detector)
    model = UNet(n_channels=config.n_channels, n_classes=1)
    ck    = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck.get('state_dict', ck.get('model_state_dict')), strict=False)
    model = model.to(device).eval()
    print(f"[Coarse UNet] {ckpt_path}")
    return model

def load_fine_lvit(device: torch.device) -> torch.nn.Module:
    """Load LViT ROI-fine model từ config.test_session (roi subdir)."""
    test_session = getattr(config, 'test_session', None)
    if test_session is None:
        raise ValueError("Config.py thiếu test_session.")

    ckpt_dir = os.path.join(config.base_save_dir, config.task_name,
                            config.model_name, test_session, 'models', 'roi')
    ckpt_path = None
    for name in (
        'best_model.pth.tar',
        f'best_model-{config.model_name}_roi.pth.tar',
        f'best_model-{config.model_name}.pth.tar',
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
        raise FileNotFoundError(f"Fine LViT checkpoint không tìm thấy trong: {ckpt_dir}")

    cfg_vit = config.get_CTranS_config()
    model   = LViT(cfg_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    ck      = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck.get('state_dict', ck.get('model_state_dict')), strict=False)
    model = model.to(device).eval()
    print(f"[Fine LViT] {ckpt_path}")
    return model

# ─── Coarse ROI detection ─────────────────────────────────────────────────────

def _coarse_prob_map(coarse_model, image_bgr, device):
    """Coarse UNet → probability map full resolution."""
    H, W   = image_bgr.shape[:2]
    tensor = preprocess_image(image_bgr, COARSE_INPUT_SIZE).to(device)
    scale  = COARSE_INPUT_SIZE / max(H, W)
    new_h  = int(H * scale)
    new_w  = int(W * scale)

    with torch.no_grad():
        out  = coarse_model(tensor)
        prob = out.squeeze().cpu().numpy()

    prob_unpad = prob[:new_h, :new_w]
    return cv2.resize(prob_unpad, (W, H), interpolation=cv2.INTER_LINEAR)

def _ensure_min_roi_size(x1, y1, x2, y2, H, W, min_size):
    if x2 - x1 < min_size:
        cx = (x1 + x2) // 2
        x1 = max(0, cx - min_size // 2)
        x2 = min(W, x1 + min_size)
        x1 = max(0, x2 - min_size)
    if y2 - y1 < min_size:
        cy = (y1 + y2) // 2
        y1 = max(0, cy - min_size // 2)
        y2 = min(H, y1 + min_size)
        y1 = max(0, y2 - min_size)
    return x1, y1, x2, y2

def _box_from_xywh(x, y, w, h, H, W, padding_px):
    x1 = max(0, x - padding_px)
    y1 = max(0, y - padding_px)
    x2 = min(W, x + w + padding_px)
    y2 = min(H, y + h + padding_px)
    return _ensure_min_roi_size(x1, y1, x2, y2, H, W, FINE_INPUT_SIZE)

def detect_rois(coarse_model: torch.nn.Module,
                image_bgr: np.ndarray,
                device: torch.device):
    """
    Coarse UNet → danh sách bbox (x1,y1,x2,y2), mỗi connected component một ROI.
    Trả None nếu không detect được.
    """
    H, W = image_bgr.shape[:2]
    prob_orig = _coarse_prob_map(coarse_model, image_bgr, device)

    coarse_bin = (prob_orig > COARSE_THRESHOLD).astype(np.uint8)
    if coarse_bin.sum() / (H * W) < ROI_MIN_AREA_PCT:
        return None

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    coarse_bin = cv2.morphologyEx(coarse_bin, cv2.MORPH_OPEN,  k_open,  iterations=1)
    coarse_bin = cv2.morphologyEx(coarse_bin, cv2.MORPH_CLOSE, k_close, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(coarse_bin, connectivity=8)
    candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < H * W * ROI_MIN_AREA_PCT:
            continue
        if area / (H * W) > ROI_MAX_FRAC:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        mask_i    = labels == i
        mean_conf = prob_orig[mask_i].mean()
        candidates.append((mean_conf, area, x, y, w, h))

    if not candidates:
        return None

    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    rois = []
    for _, _, x, y, w, h in candidates[:MAX_ROIS]:
        rois.append(_box_from_xywh(x, y, w, h, H, W, ROI_PADDING))
    return rois

# ─── Fine segmentation ────────────────────────────────────────────────────────

def _fine_prob_numpy(pred: torch.Tensor) -> np.ndarray:
    """LViT n_classes==1 đã sigmoid; multi-class cần sigmoid thêm."""
    if config.n_labels == 1:
        return pred.squeeze().cpu().numpy()
    return torch.sigmoid(pred).squeeze().cpu().numpy()

def make_gaussian_weight(win_size: int, sigma_ratio: float = 0.25) -> np.ndarray:
    sigma = win_size * sigma_ratio
    cx = cy = win_size // 2
    y, x = np.ogrid[:win_size, :win_size]
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)

def _forward_tile(fine_model, roi_crop, py, px, text_t, device):
    """Resize tile to FINE_INPUT_SIZE before LViT; map prob back to actual tile size."""
    roi_h, roi_w = roi_crop.shape[:2]
    y2 = min(py + FINE_INPUT_SIZE, roi_h)
    x2 = min(px + FINE_INPUT_SIZE, roi_w)
    tile = roi_crop[py:y2, px:x2]
    th, tw = tile.shape[:2]

    tile_224 = tile if (th == FINE_INPUT_SIZE and tw == FINE_INPUT_SIZE) \
        else cv2.resize(tile, (FINE_INPUT_SIZE, FINE_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

    tensor = preprocess_patch(tile_224).to(device)
    with torch.no_grad():
        prob = _fine_prob_numpy(fine_model(tensor, text_t))

    if th != FINE_INPUT_SIZE or tw != FINE_INPUT_SIZE:
        prob = cv2.resize(prob, (tw, th), interpolation=cv2.INTER_LINEAR)
    return prob, th, tw

def segment_roi(fine_model: torch.nn.Module,
                image_bgr: np.ndarray,
                roi_box: tuple,
                text_tensor: torch.Tensor,
                device: torch.device) -> np.ndarray:
    """
    Fine LViT trên ROI. ROI nhỏ → 1 pass resize 224; ROI lớn → tiling + Gaussian blend.
    Returns: prob_map (H_roi, W_roi) float [0,1]
    """
    x1, y1, x2, y2 = roi_box
    roi_crop     = image_bgr[y1:y2, x1:x2]
    roi_h, roi_w = roi_crop.shape[:2]
    text_t       = text_tensor.to(device)

    weight_map = make_gaussian_weight(FINE_INPUT_SIZE) if USE_GAUSSIAN \
        else np.ones((FINE_INPUT_SIZE, FINE_INPUT_SIZE), np.float32)

    if roi_h <= FINE_INPUT_SIZE and roi_w <= FINE_INPUT_SIZE:
        patch  = cv2.resize(roi_crop, (FINE_INPUT_SIZE, FINE_INPUT_SIZE))
        tensor = preprocess_patch(patch).to(device)
        with torch.no_grad():
            prob_224 = _fine_prob_numpy(fine_model(tensor, text_t))
        return cv2.resize(prob_224, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

    pred_map  = np.zeros((roi_h, roi_w), np.float32)
    count_map = np.zeros((roi_h, roi_w), np.float32)

    ys = list(range(0, max(1, roi_h - FINE_INPUT_SIZE), FINE_STRIDE))
    if not ys or ys[-1] + FINE_INPUT_SIZE < roi_h:
        ys.append(max(0, roi_h - FINE_INPUT_SIZE))
    xs = list(range(0, max(1, roi_w - FINE_INPUT_SIZE), FINE_STRIDE))
    if not xs or xs[-1] + FINE_INPUT_SIZE < roi_w:
        xs.append(max(0, roi_w - FINE_INPUT_SIZE))

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

def merge_rois_to_full(fine_model: torch.nn.Module,
                       image_bgr: np.ndarray,
                       roi_boxes: list,
                       text_tensor: torch.Tensor,
                       device: torch.device,
                       H: int, W: int) -> np.ndarray:
    """Chạy fine model trên từng ROI, gộp prob map bằng element-wise max."""
    full_map = np.zeros((H, W), dtype=np.float32)
    for roi_box in roi_boxes:
        prob_roi = segment_roi(fine_model, image_bgr, roi_box, text_tensor, device)
        x1, y1, x2, y2 = roi_box
        sl = full_map[y1:y2, x1:x2]
        full_map[y1:y2, x1:x2] = np.maximum(sl, prob_roi)
    return full_map

# ─── Metrics ─────────────────────────────────────────────────────────────────

def dice_score(pred_bin, gt_bin):
    inter = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum()
    return float(2 * inter / (union + 1e-6))

def iou_score(pred_bin, gt_bin):
    return float(jaccard_score(gt_bin.ravel(), pred_bin.ravel(), zero_division=0))

# ─── Visualization ────────────────────────────────────────────────────────────

def save_vis(image_bgr, pred_bin, gt_bin, roi_boxes, save_path,
             dice, iou, used_fallback):
    p1 = image_bgr.copy()
    boxes = roi_boxes if isinstance(roi_boxes, list) else ([roi_boxes] if roi_boxes else [])
    color = (0, 165, 255) if used_fallback else (0, 255, 0)
    for i, box in enumerate(boxes):
        cv2.rectangle(p1, box[:2], box[2:], color, 3)
    if boxes:
        cv2.putText(p1, 'fallback' if used_fallback else f'{len(boxes)} ROI(s)',
                    (boxes[0][0] + 4, boxes[0][1] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    p2  = cv2.cvtColor((gt_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    p3  = image_bgr.copy()
    ov  = np.zeros_like(p3); ov[pred_bin == 1] = (0, 0, 200)
    p3  = cv2.addWeighted(p3, 0.6, ov, 0.4, 0)
    cv2.putText(p3, f"Dice:{dice:.3f} IoU:{iou:.3f}",
                (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    th = 512
    def _r(p):
        s = th / p.shape[0]
        return cv2.resize(p, (int(p.shape[1] * s), th))

    cv2.imwrite(save_path, np.hstack([_r(p1), _r(p2), _r(p3)]))

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")
    print(f"Task    : {config.task_name}")
    print(f"Model   : {config.model_name} (session={getattr(config,'test_session','?')})")
    print(f"Coarse  : UNet (session={getattr(config,'coarse_session','?')})")

    # ── Load models ───────────────────────────────────────────────────────────
    coarse_model = load_coarse_unet(device)
    fine_model   = load_fine_lvit(device)

    # ── Text embedder ─────────────────────────────────────────────────────────
    try:
        from Load_Dataset import HFTextEmbedder
        from utils import read_text
        embedder   = HFTextEmbedder(model_name="bert-base-uncased", max_tokens=10)
        text_excel = os.path.join(config.test_dataset, 'Test_text.xlsx')
        raw_text   = read_text(text_excel) if os.path.exists(text_excel) else {}
    except Exception as e:
        print(f"[warn] Text embedder error: {e}. Using zero embeddings.")
        embedder = None
        raw_text = {}

    DEFAULT_TEXT = 'bone tumor xray lesion segmentation EOF XXX EOF XXX EOF XXX EOF XXX'

    def get_text_tensor(img_fn, mask_fn):
        if embedder is None:
            return torch.zeros(1, 10, 768)
        txt = (raw_text.get(os.path.basename(mask_fn))
               or raw_text.get(img_fn)
               or DEFAULT_TEXT)
        emb = embedder.encode(txt)                    # (10, 768) np.ndarray
        return torch.from_numpy(emb).unsqueeze(0).float()

    # ── IO dirs ───────────────────────────────────────────────────────────────
    img_dir   = os.path.join(config.test_dataset, 'img')
    mask_dir  = os.path.join(config.test_dataset, 'labelcol')
    out_root  = os.path.join(config.save_path, 'roi_inference')
    pred_dir  = os.path.join(out_root, 'preds')
    vis_dir   = os.path.join(out_root, 'vis')
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(vis_dir,  exist_ok=True)

    img_files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Test samples: {len(img_files)}\n")

    dice_list, iou_list, fallback_imgs = [], [], []
    processed_files = []

    with tqdm(img_files, desc='Coarse→Fine', ncols=90) as pbar:
        for img_fn in pbar:
            stem      = os.path.splitext(img_fn)[0]
            image_bgr = cv2.imread(os.path.join(img_dir, img_fn))
            if image_bgr is None:
                continue
            H, W = image_bgr.shape[:2]

            mask_path = None
            for ext in (img_fn, stem + '.png', stem + '.jpg'):
                cand = os.path.join(mask_dir, ext)
                if os.path.exists(cand):
                    mask_path = cand
                    break
            if mask_path is None:
                continue

            mask_orig = cv2.imread(mask_path, 0)
            mask_orig = align_mask_to_image(mask_orig, H, W)
            gt_bin    = (mask_orig > 0).astype(np.uint8)

            text_t = get_text_tensor(img_fn, mask_path)

            roi_boxes     = detect_rois(coarse_model, image_bgr, device)
            used_fallback = False

            if roi_boxes is None:
                if ROI_FALLBACK:
                    roi_boxes     = [(0, 0, W, H)]
                    used_fallback = True
                    fallback_imgs.append(img_fn)
                else:
                    pred_bin = np.zeros((H, W), dtype=np.uint8)
                    d = dice_score(pred_bin, gt_bin)
                    i = iou_score(pred_bin, gt_bin)
                    dice_list.append(d)
                    iou_list.append(i)
                    processed_files.append(img_fn)
                    continue

            prob_full = merge_rois_to_full(
                fine_model, image_bgr, roi_boxes, text_t, device, H, W
            )
            pred_bin = (prob_full > THRESHOLD).astype(np.uint8)

            d = dice_score(pred_bin, gt_bin)
            i = iou_score(pred_bin, gt_bin)
            dice_list.append(d)
            iou_list.append(i)
            processed_files.append(img_fn)

            cv2.imwrite(os.path.join(pred_dir, stem + '_pred.png'), pred_bin * 255)
            save_vis(image_bgr, pred_bin, gt_bin, roi_boxes,
                     os.path.join(vis_dir, stem + '_vis.jpg'),
                     d, i, used_fallback)

            pbar.set_postfix(dice=f'{d:.4f}', rois=len(roi_boxes),
                             fb='Y' if used_fallback else 'N')

    # ── Summary ───────────────────────────────────────────────────────────────
    mean_dice = float(np.mean(dice_list)) if dice_list else 0.0
    mean_iou  = float(np.mean(iou_list))  if iou_list  else 0.0

    print(f"\n{'='*60}")
    print(f"  Coarse-to-Fine ROI Inference Results")
    print(f"  Samples   : {len(dice_list)}")
    print(f"  Mean Dice : {mean_dice:.4f} ({mean_dice*100:.2f}%)")
    print(f"  Mean IoU  : {mean_iou:.4f}  ({mean_iou*100:.2f}%)")
    print(f"  Fallback  : {len(fallback_imgs)} images")
    print(f"{'='*60}")

    result_path = os.path.join(out_root, 'results.txt')
    with open(result_path, 'w') as f:
        f.write("Coarse-to-Fine ROI Inference\n")
        f.write(f"Fine   : {config.model_name} ({getattr(config,'test_session','?')})\n")
        f.write(f"Coarse : UNet ({getattr(config,'coarse_session','?')})\n")
        f.write(f"CoarseSize={COARSE_INPUT_SIZE} Threshold={COARSE_THRESHOLD} Padding={ROI_PADDING}\n")
        f.write(f"FineSize={FINE_INPUT_SIZE} Threshold={THRESHOLD}\n")
        f.write(f"Samples    : {len(dice_list)}\n")
        f.write(f"Mean Dice  : {mean_dice:.4f}\n")
        f.write(f"Mean IoU   : {mean_iou:.4f}\n")
        f.write(f"Fallback   : {len(fallback_imgs)}\n")
        if fallback_imgs:
            f.write("Fallback images: " + ", ".join(fallback_imgs) + "\n")
        f.write("\nPer-sample:\n")
        for fn, d, i in zip(processed_files, dice_list, iou_list):
            fb = '*' if fn in fallback_imgs else ' '
            f.write(f"  {fb}{fn}: Dice={d:.4f}, IoU={i:.4f}\n")

    print(f"\n  Results : {result_path}")
    print(f"  Preds   : {pred_dir}")
    print(f"  Vis     : {vis_dir}")


if __name__ == '__main__':
    main()
