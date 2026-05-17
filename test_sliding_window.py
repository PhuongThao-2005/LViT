# test_sliding_window.py — Phase 2A
# Inference sliding window 224×224 trên ảnh gốc full size
# Dùng model đã train (best checkpoint từ Phase 1)
# Text giữ nguyên global text, feed cho mọi patch
#
# Cách dùng:
#   python test_sliding_window.py
#   (Sau khi set test_session và model_name đúng trong Config.py)

import os
import sys
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


# ─── Cấu hình Sliding Window ─────────────────────────────────────────────────
WIN_SIZE = 224          # kích thước window — giữ nguyên 224 để dùng model cũ
STRIDE   = 112          # stride 50% overlap — giảm boundary artifact
THRESHOLD = 0.5         # ngưỡng binary mask
USE_GAUSSIAN_WEIGHT = True   # True = Gaussian weighting, False = average đơn giản
# ─────────────────────────────────────────────────────────────────────────────


def make_gaussian_weight(win_size: int, sigma_ratio: float = 0.25) -> np.ndarray:
    """
    Tạo weight map hình chuông Gaussian: trung tâm = 1.0, biên ≈ 0.
    sigma_ratio: sigma = win_size * sigma_ratio
    """
    sigma = win_size * sigma_ratio
    cx = cy = win_size // 2
    y, x = np.ogrid[:win_size, :win_size]
    w = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return w.astype(np.float32)


def sliding_window_predict(model, image_bgr: np.ndarray, text_tensor: torch.Tensor,
                            win_size: int = WIN_SIZE, stride: int = STRIDE,
                            device: torch.device = torch.device('cuda'),
                            use_gaussian: bool = USE_GAUSSIAN_WEIGHT) -> np.ndarray:
    """
    image_bgr : ảnh gốc full size (H×W×3, BGR, uint8)
    text_tensor: (1, max_tokens, 768) — BERT embedding của text annotation
    Trả về: mask (H×W, float32, 0~1)
    """
    H, W = image_bgr.shape[:2]
    pred_map  = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    weight_map = make_gaussian_weight(win_size) if use_gaussian else np.ones((win_size, win_size), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        # Sinh danh sách tọa độ top-left của các window
        ys = list(range(0, H - win_size, stride))
        if not ys or ys[-1] + win_size < H:
            ys.append(max(0, H - win_size))   # đảm bảo cover đến cuối

        xs = list(range(0, W - win_size, stride))
        if not xs or xs[-1] + win_size < W:
            xs.append(max(0, W - win_size))

        text_t = text_tensor.to(device)       # (1, max_tokens, 768)
        for y in ys:
            for x in xs:
                # 1. Crop patch
                patch = image_bgr[y:y+win_size, x:x+win_size]   # (224,224,3) BGR

                # 2. Preprocess — normalize giống ValGenerator
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                # Normalize mean/std ImageNet (giống LViT training)
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                patch_norm = (patch_rgb - mean) / std
                patch_t = torch.from_numpy(patch_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

                # 3. Forward pass — feed global text
                output = model(patch_t, text_t)       # (1, 1, 224, 224)
                prob = output.squeeze().cpu().numpy()  # (224, 224), float 0~1

                # 4. Accumulate với weight
                pred_map[y:y+win_size, x:x+win_size]  += prob * weight_map
                count_map[y:y+win_size, x:x+win_size] += weight_map

    # Normalize
    final = pred_map / np.maximum(count_map, 1e-6)
    return final   # float32, 0~1


def compute_dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """Tính Dice coefficient trên binary mask."""
    p = pred_bin.astype(np.float32).reshape(-1)
    g = gt_bin.astype(np.float32).reshape(-1)
    return float(2 * np.sum(p * g) / (np.sum(p) + np.sum(g) + 1e-5))


def compute_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    return float(jaccard_score(gt_bin.reshape(-1), pred_bin.reshape(-1), zero_division=0))


def load_model(device: torch.device):
    _mn = str(config.model_name).lower().replace('-', '_')

    if _mn in ('lvit', 'lvit_pretrain'):
        cfg_vit = config.get_CTranS_config()
        model = LViT(cfg_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    elif _mn == 'lvit_tw':
        cfg_vit = config.get_LViT_TW_config()
        model = LViT(cfg_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    elif _mn == 'unet':
        model = UNet(n_channels=config.n_channels, n_classes=config.n_labels)
    else:
        raise ValueError(f"Unknown model_name: {config.model_name}")

    # ── Build path từ test_session (không dùng save_path) ──────────────────
    test_session = getattr(config, 'test_session', None)
    if test_session is None:
        raise ValueError("Config.py thiếu test_session. Hãy thêm: test_session = 'DEBUG_...'")

    ckpt_dir = os.path.join(config.base_save_dir, config.task_name,
                            config.model_name, test_session, 'models')
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pth.tar')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ckpt_dir, 'latest.pth.tar')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint tại: {ckpt_path}\n"
            f"test_session hiện tại: {test_session}\n"
            f"base_save_dir: {config.base_save_dir}"
        )
    # ───────────────────────────────────────────────────────────────────────

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_key = "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
    model.load_state_dict(checkpoint[state_key], strict=False)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model : {config.model_name}")
    print(f"Window: {WIN_SIZE}×{WIN_SIZE}, stride={STRIDE}, gaussian={USE_GAUSSIAN_WEIGHT}")

    # Load model
    model = load_model(device)

    # Load text annotations cho test set
    test_text_path = os.path.join(config.test_dataset, 'Test_text.xlsx')
    if not os.path.exists(test_text_path):
        # Fallback: tạo dict với default text
        label_dir = os.path.join(config.test_dataset, 'labelcol')
        test_text = {f: 'chest xray lesion segmentation EOF XXX EOF XXX EOF XXX EOF XXX'
                     for f in os.listdir(label_dir)}
        print(f"Test_text.xlsx not found. Using default text for {len(test_text)} samples.")
    else:
        test_text = read_text(test_text_path)

    # Text embedder — giống Load_Dataset.py
    embedder = HFTextEmbedder(model_name="bert-base-uncased", max_tokens=10)

    # Tìm test images
    img_dir  = os.path.join(config.test_dataset, 'img')
    mask_dir = os.path.join(config.test_dataset, 'labelcol')

    image_files = sorted([f for f in os.listdir(img_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Test samples: {len(image_files)}")

    # Output dir
    out_dir = os.path.join(config.save_path, 'sliding_window_test')
    os.makedirs(out_dir, exist_ok=True)

    dice_list = []
    iou_list  = []

    with tqdm(image_files, desc='Sliding Window Inference', ncols=80) as pbar:
        for img_fn in pbar:
            stem = os.path.splitext(img_fn)[0]

            # Load ảnh GỐC — KHÔNG resize
            img_path = os.path.join(img_dir, img_fn)
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                print(f"Cannot read {img_path}, skip.")
                continue
            H_orig, W_orig = image_bgr.shape[:2]

            # Load mask GỐC — KHÔNG resize
            mask_fn = None
            for ext in (img_fn, stem+'.png', stem+'.jpg'):
                cand = os.path.join(mask_dir, ext)
                if os.path.exists(cand):
                    mask_fn = cand
                    break
            if mask_fn is None:
                print(f"No mask for {img_fn}, skip.")
                continue

            mask_orig = cv2.imread(mask_fn, 0)
            mask_orig = cv2.resize(mask_orig, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
            gt_bin = (mask_orig > 0).astype(np.uint8)

            # Load & encode text (dùng mask filename làm key, giống Load_Dataset.py)
            mask_basename = os.path.basename(mask_fn)
            text_str = test_text.get(mask_basename, test_text.get(img_fn,
                        'chest xray lesion segmentation EOF XXX EOF XXX EOF XXX EOF XXX'))
            text_emb = embedder.encode(text_str)                      # (max_tokens, 768)
            text_t   = torch.from_numpy(text_emb).unsqueeze(0).float()  # (1, max_tokens, 768)

            # Sliding window predict
            prob_map = sliding_window_predict(model, image_bgr, text_t, device=device)

            # Binary threshold
            pred_bin = (prob_map > THRESHOLD).astype(np.uint8)

            # Tính Dice và IoU trên full size
            dice = compute_dice(pred_bin, gt_bin)
            iou  = compute_iou(pred_bin, gt_bin)
            dice_list.append(dice)
            iou_list.append(iou)

            # Save predicted mask
            save_path = os.path.join(out_dir, stem + '_sw_pred.png')
            cv2.imwrite(save_path, pred_bin * 255)

            pbar.set_postfix({'dice': f'{dice:.4f}', 'iou': f'{iou:.4f}'})

    # Kết quả tổng hợp
    mean_dice = np.mean(dice_list)
    mean_iou  = np.mean(iou_list)

    print()
    print("=" * 50)
    print(f"  Phase 2A — Sliding Window Inference Results")
    print(f"  Model   : {config.model_name}")
    print(f"  Window  : {WIN_SIZE}×{WIN_SIZE}, stride={STRIDE}")
    print(f"  Samples : {len(dice_list)}")
    print(f"  Mean Dice: {mean_dice:.4f} ({mean_dice*100:.2f}%)")
    print(f"  Mean IoU : {mean_iou:.4f} ({mean_iou*100:.2f}%)")
    print("=" * 50)
    print()
    print(" Comparison with Phase 1 (resize 224×224):")
    print("   Phase 1 best val Dice: 0.6856  (epoch 189)")
    print(f"  Phase 2A SW Dice     : {mean_dice:.4f}")
    print(f"  Δ Dice               : {(mean_dice - 0.6856):+.4f}")

    # Save kết quả ra file txt
    result_txt = os.path.join(out_dir, 'results.txt')
    with open(result_txt, 'w') as f:
        f.write(f"Phase 2A Sliding Window Results\n")
        f.write(f"Model: {config.model_name}\n")
        f.write(f"Win={WIN_SIZE}, Stride={STRIDE}, Gaussian={USE_GAUSSIAN_WEIGHT}\n")
        f.write(f"Samples: {len(dice_list)}\n")
        f.write(f"Mean Dice: {mean_dice:.4f}\n")
        f.write(f"Mean IoU : {mean_iou:.4f}\n")
        f.write("\nPer-sample results:\n")
        for fn, d, i in zip(image_files[:len(dice_list)], dice_list, iou_list):
            f.write(f"  {fn}: Dice={d:.4f}, IoU={i:.4f}\n")
    print(f"\n Results saved to: {result_txt}")


if __name__ == '__main__':
    main()
