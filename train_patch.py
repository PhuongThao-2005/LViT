import os, json, sys, warnings, csv
warnings.filterwarnings("ignore")

import cv2, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import jaccard_score

import Config as config
from nets.LViT import LViT
from patch_dataset import PatchDataset
from LV_loss.loss import BinaryDiceLoss
from Load_Dataset import HFTextEmbedder
from utils import read_text

sys.path.insert(0, os.path.dirname(__file__))
from test_sliding_window import sliding_window_predict


# ── Full-image validation ─────────────────────────────────────────────────────
def validate_full_image(model, val_dataset_path, val_text, embedder, device,
                         max_images: int = None):
    """
    Validation đúng research setting: full image → sliding inference → Dice.
    max_images: giới hạn số ảnh (None = dùng hết, số nhỏ = nhanh hơn)
    """
    img_dir  = os.path.join(val_dataset_path, 'img')
    mask_dir = os.path.join(val_dataset_path, 'labelcol')
    img_files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith(('.png','.jpg','.jpeg'))])

    if max_images:
        # Lấy subset cố định (không random) để kết quả comparable qua các epoch
        step = max(1, len(img_files) // max_images)
        img_files = img_files[::step][:max_images]

    dice_list, iou_list   = [], []
    pred_ratio_list       = []   # để detect model collapse
    model.eval()

    with torch.no_grad():
        for img_fn in tqdm(img_files, desc='Val (full-img)', ncols=70, leave=False):
            stem = os.path.splitext(img_fn)[0]
            image_bgr = cv2.imread(os.path.join(img_dir, img_fn))
            if image_bgr is None: continue

            H, W = image_bgr.shape[:2]
            mask_fn = None
            for ext in (img_fn, stem+'.png', stem+'.jpg'):
                cand = os.path.join(mask_dir, ext)
                if os.path.exists(cand): mask_fn = cand; break
            if mask_fn is None: continue

            mask_orig = cv2.imread(mask_fn, 0)
            mask_orig = cv2.resize(mask_orig, (W, H), interpolation=cv2.INTER_NEAREST)
            gt_bin    = (mask_orig > 0).astype(np.uint8)

            mask_basename = os.path.basename(mask_fn)
            text_str = val_text.get(mask_basename, val_text.get(img_fn, 'tumor lesion segmentation'))
            text_emb = embedder.encode(text_str)
            text_t   = torch.from_numpy(text_emb).unsqueeze(0).float()

            prob_map = sliding_window_predict(model, image_bgr, text_t, device=device)
            pred_bin = (prob_map > 0.5).astype(np.uint8)

            p = pred_bin.reshape(-1).astype(np.float32)
            g = gt_bin.reshape(-1).astype(np.float32)
            dice = float(2 * np.sum(p * g) / (np.sum(p) + np.sum(g) + 1e-5))
            iou  = float(jaccard_score(g, p, zero_division=0))

            dice_list.append(dice)
            iou_list.append(iou)
            pred_ratio_list.append(float(pred_bin.mean()))

    mean_pred_ratio = float(np.mean(pred_ratio_list)) if pred_ratio_list else 0.0
    if mean_pred_ratio < 0.001:
        print(f"WARNING: model collapse! mean pred_ratio={mean_pred_ratio:.5f} ≈ 0")

    return (float(np.mean(dice_list)),
            float(np.mean(iou_list)),
            mean_pred_ratio)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'   # AMP chỉ dùng trên GPU
    print(f"Device: {device} | AMP: {use_amp} | Session: {config.session_name}")

    os.makedirs(config.model_path, exist_ok=True)
    os.makedirs(config.tensorboard_folder, exist_ok=True)

    # Save training config ngay khi bắt đầu
    train_cfg = {
        "session": config.session_name, "task_name": config.task_name,
        "win_size": 224, "stride": 112,
        "neg_ratio": PatchDataset.NEG_RATIO, "min_pos": PatchDataset.MIN_POS,
        "batch_size": config.batch_size, "lr": config.learning_rate,
        "epochs": config.epochs, "seed": config.seed, "amp": use_amp
    }
    with open(os.path.join(config.save_path, 'train_config.json'), 'w') as f:
        json.dump(train_cfg, f, indent=2)

    # Text & embedder
    train_text = read_text(os.path.join(config.train_dataset, 'Train_text.xlsx'))
    val_text   = read_text(os.path.join(config.val_dataset,   'Val_text.xlsx'))
    embedder   = HFTextEmbedder(model_name="bert-base-uncased", max_tokens=10)

    # Train dataset
    cache_path    = os.path.join(config.save_path, 'train_patch_metadata.json')
    train_dataset = PatchDataset(
        config.train_dataset, train_text,
        augment=True, seed=config.seed, cache_json=cache_path
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=2, pin_memory=use_amp
    )

    # Model + optimizer + AMP scaler
    model     = LViT(config.get_CTranS_config(),
                     n_channels=config.n_channels,
                     n_classes=config.n_labels).to(device)
    criterion = BinaryDiceLoss()
    bce_loss  = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    scaler    = GradScaler(enabled=use_amp)

    best_dice      = 0.0
    patience_count = 0
    log_rows       = []

    # Val schedule:
    # epoch 1        → val đầu tiên để kiểm tra pipeline
    # epoch 1–30     → val mỗi 10 epoch (nhanh, subset 30 ảnh)
    # epoch 31+      → val mỗi 10 epoch (full val set)
    WARMUP_EPOCHS   = 30
    VAL_SUBSET_SIZE = 30   # số ảnh val trong warmup

    for epoch in range(1, config.epochs + 1):

        # ── TRAIN ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Ep{epoch:03d}", ncols=72, leave=False):
            imgs  = batch['image'].to(device)
            masks = batch['label'].to(device)
            texts = batch['text'].to(device)

            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                preds = model(imgs, texts)
                loss  = criterion(preds, masks) + bce_loss(preds, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # ── VALIDATION ─────────────────────────────────────────────────────
        do_val    = (epoch == 1) or (epoch % 10 == 0)
        subset    = VAL_SUBSET_SIZE if epoch <= WARMUP_EPOCHS else None

        if do_val:
            val_dice, val_iou, pred_ratio = validate_full_image(
                model, config.val_dataset, val_text, embedder, device,
                max_images=subset
            )
            subset_tag = f"(subset {subset})" if subset else "(full)"
            print(f"Ep{epoch:03d} | loss={train_loss:.4f} | "
                  f"val_dice={val_dice:.4f} | val_iou={val_iou:.4f} | "
                  f"pred_ratio={pred_ratio:.4f} {subset_tag}")

            log_rows.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                             "val_dice": round(val_dice, 6),
                             "val_iou": round(val_iou, 6),
                             "pred_ratio": round(pred_ratio, 6),
                             "val_subset": subset or "full"})

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({'model_state_dict': model.state_dict(),
                            'epoch': epoch, 'val_dice': val_dice},
                           os.path.join(config.model_path, 'best_model.pth.tar'))
                print(f"Best model! Dice={best_dice:.4f}")
                patience_count = 0
            else:
                patience_count += 10

            if patience_count >= config.early_stopping_patience:
                print(f"Early stopping tại epoch {epoch}")
                break
        else:
            print(f"Ep{epoch:03d} | loss={train_loss:.4f}")

        # Checkpoint định kỳ
        if epoch % config.save_frequency == 0:
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch},
                       os.path.join(config.model_path, f'epoch-{epoch:04d}.pth.tar'))

    # Save log CSV
    log_path = os.path.join(config.save_path, 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader(); writer.writerows(log_rows)
    print(f"\nBest Val Dice: {best_dice:.4f} | Log → {log_path}")


if __name__ == '__main__':
    main()