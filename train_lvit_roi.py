# -*- coding: utf-8 -*-
"""
train_lvit_roi.py — Train LViT fine segmentation trên ROI patches

Pipeline:
  1. Build ROIPatchDataset từ GT mask bbox (per component) + jitter
  2. Train LViT (WeightedDiceBCE loss, Adam optimizer)
  3. Lưu checkpoint giống train_model.py: models/roi/latest.pth.tar + best_model.pth.tar

Metrics:
  Each epoch: Dice_patch224 on ROI crops (fast).
  Every roi_full_eval_every (default 10): same pipeline as inference_roi — coarse UNet → ROI(s)
  → LViT (current training weights) → paste full image → Dice_full / IoU_full on val subset.
  Requires config.coarse_session and coarse checkpoint. Set roi_full_eval_every = 0 to skip.

Cách chạy:
  python train_lvit_roi.py

Config.py (optional):
  coarse_session — required for periodic C2F val (same as inference_roi)
  roi_padding, roi_jitter, roi_patch_size, lvit_pretrained_path
  roi_num_workers, roi_full_eval_every, roi_full_eval_max_samples
"""

import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import jaccard_score

import Config as config
from nets.LViT import LViT
from utils import (
    CosineAnnealingWarmRestarts,
    WeightedDiceBCE,
    read_text,
)

from roi_dataset import ROIPatchDataset, roi_collate_fn, imread_bgr, imread_gray, align_mask_to_image

logger: logging.Logger = None  # type: ignore


def setup_logger(log_path: str) -> logging.Logger:
    """Giống train_model.logger_config: một logger, không propagate lên root."""
    _logger = logging.getLogger('train_roi')
    if _logger.hasHandlers():
        _logger.handlers.clear()
    _logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(message)s')
    fh = logging.FileHandler(log_path, encoding='UTF-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    _logger.addHandler(fh)
    _logger.addHandler(ch)
    _logger.propagate = False
    return _logger


def save_checkpoint(state: dict, save_path: str, filename: str) -> str:
    """Giống train_model.save_checkpoint."""
    logger.info('Saving checkpoint: %s', filename)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    out_path = os.path.join(save_path, filename)
    torch.save(state, out_path)
    return out_path


def list_val_image_pairs(img_dir: str, mask_dir: str):
    """Một entry / ảnh val (không tách theo component)."""
    pairs = []
    img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    for fn in sorted(os.listdir(img_dir)):
        if not fn.lower().endswith(img_extensions):
            continue
        stem     = os.path.splitext(fn)[0]
        img_path = os.path.join(img_dir, fn)
        mask_path = None
        for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
            cand = os.path.join(mask_dir, stem + ext)
            if os.path.exists(cand):
                mask_path = cand
                break
        if mask_path is None:
            continue
        pairs.append((img_path, mask_path, stem, fn))
    return pairs


def _text_tensor_from_dict(img_fn: str, mask_path: str, text_dict: dict, device: torch.device):
    for key in (os.path.basename(mask_path),
                  os.path.basename(img_fn),
                  os.path.splitext(os.path.basename(img_fn))[0]):
        if key in text_dict:
            emb = text_dict[key]
            if not isinstance(emb, np.ndarray):
                emb = np.asarray(emb, dtype=np.float32)
            else:
                emb = emb.astype(np.float32)
            return torch.from_numpy(emb).unsqueeze(0).to(device)
    return torch.zeros(1, 10, 768, device=device, dtype=torch.float32)


def _dice_full_image(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    p = pred_bin.reshape(-1).astype(np.float64)
    g = gt_bin.reshape(-1).astype(np.float64)
    return float(2.0 * np.dot(p, g) / (p.sum() + g.sum() + 1e-6))


def _iou_full_image(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    return float(
        jaccard_score(gt_bin.reshape(-1), pred_bin.reshape(-1), zero_division=0),
    )


def run_periodic_c2f_eval(
    coarse_model: nn.Module,
    fine_model: nn.Module,
    val_pairs: list,
    text_dict: dict,
    device: torch.device,
    max_samples: int,
    epoch: int,
    writer: SummaryWriter,
) -> tuple:
    """
    Coarse UNet → ROI(s) → LViT (weights đang train) → paste full → Dice/IoU.
    Dùng cùng logic với inference_roi (merge_rois_to_full, THRESHOLD, ROI_FALLBACK).
    """
    import inference_roi as ir

    if not val_pairs:
        return float('nan'), float('nan'), 0

    subset = val_pairs
    if max_samples and len(subset) > max_samples:
        subset = subset[:max_samples]

    was_training = fine_model.training
    fine_model.eval()
    coarse_model.eval()

    dice_list, iou_list = [], []
    for img_path, mask_path, _stem, img_fn in tqdm(
        subset,
        desc='C2F val',
        leave=False,
        ncols=90,
    ):
        try:
            image_bgr = imread_bgr(img_path)
            mask_raw  = imread_gray(mask_path)
            if image_bgr is None or mask_raw is None:
                continue
            H, W = image_bgr.shape[:2]
            mask_raw = align_mask_to_image(mask_raw, H, W)
            gt_bin   = (mask_raw > 0).astype(np.uint8)

            text_t   = _text_tensor_from_dict(img_fn, mask_path, text_dict, device)
            roi_boxes = ir.detect_rois(coarse_model, image_bgr, device)

            if roi_boxes is None:
                if ir.ROI_FALLBACK:
                    roi_boxes = [(0, 0, W, H)]
                    prob_full = ir.merge_rois_to_full(
                        fine_model, image_bgr, roi_boxes, text_t, device, H, W,
                    )
                    pred_bin = (prob_full > ir.THRESHOLD).astype(np.uint8)
                else:
                    pred_bin = np.zeros((H, W), dtype=np.uint8)
            else:
                prob_full = ir.merge_rois_to_full(
                    fine_model, image_bgr, roi_boxes, text_t, device, H, W,
                )
                pred_bin = (prob_full > ir.THRESHOLD).astype(np.uint8)

            dice_list.append(_dice_full_image(pred_bin, gt_bin))
            iou_list.append(_iou_full_image(pred_bin, gt_bin))
        except Exception as e:
            logger.warning('C2F val skip %s: %s', img_fn, e)

    if was_training:
        fine_model.train()

    if not dice_list:
        return float('nan'), float('nan'), 0

    mean_d = float(np.mean(dice_list))
    mean_i = float(np.mean(iou_list))
    if writer:
        writer.add_scalar('Dice_c2f_full/Val', mean_d, epoch)
        writer.add_scalar('IoU_c2f_full/Val',  mean_i, epoch)
    return mean_d, mean_i, len(dice_list)


def run_epoch(loader, model, criterion, optimizer, writer,
              epoch: int, is_train: bool, lr_scheduler=None):
    """
    Returns: (mean_loss, mean_dice_patch224)
    Dice là trung bình batch trên tensor 224×224 (ROI patch), không phải full ảnh.
    """
    model.train(is_train)
    phase = 'Train' if is_train else 'Val'

    total_loss = 0.0
    total_dice = 0.0
    n_batches  = 0

    for batch in loader:
        images = batch['image'].cuda(non_blocking=True)
        masks  = batch['mask'].cuda(non_blocking=True)
        texts  = batch['text'].cuda(non_blocking=True)

        with torch.set_grad_enabled(is_train):
            preds = model(images, texts)
            loss  = criterion(preds, masks)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            prob     = preds if config.n_labels == 1 else torch.sigmoid(preds)
            pred_bin = (prob > 0.5).float()
            inter      = (pred_bin * masks).sum(dim=(1, 2, 3))
            union      = pred_bin.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice_batch = ((2.0 * inter + 1e-6) / (union + 1e-6)).mean().item()

        total_loss += loss.item()
        total_dice += dice_batch
        n_batches  += 1

    if is_train and lr_scheduler is not None:
        lr_scheduler.step()

    mean_loss = total_loss / max(n_batches, 1)
    mean_dice = total_dice / max(n_batches, 1)

    if writer:
        writer.add_scalar(f'Loss/{phase}', mean_loss, epoch)
        writer.add_scalar(f'Dice_patch224/{phase}', mean_dice, epoch)

    logger.info(
        '\t [{}] Epoch {} Loss={:.4f} Dice_patch224={:.4f}'.format(
            phase, epoch + 1, mean_loss, mean_dice,
        ),
    )
    return mean_loss, mean_dice


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_text_dict(split: str) -> dict:
    try:
        from Load_Dataset import HFTextEmbedder
        embedder     = HFTextEmbedder(model_name="bert-base-uncased", max_tokens=10)
        use_embedder = True
    except (ImportError, Exception) as e:
        logger.warning('HFTextEmbedder không load được (%s), dùng zero embedding.', e)
        use_embedder = False

    if config.task_name == 'MoNuSeg':
        excel_path = config.train_dataset if split == 'train' else config.val_dataset
        excel_file = os.path.join(
            excel_path,
            'Train_text.xlsx' if split == 'train' else 'Val_text.xlsx',
        )
    elif config.task_name == 'Covid19':
        excel_file = os.path.join(config.task_dataset, 'Train_Val_text.xlsx')
    else:
        base = config.train_dataset if split == 'train' else config.val_dataset
        excel_file = os.path.join(base, f'{split.capitalize()}_text.xlsx')

    raw_text = {}
    if excel_file and os.path.exists(excel_file):
        raw_text = read_text(excel_file)
    else:
        logger.warning('Text Excel không tìm thấy: %s — dùng empty text.', excel_file)

    if not use_embedder:
        return {k: np.zeros((10, 768), dtype=np.float32) for k in raw_text}

    text_dict = {}
    for fn, txt in raw_text.items():
        text_dict[fn] = embedder.encode(txt)
    return text_dict


def main():
    set_seed(config.seed)

    roi_padding    = getattr(config, 'roi_padding',    80)
    roi_jitter     = getattr(config, 'roi_jitter',     40)
    roi_patch_size = getattr(config, 'roi_patch_size', 224)
    n_workers      = getattr(
        config, 'roi_num_workers',
        min(4, max(1, (os.cpu_count() or 4))),
    )

    logger.info('=' * 65)
    logger.info('Train LViT — ROI Patch Mode')
    logger.info('  Task       : %s', config.task_name)
    logger.info('  Model      : %s', config.model_name)
    logger.info('  Patch size : %s', roi_patch_size)
    logger.info('  ROI padding: %s  jitter: ±%s', roi_padding, roi_jitter)
    logger.info('  Epochs     : %s', config.epochs)
    logger.info('  Batch size : %s', config.batch_size)
    logger.info('  DataLoader workers: %s', n_workers)
    logger.info('=' * 65)
    logger.info(
        'Validation Dice_patch224 = ROI crops 224². Every %s epochs: C2F val = coarse UNet → '
        'ROI → LViT (current weights) → paste → full-image Dice/IoU (same as inference_roi).',
        getattr(config, 'roi_full_eval_every', 10),
    )
    logger.info('Save dir: %s', config.save_path)

    logger.info('Building text embeddings...')
    train_text_dict = build_text_dict('train')
    val_text_dict   = build_text_dict('val')
    logger.info('  Train text keys: %s', len(train_text_dict))
    logger.info('  Val   text keys: %s', len(val_text_dict))

    val_img_dir   = os.path.join(config.val_dataset, 'img')
    val_mask_dir  = os.path.join(config.val_dataset, 'labelcol')
    val_image_pairs = list_val_image_pairs(val_img_dir, val_mask_dir)
    logger.info('  Val images for C2F eval: %s', len(val_image_pairs))

    c2f_eval_every = getattr(config, 'roi_full_eval_every', 10)
    c2f_max_samples = getattr(config, 'roi_full_eval_max_samples', 32)
    logger.info(
        '  C2F full-image eval: every %s epochs | max_samples=%s (0=all) | needs coarse_session',
        c2f_eval_every,
        c2f_max_samples,
    )

    train_dataset = ROIPatchDataset(
        img_dir    = os.path.join(config.train_dataset, 'img'),
        mask_dir   = os.path.join(config.train_dataset, 'labelcol'),
        text_dict  = train_text_dict,
        patch_size = roi_patch_size,
        padding    = roi_padding,
        jitter     = roi_jitter,
        augment    = True,
        split      = 'train',
    )
    val_dataset = ROIPatchDataset(
        img_dir    = os.path.join(config.val_dataset, 'img'),
        mask_dir   = os.path.join(config.val_dataset, 'labelcol'),
        text_dict  = val_text_dict,
        patch_size = roi_patch_size,
        padding    = roi_padding,
        jitter     = 0,
        augment    = False,
        split      = 'val',
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size      = config.batch_size,
        shuffle         = True,
        num_workers     = n_workers,
        pin_memory      = True,
        worker_init_fn  = worker_init_fn,
        collate_fn      = roi_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size      = config.batch_size,
        shuffle         = False,
        num_workers     = n_workers,
        pin_memory      = True,
        worker_init_fn  = worker_init_fn,
        collate_fn      = roi_collate_fn,
    )

    config_vit = config.get_CTranS_config()
    model = LViT(
        config_vit,
        n_channels=config.n_channels,
        n_classes=config.n_labels,
    )

    pretrained_path = getattr(config, 'lvit_pretrained_path', None)
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info('Loading pretrained LViT from: %s', pretrained_path)
        ck          = torch.load(pretrained_path, map_location='cpu')
        state       = ck.get('state_dict', ck)
        model_state = model.state_dict()
        compatible  = {
            k: v for k, v in state.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        model_state.update(compatible)
        model.load_state_dict(model_state)
        logger.info('  Loaded %s/%s layers from pretrained.', len(compatible), len(model_state))
    else:
        logger.info('Training from scratch (no pretrained checkpoint).')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        logger.info('Using %d GPUs (DataParallel).', torch.cuda.device_count())
        model = nn.DataParallel(model)

    device = next(model.parameters()).device

    coarse_model = None
    if c2f_eval_every > 0:
        try:
            import inference_roi as ir
            coarse_model = ir.load_coarse_unet(device)
            coarse_model.eval()
            for p in coarse_model.parameters():
                p.requires_grad = False
            logger.info('Coarse UNet loaded for periodic C2F validation.')
        except Exception as e:
            logger.warning('Periodic C2F val disabled (coarse UNet): %s', e)
            c2f_eval_every = 0

    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
    )
    lr_scheduler = (
        CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
        if config.cosineLR else None
    )

    tb_dir = os.path.join(config.tensorboard_folder, 'roi')
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    ckpt_dir = os.path.join(config.model_path, 'roi')
    os.makedirs(ckpt_dir, exist_ok=True)

    max_dice = 0.0
    best_epoch = 1
    early_stopping_count = 0
    model_type = config.model_name + '_roi'

    for epoch in range(config.epochs):
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs))
        logger.info('%s', config.session_name)

        run_epoch(
            train_loader, model, criterion, optimizer,
            writer, epoch, is_train=True, lr_scheduler=lr_scheduler,
        )

        logger.info('Validation')
        with torch.no_grad():
            val_loss, val_dice = run_epoch(
                val_loader, model, criterion, optimizer,
                writer, epoch, is_train=False,
            )

        if (
            c2f_eval_every > 0
            and coarse_model is not None
            and (epoch + 1) % c2f_eval_every == 0
            and val_image_pairs
        ):
            mean_fd, mean_fi, n_full = run_periodic_c2f_eval(
                coarse_model,
                model,
                val_image_pairs,
                val_text_dict,
                device,
                c2f_max_samples,
                epoch,
                writer,
            )
            if n_full > 0:
                sub_note = (
                    'all val images' if not c2f_max_samples else f'first {c2f_max_samples} images'
                )
                logger.info(
                    '\t [C2F Val] epoch %s  n=%s (%s)  Dice=%.4f  IoU=%.4f',
                    epoch + 1,
                    n_full,
                    sub_note,
                    mean_fd,
                    mean_fi,
                )

        if val_dice > max_dice - 1e-4:
            logger.info(
                '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(
                    max_dice, val_dice,
                ),
            )
            max_dice = val_dice
            best_epoch = epoch + 1
            early_stopping_count = 0
        else:
            logger.info(
                '\t Mean dice:{:.4f} does not increase, '
                'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch),
            )
            early_stopping_count += 1

        state = {
            'epoch':                epoch,
            'model':                model_type,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'val_loss':             val_loss,
            'best_dice':            max_dice,
            'early_stopping_count': early_stopping_count,
            'best_epoch':           best_epoch,
            'roi_padding':          roi_padding,
            'roi_patch_size':       roi_patch_size,
        }
        save_checkpoint(state, ckpt_dir, 'latest.pth.tar')
        if val_dice >= max_dice - 1e-4 and early_stopping_count == 0:
            save_checkpoint(state, ckpt_dir, 'best_model.pth.tar')

        save_freq = getattr(config, 'save_frequency', 0)
        if save_freq > 0 and ((epoch + 1) % save_freq == 0):
            save_checkpoint(state, ckpt_dir, 'epoch-{:04d}.pth.tar'.format(epoch + 1))

        logger.info(
            '\t early_stopping_count: {}/{}'.format(
                early_stopping_count, config.early_stopping_patience,
            ),
        )

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early stopping!')
            break

    writer.close()
    logger.info('\nDone. Best val Dice_patch224: %.4f @ epoch %s', max_dice, best_epoch)
    logger.info('Checkpoint dir: %s', ckpt_dir)


if __name__ == '__main__':
    cudnn.benchmark     = False
    cudnn.deterministic = True

    os.makedirs(config.save_path, exist_ok=True)

    log_path = os.path.join(
        config.save_path,
        f'{config.session_name}_roi_train.log',
    )
    logger = setup_logger(log_path)

    main()
