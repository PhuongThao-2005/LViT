# -*- coding: utf-8 -*-
import torch.optim
import torch.nn as nn
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
import Config
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D, LV2D, load_unlabeled_stems_from_labels_xlsx
from nets.LViT import LViT as LViT_base
from nets.EfficientLViT import LViT as EfficientLViT
from torch.utils.data import DataLoader
import logging
import csv
from Train_one_epoch import train_one_epoch, print_summary
import Config as config
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE, WeightedDiceCE, read_text, read_text_LV, save_on_batch
import argparse
import torch.cuda.amp as amp


logger = logging.getLogger('LViT')


def logger_config(log_path):
    logger = logging.getLogger('LViT')
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    logger.propagate = False
    return logger


def save_checkpoint(state, save_path, filename):
    logger.info("Saving checkpoint: %s", filename)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    out_path = os.path.join(save_path, filename)
    torch.save(state, out_path)
    return out_path


def find_latest_resume_checkpoint(explicit_path=None):
    if explicit_path:
        return explicit_path
    import glob
    latest_pattern = os.path.join(config.base_save_dir, "**", "models", "latest.pth.tar")
    cands = glob.glob(latest_pattern, recursive=True)
    if cands:
        cands.sort(key=os.path.getmtime, reverse=True)
        return cands[0]
    old_pattern = os.path.join(config.base_save_dir, "**", "models", "*.pth.tar")
    cands = glob.glob(old_pattern, recursive=True)
    if cands:
        cands.sort(key=os.path.getmtime, reverse=True)
        return cands[0]
    return None


def update_paths_for_resume(checkpoint_path):
    ckpt_abs = os.path.abspath(checkpoint_path)
    model_dir = os.path.dirname(ckpt_abs)
    save_dir = os.path.dirname(model_dir)
    session_name = os.path.basename(save_dir.rstrip("\\/"))
    config.session_name = session_name
    config.save_path = save_dir + os.sep
    config.model_path = os.path.join(config.save_path, "models") + os.sep
    config.tensorboard_folder = os.path.join(config.save_path, "tensorboard_logs") + os.sep
    config.logger_path = os.path.join(config.save_path, session_name + ".log")
    config.visualize_path = os.path.join(config.save_path, "visualize_val") + os.sep


def load_checkpoint(model, device, checkpoint_path):
    logger.info("Loading checkpoint from: %s", checkpoint_path)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    model.load_state_dict(ckpt[state_key], strict=False)
    logger.info("Checkpoint info:")
    logger.info("  - Epoch: %s", ckpt.get("epoch", "unknown"))
    logger.info("  - Best dice: %s", ckpt.get("best_dice", "unknown"))
    logger.info("  - Early stopping count: %s", ckpt.get("early_stopping_count", "unknown"))
    return model, ckpt


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


def dynamic_pad_collate(batch):
    """
    Collate fn cho variable-size images (khi resize_images=False).
    Pad tất cả ảnh/mask trong batch về cùng (max_H, max_W) bằng zero-padding.
    Text tokens đã cùng size nên stack bình thường.
    """
    import torch.nn.functional as TF
    samples = [b[0] for b in batch]
    names   = [b[1] for b in batch]

    images = [s['image'] for s in samples]
    labels = [s['label'] for s in samples]
    texts  = [s['text']  for s in samples]

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    images_pad = torch.stack([
        TF.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1]))
        for img in images
    ])

    labels_pad = []
    for lbl in labels:
        if lbl.dim() == 2:
            lbl_p = TF.pad(lbl.unsqueeze(0).float(),
                           (0, max_w - lbl.shape[1], 0, max_h - lbl.shape[0]))
            labels_pad.append(lbl_p.squeeze(0).long())
        else:
            lbl_p = TF.pad(lbl.float(),
                           (0, max_w - lbl.shape[2], 0, max_h - lbl.shape[1]))
            labels_pad.append(lbl_p.long())
    labels_pad = torch.stack(labels_pad)

    texts_pad = torch.stack(texts)
    return {'image': images_pad, 'label': labels_pad, 'text': texts_pad}, names


def load_text_or_default(dataset_dir, text_filename):
    text_path = os.path.join(dataset_dir, text_filename)
    if os.path.exists(text_path):
        try:
            return read_text(text_path)
        except Exception as exc:
            logger.warning('Cannot read %s (%s). Fallback to default prompts.', text_path, str(exc))
    label_dir = os.path.join(dataset_dir, 'labelcol')
    text = {}
    default_prompt = 'chest xray lesion segmentation EOF XXX EOF XXX EOF XXX EOF XXX'
    if os.path.isdir(label_dir):
        for mask_name in os.listdir(label_dir):
            text[mask_name] = default_prompt
    logger.warning('Text file not found: %s. Use default prompts for %d masks.', text_path, len(text))
    return text


def load_unlabeled_stems(plan_csv_path):
    unlabeled = set()
    if not plan_csv_path:
        return unlabeled
    if not os.path.exists(plan_csv_path):
        logger.warning('Label plan CSV not found: %s. Use all labels.', plan_csv_path)
        return unlabeled
    try:
        with open(plan_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = str(row.get("split", "")).lower()
                is_labeled = str(row.get("is_labeled", "")).lower()
                if split == "train" and is_labeled in ("false", "0", "no"):
                    image_id = str(row.get("image_id", ""))
                    if image_id:
                        unlabeled.add(os.path.splitext(image_id)[0])
        logger.info('Loaded %d unlabeled train samples from plan CSV.', len(unlabeled))
    except Exception as exc:
        logger.warning('Cannot read label plan CSV %s (%s). Use all labels.', plan_csv_path, str(exc))
    return unlabeled


##################################################################################
# =================================================================================
#          Main Loop
# =================================================================================
##################################################################################
def main_loop(model, batch_size=config.batch_size, model_type='', tensorboard=True, resume_ckpt=None, start_epoch=0):
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    logger.info("Training device (from model): %s", device)

    train_output_size = [config.img_size, config.img_size] if config.resize_images else None
    train_tf = transforms.Compose([RandomGenerator(output_size=train_output_size)])
    val_tf = ValGenerator(output_size=train_output_size)
    image_size = config.img_size if config.resize_images else None

    if not config.resize_images and not (config.use_efficient_lvit or config.model_name == 'EfficientLViT'):
        raise ValueError('resize_images=False is only supported when using EfficientLViT')

    unlabeled_train_stems = load_unlabeled_stems(getattr(config, "label_plan_csv", ""))
    train_labels_xlsx = os.path.join(config.train_dataset, "Train_labels.xlsx")
    if os.path.isfile(train_labels_xlsx):
        from_xlsx = load_unlabeled_stems_from_labels_xlsx(train_labels_xlsx)
        if from_xlsx:
            unlabeled_train_stems |= from_xlsx
            logger.info("Merged %d unlabeled train stems from labels xlsx: %s",
                        len(from_xlsx), train_labels_xlsx)

    val_labels_xlsx = os.path.join(config.val_dataset, "Val_labels.xlsx")
    val_unlabeled_stems = (
        load_unlabeled_stems_from_labels_xlsx(val_labels_xlsx)
        if os.path.isfile(val_labels_xlsx) else set()
    )
    if val_unlabeled_stems:
        logger.info("Val set: %d samples will use empty masks per Val_labels.xlsx",
                    len(val_unlabeled_stems))

    if config.task_name == 'MoNuSeg':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        val_text   = read_text(config.val_dataset   + 'Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=image_size, unlabeled_image_stems=unlabeled_train_stems)
        val_dataset   = ImageToImage2D(config.val_dataset,   config.task_name, val_text,   val_tf,
                                       image_size=image_size, unlabeled_image_stems=val_unlabeled_stems)
    elif config.task_name == 'Covid19':
        text = read_text(config.task_dataset + 'Train_Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, text, train_tf,
                                       image_size=image_size, unlabeled_image_stems=unlabeled_train_stems)
        val_dataset   = ImageToImage2D(config.val_dataset,   config.task_name, text, val_tf,
                                       image_size=image_size, unlabeled_image_stems=val_unlabeled_stems)
    elif str(config.task_name).startswith('BTRXD'):
        train_text = load_text_or_default(config.train_dataset, 'Train_text.xlsx')
        val_text   = load_text_or_default(config.val_dataset,   'Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=image_size, unlabeled_image_stems=unlabeled_train_stems)
        val_dataset   = ImageToImage2D(config.val_dataset,   config.task_name, val_text,   val_tf,
                                       image_size=image_size, unlabeled_image_stems=val_unlabeled_stems)
    else:
        train_text = load_text_or_default(config.train_dataset, 'Train_text.xlsx')
        val_text   = load_text_or_default(config.val_dataset,   'Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=image_size, unlabeled_image_stems=unlabeled_train_stems)
        val_dataset   = ImageToImage2D(config.val_dataset,   config.task_name, val_text,   val_tf,
                                       image_size=image_size, unlabeled_image_stems=val_unlabeled_stems)

    _collate = dynamic_pad_collate if not config.resize_images else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              worker_init_fn=worker_init_fn, num_workers=0,
                              pin_memory=use_cuda, collate_fn=_collate)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=True,
                              worker_init_fn=worker_init_fn, num_workers=0,
                              pin_memory=use_cuda, collate_fn=_collate)

    lr = config.learning_rate
    logger.info(model_type)

    if torch.cuda.device_count() > 1 and use_cuda:
        logger.info("Using %d GPUs (DataParallel).", torch.cuda.device_count())
        model = nn.DataParallel(model)

    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler = None

    if resume_ckpt:
        optimizer_state = resume_ckpt.get('optimizer_state_dict', resume_ckpt.get('optimizer'))
        scheduler_state = resume_ckpt.get('scheduler_state_dict', resume_ckpt.get('scheduler'))
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            logger.info('Loaded optimizer state')
        if lr_scheduler is not None and scheduler_state is not None:
            lr_scheduler.load_state_dict(scheduler_state)
            logger.info('Loaded scheduler state')
        logger.info('Current LR: %.6g', optimizer.param_groups[0]['lr'])

    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info("TensorBoard log dir: %s", log_dir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    # ── FIX: khởi tạo scaler có điều kiện theo config.use_amp ────────────────
    use_amp = getattr(config, 'use_amp', False)
    scaler  = amp.GradScaler() if (use_amp and use_cuda) else None
    if scaler is not None:
        logger.info("AMP (FP16) enabled — GradScaler active.")
    else:
        logger.info("AMP disabled — training in FP32.")
    # ─────────────────────────────────────────────────────────────────────────

    max_dice = float(resume_ckpt.get("best_dice", 0.0)) if resume_ckpt else 0.0
    best_epoch = int(resume_ckpt.get("best_epoch", max(0, start_epoch - 1))) if resume_ckpt else 1
    early_stopping_count = int(resume_ckpt.get("early_stopping_count", 0)) if resume_ckpt else 0
    logger.info("Save dir: %s", config.save_path)

    for epoch in range(start_epoch, config.epochs):
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs))
        logger.info(config.session_name)

        # ── Train ─────────────────────────────────────────────────────────────
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        # FIX: truyền scaler vào train loop
        train_one_epoch(train_loader, model, criterion, optimizer, writer,
                        epoch, None, model_type, logger, scaler=scaler)

        # ── Validation (không dùng scaler — chỉ inference) ───────────────────
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,
                                                 optimizer, writer, epoch,
                                                 lr_scheduler, model_type, logger,
                                                 scaler=None)   # FIX: val không dùng AMP scaler

        # ── Save best model ───────────────────────────────────────────────────
        if val_dice > max_dice - 1e-4:
            logger.info('\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(
                max_dice, val_dice))
            max_dice = val_dice
            best_epoch = epoch + 1
            early_stopping_count = 0
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))
            early_stopping_count += 1

        state = {
            'epoch': epoch,
            'model': model_type,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'val_loss': val_loss,
            'best_dice': max_dice,
            'early_stopping_count': early_stopping_count,
            'best_epoch': best_epoch,
            # FIX: lưu scaler state để resume AMP đúng
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        }
        save_checkpoint(state, config.model_path, "latest.pth.tar")
        if val_dice >= max_dice and early_stopping_count == 0:
            save_checkpoint(state, config.model_path, "best_model.pth.tar")
        if config.save_frequency > 0 and ((epoch + 1) % config.save_frequency == 0):
            save_checkpoint(state, config.model_path, "epoch-{:04d}.pth.tar".format(epoch + 1))

        logger.info('\t early_stopping_count: {}/{}'.format(
            early_stopping_count, config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LViT model')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to specific checkpoint file to resume from')
    args = parser.parse_args()

    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    ckpt_path = None
    if args.resume or args.checkpoint:
        ckpt_path = find_latest_resume_checkpoint(args.checkpoint)
        if not ckpt_path:
            raise FileNotFoundError("No checkpoint found for resume.")
        update_paths_for_resume(ckpt_path)
    elif not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)

    config_vit = config.get_CTranS_config()
    logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
    logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
    logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))

    model_cls = EfficientLViT if config.use_efficient_lvit or config.model_name == 'EfficientLViT' else LViT_base
    if model_cls is EfficientLViT:
        config.window_size    = getattr(config, 'efficient_lvit_window_size', 7)
        config.vit_depth      = getattr(config, 'efficient_lvit_depth', 1)
        config.vit_num_heads  = getattr(config, 'efficient_lvit_num_heads', 4)
        config.vit_key_dim    = getattr(config, 'efficient_lvit_key_dim', 16)

    model  = model_cls(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device: %s', str(device))
    model  = model.to(device)

    ckpt = None
    start_epoch = 0
    if ckpt_path is not None:
        model, ckpt = load_checkpoint(model, device, ckpt_path)
        if ckpt:
            start_epoch = int(ckpt.get('epoch', -1)) + 1
            logger.info('Resuming from epoch %d', start_epoch)
        else:
            logger.info('Starting fresh training (no valid checkpoint found)')

    model = main_loop(model, batch_size=config.batch_size, model_type=config.model_name,
                      tensorboard=True, resume_ckpt=ckpt, start_epoch=start_epoch)