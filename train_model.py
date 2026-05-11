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
from nets.LViT import LViT
from torch.utils.data import DataLoader
import logging
import csv
from Train_one_epoch import train_one_epoch, print_summary
import Config as config
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE, WeightedDiceCE, read_text, read_text_LV, save_on_batch
# from thop import profile  # optional: uncomment for FLOPs/params during training, or profile in a separate script
import argparse

def logger_config(log_path):
    logger = logging.getLogger('LViT')  # Use named logger to avoid conflicts
    if logger.hasHandlers():
        logger.handlers.clear()  # Clear existing handlers to avoid duplicates
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    # Avoid duplicate lines in notebooks (root logger often has its own handler).
    logger.propagate = False
    return logger


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)


def load_checkpoint(model, device, checkpoint_path=None):
    '''
        Load checkpoint from file.
        If checkpoint_path is None, find the latest checkpoint automatically.
    '''
    if checkpoint_path is None:
        # Find latest checkpoint automatically
        import glob
        checkpoint_pattern = os.path.join(config.base_save_dir, "**", "models", "*.pth.tar")
        cands = glob.glob(checkpoint_pattern, recursive=True)

        if not cands:
            logger.warning("No checkpoint found. Starting training from scratch.")
            return model, None

        # Sort by modification time (newest first)
        cands.sort(key=os.path.getmtime, reverse=True)
        checkpoint_path = cands[0]

    logger.info('Loading checkpoint from: {}'.format(checkpoint_path))

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    # Display checkpoint info
    epoch_saved = ckpt.get('epoch', 'unknown')
    val_loss = ckpt.get('val_loss', 'unknown')
    is_best = ckpt.get('best_model', False)

    logger.info('Checkpoint info:')
    logger.info('  - Epoch: {}'.format(epoch_saved + 1 if isinstance(epoch_saved, int) else epoch_saved))
    logger.info('  - Val loss: {}'.format(val_loss))
    logger.info('  - Is best model: {}'.format(is_best))

    return model, ckpt

def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


def load_text_or_default(dataset_dir, text_filename):
    text_path = os.path.join(dataset_dir, text_filename)
    if os.path.exists(text_path):
        try:
            return read_text(text_path)
        except Exception as exc:
            logger.warning('Cannot read %s (%s). Fallback to default prompts.', text_path, str(exc))

    # Fallback for custom datasets without xlsx annotations.
    # LViT still expects a text prompt per mask file.
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
#          Main Loop: load model,
# =================================================================================
##################################################################################
def main_loop(model, batch_size=config.batch_size, model_type='', tensorboard=True, resume_ckpt=None, start_epoch=0):
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    logger.info("Training device (from model): %s", device)

    # Load train and val data
    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    unlabeled_train_stems = load_unlabeled_stems(getattr(config, "label_plan_csv", ""))
    train_labels_xlsx = os.path.join(config.train_dataset, "Train_labels.xlsx")
    if os.path.isfile(train_labels_xlsx):
        from_xlsx = load_unlabeled_stems_from_labels_xlsx(train_labels_xlsx)
        if from_xlsx:
            unlabeled_train_stems |= from_xlsx
            logger.info(
                "Merged %d unlabeled train stems from labels xlsx: %s",
                len(from_xlsx),
                train_labels_xlsx,
            )
    val_labels_xlsx = os.path.join(config.val_dataset, "Val_labels.xlsx")
    val_unlabeled_stems = (
        load_unlabeled_stems_from_labels_xlsx(val_labels_xlsx)
        if os.path.isfile(val_labels_xlsx)
        else set()
    )
    if val_unlabeled_stems:
        logger.info(
            "Val set: %d samples will use empty masks per Val_labels.xlsx",
            len(val_unlabeled_stems),
        )

    if config.task_name == 'MoNuSeg':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        val_text = read_text(config.val_dataset + 'Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=config.img_size, unlabeled_image_stems=unlabeled_train_stems)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf,
                                    image_size=config.img_size, unlabeled_image_stems=val_unlabeled_stems)
    elif config.task_name == 'Covid19':
        text = read_text(config.task_dataset + 'Train_Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, text, train_tf,
                                       image_size=config.img_size, unlabeled_image_stems=unlabeled_train_stems)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, text, val_tf,
                                    image_size=config.img_size, unlabeled_image_stems=val_unlabeled_stems)
    else:
        train_text = load_text_or_default(config.train_dataset, 'Train_text.xlsx')
        val_text = load_text_or_default(config.val_dataset, 'Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=config.img_size, unlabeled_image_stems=unlabeled_train_stems)
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf,
                                    image_size=config.img_size, unlabeled_image_stems=val_unlabeled_stems)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn,
                              num_workers=0, pin_memory=use_cuda)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn,
                            num_workers=0, pin_memory=use_cuda)

    lr = config.learning_rate
    logger.info(model_type)

    # Optional THOP FLOPs/params (slow; run later for benchmarking if needed):
    # from thop import profile
    # model.eval()
    # dummy_in = torch.randn(batch_size, 3, config.img_size, config.img_size, device=device)
    # dummy_txt = torch.randn(batch_size, 10, 768, device=device)
    # flops, params = profile(model, inputs=(dummy_in, dummy_txt))
    # logger.info("THOP flops: %s, params: %s", flops, params)
    # model.train(True)

    if torch.cuda.device_count() > 1 and use_cuda:
        logger.info("Using %d GPUs (DataParallel).", torch.cuda.device_count())
        model = nn.DataParallel(model)
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize
    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    else:
        lr_scheduler = None

    if resume_ckpt:
        if 'optimizer' in resume_ckpt and resume_ckpt['optimizer'] is not None:
            optimizer.load_state_dict(resume_ckpt['optimizer'])
            logger.info('Optimizer state restored from checkpoint.')
        if lr_scheduler is not None and 'scheduler' in resume_ckpt and resume_ckpt['scheduler'] is not None:
            lr_scheduler.load_state_dict(resume_ckpt['scheduler'])
            logger.info('Scheduler state restored from checkpoint.')
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info("TensorBoard log dir: %s", log_dir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_dice = 0.0
    best_epoch = 1
    for epoch in range(start_epoch, config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs))
        logger.info(config.session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, model_type, logger)  # sup

        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,
                                                 optimizer, writer, epoch, lr_scheduler, model_type, logger)
        # =============================================================
        #       Save best model
        # =============================================================
        # Save checkpoint every save_frequency epochs
        if (epoch + 1) % config.save_frequency == 0:
            logger.info('\t Saving checkpoint at epoch {}'.format(epoch + 1))
            save_checkpoint({'epoch': epoch,
                             'best_model': False,
                             'model': model_type,
                             'state_dict': model.state_dict(),
                             'val_loss': val_loss,
                             'optimizer': optimizer.state_dict(),
                             'scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None}, config.model_path)

        if val_dice > max_dice:
            if epoch + 1 > 5:
                logger.info(
                    '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict(),
                                 'scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None}, config.model_path)
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':
    # Parse command line arguments
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
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)

    # Build model
    config_vit = config.get_CTranS_config()
    logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
    logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
    logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
    model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device: %s', str(device))
    model = model.to(device)

    # Resume from checkpoint if requested
    ckpt = None
    start_epoch = 0
    if args.resume or args.checkpoint:
        model, ckpt = load_checkpoint(model, device, args.checkpoint)
        if ckpt:
            start_epoch = int(ckpt.get('epoch', -1)) + 1
            logger.info('Resuming from epoch index: %d', start_epoch)
            logger.info('Checkpoint loaded successfully. Continuing training...')
        else:
            logger.info('Starting fresh training (no valid checkpoint found)')

    # Continue with normal training
    model = main_loop(model, batch_size=config.batch_size, model_type=config.model_name, tensorboard=True, resume_ckpt=ckpt, start_epoch=start_epoch)
