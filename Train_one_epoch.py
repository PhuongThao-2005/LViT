# -*- coding: utf-8 -*-
import torch.optim
import os
import time
from utils import *
import Config as config
import warnings
from torchinfo import summary
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")
import torch.cuda.amp as amp


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger):
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)


def train_one_epoch(loader, model, criterion, optimizer, writer,
                    epoch, lr_scheduler, model_type, logger, scaler=None):
    logging_mode = 'Train' if model.training else 'Val'
    device = next(model.parameters()).device
    accumulation_steps = max(1, int(getattr(config, "accumulation_steps", 1)))
    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum = 0.0, 0.0
    dices = []
    average_loss   = 0.0
    train_dice_avg = 0.0

    if model.training:
        optimizer.zero_grad()

    for i, (sampled_batch, names) in enumerate(loader, 1):
        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = type(criterion).__name__

        images = sampled_batch['image'].to(device).float()
        masks  = sampled_batch['label'].to(device).float()
        text   = sampled_batch['text'].to(device).float()

        if text.shape[1] > 10:
            text = text[:, :10, :]

        labeled_mask = masks.sum(dim=(-2, -1)).view(-1) > 0

        with amp.autocast(enabled=(scaler is not None)):
            preds = model(images, text)
            if labeled_mask.any():
                if labeled_mask.all():
                    loss = criterion(preds, masks)
                else:
                    labeled_idx = labeled_mask.nonzero(as_tuple=True)[0]
                    loss = criterion(preds[labeled_idx], masks[labeled_idx])
            else:
                continue  # skip toàn batch unlabeled

        loss_val = loss.item()  # lưu trước khi chia accumulation

        loss = loss / accumulation_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if i % accumulation_steps == 0 or i == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        # Dice/IoU chỉ tính trên labeled samples
        if labeled_mask.all():
            train_dice = criterion._show_dice(
                preds.detach().clone(), masks.float().detach().clone())
            train_iou  = iou_on_batch(masks, preds)
        else:
            labeled_idx = labeled_mask.nonzero(as_tuple=True)[0]
            train_dice  = criterion._show_dice(
                preds[labeled_idx].detach().clone(),
                masks[labeled_idx].float().detach().clone())
            train_iou   = iou_on_batch(masks[labeled_idx], preds[labeled_idx])

        batch_time = time.time() - end

        if epoch % config.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = config.visualize_path + str(epoch) + '/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images, masks, preds, names, vis_path)

        dices.append(train_dice)
        time_sum  += len(images) * batch_time
        loss_sum  += len(images) * loss_val
        iou_sum   += len(images) * train_iou
        dice_sum  += len(images) * train_dice

        if i == len(loader):
            denom = config.batch_size * (i - 1) + len(images)
        else:
            denom = i * config.batch_size
        denom = max(denom, 1)

        average_loss      = loss_sum / denom
        average_time      = time_sum / denom
        train_iou_average = iou_sum  / denom
        train_dice_avg    = dice_sum / denom

        end = time.time()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), loss_val, loss_name,
                          batch_time, average_loss, average_time,
                          train_iou, train_iou_average,
                          train_dice, train_dice_avg, 0, 0, logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),
                          logger=logger)

        if config.tensorboard and writer is not None:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, loss_val, step)
            writer.add_scalar(logging_mode + '_iou',  train_iou,  step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss, train_dice_avg