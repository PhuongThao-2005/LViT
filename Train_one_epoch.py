# -*- coding: utf-8 -*-
import torch
import torch.optim
from torch.cuda.amp import autocast
import os
import time
from utils import *
import Config as config
import warnings
from torchinfo import summary
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary


##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger, scaler=None):
    logging_mode = 'Train' if model.training else 'Val'
    device = next(model.parameters()).device
    accumulation_steps = max(1, int(getattr(config, "accumulation_steps", 1)))
    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
    dices = []
    if model.training:
        optimizer.zero_grad()
    for i, (sampled_batch, names) in enumerate(loader, 1):

        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Move data to the same device as the model (GPU or CPU).
        images, masks, text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
        if text.shape[1] > 10:
            text = text[ :, :10, :]
        
        images, masks, text = images.to(device), masks.to(device), text.to(device)


        # ====================================================
        #             Compute loss
        # ====================================================

        if model.training and scaler is not None:
            # Use AMP (Automatic Mixed Precision) for training
            with autocast():
                preds = model(images, text)
                out_loss = criterion(preds, masks.float())  # Loss
            
            scaler.scale(out_loss / accumulation_steps).backward()
            if (i % accumulation_steps == 0) or (i == len(loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Standard training without AMP
            preds = model(images, text)
            out_loss = criterion(preds, masks.float())  # Loss

            if model.training:
                (out_loss / accumulation_steps).backward()
                if (i % accumulation_steps == 0) or (i == len(loader)):
                    optimizer.step()
                    optimizer.zero_grad()

        train_dice = criterion._show_dice(preds, masks.float())
        train_iou = iou_on_batch(masks,preds)

        batch_time = time.time() - end
        if epoch % config.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        iou_sum += len(images) * train_iou
        # acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice

        if i == len(loader):
            average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
            average_time = time_sum / (config.batch_size*(i-1) + len(images))
            train_iou_average = iou_sum / (config.batch_size*(i-1) + len(images))
            # train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
        else:
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)
            # train_acc_average = acc_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)

        end = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                          average_loss, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, 0, 0,  logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)

        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            # writer.add_scalar(logging_mode + '_acc', train_acc, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss, train_dice_avg