# -*- coding: utf-8 -*-
import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True  # Use cosineLR or not
n_channels = 3
n_labels = 1  # MoNuSeg & Covid19
# epochs = 2000
epochs = 1
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 10
early_stopping_patience = 50

pretrain = False
# task_name supports custom datasets if folder layout matches README.
task_name = 'BTRXD_tumor_l100'
# task_name = 'MoNuSeg'
# task_name = 'Covid19'
learning_rate = 1e-3  # MoNuSeg/BTRXD: 1e-3, Covid19: 3e-4
batch_size = 2  # For LViT-T, 2 is better than 4

model_name = 'LViT'
# model_name = 'LViT_pretrain'

train_dataset = './datasets/' + task_name + '/Train_Folder/'
val_dataset = './datasets/' + task_name + '/Val_Folder/'
test_dataset = './datasets/' + task_name + '/Test_Folder/'
task_dataset = './datasets/' + task_name + '/Train_Folder/'
# Semi-supervised plan CSV with columns: image_id,split,is_labeled
# Keep images/masks only in BTRXD_tumor_l100 and switch ratios via these plans.
label_plan_csv = './datasets/' + task_name + '/label_plan_25.csv'
# label_plan_csv = './datasets/' + task_name + '/label_plan_50.csv'
# label_plan_csv = './datasets/' + task_name + '/label_plan_100.csv'
session_name = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
base_save_dir = '/kaggle/working/' if os.path.isdir('/kaggle/working') else './'
save_path = os.path.join(base_save_dir, task_name, model_name, session_name) + os.sep
model_path = os.path.join(save_path, 'models') + os.sep
tensorboard_folder = os.path.join(save_path, 'tensorboard_logs') + os.sep
logger_path = os.path.join(save_path, session_name + '.log')
visualize_path = os.path.join(save_path, 'visualize_val') + os.sep


##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64  # base channel of U-Net
    config.n_classes = 1
    return config


# used in testing phase, copy the session name in training phase
# test_session = "Test_session_05.23_14h19"  # dice=79.98, IoU=66.83