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
n_labels = 1

epochs = 200

img_size = 224
resize_images = False  # set False to use original image resolution (best with EfficientLViT)

print_frequency = 50
save_frequency = 10
vis_frequency = 50
max_tokens = 10   # BERT token count per sample

early_stopping_patience = 100

pretrain = False

task_name = 'BTRXD_fullsize'

learning_rate = 3e-4 
batch_size = 4
accumulation_steps = 2  # Effective batch size = batch_size * accumulation_steps

model_name = 'EfficientLViT'   # or 'EfficientLViT' — see use_efficient_lvit below

# EfficientLViT ViT branch (EfficientLViT.py): local-window attention, resolution-flexible stem
use_efficient_lvit = True
efficient_lvit_text_dim = 768
efficient_lvit_window_size = 14    # 7 for 224; try 14 if img_size >= 512
efficient_lvit_depth = 1
efficient_lvit_num_heads = 4
efficient_lvit_key_dim = 16

window_size = efficient_lvit_window_size
vit_depth = efficient_lvit_depth
vit_num_heads = efficient_lvit_num_heads
vit_key_dim = efficient_lvit_key_dim

train_dataset = './datasets/' + task_name + '/Train_Folder/'
val_dataset = './datasets/' + task_name + '/Val_Folder/'
test_dataset = './datasets/' + task_name + '/Test_Folder/'
task_dataset = './datasets/' + task_name + '/Train_Folder/'

label_plan_csv = './datasets/' + task_name + '/label_plan_100.csv'

session_name = 'DEBUG_' + time.strftime('%m.%d_%Hh%M')

base_save_dir = '/kaggle/working/'

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