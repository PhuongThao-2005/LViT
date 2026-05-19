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

print_frequency = 50
save_frequency = 10
vis_frequency = 50
early_stopping_patience = 100

pretrain = False

model_name = 'LViT'
learning_rate = 3e-4
batch_size = 4
accumulation_steps = 2  # Effective batch size = batch_size * accumulation_steps

# =========================
# ROI training (train_lvit_roi.py)
# =========================

roi_patch_size = 224      # LViT input (must stay 224)
roi_padding = 80          # padding around GT bbox (train) / coarse bbox (inference)
roi_jitter = 40           # ±px random bbox jitter (train only)
roi_num_workers = 4       # DataLoader workers (use 2 on Kaggle if warned)

# Periodic val: coarse UNet → ROI → LViT (weights đang train) → paste → Dice/IoU full image
# Giống inference_roi.py — cần coarse_session + checkpoint coarse đã train
roi_full_eval_every = 20       # 0 = tắt C2F eval khi train
roi_full_eval_max_samples = 32 # 0 = toàn bộ ảnh val (chậm trên fullsize)

# Optional: load LViT full-image checkpoint trước khi train ROI (None = train from scratch)
lvit_pretrained_path = None

# =========================
# DATASET
# =========================

task_name = 'BTRXD_fullsize'

train_dataset = './datasets/' + task_name + '/Train_Folder/'
val_dataset   = './datasets/' + task_name + '/Val_Folder/'
test_dataset  = './datasets/' + task_name + '/Test_Folder/'
task_dataset  = './datasets/' + task_name + '/Train_Folder/'

label_plan_csv = './datasets/' + task_name + '/label_plan_100.csv'

# =========================
# SAVE PATH
# =========================

base_save_dir = '/kaggle/working/'

session_name = 'ROI_E2E_' + time.strftime('%m.%d_%Hh%M')

save_path = os.path.join(
    base_save_dir,
    task_name,
    model_name,
    session_name,
) + os.sep

model_path = os.path.join(save_path, 'models') + os.sep
tensorboard_folder = os.path.join(save_path, 'tensorboard_logs') + os.sep
logger_path = os.path.join(save_path, session_name + '.log')
visualize_path = os.path.join(save_path, 'visualize_val') + os.sep

# =========================
# CHECKPOINT SESSIONS (inference / C2F val)
# =========================
# train_lvit_roi: fine weights = model đang train; coarse = load từ coarse_session
# inference_roi / test_roi_based: fine từ test_session (models/roi/best_model.pth.tar)

test_session   = 'DEBUG_05.12_23h37'   # session đã train LViT ROI (sau khi train xong, đổi = session_name)
coarse_session = 'DEBUG_05.13_11h38'   # session UNet coarse (bắt buộc cho C2F eval mỗi N epoch)

# =========================
# LViT CONFIG
# =========================

def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()

    config.KV_size = 960
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
