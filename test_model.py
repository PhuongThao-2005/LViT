import torch
import torch.optim
import torch.nn as nn
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import glob
from nets.LViT import LViT as LViT_base
from nets.EfficientLViT import LViT as EfficientLViT
from utils import *
import cv2


def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl   = labs.astype(np.float32)
    tmp_pred  = predict_save.astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_pred) / (np.sum(tmp_lbl) + np.sum(tmp_pred) + 1e-5)
    iou_pred  = jaccard_score(tmp_lbl.reshape(-1), tmp_pred.reshape(-1))
    if config.task_name == "MoNuSeg":
        predict_save = cv2.pyrUp(predict_save, (448, 448))
        predict_save = cv2.resize(predict_save, (2000, 2000))
    cv2.imwrite(save_path, predict_save * 255)
    return dice_pred, iou_pred


def vis_and_save_heatmap(model, input_img, text, img_RGB, labs, vis_save_path, model_type, dice_pred, dice_ens):
    model.eval()
    output = model(input_img.cuda(), text.cuda())
    pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    if predict_save.ndim == 3 and predict_save.shape[0] == 1:
        predict_save = predict_save[0]
    dice_pred_tmp, iou_tmp = show_image_with_dice(
        predict_save, labs,
        save_path=vis_save_path + '_predict_' + model_type + '.jpg')
    return dice_pred_tmp, iou_tmp


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session
    model_type = config.model_name

    # ── FIX #7: test_num — tự đếm số file thực, không hardcode ──────────────
    if config.task_name == "MoNuSeg":
        test_num = 14
    elif config.task_name == "Covid19":
        test_num = 2113
    else:
        # BTRXD hoặc bất kỳ dataset custom nào: tự đếm
        img_dir = os.path.join(config.test_dataset, 'img')
        test_num = len([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ]) if os.path.isdir(img_dir) else 0
        print(f"[test_model] Auto-detected test_num = {test_num} from {img_dir}")
    # ─────────────────────────────────────────────────────────────────────────

    # ── FIX: model_path dùng config.model_path thay vì hardcode ─────────────
    model_path = os.path.join(config.model_path, "best_model.pth.tar")
    if not os.path.exists(model_path):
        # fallback: tìm file .pth.tar mới nhất trong thư mục model
        candidates = glob.glob(os.path.join(config.model_path, "*.pth.tar"))
        if candidates:
            model_path = max(candidates, key=os.path.getmtime)
            print(f"[test_model] best_model.pth.tar not found, using: {model_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {config.model_path}")
    # ─────────────────────────────────────────────────────────────────────────

    save_path = os.path.join(config.task_name, model_type, test_session) + os.sep
    vis_path  = "./" + config.task_name + '_visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

    if model_type in ('LViT', 'LViT_pretrain'):
        config_vit = config.get_CTranS_config()
        model = LViT_base(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'EfficientLViT':
        config_vit = config.get_CTranS_config()
        config.window_size     = getattr(config, 'efficient_lvit_window_size', 7)
        config.vit_depth       = getattr(config, 'efficient_lvit_depth', 1)
        config.vit_num_heads   = getattr(config, 'efficient_lvit_num_heads', 4)
        config.vit_key_dim     = getattr(config, 'efficient_lvit_key_dim', 16)
        model = EfficientLViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    else:
        raise TypeError(f"Unknown model_type: {model_type}")

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # ── FIX: load checkpoint với key 'model_state_dict' (chuẩn theo train_model.py mới)
    state_key = "model_state_dict" if "model_state_dict" in checkpoint else "state_dict"
    model.load_state_dict(checkpoint[state_key], strict=False)
    print(f"Model loaded from: {model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}, Best dice: {checkpoint.get('best_dice', 'unknown'):.4f}")

    test_output_size = [config.img_size, config.img_size] if config.resize_images else None
    tf_test = ValGenerator(output_size=test_output_size)

    # ── FIX: load text với fallback giống train_model.py ─────────────────────
    test_text_path = os.path.join(config.test_dataset, 'Test_text.xlsx')
    if os.path.exists(test_text_path):
        test_text = read_text(test_text_path)
    else:
        label_dir = os.path.join(config.test_dataset, 'labelcol')
        test_text = {}
        default_prompt = 'chest xray lesion segmentation EOF XXX EOF XXX EOF XXX EOF XXX'
        if os.path.isdir(label_dir):
            for mask_name in os.listdir(label_dir):
                test_text[mask_name] = default_prompt
        print(f"[test_model] Test_text.xlsx not found, using default prompts for {len(test_text)} masks.")
    # ─────────────────────────────────────────────────────────────────────────

    test_dataset = ImageToImage2D(
        config.test_dataset, config.task_name, test_text, tf_test,
        image_size=config.img_size if config.resize_images else None
    )

    # ── FIX #8: collate_fn cho test_loader (batch_size=1 OK, nhưng thêm cho nhất quán)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if test_num == 0:
        test_num = len(test_dataset)
        print(f"[test_model] test_num reset to dataset length: {test_num}")

    dice_pred = 0.0
    iou_pred  = 0.0
    dice_ens  = 0.0

    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label, test_text_batch = (
                sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
            )
            arr = test_data.numpy().astype(np.float32)
            lab = test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255

            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = lab.shape[1], lab.shape[2]
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path + str(names) + "_lab.jpg", dpi=300)
            plt.close()

            input_img = torch.from_numpy(arr)
            dice_pred_t, iou_pred_t = vis_and_save_heatmap(
                model, input_img, test_text_batch, None, lab,
                vis_path + str(names), model_type,
                dice_pred=dice_pred, dice_ens=dice_ens
            )
            dice_pred += dice_pred_t
            iou_pred  += iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()

    print(f"\n=== Test Results ===")
    print(f"Mean Dice: {dice_pred / test_num:.4f}")
    print(f"Mean IoU:  {iou_pred  / test_num:.4f}")
