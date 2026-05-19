# -*- coding: utf-8 -*-
"""
roi_dataset.py — ROI Patch Dataset cho LViT fine segmentation

Train-time : crop từ GT mask bbox (per connected component) + random jitter
Val-time   : crop từ GT mask bbox + fixed padding (no jitter)
"""

import os
import random
import logging
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

_log = logging.getLogger(__name__)


def imread_bgr(path: str):
    """
    Đọc ảnh BGR. Với PNG dùng PIL trước để tránh spam cảnh báo libpng iCCP từ OpenCV.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.png':
        try:
            from PIL import Image
            arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    return cv2.imread(path, cv2.IMREAD_COLOR)


def imread_gray(path: str):
    """Đọc mask grayscale; PNG qua PIL để giảm cảnh báo iCCP."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.png':
        try:
            from PIL import Image
            return np.asarray(Image.open(path).convert('L'), dtype=np.uint8)
        except Exception:
            pass
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# ─── Augmentation helpers ─────────────────────────────────────────────────────

def random_flip(image, mask):
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask  = cv2.flip(mask,  1)
    if random.random() > 0.5:
        image = cv2.flip(image, 0)
        mask  = cv2.flip(mask,  0)
    return image, mask

def random_rotate(image, mask):
    angle = random.uniform(-20, 20)
    h, w  = image.shape[:2]
    M     = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    mask  = cv2.warpAffine(mask,  M, (w, h),
                           flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask

def random_brightness_contrast(image):
    alpha = random.uniform(0.8, 1.2)   # contrast
    beta  = random.uniform(-20, 20)    # brightness
    return np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalize(image_rgb_float):
    """image_rgb_float: H×W×3, range [0,1] → normalized tensor C×H×W"""
    img = (image_rgb_float - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).float()

# ─── Bbox utilities ──────────────────────────────────────────────────────────

def align_mask_to_image(mask_raw: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Chỉ resize mask khi kích thước khác ảnh (lệch metadata).
    Không downscale ảnh full-size — ROI crop xử lý sau đó.
    """
    if mask_raw is None:
        return None
    mh, mw = mask_raw.shape[:2]
    if mh == H and mw == W:
        return mask_raw
    return cv2.resize(mask_raw, (W, H), interpolation=cv2.INTER_NEAREST)

def mask_to_bbox(mask_bin):
    """
    Returns (x1, y1, x2, y2) bounding box của tất cả non-zero pixels.
    Trả None nếu mask rỗng.
    """
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def mask_component_ids(mask_bin: np.ndarray, min_area_px: int = 100, max_components: int = 20):
    """
    Trả danh sách label id (>=1) của từng connected component trong mask.
    Nếu không tách được component → [0] (dùng union bbox toàn mask).
    """
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        mask_bin.astype(np.uint8), connectivity=8
    )
    ids = []
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area_px:
            ids.append(lbl)
    if not ids:
        return [0]
    ids.sort(key=lambda lbl: stats[lbl, cv2.CC_STAT_AREA], reverse=True)
    return ids[:max_components]

def bbox_from_component(mask_bin: np.ndarray, comp_id: int):
    """Bbox của một component (comp_id>=1) hoặc toàn mask (comp_id==0)."""
    if comp_id <= 0:
        return mask_to_bbox(mask_bin)
    comp_mask = (mask_bin > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(comp_mask)
    if comp_id >= num_labels:
        return mask_to_bbox(mask_bin)
    return mask_to_bbox((labels == comp_id).astype(np.uint8))

def expand_bbox(x1, y1, x2, y2, H, W, padding, jitter=0):
    """
    Mở rộng bbox bằng padding + random jitter, rồi clamp về biên ảnh.
    """
    if jitter > 0:
        x1 -= padding + random.randint(-jitter, jitter)
        y1 -= padding + random.randint(-jitter, jitter)
        x2 += padding + random.randint(-jitter, jitter)
        y2 += padding + random.randint(-jitter, jitter)
    else:
        x1 -= padding
        y1 -= padding
        x2 += padding
        y2 += padding

    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(W, x2);  y2 = min(H, y2)

    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)

    return x1, y1, x2, y2

def crop_and_resize(image, mask, x1, y1, x2, y2, patch_size):
    """Crop ROI rồi resize về patch_size×patch_size."""
    roi_img  = image[y1:y2, x1:x2]
    roi_mask = mask [y1:y2, x1:x2]

    if roi_img.size == 0:
        roi_img  = image
        roi_mask = mask

    roi_img  = cv2.resize(roi_img,  (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    roi_mask = cv2.resize(roi_mask, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
    return roi_img, roi_mask

# ─── Dataset ─────────────────────────────────────────────────────────────────

class ROIPatchDataset(Dataset):
    """
    Dataset trả về patch 224×224 được crop từ bbox của GT mask.
    Mỗi connected component (khối u tách rời) → một sample riêng.

    Args:
        img_dir    : thư mục chứa ảnh (.png/.jpg)
        mask_dir   : thư mục chứa mask (grayscale, cùng tên ảnh)
        text_dict  : dict {filename_key: np.ndarray shape (10, 768)}
        patch_size : kích thước output patch (default 224)
        padding    : số pixel padding quanh GT bbox (default 80)
        jitter     : random jitter ±jitter px (train only)
        augment    : có dùng augmentation không (chỉ khi split='train')
        split      : 'train' hoặc 'val'
        skip_empty : bỏ qua ảnh không có mask (default True)
        max_components_per_image : tối đa component / ảnh (default 20)
    """

    def __init__(
        self,
        img_dir:    str,
        mask_dir:   str,
        text_dict:  dict,
        patch_size: int  = 224,
        padding:    int  = 80,
        jitter:     int  = 40,
        augment:    bool = True,
        split:      str  = 'train',
        skip_empty: bool = True,
        max_components_per_image: int = 20,
    ):
        self.img_dir    = img_dir
        self.mask_dir   = mask_dir
        self.text_dict  = text_dict
        self.patch_size = patch_size
        self.padding    = padding
        self.jitter     = jitter if split == 'train' else 0
        self.augment    = augment and (split == 'train')
        self.split      = split
        self.max_components = max_components_per_image

        self.samples = []
        img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        n_images = 0
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

            m = imread_gray(mask_path)
            if skip_empty and (m is None or m.max() == 0):
                continue

            n_images += 1
            mask_bin = (m > 0).astype(np.uint8)
            H_m, W_m = mask_bin.shape[:2]
            min_area = max(100, int(H_m * W_m * 0.0001))
            comp_ids = mask_component_ids(
                mask_bin, min_area_px=min_area, max_components=self.max_components
            )
            for cid in comp_ids:
                self.samples.append((img_path, mask_path, fn, stem, cid))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"ROIPatchDataset: không tìm thấy sample hợp lệ trong {img_dir}"
            )

        _log.info(
            "[ROIPatchDataset][%s] %d patches from %d images | patch=%s padding=%s jitter=%s aug=%s",
            split, len(self.samples), n_images, patch_size, padding, self.jitter, self.augment,
        )

    def __len__(self):
        return len(self.samples)

    def _get_text_embedding(self, img_fn, mask_fn):
        for key in (os.path.basename(mask_fn),
                    os.path.basename(img_fn),
                    os.path.splitext(os.path.basename(img_fn))[0]):
            if key in self.text_dict:
                emb = self.text_dict[key]
                if isinstance(emb, np.ndarray):
                    return emb.astype(np.float32)
                return np.array(emb, dtype=np.float32)
        return np.zeros((10, 768), dtype=np.float32)

    def __getitem__(self, idx):
        img_path, mask_path, img_fn, stem, comp_id = self.samples[idx]

        image_bgr = imread_bgr(img_path)
        mask_raw  = imread_gray(mask_path)
        if image_bgr is None or mask_raw is None:
            fallback = (idx + 1) % len(self)
            return self.__getitem__(fallback)

        H, W = image_bgr.shape[:2]
        mask_raw = align_mask_to_image(mask_raw, H, W)
        mask_bin = (mask_raw > 0).astype(np.uint8)

        bbox = bbox_from_component(mask_bin, comp_id)
        if bbox is None:
            x1, y1, x2, y2 = 0, 0, W, H
        else:
            x1, y1, x2, y2 = expand_bbox(
                *bbox, H=H, W=W,
                padding=self.padding,
                jitter=self.jitter
            )

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        patch_img, patch_mask = crop_and_resize(
            image_rgb, mask_bin, x1, y1, x2, y2, self.patch_size
        )

        if self.augment:
            patch_img, patch_mask = random_flip(patch_img, patch_mask)
            patch_img, patch_mask = random_rotate(patch_img, patch_mask)
            patch_img = random_brightness_contrast(patch_img)

        img_float  = patch_img.astype(np.float32) / 255.0
        img_tensor = normalize(img_float)

        mask_tensor = torch.from_numpy(
            patch_mask.astype(np.float32)
        ).unsqueeze(0)

        text_emb    = self._get_text_embedding(img_fn, mask_path)
        text_tensor = torch.from_numpy(text_emb)

        return {
            'image': img_tensor,                              # (3, 224, 224)
            'mask':  mask_tensor,                             # (1, 224, 224) float 0/1
            'text':  text_tensor,                             # (10, 768)
            'roi':   torch.tensor([x1, y1, x2, y2], dtype=torch.float32),
            'stem':  stem,
            'comp_id': comp_id,
        }

# ─── Collate fn ──────────────────────────────────────────────────────────────

def roi_collate_fn(batch):
    images = torch.stack([b['image'] for b in batch])
    masks  = torch.stack([b['mask']  for b in batch])
    texts  = torch.stack([b['text']  for b in batch])
    rois   = torch.stack([b['roi']   for b in batch])
    stems  = [b['stem'] for b in batch]
    return {'image': images, 'mask': masks, 'text': texts, 'roi': rois, 'stem': stems}
