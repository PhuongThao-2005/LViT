import os, json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from Load_Dataset import HFTextEmbedder
from utils import read_text


class PatchDataset(Dataset):
    WIN_SIZE  = 224
    MIN_POS   = 4
    NEG_RATIO = 1

    def __init__(self, dataset_path: str, text_dict: dict,
                 augment: bool = True,
                 win_size: int = WIN_SIZE,
                 neg_ratio: int = NEG_RATIO,
                 seed: int = 666,
                 cache_json: str = None):

        self.img_dir   = os.path.join(dataset_path, 'img')
        self.mask_dir  = os.path.join(dataset_path, 'labelcol')
        self.text_dict = text_dict
        self.augment   = augment
        self.win_size  = win_size
        self.neg_ratio = neg_ratio
        self.seed      = seed
        self.rng       = np.random.RandomState(seed)

        self.embedder    = HFTextEmbedder(model_name="bert-base-uncased", max_tokens=10)
        self._text_cache = {}   # cache: text_str → np.ndarray

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if cache_json and os.path.exists(cache_json):
            with open(cache_json) as f:
                self.patches = json.load(f)
            print(f"PatchDataset: loaded {len(self.patches)} patches ← {cache_json}")
        else:
            self.patches = []
            self._build_patch_list()
            if cache_json:
                os.makedirs(os.path.dirname(cache_json), exist_ok=True)
                with open(cache_json, 'w') as f:
                    json.dump(self.patches, f)
                print(f"PatchDataset: saved {len(self.patches)} patches → {cache_json}")

        print(f"PatchDataset ready: {len(self.patches)} patches | "
              f"neg_ratio={neg_ratio} | seed={seed}")

    # ── Patch list generation ─────────────────────────────────────────────────
    def _sample_centers_from_components(self, mask_bin: np.ndarray):
        """
        Thay vì sample theo pixel (bias về lesion lớn),
        dùng connected components → 1 center / component.
        Mỗi component bổ sung thêm 1 center ngẫu nhiên bên trong nó.
        """
        num_labels, labels = cv2.connectedComponents(mask_bin.astype(np.uint8))
        centers = []
        for lbl in range(1, num_labels):   # bỏ background (label 0)
            ys, xs = np.where(labels == lbl)
            # Center của component
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            centers.append((cy, cx))
            # Thêm 1 điểm random bên trong component để tăng diversity
            if len(ys) > 1:
                pick = self.rng.randint(0, len(ys))
                centers.append((int(ys[pick]), int(xs[pick])))
        return centers

    def _build_patch_list(self):
        ws  = self.win_size
        img_files = sorted([f for f in os.listdir(self.img_dir)
                             if f.lower().endswith(('.png','.jpg','.jpeg'))])

        for img_fn in img_files:
            stem = os.path.splitext(img_fn)[0]
            mask_fn = None
            for ext in (img_fn, stem+'.png', stem+'.jpg'):
                cand = os.path.join(self.mask_dir, ext)
                if os.path.exists(cand): mask_fn = cand; break
            if mask_fn is None: continue

            img_path  = os.path.join(self.img_dir, img_fn)
            mask_full = cv2.imread(mask_fn, 0)
            if mask_full is None: continue

            H, W = mask_full.shape[:2]
            text_str = self.text_dict.get(
                img_fn, self.text_dict.get(
                os.path.basename(mask_fn), 'tumor lesion segmentation'))

            # Ảnh nhỏ hơn window → resize patch duy nhất
            if H < ws or W < ws:
                self.patches.append([img_path, mask_fn, 0, 0, text_str, H, W, True])
                continue

            mask_bin = (mask_full > 0).astype(np.uint8)
            if mask_bin.sum() == 0:
                continue   # bỏ ảnh không có tumor

            # ── Positive patches (component-based, không pixel-bias) ──────
            centers    = self._sample_centers_from_components(mask_bin)
            pos_patches = []
            for cy, cx in centers:
                y0 = int(np.clip(cy - ws//2, 0, H - ws))
                x0 = int(np.clip(cx - ws//2, 0, W - ws))
                pos_patches.append([img_path, mask_fn, y0, x0, text_str, H, W, False])

            # Fallback nếu thiếu MIN_POS → crop quanh lesion center (KHÔNG random toàn ảnh)
            if len(pos_patches) < self.MIN_POS:
                lesion_cy = int(np.mean(np.where(mask_bin > 0)[0]))
                lesion_cx = int(np.mean(np.where(mask_bin > 0)[1]))
                while len(pos_patches) < self.MIN_POS:
                    off_y = int(self.rng.randint(-64, 65))
                    off_x = int(self.rng.randint(-64, 65))
                    y0 = int(np.clip(lesion_cy - ws//2 + off_y, 0, H - ws))
                    x0 = int(np.clip(lesion_cx - ws//2 + off_x, 0, W - ws))
                    pos_patches.append([img_path, mask_fn, y0, x0, text_str, H, W, False])

            # ── Negative patches ─────────────────────────────────────────
            n_neg = len(pos_patches) * self.neg_ratio
            neg_patches = []
            attempts = 0
            while len(neg_patches) < n_neg and attempts < n_neg * 20:
                y0 = int(self.rng.randint(0, H - ws + 1))
                x0 = int(self.rng.randint(0, W - ws + 1))
                if (mask_full[y0:y0+ws, x0:x0+ws] > 0).sum() < ws * ws * 0.01:
                    neg_patches.append([img_path, mask_fn, y0, x0, text_str, H, W, False])
                attempts += 1

            self.patches.extend(pos_patches)
            self.patches.extend(neg_patches)

    # ── Text cache ────────────────────────────────────────────────────────────
    def _encode_text(self, text_str: str) -> np.ndarray:
        if text_str not in self._text_cache:
            self._text_cache[text_str] = self.embedder.encode(text_str)
        return self._text_cache[text_str]

    # ── __getitem__ ───────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_path, mask_fn, y0, x0, text_str, H_orig, W_orig, small_img = self.patches[idx]
        ws = self.win_size

        image_bgr = cv2.imread(img_path)
        mask_full = cv2.imread(mask_fn, 0)

        if small_img:
            image_bgr = cv2.resize(image_bgr, (ws, ws))
            mask_full = cv2.resize(mask_full, (ws, ws), interpolation=cv2.INTER_NEAREST)
            patch = image_bgr; patch_mask = mask_full
        else:
            patch      = image_bgr[y0:y0+ws, x0:x0+ws]
            patch_mask = mask_full[y0:y0+ws, x0:x0+ws]

        # Augmentation — dùng np.random global (intentionally non-reproducible cho diversity)
        if self.augment:
            if np.random.random() > 0.5:
                patch = cv2.flip(patch, 1); patch_mask = cv2.flip(patch_mask, 1)
            if np.random.random() > 0.5:
                patch = cv2.flip(patch, 0); patch_mask = cv2.flip(patch_mask, 0)
            k = np.random.choice([0, 1, 2, 3])
            if k:
                patch = np.rot90(patch, k).copy()
                patch_mask = np.rot90(patch_mask, k).copy()

        # Normalize
        patch_rgb  = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        patch_norm = (patch_rgb - self.mean) / self.std
        image_t    = torch.from_numpy(patch_norm.transpose(2, 0, 1)).float()

        patch_mask = (patch_mask > 0).astype(np.float32)
        mask_t     = torch.from_numpy(patch_mask[np.newaxis]).float()

        text_t = torch.from_numpy(self._encode_text(text_str)).float()

        return {'image': image_t, 'label': mask_t, 'text': text_t}