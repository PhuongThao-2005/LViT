# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage
import pandas as pd
try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None


class HFTextEmbedder:
    """
    HuggingFace text embedding wrapper that returns token embeddings with shape
    [max_tokens, 768] (BERT hidden size), matching existing LViT text pipeline.
    """
    def __init__(self, model_name: str = "bert-base-uncased", max_tokens: int = 10):
        if AutoTokenizer is None or AutoModel is None:
            raise ImportError(
                "transformers is required for text embedding. "
                "Please install it (e.g. pip install transformers)."
            )
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 768))
        self._cache = {}

    def encode(self, text: str) -> np.ndarray:
        key = text.strip()
        if key in self._cache:
            return self._cache[key]

        with torch.no_grad():
            encoded = self.tokenizer(
                key,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_tokens + 2,
                return_tensors="pt",
            )
            outputs = self.model(**encoded)
            token_embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden]
            token_embeddings = token_embeddings[1:-1]  # drop [CLS], [SEP]
            if token_embeddings.shape[0] == 0:
                arr = np.zeros((self.max_tokens, self.hidden_size), dtype=np.float32)
                self._cache[key] = arr
                return arr

            if token_embeddings.shape[0] > self.max_tokens:
                token_embeddings = token_embeddings[: self.max_tokens]

            arr = token_embeddings.cpu().numpy().astype(np.float32)
            if arr.shape[0] < self.max_tokens:
                pad = np.zeros((self.max_tokens - arr.shape[0], self.hidden_size), dtype=np.float32)
                arr = np.vstack([arr, pad])

            self._cache[key] = arr
            return arr


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # if x != self.output_size[0] or y != self.output_size[1]:
        if self.output_size and (x != self.output_size[0] or y != self.output_size[1]):
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)  # OSIC
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if self.output_size and (x != self.output_size[0] or y != self.output_size[1]):
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
        return sample


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def load_unlabeled_stems_from_labels_xlsx(xlsx_path: str) -> set:
    """
    Read per-sample mask supervision from an xlsx next to img/labelcol.

    Expected columns (case-insensitive):
      - Image : mask filename (e.g. IMG000001.png) or image filename
      - is_labeled OR use_mask OR labeled :
          if False/0/no → sample is treated as unlabeled (empty mask at train time)

    Returns:
        set of image stems (no extension) for which pixel mask should be cleared.
    """
    if not xlsx_path or not os.path.isfile(xlsx_path):
        return set()

    try:
        df = pd.read_excel(xlsx_path)
    except Exception:
        return set()

    if df.empty:
        return set()

    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    col_img = pick("Image", "image_id", "image", "mask", "file")
    if col_img is None:
        return set()

    col_labeled = pick("is_labeled", "labeled", "has_label", "label_mask")
    col_use_mask = pick("use_mask")
    if col_labeled is None and col_use_mask is None:
        return set()

    truthy = {True, 1, "1", "true", "yes", "y"}

    def is_true(v):
        if pd.isna(v):
            return False
        if isinstance(v, str):
            return v.strip().lower() in truthy
        return v in truthy

    unlabeled = set()
    for _, row in df.iterrows():
        raw = row[col_img]
        if pd.isna(raw):
            continue
        stem = os.path.splitext(str(raw).strip())[0]

        if col_labeled is not None:
            if not is_true(row[col_labeled]):
                unlabeled.add(stem)
        else:
            if not is_true(row[col_use_mask]):
                unlabeled.add(stem)

    return unlabeled


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class LV2D(Dataset):
    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_path = os.path.join(dataset_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.text_embedder = HFTextEmbedder(model_name="bert-base-uncased", max_tokens=10)

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.output_path))

    def __getitem__(self, idx):

        mask_filename = self.mask_list[idx]  # Co
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        mask = correct_dims(mask)
        text = self.rowtext[mask_filename]
        text = self.text_embedder.encode(text)
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'label': mask, 'text': text}

        return sample, mask_filename


class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224,
                 unlabeled_image_stems: set | None = None) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.text_embedder = HFTextEmbedder(model_name="bert-base-uncased", max_tokens=10)
        self.unlabeled_image_stems = unlabeled_image_stems if unlabeled_image_stems is not None else set()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]  # MoNuSeg
        image_stem, _ = os.path.splitext(image_filename)
        # Support datasets where masks share the same extension as images (jpg/jpeg/png).
        candidate_masks = [
            image_filename,
            image_stem + ".png",
            image_stem + ".jpg",
            image_stem + ".jpeg",
        ]
        mask_filename = None
        for candidate in candidate_masks:
            if os.path.exists(os.path.join(self.output_path, candidate)):
                mask_filename = candidate
                break
        if mask_filename is None:
            raise FileNotFoundError("Cannot find mask for image: {}".format(image_filename))
        # mask_filename = self.mask_list[idx]  # Covid19
        # image_filename = mask_filename.replace('mask_', '')  # Covid19
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        if self.image_size is not None:
            image = cv2.resize(image, (self.image_size, self.image_size))

        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        if self.image_size is not None:
            mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        if image_stem in self.unlabeled_image_stems:
            mask = np.zeros_like(mask, dtype=np.uint8)

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        text = self.rowtext[mask_filename]
        text = self.text_embedder.encode(text)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'image': image, 'label': mask, 'text': text}

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename