# -*- coding: utf-8 -*-
"""
prepare_dataset.py
==================
Tạo cấu trúc thư mục chuẩn LViT từ raw BTRXD dataset.

Input:
    - dataset.xlsx          : metadata (3746 ảnh)
    - raw_images/           : thư mục chứa toàn bộ ảnh gốc (IMG000001.jpeg ...)
    - raw_masks/            : thư mục chứa mask (chỉ ~2000 ảnh có bệnh)

Output (cấu trúc LViT gốc):
    data/BTRXD/
    ├── Train_Folder/
    │   ├── img/            ← ảnh train (resize 224×224)
    │   ├── labelcol/       ← mask train (binary PNG, 224×224)
    │   └── Train_text.xlsx ← text descriptions cho ảnh train
    ├── Val_Folder/
    │   ├── img/
    │   ├── labelcol/
    │   └── Val_text.xlsx
    └── Test_Folder/
        ├── img/
        ├── labelcol/
        └── Test_text.xlsx

Cách dùng:
    python prepare_dataset.py \
        --excel      dataset.xlsx \
        --img_dir    raw_images/ \
        --mask_dir   raw_masks/ \
        --out_dir    data/BTRXD/ \
        --img_size   224 \
        --seed       42

Notes:
    - Split 70 / 15 / 15 với stratified sampling (giữ tỷ lệ benign/malignant/no_tumor)
    - Ảnh không có mask → tạo mask rỗng (toàn 0) 
    - Mask được binarize: pixel > 127 → 255, còn lại → 0
    - Tên file output đổi sang .png hết để thống nhất
    - File text Excel mỗi split theo format MoNuSeg: columns [Image, Description]
"""

import os
import shutil
import argparse
import random
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ─────────────────────────────────────────────────────────────
#  Hằng số
# ─────────────────────────────────────────────────────────────
SPLIT_RATIO  = (0.70, 0.15, 0.15)   # train / val / test
SPLIT_NAMES  = ["Train_Folder", "Val_Folder", "Test_Folder"]
TEXT_NAMES   = ["Train_text.xlsx", "Val_text.xlsx", "Test_text.xlsx"]
IMG_SIZE     = 224


# ─────────────────────────────────────────────────────────────
#  Tạo stratum label để stratified split
# ─────────────────────────────────────────────────────────────
def make_stratum(row: pd.Series) -> str:
    """
    Gán nhãn stratum cho mỗi ảnh dựa trên:
        - no_tumor                    : không có bệnh
        - ben_upper / ben_lower / ben_pelvis : benign theo vị trí chi
        - mal_upper / mal_lower / mal_pelvis : malignant theo vị trí chi
    Giúp split đảm bảo tỷ lệ bệnh đồng đều trong train/val/test.
    """
    if row.get("tumor", 0) == 0:
        return "no_tumor"
    bm = "mal" if row.get("malignant", 0) == 1 else "ben"
    if row.get("upper limb", 0) == 1:
        limb = "upper"
    elif row.get("lower limb", 0) == 1:
        limb = "lower"
    elif row.get("pelvis", 0) == 1:
        limb = "pelvis"
    else:
        limb = "other"
    return f"{bm}_{limb}"


# ─────────────────────────────────────────────────────────────
#  Stratified split
# ─────────────────────────────────────────────────────────────
def stratified_split(
    df: pd.DataFrame,
    ratios: tuple = SPLIT_RATIO,
    seed: int = 42,
) -> tuple:
    """
    Trả về 3 DataFrame: train_df, val_df, test_df
    Đảm bảo mỗi stratum được phân chia đều theo ratios.
    """
    rng = random.Random(seed)

    train_idx, val_idx, test_idx = [], [], []

    for stratum, group in df.groupby("stratum"):
        indices = group.index.tolist()
        rng.shuffle(indices)
        n = len(indices)

        n_val  = max(1, round(n * ratios[1]))
        n_test = max(1, round(n * ratios[2]))
        # Đảm bảo không vượt quá tổng
        n_val  = min(n_val,  n - 2)
        n_test = min(n_test, n - n_val - 1)
        n_train = n - n_val - n_test

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df   = df.loc[val_idx].reset_index(drop=True)
    test_df  = df.loc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────
#  Tìm file ảnh (hỗ trợ .jpeg và .jpg)
# ─────────────────────────────────────────────────────────────
def find_image_file(img_dir: str, image_id: str) -> str | None:
    """
    Tìm file ảnh từ image_id (có thể là IMG000001.jpeg hoặc .jpg).
    Trả về đường dẫn đầy đủ hoặc None nếu không tìm thấy.
    """
    for ext in [".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"]:
        base = os.path.splitext(image_id)[0]
        path = os.path.join(img_dir, base + ext)
        if os.path.exists(path):
            return path
    # Thử dùng tên gốc đúng như trong Excel
    path = os.path.join(img_dir, image_id)
    if os.path.exists(path):
        return path
    return None


# ─────────────────────────────────────────────────────────────
#  Xử lý và lưu 1 ảnh
# ─────────────────────────────────────────────────────────────
def process_image(src_path: str, dst_path: str, img_size: int = IMG_SIZE):
    """
    Load ảnh gốc → resize về img_size×img_size → lưu PNG.
    Giữ RGB (3 kênh) cho ảnh; convert L nếu là grayscale.
    """
    img = Image.open(src_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.LANCZOS)
    img.save(dst_path, format="PNG")


# ─────────────────────────────────────────────────────────────
#  Xử lý và lưu 1 mask
# ─────────────────────────────────────────────────────────────
def process_mask(src_path: str | None, dst_path: str, img_size: int = IMG_SIZE):
    """
    Nếu có mask:
        - Load → convert grayscale → binarize (>127 → 255) → resize → lưu PNG
    Nếu không có mask (ảnh không bệnh):
        - Tạo mask rỗng toàn 0 → lưu PNG
    """
    if src_path is not None and os.path.exists(src_path):
        mask = Image.open(src_path).convert("L")
        arr  = np.array(mask)
        arr  = (arr > 127).astype(np.uint8) * 255   # binarize
        mask = Image.fromarray(arr)
        mask = mask.resize((img_size, img_size), Image.NEAREST)
    else:
        # Ảnh không bệnh → mask rỗng
        mask = Image.fromarray(np.zeros((img_size, img_size), dtype=np.uint8))
    mask.save(dst_path, format="PNG")


def process_mask_from_annotation(ann_path: str, dst_path: str, img_size: int = IMG_SIZE):
    """
    Build binary mask from LabelMe annotation JSON.
    Segmentation mode: only polygon shapes are used.
    """
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    width = int(data.get("imageWidth", img_size))
    height = int(data.get("imageHeight", img_size))
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for shape in data.get("shapes", []):
        shape_type = shape.get("shape_type", "").lower()
        points = shape.get("points", [])
        if not points:
            continue

        if shape_type == "polygon" and len(points) >= 3:
            polygon = [(float(p[0]), float(p[1])) for p in points]
            draw.polygon(polygon, fill=255, outline=255)

    mask = mask.resize((img_size, img_size), Image.NEAREST)
    arr = np.array(mask)
    arr = (arr > 0).astype(np.uint8) * 255
    Image.fromarray(arr).save(dst_path, format="PNG")


# ─────────────────────────────────────────────────────────────
#  Build text Excel cho 1 split
# ─────────────────────────────────────────────────────────────
def row_to_short_text(row: pd.Series) -> str:
    """Generate concise diagnosis text for LViT text supervision."""
    parts = []
    if row.get("tumor", 0) == 1:
        if row.get("malignant", 0) == 1:
            parts.append("malignant bone tumor")
        else:
            parts.append("benign bone tumor")
    else:
        parts.append("normal bone")

    if row.get("upper limb", 0) == 1:
        parts.append("upper limb")
    elif row.get("lower limb", 0) == 1:
        parts.append("lower limb")
    elif row.get("pelvis", 0) == 1:
        parts.append("pelvis")

    return ", ".join(parts) + "."


def build_text_excel(split_df: pd.DataFrame, out_path: str):
    """
    Build MoNuSeg-like text file with columns:
      - Image: mask filename (e.g. IMG000001.png)
      - Description: short diagnostic sentence
    """
    rows = []
    for _, row in split_df.iterrows():
        img_id = os.path.splitext(str(row["image_id"]))[0] + ".png"
        rows.append({"Image": img_id, "Description": row_to_short_text(row)})
    pd.DataFrame(rows).to_excel(out_path, index=False)
    return len(rows)


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── 1) Load metadata ──────────────────────
    print(f"\n[1/5] Đọc metadata: {args.excel}")
    df = pd.read_excel(args.excel, sheet_name="Sheet1")
    print(f"      Tổng ảnh: {len(df)}")
    print(f"      Có mask (tumor=1): {df['tumor'].sum()}")
    print(f"      Không mask (tumor=0): {(df['tumor']==0).sum()}")

    if args.only_tumor:
        df = df[df["tumor"] == 1].copy().reset_index(drop=True)
        print(f"      Keep tumor-only images: {len(df)}")
    else:
        print("      Keep all images (tumor and non-tumor)")

    # ── 2) Tạo stratum & split ────────────────
    print(f"\n[2/5] Stratified split {int(SPLIT_RATIO[0]*100)}/{int(SPLIT_RATIO[1]*100)}/{int(SPLIT_RATIO[2]*100)}...")
    df["stratum"] = df.apply(make_stratum, axis=1)
    train_df, val_df, test_df = stratified_split(df, SPLIT_RATIO, args.seed)
    for split_df in (train_df, val_df, test_df):
        split_df["is_labeled"] = True

    # Semi-supervised setting for train set:
    # labeled_fraction in {1.0, 0.5, 0.25}; remaining train samples are unlabeled (empty masks).
    rng = random.Random(args.seed)
    train_ids = train_df["image_id"].tolist()
    n_train = len(train_ids)
    n_labeled = int(round(n_train * args.labeled_fraction))
    n_labeled = max(1, min(n_labeled, n_train))
    labeled_ids = set(rng.sample(train_ids, n_labeled))
    train_df["is_labeled"] = train_df["image_id"].isin(labeled_ids)

    splits = [train_df, val_df, test_df]
    for name, sdf in zip(SPLIT_NAMES, splits):
        n_total   = len(sdf)
        if name == "Train_Folder":
            n_labeled_split = int(sdf["is_labeled"].sum())
            print(f"      {name:15s}: {n_total:4d} images  "
                  f"(labeled={n_labeled_split}, unlabeled={n_total - n_labeled_split}, "
                  f"fraction={args.labeled_fraction})")
        else:
            print(f"      {name:15s}: {n_total:4d} images  (labeled={n_total}, unlabeled=0)")

    # ── 3) Prepare text supervision ─────────────
    print("\n[3/5] Prepare MoNuSeg-like text descriptions...")

    # ── 4) Copy & process files ───────────────
    print(f"\n[4/5] Xử lý và copy files vào {args.out_dir} ...")

    # Index annotation json files (preferred if provided)
    if os.path.exists(args.ann_dir):
        available_annotations = {
            os.path.splitext(f)[0]: os.path.join(args.ann_dir, f)
            for f in os.listdir(args.ann_dir)
            if f.lower().endswith(".json")
        }
        print(f"      Found {len(available_annotations)} annotation files in {args.ann_dir}")
    else:
        available_annotations = {}
        print(f"      [WARN] ann_dir not found: {args.ann_dir}")

    # Index mask image files (fallback)
    if os.path.exists(args.mask_dir):
        available_masks = {
            os.path.splitext(f)[0]: os.path.join(args.mask_dir, f)
            for f in os.listdir(args.mask_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        }
        print(f"      Tìm thấy {len(available_masks)} mask files trong {args.mask_dir}")
    else:
        available_masks = {}
        print(f"      [WARN] Không tìm thấy mask_dir: {args.mask_dir}")

    stats = defaultdict(int)

    for split_name, split_df in zip(SPLIT_NAMES, splits):
        img_out_dir  = os.path.join(args.out_dir, split_name, "img")
        mask_out_dir = os.path.join(args.out_dir, split_name, "labelcol")
        os.makedirs(img_out_dir,  exist_ok=True)
        os.makedirs(mask_out_dir, exist_ok=True)

        desc = f"  {split_name}"
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=desc):
            raw_id  = str(row["image_id"])                    # IMG000001.jpeg
            base_id = os.path.splitext(raw_id)[0]             # IMG000001
            out_name = base_id + ".png"                       # chuẩn hóa sang .png

            # ── Ảnh ──
            img_src = find_image_file(args.img_dir, raw_id)
            if img_src is None:
                stats["img_missing"] += 1
                # Tạo ảnh đen giả nếu thiếu
                fake = Image.fromarray(
                    np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)
                )
                fake.save(os.path.join(img_out_dir, out_name))
            else:
                try:
                    process_image(img_src, os.path.join(img_out_dir, out_name), args.img_size)
                    stats["img_ok"] += 1
                except Exception as e:
                    stats["img_error"] += 1
                    print(f"\n  [ERR] {raw_id}: {e}")

            # ── Mask ──
            ann_src = available_annotations.get(base_id, None)
            mask_src = available_masks.get(base_id, None)
            if split_name == "Train_Folder" and not bool(row.get("is_labeled", True)):
                process_mask(None, os.path.join(mask_out_dir, out_name), args.img_size)
                stats["mask_unlabeled_train"] += 1
            elif ann_src is not None:
                try:
                    process_mask_from_annotation(ann_src, os.path.join(mask_out_dir, out_name), args.img_size)
                    stats["mask_from_ann"] += 1
                except Exception as e:
                    process_mask(None, os.path.join(mask_out_dir, out_name), args.img_size)
                    stats["mask_ann_error"] += 1
                    print(f"\n  [ERR] annotation {raw_id}: {e}")
            elif row.get("tumor", 0) == 0:
                # Ảnh không bệnh → mask rỗng (chắc chắn)
                process_mask(None, os.path.join(mask_out_dir, out_name), args.img_size)
                stats["mask_empty"] += 1
            elif mask_src is not None:
                # Ảnh có bệnh và có mask thật
                try:
                    process_mask(mask_src, os.path.join(mask_out_dir, out_name), args.img_size)
                    stats["mask_real"] += 1
                except Exception as e:
                    process_mask(None, os.path.join(mask_out_dir, out_name), args.img_size)
                    stats["mask_error"] += 1
                    print(f"\n  [ERR] mask {raw_id}: {e}")
            else:
                # Có bệnh nhưng không tìm thấy file mask → cảnh báo, dùng rỗng
                process_mask(None, os.path.join(mask_out_dir, out_name), args.img_size)
                stats["mask_notfound"] += 1

    # ── 5) Tạo text Excel cho mỗi split ──────
    print(f"\n[5/5] Create text Excel for each split...")
    for split_name, split_df, txt_name in zip(SPLIT_NAMES, splits, TEXT_NAMES):
        out_path = os.path.join(args.out_dir, split_name, txt_name)
        count = build_text_excel(split_df, out_path)
        print(f"      {txt_name:20s}: {count} rows")

    # ── Summary ───────────────────────────────
    print("\n" + "=" * 55)
    print("  HOÀN TẤT")
    print("=" * 55)
    print(f"  Ảnh copy OK       : {stats['img_ok']}")
    print(f"  Ảnh thiếu file    : {stats['img_missing']}")
    print(f"  Train unlabeled   : {stats['mask_unlabeled_train']}")
    print(f"  Mask từ annotation: {stats['mask_from_ann']}")
    print(f"  Mask ann lỗi      : {stats['mask_ann_error']}")
    print(f"  Mask thật (tumor) : {stats['mask_real']}")
    print(f"  Mask rỗng (normal): {stats['mask_empty']}")
    print(f"  Mask không tìm thấy: {stats['mask_notfound']}")
    print(f"  Mask lỗi          : {stats['mask_error']}")
    print()
    print("  Cấu trúc thư mục:")
    for split_name in SPLIT_NAMES:
        img_dir  = os.path.join(args.out_dir, split_name, "img")
        mask_dir = os.path.join(args.out_dir, split_name, "labelcol")
        n_img  = len(os.listdir(img_dir))  if os.path.exists(img_dir)  else 0
        n_mask = len(os.listdir(mask_dir)) if os.path.exists(mask_dir) else 0
        print(f"    {split_name}/")
        print(f"      img/      : {n_img} files")
        print(f"      labelcol/ : {n_mask} files")

    # Lưu split info để tham khảo sau
    split_log_path = os.path.join(args.out_dir, "split_info.csv")
    train_log = train_df.copy()
    train_log["split"] = "train"
    val_log = val_df.copy()
    val_log["split"] = "val"
    test_log = test_df.copy()
    test_log["split"] = "test"
    df_log = pd.concat([train_log, val_log, test_log], ignore_index=True)
    df_log[["image_id", "tumor", "benign", "malignant", "stratum", "split", "is_labeled"]].to_csv(
        split_log_path, index=False
    )
    print(f"\n  Split log: {split_log_path}")
    print(f"  (Dùng file này để kiểm tra lại ảnh nào thuộc split nào)\n")

    # ── Verify 1 ảnh ngẫu nhiên ───────────────
    _verify_sample(args.out_dir, train_df, args.img_size)


# ─────────────────────────────────────────────────────────────
#  Verify: load lại 1 ảnh để kiểm tra kích thước
# ─────────────────────────────────────────────────────────────
def _verify_sample(out_dir: str, train_df: pd.DataFrame, img_size: int):
    """Kiểm tra nhanh 1 ảnh + mask từ Train_Folder."""
    sample_id = os.path.splitext(str(train_df.iloc[0]["image_id"]))[0] + ".png"
    img_path  = os.path.join(out_dir, "Train_Folder", "img",      sample_id)
    mask_path = os.path.join(out_dir, "Train_Folder", "labelcol", sample_id)

    ok = True
    if os.path.exists(img_path):
        img = Image.open(img_path)
        assert img.size == (img_size, img_size), f"Ảnh sai size: {img.size}"
        assert img.mode == "RGB", f"Ảnh sai mode: {img.mode}"
        print(f"  [OK] img sample    : {sample_id}  size={img.size}  mode={img.mode}")
    else:
        print(f"  [WARN] Không tìm thấy ảnh sample: {img_path}")
        ok = False

    if os.path.exists(mask_path):
        mask = Image.open(mask_path)
        arr  = np.array(mask)
        assert mask.size == (img_size, img_size), f"Mask sai size: {mask.size}"
        uniq = np.unique(arr).tolist()
        print(f"  [OK] mask sample   : {sample_id}  size={mask.size}  unique={uniq}")
    else:
        print(f"  [WARN] Không tìm thấy mask sample: {mask_path}")
        ok = False

    if ok:
        print("\n  Verify passed — dataset sẵn sàng để train!\n")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chuẩn bị dataset BTRXD theo cấu trúc LViT"
    )
    parser.add_argument(
        "--excel",    default="dataset.xlsx",
        help="File metadata gốc (dataset.xlsx)",
    )
    parser.add_argument(
        "--img_dir",  default="raw_images/",
        help="Thư mục chứa ảnh gốc (IMG000001.jpeg ...)",
    )
    parser.add_argument(
        "--ann_dir", default="datasets/BTRXD/Annotations/",
        help="Folder LabelMe annotations (.json). Priority over --mask_dir",
    )
    parser.add_argument(
        "--mask_dir", default="raw_masks/",
        help="Thư mục chứa mask gốc (chỉ ảnh có bệnh)",
    )
    parser.add_argument(
        "--out_dir",  default="data/BTRXD/",
        help="Thư mục output",
    )
    parser.add_argument(
        "--img_size", type=int, default=224,
        help="Kích thước ảnh output (default: 224)",
    )
    parser.add_argument(
        "--seed",     type=int, default=42,
        help="Random seed để reproducible",
    )
    parser.add_argument(
        "--only_tumor", action="store_true",
        help="Use only tumor-positive images (ignore non-disease images)",
    )
    parser.add_argument(
        "--labeled_fraction", type=float, default=1.0,
        help="Train labeled fraction: use 1.0, 0.5, or 0.25",
    )
    args = parser.parse_args()

    # Validation
    if not os.path.exists(args.excel):
        raise FileNotFoundError(f"Không tìm thấy: {args.excel}")
    if not os.path.exists(args.img_dir):
        raise FileNotFoundError(
            f"Không tìm thấy img_dir: {args.img_dir}\n"
            f"Hãy đảm bảo thư mục ảnh gốc tồn tại."
        )
    if args.labeled_fraction not in (1.0, 0.5, 0.25):
        raise ValueError("--labeled_fraction must be one of: 1.0, 0.5, 0.25")

    main(args)