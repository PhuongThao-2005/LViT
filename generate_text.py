# -*- coding: utf-8 -*-
"""
generate_text.py
================
Generate short clinical text files for BTRXD in MoNuSeg/LViT format.

Input:
    - dataset.xlsx
Output:
    - BTRXD_text.xlsx with two columns: Image, Description

Usage:
    python generate_text.py --excel dataset.xlsx --out datasets/BTRXD/BTRXD_text.xlsx
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────
#  Cấu hình cột trong dataset.xlsx
# ─────────────────────────────────────────────
BONE_COLS = [
    "hand", "ulna", "radius", "humerus", "foot",
    "tibia", "fibula", "femur", "hip bone",
    "ankle-joint", "knee-joint", "hip-joint",
    "wrist-joint", "elbow-joint", "shoulder-joint",
]
TUMOR_TYPE_COLS = [
    "osteochondroma", "multiple osteochondromas", "simple bone cyst",
    "giant cell tumor", "osteofibroma", "synovial osteochondroma",
    "other bt", "osteosarcoma", "other mt",
]
VIEW_COLS  = ["frontal", "lateral", "oblique"]
LIMB_COLS  = ["upper limb", "lower limb", "pelvis"]


def row_to_text(row: pd.Series) -> str:
    """
    Standard template:
      [benign/malignant] + [body region] + [specific location] + [X-ray view]
    """
    if row.get("tumor", 0) != 1:
        tumor_tag = "benign"
    elif row.get("malignant", 0) == 1:
        tumor_tag = "malignant"
    else:
        tumor_tag = "benign"

    body_regions = [c for c in LIMB_COLS if row.get(c, 0) == 1]
    body_region = ", ".join(body_regions) if body_regions else "unspecified body region"

    bones = [c for c in BONE_COLS if row.get(c, 0) == 1]
    location = ", ".join(bones) if bones else "unspecified location"

    views = [c for c in VIEW_COLS if row.get(c, 0) == 1]
    xray_view = ", ".join(views) if views else "unspecified view"

    # Keep text short and stable:
    # [benign/malignant] + [body region] + [specific location] + [X-ray view]
    return f"{tumor_tag}; {body_region}; {location}; {xray_view}"


def build_text_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  build text"):
        image_name = str(row["image_id"])
        base = image_name.rsplit(".", 1)[0]
        image_key = base + ".png"
        rows.append({"Image": image_key, "Description": row_to_text(row)})
    return pd.DataFrame(rows)


def write_split_texts(df_all: pd.DataFrame, split_csv: str, out_root: str) -> None:
    split_df = pd.read_csv(split_csv)
    merged = df_all.merge(split_df[["image_id", "split"]], on="image_id", how="inner")
    split_to_folder = {
        "train": ("Train_Folder", "Train_text.xlsx"),
        "val": ("Val_Folder", "Val_text.xlsx"),
        "test": ("Test_Folder", "Test_text.xlsx"),
    }
    for split_key, (folder, file_name) in split_to_folder.items():
        sub = merged[merged["split"].str.lower() == split_key].copy()
        if sub.empty:
            continue
        out_df = build_text_dataframe(sub)
        out_dir = os.path.join(out_root, folder)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, file_name)
        out_df.to_excel(out_path, index=False)
        print(f"Saved {len(out_df)} rows -> {out_path}")


def main(args):
    print(f"[1/2] Read metadata: {args.excel}")
    df = pd.read_excel(args.excel, sheet_name="Sheet1")
    print(f"      Total images: {len(df)}")

    if args.split_csv and args.out_root:
        print("[2/2] Build split text files (Train/Val/Test)...")
        write_split_texts(df, args.split_csv, args.out_root)
        return

    print("[2/2] Build single clinical text file...")
    out_df = build_text_dataframe(df)
    out_df.to_excel(args.out, index=False)
    print(f"\nSaved: {args.out}")
    print("Expected columns:", out_df.columns.tolist())
    print("\nSample rows:")
    print(out_df.head(5).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BTRXD text file in MoNuSeg style")
    parser.add_argument("--excel", default="dataset.xlsx", help="Path to dataset.xlsx")
    parser.add_argument("--out", default="datasets/BTRXD/BTRXD_text.xlsx", help="Output text xlsx path")
    parser.add_argument("--split-csv", default="", help="split_info.csv with image_id, split columns")
    parser.add_argument("--out-root", default="", help="Dataset root containing Train_Folder/Val_Folder/Test_Folder")
    args = parser.parse_args()
    main(args)