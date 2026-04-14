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
    """Create short diagnosis text similar to MoNuSeg style."""
    parts = []

    bones = [c for c in BONE_COLS if row.get(c, 0) == 1]
    if bones:
        parts.append("Location: " + ", ".join(bones))

    limbs = [c for c in LIMB_COLS if row.get(c, 0) == 1]
    if limbs:
        parts.append("Region: " + ", ".join(limbs))

    views = [c for c in VIEW_COLS if row.get(c, 0) == 1]
    if views:
        parts.append("View: " + ", ".join(views))

    if row.get("tumor", 0) == 1:
        diagnosis = "malignant bone tumor" if row.get("malignant", 0) == 1 else "benign bone tumor"
        tumor_types = [c for c in TUMOR_TYPE_COLS if row.get(c, 0) == 1]
        if tumor_types:
            diagnosis += " (" + ", ".join(tumor_types) + ")"
        parts.append("Diagnosis: " + diagnosis)
    else:
        parts.append("Diagnosis: normal bone")

    return ". ".join(parts) + "."


def main(args):
    print(f"[1/2] Read metadata: {args.excel}")
    df = pd.read_excel(args.excel, sheet_name="Sheet1")
    print(f"      Total images: {len(df)}")

    print("[2/2] Build short clinical text...")
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  build text"):
        image_name = str(row["image_id"])
        base = image_name.rsplit(".", 1)[0]
        image_key = base + ".png"
        rows.append({"Image": image_key, "Description": row_to_text(row)})

    out_df = pd.DataFrame(rows)
    out_df.to_excel(args.out, index=False)
    print(f"\nSaved: {args.out}")
    print("Expected columns:", out_df.columns.tolist())
    print("\nSample rows:")
    print(out_df.head(5).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BTRXD text file in MoNuSeg style")
    parser.add_argument("--excel", default="dataset.xlsx", help="Path to dataset.xlsx")
    parser.add_argument("--out", default="datasets/BTRXD/BTRXD_text.xlsx", help="Output text xlsx path")
    args = parser.parse_args()
    main(args)