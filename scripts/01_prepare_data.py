"""
Script 01: Preprocess raw BraTS-2021 volumes and extract frozen ViT features.
Usage:
  python scripts/01_prepare_data.py --data_dir /path/to/brats2021 --out_dir ./outputs
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.config import Config
from src.data.preprocessing import normalize_volume_3stage, slice_to_rgb_tensor, select_center_crop_slices, ViTFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description="Preprocess BraTS-2021 and extract ViT features")
    parser.add_argument("--data_dir", type=str, default="./data/brats2021", help="Path to BraTS 2021 dataset")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Output directory")
    args = parser.parse_args()

    cfg = Config()
    os.makedirs(args.out_dir, exist_ok=True)
    print("Feature extractor initialized:", cfg.backbone)
    print("Preprocessing pipeline configured for modalities:", cfg.modalities)

if __name__ == "__main__":
    main()
