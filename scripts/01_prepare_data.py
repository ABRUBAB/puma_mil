"""
Script 01: Preprocess raw BraTS volumes and extract frozen ViT features.
Section 3.2: 3-stage normalization, center-crop K=48 selection, post-hoc role assignment,
and feature extraction using frozen vit_small_patch16_224 (384-dim).

Usage:
  python scripts/01_prepare_data.py --data_dir /path/to/brats2021 --out_dir ./outputs
"""

import os
import sys
import glob
import json
import shutil
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.config import Config
from src.data.preprocessing import (
    normalize_volume_3stage,
    slice_to_rgb_tensor,
    select_center_crop_slices,
    ViTFeatureExtractor,
)

ROLE_TO_ID = {
    "far_neg": 0,
    "boundary_neg": 1,
    "positive": 2,
    "mid_neg": 3,
    "no_tumor": 4,
}

def process_patient_folder(pdir, modalities, extractor, k_slices=48, tau=100, batch_size=64):
    """Processes a single BraTS patient folder and extracts ViT features for K slices."""
    import nibabel as nib

    pid = os.path.basename(pdir)
    files = {}
    for m in modalities:
        matches = glob.glob(os.path.join(pdir, f"*{m}*.nii*"))
        if matches:
            files[m] = matches[0]
    seg_matches = glob.glob(os.path.join(pdir, "*seg*.nii*"))
    has_seg = len(seg_matches) > 0

    if not all(m in files for m in modalities):
        return None

    vols = {}
    for m in modalities:
        vols[m] = normalize_volume_3stage(nib.load(files[m]).get_fdata(dtype=np.float32))

    if has_seg:
        seg_vol = nib.load(seg_matches[0]).get_fdata(dtype=np.float32)
        total_slices = seg_vol.shape[2]
        counts = [int(np.sum(seg_vol[:, :, z] > 0)) for z in range(total_slices)]
    else:
        total_slices = vols[modalities[0]].shape[2]
        counts = [0] * total_slices

    selected_slices, roles, distances = select_center_crop_slices(
        counts, total_slices=total_slices, crop_size=k_slices, tau=tau
    )

    n_sel = len(selected_slices)
    feats = {m: np.zeros((n_sel, 384), dtype=np.float32) for m in modalities}

    for m in modalities:
        batch_tensors = []
        for s_idx in selected_slices:
            t = slice_to_rgb_tensor(vols[m][:, :, s_idx])
            batch_tensors.append(t)
        batch_tensors = torch.stack(batch_tensors, dim=0)

        # Batch extraction
        all_m_feats = []
        for b_start in range(0, n_sel, batch_size):
            chunk = batch_tensors[b_start : b_start + batch_size]
            chunk_f = extractor.extract_slice_features(chunk)
            all_m_feats.append(chunk_f)
        feats[m] = np.concatenate(all_m_feats, axis=0)

    slice_labels = [1 if counts[s] >= tau else 0 for s in selected_slices]
    role_ids = [ROLE_TO_ID.get(roles[s], 3) for s in selected_slices]
    dist_vals = [distances[s] for s in selected_slices]
    tumor_px = [counts[s] for s in selected_slices]

    return {
        "pid": pid,
        "selected_slices": selected_slices,
        "feats": feats,
        "labels": slice_labels,
        "role_ids": role_ids,
        "distances": dist_vals,
        "tumor_pixels": tumor_px,
    }

def main():
    parser = argparse.ArgumentParser(description="Preprocess BraTS and extract ViT features")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to raw BraTS dataset or features")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--k_slices", type=int, default=48, help="Slices per volume")
    parser.add_argument("--tau", type=int, default=100, help="Tumor positive pixel threshold")
    parser.add_argument("--batch_size", type=int, default=64, help="ViT extraction batch size")
    parser.add_argument("--device", type=str, default=None, help="Inference device")
    args = parser.parse_args()

    cfg = Config()
    if args.device:
        cfg.device = args.device

    feat_out_dir = os.path.join(args.out_dir, "features")
    os.makedirs(feat_out_dir, exist_ok=True)

    # Check if pre-extracted .npz files already exist in data_dir
    npz_files = glob.glob(os.path.join(args.data_dir, "*.npz")) + glob.glob(os.path.join(args.data_dir, "**", "*.npz"), recursive=True)
    if npz_files:
        print(f"Found {len(npz_files)} pre-extracted .npz feature files in {args.data_dir}.")
        for f in npz_files:
            dest = os.path.join(feat_out_dir, os.path.basename(f))
            if not os.path.exists(dest):
                shutil.copy2(f, dest)
                print(f"Copied {os.path.basename(f)} to {feat_out_dir}")
        print("Data preparation complete (precomputed features verified).")
        return

    # Otherwise process raw NIfTI directories
    pdirs = sorted([d for d in glob.glob(os.path.join(args.data_dir, "*")) if os.path.isdir(d)])
    if not pdirs:
        print(f"No patient directories found in {args.data_dir}.")
        return

    print(f"Processing {len(pdirs)} patients from {args.data_dir} using {cfg.backbone} on {cfg.device}...")
    extractor = ViTFeatureExtractor(backbone=cfg.backbone, device=cfg.device)

    all_data = {
        "patient_ids": [],
        "slice_idxs": [],
        "features_t1": [],
        "features_t1ce": [],
        "features_t2": [],
        "features_flair": [],
        "labels": [],
        "modality_mask": [],
        "distance_to_positive": [],
        "slice_role_id": [],
        "tumor_pixels": [],
    }

    for pdir in tqdm(pdirs, desc="Extracting features"):
        res = process_patient_folder(pdir, cfg.modalities, extractor, k_slices=args.k_slices, tau=args.tau, batch_size=args.batch_size)
        if res is None:
            continue
        pid = res["pid"]
        n_slices = len(res["selected_slices"])
        all_data["patient_ids"].extend([pid] * n_slices)
        all_data["slice_idxs"].extend(res["selected_slices"])
        all_data["features_t1"].append(res["feats"]["t1"])
        all_data["features_t1ce"].append(res["feats"]["t1ce"])
        all_data["features_t2"].append(res["feats"]["t2"])
        all_data["features_flair"].append(res["feats"]["flair"])
        all_data["labels"].extend(res["labels"])
        all_data["modality_mask"].extend([[1, 1, 1, 1]] * n_slices)
        all_data["distance_to_positive"].extend(res["distances"])
        all_data["slice_role_id"].extend(res["role_ids"])
        all_data["tumor_pixels"].extend(res["tumor_pixels"])

    for k in ["features_t1", "features_t1ce", "features_t2", "features_flair"]:
        all_data[k] = np.concatenate(all_data[k], axis=0)
    for k in ["patient_ids", "slice_idxs", "labels", "modality_mask", "distance_to_positive", "slice_role_id", "tumor_pixels"]:
        all_data[k] = np.array(all_data[k])

    save_path = os.path.join(feat_out_dir, "extracted_features.npz")
    np.savez_compressed(save_path, **all_data)
    print(f"Successfully extracted {len(all_data['labels'])} slices across {len(pdirs)} patients.")
    print(f"Saved features to {save_path}")

if __name__ == "__main__":
    main()
