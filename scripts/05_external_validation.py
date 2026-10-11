"""
Script 05: Fully Locked External Transfer on BraTS-2020 (Section 4.4 & Figure 13).
Evaluates the internal 3-seed ensemble on 17,664 slices across 369 BraTS-2020 patients
WITHOUT retraining, fine-tuning, or threshold adaptation.

Usage:
  python scripts/05_external_validation.py --checkpoints_dir ./outputs/checkpoints --data_dir ./data/brats2020 --out_dir ./outputs
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.config import Config
from src.models.puma_mil import PUMAMILLite
from src.data.dataset import CenterWindowDataset
from src.evaluation.metrics import metrics_from_probs
from src.evaluation.temperature_scaling import apply_temperature

def main():
    parser = argparse.ArgumentParser(description="Locked External Validation on BraTS-2020")
    parser.add_argument("--checkpoints_dir", type=str, default="./outputs/checkpoints", help="Path to trained model checkpoints")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to BraTS-2020 precomputed features")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.42, help="Decision threshold tuned on internal tune split")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    # Locate external dataset features
    candidates = [
        os.path.join(args.data_dir, "external_2020_features.npz"),
        os.path.join(args.data_dir, "brats2020_features.npz"),
        os.path.join(args.data_dir, "test_features.npz"),
    ]
    feat_file = None
    for c in candidates:
        if os.path.isfile(c):
            feat_file = c
            break
    if feat_file is None:
        npzs = glob.glob(os.path.join(args.data_dir, "*.npz"))
        if npzs:
            feat_file = npzs[0]

    if feat_file is None:
        print(f"Error: No .npz features found in {args.data_dir}.")
        return

    print(f"Loading external features from {feat_file}...")
    data_dict = dict(np.load(feat_file, allow_pickle=True))
    ds = CenterWindowDataset(data_dict, window_size=11)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    print(f"External dataset contains {len(ds)} instances across {len(ds.patient_data)} patients.")

    ckpt_files = sorted(glob.glob(os.path.join(args.checkpoints_dir, "*.pth")))
    if not ckpt_files:
        print(f"Error: No checkpoints found in {args.checkpoints_dir}.")
        return

    print(f"Evaluating {len(ckpt_files)} ensemble checkpoints with fixed threshold t={args.threshold}...")
    all_seed_probs = []
    all_mod_weights = []

    for ckpt_path in ckpt_files:
        model = PUMAMILLite(d_in=384, d_model=128, max_slices=48, top_k=4).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        seed_logits = []
        seed_weights = []
        labels_list = []

        with torch.no_grad():
            for batch in loader:
                t1, t1ce, t2, flair, _, center_lbl, seq_mask, mod_mask, center_pos, _, _, _, _, _ = batch
                t1, t1ce, t2, flair = t1.to(device), t1ce.to(device), t2.to(device), flair.to(device)
                seq_mask, mod_mask = seq_mask.to(device), mod_mask.to(device)

                _, slice_logits, _, mod_w, _, _ = model(
                    t1, t1ce, t2, flair, seq_mask, mod_mask, return_aux=True
                )
                c_idx = center_pos[0].item()
                seed_logits.append(slice_logits[:, c_idx, :].cpu())
                seed_weights.append(mod_w[:, c_idx, :].cpu())
                labels_list.append(center_lbl)

        logits = torch.cat(seed_logits, dim=0).numpy()
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        all_seed_probs.append(probs)
        all_mod_weights.append(torch.cat(seed_weights, dim=0).numpy())

    mean_probs = np.mean(all_seed_probs, axis=0)
    labels = torch.cat(labels_list, dim=0).numpy()
    metrics = metrics_from_probs(mean_probs, labels, threshold=args.threshold)

    print("\n" + "=" * 65)
    print(f" BraTS-2020 Fully Locked External Validation Results (t={args.threshold})")
    print("=" * 65)
    for m in ["accuracy", "macro_f1", "auc", "pr_auc", "sensitivity", "specificity", "balanced_acc"]:
        print(f"  {m:15s}: {metrics.get(m, float('nan')):.4f}")
    print("=" * 65)

    out_file = os.path.join(res_dir, "external_validation_metrics.json")
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved external validation metrics to {out_file}")

if __name__ == "__main__":
    main()
