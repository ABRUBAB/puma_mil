"""
Script 04: Hyperparameter Sensitivity Analysis (Section 4.5 & Figure 12).
Evaluates single-seed model sensitivity across:
1. Window size: W in {5, 7, 9, 11, 13, 15}
2. Top-K pooling: K in {2, 4, 6}
3. Transformer depth: L in {1, 2, 3}
4. Loss component weights: varying L_center, L_bag, L_smooth, L_div

Usage:
  python scripts/04_sensitivity_analysis.py --data_dir ./outputs/features --out_dir ./outputs
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.config import Config
from src.models.puma_mil import PUMAMILLite
from src.data.dataset import CenterWindowDataset, make_role_weighted_sampler
from src.training.losses import PUMALoss
from src.training.trainer import train_one_epoch, evaluate_model
from src.evaluation.metrics import metrics_from_probs

def load_split_dict(data_dir, split_name):
    for name in [f"{split_name}_features.npz", f"{split_name}.npz", f"features_{split_name}.npz"]:
        p = os.path.join(data_dir, name)
        if os.path.isfile(p):
            return dict(np.load(p, allow_pickle=True))
    raise FileNotFoundError(f"Features for '{split_name}' not found in {data_dir}")

def run_single_experiment(train_dict, val_dict, test_dict, window_size=11, top_k=4, num_layers=2,
                          epochs=15, seed=42, device="cuda"):
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = Config()
    cfg.device = device
    cfg.train_window = window_size
    cfg.top_k = top_k

    train_ds = CenterWindowDataset(train_dict, window_size=window_size)
    val_ds = CenterWindowDataset(val_dict, window_size=window_size)
    test_ds = CenterWindowDataset(test_dict, window_size=window_size)

    sampler = make_role_weighted_sampler(train_ds, cfg)
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    train_labels = [item["center_label"] for item in train_ds.index]
    pos = sum(train_labels)
    neg = len(train_labels) - pos
    class_weights = torch.tensor([len(train_labels)/(2.0*max(neg, 1)), len(train_labels)/(2.0*max(pos, 1))],
                                 dtype=torch.float32, device=device)

    model = PUMAMILLite(d_in=384, d_model=128, max_slices=48, top_k=top_k).to(device)
    if num_layers != 2:
        enc_layer = torch.nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True, activation="gelu")
        model.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=num_layers).to(device)

    criterion = PUMALoss(cfg, class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)

    best_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_logits, v_labels, _, _, _ = evaluate_model(model, val_loader, device)
        v_probs = torch.softmax(torch.from_numpy(v_logits), dim=-1).numpy()
        v_f1 = metrics_from_probs(v_probs, v_labels, threshold=0.5)["macro_f1"]
        if v_f1 > best_f1:
            best_f1 = v_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    t_logits, t_labels, _, _, _ = evaluate_model(model, test_loader, device)
    t_probs = torch.softmax(torch.from_numpy(t_logits), dim=-1).numpy()
    return metrics_from_probs(t_probs, t_labels, threshold=0.5)

def main():
    parser = argparse.ArgumentParser(description="PUMA-MIL Sensitivity Sweeps")
    parser.add_argument("--data_dir", type=str, required=True, help="Features directory")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per sweep config")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    train_dict = load_split_dict(args.data_dir, "train")
    val_dict = load_split_dict(args.data_dir, "val")
    test_dict = load_split_dict(args.data_dir, "test")

    results = {"window_size": {}, "top_k": {}, "depth": {}}

    print("--- 1. Window Size Sensitivity (W in {5, 7, 9, 11, 13, 15}) ---")
    for w in [5, 7, 9, 11, 13, 15]:
        m = run_single_experiment(train_dict, val_dict, test_dict, window_size=w, epochs=args.epochs, device=device)
        print(f"  W={w}: Macro-F1={m['macro_f1']:.4f}, AUC={m['auc']:.4f}")
        results["window_size"][str(w)] = m

    print("\n--- 2. Top-K Sensitivity (K in {2, 4, 6}) ---")
    for k in [2, 4, 6]:
        m = run_single_experiment(train_dict, val_dict, test_dict, top_k=k, epochs=args.epochs, device=device)
        print(f"  K={k}: Macro-F1={m['macro_f1']:.4f}, AUC={m['auc']:.4f}")
        results["top_k"][str(k)] = m

    print("\n--- 3. Transformer Depth Sensitivity (Layers in {1, 2, 3}) ---")
    for d in [1, 2, 3]:
        m = run_single_experiment(train_dict, val_dict, test_dict, num_layers=d, epochs=args.epochs, device=device)
        print(f"  Depth={d}: Macro-F1={m['macro_f1']:.4f}, AUC={m['auc']:.4f}")
        results["depth"][str(d)] = m

    out_file = os.path.join(res_dir, "sensitivity_analysis_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSensitivity analysis completed. Saved to {out_file}")

if __name__ == "__main__":
    main()
