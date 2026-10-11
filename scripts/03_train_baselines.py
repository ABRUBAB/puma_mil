"""
Script 03: Train competitive MIL baselines under identical fair protocol.
Models: ABMIL, TransMIL, CLAM-SB, ViTSliceMLP.
Guarantees:
- Same patient splits and frozen ViT-S/16 features
- Same W=11 axial window and center slice targets
- Same 3 random seeds (42, 123, 456)
- Same AdamW optimizer (lr=8e-4, weight_decay=1e-4) and early stopping patience (8)
- Same tune-split temperature scaling and threshold tuning
- Same evaluation metrics and test set

Usage:
  python scripts/03_train_baselines.py --data_dir ./outputs/features --out_dir ./outputs --model all
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.config import Config
from src.models.baselines import ABMIL, TransMIL, CLAM_SB, ViTSliceMLP
from src.data.dataset import CenterWindowDataset
from src.evaluation.metrics import metrics_from_probs, find_best_threshold, bootstrap_ci
from src.evaluation.temperature_scaling import fit_temperature, apply_temperature

def load_split_dict(data_dir, split_name):
    for name in [f"{split_name}_features.npz", f"{split_name}.npz", f"features_{split_name}.npz"]:
        p = os.path.join(data_dir, name)
        if os.path.isfile(p):
            return dict(np.load(p, allow_pickle=True))
    raise FileNotFoundError(f"Could not find features for split '{split_name}' in {data_dir}")

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_class_balanced_sampler(dataset: CenterWindowDataset) -> WeightedRandomSampler:
    labels = [item["center_label"] for item in dataset.index]
    pos = sum(labels)
    neg = len(labels) - pos
    w_pos = len(labels) / (2.0 * max(pos, 1))
    w_neg = len(labels) / (2.0 * max(neg, 1))
    weights = [w_pos if y == 1 else w_neg for y in labels]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), replacement=True)

def train_baseline_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        t1, t1ce, t2, flair, _, center_lbl, seq_mask, _, _, _, _, _, _, _ = batch
        t1, t1ce, t2, flair = t1.to(device), t1ce.to(device), t2.to(device), flair.to(device)
        center_lbl, seq_mask = center_lbl.to(device), seq_mask.to(device)

        optimizer.zero_grad()
        out = model(t1, t1ce, t2, flair, seq_mask)
        logits = out[0] if isinstance(out, tuple) else out
        loss = criterion(logits, center_lbl)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(center_lbl)
        correct += (logits.argmax(dim=-1) == center_lbl).sum().item()
        total += len(center_lbl)
    return total_loss / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def eval_baseline(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        t1, t1ce, t2, flair, _, center_lbl, seq_mask, _, _, _, _, _, _, _ = batch
        t1, t1ce, t2, flair = t1.to(device), t1ce.to(device), t2.to(device), flair.to(device)
        seq_mask = seq_mask.to(device)

        out = model(t1, t1ce, t2, flair, seq_mask)
        logits = out[0] if isinstance(out, tuple) else out
        all_logits.append(logits.cpu())
        all_labels.append(center_lbl)
    return torch.cat(all_logits, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()

def main():
    parser = argparse.ArgumentParser(description="Train Competitive Baselines")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to features")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--model", type=str, choices=["abmil", "transmil", "clam", "vitmlp", "all"], default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    train_ds = CenterWindowDataset(load_split_dict(args.data_dir, "train"), window_size=11)
    val_ds = CenterWindowDataset(load_split_dict(args.data_dir, "val"), window_size=11)
    tune_ds = CenterWindowDataset(load_split_dict(args.data_dir, "tune"), window_size=11)
    test_ds = CenterWindowDataset(load_split_dict(args.data_dir, "test"), window_size=11)

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    tune_loader = DataLoader(tune_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    models_to_run = ["abmil", "transmil", "clam", "vitmlp"] if args.model == "all" else [args.model]
    all_results = {}

    model_factories = {
        "abmil": lambda: ABMIL(d_in=384, d_hidden=256),
        "transmil": lambda: TransMIL(d_in=384, d_model=128),
        "clam": lambda: CLAM_SB(d_in=384, d_hidden=256),
        "vitmlp": lambda: ViTSliceMLP(d_in=384, window_size=11),
    }

    for m_name in models_to_run:
        print(f"\n{'='*25} Training Baseline: {m_name.upper()} {'='*25}")
        seed_test_probs, seed_thresholds = [], []

        for seed in args.seeds:
            set_seed(seed)
            sampler = make_class_balanced_sampler(train_ds)
            train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)

            model = model_factories[m_name]().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            best_f1, best_state = -1.0, None
            patience = args.patience

            for epoch in range(1, args.epochs + 1):
                train_baseline_epoch(model, train_loader, criterion, optimizer, device)
                v_logits, v_labels = eval_baseline(model, val_loader, device)
                v_probs = torch.softmax(torch.from_numpy(v_logits), dim=-1).numpy()
                v_f1 = metrics_from_probs(v_probs, v_labels, threshold=0.5)["macro_f1"]

                if v_f1 > best_f1:
                    best_f1 = v_f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience = args.patience
                else:
                    patience -= 1
                    if patience <= 0:
                        break

            model.load_state_dict(best_state)
            tune_logits, tune_labels = eval_baseline(model, tune_loader, device)
            T = fit_temperature(tune_logits, tune_labels)
            t_star, _ = find_best_threshold(apply_temperature(tune_logits, T), tune_labels)
            seed_thresholds.append(t_star)

            test_logits, test_labels = eval_baseline(model, test_loader, device)
            cal_probs = apply_temperature(test_logits, T)
            seed_test_probs.append(cal_probs)

        ens_probs = np.mean(seed_test_probs, axis=0)
        t_final = float(np.mean(seed_thresholds))
        ens_metrics = metrics_from_probs(ens_probs, test_labels, threshold=t_final)
        ci = bootstrap_ci(ens_probs, test_labels, threshold=t_final, n_resamples=2000, seed=42)

        print(f"Results for {m_name.upper()}:")
        print(f"  Macro-F1: {ens_metrics['macro_f1']:.4f} [{ci['macro_f1']['ci_low']:.4f}, {ci['macro_f1']['ci_high']:.4f}]")
        print(f"  AUC:      {ens_metrics['auc']:.4f}")
        print(f"  Accuracy: {ens_metrics['accuracy']:.4f}")

        all_results[m_name] = {
            "metrics": ens_metrics,
            "bootstrap_ci": ci,
            "threshold": t_final,
        }

    out_file = os.path.join(res_dir, "baseline_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll baseline results saved to {out_file}")

if __name__ == "__main__":
    main()
