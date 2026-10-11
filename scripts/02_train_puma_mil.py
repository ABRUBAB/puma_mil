"""
Script 02: Train PUMA-MIL ensemble across 3 random seeds (42, 123, 456).
Algorithm 2 & Section 3.4:
- Reliability-weighted modality fusion with 10% stochastic modality dropout.
- 2-layer Transformer sequence encoder over W=11 axial window.
- Hybrid attention + top-K pooling with EMA prototype similarity.
- Multi-objective loss: role-aware asymmetric focal loss, bag loss, smoothness loss, diversity loss.
- Post-training: threshold tuning on tune split, temperature scaling, 3-seed ensemble averaging.
- Bootstrap 95% confidence intervals (2,000 resamples).

Usage:
  python scripts/02_train_puma_mil.py --data_dir ./outputs/features --out_dir ./outputs
"""

import os
import sys
import json
import glob
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.config import Config
from src.models.puma_mil import PUMAMILLite
from src.data.dataset import CenterWindowDataset, make_role_weighted_sampler
from src.training.losses import PUMALoss
from src.training.trainer import train_one_epoch, evaluate_model
from src.evaluation.metrics import metrics_from_probs, find_best_threshold, bootstrap_ci
from src.evaluation.temperature_scaling import fit_temperature, apply_temperature

def load_split_dict(data_dir, split_name):
    """Finds and loads .npz feature file for a split."""
    candidates = [
        os.path.join(data_dir, f"{split_name}_features.npz"),
        os.path.join(data_dir, f"{split_name}.npz"),
        os.path.join(data_dir, f"features_{split_name}.npz"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return dict(np.load(c, allow_pickle=True))
    # Fallback to single combined file if split_name in it
    for f in glob.glob(os.path.join(data_dir, "*.npz")):
        d = dict(np.load(f, allow_pickle=True))
        if f"{split_name}_features_t1" in d:
            prefix = f"{split_name}_"
            return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}
    raise FileNotFoundError(f"Could not find features for split '{split_name}' in {data_dir}")

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train PUMA-MIL Model Ensemble")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to precomputed features")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Ensemble random seeds")
    parser.add_argument("--epochs", type=int, default=32, help="Max training epochs per seed")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate")
    parser.add_argument("--window_size", type=int, default=11, help="Axial window width")
    parser.add_argument("--top_k", type=int, default=4, help="Top-K instances for hybrid pooling")
    parser.add_argument("--proto_momentum", type=float, default=0.90, help="EMA prototype momentum")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    cfg = Config()
    if args.device:
        cfg.device = args.device
    cfg.work_dir = args.out_dir
    cfg.seeds = args.seeds
    cfg.total_epochs = args.epochs
    cfg.early_stopping_patience = args.patience
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.train_window = args.window_size
    cfg.top_k = args.top_k
    cfg.proto_momentum = args.proto_momentum

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    print(f"Loading data splits from {args.data_dir}...")
    train_dict = load_split_dict(args.data_dir, "train")
    val_dict = load_split_dict(args.data_dir, "val")
    tune_dict = load_split_dict(args.data_dir, "tune")
    test_dict = load_split_dict(args.data_dir, "test")

    train_ds = CenterWindowDataset(train_dict, window_size=cfg.train_window)
    val_ds = CenterWindowDataset(val_dict, window_size=cfg.train_window)
    tune_ds = CenterWindowDataset(tune_dict, window_size=cfg.train_window)
    test_ds = CenterWindowDataset(test_dict, window_size=cfg.train_window)

    print(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Tune={len(tune_ds)}, Test={len(test_ds)}")

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    tune_loader = DataLoader(tune_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # Class balance weights
    train_labels = [item["center_label"] for item in train_ds.index]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    w0 = len(train_labels) / (2.0 * max(neg_count, 1))
    w1 = len(train_labels) / (2.0 * max(pos_count, 1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=cfg.device)

    seed_models = []
    seed_temperatures = []
    seed_thresholds = []
    seed_test_probs = []
    histories = {}

    for seed in cfg.seeds:
        print(f"\n{'='*30} Seed {seed} {'='*30}")
        set_seed(seed)

        sampler = make_role_weighted_sampler(train_ds, cfg)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler)

        model = PUMAMILLite(
            d_in=384,
            d_model=cfg.d_model,
            max_slices=cfg.max_slices,
            top_k=cfg.top_k,
            proto_momentum=cfg.proto_momentum,
            modality_dropout=cfg.modality_dropout,
        ).to(cfg.device)

        criterion = PUMALoss(cfg, class_weights=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        best_val_f1 = -1.0
        best_state = None
        patience_left = cfg.early_stopping_patience
        seed_history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}

        for epoch in range(1, cfg.total_epochs + 1):
            t_loss, t_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, cfg.device, clip_grad=cfg.grad_clip_norm
            )
            v_logits, v_labels, _, _, _ = evaluate_model(model, val_loader, cfg.device)
            v_probs = torch.softmax(torch.from_numpy(v_logits), dim=-1).numpy()
            v_metrics = metrics_from_probs(v_probs, v_labels, threshold=0.5)

            seed_history["train_loss"].append(t_loss)
            seed_history["val_macro_f1"].append(v_metrics["macro_f1"])

            if v_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = v_metrics["macro_f1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_left = cfg.early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"Early stopping at epoch {epoch} (Best Val Macro-F1: {best_val_f1:.4f})")
                    break

        histories[seed] = seed_history
        model.load_state_dict(best_state)
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_seed{seed}.pth")
        torch.save(best_state, ckpt_path)
        print(f"Saved best model checkpoint for seed {seed} -> {ckpt_path}")

        # Post-training calibration on tune split
        tune_logits, tune_labels, _, _, _ = evaluate_model(model, tune_loader, cfg.device)
        T = fit_temperature(tune_logits, tune_labels)
        seed_temperatures.append(T)
        cal_tune_probs = apply_temperature(tune_logits, T)
        t_star, _ = find_best_threshold(cal_tune_probs, tune_labels)
        seed_thresholds.append(t_star)

        # Test inference
        test_logits, test_labels, _, _, _ = evaluate_model(model, test_loader, cfg.device)
        cal_test_probs = apply_temperature(test_logits, T)
        seed_test_probs.append(cal_test_probs)
        seed_metrics = metrics_from_probs(cal_test_probs, test_labels, threshold=t_star)
        print(f"Seed {seed} Test: Macro-F1={seed_metrics['macro_f1']:.4f}, AUC={seed_metrics['auc']:.4f}, Acc={seed_metrics['accuracy']:.4f}")

    # Build 3-seed Ensemble
    mean_test_probs = np.mean(seed_test_probs, axis=0)
    avg_thresh = float(np.mean(seed_thresholds))
    ensemble_metrics = metrics_from_probs(mean_test_probs, test_labels, threshold=avg_thresh)
    ci_results = bootstrap_ci(mean_test_probs, test_labels, threshold=avg_thresh, n_resamples=2000, seed=42)

    print("\n" + "=" * 65)
    print(" PUMA-MIL 3-Seed Ensemble Test Evaluation")
    print("=" * 65)
    for m in ["accuracy", "macro_f1", "auc", "pr_auc", "sensitivity", "specificity", "balanced_acc", "ece"]:
        val = ensemble_metrics.get(m, float("nan"))
        ci_str = f" [{ci_results[m]['ci_low']:.4f}, {ci_results[m]['ci_high']:.4f}]" if m in ci_results else ""
        print(f"  {m:15s}: {val:.4f}{ci_str}")
    print("=" * 65)

    out_metrics = {
        "ensemble_metrics": ensemble_metrics,
        "bootstrap_ci": ci_results,
        "threshold": avg_thresh,
        "seed_temperatures": seed_temperatures,
        "seed_thresholds": seed_thresholds,
    }
    with open(os.path.join(res_dir, "ensemble_test_metrics.json"), "w") as f:
        json.dump(out_metrics, f, indent=2)

    with open(os.path.join(res_dir, "training_histories.json"), "w") as f:
        json.dump(histories, f, indent=2)

    print(f"Training completed successfully. Results saved in {res_dir}")

if __name__ == "__main__":
    main()
