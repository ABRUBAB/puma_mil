# ============================================================
# PUMA-MIL CLEAN RUN — CELL 3: JOURNAL-STANDARD FIGURES
# Generates Fig01–Fig13 at 600 DPI for Q1 journal submission
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score, balanced_accuracy_score,
    recall_score
)

# ─── CONFIG ──────────────────────────────────────────────────
WORK = "/kaggle/working"
INPUT = "/kaggle/input/datasets/absanim/phase-1-output"
OUT = os.path.join(WORK, "latest_outputs", "output_figs")
RES_OUT = os.path.join(WORK, "latest_outputs", "results")
os.makedirs(OUT, exist_ok=True)
os.makedirs(RES_OUT, exist_ok=True)

DPI = 600
FONT = {"family": "serif", "size": 10}
matplotlib.rc("font", **FONT)
matplotlib.rc("axes", linewidth=0.8)
matplotlib.rc("xtick", direction="in")
matplotlib.rc("ytick", direction="in")

COLORS = {
    "train": "#2563EB",
    "val": "#DC2626",
    "puma": "#2563EB",
    "abmil": "#16A34A",
    "transmil": "#DC2626",
    "clam": "#9333EA",
    "vitmlp": "#EA580C",
}


def load_json(name, dirs=None):
    if dirs is None:
        dirs = [WORK, os.path.join(WORK, "results"), INPUT, os.path.join(INPUT, "results")]
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(f"{name} not found")


def load_csv(name, dirs=None):
    if dirs is None:
        dirs = [WORK, os.path.join(WORK, "results"), INPUT, os.path.join(INPUT, "results")]
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"{name} not found")


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {name}")


# ============================================================
# Fig01: Training Loss Curve (Train + Val)
# ============================================================
def fig01_loss_curves():
    hist = load_json("training_histories_v3.json")
    # hist is a list of {"seed": ..., "history": [rows]}
    # Use best seed (456) — index 2 in [42, 123, 456]
    best = hist[2]["history"]  # list of epoch rows
    epochs = [r["epoch"] for r in best]
    train_loss = [r["train_total_loss"] for r in best]
    val_loss = [r.get("val_total_loss") for r in best]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(epochs, train_loss, color=COLORS["train"], linewidth=1.5,
            marker="o", markersize=3, label="Training Loss")
    if val_loss[0] is not None:
        ax.plot(epochs, val_loss, color=COLORS["val"], linewidth=1.5,
                marker="s", markersize=3, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("PUMA-MIL Training and Validation Loss")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    savefig(fig, "Fig01_loss_curves.png")


# ============================================================
# Fig02: Accuracy Curve (Train + Val)
# ============================================================
def fig02_accuracy_curves():
    hist = load_json("training_histories_v3.json")
    best = hist[2]["history"]  # seed 456 epoch rows
    epochs = [r["epoch"] for r in best]
    train_acc = [r.get("train_accuracy") for r in best]
    val_acc = [r.get("val_accuracy") or r.get("val_bal_acc") for r in best]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    if train_acc[0] is not None:
        ax.plot(epochs, train_acc, color=COLORS["train"], linewidth=1.5,
                marker="o", markersize=3, label="Training Accuracy")
    if val_acc[0] is not None:
        ax.plot(epochs, val_acc, color=COLORS["val"], linewidth=1.5,
                marker="s", markersize=3, label="Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("PUMA-MIL Training and Validation Accuracy")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.set_ylim(0.8, 1.01)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    savefig(fig, "Fig02_accuracy_curves.png")


# ============================================================
# Fig03: Baseline Comparison Loss Curves
# ============================================================
def fig03_baseline_loss_curves():
    try:
        bh = load_json("phase3b_training_histories.json")
    except FileNotFoundError:
        print("  ⚠ Skipping Fig03: baseline histories not found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    for idx, (name, color) in enumerate([
        ("ABMIL", COLORS["abmil"]), ("TransMIL", COLORS["transmil"]),
        ("CLAM-SB", COLORS["clam"]), ("ViT-Slice-MLP", COLORS["vitmlp"])
    ]):
        ax = axes[idx]
        if name not in bh:
            ax.set_title(f"{name} (no data)")
            continue
        # Use first seed
        first_key = list(bh[name].keys())[0]
        h = bh[name][first_key]
        epochs = [r["epoch"] for r in h]
        ax.plot(epochs, [r["train_loss"] for r in h], color=color,
                linewidth=1.5, label="Train")
        if "val_loss" in h[0]:
            ax.plot(epochs, [r["val_loss"] for r in h], color=color,
                    linewidth=1.5, linestyle="--", label="Val")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8, frameon=True)
        ax.grid(alpha=0.2, linewidth=0.5)
    plt.suptitle("Baseline Training Curves", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, "Fig03_baseline_loss_curves.png")


# ============================================================
# Fig04: Baseline Comparison Accuracy Curves
# ============================================================
def fig04_baseline_accuracy_curves():
    try:
        bh = load_json("phase3b_training_histories.json")
    except FileNotFoundError:
        print("  ⚠ Skipping Fig04: baseline histories not found")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    for idx, (name, color) in enumerate([
        ("ABMIL", COLORS["abmil"]), ("TransMIL", COLORS["transmil"]),
        ("CLAM-SB", COLORS["clam"]), ("ViT-Slice-MLP", COLORS["vitmlp"])
    ]):
        ax = axes[idx]
        if name not in bh:
            ax.set_title(f"{name} (no data)")
            continue
        first_key = list(bh[name].keys())[0]
        h = bh[name][first_key]
        epochs = [r["epoch"] for r in h]
        if "train_accuracy" in h[0]:
            ax.plot(epochs, [r["train_accuracy"] for r in h], color=color,
                    linewidth=1.5, label="Train Acc")
        ax.plot(epochs, [r["val_acc"] for r in h], color=color,
                linewidth=1.5, linestyle="--", label="Val Acc")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8, frameon=True)
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.set_ylim(0.7, 1.01)
    plt.suptitle("Baseline Accuracy Curves", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, "Fig04_baseline_accuracy_curves.png")


# ============================================================
# Fig05: Ensemble Confusion Matrix (Internal Test)
# ============================================================
def fig05_confusion_matrix():
    df = load_csv("slice_predictions_v3.csv")
    cm = confusion_matrix(df["label"], df["pred"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Tumor", "Tumor"])
    ax.set_yticklabels(["Non-Tumor", "Tumor"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("PUMA-MIL Ensemble — Internal Test Set")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    savefig(fig, "Fig05_confusion_matrix.png")


# ============================================================
# Fig06: ROC Curve (Ensemble)
# ============================================================
def fig06_roc_curve():
    df = load_csv("slice_predictions_v3.csv")
    fpr, tpr, _ = roc_curve(df["label"], df["prob_1"])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color=COLORS["puma"], linewidth=1.8,
            label=f"PUMA-MIL (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Internal Test Set")
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    savefig(fig, "Fig06_roc_curve.png")


# ============================================================
# Fig07: PR Curve (Ensemble)
# ============================================================
def fig07_pr_curve():
    df = load_csv("slice_predictions_v3.csv")
    prec, rec, _ = precision_recall_curve(df["label"], df["prob_1"])
    ap = average_precision_score(df["label"], df["prob_1"])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(rec, prec, color=COLORS["puma"], linewidth=1.8,
            label=f"PUMA-MIL (AP = {ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Internal Test Set")
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    savefig(fig, "Fig07_pr_curve.png")


# ============================================================
# Fig08: Competitive Baselines Comparison Bar Chart
# ============================================================
def fig08_competitive_comparison():
    comp = load_json("phase3b_competitive_results.json")
    agg = load_json("aggregated_test_metrics_v3.json")

    names = ["PUMA-MIL"] + list(comp.keys())
    metrics_keys = ["f1", "auc", "balanced_accuracy", "specificity", "sensitivity"]
    labels = ["Macro-F1", "AUC", "Bal. Acc.", "Specificity", "Sensitivity"]
    colors = [COLORS["puma"], COLORS["abmil"], COLORS["transmil"],
              COLORS["clam"], COLORS["vitmlp"]]

    data = {}
    ens = agg["ensemble"]
    data["PUMA-MIL"] = {
        "f1": ens["test_f1"], "auc": ens["test_auc"],
        "balanced_accuracy": ens["test_balanced_accuracy"],
        "specificity": ens["test_specificity"],
        "sensitivity": ens["test_sensitivity"],
    }
    for name, vals in comp.items():
        data[name] = {k: vals[k] for k in metrics_keys}

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(metrics_keys))
    width = 0.15
    for i, name in enumerate(names):
        vals = [data[name][k] for k in metrics_keys]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=colors[i % len(colors)], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=45)

    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("PUMA-MIL vs. Competitive Baselines (3-Seed Ensemble)")
    ax.legend(fontsize=8, frameon=True, ncol=len(names),
              bbox_to_anchor=(0.5, -0.15), loc="upper center")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    savefig(fig, "Fig08_competitive_comparison.png")


# ============================================================
# Fig09: Sensitivity Analysis (2×2 grid)
# ============================================================
def fig09_sensitivity():
    sens = load_json("phase3c_sensitivity_results.json")
    experiments = {}
    for r in sens:
        exp = r["experiment"]
        experiments.setdefault(exp, []).append(r)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    exp_order = ["window_size", "top_k", "n_layers", "loss_weights"]
    exp_labels = ["Window Size (W)", "Top-K", "Transformer Layers", "Loss Weights"]

    for idx, (exp_name, title) in enumerate(zip(exp_order, exp_labels)):
        ax = axes[idx]
        if exp_name not in experiments:
            ax.set_title(f"{title} (no data)")
            continue
        items = experiments[exp_name]
        configs = [r["config"] for r in items]
        f1s = [r["f1"] for r in items]
        aucs = [r["auc"] for r in items]
        x = np.arange(len(configs))
        w = 0.35
        ax.bar(x - w/2, f1s, w, label="Macro-F1", color=COLORS["puma"], alpha=0.85)
        ax.bar(x + w/2, aucs, w, label="AUC", color=COLORS["val"], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.88, 1.01)
        ax.legend(fontsize=7, frameon=True)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    fig.suptitle("Hyperparameter Sensitivity Analysis", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, "Fig09_sensitivity_analysis.png")


# ============================================================
# Fig10: Calibration / Reliability Diagram
# ============================================================
def fig10_calibration():
    bins = load_csv("phase4a_reliability_bins.csv")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect Calibration")
    ax.bar(bins["mean_confidence"], bins["accuracy"], width=0.08,
           alpha=0.6, color=COLORS["puma"], edgecolor="white", label="PUMA-MIL")
    ax.plot(bins["mean_confidence"], bins["accuracy"], "o-",
            color=COLORS["puma"], markersize=5, linewidth=1.5)
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Observed Accuracy")
    ax.set_title("Reliability Diagram (Calibration)")
    ax.legend(fontsize=9, frameon=True)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.45, 1.02)

    agg = load_json("aggregated_test_metrics_v3.json")
    ece = agg["ensemble"]["test_ece"]
    ax.text(0.55, 0.95, f"ECE = {ece:.4f}", transform=ax.transAxes,
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
    savefig(fig, "Fig10_calibration.png")


# ============================================================
# Fig11: Selective Prediction Curve
# ============================================================
def fig11_selective_prediction():
    sp = load_csv("phase4a_selective_prediction.csv")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(sp["coverage"], sp["macro_f1_phase2_ensemble"], "o-",
            color=COLORS["puma"], linewidth=1.5, markersize=4, label="PUMA-MIL Ensemble")
    ax.set_xlabel("Coverage (Fraction Retained)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Selective Prediction — F1 vs. Coverage")
    ax.legend(fontsize=9, frameon=True)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.set_ylim(0.93, 1.005)
    savefig(fig, "Fig11_selective_prediction.png")


# ============================================================
# Fig12: Modality Reliability Weights
# ============================================================
def fig12_modality_weights():
    mw = load_csv("phase4b_modality_reliability_weights.csv")
    tumor = mw[mw["class_name"] == "Tumor"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    modalities = tumor["modality"].values
    means = tumor["mean_weight"].values
    stds = tumor["std_weight"].values
    colors_bar = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]
    bars = ax.bar(modalities, means, yerr=stds, capsize=4,
                  color=colors_bar[:len(modalities)], edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Mean Reliability Weight")
    ax.set_title("Modality Reliability Weights (Tumor Class)")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    # Add note about uniform weights for non-tumor
    ax.text(0.5, -0.18, "Note: Non-Tumor class shows uniform weights (0.25±0.00) by design",
            transform=ax.transAxes, fontsize=7, ha="center", style="italic", color="gray")
    savefig(fig, "Fig12_modality_weights.png")


# ============================================================
# Fig13: External Validation Confusion Matrix (BraTS2020)
# ============================================================
def fig13_external_cm():
    ext = load_json("external_aggregated_metrics_locked_v3.json")
    cm = np.array(ext["slice_level_external"]["confusion_matrix"])
    f1 = ext["slice_level_external"]["macro_f1"]
    ext_auc = ext["slice_level_external"]["auc"]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Oranges", interpolation="nearest")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Tumor", "Tumor"])
    ax.set_yticklabels(["Non-Tumor", "Tumor"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"External Validation — BraTS2020\n(F1={f1:.4f}, AUC={ext_auc:.4f})")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    savefig(fig, "Fig13_external_confusion_matrix.png")


# ============================================================
# CONSOLIDATION: copy results + checkpoints
# ============================================================
def consolidate():
    import shutil
    ckpt_dir = os.path.join(WORK, "latest_outputs", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Copy checkpoints
    for f in ["pumamil_best_v3.pth", "pumamil_best_seed42.pth",
              "pumamil_best_seed123.pth", "pumamil_best_seed456.pth"]:
        src = os.path.join(WORK, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(ckpt_dir, f))
            print(f"  ✓ Checkpoint: {f}")

    # Copy result files
    result_files = [
        "aggregated_test_metrics_v3.json", "per_seed_test_metrics_v3.csv",
        "slice_predictions_v3.csv", "patient_predictions_v3.csv",
        "training_histories_v3.json", "hyperparameters_v3.json",
        "phase3b_competitive_results.json", "phase3b_training_histories.json",
        "phase3b_competitive_predictions.json",
    ]
    # Also copy results from INPUT (previous run) that we didn't re-run
    old_results = [
        "phase3_results_v8.json", "phase3c_sensitivity_results.json",
        "phase4a_reliability_bins.csv", "phase4a_selective_prediction.csv",
        "phase4a_subgroup_by_role.csv", "phase4a_mcnemar_vs_phase3.csv",
        "phase4b_modality_reliability_weights.csv",
        "external_aggregated_metrics_locked_v3.json",
        "external_slice_predictions_locked_v3.csv",
        "patient_splits_v3.json",
    ]

    for f in result_files:
        src = os.path.join(WORK, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(RES_OUT, f))
    for f in old_results:
        for base in [INPUT, os.path.join(INPUT, "results"), WORK]:
            src = os.path.join(base, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(RES_OUT, f))
                break

    print(f"\n  Consolidated to: {os.path.join(WORK, 'latest_outputs')}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("PUMA-MIL — Generating Journal-Standard Figures")
    print("=" * 60)

    print("\n─── Consolidating files ───")
    consolidate()

    print("\n─── Generating Figures ───")
    fig01_loss_curves()
    fig02_accuracy_curves()
    fig03_baseline_loss_curves()
    fig04_baseline_accuracy_curves()
    fig05_confusion_matrix()
    fig06_roc_curve()
    fig07_pr_curve()
    fig08_competitive_comparison()
    fig09_sensitivity()
    fig10_calibration()
    fig11_selective_prediction()
    fig12_modality_weights()
    fig13_external_cm()

    print(f"\n{'=' * 60}")
    print(f"✓ ALL {len(os.listdir(OUT))} FIGURES GENERATED")
    print(f"  Output: {OUT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
