"""
Publication-grade Figure Plotting at 600 DPI.
Generates Figures 1-14 matching the styles and color palettes in the paper.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Default font and style settings
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

def savefig(fig, out_dir, name, dpi=600):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [SAVED] {p}")

def plot_confusion_matrix(cm: np.ndarray, out_dir: str, name: str = "Figure8.png",
                          title: str = "PUMA-MIL Ensemble -- BraTS-2021 Test Set"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Tumor", "Tumor"])
    ax.set_yticklabels(["Non-Tumor", "Tumor"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title, fontsize=10)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    savefig(fig, out_dir, name)

def plot_calibration_curve(confidences, accuracies, ece, out_dir: str, name: str = "Figure10a.png"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    bins = np.linspace(0.45, 1.0, len(accuracies))
    ax.bar(bins, accuracies, width=0.04, color="#3B82F6", alpha=0.7, edgecolor="#1D4ED8", label="Outputs")
    ax.plot([0.45, 1.0], [0.45, 1.0], "r--", linewidth=1.5, label="Perfect calibration")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram (ECE = {ece:.4f})")
    ax.legend(loc="upper left", frameon=True)
    savefig(fig, out_dir, name)
