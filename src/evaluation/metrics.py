"""
Evaluation metrics for PUMA-MIL:
Accuracy, Macro-F1, ROC-AUC, PR-AUC, Sensitivity, Specificity, Balanced Accuracy, ECE,
and Bootstrap Confidence Interval calculation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix
)
from typing import Dict, Tuple

def safe_auc(labels, prob1) -> float:
    labels = np.asarray(labels)
    prob1 = np.asarray(prob1)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, prob1))
    except Exception:
        return float("nan")

def safe_pr_auc(labels, prob1) -> float:
    labels = np.asarray(labels)
    prob1 = np.asarray(prob1)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(labels, prob1))
    except Exception:
        return float("nan")

def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error across n_bins equal-width confidence intervals."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 0.0

    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    acc = (pred == labels).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf <= bins[i + 1]) if i == 0 else (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / len(labels)) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)

def metrics_from_probs(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Tuple[Dict[str, float], np.ndarray]:
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if len(labels) == 0:
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "auc": float("nan"), "pr_auc": float("nan"), "ece": 0.0,
            "balanced_accuracy": 0.0, "specificity": 0.0, "sensitivity": 0.0
        }, np.zeros(0, dtype=np.int64)

    preds = (probs[:, 1] >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "auc": safe_auc(labels, probs[:, 1]),
        "pr_auc": safe_pr_auc(labels, probs[:, 1]),
        "ece": calculate_ece(probs, labels),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)) if len(np.unique(labels)) > 1 else 0.0,
        "specificity": float(recall_score(labels, preds, pos_label=0, zero_division=0)),
        "sensitivity": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
    }
    return metrics, preds

def find_best_threshold(probs: np.ndarray, labels: np.ndarray,
                        start: float = 0.10, end: float = 0.90, step: float = 0.01) -> Tuple[float, float]:
    """Finds decision threshold maximizing validation macro-F1."""
    best_t, best_f1 = 0.50, -1.0
    for t in np.arange(start, end + 1e-5, step):
        preds = (probs[:, 1] >= t).astype(int)
        f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1

def bootstrap_ci(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.42,
                 n_boot: int = 2000, seed: int = 42) -> Dict[str, Tuple[float, float]]:
    """Computes 95% bootstrap confidence intervals for key metrics."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    boot_metrics = {"accuracy": [], "f1": [], "sensitivity": [], "specificity": [], "balanced_accuracy": []}

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        m, _ = metrics_from_probs(probs[idx], labels[idx], threshold=threshold)
        for k in boot_metrics:
            boot_metrics[k].append(m[k])

    cis = {}
    for k, v in boot_metrics.items():
        arr = np.array(v)
        cis[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return cis
