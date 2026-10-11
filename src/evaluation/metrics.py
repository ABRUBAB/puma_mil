"""
Evaluation metrics for PUMA-MIL:
- Accuracy, Precision, Recall, Macro-F1
- ROC-AUC and PR-AUC
- Expected Calibration Error (ECE)
- Percentile Bootstrap 95% Confidence Intervals (2,000 resamples)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    balanced_accuracy_score,
)
from typing import Dict, Tuple, Any

class MetricsResult(tuple):
    """
    Dual-compatibility return object:
    Can be unpacked as (metrics_dict, preds) OR used directly as a metrics dict!
    """
    def __new__(cls, metrics_dict: Dict[str, float], preds: np.ndarray):
        return super().__new__(cls, (metrics_dict, preds))

    def __init__(self, metrics_dict: Dict[str, float], preds: np.ndarray):
        self.metrics = metrics_dict
        self.preds = preds

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.metrics[item]
        return super().__getitem__(item)

    def __contains__(self, item):
        return item in self.metrics

    def get(self, key, default=None):
        return self.metrics.get(key, default)

    def keys(self):
        return self.metrics.keys()

    def values(self):
        return self.metrics.values()

    def items(self):
        return self.metrics.items()

class CIInterval(dict):
    """
    Dual-compatibility CI object:
    Can be indexed as dict (['ci_low'], ['ci_high']) OR tuple ([0], [1]) OR attributes (.ci_low, .ci_high).
    """
    def __init__(self, low: float, high: float, mean: float = None):
        super().__init__(ci_low=low, ci_high=high, mean=mean if mean is not None else (low + high) / 2.0)
        self.ci_low = low
        self.ci_high = high
        self.mean = self["mean"]

    def __getitem__(self, item):
        if isinstance(item, int):
            return [self.ci_low, self.ci_high][item]
        return super().__getitem__(item)

def safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, probs))
    except Exception:
        return float("nan")

def safe_pr_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        if len(np.unique(labels)) < 2:
            return float("nan")
        p, r, _ = precision_recall_curve(labels, probs)
        return float(auc(r, p))
    except Exception:
        return float("nan")

def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
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

def metrics_from_probs(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> MetricsResult:
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if len(labels) == 0:
        empty = {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "macro_f1": 0.0,
            "auc": float("nan"), "pr_auc": float("nan"), "ece": 0.0,
            "balanced_accuracy": 0.0, "balanced_acc": 0.0, "specificity": 0.0, "sensitivity": 0.0
        }
        return MetricsResult(empty, np.zeros(0, dtype=np.int64))

    preds = (probs[:, 1] >= threshold).astype(int)
    f1_val = float(f1_score(labels, preds, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(labels, preds)) if len(np.unique(labels)) > 1 else 0.0
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1": f1_val,
        "macro_f1": f1_val,
        "auc": safe_auc(labels, probs[:, 1]),
        "pr_auc": safe_pr_auc(labels, probs[:, 1]),
        "ece": calculate_ece(probs, labels),
        "balanced_accuracy": bal_acc,
        "balanced_acc": bal_acc,
        "specificity": float(recall_score(labels, preds, pos_label=0, zero_division=0)),
        "sensitivity": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
    }
    return MetricsResult(metrics, preds)

def find_best_threshold(probs: np.ndarray, labels: np.ndarray,
                        start: float = 0.10, end: float = 0.90, step: float = 0.01) -> Tuple[float, float]:
    """Finds decision threshold maximizing validation macro-F1. Returns (best_threshold, best_macro_f1)."""
    best_t, best_f1 = 0.50, -1.0
    for t in np.arange(start, end + 1e-5, step):
        preds = (probs[:, 1] >= t).astype(int)
        f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1

def bootstrap_ci(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.42,
                 n_resamples: int = 2000, seed: int = 42) -> Dict[str, CIInterval]:
    """Computes 95% bootstrap confidence intervals for key metrics."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    boot_metrics = {"accuracy": [], "macro_f1": [], "f1": [], "sensitivity": [], "specificity": [], "balanced_accuracy": [], "balanced_acc": []}

    for _ in range(n_resamples):
        idx = rng.choice(n, n, replace=True)
        m, _ = metrics_from_probs(probs[idx], labels[idx], threshold=threshold)
        for k in boot_metrics:
            boot_metrics[k].append(m[k])

    cis = {}
    for k, v in boot_metrics.items():
        arr = np.array(v)
        cis[k] = CIInterval(
            low=float(np.percentile(arr, 2.5)),
            high=float(np.percentile(arr, 97.5)),
            mean=float(np.mean(arr)),
        )
    return cis
