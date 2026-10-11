"""
Selective Prediction Analysis:
Evaluates Macro-F1 and accuracy as coverage decreases from 100% to 50%
by deferring uncertain slices based on prediction confidence |p - 0.5|.
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from typing import Dict, List, Any

def evaluate_selective_prediction(probs: np.ndarray, labels: np.ndarray,
                                  threshold: float = 0.5,
                                  coverages: List[float] = None) -> List[Dict[str, Any]]:
    if coverages is None:
        coverages = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]

    prob1 = probs[:, 1]
    confidence = np.abs(prob1 - 0.5)
    order = np.argsort(-confidence)  # most confident first

    results = []
    n_total = len(labels)

    for cov in coverages:
        k = max(1, int(np.round(n_total * cov)))
        subset_idx = order[:k]
        sub_preds = (prob1[subset_idx] >= threshold).astype(int)
        sub_labels = labels[subset_idx]

        f1 = float(f1_score(sub_labels, sub_preds, average="macro", zero_division=0))
        acc = float(accuracy_score(sub_labels, sub_preds))
        results.append({
            "coverage": cov,
            "retained_count": k,
            "macro_f1": f1,
            "accuracy": acc
        })

    return results
