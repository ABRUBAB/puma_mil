"""PUMA-MIL evaluation module."""

from .metrics import (
    safe_auc,
    safe_pr_auc,
    calculate_ece,
    metrics_from_probs,
    find_best_threshold,
    bootstrap_ci,
)
from .selective_prediction import evaluate_selective_prediction
from .temperature_scaling import fit_temperature, apply_temperature

__all__ = [
    "safe_auc",
    "safe_pr_auc",
    "calculate_ece",
    "metrics_from_probs",
    "find_best_threshold",
    "bootstrap_ci",
    "evaluate_selective_prediction",
    "fit_temperature",
    "apply_temperature",
]
