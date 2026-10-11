"""
PUMA-MIL: Prototype-Updated Multi-Modal Attention-Based Multiple Instance Learning
for Slice-Level Glioma Detection on Multi-Sequence MRI.

Author: Abdullah Rubab (rubab2712@gmail.com)
GitHub: https://github.com/ABRUBAB/puma_mil
"""

from .models import PUMAMILLite, ABMIL, TransMIL, CLAM_SB, ViTSliceMLP
from .data import CenterWindowDataset, ViTFeatureExtractor
from .training import PUMALoss, train_one_epoch, evaluate_model
from .evaluation import metrics_from_probs, calculate_ece, fit_temperature, bootstrap_ci
from .utils import Config

__version__ = "1.0.0"
__author__ = "Abdullah Rubab"

__all__ = [
    "PUMAMILLite",
    "ABMIL",
    "TransMIL",
    "CLAM_SB",
    "ViTSliceMLP",
    "CenterWindowDataset",
    "ViTFeatureExtractor",
    "PUMALoss",
    "train_one_epoch",
    "evaluate_model",
    "metrics_from_probs",
    "calculate_ece",
    "fit_temperature",
    "bootstrap_ci",
    "Config",
]
