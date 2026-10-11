"""PUMA-MIL interpretability module."""

from .gradcam import ViTGradCAM
from .lime_explainer import explain_slice_lime
from .shap_explainer import compute_shap_values
from .attention_rollout import compute_attention_rollout

__all__ = [
    "ViTGradCAM",
    "explain_slice_lime",
    "compute_shap_values",
    "compute_attention_rollout",
]
