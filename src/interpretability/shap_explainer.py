"""
SHAP GradientExplainer for Pixel-Level Feature Attribution on MRI slices.
"""

import numpy as np
import torch
import shap

def compute_shap_values(model, background_tensor: torch.Tensor, test_tensor: torch.Tensor):
    """
    Computes SHAP values using GradientExplainer.
    Input:
      background_tensor: [N_bg, 3, 224, 224]
      test_tensor: [N_test, 3, 224, 224]
    """
    explainer = shap.GradientExplainer(model, background_tensor)
    shap_values = explainer.shap_values(test_tensor)
    return shap_values
