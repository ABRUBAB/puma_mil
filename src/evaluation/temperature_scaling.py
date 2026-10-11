"""
Post-hoc Temperature Scaling via L-BFGS optimization on the tuning set.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def fit_temperature(logits: np.ndarray, labels: np.ndarray, device: str = "cpu") -> float:
    """
    Fits a single scalar temperature T > 0 on tuning set logits:
    min_T - sum log softmax(logits / T)
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 1.0

    T = nn.Parameter(torch.ones(1, device=device))
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)

    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits_t / T.clamp_min(0.5), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.detach().clamp_min(0.5).cpu().item())

def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Computes calibrated probabilities: softmax(logits / T)."""
    t_logits = logits / max(temperature, 1e-4)
    exp = np.exp(t_logits - np.max(t_logits, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)
