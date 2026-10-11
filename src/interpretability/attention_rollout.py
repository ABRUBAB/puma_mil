"""
Attention Rollout and Cross-Modal Attention Analysis.
Computes accumulated attention flows through the sequential Transformer encoder.
"""

import numpy as np
import torch

def compute_attention_rollout(all_layer_attentions: list, discard_ratio: float = 0.9) -> np.ndarray:
    """
    Computes attention rollout across Transformer layers (Abnar & Zuidema, 2020).
    Input: list of [B, n_heads, W, W] attention matrices.
    """
    result = torch.eye(all_layer_attentions[0].size(-1))
    with torch.no_grad():
        for attn in all_layer_attentions:
            attn_heads = attn.mean(dim=1)  # average across heads
            # Add identity matrix for residual connection
            I = torch.eye(attn_heads.size(-1)).to(attn_heads.device)
            a = 0.5 * attn_heads + 0.5 * I
            a = a / a.sum(dim=-1, keepdim=True)
            result = torch.matmul(a, result.to(a.device))
    return result.cpu().numpy()
