"""
Loss functions for PUMA-MIL:
1. Role-aware asymmetric focal loss (Section 3.4.1)
2. Temporal smoothness loss
3. Modality diversity loss
4. Combined multi-objective loss (Eq. 17)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

def diversity_loss(mod_weights: torch.Tensor) -> torch.Tensor:
    """
    Modality Diversity Regularization:
    Encourages balanced sequence usage by penalizing batch-average deviation from uniform (0.25).
    """
    mean_w = mod_weights.mean(dim=1)  # average across window slices -> [B, 4]
    target = torch.full_like(mean_w, 1.0 / mean_w.size(-1))
    return F.mse_loss(mean_w, target)

def temporal_smoothness_loss(slice_logits: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
    """
    Temporal Smoothness Loss:
    Penalizes abrupt probability jumps between adjacent valid axial slices using Smooth L1.
    """
    pos_probs = torch.softmax(slice_logits, dim=-1)[..., 1]
    diffs = pos_probs[:, 1:] - pos_probs[:, :-1]
    pair_mask = seq_mask[:, 1:] & seq_mask[:, :-1]
    if pair_mask.sum() == 0:
        return slice_logits.sum() * 0.0
    return F.smooth_l1_loss(diffs[pair_mask], torch.zeros_like(diffs[pair_mask]))

def role_aware_asymmetric_focal_loss(center_logits: torch.Tensor, center_targets: torch.Tensor,
                                     role_ids: torch.Tensor, class_weights: torch.Tensor,
                                     gamma_pos: float = 1.5, gamma_neg: float = 3.0,
                                     role_multipliers: Dict[int, float] = None) -> torch.Tensor:
    """
    Role-Aware Asymmetric Focal Loss (Eq. 18):
    L_center = -w1 * y * (1-p)^gamma+ * log(p) - w0 * (1-y) * R_r * p^gamma- * log(1-p)
    """
    if role_multipliers is None:
        role_multipliers = {0: 1.25, 1: 1.60, 2: 1.00, 3: 1.00, 4: 1.15}

    prob1 = torch.softmax(center_logits, dim=-1)[:, 1].clamp(1e-6, 1.0 - 1e-6)
    y = center_targets.float()

    pos_w = float(class_weights[1])
    neg_w = float(class_weights[0])

    role_mult = torch.tensor(
        [role_multipliers.get(int(r.item()), 1.0) for r in role_ids],
        dtype=torch.float32,
        device=center_logits.device
    )

    pos_term = -pos_w * y * ((1.0 - prob1) ** gamma_pos) * torch.log(prob1)
    neg_term = -neg_w * (1.0 - y) * role_mult * (prob1 ** gamma_neg) * torch.log(1.0 - prob1)

    return (pos_term + neg_term).mean()

class PUMALoss(nn.Module):
    """
    Total multi-objective loss (Eq. 17):
    L = 1.0 * L_center + 0.2 * L_bag + 0.06 * L_smooth + 0.01 * L_div
    """
    def __init__(self, cfg, class_weights: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.class_weights = class_weights

    def forward(self, center_logits, center_targets, role_ids, bag_logits, bag_targets,
                slice_logits, seq_mask, mod_weights):
        # 1. Primary center loss
        l_center = role_aware_asymmetric_focal_loss(
            center_logits, center_targets, role_ids, self.class_weights,
            gamma_pos=self.cfg.center_gamma_pos,
            gamma_neg=self.cfg.center_gamma_neg,
            role_multipliers=self.cfg.role_loss_multiplier
        )

        # 2. Auxiliary bag loss
        l_bag = F.cross_entropy(bag_logits, bag_targets)

        # 3. Temporal smoothness loss
        l_smooth = temporal_smoothness_loss(slice_logits, seq_mask)

        # 4. Modality diversity loss
        l_div = diversity_loss(mod_weights)

        # Total
        total = (
            self.cfg.center_loss_weight * l_center +
            self.cfg.bag_loss_weight * l_bag +
            self.cfg.smoothness_loss_weight * l_smooth +
            self.cfg.diversity_loss_weight * l_div
        )

        return total, {
            "l_center": l_center.item(),
            "l_bag": l_bag.item(),
            "l_smooth": l_smooth.item(),
            "l_div": l_div.item(),
            "l_total": total.item()
        }
