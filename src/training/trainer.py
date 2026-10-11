"""
Training Engine for PUMA-MIL.
Handles multi-seed training, validation loops, early stopping, and checkpointing.
"""

import os
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List

from src.models.puma_mil import PUMAMILLite
from src.training.losses import PUMALoss
from src.data.dataset import make_role_weighted_sampler

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: PUMALoss,
                    optimizer: torch.optim.Optimizer, device: str, clip_grad: float = 1.0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        t1, t1ce, t2, flair, bag_lbl, center_lbl, seq_mask, mod_mask, center_pos, role_id, _, _, _, _ = batch
        t1 = t1.to(device)
        t1ce = t1ce.to(device)
        t2 = t2.to(device)
        flair = flair.to(device)
        bag_lbl = bag_lbl.to(device)
        center_lbl = center_lbl.to(device)
        seq_mask = seq_mask.to(device)
        mod_mask = mod_mask.to(device)
        role_id = role_id.to(device)

        optimizer.zero_grad()
        bag_logits, slice_logits, bag_embed, mod_weights, _, _ = model(
            t1, t1ce, t2, flair, seq_mask, mod_mask, return_aux=True
        )

        center_idx = center_pos[0].item()
        center_logits = slice_logits[:, center_idx, :]

        loss, _ = criterion(
            center_logits, center_lbl, role_id, bag_logits, bag_lbl,
            slice_logits, seq_mask, mod_weights
        )

        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        model.update_prototypes(bag_embed.detach(), bag_lbl)

        total_loss += loss.item() * len(center_lbl)
        preds = center_logits.argmax(dim=-1)
        correct += (preds == center_lbl).sum().item()
        total += len(center_lbl)

    return total_loss / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    all_logits = []
    all_labels = []
    all_roles = []
    all_pids = []
    all_sidxs = []

    for batch in loader:
        t1, t1ce, t2, flair, _, center_lbl, seq_mask, mod_mask, center_pos, role_id, pids, sidxs, _, _ = batch
        t1 = t1.to(device)
        t1ce = t1ce.to(device)
        t2 = t2.to(device)
        flair = flair.to(device)
        seq_mask = seq_mask.to(device)
        mod_mask = mod_mask.to(device)

        _, slice_logits, _, _, _, _ = model(
            t1, t1ce, t2, flair, seq_mask, mod_mask, return_aux=True
        )
        center_idx = center_pos[0].item()
        center_logits = slice_logits[:, center_idx, :]

        all_logits.append(center_logits.cpu())
        all_labels.append(center_lbl)
        all_roles.append(role_id)
        all_pids.extend(pids)
        all_sidxs.extend(sidxs.tolist())

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    roles = torch.cat(all_roles, dim=0).numpy()
    return logits, labels, roles, all_pids, all_sidxs
