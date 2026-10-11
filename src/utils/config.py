"""
Central Configuration module for PUMA-MIL.
Supports default hyperparameter settings from the paper and CLI overrides.
"""

import os
import argparse
import torch
from dataclasses import dataclass, field
from typing import Tuple, List, Dict

@dataclass
class Config:
    # --- Paths ---
    work_dir: str = "./outputs"
    data_dir: str = "./data"
    phase1_dir: str = "./data/phase-1-output"
    brats2021_dir: str = "./data/brats2021"
    brats2020_dir: str = "./data/brats2020"
    checkpoints_dir: str = "./outputs/checkpoints"
    results_dir: str = "./outputs/results"
    figures_dir: str = "./outputs/output_figs"

    # --- Hardware & Reproducibility ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])

    # --- Preprocessing & Feature Extraction ---
    img_size: int = 224
    patch_size: int = 16
    modalities: Tuple[str, ...] = ("t1", "t1ce", "t2", "flair")
    modality_labels: Tuple[str, ...] = ("T1", "T1ce", "T2", "FLAIR")
    backbone: str = "vit_small_patch16_224"
    d_in: int = 384                 # ViT-S/16 CLS token dimension
    tumor_pixel_threshold: int = 100
    max_selected_slices: int = 48
    boundary_neg_radius: int = 6
    far_neg_min_distance: int = 12

    # --- PUMA-MIL Architecture ---
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    n_layers: int = 2
    dropout: float = 0.10
    modality_dropout: float = 0.10
    train_window: int = 11          # W = 11 axial slices
    top_k: int = 4                  # K = 4 most positive/negative instances
    proto_momentum: float = 0.90    # EMA momentum for class prototypes mu

    # --- Optimization & Loss ---
    total_epochs: int = 32
    batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 8e-4
    weight_decay: float = 1e-4
    patience: int = 8
    clip_grad_norm: float = 1.0

    # Loss weights (Eq. 17)
    center_loss_weight: float = 1.00
    bag_loss_weight: float = 0.20
    smoothness_loss_weight: float = 0.06
    diversity_loss_weight: float = 0.01

    # Asymmetric focal loss focusing parameters
    center_gamma_pos: float = 1.5
    center_gamma_neg: float = 3.0

    # Role multipliers R_r (Table 2 & Section 3.4.1)
    role_loss_multiplier: Dict[int, float] = field(default_factory=lambda: {
        0: 1.25,   # far_neg
        1: 1.60,   # boundary_neg
        2: 1.00,   # positive
        3: 1.00,   # mid_neg
        4: 1.15,   # no_tumor
    })

    # Weighted sampler boosts
    sampler_positive_boost: float = 1.20
    sampler_boundary_neg_boost: float = 1.45
    sampler_far_neg_boost: float = 1.20
    sampler_no_tumor_boost: float = 1.10

    # --- Output & DPI ---
    fig_dpi: int = 600

    def ensure_dirs(self):
        for d in [self.work_dir, self.checkpoints_dir, self.results_dir, self.figures_dir]:
            os.makedirs(d, exist_ok=True)

# Slice role mappings
SLICE_ROLE_TO_ID = {
    "far_neg": 0,
    "boundary_neg": 1,
    "positive": 2,
    "mid_neg": 3,
    "no_tumor": 4,
}
ID_TO_SLICE_ROLE = {v: k for k, v in SLICE_ROLE_TO_ID.items()}
