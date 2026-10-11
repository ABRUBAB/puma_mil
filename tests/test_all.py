"""
Unit and integration test suite for PUMA-MIL.
"""

import os
import sys
import numpy as np
import pytest
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import (
    PUMAMILLite,
    ReliabilityFusion,
    HybridMILPool,
    ABMIL,
    TransMIL,
    CLAM_SB,
    ViTSliceMLP,
)
from src.data import (
    CenterWindowDataset,
    normalize_volume_3stage,
    select_center_crop_slices,
    make_role_weighted_sampler,
)
from src.training import (
    PUMALoss,
    diversity_loss,
    temporal_smoothness_loss,
    role_aware_asymmetric_focal_loss,
)
from src.evaluation import (
    safe_auc,
    safe_pr_auc,
    calculate_ece,
    metrics_from_probs,
    find_best_threshold,
    bootstrap_ci,
    evaluate_selective_prediction,
    fit_temperature,
    apply_temperature,
)
from src.utils import Config


def test_puma_mil_model_forward():
    model = PUMAMILLite(d_in=384, d_model=128, max_slices=48, top_k=4)
    model.eval()

    bsz, w = 2, 11
    t1 = torch.randn(bsz, w, 384)
    t1ce = torch.randn(bsz, w, 384)
    t2 = torch.randn(bsz, w, 384)
    flair = torch.randn(bsz, w, 384)
    seq_mask = torch.ones(bsz, w, dtype=torch.bool)
    mod_mask = torch.ones(bsz, 4, dtype=torch.bool)

    # Standard forward
    bag_logits = model(t1, t1ce, t2, flair, seq_mask, mod_mask, return_aux=False)
    assert bag_logits.shape == (bsz, 2)

    # Auxiliary forward
    bag_l, slice_l, bag_emb, mod_w, attn_w, rel = model(
        t1, t1ce, t2, flair, seq_mask, mod_mask, return_aux=True
    )
    assert bag_l.shape == (bsz, 2)
    assert slice_l.shape == (bsz, w, 2)
    assert bag_emb.shape == (bsz, 128 * 3)
    assert mod_w.shape == (bsz, w, 4)
    assert attn_w.shape == (bsz, w)
    assert rel.shape == (bsz, w)


def test_puma_mil_prototypes():
    model = PUMAMILLite(d_in=384, d_model=128, max_slices=48, top_k=4)
    bag_embed = torch.randn(4, 128 * 3)
    bag_labels = torch.tensor([0, 1, 0, 1])

    assert model.proto_counts[0] == 0
    model.update_prototypes(bag_embed, bag_labels)
    assert model.proto_counts[0] == 2
    assert model.proto_counts[1] == 2

    sims = model.prototype_sims(bag_embed)
    assert sims.shape == (4, 2)


def test_baselines_forward():
    bsz, w = 2, 11
    t1 = torch.randn(bsz, w, 384)
    t1ce = torch.randn(bsz, w, 384)
    t2 = torch.randn(bsz, w, 384)
    flair = torch.randn(bsz, w, 384)
    seq_mask = torch.ones(bsz, w, dtype=torch.bool)

    abmil = ABMIL(d_in=384, d_hidden=256)
    out = abmil(t1, t1ce, t2, flair, seq_mask)
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (bsz, 2)

    transmil = TransMIL(d_in=384, d_model=128)
    logits = transmil(t1, t1ce, t2, flair, seq_mask)
    assert logits.shape == (bsz, 2)

    clam = CLAM_SB(d_in=384, d_hidden=256)
    logits = clam(t1, t1ce, t2, flair, seq_mask, return_inst_loss=False)
    assert logits.shape == (bsz, 2)
    logits, inst_loss = clam(t1, t1ce, t2, flair, seq_mask, return_inst_loss=True, center_label=torch.tensor([0, 1]))
    assert logits.shape == (bsz, 2)
    assert inst_loss.ndim == 0

    vitmlp = ViTSliceMLP(d_in=384, window_size=11)
    logits = vitmlp(t1, t1ce, t2, flair, seq_mask)
    assert logits.shape == (bsz, 2)


def test_losses():
    bsz, w = 4, 11
    slice_logits = torch.randn(bsz, w, 2, requires_grad=True)
    center_logits = slice_logits[:, 5, :]
    center_lbl = torch.randint(0, 2, (bsz,))
    bag_lbl = torch.randint(0, 2, (bsz,))
    bag_logits = torch.randn(bsz, 2, requires_grad=True)
    role_id = torch.tensor([0, 1, 2, 4])
    seq_mask = torch.ones(bsz, w, dtype=torch.bool)
    mod_weights = torch.softmax(torch.randn(bsz, w, 4), dim=-1)

    cfg = Config()
    class_weights = torch.tensor([1.0, 1.0])
    loss_fn = PUMALoss(cfg, class_weights)

    total_loss, details = loss_fn(
        center_logits, center_lbl, role_id, bag_logits, bag_lbl, slice_logits, seq_mask, mod_weights
    )
    assert total_loss.item() > 0
    assert "loss_center" in details
    assert "loss_bag" in details
    assert "loss_smooth" in details
    assert "loss_div" in details

    total_loss.backward()
    assert slice_logits.grad is not None
    assert bag_logits.grad is not None


def test_metrics():
    labels = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.1, 0.9],
        [0.3, 0.7],
        [0.7, 0.3],
        [0.2, 0.8],
        [0.1, 0.9],
        [0.6, 0.4]
    ])

    m = metrics_from_probs(probs, labels, threshold=0.5)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["auc"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0

    ece = calculate_ece(probs, labels, n_bins=5)
    assert 0.0 <= ece <= 1.0

    t_best, best_f1 = find_best_threshold(probs, labels)
    assert 0.0 <= t_best <= 1.0
    assert best_f1 == 1.0

    sel = evaluate_selective_prediction(probs, labels, threshold=0.5)
    assert len(sel) > 0
    assert "coverage" in sel[0]
    assert "macro_f1" in sel[0]

    ci = bootstrap_ci(probs, labels, n_resamples=50, seed=42)
    assert "accuracy" in ci
    assert "ci_low" in ci["accuracy"]


def test_dataset():
    n_slices = 20
    data_dict = {
        "patient_ids": ["pat1"] * n_slices,
        "slice_idxs": list(range(n_slices)),
        "features_t1": np.random.randn(n_slices, 384).astype(np.float32),
        "features_t1ce": np.random.randn(n_slices, 384).astype(np.float32),
        "features_t2": np.random.randn(n_slices, 384).astype(np.float32),
        "features_flair": np.random.randn(n_slices, 384).astype(np.float32),
        "labels": np.array([0]*10 + [1]*10),
        "modality_mask": np.ones((n_slices, 4), dtype=np.int64),
        "distance_to_positive": np.zeros(n_slices, dtype=np.int64),
        "slice_role_id": np.array([0]*5 + [1]*5 + [2]*10),
        "tumor_pixels": np.zeros(n_slices, dtype=np.int64),
    }

    ds = CenterWindowDataset(data_dict, window_size=11)
    assert len(ds) == n_slices

    item = ds[0]
    assert len(item) == 14
    t1, t1ce, t2, flair, bag_lbl, center_lbl, seq_mask, mod_mask = item[:8]
    assert t1.shape == (11, 384)
    assert seq_mask.shape == (11,)
    assert mod_mask.shape == (4,)
    assert center_lbl.item() == 0

    cfg = Config()
    sampler = make_role_weighted_sampler(ds, cfg)
    assert len(sampler) == n_slices


def test_temperature_scaling():
    logits = np.array([[-2.0, 2.0], [1.5, -1.5], [-1.0, 1.0], [2.0, -2.0]])
    labels = np.array([1, 0, 1, 0])

    T = fit_temperature(logits, labels)
    assert T > 0.0

    cal_probs = apply_temperature(logits, T)
    assert cal_probs.shape == (4, 2)
    np.testing.assert_allclose(cal_probs.sum(axis=-1), np.ones(4), atol=1e-5)
