# ============================================================
# PUMA-MIL CLEAN RUN — CELL 2: COMPETITIVE BASELINES
# ABMIL, TransMIL, CLAM-SB, ViT-Slice-MLP
# Fixed: logs val_loss + train_accuracy for proper curves
# ============================================================
# Fairness guarantees:
#   - Same precomputed ViT features (frozen vit_small_patch16_224)
#   - Same patient-level stratified splits (train/val/tune/test)
#   - Same CenterWindowDataset with W=11, center-slice labels
#   - CLASS-BALANCED sampling (no role-aware boosts -- those are
#     PUMA-MIL's contribution, giving them to baselines would be
#     an unearned advantage)
#   - Same evaluation: temperature scaling on tune, threshold
#     tuning, metrics_from_probs on test, ECE with 5 bins
#   - Same model selection: maximum validation macro-F1
#   - Same optimizer (AdamW), lr=8e-4, early stopping patience 8
#   - Same training budget (32 epochs max)
#   - 3-SEED ENSEMBLE {42, 123, 456} with probability averaging
#     (identical to PUMA-MIL ensembling protocol)
# ============================================================

import os
import gc
import json
import time
import math
import copy
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    average_precision_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG  (identical to Phase 3)
# ============================================================

class Config:
    PHASE1_DIR = "/kaggle/input/datasets/absanim/phase-1-output"
    WORK_DIR = "/kaggle/working"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    SEEDS = [42, 123, 456]   # same 3 seeds as PUMA-MIL ensemble

    NUM_CLASSES = 2
    D_IN = 384          # ViT-Small CLS token dimension
    D_MODEL = 128       # projection dimension (same as PUMA-MIL)
    MAX_SLICES = 48
    TRAIN_WINDOW = 11
    TOP_K = 4

    FAST_DEBUG = False

    TOTAL_EPOCHS = 14 if FAST_DEBUG else 32
    LR = 8e-4
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 64
    NUM_WORKERS = 0
    PATIENCE = 4 if FAST_DEBUG else 8
    CLIP_GRAD_NORM = 1.0

    FIG_DPI = 300

    # NOTE: Role-aware sampler boosts are PUMA-MIL's contribution.
    # Baselines use plain class-balanced sampling only.


CFG = Config()

SLICE_ROLE_TO_ID = {
    "far_neg": 0,
    "boundary_neg": 1,
    "positive": 2,
    "mid_neg": 3,
    "no_tumor": 4,
}


# ============================================================
# UTILITIES  (copied from Phase 3 for self-containment)
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def safe_auc(labels, prob1):
    labels, prob1 = np.asarray(labels), np.asarray(prob1)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, prob1))
    except Exception:
        return float("nan")


def safe_pr_auc(labels, prob1):
    labels, prob1 = np.asarray(labels), np.asarray(prob1)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(labels, prob1))
    except Exception:
        return float("nan")


def calculate_ece(probs, labels, n_bins=5):
    probs, labels = np.asarray(probs), np.asarray(labels)
    if len(labels) == 0:
        return 0.0
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    acc = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf <= bins[i + 1]) if i == 0 else (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / len(labels)) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def metrics_from_probs(probs, labels, threshold=0.5):
    probs, labels = np.asarray(probs), np.asarray(labels)
    if len(labels) == 0:
        return {k: 0.0 for k in ["accuracy","precision","recall","f1","auc","pr_auc","ece","balanced_accuracy","specificity","sensitivity"]}, np.zeros(0, dtype=np.int64)
    preds = (probs[:, 1] >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "auc": safe_auc(labels, probs[:, 1]),
        "pr_auc": safe_pr_auc(labels, probs[:, 1]),
        "ece": float(calculate_ece(probs, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)) if len(np.unique(labels)) > 1 else 0.0,
        "specificity": float(recall_score(labels, preds, pos_label=0, zero_division=0)),
        "sensitivity": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
    }, preds


def find_best_threshold(probs, labels):
    probs, labels = np.asarray(probs), np.asarray(labels)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.50, 0.0
    best_t, best_f1 = 0.50, -1.0
    for t in np.arange(0.10, 0.91, 0.01):
        preds = (probs[:, 1] >= t).astype(int)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def fit_temperature(logits, labels, device):
    logits, labels = np.asarray(logits), np.asarray(labels)
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


# ============================================================
# DATA LOADING  (identical to Phase 3)
# ============================================================

def load_npz(name):
    candidates = [
        os.path.join(CFG.PHASE1_DIR, f"{name}_features_4mod_v3.npz"),
        os.path.join(CFG.PHASE1_DIR, "results", f"{name}_features_4mod_v3.npz"),
        os.path.join(CFG.WORK_DIR, f"{name}_features_4mod_v3.npz"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with np.load(path, allow_pickle=True) as data:
                return {k: data[k] for k in data.files}
    raise FileNotFoundError(f"Could not find {name}_features_4mod_v3.npz")


def _zeros_if_missing(data_dict, key, shape, dtype):
    return data_dict[key] if key in data_dict else np.zeros(shape, dtype=dtype)


class CenterWindowDataset(Dataset):
    """Identical to Phase 3 CenterWindowDataset for fairness."""
    def __init__(self, data_dict, window_size=11):
        assert window_size % 2 == 1
        self.window_size = window_size
        self.half = window_size // 2

        patient_ids = np.asarray(data_dict["patient_ids"]).astype(str)
        slice_idxs = np.asarray(data_dict["slice_idxs"]).astype(np.int64)
        f_t1 = np.asarray(data_dict["features_t1"], dtype=np.float32)
        f_t1ce = np.asarray(data_dict["features_t1ce"], dtype=np.float32)
        f_t2 = np.asarray(data_dict["features_t2"], dtype=np.float32)
        f_flair = np.asarray(data_dict["features_flair"], dtype=np.float32)
        labels = np.asarray(data_dict["labels"], dtype=np.int64)
        mod_mask = np.asarray(data_dict["modality_mask"], dtype=np.int64)
        n = len(labels)
        dist = _zeros_if_missing(data_dict, "distance_to_positive", (n,), np.int64).astype(np.int64)
        role_id = _zeros_if_missing(data_dict, "slice_role_id", (n,), np.int64).astype(np.int64)

        self.patient_data = {}
        pid_to_idx = {}
        for i, pid in enumerate(patient_ids):
            pid_to_idx.setdefault(pid, []).append(i)
        for pid in sorted(pid_to_idx):
            idxs = np.asarray(pid_to_idx[pid], dtype=np.int64)
            order = np.argsort(slice_idxs[idxs], kind="mergesort")
            idxs = idxs[order]
            self.patient_data[pid] = {
                "t1": f_t1[idxs], "t1ce": f_t1ce[idxs],
                "t2": f_t2[idxs], "flair": f_flair[idxs],
                "labels": labels[idxs], "slice_idxs": slice_idxs[idxs],
                "modality_mask": mod_mask[idxs],
                "distance_to_positive": dist[idxs],
                "slice_role_id": role_id[idxs],
            }

        self.index = []
        for pid in sorted(self.patient_data):
            y = self.patient_data[pid]["labels"]
            roles = self.patient_data[pid]["slice_role_id"]
            sx = self.patient_data[pid]["slice_idxs"]
            bag_label = int(y.max() > 0)
            for local_idx in range(len(y)):
                self.index.append({
                    "pid": pid, "local_idx": int(local_idx),
                    "center_label": int(y[local_idx]),
                    "bag_label": bag_label,
                    "slice_role_id": int(roles[local_idx]),
                    "slice_idx": int(sx[local_idx]),
                })

    def __len__(self):
        return len(self.index)

    def _build_window(self, arr, center_idx):
        out, mask = [], []
        for offset in range(-self.half, self.half + 1):
            j = center_idx + offset
            if 0 <= j < len(arr):
                out.append(arr[j]); mask.append(True)
            else:
                out.append(np.zeros(arr.shape[1], dtype=np.float32)); mask.append(False)
        return np.stack(out, axis=0), np.array(mask, dtype=bool)

    def __getitem__(self, idx):
        meta = self.index[idx]
        pid, c = meta["pid"], meta["local_idx"]
        d = self.patient_data[pid]
        t1, seq_mask = self._build_window(d["t1"], c)
        t1ce, _ = self._build_window(d["t1ce"], c)
        t2, _ = self._build_window(d["t2"], c)
        flair, _ = self._build_window(d["flair"], c)
        valid_mm = d["modality_mask"]
        bag_mod_mask = (valid_mm.sum(axis=0) > 0).astype(np.int64)
        if bag_mod_mask.sum() == 0:
            bag_mod_mask[:] = 1
        return (
            torch.tensor(t1, dtype=torch.float32),
            torch.tensor(t1ce, dtype=torch.float32),
            torch.tensor(t2, dtype=torch.float32),
            torch.tensor(flair, dtype=torch.float32),
            torch.tensor(int(meta["bag_label"]), dtype=torch.long),
            torch.tensor(int(meta["center_label"]), dtype=torch.long),
            torch.tensor(seq_mask, dtype=torch.bool),
            torch.tensor(bag_mod_mask, dtype=torch.bool),
            torch.tensor(self.half, dtype=torch.long),
            torch.tensor(int(meta["slice_role_id"]), dtype=torch.long),
            str(pid),
            torch.tensor(int(meta["slice_idx"]), dtype=torch.long),
        )


def compute_class_balanced_weights(ds):
    """Plain class-balanced sampling: only correct for class imbalance.
    No role-aware boosts (those are PUMA-MIL's contribution).
    """
    labels = np.array([int(item["center_label"]) for item in ds.index])
    n0, n1 = (labels == 0).sum(), (labels == 1).sum()
    n_total = len(labels)
    w0 = n_total / (2.0 * max(n0, 1))
    w1 = n_total / (2.0 * max(n1, 1))
    weights = np.where(labels == 1, w1, w0).astype(np.float64)
    return weights


def make_loader(data_dict, training=False):
    ds = CenterWindowDataset(data_dict, window_size=CFG.TRAIN_WINDOW)
    if training:
        weights = compute_class_balanced_weights(ds)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights), replacement=True,
        )
        loader = DataLoader(ds, batch_size=CFG.BATCH_SIZE, sampler=sampler,
                            num_workers=CFG.NUM_WORKERS, pin_memory=torch.cuda.is_available())
    else:
        loader = DataLoader(ds, batch_size=CFG.EVAL_BATCH_SIZE, shuffle=False,
                            num_workers=CFG.NUM_WORKERS, pin_memory=torch.cuda.is_available())
    return ds, loader


def load_data():
    candidates = [
        os.path.join(CFG.PHASE1_DIR, "patient_splits_v3.json"),
        os.path.join(CFG.PHASE1_DIR, "results", "patient_splits_v3.json"),
        os.path.join(CFG.WORK_DIR, "patient_splits_v3.json"),
    ]
    splits = None
    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r") as f:
                splits = json.load(f)
            break
    if splits is None:
        raise FileNotFoundError("patient_splits_v3.json not found")
    tr = load_npz("train"); vl = load_npz("val")
    tu = load_npz("tune"); te = load_npz("test")
    for name, d in [("train", tr), ("val", vl), ("tune", tu), ("test", te)]:
        required = ["features_t1","features_t1ce","features_t2","features_flair","labels","patient_ids","slice_idxs","modality_mask"]
        missing = [k for k in required if k not in d]
        if missing:
            raise KeyError(f"{name} missing: {missing}")
    print(f"Train: {len(tr['labels'])}, Val: {len(vl['labels'])}, "
          f"Tune: {len(tu['labels'])}, Test: {len(te['labels'])}")
    return splits, tr, vl, tu, te


# ============================================================
# SIMPLE FUSION: concat 4 modalities -> project to D_MODEL
# All competitive baselines use the same input representation
# ============================================================

class SimpleModalityFusion(nn.Module):
    """Fuse 4 modality features by concatenation + projection.
    Input:  4 tensors of shape (B, W, 384) -> Output: (B, W, D_MODEL)
    """
    def __init__(self, d_in=384, d_model=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in * 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, t1, t1ce, t2, flair, seq_mask):
        # Concatenate all 4 modalities along feature dim
        x = torch.cat([t1, t1ce, t2, flair], dim=-1)  # (B, W, 1536)
        return self.proj(x)  # (B, W, d_model)


# ============================================================
# MODEL 1: ABMIL  (Ilse et al., ICML 2018)
# Attention-Based MIL with gated attention mechanism
# Applied to the 11-slice window: center-slice classification
# ============================================================

class ABMIL(nn.Module):
    """Attention-Based MIL (Ilse et al. 2018).
    Uses gated attention on the 11-slice window, then classifies
    the attention-pooled bag representation.
    """
    def __init__(self, d_in=384, d_model=128, d_attn=64):
        super().__init__()
        self.fusion = SimpleModalityFusion(d_in, d_model)
        # Gated attention mechanism (Eq. 9 in Ilse et al.)
        self.attn_V = nn.Sequential(nn.Linear(d_model, d_attn), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(d_model, d_attn), nn.Sigmoid())
        self.attn_w = nn.Linear(d_attn, 1)
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, t1, t1ce, t2, flair, seq_mask, **kwargs):
        h = self.fusion(t1, t1ce, t2, flair, seq_mask)  # (B, W, d_model)
        # Gated attention
        a_V = self.attn_V(h)   # (B, W, d_attn)
        a_U = self.attn_U(h)   # (B, W, d_attn)
        a = self.attn_w(a_V * a_U).squeeze(-1)  # (B, W)
        a = a.masked_fill(~seq_mask, -1e9)
        a = torch.softmax(a, dim=1)  # (B, W)
        # Attention pooling
        bag = (h * a.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
        logits = self.classifier(bag)  # (B, 2)
        return logits


# ============================================================
# MODEL 2: TransMIL  (Shao et al., NeurIPS 2021)
# Transformer-based MIL with PPEG and pyramid positions
# Simplified to match our feature setting (precomputed ViT features)
# ============================================================

class PPEG(nn.Module):
    """Pyramid Position Encoding Generator (from TransMIL).
    Adapted: since our instances are 1D sequence (slices), we use
    1D convolutions instead of 2D.
    """
    def __init__(self, d_model=128):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.conv1 = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, 5, padding=2, groups=d_model)
        self.conv3 = nn.Conv1d(d_model, d_model, 7, padding=3, groups=d_model)

    def forward(self, x):
        # x: (B, W, d_model)
        cls_token = x[:, :1, :]
        feat = x[:, 1:, :]  # skip CLS for position encoding
        xt = feat.transpose(1, 2)  # (B, d_model, W-1)
        x1 = self.conv1(xt).transpose(1, 2)
        x2 = self.conv2(xt).transpose(1, 2)
        x3 = self.conv3(xt).transpose(1, 2)
        feat = feat + x1 + x2 + x3
        return torch.cat([cls_token, feat], dim=1)


class TransMIL(nn.Module):
    """TransMIL (Shao et al., NeurIPS 2021).
    CLS token + Transformer encoder + PPEG for positional info.
    Applied on the 11-slice window with 4-modality fusion.
    """
    def __init__(self, d_in=384, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.fusion = SimpleModalityFusion(d_in, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.ppeg = PPEG(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.10, batch_first=True, activation="gelu",
        )
        self.encoder1 = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, t1, t1ce, t2, flair, seq_mask, **kwargs):
        h = self.fusion(t1, t1ce, t2, flair, seq_mask)  # (B, W, d_model)
        B = h.size(0)
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)  # (B, W+1, d_model)
        # Extend mask for CLS
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=h.device)
        full_mask = torch.cat([cls_mask, seq_mask], dim=1)
        # Transformer + PPEG
        h = self.encoder1(h, src_key_padding_mask=~full_mask)
        h = self.ppeg(h)
        h = self.encoder1(h, src_key_padding_mask=~full_mask)
        h = self.norm(h)
        # Use CLS token for classification
        logits = self.classifier(h[:, 0, :])
        return logits


# ============================================================
# MODEL 3: CLAM  (Lu et al., Nature BME 2021)
# Clustering-constrained Attention MIL
# Single-branch version (CLAM-SB) adapted for our setting
# ============================================================

class CLAM_SB(nn.Module):
    """CLAM Single-Branch (Lu et al. 2021).
    Gated attention + instance-level clustering loss (auxiliary).
    We expose the instance loss to the training loop.
    """
    def __init__(self, d_in=384, d_model=128, d_attn=64, n_clusters=8):
        super().__init__()
        self.fusion = SimpleModalityFusion(d_in, d_model)
        # Attention (gated)
        self.attn_V = nn.Sequential(nn.Linear(d_model, d_attn), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(d_model, d_attn), nn.Sigmoid())
        self.attn_w = nn.Linear(d_attn, 1)
        # Instance classifier for top-k (clustering constraint)
        self.inst_classifier = nn.Sequential(
            nn.Linear(d_model, 2),
        )
        self.n_clusters = n_clusters
        # Bag classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, t1, t1ce, t2, flair, seq_mask, return_inst_loss=False, center_label=None, **kwargs):
        h = self.fusion(t1, t1ce, t2, flair, seq_mask)  # (B, W, d_model)
        # Gated attention
        a_V = self.attn_V(h)
        a_U = self.attn_U(h)
        a = self.attn_w(a_V * a_U).squeeze(-1)
        a = a.masked_fill(~seq_mask, -1e9)
        a_weights = torch.softmax(a, dim=1)
        # Attention pooling
        bag = (h * a_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(bag)

        inst_loss = torch.tensor(0.0, device=h.device)
        if return_inst_loss and center_label is not None:
            # Instance-level clustering constraint:
            # For positive bags, top-attended instances should be positive
            # For negative bags, top-attended instances should be negative
            for b_idx in range(h.size(0)):
                valid = seq_mask[b_idx]
                if valid.sum() == 0:
                    continue
                valid_h = h[b_idx, valid]      # (n_valid, d_model)
                valid_a = a_weights[b_idx, valid]  # (n_valid,)
                k = min(self.n_clusters, valid_h.size(0))
                topk_idx = torch.topk(valid_a, k).indices
                inst_logits = self.inst_classifier(valid_h[topk_idx])
                inst_labels = torch.full((k,), int(center_label[b_idx].item()),
                                         dtype=torch.long, device=h.device)
                inst_loss = inst_loss + F.cross_entropy(inst_logits, inst_labels)
            inst_loss = inst_loss / max(h.size(0), 1)

        if return_inst_loss:
            return logits, inst_loss
        return logits


# ============================================================
# MODEL 4: ViT-Slice-MLP (single-slice deep classifier)
# Strong single-slice baseline using the same ViT features
# ============================================================

class ViTSliceMLP(nn.Module):
    """Deep MLP classifier on center-slice ViT features.
    Uses only the center slice (no window context) with
    4-modality concatenation and a 3-layer MLP.
    This represents a 'best single-slice ViT classifier'.
    """
    def __init__(self, d_in=384, d_hidden=256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_in * 4, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(d_hidden // 2, 2),
        )

    def forward(self, t1, t1ce, t2, flair, seq_mask, center_pos=None, **kwargs):
        # Use only center slice
        if center_pos is None:
            c = t1.size(1) // 2
        else:
            c = center_pos[0].item()  # same for all in batch
        x = torch.cat([t1[:, c], t1ce[:, c], t2[:, c], flair[:, c]], dim=-1)
        return self.classifier(x)


# ============================================================
# TRAINING AND EVALUATION LOOP
# ============================================================

def evaluate_model(model, loader, device, threshold=0.5, temperature=1.0, model_type="bag"):
    model.eval()
    all_logits, all_probs, all_labels, all_pids, all_slice_idxs = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            t1, t1ce, t2, flair = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            center_y = batch[5]
            seq_mask = batch[6].to(device)
            center_pos = batch[8].to(device)
            pid = batch[10]
            slice_idx = batch[11]

            logits = model(t1, t1ce, t2, flair, seq_mask, center_pos=center_pos)
            logits = logits / max(float(temperature), 1e-6)
            probs = torch.softmax(logits, dim=-1)

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(center_y.numpy())
            all_pids.extend(list(pid))
            all_slice_idxs.append(slice_idx.numpy())

    logits = np.concatenate(all_logits, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    slice_idxs = np.concatenate(all_slice_idxs, axis=0).astype(np.int64)
    metrics, preds = metrics_from_probs(probs, labels, threshold=threshold)

    return {
        "logits": logits, "probs": probs, "labels": labels, "preds": preds,
        "pids": np.asarray(all_pids).astype(str), "slice_idxs": slice_idxs,
        "metrics": metrics,
    }


def compute_val_loss(model, val_loader, ce_weight, use_inst_loss, device):
    """Compute validation loss (CE) without gradients."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            t1 = batch[0].to(device)
            t1ce = batch[1].to(device)
            t2 = batch[2].to(device)
            flair = batch[3].to(device)
            center_y = batch[5].to(device)
            seq_mask = batch[6].to(device)
            center_pos = batch[8].to(device)
            logits = model(t1, t1ce, t2, flair, seq_mask, center_pos=center_pos)
            loss = F.cross_entropy(logits, center_y, weight=ce_weight)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def train_single_seed(
    name, model_cls, model_kwargs, train_data, val_data,
    seed=42, use_inst_loss=False
):
    """Train one seed of a competitive baseline.
    Model selection uses pure validation macro-F1 (same as PUMA-MIL).
    """
    set_seed(seed)
    clear_memory()
    model = model_cls(**model_kwargs).to(CFG.DEVICE)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Seed {seed} | Params: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)

    _, train_loader = make_loader(train_data, training=True)
    _, val_loader = make_loader(val_data, training=False)

    # Compute class weights for CE loss
    train_labels = np.asarray(train_data["labels"])
    n0, n1 = (train_labels == 0).sum(), (train_labels == 1).sum()
    ce_weight = torch.tensor([len(train_labels) / (2.0 * max(n0, 1)),
                              len(train_labels) / (2.0 * max(n1, 1))],
                             dtype=torch.float32, device=CFG.DEVICE)

    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0     # FIX #2: pure macro-F1 for model selection
    best_epoch = 0
    bad_epochs = 0
    history = []

    t_start = time.time()

    for epoch in range(1, CFG.TOTAL_EPOCHS + 1):
        model.train()
        loss_meter = []
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            t1 = batch[0].to(CFG.DEVICE)
            t1ce = batch[1].to(CFG.DEVICE)
            t2 = batch[2].to(CFG.DEVICE)
            flair = batch[3].to(CFG.DEVICE)
            center_y = batch[5].to(CFG.DEVICE)
            seq_mask = batch[6].to(CFG.DEVICE)
            center_pos = batch[8].to(CFG.DEVICE)

            optimizer.zero_grad(set_to_none=True)

            if use_inst_loss:
                logits, inst_loss = model(
                    t1, t1ce, t2, flair, seq_mask,
                    return_inst_loss=True, center_label=center_y, center_pos=center_pos,
                )
                ce_loss = F.cross_entropy(logits, center_y, weight=ce_weight)
                loss = ce_loss + 0.3 * inst_loss
            else:
                logits = model(t1, t1ce, t2, flair, seq_mask, center_pos=center_pos)
                loss = F.cross_entropy(logits, center_y, weight=ce_weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.CLIP_GRAD_NORM)
            optimizer.step()
            loss_meter.append(loss.item())

            # Track train accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                train_correct += (preds == center_y).sum().item()
                train_total += center_y.size(0)

        # Validation — select by pure macro-F1
        val_result = evaluate_model(model, val_loader, CFG.DEVICE)
        vm = val_result["metrics"]
        val_f1 = vm["f1"]

        # Compute val_loss
        val_loss = compute_val_loss(model, val_loader, ce_weight, use_inst_loss, CFG.DEVICE)

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(loss_meter)),
            "train_accuracy": train_correct / max(train_total, 1),
            "val_loss": float(val_loss),
            "val_f1": vm["f1"], "val_auc": vm["auc"],
            "val_acc": vm["accuracy"], "val_spec": vm["specificity"],
            "val_sens": vm["sensitivity"],
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 4 == 0 or epoch == 1:
            print(f"    Ep {epoch:3d}  loss={np.mean(loss_meter):.4f}  "
                  f"val_F1={vm['f1']:.4f}  val_AUC={vm['auc']:.4f}  "
                  f"{'*BEST*' if bad_epochs == 0 else ''}")

        if bad_epochs >= CFG.PATIENCE:
            print(f"    Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t_start
    print(f"    Best epoch: {best_epoch}  ({elapsed:.1f}s)")

    model.load_state_dict(best_state)
    model.eval()
    return model, param_count, elapsed, history


def train_competitive_baseline(
    name, model_cls, model_kwargs, train_data, val_data, tune_data, test_data,
    use_inst_loss=False
):
    """Train a 3-seed ensemble of a competitive baseline.
    FIX #1: Identical ensembling protocol to PUMA-MIL.
    """
    print(f"\n{'='*72}")
    print(f"TRAINING 3-SEED ENSEMBLE: {name}")
    print(f"{'='*72}")

    seed_models = []
    all_histories = {}
    total_time = 0.0
    param_count = 0

    for seed in CFG.SEEDS:
        print(f"\n  --- Seed {seed} ---")
        model, pc, elapsed, hist = train_single_seed(
            name, model_cls, model_kwargs,
            train_data, val_data, seed=seed,
            use_inst_loss=use_inst_loss,
        )
        seed_models.append(model)
        all_histories[f"seed_{seed}"] = hist
        total_time += elapsed
        param_count = pc

    # ---- Per-seed temperature scaling + threshold on tune ----
    _, tune_loader = make_loader(tune_data, training=False)
    _, test_loader = make_loader(test_data, training=False)

    seed_temperatures = []
    seed_thresholds = []
    seed_test_probs = []
    seed_test_metrics = []

    for i, (seed, model) in enumerate(zip(CFG.SEEDS, seed_models)):
        # Temperature scaling
        tune_raw = evaluate_model(model, tune_loader, CFG.DEVICE, threshold=0.5, temperature=1.0)
        temp = fit_temperature(tune_raw["logits"], tune_raw["labels"], CFG.DEVICE)
        seed_temperatures.append(temp)

        # Threshold tuning with temperature
        tune_cal = evaluate_model(model, tune_loader, CFG.DEVICE, threshold=0.5, temperature=temp)
        thr, tune_f1 = find_best_threshold(tune_cal["probs"], tune_cal["labels"])
        seed_thresholds.append(thr)

        # Per-seed test
        test_res = evaluate_model(model, test_loader, CFG.DEVICE, threshold=thr, temperature=temp)
        seed_test_probs.append(test_res["probs"])
        seed_test_metrics.append(test_res["metrics"])
        tm = test_res["metrics"]
        print(f"  Seed {seed}: F1={tm['f1']:.4f}  AUC={tm['auc']:.4f}  "
              f"Spec={tm['specificity']:.4f}  Sens={tm['sensitivity']:.4f}")

    # ---- Ensemble: average softmax probabilities ----
    ensemble_probs = np.mean(seed_test_probs, axis=0)
    # Tune ensemble threshold on tune set
    tune_ensemble_probs = []
    for model, temp in zip(seed_models, seed_temperatures):
        tune_res = evaluate_model(model, tune_loader, CFG.DEVICE, threshold=0.5, temperature=temp)
        tune_ensemble_probs.append(tune_res["probs"])
    tune_ens_probs = np.mean(tune_ensemble_probs, axis=0)
    tune_labels = tune_res["labels"]
    ens_thr, ens_tune_f1 = find_best_threshold(tune_ens_probs, tune_labels)

    # Final ensemble test metrics
    test_labels = test_res["labels"]  # same for all seeds
    ens_metrics, ens_preds = metrics_from_probs(ensemble_probs, test_labels, threshold=ens_thr)

    print(f"\n  ENSEMBLE RESULTS ({name}):")
    print(f"  Threshold={ens_thr:.2f}")
    print(f"  Acc={ens_metrics['accuracy']:.4f}  F1={ens_metrics['f1']:.4f}  "
          f"AUC={ens_metrics['auc']:.4f}  BalAcc={ens_metrics['balanced_accuracy']:.4f}")
    print(f"  Sens={ens_metrics['sensitivity']:.4f}  Spec={ens_metrics['specificity']:.4f}  "
          f"ECE={ens_metrics['ece']:.4f}")

    return {
        "name": name,
        "temperatures": seed_temperatures,
        "thresholds": seed_thresholds,
        "ensemble_threshold": float(ens_thr),
        "ensemble_tune_f1": float(ens_tune_f1),
        "trainable_params": param_count,
        "training_time_s": total_time,
        "test_metrics": ens_metrics,
        "per_seed_metrics": seed_test_metrics,
        "test_preds": ens_preds,
        "test_labels": test_labels,
        "test_probs": ensemble_probs,
        "test_pids": test_res["pids"],
        "test_slice_idxs": test_res["slice_idxs"],
        "histories": all_histories,
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_confusion_matrix(labels, preds, name, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-Tumor","Tumor"])
    ax.set_yticklabels(["Non-Tumor","Tumor"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
    ax.set_title(f"CM: {name}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=CFG.FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_training_curves(history, name, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = [h["epoch"] for h in history]
    axes[0].plot(epochs, [h["train_loss"] for h in history], 'b-')
    axes[0].set_title(f"{name}: Training Loss"); axes[0].set_xlabel("Epoch")
    axes[1].plot(epochs, [h["val_f1"] for h in history], 'g-', label="F1")
    axes[1].plot(epochs, [h["val_acc"] for h in history], 'b--', label="Acc")
    axes[1].legend(); axes[1].set_title(f"{name}: Validation")
    axes[2].plot(epochs, [h["val_sens"] for h in history], 'r-', label="Sens")
    axes[2].plot(epochs, [h["val_spec"] for h in history], 'b-', label="Spec")
    axes[2].legend(); axes[2].set_title(f"{name}: Sens/Spec")
    plt.tight_layout()
    plt.savefig(save_path, dpi=CFG.FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_comparison_chart(all_results, save_path):
    names = [r["name"] for r in all_results]
    metrics_list = ["f1", "auc", "balanced_accuracy", "specificity", "sensitivity"]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    width = 0.15
    for i, m in enumerate(metrics_list):
        vals = [r["test_metrics"][m] for r in all_results]
        ax.bar(x + i * width, vals, width, label=m.upper())
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0.80, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Competitive Baselines Comparison")
    plt.tight_layout()
    plt.savefig(save_path, dpi=CFG.FIG_DPI, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(CFG.SEEDS[0])
    splits, tr, vl, tu, te = load_data()

    all_results = []

    # Define all baselines: (name, model_class, model_kwargs, use_inst_loss)
    baseline_specs = [
        ("ABMIL",         ABMIL,       {"d_in": CFG.D_IN, "d_model": CFG.D_MODEL},   False),
        ("TransMIL",      TransMIL,    {"d_in": CFG.D_IN, "d_model": CFG.D_MODEL},   False),
        ("CLAM-SB",       CLAM_SB,     {"d_in": CFG.D_IN, "d_model": CFG.D_MODEL},   True),
        ("ViT-Slice-MLP", ViTSliceMLP, {"d_in": CFG.D_IN},                           False),
    ]

    for bname, model_cls, model_kwargs, use_inst in baseline_specs:
        r = train_competitive_baseline(
            bname, model_cls, model_kwargs,
            tr, vl, tu, te,
            use_inst_loss=use_inst,
        )
        all_results.append(r)
        plot_confusion_matrix(r["test_labels"], r["test_preds"], bname,
                             os.path.join(CFG.WORK_DIR, f"phase3b_cm_{bname}.png"))
        # Plot first seed's history
        first_hist = list(r["histories"].values())[0]
        plot_training_curves(first_hist, bname,
                            os.path.join(CFG.WORK_DIR, f"phase3b_curves_{bname}.png"))
        clear_memory()

    # ---- Summary ----
    print(f"\n{'='*72}")
    print("COMPETITIVE BASELINES — 3-SEED ENSEMBLE SUMMARY")
    print(f"{'='*72}")
    print(f"{'Method':<20} {'Acc':>7} {'F1':>7} {'AUC':>7} {'BalAcc':>7} {'Spec':>7} {'Sens':>7} {'ECE':>7}")
    print("-" * 79)
    for r in all_results:
        m = r["test_metrics"]
        print(f"{r['name']:<20} {m['accuracy']:>7.4f} {m['f1']:>7.4f} {m['auc']:>7.4f} "
              f"{m['balanced_accuracy']:>7.4f} {m['specificity']:>7.4f} {m['sensitivity']:>7.4f} "
              f"{m['ece']:>7.4f}")
    print()
    print("Per-seed breakdown:")
    for r in all_results:
        for i, (seed, sm) in enumerate(zip(CFG.SEEDS, r["per_seed_metrics"])):
            print(f"  {r['name']} seed {seed}: F1={sm['f1']:.4f} AUC={sm['auc']:.4f} "
                  f"Spec={sm['specificity']:.4f} Sens={sm['sensitivity']:.4f}")

    # ---- Save ----
    plot_comparison_chart(all_results,
                         os.path.join(CFG.WORK_DIR, "phase3b_comparison_chart.png"))

    # Save predictions for McNemar's test
    pred_store = {}
    for r in all_results:
        pred_store[r["name"]] = {
            "patient_id": r["test_pids"].tolist(),
            "slice_idx": r["test_slice_idxs"].astype(int).tolist(),
            "label": r["test_labels"].astype(int).tolist(),
            "pred": r["test_preds"].astype(int).tolist(),
            "prob_1": r["test_probs"][:, 1].astype(float).tolist(),
        }
    save_json(pred_store, os.path.join(CFG.WORK_DIR, "phase3b_competitive_predictions.json"))

    # Save summary metrics
    summary = {}
    for r in all_results:
        summary[r["name"]] = {
            "family": "competitive_baseline",
            "ensemble": "3-seed average",
            "seeds": CFG.SEEDS,
            "temperatures": r["temperatures"],
            "ensemble_threshold": r["ensemble_threshold"],
            "ensemble_tune_f1": r["ensemble_tune_f1"],
            "trainable_params": r["trainable_params"],
            "training_time_s": r["training_time_s"],
            "per_seed_metrics": r["per_seed_metrics"],
            **r["test_metrics"],
        }
    save_json(summary, os.path.join(CFG.WORK_DIR, "phase3b_competitive_results.json"))

    # Save histories
    histories = {r["name"]: r["histories"] for r in all_results}
    save_json(histories, os.path.join(CFG.WORK_DIR, "phase3b_training_histories.json"))

    print(f"\n{'='*72}")
    print("ALL COMPETITIVE BASELINES COMPLETE (3-SEED ENSEMBLE)")
    print(f"Outputs saved to {CFG.WORK_DIR}/phase3b_*")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
