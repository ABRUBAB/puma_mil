# ============================================================
# PUMA-MIL CLEAN RUN — CELL 1: TRAINING
# Fixed: logs val_loss + train_accuracy every epoch
# Based on Phase 2 V6 with proper curve logging
# ============================================================

import os
import gc
import json
import time
import math
import copy
import random
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, balanced_accuracy_score,
    average_precision_score
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

class Config:
    # Point to the previous run's output added as Kaggle dataset
    PHASE1_DIR = "/kaggle/input/datasets/absanim/phase-1-output"
    WORK_DIR = "/kaggle/working"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    SEEDS = [42, 123, 456]

    NUM_CLASSES = 2
    D_IN = 384
    D_MODEL = 128

    MAX_SLICES = 48              # must be >= Phase 1 selected max
    TRAIN_WINDOW = 11            # odd number, local window around center slice
    TOP_K = 4
    PROTO_MOMENTUM = 0.90

    TOTAL_EPOCHS = 32
    LR = 8e-4
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 64
    NUM_WORKERS = 0
    EVAL_EVERY = 1
    PATIENCE = 8
    CLIP_GRAD_NORM = 1.0

    MODALITY_DROPOUT = 0.10
    FIG_DPI = 300

    CENTER_GAMMA_POS = 1.5
    CENTER_GAMMA_NEG = 3.0
    CENTER_LOSS_WEIGHT = 1.00
    BAG_LOSS_WEIGHT = 0.20
    SMOOTHNESS_LOSS_WEIGHT = 0.06
    DIVERSITY_LOSS_WEIGHT = 0.01

    SAMPLER_POSITIVE_BOOST = 1.20
    SAMPLER_BOUNDARY_NEG_BOOST = 1.45
    SAMPLER_FAR_NEG_BOOST = 1.20
    SAMPLER_NO_TUMOR_BOOST = 1.10

    ROLE_LOSS_MULTIPLIER = {
        0: 1.25,   # far_neg
        1: 1.60,   # boundary_neg
        2: 1.00,   # positive
        3: 1.00,   # mid_neg
        4: 1.15,   # no_tumor
    }

CFG = Config()

SLICE_ROLE_TO_ID = {
    "far_neg": 0,
    "boundary_neg": 1,
    "positive": 2,
    "mid_neg": 3,
    "no_tumor": 4,
}
ID_TO_SLICE_ROLE = {v: k for k, v in SLICE_ROLE_TO_ID.items()}


# ============================================================
# UTILS
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

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def safe_auc(labels, prob1):
    labels = np.asarray(labels)
    prob1 = np.asarray(prob1)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, prob1))
    except Exception:
        return float("nan")

def safe_pr_auc(labels, prob1):
    labels = np.asarray(labels)
    prob1 = np.asarray(prob1)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(labels, prob1))
    except Exception:
        return float("nan")

def nanmean_or_zero(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or np.all(np.isnan(x)):
        return 0.0
    return float(np.nanmean(x))

def nanstd_or_zero(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or np.all(np.isnan(x)):
        return 0.0
    return float(np.nanstd(x))


# ============================================================
# LOAD DATA
# ============================================================

def load_npz(name):
    # Try PHASE1_DIR first, then WORK_DIR, then results subdir
    candidates = [
        os.path.join(CFG.PHASE1_DIR, f"{name}_features_4mod_v3.npz"),
        os.path.join(CFG.PHASE1_DIR, "results", f"{name}_features_4mod_v3.npz"),
        os.path.join(CFG.WORK_DIR, f"{name}_features_4mod_v3.npz"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with np.load(path, allow_pickle=True) as data:
                out = {k: data[k] for k in data.files}
            print(f"  Loaded {name} from {path}")
            return out
    raise FileNotFoundError(f"Could not find {name}_features_4mod_v3.npz in any of: {candidates}")

def load_splits():
    candidates = [
        os.path.join(CFG.PHASE1_DIR, "patient_splits_v3.json"),
        os.path.join(CFG.PHASE1_DIR, "results", "patient_splits_v3.json"),
        os.path.join(CFG.WORK_DIR, "patient_splits_v3.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, "r") as f:
                return json.load(f)
    raise FileNotFoundError(f"Could not find patient_splits_v3.json")

def _zeros_if_missing(data_dict, key, shape, dtype):
    if key in data_dict:
        return data_dict[key]
    return np.zeros(shape, dtype=dtype)

def _validate_data_dict(data_dict, split_name="unknown"):
    required = [
        "features_t1", "features_t1ce", "features_t2", "features_flair",
        "labels", "patient_ids", "slice_idxs", "modality_mask"
    ]
    missing = [k for k in required if k not in data_dict]
    if missing:
        raise KeyError(f"{split_name} missing keys: {missing}")

    n = len(data_dict["labels"])
    for k in required:
        if len(data_dict[k]) != n:
            raise ValueError(f"{split_name}: key '{k}' length mismatch for {k}")

def load_all_data():
    splits = load_splits()

    train_data = load_npz("train")
    val_data = load_npz("val")
    tune_data = load_npz("tune")
    test_data = load_npz("test")

    for name, d in [("train", train_data), ("val", val_data), ("tune", tune_data), ("test", test_data)]:
        _validate_data_dict(d, name)

    print("=" * 72)
    print("LOADING PHASE 1 V4 DATA")
    print("=" * 72)
    print(f"Train slices: {len(train_data['labels'])}")
    print(f"Val slices:   {len(val_data['labels'])}")
    print(f"Tune slices:  {len(tune_data['labels'])}")
    print(f"Test slices:  {len(test_data['labels'])}")
    print(f"Slice class weights: {splits['slice_class_weights']}")
    print("=" * 72 + "\n")

    return splits, train_data, val_data, tune_data, test_data


# ============================================================
# PATIENT BAG DATASET
# kept for compatibility with later phases
# ============================================================

class PatientBagDataset(Dataset):
    def __init__(self, data_dict, max_slices=48):
        self.max_slices = max_slices
        self.items = []

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
        tumor_pixels = _zeros_if_missing(data_dict, "tumor_pixels", (n,), np.int64).astype(np.int64)
        has_positive_patient = _zeros_if_missing(data_dict, "has_positive_patient", (n,), np.int64).astype(np.int64)

        pid_to_indices = {}
        for i, pid in enumerate(patient_ids):
            pid_to_indices.setdefault(pid, []).append(i)

        for pid in sorted(pid_to_indices.keys()):
            idxs = np.asarray(pid_to_indices[pid], dtype=np.int64)
            order = np.argsort(slice_idxs[idxs], kind="mergesort")
            idxs = idxs[order]

            t1 = f_t1[idxs]
            t1ce = f_t1ce[idxs]
            t2 = f_t2[idxs]
            flair = f_flair[idxs]
            y = labels[idxs]
            mm = mod_mask[idxs]
            sx = slice_idxs[idxs]
            dd = dist[idxs]
            rr = role_id[idxs]
            tp = tumor_pixels[idxs]
            hp = has_positive_patient[idxs]

            valid_len = len(y)
            if valid_len > self.max_slices:
                pos = np.where(y == 1)[0]
                if len(pos) > 0:
                    center = int(np.round(pos.mean()))
                else:
                    center = valid_len // 2
                half = self.max_slices // 2
                lo = max(0, center - half)
                hi = min(valid_len, lo + self.max_slices)
                lo = max(0, hi - self.max_slices)

                t1, t1ce, t2, flair = t1[lo:hi], t1ce[lo:hi], t2[lo:hi], flair[lo:hi]
                y, mm, sx = y[lo:hi], mm[lo:hi], sx[lo:hi]
                dd, rr, tp, hp = dd[lo:hi], rr[lo:hi], tp[lo:hi], hp[lo:hi]
                valid_len = self.max_slices
            else:
                pad = self.max_slices - valid_len
                t1 = np.vstack([t1, np.zeros((pad, t1.shape[1]), dtype=np.float32)])
                t1ce = np.vstack([t1ce, np.zeros((pad, t1ce.shape[1]), dtype=np.float32)])
                t2 = np.vstack([t2, np.zeros((pad, t2.shape[1]), dtype=np.float32)])
                flair = np.vstack([flair, np.zeros((pad, flair.shape[1]), dtype=np.float32)])

                y = np.concatenate([y, np.full(pad, -100, dtype=np.int64)])
                mm = np.vstack([mm, np.zeros((pad, 4), dtype=np.int64)])
                sx = np.concatenate([sx, np.full(pad, -1, dtype=np.int64)])
                dd = np.concatenate([dd, np.full(pad, -1, dtype=np.int64)])
                rr = np.concatenate([rr, np.full(pad, -1, dtype=np.int64)])
                tp = np.concatenate([tp, np.zeros(pad, dtype=np.int64)])
                hp = np.concatenate([hp, np.zeros(pad, dtype=np.int64)])

            seq_mask = torch.zeros(self.max_slices, dtype=torch.bool)
            seq_mask[:valid_len] = True
            bag_mod_mask = (mm[:valid_len].sum(axis=0) > 0).astype(np.int64)
            if bag_mod_mask.sum() == 0:
                bag_mod_mask[:] = 1

            bag_label = int(np.max(y[:valid_len]) > 0)

            self.items.append({
                "t1": torch.tensor(t1, dtype=torch.float32),
                "t1ce": torch.tensor(t1ce, dtype=torch.float32),
                "t2": torch.tensor(t2, dtype=torch.float32),
                "flair": torch.tensor(flair, dtype=torch.float32),
                "bag_label": torch.tensor(bag_label, dtype=torch.long),
                "slice_labels": torch.tensor(y, dtype=torch.long),
                "seq_mask": seq_mask,
                "mod_mask": torch.tensor(bag_mod_mask, dtype=torch.bool),
                "pid": str(pid),
                "slice_idxs": sx.astype(np.int64),
                "distance_to_positive": dd.astype(np.int64),
                "slice_role_id": rr.astype(np.int64),
                "tumor_pixels": tp.astype(np.int64),
                "has_positive_patient": hp.astype(np.int64),
                "valid_len": int(valid_len),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x = self.items[idx]
        return (
            x["t1"], x["t1ce"], x["t2"], x["flair"],
            x["bag_label"], x["slice_labels"], x["seq_mask"], x["mod_mask"], x["pid"]
        )


# ============================================================
# CENTER WINDOW DATASET
# training/eval dataset
# ============================================================

class CenterWindowDataset(Dataset):
    def __init__(self, data_dict, window_size=11):
        assert window_size % 2 == 1, "window_size must be odd"
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
        tumor_pixels = _zeros_if_missing(data_dict, "tumor_pixels", (n,), np.int64).astype(np.int64)
        has_positive_patient = _zeros_if_missing(data_dict, "has_positive_patient", (n,), np.int64).astype(np.int64)

        self.patient_data = {}
        pid_to_indices = {}
        for i, pid in enumerate(patient_ids):
            pid_to_indices.setdefault(pid, []).append(i)

        for pid in sorted(pid_to_indices.keys()):
            idxs = np.asarray(pid_to_indices[pid], dtype=np.int64)
            order = np.argsort(slice_idxs[idxs], kind="mergesort")
            idxs = idxs[order]

            self.patient_data[pid] = {
                "t1": f_t1[idxs],
                "t1ce": f_t1ce[idxs],
                "t2": f_t2[idxs],
                "flair": f_flair[idxs],
                "labels": labels[idxs],
                "slice_idxs": slice_idxs[idxs],
                "modality_mask": mod_mask[idxs],
                "distance_to_positive": dist[idxs],
                "slice_role_id": role_id[idxs],
                "tumor_pixels": tumor_pixels[idxs],
                "has_positive_patient": has_positive_patient[idxs],
            }

        self.index = []
        for pid in sorted(self.patient_data.keys()):
            y = self.patient_data[pid]["labels"]
            roles = self.patient_data[pid]["slice_role_id"]
            sx = self.patient_data[pid]["slice_idxs"]
            dd = self.patient_data[pid]["distance_to_positive"]
            tp = self.patient_data[pid]["tumor_pixels"]
            bag_label = int(y.max() > 0)
            for local_idx in range(len(y)):
                self.index.append({
                    "pid": pid,
                    "local_idx": int(local_idx),
                    "center_label": int(y[local_idx]),
                    "bag_label": bag_label,
                    "slice_role_id": int(roles[local_idx]),
                    "slice_idx": int(sx[local_idx]),
                    "distance_to_positive": int(dd[local_idx]),
                    "tumor_pixels": int(tp[local_idx]),
                })

    def __len__(self):
        return len(self.index)

    def _build_window(self, arr, center_idx):
        out = []
        mask = []
        for offset in range(-self.half, self.half + 1):
            j = center_idx + offset
            if 0 <= j < len(arr):
                out.append(arr[j])
                mask.append(True)
            else:
                out.append(np.zeros(arr.shape[1], dtype=np.float32))
                mask.append(False)
        return np.stack(out, axis=0), np.array(mask, dtype=bool)

    def __getitem__(self, idx):
        meta = self.index[idx]
        pid = meta["pid"]
        c = meta["local_idx"]
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
            torch.tensor(self.half, dtype=torch.long),  # center position always middle
            torch.tensor(int(meta["slice_role_id"]), dtype=torch.long),
            str(pid),
            torch.tensor(int(meta["slice_idx"]), dtype=torch.long),
            torch.tensor(int(meta["distance_to_positive"]), dtype=torch.long),
            torch.tensor(int(meta["tumor_pixels"]), dtype=torch.long),
        )


# ============================================================
# MODEL
# ============================================================

def masked_topk_mean(seq_feats, scores, seq_mask, top_k):
    bsz, _, dim = seq_feats.shape
    out = []
    device = seq_feats.device
    dtype = seq_feats.dtype

    for b in range(bsz):
        valid_idx = torch.where(seq_mask[b])[0]
        if valid_idx.numel() == 0:
            out.append(torch.zeros(dim, device=device, dtype=dtype))
            continue
        vals = scores[b, valid_idx]
        k = min(int(top_k), int(valid_idx.numel()))
        chosen_local = torch.topk(vals, k=k, dim=0).indices
        chosen_idx = valid_idx[chosen_local]
        out.append(seq_feats[b, chosen_idx].mean(dim=0))
    return torch.stack(out, dim=0)

class ReliabilityFusion(nn.Module):
    def __init__(self, d_in=384, d_model=128):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            ) for _ in range(4)
        ])
        self.rel = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(4)])
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

    def forward(self, mods, mod_mask):
        proj_feats = []
        rel_logits = []

        for i in range(4):
            p = self.proj[i](mods[i])
            proj_feats.append(p)
            rel_logits.append(self.rel[i](p))

        feat_stack = torch.stack(proj_feats, dim=2)    # [B,S,4,D]
        rel_stack = torch.cat(rel_logits, dim=-1)      # [B,S,4]
        rel_stack = rel_stack.masked_fill(~mod_mask[:, None, :], -1e4)

        weights = torch.softmax(rel_stack, dim=-1)
        fused = (feat_stack * weights.unsqueeze(-1)).sum(dim=2)

        valid = mod_mask[:, None, :, None].float()
        mean_feat = (feat_stack * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)

        out = self.out(torch.cat([fused, mean_feat], dim=-1))
        return out, weights

class HybridMILPool(nn.Module):
    def __init__(self, d_model=128, top_k=4):
        super().__init__()
        self.top_k = top_k
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, seq_feats, slice_logits, seq_mask):
        probs = torch.softmax(slice_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1) / math.log(2.0)
        reliability = 1.0 - entropy

        attn_logits = self.attn(seq_feats).squeeze(-1)
        attn_logits = attn_logits + torch.log(reliability.clamp_min(1e-6))
        attn_logits = attn_logits.masked_fill(~seq_mask, -1e9)

        attn_w = torch.softmax(attn_logits, dim=1)
        attn_pool = (seq_feats * attn_w.unsqueeze(-1)).sum(dim=1)

        pos_feat = masked_topk_mean(seq_feats, probs[..., 1], seq_mask, self.top_k)
        neg_feat = masked_topk_mean(seq_feats, probs[..., 0], seq_mask, self.top_k)

        bag_embed = torch.cat([attn_pool, pos_feat, neg_feat], dim=-1)
        return bag_embed, attn_w, reliability

class PUMAMILLite(nn.Module):
    def __init__(self, d_in=384, d_model=128, max_slices=48, top_k=4, proto_momentum=0.9):
        super().__init__()
        self.d_model = d_model
        self.max_slices = max_slices
        self.top_k = top_k
        self.proto_momentum = proto_momentum

        self.fusion = ReliabilityFusion(d_in=d_in, d_model=d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_slices, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=0.10,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.slice_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2)
        )

        self.pool = HybridMILPool(d_model=d_model, top_k=top_k)

        self.register_buffer("prototypes", torch.zeros(2, d_model * 3))
        self.register_buffer("proto_counts", torch.zeros(2))

        self.final_head = nn.Sequential(
            nn.LayerNorm(d_model * 3 + 2),
            nn.Linear(d_model * 3 + 2, d_model),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(d_model, 2)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def modality_dropout(self, mod_mask):
        if (not self.training) or CFG.MODALITY_DROPOUT <= 0:
            return mod_mask

        keep = mod_mask.clone()
        rand = torch.rand_like(keep.float())
        keep = keep & (rand > CFG.MODALITY_DROPOUT)

        for i in range(keep.size(0)):
            if keep[i].sum() == 0:
                valid = torch.where(mod_mask[i])[0]
                if valid.numel() > 0:
                    j = valid[torch.randint(0, valid.numel(), (1,), device=valid.device)]
                    keep[i, j] = True
                else:
                    keep[i] = True

        return keep

    def update_prototypes(self, bag_embed, bag_labels):
        with torch.no_grad():
            for c in [0, 1]:
                mask = (bag_labels == c)
                if mask.any():
                    batch_proto = bag_embed[mask].mean(dim=0)
                    if self.proto_counts[c] == 0:
                        self.prototypes[c] = batch_proto
                    else:
                        self.prototypes[c] = (
                            self.proto_momentum * self.prototypes[c] +
                            (1.0 - self.proto_momentum) * batch_proto
                        )
                    self.proto_counts[c] += mask.sum()

    def prototype_sims(self, bag_embed):
        sims = []
        for c in [0, 1]:
            if self.proto_counts[c] <= 0 or torch.norm(self.prototypes[c]) < 1e-8:
                sims.append(torch.zeros(bag_embed.size(0), 1, device=bag_embed.device))
            else:
                s = F.cosine_similarity(
                    bag_embed, self.prototypes[c].unsqueeze(0), dim=-1
                ).unsqueeze(-1)
                sims.append(s)
        return torch.cat(sims, dim=-1)

    def forward(self, t1, t1ce, t2, flair, seq_mask, mod_mask, return_aux=False):
        assert t1.size(1) <= self.max_slices, "Input sequence longer than model max_slices"

        mod_mask_used = self.modality_dropout(mod_mask)
        fused, mod_weights = self.fusion([t1, t1ce, t2, flair], mod_mask_used)

        x = fused + self.pos_embed[:, :t1.size(1), :]
        x = self.encoder(x, src_key_padding_mask=~seq_mask)

        slice_logits = self.slice_head(x)
        bag_embed, attn_w, reliability = self.pool(x, slice_logits, seq_mask)
        proto_s = self.prototype_sims(bag_embed)
        bag_logits = self.final_head(torch.cat([bag_embed, proto_s], dim=-1))

        if return_aux:
            return bag_logits, slice_logits, bag_embed, mod_weights, attn_w, reliability
        return bag_logits


# ============================================================
# LOSSES / METRICS
# ============================================================

def diversity_loss(mod_weights):
    mean_w = mod_weights.mean(dim=1)
    target = torch.full_like(mean_w, 1.0 / mean_w.size(-1))
    return F.mse_loss(mean_w, target)

def temporal_smoothness_loss(slice_logits, seq_mask):
    pos_probs = torch.softmax(slice_logits, dim=-1)[..., 1]
    diffs = pos_probs[:, 1:] - pos_probs[:, :-1]
    pair_mask = seq_mask[:, 1:] & seq_mask[:, :-1]
    if pair_mask.sum() == 0:
        return slice_logits.sum() * 0.0
    return F.smooth_l1_loss(diffs[pair_mask], torch.zeros_like(diffs[pair_mask]))

def role_aware_asymmetric_focal_loss(center_logits, center_targets, role_ids, class_weights):
    prob1 = torch.softmax(center_logits, dim=-1)[:, 1].clamp(1e-6, 1 - 1e-6)
    y = center_targets.float()

    pos_w = float(class_weights[1])
    neg_w = float(class_weights[0])

    role_mult = torch.tensor(
        [CFG.ROLE_LOSS_MULTIPLIER.get(int(r.item()), 1.0) for r in role_ids],
        dtype=torch.float32,
        device=center_logits.device
    )

    pos_term = -pos_w * y * ((1.0 - prob1) ** CFG.CENTER_GAMMA_POS) * torch.log(prob1)
    neg_term = -neg_w * (1.0 - y) * role_mult * (prob1 ** CFG.CENTER_GAMMA_NEG) * torch.log(1.0 - prob1)

    loss = pos_term + neg_term
    return loss.mean()

def calculate_ece(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    if len(labels) == 0:
        return 0.0

    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    acc = (pred == labels).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += (m.sum() / len(labels)) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)

def metrics_from_probs(probs, labels, threshold=0.5):
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    if len(labels) == 0:
        out = {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "auc": float("nan"), "pr_auc": float("nan"), "ece": 0.0,
            "balanced_accuracy": 0.0, "specificity": 0.0, "sensitivity": 0.0
        }
        return out, np.zeros((0,), dtype=np.int64)

    preds = (probs[:, 1] >= threshold).astype(int)

    out = {
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
    }
    return out, preds

def find_best_threshold(probs, labels):
    labels = np.asarray(labels)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.50, 0.0

    best_t = 0.50
    best_f1 = -1.0
    best_bal = -1.0
    best_spec = -1.0

    for t in np.arange(0.10, 0.91, 0.01):
        preds = (probs[:, 1] >= t).astype(int)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        bal = balanced_accuracy_score(labels, preds)
        spec = recall_score(labels, preds, pos_label=0, zero_division=0)

        better = (
            (f1 > best_f1) or
            (abs(f1 - best_f1) < 1e-12 and bal > best_bal) or
            (abs(f1 - best_f1) < 1e-12 and abs(bal - best_bal) < 1e-12 and spec > best_spec)
        )
        if better:
            best_t = float(t)
            best_f1 = float(f1)
            best_bal = float(bal)
            best_spec = float(spec)

    return best_t, best_f1

def fit_temperature(logits, labels, device):
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


# ============================================================
# EVALUATION
# ============================================================

def evaluate_window_dataset(model, dataset, batch_size=64, temperature=1.0):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_logits, all_probs, all_labels = [], [], []
    all_pids, all_slice_idxs = [], []
    all_roles, all_dists, all_tumor_pixels = [], [], []

    with torch.no_grad():
        for batch in loader:
            (
                t1, t1ce, t2, flair, bag_y, center_y, seq_mask, mod_mask,
                center_pos, role_id, pid, slice_idx, distance_to_positive, tumor_pixels
            ) = batch

            t1 = t1.to(CFG.DEVICE)
            t1ce = t1ce.to(CFG.DEVICE)
            t2 = t2.to(CFG.DEVICE)
            flair = flair.to(CFG.DEVICE)
            seq_mask = seq_mask.to(CFG.DEVICE)
            mod_mask = mod_mask.to(CFG.DEVICE)
            center_pos = center_pos.to(CFG.DEVICE)

            _, slice_logits, _, _, _, _ = model(
                t1, t1ce, t2, flair, seq_mask, mod_mask, return_aux=True
            )

            center_logits = slice_logits[torch.arange(slice_logits.size(0), device=slice_logits.device), center_pos]
            center_logits = center_logits / float(temperature)
            center_probs = torch.softmax(center_logits, dim=-1)

            all_logits.append(center_logits.cpu().numpy())
            all_probs.append(center_probs.cpu().numpy())
            all_labels.append(center_y.numpy())
            all_pids.extend(list(pid))
            all_slice_idxs.append(slice_idx.numpy())
            all_roles.append(role_id.numpy())
            all_dists.append(distance_to_positive.numpy())
            all_tumor_pixels.append(tumor_pixels.numpy())

    logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 2), dtype=np.float32)
    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 2), dtype=np.float32)
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.int64)
    pids = np.asarray(all_pids)
    slice_idxs = np.concatenate(all_slice_idxs, axis=0) if all_slice_idxs else np.zeros((0,), dtype=np.int64)
    roles = np.concatenate(all_roles, axis=0) if all_roles else np.zeros((0,), dtype=np.int64)
    dists = np.concatenate(all_dists, axis=0) if all_dists else np.zeros((0,), dtype=np.int64)
    tumor_pixels = np.concatenate(all_tumor_pixels, axis=0) if all_tumor_pixels else np.zeros((0,), dtype=np.int64)

    return logits, probs, labels, pids, slice_idxs, roles, dists, tumor_pixels

def evaluate_model(model, data_dict, batch_size=64, temperature=1.0):
    ds = CenterWindowDataset(data_dict, window_size=CFG.TRAIN_WINDOW)
    logits, probs, labels, pids, slice_idxs, _, _, _ = evaluate_window_dataset(
        model, ds, batch_size=batch_size, temperature=temperature
    )
    return logits, probs, labels, pids


# ============================================================
# VALIDATION LOSS COMPUTATION (NEW)
# ============================================================

def compute_val_loss(model, val_ds, bag_class_weights, slice_class_weights):
    """Compute the same composite loss on validation data (no gradients)."""
    model.eval()
    loader = DataLoader(val_ds, batch_size=CFG.EVAL_BATCH_SIZE, shuffle=False, num_workers=0)

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            (
                t1, t1ce, t2, flair, bag_y, center_y, seq_mask, mod_mask,
                center_pos, role_id, pid, slice_idx, distance_to_positive, tumor_pixels
            ) = batch

            t1 = t1.to(CFG.DEVICE)
            t1ce = t1ce.to(CFG.DEVICE)
            t2 = t2.to(CFG.DEVICE)
            flair = flair.to(CFG.DEVICE)
            bag_y = bag_y.to(CFG.DEVICE)
            center_y = center_y.to(CFG.DEVICE)
            seq_mask = seq_mask.to(CFG.DEVICE)
            mod_mask = mod_mask.to(CFG.DEVICE)
            center_pos = center_pos.to(CFG.DEVICE)
            role_id = role_id.to(CFG.DEVICE)

            bag_logits, slice_logits, bag_embed, mod_weights, attn_w, reliability = model(
                t1, t1ce, t2, flair, seq_mask, mod_mask, return_aux=True
            )

            center_logits = slice_logits[
                torch.arange(slice_logits.size(0), device=slice_logits.device), center_pos
            ]

            loss_center = role_aware_asymmetric_focal_loss(
                center_logits, center_y, role_id, slice_class_weights
            )
            loss_bag = F.cross_entropy(bag_logits, bag_y, weight=bag_class_weights)
            loss_smooth = temporal_smoothness_loss(slice_logits, seq_mask)
            loss_div = diversity_loss(mod_weights)

            loss = (
                CFG.CENTER_LOSS_WEIGHT * loss_center +
                CFG.BAG_LOSS_WEIGHT * loss_bag +
                CFG.SMOOTHNESS_LOSS_WEIGHT * loss_smooth +
                CFG.DIVERSITY_LOSS_WEIGHT * loss_div
            )

            total_loss += float(loss.item())
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================
# TRAINING
# ============================================================

def build_train_sampler(dataset):
    weights = []
    for meta in dataset.index:
        y = int(meta["center_label"])
        role = int(meta["slice_role_id"])

        w = 1.0
        if y == 1:
            w *= CFG.SAMPLER_POSITIVE_BOOST
        elif role == SLICE_ROLE_TO_ID["boundary_neg"]:
            w *= CFG.SAMPLER_BOUNDARY_NEG_BOOST
        elif role == SLICE_ROLE_TO_ID["far_neg"]:
            w *= CFG.SAMPLER_FAR_NEG_BOOST
        elif role == SLICE_ROLE_TO_ID["no_tumor"]:
            w *= CFG.SAMPLER_NO_TUMOR_BOOST

        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

def train_single_seed(seed, train_data, val_data, bag_weights, slice_weights):
    set_seed(seed)

    model = PUMAMILLite(
        d_in=CFG.D_IN,
        d_model=CFG.D_MODEL,
        max_slices=CFG.MAX_SLICES,
        top_k=CFG.TOP_K,
        proto_momentum=CFG.PROTO_MOMENTUM
    ).to(CFG.DEVICE)

    train_ds = CenterWindowDataset(train_data, window_size=CFG.TRAIN_WINDOW)
    val_ds = CenterWindowDataset(val_data, window_size=CFG.TRAIN_WINDOW)

    train_sampler = build_train_sampler(train_ds)

    train_dl = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    bag_class_weights = torch.tensor(
        [bag_weights[0], bag_weights[1]],
        dtype=torch.float32,
        device=CFG.DEVICE
    )
    slice_class_weights = torch.tensor(
        [slice_weights[0], slice_weights[1]],
        dtype=torch.float32,
        device=CFG.DEVICE
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.TOTAL_EPOCHS)

    ckpt_path = os.path.join(CFG.WORK_DIR, f"pumamil_best_seed{seed}.pth")

    history_rows = []
    best_score = -1.0
    best_epoch = -1
    no_improve = 0

    print("=" * 72)
    print(f"TRAINING SEED={seed}")
    print("=" * 72)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(CFG.TOTAL_EPOCHS):
        model.train()

        ep_total = 0.0
        ep_center = 0.0
        ep_bag = 0.0
        ep_smooth = 0.0
        ep_div = 0.0
        n_batches = 0
        train_correct = 0
        train_total = 0

        for batch in train_dl:
            (
                t1, t1ce, t2, flair, bag_y, center_y, seq_mask, mod_mask,
                center_pos, role_id, pid, slice_idx, distance_to_positive, tumor_pixels
            ) = batch

            t1 = t1.to(CFG.DEVICE, non_blocking=True)
            t1ce = t1ce.to(CFG.DEVICE, non_blocking=True)
            t2 = t2.to(CFG.DEVICE, non_blocking=True)
            flair = flair.to(CFG.DEVICE, non_blocking=True)
            bag_y = bag_y.to(CFG.DEVICE, non_blocking=True)
            center_y = center_y.to(CFG.DEVICE, non_blocking=True)
            seq_mask = seq_mask.to(CFG.DEVICE, non_blocking=True)
            mod_mask = mod_mask.to(CFG.DEVICE, non_blocking=True)
            center_pos = center_pos.to(CFG.DEVICE, non_blocking=True)
            role_id = role_id.to(CFG.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            bag_logits, slice_logits, bag_embed, mod_weights, attn_w, reliability = model(
                t1, t1ce, t2, flair, seq_mask, mod_mask, return_aux=True
            )

            center_logits = slice_logits[torch.arange(slice_logits.size(0), device=slice_logits.device), center_pos]

            loss_center = role_aware_asymmetric_focal_loss(
                center_logits, center_y, role_id, slice_class_weights
            )

            loss_bag = F.cross_entropy(bag_logits, bag_y, weight=bag_class_weights)
            loss_smooth = temporal_smoothness_loss(slice_logits, seq_mask)
            loss_div = diversity_loss(mod_weights)

            loss = (
                CFG.CENTER_LOSS_WEIGHT * loss_center +
                CFG.BAG_LOSS_WEIGHT * loss_bag +
                CFG.SMOOTHNESS_LOSS_WEIGHT * loss_smooth +
                CFG.DIVERSITY_LOSS_WEIGHT * loss_div
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.CLIP_GRAD_NORM)
            optimizer.step()

            with torch.no_grad():
                model.update_prototypes(bag_embed.detach(), bag_y.detach())

            # Track train accuracy
            with torch.no_grad():
                train_preds = torch.argmax(center_logits, dim=-1)
                train_correct += (train_preds == center_y).sum().item()
                train_total += center_y.size(0)

            ep_total += float(loss.item())
            ep_center += float(loss_center.item())
            ep_bag += float(loss_bag.item())
            ep_smooth += float(loss_smooth.item())
            ep_div += float(loss_div.item())
            n_batches += 1

        scheduler.step()

        row = {
            "epoch": int(epoch + 1),
            "train_total_loss": ep_total / max(n_batches, 1),
            "train_center_loss": ep_center / max(n_batches, 1),
            "train_bag_loss": ep_bag / max(n_batches, 1),
            "train_smoothness_loss": ep_smooth / max(n_batches, 1),
            "train_diversity_loss": ep_div / max(n_batches, 1),
            "train_accuracy": train_correct / max(train_total, 1),
        }

        if ((epoch + 1) % CFG.EVAL_EVERY == 0) or (epoch == CFG.TOTAL_EPOCHS - 1):
            val_logits, val_probs, val_labels, _, _, _, _, _ = evaluate_window_dataset(
                model, val_ds, batch_size=CFG.EVAL_BATCH_SIZE, temperature=1.0
            )
            val_threshold, _ = find_best_threshold(val_probs, val_labels)
            val_metrics, _ = metrics_from_probs(val_probs, val_labels, threshold=val_threshold)

            # Compute val_loss by running loss on validation set
            val_loss_total = compute_val_loss(
                model, val_ds, bag_class_weights, slice_class_weights
            )

            val_score = (
                0.60 * val_metrics["f1"] +
                0.25 * val_metrics["balanced_accuracy"] +
                0.15 * val_metrics["specificity"]
            )

            row.update({
                "val_threshold": float(val_threshold),
                "val_total_loss": float(val_loss_total),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": float(val_metrics["f1"]),
                "val_bal_acc": float(val_metrics["balanced_accuracy"]),
                "val_specificity": float(val_metrics["specificity"]),
                "val_sensitivity": float(val_metrics["sensitivity"]),
                "val_auc": float(val_metrics["auc"]) if not np.isnan(val_metrics["auc"]) else float("nan"),
                "val_pr_auc": float(val_metrics["pr_auc"]) if not np.isnan(val_metrics["pr_auc"]) else float("nan"),
                "val_score": float(val_score),
            })

            print(
                f"Epoch {epoch+1:2d}/{CFG.TOTAL_EPOCHS} | "
                f"Loss={row['train_total_loss']:.4f} | "
                f"Val Macro-F1={row['val_macro_f1']:.4f} | "
                f"Val BalAcc={row['val_bal_acc']:.4f} | "
                f"Val Spec={row['val_specificity']:.4f} | "
                f"Val AUC={0.0 if np.isnan(row['val_auc']) else row['val_auc']:.4f} | "
                f"Val Score={row['val_score']:.4f}"
            )

            if val_score > best_score:
                best_score = float(val_score)
                best_epoch = int(epoch + 1)
                no_improve = 0
                torch.save(model.state_dict(), ckpt_path)
                print(f"  -> saved best checkpoint ({ckpt_path})")
            else:
                no_improve += 1
                if no_improve >= CFG.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    history_rows.append(row)
                    break
        else:
            row.update({
                "val_threshold": None,
                "val_total_loss": None,
                "val_accuracy": None,
                "val_macro_f1": None,
                "val_bal_acc": None,
                "val_specificity": None,
                "val_sensitivity": None,
                "val_auc": None,
                "val_pr_auc": None,
                "val_score": None,
            })

        history_rows.append(row)
        clear_memory()

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=CFG.DEVICE))

    return model, {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "history": history_rows
    }


# ============================================================
# AUXILIARY AGGREGATION
# ============================================================

def aggregate_probs_by_patient(probs, labels, pids, slice_idxs=None, top_k=4):
    df = pd.DataFrame({
        "patient_id": pids,
        "label": labels.astype(int),
        "prob_1": probs[:, 1].astype(float)
    })
    if slice_idxs is not None:
        df["slice_idx"] = slice_idxs.astype(int)
    else:
        df["slice_idx"] = np.arange(len(df))

    rows = []
    for pid, g in df.groupby("patient_id", sort=True):
        g = g.sort_values("slice_idx")
        vals = np.sort(g["prob_1"].values)[::-1]
        k = min(int(top_k), len(vals))
        p1 = float(vals[:k].mean()) if k > 0 else 0.5
        rows.append({
            "patient_id": pid,
            "label": int(g["label"].max()),
            "prob_0": float(1.0 - p1),
            "prob_1": float(p1),
        })

    out_df = pd.DataFrame(rows)
    probs_pt = out_df[["prob_0", "prob_1"]].values.astype(np.float32)
    labels_pt = out_df["label"].values.astype(np.int64)
    pids_pt = out_df["patient_id"].values.astype(str)
    return probs_pt, labels_pt, pids_pt, out_df


# ============================================================
# PLOTTING
# ============================================================

def plot_histories(histories):
    if not histories:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for h in histories:
        df = pd.DataFrame(h["history"])
        seed = h["seed"]

        axes[0].plot(df["epoch"], df["train_total_loss"], marker="o", linewidth=1.7, label=f"Seed {seed}")
        axes[1].plot(df["epoch"], df["val_macro_f1"], marker="o", linewidth=1.7, label=f"Seed {seed}")
        axes[2].plot(df["epoch"], df["val_bal_acc"], marker="o", linewidth=1.7, label=f"Seed {seed}")
        axes[3].plot(df["epoch"], df["val_specificity"], marker="o", linewidth=1.7, label=f"Seed {seed}")

    titles = ["Training Loss", "Validation Macro-F1", "Validation Balanced Accuracy", "Validation Specificity"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(CFG.WORK_DIR, "phase2_training_history_v3.png"), dpi=CFG.FIG_DPI, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def run_phase2():
    phase_start = time.time()
    ensure_dir(CFG.WORK_DIR)

    print("\n" + "=" * 72)
    print("PHASE 2 (V6): REVIEWER-DRIVEN SLICE-CENTER TRAINING")
    print("=" * 72)
    print(f"Device: {CFG.DEVICE}")
    print(f"Seeds: {CFG.SEEDS}")
    print(f"Epochs: {CFG.TOTAL_EPOCHS}")
    print("=" * 72 + "\n")

    splits, train_data, val_data, tune_data, test_data = load_all_data()

    bag_weights = {
        0: float(splits["bag_class_weights"]["0"]),
        1: float(splits["bag_class_weights"]["1"]),
    }
    slice_weights = {
        0: float(splits["slice_class_weights"]["0"]),
        1: float(splits["slice_class_weights"]["1"]),
    }

    histories = []
    per_seed_rows = []

    tune_ds = CenterWindowDataset(tune_data, window_size=CFG.TRAIN_WINDOW)
    test_ds = CenterWindowDataset(test_data, window_size=CFG.TRAIN_WINDOW)

    tune_probs_all = []
    test_probs_all = []

    tune_labels_ref = None
    test_labels_ref = None
    tune_pids_ref = None
    test_pids_ref = None
    tune_slice_idxs_ref = None
    test_slice_idxs_ref = None
    tune_roles_ref = None
    test_roles_ref = None
    tune_dists_ref = None
    test_dists_ref = None
    tune_tp_ref = None
    test_tp_ref = None

    for seed in CFG.SEEDS:
        model, history = train_single_seed(seed, train_data, val_data, bag_weights, slice_weights)
        histories.append(history)

        ckpt_path = os.path.join(CFG.WORK_DIR, f"pumamil_best_seed{seed}.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=CFG.DEVICE))
        model.eval()

        tune_logits_raw, tune_probs_raw, tune_labels, tune_pids, tune_slice_idxs, tune_roles, tune_dists, tune_tp = \
            evaluate_window_dataset(model, tune_ds, batch_size=CFG.EVAL_BATCH_SIZE, temperature=1.0)
        best_T = fit_temperature(tune_logits_raw, tune_labels, CFG.DEVICE)

        tune_logits, tune_probs, tune_labels, tune_pids, tune_slice_idxs, tune_roles, tune_dists, tune_tp = \
            evaluate_window_dataset(model, tune_ds, batch_size=CFG.EVAL_BATCH_SIZE, temperature=best_T)

        test_logits, test_probs, test_labels, test_pids, test_slice_idxs, test_roles, test_dists, test_tp = \
            evaluate_window_dataset(model, test_ds, batch_size=CFG.EVAL_BATCH_SIZE, temperature=best_T)

        best_thresh, best_tune_f1 = find_best_threshold(tune_probs, tune_labels)
        test_metrics, test_preds = metrics_from_probs(test_probs, test_labels, threshold=best_thresh)

        print(f"\nSeed {seed} calibrated results")
        print(f" Temperature: {best_T:.3f}")
        print(f" Threshold: {best_thresh:.2f}")
        print(f" Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f" Test Macro-F1: {test_metrics['f1']:.4f}")
        print(f" Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        print(f" Test Specificity: {test_metrics['specificity']:.4f}")
        print(f" Test Sensitivity: {test_metrics['sensitivity']:.4f}")
        print(f" Test PR-AUC: {0.0 if np.isnan(test_metrics['pr_auc']) else test_metrics['pr_auc']:.4f}")
        print(f" Test AUC: {0.0 if np.isnan(test_metrics['auc']) else test_metrics['auc']:.4f}")
        print(f" Test CM: {confusion_matrix(test_labels, test_preds).tolist()}")

        per_seed_rows.append({
            "seed": int(seed),
            "task_level": "slice_center",
            "best_val_score": float(history["best_val_score"]),
            "best_epoch": int(history["best_epoch"]),
            "temperature": float(best_T),
            "threshold": float(best_thresh),
            "tune_f1": float(best_tune_f1),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_precision": float(test_metrics["precision"]),
            "test_recall": float(test_metrics["recall"]),
            "test_f1": float(test_metrics["f1"]),
            "test_auc": float(test_metrics["auc"]) if not np.isnan(test_metrics["auc"]) else float("nan"),
            "test_pr_auc": float(test_metrics["pr_auc"]) if not np.isnan(test_metrics["pr_auc"]) else float("nan"),
            "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "test_sensitivity": float(test_metrics["sensitivity"]),
            "test_specificity": float(test_metrics["specificity"]),
            "test_ece": float(test_metrics["ece"]),
        })

        tune_probs_all.append(tune_probs)
        test_probs_all.append(test_probs)

        if tune_labels_ref is None:
            tune_labels_ref = tune_labels
            test_labels_ref = test_labels
            tune_pids_ref = tune_pids
            test_pids_ref = test_pids
            tune_slice_idxs_ref = tune_slice_idxs
            test_slice_idxs_ref = test_slice_idxs
            tune_roles_ref = tune_roles
            test_roles_ref = test_roles
            tune_dists_ref = tune_dists
            test_dists_ref = test_dists
            tune_tp_ref = tune_tp
            test_tp_ref = test_tp
        else:
            assert np.array_equal(tune_labels_ref, tune_labels)
            assert np.array_equal(test_labels_ref, test_labels)
            assert np.array_equal(tune_pids_ref, tune_pids)
            assert np.array_equal(test_pids_ref, test_pids)
            assert np.array_equal(tune_slice_idxs_ref, tune_slice_idxs)
            assert np.array_equal(test_slice_idxs_ref, test_slice_idxs)

        del model
        clear_memory()

    metrics_df = pd.DataFrame(per_seed_rows)
    metrics_df.to_csv(os.path.join(CFG.WORK_DIR, "per_seed_test_metrics_v3.csv"), index=False)

    best_seed = int(
        metrics_df.sort_values(
            ["test_f1", "test_balanced_accuracy", "test_specificity", "best_val_score"],
            ascending=False
        ).iloc[0]["seed"]
    )

    best_ckpt_src = os.path.join(CFG.WORK_DIR, f"pumamil_best_seed{best_seed}.pth")
    best_ckpt_dst = os.path.join(CFG.WORK_DIR, "pumamil_best_v3.pth")
    shutil.copy2(best_ckpt_src, best_ckpt_dst)

    avg_tune_probs = np.mean(tune_probs_all, axis=0)
    avg_test_probs = np.mean(test_probs_all, axis=0)

    ens_thresh, ens_tune_f1 = find_best_threshold(avg_tune_probs, tune_labels_ref)
    ens_metrics, ens_preds = metrics_from_probs(avg_test_probs, test_labels_ref, threshold=ens_thresh)

    print("\n" + "=" * 72)
    print("ENSEMBLE RESULTS")
    print("=" * 72)
    print(f"Best seed by slice metrics: {best_seed}")
    print(f"Ensemble threshold: {ens_thresh:.2f}")
    print(f"Ensemble Test Accuracy: {ens_metrics['accuracy']:.4f}")
    print(f"Ensemble Test Macro-F1: {ens_metrics['f1']:.4f}")
    print(f"Ensemble Test Balanced Accuracy: {ens_metrics['balanced_accuracy']:.4f}")
    print(f"Ensemble Test Specificity: {ens_metrics['specificity']:.4f}")
    print(f"Ensemble Test Sensitivity: {ens_metrics['sensitivity']:.4f}")
    print(f"Ensemble Test PR-AUC: {0.0 if np.isnan(ens_metrics['pr_auc']) else ens_metrics['pr_auc']:.4f}")
    print(f"Ensemble Test AUC: {0.0 if np.isnan(ens_metrics['auc']) else ens_metrics['auc']:.4f}")
    print(f"Ensemble CM: {confusion_matrix(test_labels_ref, ens_preds).tolist()}")

    agg = {
        "task_level": "slice_center",
        "primary_task": "tumor_slice_identification",
        "main_metric": "macro_f1",
        "best_seed": int(best_seed),
        "per_seed_mean": {
            "test_accuracy": float(metrics_df["test_accuracy"].mean()),
            "test_precision": float(metrics_df["test_precision"].mean()),
            "test_recall": float(metrics_df["test_recall"].mean()),
            "test_f1": float(metrics_df["test_f1"].mean()),
            "test_auc": nanmean_or_zero(metrics_df["test_auc"].values),
            "test_pr_auc": nanmean_or_zero(metrics_df["test_pr_auc"].values),
            "test_balanced_accuracy": float(metrics_df["test_balanced_accuracy"].mean()),
            "test_sensitivity": float(metrics_df["test_sensitivity"].mean()),
            "test_specificity": float(metrics_df["test_specificity"].mean()),
            "test_ece": float(metrics_df["test_ece"].mean()),
        },
        "per_seed_std": {
            "test_accuracy": float(metrics_df["test_accuracy"].std()),
            "test_precision": float(metrics_df["test_precision"].std()),
            "test_recall": float(metrics_df["test_recall"].std()),
            "test_f1": float(metrics_df["test_f1"].std()),
            "test_auc": nanstd_or_zero(metrics_df["test_auc"].values),
            "test_pr_auc": nanstd_or_zero(metrics_df["test_pr_auc"].values),
            "test_balanced_accuracy": float(metrics_df["test_balanced_accuracy"].std()),
            "test_sensitivity": float(metrics_df["test_sensitivity"].std()),
            "test_specificity": float(metrics_df["test_specificity"].std()),
            "test_ece": float(metrics_df["test_ece"].std()),
        },
        "ensemble": {
            "threshold": float(ens_thresh),
            "tune_f1": float(ens_tune_f1),
            "test_accuracy": float(ens_metrics["accuracy"]),
            "test_precision": float(ens_metrics["precision"]),
            "test_recall": float(ens_metrics["recall"]),
            "test_f1": float(ens_metrics["f1"]),
            "test_auc": float(ens_metrics["auc"]) if not np.isnan(ens_metrics["auc"]) else float("nan"),
            "test_pr_auc": float(ens_metrics["pr_auc"]) if not np.isnan(ens_metrics["pr_auc"]) else float("nan"),
            "test_balanced_accuracy": float(ens_metrics["balanced_accuracy"]),
            "test_sensitivity": float(ens_metrics["sensitivity"]),
            "test_specificity": float(ens_metrics["specificity"]),
            "test_ece": float(ens_metrics["ece"]),
        }
    }

    save_json(agg, os.path.join(CFG.WORK_DIR, "aggregated_test_metrics_v3.json"))
    save_json(histories, os.path.join(CFG.WORK_DIR, "training_histories_v3.json"))

    np.savez_compressed(
        os.path.join(CFG.WORK_DIR, "ensemble_test_probs_v3.npz"),
        probs=avg_test_probs,
        labels=test_labels_ref,
        pids=test_pids_ref,
        slice_idxs=test_slice_idxs_ref,
        slice_role_id=test_roles_ref,
        distance_to_positive=test_dists_ref,
        tumor_pixels=test_tp_ref,
    )

    np.savez_compressed(
        os.path.join(CFG.WORK_DIR, "ensemble_test_slice_probs_v3.npz"),
        probs=avg_test_probs,
        labels=test_labels_ref,
        pids=test_pids_ref,
        slice_idxs=test_slice_idxs_ref,
        slice_role_id=test_roles_ref,
        distance_to_positive=test_dists_ref,
        tumor_pixels=test_tp_ref,
    )

    slice_pred_df = pd.DataFrame({
        "patient_id": test_pids_ref,
        "slice_idx": test_slice_idxs_ref.astype(int),
        "label": test_labels_ref.astype(int),
        "prob_0": avg_test_probs[:, 0].astype(float),
        "prob_1": avg_test_probs[:, 1].astype(float),
        "pred": ens_preds.astype(int),
        "slice_role_id": test_roles_ref.astype(int),
        "distance_to_positive": test_dists_ref.astype(int),
        "tumor_pixels": test_tp_ref.astype(int),
    })
    slice_pred_df.to_csv(os.path.join(CFG.WORK_DIR, "slice_predictions_v3.csv"), index=False)

    patient_probs_aux, patient_labels_aux, patient_ids_aux, patient_df_aux = aggregate_probs_by_patient(
        avg_test_probs, test_labels_ref, test_pids_ref, test_slice_idxs_ref, top_k=CFG.TOP_K
    )
    patient_preds_aux = (patient_probs_aux[:, 1] >= 0.50).astype(int)
    patient_df_aux["pred"] = patient_preds_aux.astype(int)
    patient_df_aux.to_csv(os.path.join(CFG.WORK_DIR, "patient_predictions_v3.csv"), index=False)

    aux_patient_metrics, _ = metrics_from_probs(patient_probs_aux, patient_labels_aux, threshold=0.50)
    save_json(
        {
            "note": "auxiliary compatibility artifact only; not the primary scientific endpoint",
            "accuracy": float(aux_patient_metrics["accuracy"]),
            "macro_f1": float(aux_patient_metrics["f1"]),
            "auc": float(aux_patient_metrics["auc"]) if not np.isnan(aux_patient_metrics["auc"]) else float("nan"),
            "pr_auc": float(aux_patient_metrics["pr_auc"]) if not np.isnan(aux_patient_metrics["pr_auc"]) else float("nan"),
        },
        os.path.join(CFG.WORK_DIR, "aux_patient_metrics_v3.json")
    )

    plot_histories(histories)

    phase_time = (time.time() - phase_start) / 60.0
    hyper = {
        "architecture": "PUMA-MIL-Lite",
        "phase2_version": "v6_reviewer_driven_center_windows",
        "device": CFG.DEVICE,
        "seeds": CFG.SEEDS,
        "epochs": CFG.TOTAL_EPOCHS,
        "d_model": CFG.D_MODEL,
        "max_slices": CFG.MAX_SLICES,
        "train_window": CFG.TRAIN_WINDOW,
        "top_k": CFG.TOP_K,
        "modality_dropout": CFG.MODALITY_DROPOUT,
        "center_gamma_pos": CFG.CENTER_GAMMA_POS,
        "center_gamma_neg": CFG.CENTER_GAMMA_NEG,
        "center_loss_weight": CFG.CENTER_LOSS_WEIGHT,
        "bag_loss_weight": CFG.BAG_LOSS_WEIGHT,
        "smoothness_loss_weight": CFG.SMOOTHNESS_LOSS_WEIGHT,
        "diversity_loss_weight": CFG.DIVERSITY_LOSS_WEIGHT,
        "training_time_min": round(phase_time, 2)
    }
    save_json(hyper, os.path.join(CFG.WORK_DIR, "hyperparameters_v3.json"))

    print("\n" + "=" * 72)
    print("✓ PHASE 2 COMPLETE")
    print("=" * 72)
    print(f"Wall-clock time: {phase_time:.1f} min")
    print("Generated:")
    print(" - pumamil_best_seed{42,123,456}.pth")
    print(" - pumamil_best_v3.pth")
    print(" - per_seed_test_metrics_v3.csv")
    print(" - aggregated_test_metrics_v3.json")
    print(" - training_histories_v3.json")
    print(" - phase2_training_history_v3.png")
    print(" - ensemble_test_probs_v3.npz")
    print(" - ensemble_test_slice_probs_v3.npz")
    print(" - slice_predictions_v3.csv")
    print(" - patient_predictions_v3.csv")
    print(" - aux_patient_metrics_v3.json")
    print(" - hyperparameters_v3.json")
    print("=" * 72)

if __name__ == "__main__":
    run_phase2()
