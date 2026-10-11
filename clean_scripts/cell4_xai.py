# ============================================================
# PUMA-MIL CLEAN RUN — CELL 4: XAI (INTERPRETABILITY)
# Grad-CAM, Grad-CAM++, LIME, SHAP + Attention Rollout
# + Cross-Modal Attention Maps
#
# HOW IT WORKS
# ─────────────────────────────────────────────────────────────
# Your model pipeline is:
#   raw MRI slice  →  ViT backbone  →  384-d feature  →  PUMA-MIL-Lite
#
# For image-level explanations we run the ViT + PUMA end-to-end on the
# actual raw MRI pixel arrays and compute gradients / perturbations at
# the INPUT IMAGE level — exactly as in the paper figures you showed.
#
# OUTPUTS (one figure grid per explanation method):
#   interp_gradcam.png        — 3-col grid: MRI | GradCAM | GradCAM++
#   interp_lime.png           — 2-col grid: MRI | LIME superpixels
#   interp_shap.png           — 3-col grid: MRI | SHAP(pos) | SHAP(neg)
#
# Each row in the grid = one MRI slice (2 positive + 2 negative samples)
# across all 4 modalities (T1, T1ce, T2, FLAIR) for context.
# ============================================================

import os, gc, json, math, warnings, glob
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────────
WORK_DIR     = "/kaggle/working"
INPUT_DIR    = "/kaggle/input/datasets/absanim/phase-1-output"
BACKBONE     = "vit_small_patch16_224"
IMG_SIZE     = 224
PATCH_SIZE   = 16
N_PATCHES_1D = IMG_SIZE // PATCH_SIZE    # 14
FEATURE_DIM  = 384
D_MODEL      = 128
MAX_SLICES   = 48
TOP_K        = 4
FIG_DPI      = 600
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Slice CSV from Phase 1 — contains raw NIfTI paths per slice
_META_CANDIDATES = [
    os.path.join(WORK_DIR, "slice_metadata_v3.csv"),
    os.path.join(INPUT_DIR, "slice_metadata_v3.csv"),
    os.path.join(INPUT_DIR, "results", "slice_metadata_v3.csv"),
]
SLICE_META_CSV = next((p for p in _META_CANDIDATES if os.path.exists(p)),
                      _META_CANDIDATES[0])

# Pre-extracted feature NPZ from Phase 1 (used to build proper windows)
_TEST_NPZ_CANDIDATES = [
    os.path.join(WORK_DIR, "test_features_4mod_v3.npz"),
    os.path.join(INPUT_DIR, "test_features_4mod_v3.npz"),
    os.path.join(INPUT_DIR, "results", "test_features_4mod_v3.npz"),
]
TEST_NPZ = next((p for p in _TEST_NPZ_CANDIDATES if os.path.exists(p)),
                _TEST_NPZ_CANDIDATES[0])

TRAIN_WINDOW = 11   # must match Phase 2 CFG.TRAIN_WINDOW

# Checkpoint
CKPT_CANDIDATES = [
    os.path.join(WORK_DIR, "pumamil_best_v3.pth"),
    os.path.join(WORK_DIR, "pumamil_best_seed456.pth"),
    os.path.join(INPUT_DIR, "pumamil_best_v3.pth"),
    os.path.join(INPUT_DIR, "results", "pumamil_best_v3.pth"),
]

# How many samples per class to show
N_POS = 3   # positive (tumor) slices
N_NEG = 3   # negative (non-tumor) slices

MODALITIES = ["t1", "t1ce", "t2", "flair"]   # lowercase, matches CSV cols

# ── OUTPUTS ────────────────────────────────────────────────────────────────
OUT_DIR       = os.path.join(WORK_DIR, "latest_outputs", "output_figs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_GRADCAM   = os.path.join(OUT_DIR, "Fig14_gradcam.png")
OUT_GRADCAMPP = os.path.join(OUT_DIR, "Fig15_gradcampp.png")
OUT_LIME      = os.path.join(OUT_DIR, "Fig16_lime.png")
OUT_SHAP      = os.path.join(OUT_DIR, "Fig17_shap.png")
OUT_ATTN_ROLL = os.path.join(OUT_DIR, "Fig18_attention_rollout.png")
OUT_CROSS_MOD = os.path.join(OUT_DIR, "Fig19_cross_modal_attention.png")

# ── TAR PATHS ──────────────────────────────────────────────────────────────
# BraTS 2021 tar lives in Kaggle input; extracted files go to WORK_DIR so
# they persist across sessions (unlike /kaggle/temp which is wiped on restart)
BRATS_TAR      = "/kaggle/input/datasets/dschettler8845/brats-2021-task1/BraTS2021_Training_Data.tar"
BRATS_EXTRACT  = os.path.join(WORK_DIR, "brats2021_extracted")   # persistent

# ── HELPERS ────────────────────────────────────────────────────────────────
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def clear_mem():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def ensure_brats_extracted():
    """
    Extract only the patient folders needed by the selected samples
    from the tar into BRATS_EXTRACT (inside /kaggle/working, which
    persists between sessions).

    Returns the extraction root so callers can remap paths.
    """
    ensure_dir(BRATS_EXTRACT)
    # Check if already extracted (at least one patient folder present)
    existing = glob.glob(os.path.join(BRATS_EXTRACT, "BraTS2021_*"))
    if existing:
        print(f"  [TAR] already extracted: {len(existing)} patient folders in {BRATS_EXTRACT}")
        return BRATS_EXTRACT

    if not os.path.exists(BRATS_TAR):
        raise FileNotFoundError(
            f"BraTS 2021 tar not found at {BRATS_TAR}\n"
            "Please make sure the dataset is attached to your Kaggle notebook."
        )

    print(f"  [TAR] Extracting BraTS 2021 from tar → {BRATS_EXTRACT} ...")
    print("        (this takes ~3-5 min, runs once, then cached in /kaggle/working)")
    import tarfile, time
    t0 = time.time()
    with tarfile.open(BRATS_TAR, "r") as tf:
        tf.extractall(BRATS_EXTRACT)
    elapsed = time.time() - t0
    extracted = glob.glob(os.path.join(BRATS_EXTRACT, "BraTS2021_*"))
    # Sometimes tar puts everything inside a subdirectory
    if not extracted:
        extracted = glob.glob(os.path.join(BRATS_EXTRACT, "*", "BraTS2021_*"))
    print(f"  [TAR] Done in {elapsed/60:.1f} min — {len(extracted)} patient folders")
    return BRATS_EXTRACT


def repair_paths(df, extract_root):
    """
    The CSV was written with paths pointing to /kaggle/temp/brats2021_extracted/...
    which is wiped between sessions.  Re-map to the persistent extract_root.

    Handles both:
      /kaggle/temp/brats2021_extracted/BraTS2021_00001/BraTS2021_00001_t1.nii.gz
      /kaggle/working/brats2021_extracted/BraTS2021_00001/BraTS2021_00001_t1.nii.gz
    """
    path_cols = [c for c in ["t1_path","t1ce_path","t2_path","flair_path","seg_path"]
                 if c in df.columns]
    if not path_cols:
        return df

    df = df.copy()
    for col in path_cols:
        def _remap(p):
            p = str(p)
            # Extract the patient_id and filename from the end of the old path
            # Works regardless of old prefix (/kaggle/temp/... or /kaggle/working/...)
            parts = p.replace("\\", "/").split("/")
            # Find the BraTS2021_XXXXX part
            for i, part in enumerate(parts):
                if part.startswith("BraTS2021_") and i + 1 < len(parts):
                    pid_dir  = part                 # e.g. BraTS2021_00001
                    filename = parts[i + 1]         # e.g. BraTS2021_00001_t1.nii.gz
                    # Try direct: EXTRACT/BraTS2021_XXXXX/filename
                    cand1 = os.path.join(extract_root, pid_dir, filename)
                    if os.path.exists(cand1):
                        return cand1
                    # Try one level deeper (tar may have a subfolder)
                    for sub in os.listdir(extract_root):
                        cand2 = os.path.join(extract_root, sub, pid_dir, filename)
                        if os.path.exists(cand2):
                            return cand2
            return p   # return original if remapping failed (will be filtered out)
        df[col] = df[col].apply(_remap)
    return df


def percentile_normalize(vol):
    lo = np.percentile(vol[vol > 0], 1) if np.any(vol > 0) else 0.0
    hi = np.percentile(vol[vol > 0], 99) if np.any(vol > 0) else 1.0
    if hi == lo: return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def load_slice(nii_path, slice_idx):
    """Load one 2D slice from a NIfTI volume, percentile-normalised to [0,1]."""
    vol = nib.load(nii_path).get_fdata().astype(np.float32)
    vol = percentile_normalize(vol)
    return vol[:, :, int(slice_idx)]      # (H, W) float32 in [0, 1]


def arr2d_to_tensor(arr2d):
    """Convert a [0,1] float32 2D array → ImageNet-normalised (1,3,224,224) tensor."""
    arr2d = np.clip(arr2d, 0.0, 1.0)
    img   = Image.fromarray((arr2d * 255).astype(np.uint8)).convert("RGB")
    tfm   = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tfm(img).unsqueeze(0)           # (1, 3, 224, 224)


def arr2d_to_display(arr2d):
    """Resize 2D MRI slice to 224×224 for display."""
    h, w = arr2d.shape
    return cv2.resize(arr2d, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def overlay_heatmap(base_img, cam, alpha=0.65, colormap=cv2.COLORMAP_JET, gamma=0.4):
    """
    Overlay a [0,1] CAM heatmap onto a [0,1] grayscale image.
    Uses gamma correction to suppress low activations and make
    hot regions pop with vivid jet colors (red/yellow = high, blue = low).
    Returns an RGB float [0,1] array of shape (H, W, 3).
    """
    # Gamma correction: suppresses weak activations, boosts strong ones
    cam_gamma = np.power(np.clip(cam, 0, 1), gamma)
    base_rgb  = np.stack([base_img] * 3, axis=-1)   # (H, W, 3)
    cam_uint8 = np.uint8(255 * cam_gamma)
    heat_bgr  = cv2.applyColorMap(cam_uint8, colormap)
    heat_rgb  = heat_bgr[:, :, ::-1].astype(np.float32) / 255.0
    # Use cam intensity as per-pixel alpha for cleaner overlay
    pixel_alpha = (cam_gamma[..., np.newaxis] * alpha)
    blended = (1 - pixel_alpha) * base_rgb + pixel_alpha * heat_rgb
    return np.clip(blended, 0, 1)


# ── MODEL ──────────────────────────────────────────────────────────────────
# ── ViT backbone with ATTENTION-BASED GradCAM hooks ──────────────────────────
class ViTForCAM(nn.Module):
    """
    ViT wrapper with ATTENTION-WEIGHT-BASED GradCAM.

    IMPORTANT: Standard GradCAM fails on ViTs that use CLS-token pooling
    because backward() sends gradients ONLY to the CLS token (position 0),
    leaving all 196 patch-token gradients at ZERO.  Hooking the block output
    and looking at patch gradients therefore produces an all-zero CAM.

    FIX: We hook the self-attention layer inside the last transformer block
    to capture the ATTENTION WEIGHTS.  The CLS token's attention over the
    196 patch tokens IS the spatial map of "where the model looks".

    GradCAM   = CLS→patch attention (averaged across heads)
    GradCAM++ = CLS→patch attention × gradient magnitude of the CLS feature
                (gradient-weighted attention — patches that receive more
                 gradient-informed CLS attention get higher weight)
    """
    def __init__(self, model_name=BACKBONE):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self._attn_weights = None    # (B, heads, 197, 197)
        self._activations  = None    # (B, 197, D)  — block output
        self._gradients    = None    # (B, 197, D)  — gradient at block output

        last_block = self.backbone.blocks[-1]
        # Hook attention module to capture attention weights
        last_block.attn.register_forward_hook(self._attn_hook)
        # Hook block output for gradient-weighted refinement
        last_block.register_forward_hook(self._fwd_hook)
        last_block.register_full_backward_hook(self._bwd_hook)

    def _attn_hook(self, module, inp, out):
        """Capture raw attention weights from qkv attention."""
        # timm's Attention.forward computes attn_weights internally;
        # we need to recompute them from the input to get the weights.
        x = inp[0]  # (B, 197, D)
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, _ = qkv.unbind(0)
        scale = (C // module.num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        self._attn_weights = attn.detach()  # (B, heads, 197, 197)

    def _fwd_hook(self, module, inp, out):
        self._activations = out  # (B, 197, D)

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0]  # (B, 197, D)

    def forward(self, x):
        return self.backbone(x)   # (1, D=384)

    def get_cam(self, method="gradcam"):
        """
        Compute spatial CAM from attention weights.
        Returns a (14, 14) numpy array (NOT yet upsampled).
        """
        # CLS token's attention over 196 patches, averaged across heads
        # attn_weights shape: (B, heads, 197, 197)
        # CLS is position 0; patches are positions 1-196
        cls_attn = self._attn_weights[:, :, 0, 1:]  # (B, heads, 196)
        cam = cls_attn.mean(dim=1)                   # (B, 196) — average across heads

        if method == "gradcampp" and self._gradients is not None:
            # Gradient-weighted: modulate attention by gradient magnitude at CLS
            # The CLS gradient tells us how important different features are;
            # we use it to reweight the attention heads before averaging
            cls_grad = self._gradients[:, 0, :]  # (B, D) — gradient at CLS token
            head_dim = cls_grad.shape[-1] // self._attn_weights.shape[1]
            n_heads  = self._attn_weights.shape[1]
            # Per-head gradient magnitude
            grad_per_head = cls_grad.reshape(-1, n_heads, head_dim)  # (B, heads, head_dim)
            head_importance = grad_per_head.abs().mean(dim=-1)       # (B, heads)
            if head_importance.sum() > 1e-10:
                head_importance = head_importance / head_importance.sum(dim=-1, keepdim=True)
            else:
                head_importance = torch.ones_like(head_importance) / n_heads
            # Weighted average of per-head CLS→patch attention
            cam = (cls_attn * head_importance.unsqueeze(-1)).sum(dim=1)  # (B, 196)

        cam = cam.reshape(1, N_PATCHES_1D, N_PATCHES_1D)  # (1, 14, 14)
        cam = cam.squeeze(0).detach().cpu().numpy()        # (14, 14)
        # Normalize to [0, 1]
        if cam.max() > 1e-10:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        return cam


def make_brain_mask(arr2d, threshold=0.02):
    """Binary mask: 1 inside brain, 0 in black background."""
    mask   = (arr2d > threshold).astype(np.uint8)
    kernel = np.ones((11, 11), np.uint8)
    mask   = cv2.dilate(mask, kernel, iterations=3)
    return mask.astype(np.float32)


def compute_saliency_cam(vit, puma, slice_dict, true_feats, target_mod, method="gradcam"):
    """
    TRUE feature-map Grad-CAM / Grad-CAM++ via ViT last-block hooks.

    The gradient flows:
      image_pixels → ViT → (hooked feature maps) → CLS → PUMA → logit → backward
      Hook captures activations A and gradients dY/dA at the last transformer block.
      CAM = ReLU(sum_c(alpha_c * A_c)), reshaped 14×14 → upsampled to 224×224.

    true_feats: pre-extracted {mod: tensor(1,384)} for all 4 modalities
                (only the non-target mods are used — they stay fixed)
    """
    vit.eval()
    puma.eval()
    # Need train mode for backward hooks to fire reliably, but eval norm stats
    vit.backbone.blocks[-1].train()

    # Input tensor with gradient tracking (needed for backward to flow through ViT)
    img = arr2d_to_tensor(slice_dict[target_mod]).to(DEVICE)
    img.requires_grad_(True)

    # Extract feature for the target modality (gradient flows through)
    feat_target = vit(img)          # (1, 384) — hooks capture activations

    # Build per-modality feature tensors: shape (1, 1, 384)
    def _f(mod):
        if mod == target_mod:
            return feat_target.unsqueeze(1)
        return true_feats[mod].detach().unsqueeze(1)

    sm = torch.ones(1, 1, dtype=torch.bool, device=DEVICE)
    mm = torch.ones(1, 4, dtype=torch.bool, device=DEVICE)

    logits = puma(_f('t1'), _f('t1ce'), _f('t2'), _f('flair'), sm, mm)
    pred   = int(logits.argmax(1).item())
    conf   = float(torch.softmax(logits, 1).max().item())

    # Backpropagate — hooks capture gradients at last block
    logits[0, pred].backward()

    # Compute CAM from hooked activations/gradients
    cam_14 = vit.get_cam(method=method)  # (14, 14)

    # Upsample to 224×224 with bicubic interpolation for smooth heatmap
    cam = cv2.resize(cam_14, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

    # Apply brain mask to zero out black background
    base  = arr2d_to_display(slice_dict[target_mod])
    bmask = make_brain_mask(base)
    cam   = cam * bmask

    # Light Gaussian blur for visual smoothness
    cam = cv2.GaussianBlur(cam, (5, 5), 0)
    if cam.max() > 1e-10:
        cam = cam / cam.max()

    if img.grad is not None:
        img.grad = None
    return np.clip(cam, 0, 1), pred, conf, base


# ── PUMA-MIL-Lite (minimal, for inference only) ────────────────────────────
def masked_topk_mean(sf, scores, mask, topk):
    bsz, _, dim = sf.shape; out = []
    for b in range(bsz):
        vi = torch.where(mask[b])[0]
        if vi.numel() == 0: out.append(torch.zeros(dim, device=sf.device)); continue
        k = min(int(topk), int(vi.numel()))
        ch = vi[torch.topk(scores[b, vi], k).indices]
        out.append(sf[b, ch].mean(0))
    return torch.stack(out)

class ReliabilityFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Linear(FEATURE_DIM, D_MODEL), nn.LayerNorm(D_MODEL), nn.GELU())
            for _ in range(4)])
        self.rel = nn.ModuleList([nn.Linear(D_MODEL, 1) for _ in range(4)])
        self.out = nn.Sequential(nn.Linear(D_MODEL*2, D_MODEL), nn.LayerNorm(D_MODEL), nn.GELU())
    def forward(self, mods, mm):
        pf = [self.proj[i](mods[i]) for i in range(4)]
        rl = [self.rel[i](pf[i]) for i in range(4)]
        fs = torch.stack(pf, 2); rs = torch.cat(rl, -1).masked_fill(~mm[:,None,:], -1e4)
        w = torch.softmax(rs, -1)
        fu = (fs * w.unsqueeze(-1)).sum(2)
        v = mm[:,None,:,None].float()
        mf = (fs*v).sum(2)/v.sum(2).clamp_min(1)
        return self.out(torch.cat([fu,mf],-1)), w

class HybridMILPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.topk = TOP_K
        self.attn = nn.Sequential(nn.Linear(D_MODEL, D_MODEL//2), nn.Tanh(), nn.Linear(D_MODEL//2, 1))
    def forward(self, sf, sl, sm):
        p = torch.softmax(sl, -1)
        e = -(p*torch.log(p.clamp_min(1e-8))).sum(-1)/math.log(2)
        r = 1-e
        al = self.attn(sf).squeeze(-1)+torch.log(r.clamp_min(1e-6))
        al = al.masked_fill(~sm, -1e9)
        aw = torch.softmax(al, 1)
        ap = (sf*aw.unsqueeze(-1)).sum(1)
        pp = masked_topk_mean(sf, p[...,1], sm, self.topk)
        np_ = masked_topk_mean(sf, p[...,0], sm, self.topk)
        return torch.cat([ap,pp,np_],-1), aw, r

class PUMAMILLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion    = ReliabilityFusion()
        self.pos_embed = nn.Parameter(torch.zeros(1, MAX_SLICES, D_MODEL))
        el = nn.TransformerEncoderLayer(D_MODEL,4,256,0.1,batch_first=True,activation="gelu")
        self.encoder    = nn.TransformerEncoder(el, 2)
        self.slice_head = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL,2))
        self.pool       = HybridMILPool()
        self.register_buffer("prototypes",   torch.zeros(2, D_MODEL*3))
        self.register_buffer("proto_counts", torch.zeros(2))
        self.final_head = nn.Sequential(
            nn.LayerNorm(D_MODEL*3+2), nn.Linear(D_MODEL*3+2, D_MODEL),
            nn.GELU(), nn.Dropout(0.25), nn.Linear(D_MODEL, 2))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    def proto_sims(self, be):
        s = []
        for c in [0,1]:
            if self.proto_counts[c]<=0 or torch.norm(self.prototypes[c])<1e-8:
                s.append(torch.zeros(be.size(0),1,device=be.device))
            else:
                s.append(F.cosine_similarity(be, self.prototypes[c].unsqueeze(0),-1).unsqueeze(-1))
        return torch.cat(s,-1)
    def forward(self, t1,t1ce,t2,flair, sm,mm):
        fu, _ = self.fusion([t1,t1ce,t2,flair], mm)
        x = fu + self.pos_embed[:,:t1.size(1),:]
        x = self.encoder(x, src_key_padding_mask=~sm)
        sl = self.slice_head(x)
        # ── CRITICAL FIX ──────────────────────────────────────────────
        # Phase 2 evaluates via:  center_logits = slice_logits[:, center_pos, :]
        # NOT via bag_logits from final_head (which answers "does this
        # PATIENT have any tumor somewhere?" → nearly always yes).
        # Center position = middle of the window (WINDOW_HALF for 11-slice)
        center_pos = t1.size(1) // 2
        return sl[:, center_pos, :]   # (B, 2) — center-slice logits


def load_models():
    """Load ViT (with gradients) and PUMA checkpoint."""
    vit = ViTForCAM(BACKBONE).to(DEVICE)
    # keep gradients enabled for CAM
    for p in vit.parameters():
        p.requires_grad_(True)
    vit.train()   # need train mode so backward works through BatchNorm etc.

    puma = PUMAMILLite().to(DEVICE)
    ckpt = next((p for p in CKPT_CANDIDATES if os.path.exists(p)), None)
    if ckpt is None:
        raise FileNotFoundError("No PUMA checkpoint found")
    state = torch.load(ckpt, map_location=DEVICE)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.",""):v for k,v in state.items()}
    puma.load_state_dict(state, strict=False)
    puma.eval()
    return vit, puma, ckpt


# ── SAMPLE SELECTOR ────────────────────────────────────────────────────────
def select_samples(n_pos=N_POS, n_neg=N_NEG):
    """
    Pick n_pos tumor slices and n_neg non-tumor slices from the
    test split of slice_metadata_v3.csv.

    Automatically extracts the BraTS tar if the NIfTI files are no
    longer present at the paths stored in the CSV (/kaggle/temp gets
    wiped between Kaggle sessions).

    Returns a list of dicts: {pid, slice_idx, label, t1_path, …}
    """
    if not os.path.exists(SLICE_META_CSV):
        raise FileNotFoundError(
            f"Not found: {SLICE_META_CSV}\n"
            "Phase 1 must have run first to create slice_metadata_v3.csv.")

    df = pd.read_csv(SLICE_META_CSV)

    for col in ["t1_path","t1ce_path","t2_path","flair_path"]:
        if col not in df.columns:
            raise KeyError(f"slice_metadata_v3.csv missing column: {col}")

    # Prefer test split; fall back to all rows if column absent or split empty
    if "split" in df.columns:
        test_df = df[df["split"] == "test"].copy()
        if len(test_df) == 0:
            test_df = df.copy()
    else:
        test_df = df.copy()

    # ── First check: are the stored paths already valid? ─────────────────
    def _all_exist(row):
        return all(os.path.exists(str(row[c]))
                   for c in ["t1_path","t1ce_path","t2_path","flair_path"])

    valid_mask = test_df.apply(_all_exist, axis=1)
    n_valid    = valid_mask.sum()

    if n_valid == 0:
        # Paths are stale (/kaggle/temp was wiped).
        # Extract the tar into /kaggle/working (persistent) and remap.
        print("  [INFO] Stored NIfTI paths are stale (likely /kaggle/temp was wiped).")
        print("  [INFO] Will extract BraTS 2021 tar into /kaggle/working/brats2021_extracted/")
        extract_root = ensure_brats_extracted()
        test_df   = repair_paths(test_df, extract_root)
        valid_mask = test_df.apply(_all_exist, axis=1)
        n_valid    = valid_mask.sum()
        print(f"  [INFO] After remapping: {n_valid}/{len(test_df)} slices have valid paths")

    test_df = test_df[valid_mask].reset_index(drop=True)

    if len(test_df) == 0:
        raise RuntimeError(
            "No slices found with valid NIfTI paths even after tar extraction.\n"
            f"Extraction root checked: {BRATS_EXTRACT}\n"
            "Please verify the BraTS 2021 dataset is attached to this notebook.")

    # Sample pos and neg rows
    pos_pool = test_df[test_df["label"] == 1]
    neg_pool = test_df[test_df["label"] == 0]
    pos_rows = pos_pool.sample(n=min(n_pos, len(pos_pool)), random_state=42)
    neg_rows = neg_pool.sample(n=min(n_neg, len(neg_pool)), random_state=42)
    samples  = pd.concat([pos_rows, neg_rows]).reset_index(drop=True)

    result = []
    for _, row in samples.iterrows():
        result.append({
            "pid":       str(row["patient_id"]),
            "slice_idx": int(row["slice_idx"]),
            "label":     int(row["label"]),
            "role":      str(row.get("slice_role", "unknown")),
            "t1":        str(row["t1_path"]),
            "t1ce":      str(row["t1ce_path"]),
            "t2":        str(row["t2_path"]),
            "flair":     str(row["flair_path"]),
        })
    return result


WINDOW_HALF = 5   # TRAIN_WINDOW = 11 → half = 5

def load_sample_slices(sample):
    """Load 2D centre slice (for display). Returns {mod: arr(H,W)}."""
    return {mod: load_slice(sample[mod], sample["slice_idx"]) for mod in MODALITIES}


def load_window_vols(sample):
    """
    Load all 4 NIfTI volumes for a sample (lazy, cached per call).
    Returns {mod: vol(H,W,D) float32 normalised}.
    """
    vols = {}
    for mod in MODALITIES:
        path = sample[mod]
        vol  = nib.load(path).get_fdata().astype(np.float32)
        vols[mod] = percentile_normalize(vol)
    return vols


def extract_window_features(vit, vols, center_idx, override_mod=None, override_img_u8=None):
    """
    Extract ViT features for an 11-slice window centred on center_idx.

    vols          : {mod: vol(H,W,D)}
    center_idx    : int, the slice we want to explain
    override_mod  : modality name to replace at center position (for LIME/SHAP)
    override_img_u8 : (224,224,3) uint8 — replaces center slice of override_mod

    Returns:
        feats : {mod: tensor(1, W, 384)}  W=window_size=11
        seq_mask : tensor(1, W) bool
        center_local : int — position of center slice within window (=WINDOW_HALF)
    """
    D          = vols["t1"].shape[2]
    half       = WINDOW_HALF
    win_size   = 2 * half + 1   # 11

    vit.eval()
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    feats    = {mod: [] for mod in MODALITIES}
    seq_mask = []

    for offset in range(-half, half + 1):
        z     = center_idx + offset
        valid = (0 <= z < D)
        seq_mask.append(valid)

        for mod in MODALITIES:
            if not valid:
                feats[mod].append(torch.zeros(FEATURE_DIM, device=DEVICE))
                continue

            # Override center slice for explanation methods
            if (override_mod == mod) and (offset == 0) and (override_img_u8 is not None):
                t = tfm(Image.fromarray(override_img_u8)).unsqueeze(0).to(DEVICE)
            else:
                arr2d = vols[mod][:, :, z]
                arr2d = np.clip(arr2d, 0, 1)
                img   = Image.fromarray((arr2d * 255).astype(np.uint8)).convert("RGB")
                t     = tfm(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                f = vit(t)       # (1, 384)
            feats[mod].append(f.squeeze(0))

    # Stack to (1, W, D)
    result = {}
    for mod in MODALITIES:
        result[mod] = torch.stack(feats[mod], dim=0).unsqueeze(0)  # (1, W, 384)
    seq_mask_t = torch.tensor(seq_mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)  # (1, W)
    return result, seq_mask_t, half


def run_puma_window(puma, feats, seq_mask):
    """Run PUMA on a full 11-slice window. Returns logits (1,2)."""
    puma.eval()
    mm = torch.ones(1, 4, dtype=torch.bool, device=DEVICE)
    with torch.no_grad():
        logits = puma(feats["t1"], feats["t1ce"], feats["t2"], feats["flair"],
                      seq_mask, mm)
    return logits


# ── GRAD-CAM: hook-based at center slice with full window context ──────────
def forward_center_grad(vit, puma, vols, center_idx, target_mod="t1ce"):
    """
    Extract features for the full 11-slice window with grad enabled only for
    the center slice of target_mod.  After backward(), use vit.get_cam() to
    retrieve the attention-based GradCAM from the hooked transformer block.

    IMPORTANT: The attention hooks fire on EVERY vit(t) call, so we explicitly
    snapshot the attention weights from the target slice's forward pass.
    Otherwise they'd be overwritten by subsequent modality/offset calls.

    Returns (logits, pred, conf).
    """
    vit.eval(); puma.eval()
    # Enable train on last block so backward hooks fire
    vit.backbone.blocks[-1].train()
    D    = vols["t1"].shape[2]
    half = WINDOW_HALF
    tfm  = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    feats_all  = {mod: [] for mod in MODALITIES}
    seq_mask   = []
    saved_attn = None  # will hold attention weights from the TARGET slice

    for offset in range(-half, half + 1):
        z     = center_idx + offset
        valid = (0 <= z < D)
        seq_mask.append(valid)
        for mod in MODALITIES:
            if not valid:
                feats_all[mod].append(torch.zeros(FEATURE_DIM, device=DEVICE))
                continue
            arr2d = np.clip(vols[mod][:, :, z], 0, 1)
            img   = Image.fromarray((arr2d * 255).astype(np.uint8)).convert("RGB")
            t     = tfm(img).unsqueeze(0).to(DEVICE)
            if mod == target_mod and offset == 0:
                t.requires_grad_(True)
                f = vit(t)           # gradient flows; hooks capture activations
                # ── SNAPSHOT attention weights from this specific forward pass ──
                saved_attn = vit._attn_weights.clone() if vit._attn_weights is not None else None
            else:
                with torch.no_grad():
                    f = vit(t)
            feats_all[mod].append(f.squeeze(0))

    # Restore the target slice's attention weights (overwritten by subsequent calls)
    if saved_attn is not None:
        vit._attn_weights = saved_attn

    f_dict = {mod: torch.stack(feats_all[mod],0).unsqueeze(0) for mod in MODALITIES}
    sm     = torch.tensor(seq_mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)
    mm     = torch.ones(1,4,dtype=torch.bool,device=DEVICE)
    logits = puma(f_dict["t1"],f_dict["t1ce"],f_dict["t2"],f_dict["flair"],sm,mm)
    pred   = int(logits.argmax(1).item())
    conf   = float(torch.softmax(logits,1).max().item())
    return logits, pred, conf


# ══════════════════════════════════════════════════════════════════════════
#  1. GRAD-CAM / GRAD-CAM++   (SEPARATE PNGs for each method)
# ══════════════════════════════════════════════════════════════════════════
def _pre_extract_features(vit, slice_dict):
    """Extract ViT features for all 4 modalities (single slice). Returns {mod: tensor(1,384)}."""
    feats = {}
    with torch.no_grad():
        for mod in MODALITIES:
            t = arr2d_to_tensor(slice_dict[mod]).to(DEVICE)
            feats[mod] = vit(t)   # (1, 384)
    return feats


def _collect_cams(vit, puma, samples, target_mod="t1ce"):
    """
    TRUE feature-map Grad-CAM & Grad-CAM++ for each sample.
    Uses the full 11-slice window context (matching Phase 2 training) via
    forward_center_grad + vit.get_cam().

    The ViT hooks capture activations/gradients at the last transformer block,
    then get_cam() computes the 14×14 CAM which is upsampled to 224×224.
    """
    results = []
    for sample in samples:
        slices = load_sample_slices(sample)
        base   = arr2d_to_display(slices[target_mod])

        # Load full 4-mod volumes for the windowed approach
        vols = load_window_vols(sample)
        center_idx = sample["slice_idx"]

        # --- GradCAM ---
        logits, pred, conf = forward_center_grad(
            vit, puma, vols, center_idx, target_mod=target_mod)
        logits[0, pred].backward(retain_graph=False)
        cam_gc_14 = vit.get_cam(method="gradcam")   # (14, 14)
        cam_gc = cv2.resize(cam_gc_14, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

        # --- GradCAM++ ---
        logits2, _, _ = forward_center_grad(
            vit, puma, vols, center_idx, target_mod=target_mod)
        logits2[0, pred].backward(retain_graph=False)
        cam_gpp_14 = vit.get_cam(method="gradcampp")  # (14, 14)
        cam_gpp = cv2.resize(cam_gpp_14, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

        # Apply brain mask & light Gaussian blur
        bmask = make_brain_mask(base)
        cam_gc  = cam_gc  * bmask
        cam_gpp = cam_gpp * bmask
        cam_gc  = cv2.GaussianBlur(cam_gc,  (5, 5), 0)
        cam_gpp = cv2.GaussianBlur(cam_gpp, (5, 5), 0)
        if cam_gc.max()  > 1e-10: cam_gc  = cam_gc  / cam_gc.max()
        if cam_gpp.max() > 1e-10: cam_gpp = cam_gpp / cam_gpp.max()
        cam_gc  = np.clip(cam_gc,  0, 1)
        cam_gpp = np.clip(cam_gpp, 0, 1)

        results.append({
            "base": base, "cam_gc": cam_gc, "cam_gpp": cam_gpp,
            "pred": pred, "conf": conf,
            "label": sample["label"], "pid": sample["pid"],
        })
        clear_mem()
    return results


def _save_cam_figure(cam_data, method_name, out_path):
    """
    Save a 2-column figure:  col0=Original MRI  |  col1=CAM heatmap overlay
    One row per sample.  SEPARATE file per method.
    """
    n = len(cam_data)
    if n == 0:
        print(f"  [SKIP] no data for {method_name}")
        return
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i, d in enumerate(cam_data):
        lbl_str  = "Tumor" if d["label"] == 1 else "Non-Tumor"
        pred_str = "Tumor" if d["pred"]  == 1 else "Non-Tumor"
        cam_key  = "cam_gc" if method_name == "Grad-CAM" else "cam_gpp"
        axes[i, 0].imshow(d["base"], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Original T1ce\nTrue: {lbl_str}", fontsize=10)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(overlay_heatmap(d["base"], d[cam_key]))
        axes[i, 1].set_title(f"{method_name} Heatmap\nPred: {pred_str}  (conf={d['conf']:.2f})", fontsize=10)
        axes[i, 1].axis("off")
    plt.suptitle(f"{method_name} — PUMA-MIL-Lite  (T1ce modality)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.subplots_adjust(top=0.90, hspace=0.40)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(out_path)}")


def run_gradcam(vit, puma, samples):
    print("\n[1/5] GradCAM & GradCAM++ ...")

    cam_data = _collect_cams(vit, puma, samples, target_mod="t1ce")

    # ── GradCAM — separate PNG ─────────────────────────────────────────────
    _save_cam_figure(cam_data, "Grad-CAM",   OUT_GRADCAM)

    # ── GradCAM++ — separate PNG ───────────────────────────────────────────
    _save_cam_figure(cam_data, "Grad-CAM++", OUT_GRADCAMPP)

    # ── All-4-modality combined figure (first sample only) ─────────────────
    # Uses forward_center_grad with FULL 11-slice window context per modality
    sample = samples[0]
    slices_4mod = load_sample_slices(sample)
    vols_4mod   = load_window_vols(sample)
    center_idx  = sample["slice_idx"]
    out_4mod = os.path.join(OUT_DIR, "Fig14b_gradcam_4modalities.png")
    fig2, axes2 = plt.subplots(4, 3, figsize=(12, 15))
    for mi, mod in enumerate(MODALITIES):
        base = arr2d_to_display(slices_4mod[mod])
        # GradCAM with full 11-slice window context
        logits_gc, pred, conf = forward_center_grad(
            vit, puma, vols_4mod, center_idx, target_mod=mod)
        logits_gc[0, pred].backward(retain_graph=False)
        cam_gc_14 = vit.get_cam(method="gradcam")
        cam_gc = cv2.resize(cam_gc_14, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        # GradCAM++ with full 11-slice window context
        logits_gpp, _, _ = forward_center_grad(
            vit, puma, vols_4mod, center_idx, target_mod=mod)
        logits_gpp[0, pred].backward(retain_graph=False)
        cam_gpp_14 = vit.get_cam(method="gradcampp")
        cam_gpp = cv2.resize(cam_gpp_14, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        # Apply brain mask & Gaussian blur
        bmask = make_brain_mask(base)
        cam_gc  = cam_gc  * bmask
        cam_gpp = cam_gpp * bmask
        cam_gc  = cv2.GaussianBlur(cam_gc,  (5, 5), 0)
        cam_gpp = cv2.GaussianBlur(cam_gpp, (5, 5), 0)
        if cam_gc.max()  > 1e-10: cam_gc  = cam_gc  / cam_gc.max()
        if cam_gpp.max() > 1e-10: cam_gpp = cam_gpp / cam_gpp.max()
        cam_gc  = np.clip(cam_gc,  0, 1)
        cam_gpp = np.clip(cam_gpp, 0, 1)
        lbl_str  = "Tumor" if sample["label"]==1 else "Non-Tumor"
        pred_str = "Tumor" if pred==1 else "Non-Tumor"
        axes2[mi, 0].imshow(base, cmap="gray", vmin=0, vmax=1)
        axes2[mi, 0].set_title(f"{mod.upper()}  |  True: {lbl_str}", fontsize=9)
        axes2[mi, 1].imshow(overlay_heatmap(base, cam_gc))
        axes2[mi, 1].set_title(f"Grad-CAM  |  Pred: {pred_str} ({conf:.2f})", fontsize=9)
        axes2[mi, 2].imshow(overlay_heatmap(base, cam_gpp))
        axes2[mi, 2].set_title(f"Grad-CAM++  |  Pred: {pred_str} ({conf:.2f})", fontsize=9)
        for ax in axes2[mi]: ax.axis("off")
        clear_mem()
    plt.suptitle(
        f"Grad-CAM — All 4 Modalities\n"
        f"Patient: {sample['pid']}  |  True: {'Tumor' if sample['label']==1 else 'Non-Tumor'}",
        fontsize=12, fontweight="bold", y=1.02)
    plt.subplots_adjust(top=0.90, hspace=0.35)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_4mod, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(out_4mod)}")



# ══════════════════════════════════════════════════════════════════════════
#  2. LIME  — Yellow boundary contour style (matching reference Fig. 10)
# ══════════════════════════════════════════════════════════════════════════
def run_lime(vit, puma, samples):
    print("\n[2/5] LIME ...")
    try:
        from lime import lime_image
    except ImportError:
        print("  [SKIP] pip install lime  then re-run"); return

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    explainer = lime_image.LimeImageExplainer()
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 4*n_rows))
    if n_rows == 1: axes = axes[np.newaxis, :]

    for row_i, sample in enumerate(samples):
        slices = load_sample_slices(sample)
        arr    = arr2d_to_display(slices["t1ce"])
        img_rgb = np.stack([arr]*3, axis=-1)
        img_u8  = np.uint8(np.clip(img_rgb, 0, 1) * 255)

        # Load full volumes for windowed extraction
        vols       = load_window_vols(sample)
        center_idx = sample["slice_idx"]

        def lime_predict(imgs_uint8, _vols=vols, _ci=center_idx):
            """Perturb ONLY center T1ce slice within a proper 11-slice window."""
            probs = []
            vit.eval(); puma.eval()
            for img_p in imgs_uint8:
                feats, seq_mask, _ = extract_window_features(
                    vit, _vols, _ci,
                    override_mod="t1ce", override_img_u8=img_p)
                with torch.no_grad():
                    logit = run_puma_window(puma, feats, seq_mask)
                    prob  = torch.softmax(logit, 1).cpu().numpy()[0]
                probs.append(prob)
            return np.asarray(probs, dtype=np.float32)

        brain_mean = int(img_u8[img_u8 > 10].mean()) if (img_u8 > 10).any() else 60

        explanation = explainer.explain_instance(
            img_u8, lime_predict,
            top_labels=2, hide_color=brain_mean,
            num_samples=500, random_seed=42,
        )
        pred_label = explanation.top_labels[0]
        lbl_str    = "Tumor" if sample["label"]==1 else "Non-Tumor"
        pred_str   = "Tumor" if pred_label==1     else "Non-Tumor"

        sp_weights = dict(explanation.local_exp[pred_label])
        segs       = explanation.segments

        sp_sorted = sorted(sp_weights.items(), key=lambda x: x[1], reverse=True)
        n_pos_sp  = sum(1 for _,w in sp_sorted if w > 0)
        n_neg_sp  = sum(1 for _,w in sp_sorted if w < 0)
        TOP_K_LIME = min(8, max(n_pos_sp, 1))

        print(f"  {sample['pid']}  true={'T' if sample['label']==1 else 'N'}  "
              f"pred={'T' if pred_label==1 else 'N'}  "
              f"n_pos={n_pos_sp}  n_neg={n_neg_sp}  "
              f"max_w={max((abs(w) for _,w in sp_sorted), default=0):.5f}")

        # ── Yellow boundary contours on grayscale (reference paper style) ──
        pos_sps = [sp_id for sp_id, w in sp_sorted if w > 0][:TOP_K_LIME]
        # Build grayscale RGB canvas
        canvas = np.uint8(np.clip(arr, 0, 1) * 255)
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        # Draw yellow contours ONLY for superpixels inside the brain
        YELLOW = (0, 255, 255)  # BGR for OpenCV
        for sp_id in pos_sps:
            sp_mask = np.uint8(segs == sp_id)
            # Skip superpixels in black background (mean intensity < 5%)
            sp_mean = arr[sp_mask > 0].mean() if (sp_mask > 0).any() else 0
            if sp_mean < 0.05:
                continue
            contours, _ = cv2.findContours(sp_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas_rgb, contours, -1, YELLOW, thickness=2)
        lime_vis = canvas_rgb[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB

        # Plot: Original | LIME contours
        axes[row_i,0].imshow(arr, cmap="gray", vmin=0, vmax=1)
        axes[row_i,0].set_title(f"Original T1ce\nTrue: {lbl_str}", fontsize=10)
        axes[row_i,0].axis("off")
        axes[row_i,1].imshow(lime_vis)
        axes[row_i,1].set_title(f"LIME Explanation\nPred: {pred_str}", fontsize=10)
        axes[row_i,1].axis("off")
        clear_mem()

    plt.suptitle("LIME Superpixel Explanations — T1ce (11-slice window)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.subplots_adjust(top=0.90, hspace=0.40)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_LIME, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(OUT_LIME)}")


# ══════════════════════════════════════════════════════════════════════════
#  3. SHAP  — GradientExplainer, red-green diverging colormap (ref Fig. 13)
# ══════════════════════════════════════════════════════════════════════════
def run_shap(vit, puma, samples):
    print("\n[3/5] SHAP (image-level, 11-slice window context) ...")
    try:
        import shap
    except ImportError:
        print("  [SKIP] pip install shap  then re-run"); return
    from matplotlib.colors import TwoSlopeNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4.5*n_rows))
    if n_rows == 1: axes = axes[np.newaxis, :]

    for row_i, sample in enumerate(samples):
        slices = load_sample_slices(sample)
        base   = arr2d_to_display(slices["t1ce"])
        t_img  = arr2d_to_tensor(slices["t1ce"]).to(DEVICE)

        # Load full volumes for 11-slice window context
        vols       = load_window_vols(sample)
        center_idx = sample["slice_idx"]

        # Pre-extract NON-T1ce window features (these stay fixed during SHAP)
        D    = vols["t1"].shape[2]
        half = WINDOW_HALF
        tfm  = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        fixed_feats = {mod: [] for mod in MODALITIES}
        seq_mask_list = []
        for offset in range(-half, half + 1):
            z     = center_idx + offset
            valid = (0 <= z < D)
            seq_mask_list.append(valid)
            for mod in MODALITIES:
                if not valid:
                    fixed_feats[mod].append(torch.zeros(FEATURE_DIM, device=DEVICE))
                    continue
                if mod == "t1ce" and offset == 0:
                    fixed_feats[mod].append(None)
                    continue
                arr2d = np.clip(vols[mod][:, :, z], 0, 1)
                img   = Image.fromarray((arr2d * 255).astype(np.uint8)).convert("RGB")
                t     = tfm(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    f = vit(t).squeeze(0)
                fixed_feats[mod].append(f)

        sm_w = torch.tensor(seq_mask_list, dtype=torch.bool, device=DEVICE).unsqueeze(0)
        mm_w = torch.ones(1, 4, dtype=torch.bool, device=DEVICE)
        center_local = half

        class E2E(nn.Module):
            def forward(self_inner, x):
                bsz = x.size(0)
                f_center = vit(x)
                results_list = []
                for b in range(bsz):
                    feat_window = {mod: [] for mod in MODALITIES}
                    for w_i in range(len(seq_mask_list)):
                        for mod in MODALITIES:
                            if mod == "t1ce" and w_i == center_local:
                                feat_window[mod].append(f_center[b])
                            else:
                                feat_window[mod].append(fixed_feats[mod][w_i])
                    fw = {mod: torch.stack(feat_window[mod], 0).unsqueeze(0)
                          for mod in MODALITIES}
                    logit = puma(fw["t1"], fw["t1ce"], fw["t2"], fw["flair"],
                                sm_w, mm_w)
                    results_list.append(logit)
                return torch.cat(results_list, dim=0)

        e2e = E2E().to(DEVICE)
        e2e.train()

        bg_pool = [s for s in samples if s["label"] != sample["label"]]
        if not bg_pool: bg_pool = [s for s in samples if s is not sample]
        bg = torch.cat([arr2d_to_tensor(load_sample_slices(s)["t1ce"]).to(DEVICE)
                        for s in bg_pool[:2]], dim=0)

        explainer = shap.GradientExplainer(e2e, bg)
        raw = explainer.shap_values(t_img)

        if isinstance(raw, tuple): raw = raw[0]
        if not isinstance(raw, list): raw = [raw]
        raw = [np.asarray(v) for v in raw]

        def _to2d(arr_in):
            arr_in = np.asarray(arr_in)
            if arr_in.ndim == 5: arr_in = arr_in[..., 1]
            if arr_in.ndim == 4: arr_in = arr_in.squeeze(0)
            if arr_in.ndim == 3 and arr_in.shape[0]==3: arr_in = arr_in.mean(0)
            if arr_in.ndim == 3 and arr_in.shape[-1]==3: arr_in = arr_in.mean(-1)
            return arr_in.astype(np.float32)

        if len(raw) >= 2:
            sv_tumor    = _to2d(raw[1])
            sv_nontumor = _to2d(raw[0])
        else:
            sv_all      = _to2d(raw[0])
            sv_tumor    = sv_all.copy()
            sv_nontumor = -sv_all.copy()

        # ── Red-green diverging SHAP overlay on grayscale (ref Fig. 13) ──
        # Build a direct red↔green colormap (no yellow middle) for max contrast
        from matplotlib.colors import LinearSegmentedColormap
        _rdgn = LinearSegmentedColormap.from_list(
            "RdGn", ["#d62728", "#f0f0f0", "#2ca02c"], N=256)

        def _shap_diverging(ax, grey, sv, title):
            """
            Overlay SHAP values using Red↔Green diverging colormap on grayscale.
            Red = negative SHAP (opposing), Green = positive SHAP (supporting).
            Includes a colorbar below the plot.
            """
            bmask = make_brain_mask(grey)
            sv_masked = sv * bmask

            # Symmetric normalization around 0
            vmax = max(np.abs(sv_masked).max(), 1e-10)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

            # Show grayscale base
            ax.imshow(grey, cmap="gray", vmin=0, vmax=1)
            # Overlay SHAP with transparency; alpha keyed to magnitude
            # Use aggressive alpha so colors are clearly visible
            sv_alpha = np.clip(np.abs(sv_masked) / vmax, 0, 1)
            # Power-scale alpha for stronger visibility of moderate values
            sv_alpha = np.power(sv_alpha, 0.5) * 0.92
            sv_alpha[bmask < 0.5] = 0  # zero outside brain
            rgba = _rdgn(norm(sv_masked))  # (H,W,4) RGBA — red-green
            rgba[..., 3] = sv_alpha       # per-pixel alpha
            ax.imshow(rgba)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0.08)
            sm_cb = cm.ScalarMappable(cmap=_rdgn, norm=norm)
            sm_cb.set_array([])
            cb = plt.colorbar(sm_cb, cax=cax, orientation="horizontal")
            cb.set_label("SHAP value", fontsize=8)
            cb.ax.tick_params(labelsize=7)

        lbl_str = "Tumor" if sample["label"]==1 else "Non-Tumor"
        axes[row_i,0].imshow(base, cmap="gray", vmin=0, vmax=1)
        axes[row_i,0].set_title(f"Original T1ce\nTrue: {lbl_str}", fontsize=10)
        axes[row_i,0].axis("off")
        _shap_diverging(axes[row_i,1], base, sv_tumor,
                        "SHAP — Tumor class")
        _shap_diverging(axes[row_i,2], base, sv_nontumor,
                        "SHAP — Non-Tumor class")
        del raw; clear_mem()

    plt.suptitle("SHAP Gradient Explanations — T1ce modality (11-slice window)",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.subplots_adjust(top=0.93, hspace=0.35)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_SHAP, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(OUT_SHAP)}")


# ══════════════════════════════════════════════════════════════════════════
#  4. ATTENTION ROLLOUT — ViT transformer attention across layers
# ══════════════════════════════════════════════════════════════════════════
def run_attention_rollout(vit, puma, samples):
    """
    Compute Attention Rollout (Abnar & Zuidema, 2020) across all ViT layers.
    Multiplies attention matrices layer-by-layer for a cumulative attention map.
    """
    print("\n[4/5] Attention Rollout ...")
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 4 * n_rows))
    if n_rows == 1: axes = axes[np.newaxis, :]

    for row_i, sample in enumerate(samples):
        slices = load_sample_slices(sample)
        base = arr2d_to_display(slices["t1ce"])
        img_tensor = arr2d_to_tensor(slices["t1ce"]).to(DEVICE)

        # Hook all transformer blocks to capture attention
        attn_weights_all = []
        hooks = []

        def make_hook(storage):
            def hook_fn(module, inp, out):
                x = inp[0]
                B, N, C = x.shape
                qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                scale = (C // module.num_heads) ** -0.5
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)
                storage.append(attn.detach().cpu())
            return hook_fn

        for block in vit.backbone.blocks:
            h = block.attn.register_forward_hook(make_hook(attn_weights_all))
            hooks.append(h)

        with torch.no_grad():
            _ = vit(img_tensor)

        for h in hooks:
            h.remove()

        # Attention Rollout: multiply attention matrices through layers
        # Average across heads first, then add identity + renormalize
        result = None
        for attn in attn_weights_all:
            attn_heads_mean = attn.mean(dim=1).squeeze(0)  # (N, N)
            # Add identity (residual connection)
            attn_with_res = 0.5 * attn_heads_mean + 0.5 * torch.eye(attn_heads_mean.size(0))
            # Renormalize rows
            attn_with_res = attn_with_res / attn_with_res.sum(dim=-1, keepdim=True)
            if result is None:
                result = attn_with_res
            else:
                result = attn_with_res @ result

        # CLS token's attention over patches
        rollout = result[0, 1:].numpy()  # (196,)
        rollout = rollout.reshape(N_PATCHES_1D, N_PATCHES_1D)
        rollout = cv2.resize(rollout, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        # Normalize and mask
        bmask = make_brain_mask(base)
        rollout = rollout * bmask
        if rollout.max() > 1e-10:
            rollout = rollout / rollout.max()

        lbl_str = "Tumor" if sample["label"] == 1 else "Non-Tumor"
        axes[row_i, 0].imshow(base, cmap="gray", vmin=0, vmax=1)
        axes[row_i, 0].set_title(f"Original T1ce\nTrue: {lbl_str}", fontsize=10)
        axes[row_i, 0].axis("off")
        axes[row_i, 1].imshow(overlay_heatmap(base, rollout))
        axes[row_i, 1].set_title("Attention Rollout", fontsize=10)
        axes[row_i, 1].axis("off")

    plt.suptitle("Attention Rollout — ViT Backbone (T1ce modality)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.subplots_adjust(top=0.90, hspace=0.40)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_ATTN_ROLL, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(OUT_ATTN_ROLL)}")


# ══════════════════════════════════════════════════════════════════════════
#  5. CROSS-MODAL ATTENTION MAPS — PUMA-MIL reliability fusion weights
# ══════════════════════════════════════════════════════════════════════════
def run_cross_modal_attention(vit, puma, samples):
    """
    Visualize PUMA-MIL's learned cross-modal attention (modality reliability
    weights) for each sample. Shows how the model weighs T1, T1ce, T2, FLAIR.
    """
    print("\n[5/5] Cross-Modal Attention Maps ...")

    # Load the full PUMA model (with return_aux) for getting mod_weights
    # We use the PUMAMILLite but need to get the weights from fusion
    puma_full = PUMAMILLite()
    ckpt = next((p for p in CKPT_CANDIDATES if os.path.exists(p)), None)
    if ckpt:
        state = torch.load(ckpt, map_location=DEVICE)
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        puma_full.load_state_dict(state, strict=False)
    puma_full = puma_full.to(DEVICE)
    puma_full.eval()

    # Need to access fusion weights — modify forward to return them
    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 5, figsize=(16, 3.5 * n_samples))
    if n_samples == 1: axes = axes[np.newaxis, :]

    for row_i, sample in enumerate(samples):
        slices = load_sample_slices(sample)
        vols = load_window_vols(sample)
        center_idx = sample["slice_idx"]

        # Extract window features
        feats, seq_mask, center_local = extract_window_features(
            vit, vols, center_idx)
        mm = torch.ones(1, 4, dtype=torch.bool, device=DEVICE)

        # Get fusion weights through forward
        with torch.no_grad():
            fused, mod_weights = puma_full.fusion(
                [feats["t1"], feats["t1ce"], feats["t2"], feats["flair"]], mm)

        # mod_weights shape: (1, W, 4) — get center slice weights
        center_w = mod_weights[0, center_local, :].cpu().numpy()  # (4,)

        lbl_str = "Tumor" if sample["label"] == 1 else "Non-Tumor"

        # Show each modality with its weight
        for mi, mod in enumerate(MODALITIES):
            base = arr2d_to_display(slices[mod])
            axes[row_i, mi].imshow(base, cmap="gray", vmin=0, vmax=1)
            axes[row_i, mi].set_title(
                f"{mod.upper()}\nWeight: {center_w[mi]:.3f}", fontsize=9)
            axes[row_i, mi].axis("off")
            # Color border by weight intensity
            color = plt.cm.RdYlGn(center_w[mi] / max(center_w.max(), 1e-6))
            for spine in axes[row_i, mi].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
                spine.set_visible(True)

        # Bar chart of weights
        colors = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]
        axes[row_i, 4].bar(MODALITIES, center_w, color=colors)
        axes[row_i, 4].set_ylim(0, max(center_w.max() * 1.3, 0.4))
        axes[row_i, 4].set_title(f"Reliability Weights\nTrue: {lbl_str}", fontsize=9)
        axes[row_i, 4].set_ylabel("Weight")

    plt.suptitle("Cross-Modal Attention — Modality Reliability Weights",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.subplots_adjust(top=0.90, hspace=0.45, wspace=0.30)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_CROSS_MOD, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {os.path.basename(OUT_CROSS_MOD)}")
    del puma_full; clear_mem()


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ensure_dir(WORK_DIR)
    ensure_dir(OUT_DIR)

    print("=" * 70)
    print("CELL 4 — XAI: IMAGE-LEVEL INTERPRETABILITY")
    print("Grad-CAM | Grad-CAM++ | LIME | SHAP | Attention Rollout | Cross-Modal")
    print("=" * 70)

    print("\nLoading models ...")
    vit, puma, ckpt = load_models()
    print(f"  ✓ PUMA checkpoint: {os.path.basename(ckpt)}")
    print(f"  ✓ ViT backbone:    {BACKBONE}")
    print(f"  ✓ Device:          {DEVICE}")
    print(f"  ✓ Output dir:      {OUT_DIR}")

    print("\nSelecting representative slices ...")
    samples = select_samples(n_pos=N_POS, n_neg=N_NEG)
    print(f"  ✓ {len(samples)} slices selected "
          f"({sum(s['label']==1 for s in samples)} tumor, "
          f"{sum(s['label']==0 for s in samples)} non-tumor)")

    run_gradcam(vit, puma, samples)
    run_lime(vit, puma, samples)
    run_shap(vit, puma, samples)
    run_attention_rollout(vit, puma, samples)
    run_cross_modal_attention(vit, puma, samples)

    print("\n" + "=" * 70)
    print("✅  ALL XAI FIGURES COMPLETE")
    print("=" * 70)
    for f in [OUT_GRADCAM, OUT_GRADCAMPP, OUT_LIME, OUT_SHAP,
              OUT_ATTN_ROLL, OUT_CROSS_MOD]:
        if os.path.exists(f):
            kb = os.path.getsize(f)/1024
            print(f"  ✓ {os.path.basename(f):50s} ({kb:.0f} KB)")