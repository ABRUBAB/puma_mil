# ============================================================
# PUMA-MIL CLEAN RUN — CELL 0: PREPROCESSING VISUALIZATION
# Generates publication-ready figures showing each preprocessing
# step exactly as implemented in phase1_data_preparation.py
# No GPU required. No model training. ~2 min runtime.
# ============================================================

import os
import glob
import json
import tarfile
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# ============================================================
# CONFIG — matches phase1_data_preparation.py exactly
# ============================================================

class Config:
    # --- Kaggle paths ---
    BRATS2021_TAR = "/kaggle/input/datasets/dschettler8845/brats-2021-task1/BraTS2021_Training_Data.tar"
    BRATS2021_EXTRACT_DIR = "/kaggle/temp/brats2021_extracted"
    WORK_DIR = "/kaggle/working"

    # --- Preprocessing params (from phase1) ---
    IMG_SIZE = 224
    MODALITIES = ("t1", "t1ce", "t2", "flair")
    MODALITY_LABELS = ("T1", "T1ce", "T2", "FLAIR")
    TUMOR_PIXEL_THRESHOLD = 100
    BACKBONE = "vit_small_patch16_224"  # frozen, 384-dim CLS token

    # --- Role-based selection params (from phase1) ---
    MAX_SELECTED_SLICES = 48
    BOUNDARY_NEG_RADIUS = 6
    FAR_NEG_MIN_DISTANCE = 12

    # --- Figure params ---
    FIG_DPI = 600

CFG = Config()

SLICE_ROLE_TO_ID = {
    "far_neg": 0, "boundary_neg": 1, "positive": 2,
    "mid_neg": 3, "no_tumor": 4,
}
ID_TO_SLICE_ROLE = {v: k for k, v in SLICE_ROLE_TO_ID.items()}

ROLE_COLORS = {
    "positive": "#2ecc71",
    "boundary_neg": "#e67e22",
    "far_neg": "#3498db",
    "mid_neg": "#9b59b6",
    "no_tumor": "#95a5a6",
}
ROLE_DISPLAY = {
    "positive": "Tumor-Positive",
    "boundary_neg": "Boundary Negative",
    "far_neg": "Far Negative",
    "mid_neg": "Mid Negative",
    "no_tumor": "No Tumor (entire volume)",
}


# ============================================================
# PREPROCESSING — exact copy from phase1_data_preparation.py
# ============================================================

def percentile_normalize(vol):
    """Exact copy of phase1_data_preparation.percentile_normalize"""
    vol = np.asarray(vol, dtype=np.float32)
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip(vol, p1, p99)
    mean, std = vol.mean(), vol.std()
    if std > 1e-6:
        vol = (vol - mean) / std
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin > 1e-6:
        vol = (vol - vmin) / (vmax - vmin)
    return vol.astype(np.float32)


def slice_preprocessor(arr2d):
    """
    Exact copy of phase1 SlicePreprocessor.__call__
    Returns: resized 224x224 pseudo-RGB numpy array (for visualization).
    """
    arr2d = np.asarray(arr2d, dtype=np.float32)
    arr2d = np.clip(arr2d, 0.0, 1.0)
    img = Image.fromarray((arr2d * 255).astype(np.uint8)).convert("RGB")
    img = img.resize((CFG.IMG_SIZE, CFG.IMG_SIZE), Image.BILINEAR)
    return np.array(img)


def imagenet_normalize(img_rgb):
    """
    ImageNet normalization as in phase1:
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    Returns float array for visualization of the normalized tensor.
    """
    arr = np.asarray(img_rgb, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normed = (arr - mean) / std
    # Rescale to [0,1] for display only
    vmin, vmax = normed.min(), normed.max()
    if vmax - vmin > 1e-6:
        normed = (normed - vmin) / (vmax - vmin)
    return normed.astype(np.float32)


# ============================================================
# DATA LOADING HELPERS
# ============================================================

def extract_2021_if_needed():
    """Extract BraTS2021 tar if not already done."""
    if os.path.isdir(CFG.BRATS2021_EXTRACT_DIR):
        dirs = glob.glob(os.path.join(CFG.BRATS2021_EXTRACT_DIR, "**", "BraTS2021_*"), recursive=True)
        if len(dirs) > 10:
            print(f"Already extracted: {len(dirs)} patient dirs found")
            return
    os.makedirs(CFG.BRATS2021_EXTRACT_DIR, exist_ok=True)
    print("Extracting BraTS2021 tar...")
    with tarfile.open(CFG.BRATS2021_TAR, "r") as tar:
        tar.extractall(CFG.BRATS2021_EXTRACT_DIR)
    print("Done extracting.")


def find_patient_dirs(n=3):
    """Find n patient directories with tumors."""
    pattern = os.path.join(CFG.BRATS2021_EXTRACT_DIR, "**", "BraTS2021_*")
    dirs = sorted(glob.glob(pattern, recursive=True))
    dirs = [d for d in dirs if os.path.isdir(d)]
    # Pick patients with clear tumors
    selected = []
    for d in dirs:
        seg_files = glob.glob(os.path.join(d, "*seg*"))
        if seg_files:
            seg = nib.load(seg_files[0]).get_fdata()
            tumor_slices = np.sum(seg > 0, axis=(0, 1))
            max_tumor = tumor_slices.max()
            if max_tumor > 500:  # nice big tumor for visualization
                selected.append(d)
                if len(selected) >= n:
                    break
    print(f"Selected {len(selected)} patients for visualization")
    return selected


def load_patient_volumes(patient_dir):
    """Load all 4 modality volumes + segmentation for a patient."""
    pid = os.path.basename(patient_dir)
    vols = {}
    for mod in CFG.MODALITIES:
        pattern = os.path.join(patient_dir, f"*{mod}*")
        # Exclude seg files
        files = [f for f in glob.glob(pattern) if "seg" not in f.lower()]
        if files:
            vols[mod] = nib.load(files[0]).get_fdata()
        else:
            raise FileNotFoundError(f"Missing {mod} for {pid}")

    seg_files = glob.glob(os.path.join(patient_dir, "*seg*"))
    seg = nib.load(seg_files[0]).get_fdata() if seg_files else np.zeros_like(vols["t1"])
    return pid, vols, seg


def find_best_tumor_slice(seg):
    """Find the axial slice with the most tumor pixels."""
    tumor_per_slice = np.sum(seg > 0, axis=(0, 1))
    return int(np.argmax(tumor_per_slice))


def find_nontumor_slice(seg):
    """Find a clean non-tumor slice (far from tumor)."""
    tumor_per_slice = np.sum(seg > 0, axis=(0, 1))
    nontumor = np.where(tumor_per_slice == 0)[0]
    if len(nontumor) == 0:
        return 0
    # Pick one roughly in the middle of the non-tumor range
    return int(nontumor[len(nontumor) // 3])


# ============================================================
# FIGURE STYLE — matches cell3_figures.py style
# ============================================================

def set_journal_style():
    """Match the style used in cell3_figures.py"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": CFG.FIG_DPI,
        "savefig.dpi": CFG.FIG_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": False,
    })


# ============================================================
# FIGURE 1: PREPROCESSING PIPELINE (4 modalities × 6 steps)
# ============================================================

def generate_preprocessing_pipeline_figure(patient_dir, out_dir):
    """
    Shows one patient, one tumor slice, all 4 modalities through
    each preprocessing step:
      (a) Raw NIfTI → (b) Percentile Clip [P1,P99] → (c) Z-Score →
      (d) Min-Max [0,1] → (e) 224×224 Pseudo-RGB → (f) ImageNet Norm
    """
    pid, vols, seg = load_patient_volumes(patient_dir)
    z = find_best_tumor_slice(seg)
    print(f"  Pipeline figure: Patient {pid}, slice z={z}")

    step_labels = [
        "(a) Raw NIfTI\nSlice",
        "(b) Percentile\nClip [$P_1$, $P_{99}$]",
        "(c) Z-Score\nNormalization",
        "(d) Min-Max\nScaling [0, 1]",
        "(e) 224×224\nPseudo-RGB",
        "(f) ImageNet\nNormalization",
    ]

    fig, axes = plt.subplots(4, 6, figsize=(14, 9.5))

    for row, mod in enumerate(CFG.MODALITIES):
        raw_vol = vols[mod].astype(np.float32)
        raw_slice = raw_vol[:, :, z]

        # Step (a): Raw
        axes[row, 0].imshow(raw_slice, cmap="gray", aspect="equal")
        axes[row, 0].set_ylabel(CFG.MODALITY_LABELS[row], fontsize=11,
                                fontweight="bold", rotation=90, labelpad=10)

        # Step (b): Percentile clip
        p1, p99 = np.percentile(raw_vol, [1, 99])
        clipped = np.clip(raw_vol, p1, p99)
        axes[row, 1].imshow(clipped[:, :, z], cmap="gray", aspect="equal")

        # Step (c): Z-score
        mean, std = clipped.mean(), clipped.std()
        zscore = (clipped - mean) / (std + 1e-8)
        axes[row, 2].imshow(zscore[:, :, z], cmap="gray", aspect="equal")

        # Step (d): Min-max to [0,1]
        vmin, vmax = zscore.min(), zscore.max()
        minmax = (zscore - vmin) / (vmax - vmin + 1e-8)
        axes[row, 3].imshow(minmax[:, :, z], cmap="gray", aspect="equal")

        # Step (e): 224×224 pseudo-RGB
        slice_01 = minmax[:, :, z]
        rgb_224 = slice_preprocessor(slice_01)
        axes[row, 4].imshow(rgb_224, aspect="equal")

        # Step (f): ImageNet normalization (display-rescaled)
        inet = imagenet_normalize(rgb_224)
        axes[row, 5].imshow(inet, aspect="equal")

    # Column titles
    for col, label in enumerate(step_labels):
        axes[0, col].set_title(label, fontsize=9, pad=8)

    # Remove ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Preprocessing Pipeline — Patient {pid}, Axial Slice z = {z}",
        fontsize=13, fontweight="bold", y=0.98
    )

    # Arrow annotations between columns
    for col in range(5):
        x_start = (col + 1) / 6.0 - 0.008
        for row in range(4):
            y_pos = 1.0 - (row + 0.5) / 4.0 - 0.02
            fig.text(x_start, y_pos, "→", fontsize=14,
                     ha="center", va="center", transform=fig.transFigure,
                     color="#555555")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "Fig_preprocessing_pipeline.png")
    fig.savefig(path, dpi=CFG.FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIGURE 2: SAMPLE SLICES (tumor-positive vs tumor-negative)
# ============================================================

def generate_sample_slices_figure(patient_dirs, out_dir):
    """
    2 rows: tumor-positive, tumor-negative
    4 columns: T1, T1ce, T2, FLAIR
    Each cell shows the preprocessed [0,1] slice with tumor overlay.
    Uses 2 different patients for variety.
    """
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))

    row_labels = ["Tumor-Positive\nSlice", "Tumor-Negative\nSlice"]

    for panel_row, (patient_dir, slice_type) in enumerate([
        (patient_dirs[0], "tumor"),
        (patient_dirs[1 % len(patient_dirs)], "nontumor"),
    ]):
        pid, vols, seg = load_patient_volumes(patient_dir)
        if slice_type == "tumor":
            z = find_best_tumor_slice(seg)
        else:
            z = find_nontumor_slice(seg)

        for col, mod in enumerate(CFG.MODALITIES):
            # Apply full preprocessing
            normed_vol = percentile_normalize(vols[mod])
            slice_2d = normed_vol[:, :, z]

            axes[panel_row, col].imshow(slice_2d, cmap="gray", aspect="equal")

            # Overlay tumor contour for positive slices
            if slice_type == "tumor":
                seg_slice = (seg[:, :, z] > 0).astype(np.float32)
                if seg_slice.max() > 0:
                    axes[panel_row, col].contour(
                        seg_slice, levels=[0.5], colors=["#e74c3c"],
                        linewidths=1.2, linestyles="solid"
                    )

            axes[panel_row, col].set_xticks([])
            axes[panel_row, col].set_yticks([])

            if panel_row == 0:
                axes[panel_row, col].set_title(
                    CFG.MODALITY_LABELS[col], fontsize=12, fontweight="bold", pad=8
                )

        # Row label
        axes[panel_row, 0].set_ylabel(
            row_labels[panel_row], fontsize=10, fontweight="bold",
            rotation=90, labelpad=15
        )

        # Patient ID annotation
        axes[panel_row, 3].annotate(
            f"{pid}\nz = {z}",
            xy=(1.02, 0.5), xycoords="axes fraction",
            fontsize=7, va="center", ha="left", color="#666666"
        )

    fig.suptitle(
        "Sample Multi-Modal MRI Slices — Tumor-Positive vs. Tumor-Negative",
        fontsize=13, fontweight="bold", y=0.98
    )

    # Add legend for contour
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#e74c3c", linewidth=1.5, label="Tumor boundary (from segmentation mask)")
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=1,
              fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    path = os.path.join(out_dir, "Fig_sample_slices.png")
    fig.savefig(path, dpi=CFG.FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIGURE 3: ROLE DISTRIBUTION BAR CHART
# ============================================================

def generate_role_distribution_figure(patient_dirs, out_dir):
    """
    Bar chart showing the 5 slice role categories from the
    reviewer-driven slice selection strategy (Algorithm 1).
    Computed on a sample of patients to show the distribution.
    """
    role_counts = {r: 0 for r in SLICE_ROLE_TO_ID}

    for patient_dir in patient_dirs[:3]:
        pid, vols, seg = load_patient_volumes(patient_dir)
        n_slices = seg.shape[2]

        # Compute tumor pixels per slice
        tumor_per_slice = np.sum(seg > 0, axis=(0, 1))

        # Label each slice using the exact algorithm from phase1
        pos_slices = set(np.where(tumor_per_slice >= CFG.TUMOR_PIXEL_THRESHOLD)[0])

        if len(pos_slices) == 0:
            for z in range(n_slices):
                role_counts["no_tumor"] += 1
            continue

        for z in range(n_slices):
            if z in pos_slices:
                role_counts["positive"] += 1
            else:
                dist = min(abs(z - p) for p in pos_slices)
                if dist <= CFG.BOUNDARY_NEG_RADIUS:
                    role_counts["boundary_neg"] += 1
                elif dist >= CFG.FAR_NEG_MIN_DISTANCE:
                    role_counts["far_neg"] += 1
                else:
                    role_counts["mid_neg"] += 1

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))

    roles = ["positive", "boundary_neg", "mid_neg", "far_neg", "no_tumor"]
    counts = [role_counts[r] for r in roles]
    colors = [ROLE_COLORS[r] for r in roles]
    labels = [ROLE_DISPLAY[r] for r in roles]
    total = sum(counts)
    pcts = [100 * c / total if total > 0 else 0 for c in counts]

    bars = ax.bar(range(len(roles)), counts, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.65)

    # Value labels on bars
    for bar, count, pct in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Number of Slices", fontsize=10)
    ax.set_title(
        f"Slice Role Distribution (Reviewer-Driven Selection)\n"
        f"$\\tau$ = {CFG.TUMOR_PIXEL_THRESHOLD} pixels, "
        f"boundary $\\leq$ {CFG.BOUNDARY_NEG_RADIUS} slices, "
        f"far $\\geq$ {CFG.FAR_NEG_MIN_DISTANCE} slices",
        fontsize=11, fontweight="bold"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(counts) * 1.18)

    plt.tight_layout()
    path = os.path.join(out_dir, "Fig_role_distribution.png")
    fig.savefig(path, dpi=CFG.FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# FIGURE 4: SEGMENTATION MASK OVERLAY (tumor anatomy)
# ============================================================

def generate_segmentation_overlay_figure(patient_dir, out_dir):
    """
    Shows tumor sub-regions on FLAIR with color-coded overlay:
    - Necrotic Core (label 1) — blue
    - Edema (label 2) — yellow
    - Enhancing Tumor (label 4) — red
    """
    pid, vols, seg = load_patient_volumes(patient_dir)
    z = find_best_tumor_slice(seg)

    flair_norm = percentile_normalize(vols["flair"])
    flair_slice = flair_norm[:, :, z]
    seg_slice = seg[:, :, z]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Panel (a): FLAIR only
    axes[0].imshow(flair_slice, cmap="gray", aspect="equal")
    axes[0].set_title("(a) FLAIR (Preprocessed)", fontsize=10, fontweight="bold")

    # Panel (b): Segmentation mask
    seg_rgb = np.zeros((*seg_slice.shape, 3), dtype=np.float32)
    seg_rgb[seg_slice == 1] = [0.2, 0.4, 0.9]   # Necrotic — blue
    seg_rgb[seg_slice == 2] = [1.0, 0.85, 0.0]   # Edema — yellow
    seg_rgb[seg_slice == 4] = [0.9, 0.15, 0.15]  # Enhancing — red
    axes[1].imshow(seg_rgb, aspect="equal")
    axes[1].set_title("(b) Ground-Truth Segmentation", fontsize=10, fontweight="bold")

    # Panel (c): Overlay
    flair_rgb = np.stack([flair_slice] * 3, axis=-1)
    alpha = 0.45
    overlay = flair_rgb.copy()
    mask = seg_slice > 0
    overlay[mask] = (1 - alpha) * flair_rgb[mask] + alpha * seg_rgb[mask]
    axes[2].imshow(overlay, aspect="equal")
    axes[2].set_title("(c) FLAIR + Segmentation Overlay", fontsize=10, fontweight="bold")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[0.2, 0.4, 0.9], label="Necrotic Core (Label 1)"),
        Patch(facecolor=[1.0, 0.85, 0.0], label="Peritumoral Edema (Label 2)"),
        Patch(facecolor=[0.9, 0.15, 0.15], label="Enhancing Tumor (Label 4)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
              fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f"Tumor Anatomy — Patient {pid}, Axial Slice z = {z}",
        fontsize=12, fontweight="bold", y=1.01
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    path = os.path.join(out_dir, "Fig_segmentation_overlay.png")
    fig.savefig(path, dpi=CFG.FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    set_journal_style()

    out_dir = os.path.join(CFG.WORK_DIR, "output_figs")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("PUMA-MIL — Cell 0: Preprocessing Visualization")
    print("=" * 72)

    # Step 1: Extract data
    extract_2021_if_needed()

    # Step 2: Find patients with visible tumors
    patient_dirs = find_patient_dirs(n=3)
    if len(patient_dirs) == 0:
        raise RuntimeError("No suitable patient directories found!")

    # Step 3: Generate all figures
    print("\n--- Figure 1: Preprocessing Pipeline ---")
    generate_preprocessing_pipeline_figure(patient_dirs[0], out_dir)

    print("\n--- Figure 2: Sample Slices ---")
    generate_sample_slices_figure(patient_dirs, out_dir)

    print("\n--- Figure 3: Role Distribution ---")
    generate_role_distribution_figure(patient_dirs, out_dir)

    print("\n--- Figure 4: Segmentation Overlay ---")
    generate_segmentation_overlay_figure(patient_dirs[0], out_dir)

    # Summary
    figs = [f for f in os.listdir(out_dir) if f.startswith("Fig_") and f.endswith(".png")]
    print("\n" + "=" * 72)
    print(f"DONE — Generated {len(figs)} figures in {out_dir}/")
    for f in sorted(figs):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:45s} {size_kb:8.0f} KB")
    print("=" * 72)


if __name__ == "__main__":
    main()
