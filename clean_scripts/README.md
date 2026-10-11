# PUMA-MIL Clean Run — Kaggle Notebook Instructions

## GPU
- **Use T4 GPU** (NOT P100 — P100's sm_60 is no longer supported by Kaggle's PyTorch)

## Datasets Needed
These should already be in your Kaggle account:
1. **Phase 1 output**: `absanim/phase-1-output` → mounts at `/kaggle/input/datasets/absanim/phase-1-output/`
2. **BraTS 2021** (for XAI only): `dschettler8845/brats-2021-task1` → mounts at `/kaggle/input/datasets/dschettler8845/brats-2021-task1/`

## Estimated Time (~3.5 hours on T4)
| Cell | File | Time |
|------|------|------|
| 1 | `cell1_training.py` | ~50 min |
| 2 | `cell2_baselines.py` | ~90 min |
| 3 | `cell3_figures.py` | ~2 min |
| 4 | `cell4_xai.py` | ~30 min |

## How to Run

### 1. Create a NEW Kaggle Notebook
- Accelerator: **GPU T4 x2**
- Add both datasets above

### 2. Paste Code into 4 Cells
Each file has `if __name__ == "__main__"` at the bottom — just paste and run:

| Cell | File |
|------|------|
| 1 | `cell1_training.py` |
| 2 | `cell2_baselines.py` |
| 3 | `cell3_figures.py` |
| 4 | `cell4_xai.py` |

### 3. Run All & Save
Run cells 1→2→3→4 sequentially → Save & Commit

### 4. Output
```
latest_outputs/
├── results/           ← All JSON + CSV metrics
├── checkpoints/       ← .pth model weights
└── output_figs/       ← 19 publication-ready figures (600 DPI)
    ├── Fig01–Fig13     (performance figures)
    └── Fig14–Fig19     (XAI figures)
```
