# PUMA-MIL: Prototype-Updated Multi-Modal Attention-Based Multiple Instance Learning for Slice-Level Glioma Detection on Multi-Sequence MRI

Official PyTorch implementation of **PUMA-MIL** by **Abdullah Rubab** (<rubab2712@gmail.com>).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 📌 Overview

Axial slice-based tumor detection provides an efficient, computationally accessible paradigm for clinical MRI reading list prioritization. However, existing slice classifiers frequently rely on static modality fusion, evaluate slices in isolation without spatial context, suffer from severe boundary slice ambiguity, and lack calibrated confidence estimates.

**PUMA-MIL** (*Prototype-Updated Multi-Modal Attention-Based Multiple Instance Learning*) addresses these challenges through a unified framework:
1. **Reliability-Weighted Modality Fusion:** Dynamically weights multi-sequence MRI representations (T1, T1ce, T2, FLAIR) per sample using learned reliability gating combined with stochastic modality dropout (10%).
2. **Context-Aware Transformer Sequence Modeling:** Analyzes axial neighborhood slices in a sliding window ($W=11$) using a 2-layer Transformer encoder with learnable positional embeddings.
3. **Hybrid MIL Pooling with EMA Prototype Anchoring:** Aggregates instance representations through entropy-modulated gated attention and deterministic top-$K$ positive/negative selection, anchored against Exponential Moving Average (EMA) class prototypes ($\mu=0.90$).
4. **Role-Aware Multi-Objective Training:** Employs an asymmetric focal loss ($\gamma_+=1.5, \gamma_-=3.0$) with role-specific multipliers prioritizing visually challenging boundary negative slices ($R_{\text{boundary}}=1.60$), joined with auxiliary bag supervision, temporal smoothness, and modality diversity regularization.

---

## 📊 Key Results

### Internal Evaluation: BraTS-2021 Test Set (4,848 slices across 101 patients)
Evaluated with a 3-seed ensemble ($\{42, 123, 456\}$) and 2,000-sample bootstrap 95% confidence intervals:

| Metric | PUMA-MIL Ensemble | 95% Bootstrap CI | ABMIL | TransMIL | CLAM-SB | ViT-Slice-MLP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Accuracy** | **0.9521** | [0.9460, 0.9581] | 0.9328 | 0.9416 | 0.9309 | 0.9022 |
| **Macro-F1** | **0.9498** | [0.9442, 0.9549] | 0.9312 | 0.9385 | 0.9298 | 0.8974 |
| **ROC-AUC** | **0.9910** | — | 0.9856 | 0.9882 | 0.9834 | 0.9678 |
| **PR-AUC** | **0.9938** | — | 0.9904 | 0.9921 | 0.9891 | 0.9782 |
| **Sensitivity** | **0.9718** | [0.9657, 0.9776] | 0.9643 | 0.9708 | 0.9588 | 0.9258 |
| **Specificity** | **0.9225** | [0.9100, 0.9339] | 0.8856 | 0.8978 | 0.8891 | 0.8669 |
| **Balanced Acc** | **0.9472** | [0.9405, 0.9535] | 0.9249 | 0.9343 | 0.9239 | 0.8964 |
| **ECE** | **0.0216** | — | 0.0384 | 0.0312 | 0.0410 | 0.0521 |

### Fully Locked External Transfer: BraTS-2020 (17,664 slices across 369 patients)
Without any retraining, parameter fine-tuning, or threshold adaptation:
- **Macro-F1:** `0.9705`
- **ROC-AUC:** `0.9967`
- **Accuracy:** `0.9723`
- **Sensitivity:** `0.9818` (10,484 TP vs. 194 FN)
- **Specificity:** `0.9505` (6,683 TN vs. 348 FP)

---

## 🏗️ Repository Architecture

```
puma_mil/
├── .gitignore
├── LICENSE                          # MIT License
├── README.md                        # Documentation
├── requirements.txt                 # Python dependencies
├── environment.yml                  # Conda environment specification
├── notebooks/
│   └── puma_mil_pipeline.ipynb      # Complete interactive end-to-end pipeline
├── scripts/
│   ├── 01_prepare_data.py           # Phase 1: Data prep & frozen ViT feature extraction
│   ├── 02_train_puma_mil.py         # Phase 2: PUMA-MIL 3-seed ensemble training
│   ├── 03_train_baselines.py        # Phase 3: Fair competitive baselines benchmark
│   ├── 04_sensitivity_analysis.py   # Phase 3c: Architectural sensitivity sweeps
│   ├── 05_external_validation.py    # Phase 4a: Locked external validation on BraTS-2020
│   ├── 06_generate_figures.py       # Generation of 600 DPI publication plots
│   └── 07_interpretability_xai.py   # Phase 4b: Multi-method XAI (Grad-CAM, LIME, SHAP)
├── src/
│   ├── models/
│   │   ├── puma_mil.py              # PUMA-MIL Lite architecture
│   │   └── baselines.py             # ABMIL, TransMIL, CLAM-SB, ViT-Slice-MLP
│   ├── data/
│   │   ├── preprocessing.py         # 3-stage normalization, center-crop, role assignment
│   │   └── dataset.py               # CenterWindowDataset & role-weighted sampling
│   ├── training/
│   │   ├── losses.py                # Role-aware asymmetric focal, bag, smoothness, diversity
│   │   └── trainer.py               # Multi-seed training engine & evaluation loops
│   ├── evaluation/
│   │   ├── metrics.py               # Comprehensive metrics & bootstrap CIs
│   │   ├── temperature_scaling.py   # Platt scaling via L-BFGS
│   │   └── selective_prediction.py  # Risk-coverage deferral analysis
│   ├── interpretability/
│   │   ├── gradcam.py               # Grad-CAM and Grad-CAM++
│   │   ├── lime_explainer.py        # LIME superpixel attribution
│   │   ├── shap_explainer.py        # SHAP GradientExplainer
│   │   └── attention_rollout.py     # Attention rollout & modality weight analysis
│   └── utils/
│       ├── config.py                # Configuration management
│       └── visualization.py         # Publication figure plotting
└── clean_scripts/                   # Modular Kaggle cell execution scripts
    ├── cell0_preprocessing_viz.py
    ├── cell1_training.py
    ├── cell2_baselines.py
    ├── cell3_figures.py
    ├── cell4_xai.py
    └── cell5_fix_figures.py
```

---

## ⚙️ Installation & Environment Setup

### Option 1: Conda Environment
```bash
git clone https://github.com/ABRUBAB/puma_mil.git
cd puma_mil

conda env create -f environment.yml
conda activate puma_mil
```

### Option 2: Pip Installation
```bash
git clone https://github.com/ABRUBAB/puma_mil.git
cd puma_mil

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Reproduction Guide

### 1. Data Preparation & Feature Extraction
Download the BraTS-2021 dataset (e.g. from Kaggle `dschettler8845/brats-2021-task1` or CBICA portal) and run:
```bash
python scripts/01_prepare_data.py --data_dir /path/to/BraTS2021 --out_dir ./outputs
```
This performs:
- 3-stage volume intensity normalization (percentile clipping, z-score, min-max to $[0,1]$).
- Center-crop $K=48$ slice selection with post-hoc role assignment (positive, boundary negative, far negative, mid negative, no-tumor).
- Feature extraction using frozen ImageNet-pretrained `vit_small_patch16_224` (384-dimensional CLS token).

### 2. Train PUMA-MIL Ensemble
Train the 3-seed ensemble ($\{42, 123, 456\}$):
```bash
python scripts/02_train_puma_mil.py --data_dir ./outputs/features --out_dir ./outputs
```

### 3. Evaluate Competitive Baselines
Train ABMIL, TransMIL, CLAM-SB, and ViT-Slice-MLP under the identical fair protocol:
```bash
python scripts/03_train_baselines.py --data_dir ./outputs/features --out_dir ./outputs
```

### 4. Locked External Validation (BraTS-2020)
Evaluate the frozen ensemble checkpoints on the 369 BraTS-2020 subjects:
```bash
python scripts/05_external_validation.py --data_dir /path/to/BraTS2020 --out_dir ./outputs
```

### 5. Multi-Method Interpretability (XAI)
Generate pixel-level and superpixel attributions:
```bash
python scripts/07_interpretability_xai.py --out_dir ./outputs/output_figs
```

---

## 📓 Running on Kaggle

For streamlined cloud execution with GPU T4/P100:
1. Open a new notebook on Kaggle and select accelerator **GPU T4 x2**.
2. Mount the datasets:
   - `absanim/phase-1-output`
   - `dschettler8845/brats-2021-task1`
3. Execute the modular scripts in `clean_scripts/`:
   - Cell 1: `clean_scripts/cell1_training.py`
   - Cell 2: `clean_scripts/cell2_baselines.py`
   - Cell 3: `clean_scripts/cell3_figures.py`
   - Cell 4: `clean_scripts/cell4_xai.py`
   - Cell 5: `clean_scripts/cell5_fix_figures.py`

---

## 👤 Author & Contact

**Abdullah Rubab**  
- Email: [rubab2712@gmail.com](mailto:rubab2712@gmail.com)  
- GitHub: [@ABRUBAB](https://github.com/ABRUBAB)  
- Repository: [https://github.com/ABRUBAB/puma_mil](https://github.com/ABRUBAB/puma_mil)

---

## 📜 Citation

```bibtex
@article{rubab2026pumamil,
  title     = {PUMA-MIL: Prototype-Updated Multi-Modal Attention-Based Multiple Instance Learning for Slice-Level Glioma Detection on Multi-Sequence MRI},
  author    = {Rubab, Abdullah},
  year      = {2026},
  publisher = {GitHub},
  journal   = {GitHub repository},
  howpublished = {\url{https://github.com/ABRUBAB/puma_mil}}
}
```

---

## 📄 License

This repository is distributed under the MIT License. See [LICENSE](LICENSE) for details.
