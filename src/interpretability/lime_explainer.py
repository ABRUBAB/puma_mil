"""
LIME Superpixel Explainability for MRI Slices.
Uses Felzenszwalb or Quickshift segmentation to generate interpretable superpixel masks.
"""

import numpy as np
from skimage.segmentation import quickshift, mark_boundaries
import torch

def explain_slice_lime(slice_rgb: np.ndarray, predict_fn, num_samples: int = 500,
                       kernel_size: int = 4, max_dist: int = 20) -> np.ndarray:
    """
    Computes LIME superpixel importance weights for an input RGB image slice.
    """
    # 1. Segment image into superpixels
    segments = quickshift(slice_rgb, kernel_size=kernel_size, max_dist=max_dist, ratio=0.2)
    n_features = len(np.unique(segments))

    # 2. Perturbation sampling
    rng = np.random.RandomState(42)
    binary_masks = rng.randint(0, 2, size=(num_samples, n_features))
    binary_masks[0, :] = 1  # include original

    perturbed_imgs = []
    for mask in binary_masks:
        img_p = slice_rgb.copy()
        zeros = np.where(mask == 0)[0]
        for z in zeros:
            img_p[segments == z] = 0
        perturbed_imgs.append(img_p)

    # 3. Predict on perturbed images
    preds = predict_fn(np.stack(perturbed_imgs, axis=0))
    pos_probs = preds[:, 1]

    # 4. Weighted Ridge regression
    dists = np.sqrt(np.sum((binary_masks - 1.0) ** 2, axis=1))
    weights = np.exp(-(dists ** 2) / (0.25 * n_features))

    X = binary_masks
    y = pos_probs
    W = np.diag(weights)
    beta = np.linalg.pinv(X.T @ W @ X + 1e-3 * np.eye(n_features)) @ (X.T @ W @ y)

    # 5. Build heatmap
    heatmap = np.zeros_like(segments, dtype=float)
    for i in range(n_features):
        heatmap[segments == i] = beta[i]

    heatmap = np.clip(heatmap, 0, None)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap, segments
