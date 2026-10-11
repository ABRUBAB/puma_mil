"""
Grad-CAM and Grad-CAM++ for Vision Transformer backbones.
Extracts gradient-weighted class activation maps from the final attention layer.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2

class ViTGradCAM:
    """Computes Grad-CAM and Grad-CAM++ on patch tokens of a Vision Transformer."""
    def __init__(self, vit_model, target_layer_name: str = "blocks.11.norm1"):
        self.model = vit_model
        self.gradients = None
        self.activations = None

        # Hook target layer
        target_module = dict(self.model.named_modules())[target_layer_name]
        target_module.register_forward_hook(self._forward_hook)
        target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = 1,
                     method: str = "gradcam", img_size: int = 224, patch_size: int = 16) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, class_idx].sum()
        score.backward(retain_graph=True)

        acts = self.activations[:, 1:, :]  # drop CLS token -> [B, 196, D]
        grads = self.gradients[:, 1:, :]   # [B, 196, D]

        if method == "gradcam++":
            grad_2 = grads.pow(2)
            grad_3 = grads.pow(3)
            eps = 1e-7
            alpha = grad_2 / (2 * grad_2 + (acts * grad_3).sum(dim=1, keepdim=True) + eps)
            weights = (alpha * F.relu(grads)).sum(dim=1, keepdim=True)
        else:
            weights = grads.mean(dim=1, keepdim=True)

        cam = (weights * acts).sum(dim=-1).squeeze(0)  # [196]
        cam = F.relu(cam)
        cam = cam.view(img_size // patch_size, img_size // patch_size).cpu().detach().numpy()

        cam = cv2.resize(cam, (img_size, img_size))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
