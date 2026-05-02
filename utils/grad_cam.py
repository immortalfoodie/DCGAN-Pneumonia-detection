"""Grad-CAM and Grad-CAM++ utilities for model interpretability."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
from torch import Tensor, nn


class GradCAM:
    """Generate Grad-CAM heatmaps from a target convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[Tensor] = None
        self.gradients: Optional[Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module: nn.Module, _input: tuple[Tensor, ...], output: Tensor) -> None:
            self.activations = output

        def backward_hook(_module: nn.Module, _grad_input: tuple[Tensor, ...], grad_output: tuple[Tensor, ...]) -> None:
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor: Tensor, class_idx: int = 0) -> np.ndarray:
        """Generate normalized heatmap as HxW numpy array."""
        self.model.zero_grad(set_to_none=True)
        output = self.model(image_tensor, mode="inference")
        score = output["classification"].view(-1)
        score.sum().backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks did not capture gradients/activations.")

        grads = self.gradients[0]
        acts = self.activations[0]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (image_tensor.shape[-1], image_tensor.shape[-2]))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

    @staticmethod
    def overlay(original_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Overlay heatmap on RGB image."""
        if original_image.ndim == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        if heatmap.shape[:2] != original_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

        heat = (heatmap * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay_img = cv2.addWeighted(original_image.astype(np.uint8), 1 - alpha, heat, alpha, 0)
        return overlay_img


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ variant with second-order gradient weighting."""

    def generate(self, image_tensor: Tensor, class_idx: int = 0) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        output = self.model(image_tensor, mode="inference")
        score = output["classification"].view(-1)
        score.sum().backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM++ hooks did not capture gradients/activations.")

        grads = self.gradients[0]
        acts = self.activations[0]

        grads_2 = grads.pow(2)
        grads_3 = grads.pow(3)
        denom = 2 * grads_2 + (acts * grads_3).sum(dim=(1, 2), keepdim=True) + 1e-8
        alpha = grads_2 / denom
        weights = (alpha * torch.relu(grads)).sum(dim=(1, 2), keepdim=True)

        cam = (weights * acts).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (image_tensor.shape[-1], image_tensor.shape[-2]))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam
