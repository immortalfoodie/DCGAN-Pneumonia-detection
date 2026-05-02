"""Unified pneumonia detection model combining GAN, classification and detection."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from config import Config
from models.classifier import ClassificationHead
from models.dcgan import Generator
from models.densenet_fpn import DenseNetFPN
from models.detector import DetectionHead


class PneumoniaDetectionModel(nn.Module):
    """End-to-end model with optional GAN-based batch balancing during training."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = DenseNetFPN(fpn_channels=Config.FPN_CHANNELS, pretrained=True)
        self.classifier = ClassificationHead(
            in_channels=Config.FPN_CHANNELS,
            hidden_dim=512,
            dropout=Config.DROPOUT,
        )
        self.detector = DetectionHead(in_channels=Config.FPN_CHANNELS)
        self.generator = Generator(latent_dim=Config.LATENT_DIM, img_channels=1)

    def load_generator_checkpoint(self, checkpoint_path: str) -> None:
        """Load pretrained generator weights when available."""
        state = torch.load(checkpoint_path, map_location="cpu")
        if "generator" in state:
            self.generator.load_state_dict(state["generator"])
        else:
            self.generator.load_state_dict(state)

    def _balance_batch_with_gan(self, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        labels = labels.float()
        positives = int((labels == 1).sum().item())
        negatives = int((labels == 0).sum().item())

        if positives == negatives:
            return images, labels

        minority_label = 1.0 if positives < negatives else 0.0
        samples_to_add = abs(positives - negatives)

        noise = torch.randn(samples_to_add, Config.LATENT_DIM, device=images.device)
        with torch.no_grad():
            synthetic = self.generator(noise)

        synthetic = nn.functional.interpolate(synthetic, size=images.shape[-2:], mode="bilinear", align_corners=False)
        synthetic = synthetic.repeat(1, 3, 1, 1)
        synth_labels = torch.full((samples_to_add,), minority_label, device=labels.device)

        balanced_images = torch.cat([images, synthetic], dim=0)
        balanced_labels = torch.cat([labels, synth_labels], dim=0)
        return balanced_images, balanced_labels

    def forward(
        self,
        images: Tensor,
        mode: str = "inference",
        labels: Optional[Tensor] = None,
        use_gan_balance: bool = False,
    ) -> Dict[str, Tensor | List[Tensor] | Dict[str, Tensor] | None]:
        """Forward pass for training and inference."""
        if mode == "train" and use_gan_balance and labels is not None and self.generator is not None:
            images, labels = self._balance_batch_with_gan(images, labels)

        pyramid_feats = self.backbone(images)
        cls_out = self.classifier(pyramid_feats["P5"])
        det_out = self.detector(pyramid_feats)

        merged_boxes = torch.cat([d["boxes"] for d in det_out], dim=1)
        merged_scores = torch.cat([d["scores"] for d in det_out], dim=1)
        pred_prob = cls_out["prob"]

        selected_boxes: List[Tensor] = []
        selected_scores: List[Tensor] = []
        for idx in range(merged_boxes.size(0)):
            nms_out = self.detector.apply_nms(merged_boxes[idx], merged_scores[idx])
            selected_boxes.append(nms_out["boxes"])
            selected_scores.append(nms_out["scores"])

        # Keep a deterministic first-item tensor output for inference APIs.
        out_boxes = selected_boxes[0] if selected_boxes else merged_boxes.new_zeros((0, 4))
        out_scores = selected_scores[0] if selected_scores else merged_scores.new_zeros((0, 1))
        label = "PNEUMONIA" if float(pred_prob[0].item()) >= 0.5 else "NORMAL"

        return {
            "classification": pred_prob,
            "label": label,
            "boxes": out_boxes,
            "scores": out_scores,
            "heatmap": None,
            "features": cls_out["features"],
        }
