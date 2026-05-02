"""DenseNet-121 backbone with Feature Pyramid Network."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNetFPN(nn.Module):
    """DenseNet backbone that exposes multi-scale FPN features."""

    def __init__(self, fpn_channels: int = 256, pretrained: bool = True, freeze_backbone: bool = False) -> None:
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = densenet121(weights=weights).features

        self.lateral_c2 = nn.Sequential(nn.Conv2d(256, fpn_channels, kernel_size=1), nn.BatchNorm2d(fpn_channels))
        self.lateral_c3 = nn.Sequential(nn.Conv2d(512, fpn_channels, kernel_size=1), nn.BatchNorm2d(fpn_channels))
        self.lateral_c4 = nn.Sequential(nn.Conv2d(1024, fpn_channels, kernel_size=1), nn.BatchNorm2d(fpn_channels))
        self.lateral_c5 = nn.Sequential(nn.Conv2d(1024, fpn_channels, kernel_size=1), nn.BatchNorm2d(fpn_channels))

        self.smooth_p3 = nn.Sequential(nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1), nn.BatchNorm2d(fpn_channels), nn.ReLU(inplace=True))
        self.smooth_p4 = nn.Sequential(nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1), nn.BatchNorm2d(fpn_channels), nn.ReLU(inplace=True))
        self.smooth_p5 = nn.Sequential(nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1), nn.BatchNorm2d(fpn_channels), nn.ReLU(inplace=True))
        if freeze_backbone:
            for layer in [self.backbone.denseblock1, self.backbone.denseblock2]:
                for param in layer.parameters():
                    param.requires_grad = False

    def _extract_backbone_features(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.backbone.conv0(x)
        x = self.backbone.norm0(x)
        x = self.backbone.relu0(x)
        x = self.backbone.pool0(x)

        c2 = self.backbone.denseblock1(x)
        x = self.backbone.transition1(c2)

        c3 = self.backbone.denseblock2(x)
        x = self.backbone.transition2(c3)

        c4 = self.backbone.denseblock3(x)
        x = self.backbone.transition3(c4)

        c5 = self.backbone.denseblock4(x)
        c5 = self.backbone.norm5(c5)
        c5 = torch.relu(c5)

        return {"C2": c2, "C3": c3, "C4": c4, "C5": c5}

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        feats = self._extract_backbone_features(x)
        c3, c4, c5 = feats["C3"], feats["C4"], feats["C5"]

        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral_c3(c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        p5 = self.smooth_p5(p5)
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)

        return {
            "P3": p3,
            "P4": p4,
            "P5": p5,
        }
