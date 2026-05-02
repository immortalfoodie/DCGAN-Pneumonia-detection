"""Anchor-free FCOS-style detector head for multi-scale lesion localization."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import Tensor, nn
from torchvision.ops import nms


class DetectionHead(nn.Module):
    """Predict bounding boxes and confidence maps from FPN levels."""

    def __init__(self, in_channels: int = 256, strides: Sequence[int] = (8, 16, 32)) -> None:
        super().__init__()
        self.strides = list(strides)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.box_head = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.score_head = nn.Sequential(nn.Conv2d(in_channels, 1, kernel_size=1), nn.Sigmoid())

    def decode_boxes(self, raw_boxes: Tensor, stride: int) -> Tensor:
        """Decode (dx, dy, dw, dh) into absolute xywh boxes."""
        bsz, _, h, w = raw_boxes.shape
        yy, xx = torch.meshgrid(torch.arange(h, device=raw_boxes.device), torch.arange(w, device=raw_boxes.device), indexing="ij")
        cx = (xx + 0.5) * stride
        cy = (yy + 0.5) * stride
        dx, dy, dw, dh = raw_boxes[:, 0], raw_boxes[:, 1], raw_boxes[:, 2], raw_boxes[:, 3]
        x = cx.unsqueeze(0) + dx * stride
        y = cy.unsqueeze(0) + dy * stride
        bw = torch.exp(dw).clamp(max=10.0) * stride
        bh = torch.exp(dh).clamp(max=10.0) * stride
        return torch.stack([x, y, bw, bh], dim=-1).view(bsz, -1, 4)

    def forward(self, pyramids: Dict[str, Tensor]) -> List[Dict[str, Tensor]]:
        outputs: List[Dict[str, Tensor]] = []
        for level_name, stride in zip(["P3", "P4", "P5"], self.strides):
            feat = self.relu(self.conv(pyramids[level_name]))
            raw_boxes = self.box_head(feat)
            raw_scores = self.score_head(feat)
            outputs.append(
                {
                    "boxes": self.decode_boxes(raw_boxes, stride),
                    "scores": raw_scores.permute(0, 2, 3, 1).reshape(raw_scores.size(0), -1, 1),
                }
            )
        return outputs

    def apply_nms(self, all_boxes: Tensor, all_scores: Tensor, iou_threshold: float = 0.5) -> Dict[str, Tensor]:
        """Apply NMS on xywh boxes and return filtered detections."""
        xywh = all_boxes
        xyxy = torch.stack(
            [
                xywh[:, 0] - 0.5 * xywh[:, 2],
                xywh[:, 1] - 0.5 * xywh[:, 3],
                xywh[:, 0] + 0.5 * xywh[:, 2],
                xywh[:, 1] + 0.5 * xywh[:, 3],
            ],
            dim=1,
        )
        keep = nms(xyxy, all_scores.squeeze(-1), iou_threshold)
        return {"boxes": all_boxes[keep], "scores": all_scores[keep]}
