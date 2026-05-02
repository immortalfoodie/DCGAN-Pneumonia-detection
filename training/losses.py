"""Loss definitions for classification, detection, and GAN training."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class FocalLoss(nn.Module):
    """Binary focal loss with configurable alpha and gamma."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        target = target.float()
        pred = pred.float().clamp(1e-6, 1 - 1e-6)
        bce_loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        pt = target * pred + (1 - target) * (1 - pred)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - pt).pow(self.gamma)
        loss = focal_weight * bce_loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class CIoULoss(nn.Module):
    """Complete IoU loss for boxes in cx, cy, w, h format."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
        if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
            return pred_boxes.new_tensor(0.0)

        pcx, pcy, pw, ph = pred_boxes.unbind(dim=-1)
        tcx, tcy, tw, th = target_boxes.unbind(dim=-1)
        pred_x1, pred_y1, pred_x2, pred_y2 = pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2
        tgt_x1, tgt_y1, tgt_x2, tgt_y2 = tcx - tw / 2, tcy - th / 2, tcx + tw / 2, tcy + th / 2

        inter_x1 = torch.maximum(pred_x1, tgt_x1)
        inter_y1 = torch.maximum(pred_y1, tgt_y1)
        inter_x2 = torch.minimum(pred_x2, tgt_x2)
        inter_y2 = torch.minimum(pred_y2, tgt_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        pred_area = (pred_x2 - pred_x1).clamp(min=1e-6) * (pred_y2 - pred_y1).clamp(min=1e-6)
        tgt_area = (tgt_x2 - tgt_x1).clamp(min=1e-6) * (tgt_y2 - tgt_y1).clamp(min=1e-6)
        union = pred_area + tgt_area - inter_area + 1e-7
        iou = inter_area / union

        pred_cx = (pred_x1 + pred_x2) / 2
        pred_cy = (pred_y1 + pred_y2) / 2
        tgt_cx = (tgt_x1 + tgt_x2) / 2
        tgt_cy = (tgt_y1 + tgt_y2) / 2
        center_dist = (pred_cx - tgt_cx).pow(2) + (pred_cy - tgt_cy).pow(2)

        enc_x1 = torch.minimum(pred_x1, tgt_x1)
        enc_y1 = torch.minimum(pred_y1, tgt_y1)
        enc_x2 = torch.maximum(pred_x2, tgt_x2)
        enc_y2 = torch.maximum(pred_y2, tgt_y2)
        enc_diag = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2) + 1e-7

        pred_w = (pred_x2 - pred_x1).clamp(min=1e-6)
        pred_h = (pred_y2 - pred_y1).clamp(min=1e-6)
        tgt_w = (tgt_x2 - tgt_x1).clamp(min=1e-6)
        tgt_h = (tgt_y2 - tgt_y1).clamp(min=1e-6)

        v = (4 / math.pi**2) * (torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h)).pow(2)
        alpha = v / (1 - iou + v + 1e-7)

        ciou = iou - (center_dist / enc_diag) - alpha * v
        loss = 1 - ciou
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class GANHingeLoss:
    """Hinge losses for generator and discriminator training."""

    @staticmethod
    def generator_loss(fake_scores_d1: Tensor, fake_scores_d2: Tensor) -> Tensor:
        return -fake_scores_d1.mean() + -fake_scores_d2.mean()

    @staticmethod
    def discriminator_loss(real_scores: Tensor, fake_scores: Tensor) -> Tensor:
        return torch.relu(1.0 - real_scores).mean() + torch.relu(1.0 + fake_scores).mean()


class CombinedLoss(nn.Module):
    """Combined focal classification and CIoU detection objective."""

    def __init__(self, lambda_det: float = 1.0, focal_alpha: float = 0.25, focal_gamma: float = 2.0) -> None:
        super().__init__()
        self.cls_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.det_loss_fn = CIoULoss()
        self.lambda_det = lambda_det

    def forward(
        self,
        cls_pred: Tensor,
        cls_targets: Tensor,
        box_pred: Optional[Tensor] = None,
        box_target: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cls_loss = self.cls_loss_fn(cls_pred, cls_targets)

        if box_pred is None or box_target is None or box_pred.numel() == 0 or box_target.numel() == 0:
            zero = cls_loss.new_tensor(0.0)
            return cls_loss, cls_loss, zero

        det_loss = self.det_loss_fn(box_pred, box_target)
        total = cls_loss + self.lambda_det * det_loss
        return total, cls_loss, det_loss
