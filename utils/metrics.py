"""Evaluation and plotting utilities for training and dashboarding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
import torch

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception:
    FrechetInceptionDistance = None  # type: ignore[assignment]


def compute_classification_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute standard binary classification metrics."""
    y_pred = (y_pred_prob >= threshold).astype(np.int32)
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc),
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


def compute_detection_metrics(pred_boxes: List[np.ndarray], true_boxes: List[np.ndarray], iou_threshold: float = 0.5) -> Dict[str, float]:
    """Compute simple detection metrics (mAP proxy at IoU threshold)."""
    tp = fp = fn = 0
    for pred, true in zip(pred_boxes, true_boxes):
        if len(pred) == 0 and len(true) == 0:
            continue
        if len(pred) == 0:
            fn += len(true)
            continue
        if len(true) == 0:
            fp += len(pred)
            continue
        matched = min(len(pred), len(true))
        tp += matched
        fp += max(0, len(pred) - matched)
        fn += max(0, len(true) - matched)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"mAP": float(precision * recall), "precision_at_50": float(precision), "recall_at_50": float(recall)}


def compute_gan_metrics(real_images_dir: str, fake_images_dir: str) -> Dict[str, float]:
    """Compute FID/KID-style GAN quality metrics."""
    if FrechetInceptionDistance is None:
        return {"fid": -1.0, "kid": -1.0}

    def _load_images(folder: Path) -> torch.Tensor:
        images = []
        for img_path in list(folder.glob("*.png"))[:64]:
            img = plt.imread(img_path)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            arr = (img[:, :, :3] * 255).astype(np.uint8)
            images.append(torch.from_numpy(arr).permute(2, 0, 1))
        if not images:
            return torch.zeros((1, 3, 128, 128), dtype=torch.uint8)
        return torch.stack(images)

    fid = FrechetInceptionDistance(feature=64)
    real = _load_images(Path(real_images_dir))
    fake = _load_images(Path(fake_images_dir))
    fid.update(real, real=True)
    fid.update(fake, real=False)
    fid_score = float(fid.compute().item())
    return {"fid": fid_score, "kid": -1.0}


def plot_roc_curve(y_true: np.ndarray, y_pred_prob: np.ndarray, save_path: str) -> None:
    """Plot and save ROC curve in PNG and HTML."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    score = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={score:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash"}, name="Chance"))
    fig.update_layout(template="plotly_dark", title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
    png_path = Path(save_path).with_suffix(".png")
    html_path = Path(save_path).with_suffix(".html")
    fig.write_html(str(html_path))
    fig.write_image(str(png_path))


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    """Plot and save confusion matrix in PNG and HTML."""
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
    png_path = Path(save_path).with_suffix(".png")
    html_path = Path(save_path).with_suffix(".html")
    fig.write_html(str(html_path))
    fig.write_image(str(png_path))


def save_metrics_json(metrics_dict: Dict[str, Any], path: str = "checkpoints/metrics.json") -> None:
    """Save or append metrics json to disk."""
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics_dict, file, indent=2)


def save_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    """Save training loss and accuracy curves as PNG."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.get("train_loss", []), label="Train Loss")
    plt.plot(history.get("val_loss", []), label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.get("train_acc", []), label="Train Acc")
    plt.plot(history.get("val_acc", []), label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
