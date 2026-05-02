"""Visualization helpers for image grids, overlays, and bounding boxes."""

from __future__ import annotations

from io import BytesIO
from typing import Iterable, List, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw
from torchvision.utils import save_image


def draw_boxes(image: np.ndarray, boxes: Sequence[Sequence[float]], color: str = "red") -> np.ndarray:
    """Draw rectangular bounding boxes on an RGB image."""
    pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    drawer = ImageDraw.Draw(pil_image)
    width, height = pil_image.size
    for box in boxes:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        # Accept both xyxy and xywh formats.
        if x2 < x1 or y2 < y1:
            x2 = x1 + max(0.0, x2)
            y2 = y1 + max(0.0, y2)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1 = max(0.0, min(x1, width - 1.0))
        y1 = max(0.0, min(y1, height - 1.0))
        x2 = max(0.0, min(x2, width - 1.0))
        y2 = max(0.0, min(y2, height - 1.0))
        if x2 <= x1 or y2 <= y1:
            continue
        drawer.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
    return np.array(pil_image)


def draw_boxes_on_image(image_np: np.ndarray, boxes: Sequence[Sequence[float]], scores: Sequence[float], threshold: float = 0.5) -> np.ndarray:
    """Draw predicted boxes on image using score threshold."""
    drawn = image_np.copy()
    if drawn.ndim == 2:
        drawn = cv2.cvtColor(drawn, cv2.COLOR_GRAY2RGB)
    h, w = drawn.shape[:2]
    cv2.rectangle(drawn, (1, 1), (w - 2, h - 2), (0, 255, 0), 2)
    for box, score in zip(boxes, scores):
        if float(score) < threshold:
            continue
        x, y, bw, bh = box
        p1 = (int(x - bw / 2), int(y - bh / 2))
        p2 = (int(x + bw / 2), int(y + bh / 2))
        cv2.rectangle(drawn, p1, p2, (255, 0, 0), 2)
    return drawn


def save_image_grid(images_tensor, filepath: str, nrow: int = 8) -> None:
    """Save tensor batch as image grid."""
    save_image(images_tensor, filepath, nrow=nrow)


def make_image_grid(images: List[np.ndarray], columns: int = 4) -> np.ndarray:
    """Create a simple image gallery grid from a list of equally sized images."""
    if not images:
        return np.zeros((128, 128, 3), dtype=np.uint8)

    h, w = images[0].shape[:2]
    rows = int(np.ceil(len(images) / columns))
    grid = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)

    for idx, image in enumerate(images):
        r, c = divmod(idx, columns)
        img = image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = img

    return grid


def encode_png_bytes(image: np.ndarray) -> bytes:
    """Encode image array to PNG bytes."""
    pil_img = Image.fromarray(image.astype(np.uint8))
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()
