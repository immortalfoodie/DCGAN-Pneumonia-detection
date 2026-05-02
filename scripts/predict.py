"""CLI inference for single-image pneumonia diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from config import Config
from models.full_model import PneumoniaDetectionModel
from utils.augmentation import get_val_transforms
from utils.visualization import draw_boxes


def load_threshold() -> float:
    """Load calibrated classification threshold from checkpoints."""
    metrics_path = Config.path(Config.CHECKPOINT_DIR) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            if "threshold" in payload:
                threshold = float(payload["threshold"])
                return threshold if 0.2 <= threshold <= 0.9 else 0.5

    model_path = Config.path(Config.CHECKPOINT_DIR) / "best_model.pth"
    if model_path.exists():
        payload = torch.load(model_path, map_location="cpu")
        if isinstance(payload, dict) and "threshold" in payload:
            threshold = float(payload["threshold"])
            return threshold if 0.2 <= threshold <= 0.9 else 0.5

    return 0.5


def preprocess_image(path: Path) -> torch.Tensor:
    """Read and preprocess image for model inference."""
    image = np.array(Image.open(path).convert("RGB"))
    transform = get_val_transforms(224)
    tensor = transform(image=image)["image"].unsqueeze(0)
    return tensor.to(Config.DEVICE)


def load_model() -> PneumoniaDetectionModel:
    """Load trained model checkpoint."""
    model_path = Config.path(Config.CHECKPOINT_DIR) / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = PneumoniaDetectionModel().to(Config.DEVICE)
    state = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict pneumonia from a single chest X-ray image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save image with predicted boxes",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="Detection confidence threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    model = load_model()
    cls_threshold = load_threshold()
    tensor = preprocess_image(image_path)

    out = model(tensor, mode="inference")

    prob = float(out["classification"].view(-1)[0].item())
    label = "PNEUMONIA" if prob >= cls_threshold else "NORMAL"
    print(f"Prediction: {label}")
    print(f"Probability: {prob:.4f}")
    print(f"Threshold: {cls_threshold:.4f}")

    if args.output:
        original = np.array(Image.open(image_path).convert("RGB"))
        boxes = out["boxes"].detach().cpu().numpy().tolist()
        rendered = draw_boxes(original, boxes)
        cv2.imwrite(str(Path(args.output)), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        print(f"Saved boxed output: {args.output}")


if __name__ == "__main__":
    main()
