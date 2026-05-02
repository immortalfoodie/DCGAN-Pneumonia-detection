"""Training script for unified classification and detection model."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from models.full_model import PneumoniaDetectionModel
from training.losses import CombinedLoss
from utils.augmentation import get_train_transforms, get_val_transforms
from utils.dataset import ChestXRayDataset, collate_fn, get_balanced_loader
from utils.metrics import compute_classification_metrics, save_training_curves


def setup_logger() -> logging.Logger:
    """Setup structured training logger."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("train_classifier")


def select_detection_targets(pred_boxes: Tensor, target_boxes: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """Create a lightweight pair set for CIoU when labels are available."""
    pred_list: List[Tensor] = []
    target_list: List[Tensor] = []

    for i, boxes in enumerate(target_boxes):
        if boxes.numel() == 0:
            continue
        pred = pred_boxes[i]
        pred_list.append(pred[:1])
        target_list.append(boxes[:1].to(pred.device))

    if not pred_list:
        empty = pred_boxes.new_zeros((0, 4))
        return empty, empty

    return torch.cat(pred_list, dim=0), torch.cat(target_list, dim=0)


def run_epoch(
    model: PneumoniaDetectionModel,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: Adam | None,
    scaler: GradScaler,
    use_amp: bool,
    train: bool,
    use_gan_balance: bool,
    max_batches: int | None = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run one training or validation epoch."""
    model.train(mode=train)
    epoch_losses: List[float] = []
    all_true: List[float] = []
    all_prob: List[float] = []

    for batch_idx, (images, labels, boxes) in enumerate(tqdm(loader, leave=False), start=1):
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with autocast(enabled=use_amp):
                outputs = model(
                    images,
                    mode="train" if train else "inference",
                    labels=labels,
                    use_gan_balance=use_gan_balance and train,
                )
                pred_boxes = torch.zeros((0, 4), device=images.device)
                tgt_boxes = torch.zeros((0, 4), device=images.device)
                loss, _, _ = criterion(outputs["classification"].squeeze(1), labels, box_pred=pred_boxes, box_target=tgt_boxes)

        if train and optimizer is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        epoch_losses.append(float(loss.item()))
        all_true.extend(labels.detach().cpu().numpy().tolist())
        all_prob.extend(outputs["classification"].detach().cpu().numpy().reshape(-1).tolist())
        if max_batches is not None and batch_idx >= max_batches:
            break

    return float(np.mean(epoch_losses)), np.array(all_true), np.array(all_prob)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Select decision threshold that maximizes F1 on validation data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def train(
    resume: bool = False,
    epochs: int = Config.EPOCHS,
    batch_size: int = Config.BATCH_SIZE,
    lr: float = Config.LR,
    no_gan: bool = False,
    image_size: int = Config.IMAGE_SIZE,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> None:
    """Train classifier + detector model with mixed precision and early stopping."""
    logger = setup_logger()

    train_ds = ChestXRayDataset(mode="train", transform=get_train_transforms(image_size))
    val_ds = ChestXRayDataset(mode="val", transform=get_val_transforms(image_size))

    train_loader = get_balanced_loader(
        train_ds,
        batch_size=batch_size,
        use_gan_samples=generator_ckpt.exists() and not no_gan,
        synthetic_dir=str(Config.path(Config.GENERATED_DIR)),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = PneumoniaDetectionModel().to(Config.DEVICE)

    generator_ckpt = Config.path(Config.CHECKPOINT_DIR) / "generator_best.pth"
    use_gan_balance = False
    if generator_ckpt.exists():
        model.load_generator_checkpoint(str(generator_ckpt))
        use_gan_balance = not no_gan
        logger.info("Loaded pretrained generator from %s", generator_ckpt)

    criterion = CombinedLoss(lambda_det=Config.LAMBDA_DET, focal_alpha=Config.FOCAL_ALPHA, focal_gamma=Config.FOCAL_GAMMA)
    optimizer = Adam(
        list(model.backbone.parameters()) + list(model.classifier.parameters()) + list(model.detector.parameters()),
        lr=lr,
        weight_decay=Config.WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    scaler = GradScaler(enabled=torch.cuda.is_available())
    use_amp = torch.cuda.is_available()

    start_epoch = 1
    best_val_auc = 0.0
    patience_counter = 0

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    checkpoint_path = Config.path(Config.CHECKPOINT_DIR) / "classifier_last.pth"
    if resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        best_val_auc = ckpt["best_val_auc"]
        patience_counter = ckpt["patience_counter"]
        logger.info("Resumed training from epoch %d", ckpt["epoch"])

    for epoch in range(start_epoch, epochs + 1):
        logger.info("Epoch %d/%d", epoch, epochs)

        train_loss, train_true, train_prob = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            use_amp,
            train=True,
            use_gan_balance=use_gan_balance,
            max_batches=max_train_batches,
        )
        val_loss, val_true, val_prob = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            scaler,
            use_amp,
            train=False,
            use_gan_balance=False,
            max_batches=max_val_batches,
        )

        train_metrics = compute_classification_metrics(train_true, train_prob)
        val_metrics = compute_classification_metrics(val_true, val_prob)
        best_threshold = find_best_threshold(val_true, val_prob)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        logger.info(
            "train_loss=%.4f val_loss=%.4f train_acc=%.4f val_acc=%.4f val_auc=%.4f val_f1=%.4f",
            train_loss,
            val_loss,
            train_metrics["accuracy"],
            val_metrics["accuracy"],
            val_metrics["auc"],
            val_metrics["f1"],
        )
        logger.info("calibrated_threshold=%.4f", best_threshold)

        scheduler.step(val_metrics["auc"])

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "threshold": best_threshold,
                },
                Config.path(Config.CHECKPOINT_DIR) / "best_model.pth",
            )
        else:
            patience_counter += 1

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
                "best_val_auc": best_val_auc,
                "patience_counter": patience_counter,
            },
            checkpoint_path,
        )

        metrics_payload = {
            "accuracy": val_metrics["accuracy"],
            "auc": val_metrics["auc"],
            "f1": val_metrics["f1"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "confusion_matrix": val_metrics["confusion_matrix"],
            "roc": val_metrics["roc"],
            "threshold": best_threshold,
            "history": history,
        }
        with open(Config.path(Config.CHECKPOINT_DIR) / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

        save_training_curves(history, Config.path(Config.CHECKPOINT_DIR) / "training_curves.png")

        if patience_counter >= Config.PATIENCE:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    logger.info("Training complete. Best val AUC: %.4f", best_val_auc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train unified pneumonia detection model")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LR)
    parser.add_argument("--no_gan", action="store_true")
    parser.add_argument("--image_size", type=int, default=Config.IMAGE_SIZE)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        resume=args.resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        no_gan=args.no_gan,
        image_size=args.image_size,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
