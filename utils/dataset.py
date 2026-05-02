"""Dataset and dataloader helpers for chest X-ray training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import Config


@dataclass
class SampleItem:
    """Single sample descriptor."""

    image_path: Path
    label: int
    boxes: Optional[np.ndarray]


class ChestXRayDataset(Dataset):
    """Dataset reader for chest_xray/{mode}/{NORMAL|PNEUMONIA}."""

    def __init__(
        self,
        root_dir: Optional[str] = None,
        mode: str = "train",
        transform: Any = None,
        include_synthetic: bool = False,
        synthetic_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.split = mode
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir else Config.resolve_data_dir()
        self.samples: List[SampleItem] = []
        self.rsna_annotations = {}
        self._scan_dataset(include_synthetic=include_synthetic, synthetic_dir=synthetic_dir)

    def _load_rsna_annotations(self) -> Dict[str, np.ndarray]:
        rsna_csv = Config.PROJECT_ROOT / "data" / "rsna" / "stage_2_train_labels.csv"
        if not rsna_csv.exists():
            return {}

        df = pd.read_csv(rsna_csv)
        grouped: Dict[str, np.ndarray] = {}
        for patient_id, group in df.groupby("patientId"):
            valid = group[group["Target"] == 1]
            boxes = []
            for _, row in valid.iterrows():
                x1 = float(row["x"])
                y1 = float(row["y"])
                x2 = x1 + float(row["width"])
                y2 = y1 + float(row["height"])
                boxes.append([x1, y1, x2, y2])
            grouped[patient_id] = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        return grouped

    def _scan_dataset(self, include_synthetic: bool = False, synthetic_dir: Optional[str] = None) -> None:
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset split not found: {split_dir}")

        class_map = {"NORMAL": 0, "PNEUMONIA": 1}
        for cls_name, label in class_map.items():
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.glob("*.jpeg"):
                patient_id = img_path.stem
                boxes = self.rsna_annotations.get(patient_id)
                self.samples.append(SampleItem(img_path, label, boxes))
            for img_path in cls_dir.glob("*.jpg"):
                patient_id = img_path.stem
                boxes = self.rsna_annotations.get(patient_id)
                self.samples.append(SampleItem(img_path, label, boxes))
            for img_path in cls_dir.glob("*.png"):
                patient_id = img_path.stem
                boxes = self.rsna_annotations.get(patient_id)
                self.samples.append(SampleItem(img_path, label, boxes))
        if include_synthetic and synthetic_dir:
            synth_root = Path(synthetic_dir)
            for img_path in list(synth_root.glob("*.png")) + list(synth_root.glob("*.jpg")):
                self.samples.append(SampleItem(img_path, 1, None))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {sample.image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        label = int(sample.label)
        boxes = sample.boxes
        if boxes is None:
            boxes = np.zeros((0, 4), dtype=np.float32)
        return {"image": image, "label": label, "path": str(sample.image_path), "boxes": torch.tensor(boxes, dtype=torch.float32)}


def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[Tensor, Tensor, List[Tensor]]:
    """Collate function supporting variable number of boxes per image."""
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([float(item["label"]) for item in batch], dtype=torch.float32)
    boxes = [item["boxes"] for item in batch]
    return images, labels, boxes


def get_balanced_loader(
    dataset: ChestXRayDataset,
    batch_size: int,
    num_workers: int = 0,
    use_gan_samples: bool = False,
    synthetic_dir: Optional[str] = None,
) -> DataLoader:
    """Create balanced dataloader with optional GAN synthetic pneumonia samples."""
    if use_gan_samples and synthetic_dir:
        augmented_dataset = ChestXRayDataset(
            root_dir=str(dataset.root_dir),
            mode=dataset.split,
            transform=dataset.transform,
            include_synthetic=True,
            synthetic_dir=synthetic_dir,
        )
        dataset = augmented_dataset

    labels = [int(item.label) for item in dataset.samples]
    class_counts = np.bincount(labels, minlength=2)
    class_weights = np.array([1.0 / max(class_counts[i], 1) for i in range(2)], dtype=np.float32)
    sample_weights = np.array([class_weights[label] for label in labels], dtype=np.float32)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
