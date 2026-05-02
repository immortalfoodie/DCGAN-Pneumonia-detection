"""Central configuration for the PneumoAI system."""

from pathlib import Path

import torch


class Config:
    """Application-wide paths and hyperparameters."""

    PROJECT_ROOT: Path = Path(__file__).resolve().parent

    # Paths - aligned with original repository layout
    DATA_DIR: str = "chest_xray"
    CHECKPOINT_DIR: str = "checkpoints"
    GENERATED_DIR: str = "generated_images"
    ASSETS_DIR: str = "Assests"

    # Input
    IMAGE_SIZE: int = 512
    GAN_IMAGE_SIZE: int = 128
    IN_CHANNELS: int = 1
    MODEL_CHANNELS: int = 3

    # GAN
    LATENT_DIM: int = 100
    GAN_LR: float = 0.0002
    GAN_BETAS: tuple[float, float] = (0.5, 0.999)
    GAN_EPOCHS: int = 100
    NUM_DISCRIMINATORS: int = 2

    # Classifier
    BATCH_SIZE: int = 16
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    EPOCHS: int = 50
    PATIENCE: int = 10
    NUM_CLASSES: int = 1
    FPN_CHANNELS: int = 256
    DROPOUT: float = 0.5

    # Loss
    FOCAL_ALPHA: float = 0.25
    FOCAL_GAMMA: float = 2.0
    LAMBDA_DET: float = 1.0

    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Baseline from original notebooks
    REPO_DENSENET_ACC: float = 87.18
    REPO_CNN_ACC: float = 91.98

    @classmethod
    def path(cls, rel: str) -> Path:
        """Resolve project-relative path from a config string field."""
        return cls.PROJECT_ROOT / rel

    @classmethod
    def resolve_data_dir(cls) -> Path:
        """Resolve chest_xray root while preserving legacy compatibility."""
        default = cls.path(cls.DATA_DIR)
        nested = cls.PROJECT_ROOT / "data" / cls.DATA_DIR
        if default.exists():
            return default
        return nested


for directory in [Config.path(Config.CHECKPOINT_DIR), Config.path(Config.GENERATED_DIR)]:
    directory.mkdir(parents=True, exist_ok=True)
