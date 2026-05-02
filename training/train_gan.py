"""Training script for Double-SGAN chest X-ray augmentation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from config import Config
from models.dcgan import Discriminator, Generator, build_double_sgan, compute_gradient_penalty
from training.losses import GANHingeLoss

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception:
    FrechetInceptionDistance = None  # type: ignore[assignment]


class MinorityClassDataset(Dataset):
    """Loads images from minority class directory for GAN training."""

    def __init__(self, data_root: Path) -> None:
        self.paths = list(data_root.glob("*.jpeg")) + list(data_root.glob("*.jpg")) + list(data_root.glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"No images found in {data_root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tensor:
        from PIL import Image

        img = Image.open(self.paths[idx]).convert("L").resize((Config.GAN_IMAGE_SIZE, Config.GAN_IMAGE_SIZE))
        tensor = torch.from_numpy(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).numpy()).float()
        tensor = tensor.view(Config.GAN_IMAGE_SIZE, Config.GAN_IMAGE_SIZE) / 255.0
        tensor = tensor.unsqueeze(0)
        return tensor * 2.0 - 1.0


def find_minority_class_dir() -> Path:
    """Detect minority class from train split and return its directory."""
    train_root = Config.resolve_data_dir() / "train"
    normal_count = len(list((train_root / "NORMAL").glob("*.*")))
    pneumonia_count = len(list((train_root / "PNEUMONIA").glob("*.*")))
    minority = "NORMAL" if normal_count < pneumonia_count else "PNEUMONIA"
    return train_root / minority


def setup_logger() -> logging.Logger:
    """Configure training logger."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("train_gan")


def compute_fid(generator: Generator, real_batch: Tensor, device: str) -> Optional[float]:
    """Compute FID score if torchmetrics FID is available."""
    if FrechetInceptionDistance is None:
        return None

    try:
        fid = FrechetInceptionDistance(feature=64).to(device)
    except (ModuleNotFoundError, ImportError):
        return None
    with torch.no_grad():
        z = torch.randn(real_batch.size(0), Config.LATENT_DIM, device=device)
        fake = generator(z)
        fake = ((fake + 1.0) * 127.5).clamp(0, 255).byte().repeat(1, 3, 1, 1)
        real = ((real_batch + 1.0) * 127.5).clamp(0, 255).byte().repeat(1, 3, 1, 1)

    fid.update(real, real=True)
    fid.update(fake, real=False)
    return float(fid.compute().item())


def save_generated_grid(generator: Generator, epoch: int, device: str) -> None:
    """Save generated image samples every few epochs using memory-safe chunks."""
    was_training = generator.training
    generator.eval()
    generated: list[Tensor] = []
    total_samples = 16
    chunk_size = 4

    with torch.no_grad():
        for start in range(0, total_samples, chunk_size):
            current_chunk = min(chunk_size, total_samples - start)
            z = torch.randn(current_chunk, Config.LATENT_DIM, device=device)
            fake_chunk = generator(z)
            generated.append(fake_chunk.detach().cpu())

    fake = torch.cat(generated, dim=0)
    fake = (fake + 1.0) / 2.0
    grid = make_grid(fake, nrow=4)
    save_image(grid, Config.path(Config.GENERATED_DIR) / f"epoch_{epoch}.png")

    if was_training:
        generator.train()


def train(
    resume: bool = False,
    batch_size: int | None = None,
    epochs: int | None = None,
    max_batches: int | None = None,
) -> None:
    """Main GAN training loop."""
    logger = setup_logger()
    device = Config.DEVICE

    dataset = MinorityClassDataset(find_minority_class_dir())
    effective_batch_size = batch_size if batch_size is not None else Config.BATCH_SIZE
    effective_epochs = epochs if epochs is not None else Config.GAN_EPOCHS
    loader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True, drop_last=True)

    generator, d1, d2 = build_double_sgan(latent_dim=Config.LATENT_DIM)
    generator, d1, d2 = generator.to(device), d1.to(device), d2.to(device)

    g_opt = Adam(generator.parameters(), lr=Config.GAN_LR, betas=Config.GAN_BETAS)
    d1_opt = Adam(d1.parameters(), lr=Config.GAN_LR, betas=Config.GAN_BETAS)
    d2_opt = Adam(d2.parameters(), lr=Config.GAN_LR, betas=Config.GAN_BETAS)

    start_epoch = 1
    best_fid = float("inf")
    best_generator_path = Config.path(Config.CHECKPOINT_DIR) / "generator_best.pth"
    checkpoint_path = Config.path(Config.CHECKPOINT_DIR) / "gan_last.pth"

    if resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        d1.load_state_dict(ckpt["d1"])
        d2.load_state_dict(ckpt["d2"])
        g_opt.load_state_dict(ckpt["g_opt"])
        d1_opt.load_state_dict(ckpt["d1_opt"])
        d2_opt.load_state_dict(ckpt["d2_opt"])
        start_epoch = ckpt["epoch"] + 1
        best_fid = ckpt.get("best_fid", best_fid)
        logger.info("Resumed GAN training from epoch %d", ckpt["epoch"])

    # Ensure downstream stages can always find a generator checkpoint, even
    # when FID metric is unavailable in the current environment.
    if not best_generator_path.exists():
        torch.save({"generator": generator.state_dict(), "initialized_only": True}, best_generator_path)
        logger.info("Initialized generator_best.pth for compatibility.")

    for epoch in range(start_epoch, effective_epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{effective_epochs}")
        for batch_idx, real in enumerate(pbar):
            real = real.to(device)
            z = torch.randn(real.size(0), Config.LATENT_DIM, device=device)
            fake = generator(z).detach()

            for disc, opt in [(d1, d1_opt), (d2, d2_opt)]:
                opt.zero_grad(set_to_none=True)
                real_scores = disc(real)
                fake_scores = disc(fake)
                d_loss = GANHingeLoss.discriminator_loss(real_scores, fake_scores)
                gp = compute_gradient_penalty(disc, real, fake)
                total_d_loss = d_loss + 10.0 * gp
                total_d_loss.backward()
                opt.step()

            g_opt.zero_grad(set_to_none=True)
            z = torch.randn(real.size(0), Config.LATENT_DIM, device=device)
            fake = generator(z)
            g_loss = GANHingeLoss.generator_loss(d1(fake), d2(fake)) / 2.0
            g_loss.backward()
            g_opt.step()

            pbar.set_postfix({"G_loss": f"{g_loss.item():.4f}", "D1_loss": f"{d1(fake.detach()).mean().item():.4f}", "D2_loss": f"{d2(fake.detach()).mean().item():.4f}"})

            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

        if epoch % 5 == 0:
            try:
                save_generated_grid(generator, epoch, device)
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" in str(exc).lower() or isinstance(exc, torch.OutOfMemoryError):
                    logger.warning("Skipping generated grid at epoch %d due OOM. Training will continue.", epoch)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise

        current_fid = None
        if epoch % 10 == 0:
            current_fid = compute_fid(generator, real, device)
            if current_fid is not None:
                logger.info("Epoch %d | FID: %.4f", epoch, current_fid)
                if current_fid < best_fid:
                    best_fid = current_fid
                    torch.save({"generator": generator.state_dict(), "initialized_only": False}, best_generator_path)

        torch.save(
            {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "d1": d1.state_dict(),
                "d2": d2.state_dict(),
                "g_opt": g_opt.state_dict(),
                "d1_opt": d1_opt.state_dict(),
                "d2_opt": d2_opt.state_dict(),
                "best_fid": best_fid,
            },
            checkpoint_path,
        )

    # Keep a fresh generator-only checkpoint for inference/demo consumers
    # even when FID dependencies are unavailable.
    torch.save({"generator": generator.state_dict(), "initialized_only": False}, best_generator_path)
    logger.info("GAN training complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Double-SGAN for chest X-ray augmentation")
    parser.add_argument("--resume", action="store_true", help="Resume from last GAN checkpoint")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size (reduces GPU memory usage)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of GAN training epochs")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit batches per epoch for faster iterative GAN training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(resume=args.resume, batch_size=args.batch_size, epochs=args.epochs, max_batches=args.max_batches)
