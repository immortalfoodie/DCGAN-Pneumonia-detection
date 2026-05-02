"""DCGAN components with self-attention and spectral normalization."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils import spectral_norm


class SelfAttention(nn.Module):
    """SAGAN-style self-attention block."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        inter_channels = max(1, in_channels // 8)
        self.query_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        value = self.value_conv(x).view(batch_size, channels, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, channels, height, width)
        return self.gamma * out + x


class SpectralNormConv2d(nn.Module):
    """Thin wrapper around spectral-normalized Conv2d."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(*args, **kwargs))

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Generator(nn.Module):
    """DCGAN generator for grayscale 128x128 X-ray synthesis."""

    def __init__(self, latent_dim: int = 100, img_channels: int = 1, base_features: int = 64) -> None:
        super().__init__()
        self.project = spectral_norm(nn.Linear(latent_dim, 4 * 4 * (base_features * 8)))
        self.block1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(base_features * 8, base_features * 4, 4, 2, 1)),
            nn.BatchNorm2d(base_features * 4),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(base_features * 4, base_features * 2, 4, 2, 1)),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(base_features * 2, base_features, 4, 2, 1)),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
        )
        self.attention = SelfAttention(base_features)
        self.block4 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(base_features, base_features // 2, 4, 2, 1)),
            nn.LayerNorm([base_features // 2, 64, 64]),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(base_features // 2, img_channels, 4, 2, 1)),
            nn.Tanh(),
        )

    def forward(self, z: Tensor) -> Tensor:
        x = self.project(z).view(z.size(0), 512, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.attention(x)
        x = self.block4(x)
        return self.out(x)


class Discriminator(nn.Module):
    """Spectral-normalized discriminator without sigmoid for hinge loss."""

    def __init__(self, img_channels: int = 1, base_features: int = 64) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, base_features, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block2 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_features, base_features * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attention = SelfAttention(base_features * 2)
        self.block3 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_features * 2, base_features * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block4 = nn.Sequential(
            spectral_norm(nn.Conv2d(base_features * 4, base_features * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Conv2d(base_features * 8, 1, 4, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.attention(x)
        x = self.block3(x)
        x = self.block4(x)
        logits = self.classifier(x)
        return logits.view(logits.size(0), -1).mean(dim=1)


def compute_gradient_penalty(discriminator: Discriminator, real: Tensor, fake: Tensor) -> Tensor:
    """Compute gradient penalty for regularized discriminator training."""
    alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    preds = discriminator(interpolated)
    grads = torch.autograd.grad(
        outputs=preds,
        inputs=interpolated,
        grad_outputs=torch.ones_like(preds),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(grads.size(0), -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


def build_double_sgan(latent_dim: int = 100) -> Tuple[Generator, Discriminator, Discriminator]:
    """Factory returning G, D1 and D2 instances."""
    generator = Generator(latent_dim=latent_dim)
    d1 = Discriminator()
    d2 = Discriminator()
    return generator, d1, d2
