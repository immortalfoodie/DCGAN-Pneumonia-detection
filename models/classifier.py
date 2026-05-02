"""Classification head for pneumonia prediction."""

from __future__ import annotations

from typing import Dict

from torch import Tensor, nn


class ClassificationHead(nn.Module):
    """Global pooling MLP classifier over semantically rich P5 features."""

    def __init__(self, in_channels: int = 256, hidden_dim: int = 512, dropout: float = 0.5) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, p5: Tensor) -> Dict[str, Tensor]:
        x = self.gap(p5).flatten(1)
        features = self.drop(self.relu(self.fc1(x)))
        prob = self.sigmoid(self.fc2(features))
        return {"prob": prob, "features": features, "cam_target": p5}
