"""Final classifier head: BatchNorm + Dropout → flatten → MLP → sigmoid."""

from __future__ import annotations

import torch
from torch import nn

from ..config import HeadConfig


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, head_cfg: HeadConfig):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, head_cfg.hidden_dim),
            nn.BatchNorm1d(head_cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=head_cfg.dropout),
            nn.Linear(head_cfg.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
