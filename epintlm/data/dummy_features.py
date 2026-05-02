"""Zero-tensor placeholder for genomic features when real chromatin data is unavailable."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class DummyGenomicFeatures(Dataset):
    """Returns torch.zeros(feat_dim) for every index. Length = `size`."""

    def __init__(self, size: int, feat_dim: int = 55):
        self.size = size
        self.feat_dim = feat_dim

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx) -> torch.Tensor:
        return torch.zeros(self.feat_dim)
