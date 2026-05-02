"""SeqGenDataset — combines tokenized enhancer/promoter ids, gene features, labels."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SeqGenDataset(Dataset):
    def __init__(self, enhancer_ids, promoter_ids, gene_data_all, labels):
        self.enhancer_ids = enhancer_ids
        self.promoter_ids = promoter_ids
        self.gene_data_all = gene_data_all
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        enh = self.enhancer_ids[idx]
        pro = self.promoter_ids[idx]
        gene = self.gene_data_all[idx]
        label = self.labels[idx]
        return enh.squeeze(), pro.squeeze(), gene, label


def register_safe_globals() -> None:
    """Register SeqGenDataset with torch.serialization for weights_only loading."""
    try:
        torch.serialization.add_safe_globals([SeqGenDataset])
    except Exception:
        pass
