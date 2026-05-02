"""FASTA loading + label-file utilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch


def read_sequences(path: str | Path) -> List[str]:
    """Return non-header lines from a FASTA file, stripped."""
    path = Path(path)
    with path.open("r") as f:
        return [line.strip() for line in f if line and not line.startswith(">")]


def tokenize_fasta(
    path: str | Path,
    tokenizer: Callable[[str], torch.Tensor],
    num_workers: int = 8,
) -> torch.Tensor:
    """Tokenize each non-header sequence in `path` and return a stacked tensor."""
    sequences = read_sequences(path)
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        tokenized = list(ex.map(tokenizer, sequences))
    arr = np.array([t.numpy() for t in tokenized], dtype=np.int64)
    return torch.from_numpy(arr)


def load_labels(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("r") as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    return np.array(labels, dtype=np.int64)
