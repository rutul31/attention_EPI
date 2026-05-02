"""k-mer tokenizer factory. Vocabulary size = 4^k + 1 (last index = null token)."""

from __future__ import annotations

import itertools
from typing import Callable, Dict

import torch

NULL_TOKEN = "null"


def build_vocab(k: int = 6) -> Dict[str, int]:
    bases = ["A", "C", "G", "T"]
    vocab = {"".join(c): idx for idx, c in enumerate(itertools.product(bases, repeat=k))}
    vocab[NULL_TOKEN] = 4 ** k
    return vocab


def create_tokenizer(k: int = 6) -> Callable[[str], torch.Tensor]:
    vocab = build_vocab(k)
    null_id = vocab[NULL_TOKEN]

    def tokenize(sequence: str) -> torch.Tensor:
        seq = sequence.upper()
        ids = [vocab.get(seq[i : i + k], null_id) for i in range(len(seq) - k + 1)]
        return torch.tensor(ids, dtype=torch.long)

    return tokenize
