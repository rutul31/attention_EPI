"""Pre-norm multi-head attention block with residual dropout."""

from __future__ import annotations

import torch
from torch import nn


class ImprovedAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.05):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, need_weights: bool = True):
        q_n = self.norm(q)
        k_n = self.norm(k)
        v_n = self.norm(v)
        out, w = self.attn(q_n, k_n, v_n, need_weights=need_weights, average_attn_weights=False)
        return self.dropout(out), w
