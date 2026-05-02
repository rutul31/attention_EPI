"""Conv + BiGRU encoder for enhancer/promoter token sequences."""

from __future__ import annotations

import torch
from torch import nn

from ..config import EncoderConfig, GRUConfig


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, cfg: EncoderConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, cfg.conv_out_channels, kernel_size=cfg.conv_kernel),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=cfg.pool_kernel, stride=cfg.pool_stride),
            nn.BatchNorm1d(cfg.conv_out_channels),
            nn.Dropout(p=cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SequenceEncoder(nn.Module):
    """Conv block (B, C_in, T) → (B, C_out, T') then BiGRU over time → (T'', B, 2*hidden).

    Forward returns the post-GRU representation. Use `forward_conv` / `forward_gru`
    separately when you need to interpose attention between the conv and GRU stages.
    """

    def __init__(self, embed_dim: int, encoder_cfg: EncoderConfig, gru_cfg: GRUConfig):
        super().__init__()
        self.conv = ConvBlock(in_channels=embed_dim, cfg=encoder_cfg)
        self.gru = nn.GRU(
            input_size=encoder_cfg.conv_out_channels,
            hidden_size=gru_cfg.hidden_size,
            num_layers=gru_cfg.num_layers,
            bidirectional=gru_cfg.bidirectional,
        )

    def forward_conv(self, embedded: torch.Tensor) -> torch.Tensor:
        """(B, T, embed_dim) → (T', B, C_out) — time-major, ready for attention or GRU."""
        conv_out = self.conv(embedded.permute(0, 2, 1))   # (B, C_out, T')
        return conv_out.permute(2, 0, 1)                  # (T', B, C_out)

    def forward_gru(self, conv_out: torch.Tensor) -> torch.Tensor:
        """(T', B, C_out) → (T', B, 2*hidden)."""
        gru_out, _ = self.gru(conv_out)
        return gru_out

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        return self.forward_gru(self.forward_conv(embedded))
