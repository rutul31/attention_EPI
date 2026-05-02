"""EPIModel — config-driven enhancer-promoter classifier with cross-attention.

Composes:
  Embedding → SequenceEncoder (Conv + BiGRU) [enhancer]
            → SequenceEncoder (Conv + BiGRU) [promoter]
  Self-attn (enh, pro), cross-attn (enh↔pro), gene-data attn
  Concat over time → BatchNorm → Dropout → flatten → MLP → sigmoid
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

from ..config import ModelConfig
from .attention import ImprovedAttention
from .encoders import SequenceEncoder
from .heads import ClassificationHead


def _attention_dim(cfg: ModelConfig) -> int:
    """Time-major encoder output channels = 2*hidden if bidirectional."""
    return cfg.gru.hidden_size * (2 if cfg.gru.bidirectional else 1)


def _expected_time_steps(seq_len: int, kmer_size: int, encoder) -> int:
    """Compute T' after k-mer tokenization → conv (kernel) → maxpool (stride)."""
    after_kmer = seq_len - kmer_size + 1
    after_conv = after_kmer - encoder.conv_kernel + 1
    return after_conv // encoder.pool_stride


class EPIModel(nn.Module):
    def __init__(self, cfg: ModelConfig, embeddings_path: Optional[str | Path] = None):
        super().__init__()
        self.cfg = cfg
        self._fc_input_dim: Optional[int] = None

        # Embeddings — load pretrained if path given, else random init
        emb = cfg.embedding
        self.embedding_en = nn.Embedding(emb.vocab_size, emb.dim)
        self.embedding_pr = nn.Embedding(emb.vocab_size, emb.dim)

        if embeddings_path is not None and Path(embeddings_path).exists():
            arr = np.load(embeddings_path)
            mat = torch.tensor(arr, dtype=torch.float32)
            if mat.shape[0] == emb.vocab_size - 1:
                # Pad with zero row for null token if file is one short (4096 → 4097)
                mat = torch.cat([mat, torch.zeros(1, mat.shape[1])], dim=0)
            assert mat.shape == (emb.vocab_size, emb.dim), (
                f"Embedding shape {mat.shape} != ({emb.vocab_size}, {emb.dim})"
            )
            self.embedding_en.weight = nn.Parameter(mat.clone(), requires_grad=False)
            self.embedding_pr.weight = nn.Parameter(mat.clone(), requires_grad=False)

        # Sequence encoders
        self.enhancer_encoder = SequenceEncoder(emb.dim, cfg.enhancer_encoder, cfg.gru)
        self.promoter_encoder = SequenceEncoder(emb.dim, cfg.promoter_encoder, cfg.gru)

        attn_dim = _attention_dim(cfg)
        if attn_dim != cfg.attention.embed_dim:
            raise ValueError(
                f"GRU output dim ({attn_dim}) must match attention.embed_dim ({cfg.attention.embed_dim})"
            )

        if cfg.attention.placement not in ("post_gru", "pre_gru", "dual"):
            raise ValueError(f"Unknown attention.placement: {cfg.attention.placement!r}")

        # Attention modules — cheap to create; we just bypass them in forward when toggled off
        self.self_attn_en = ImprovedAttention(attn_dim, cfg.attention.num_heads, cfg.attention.dropout)
        self.self_attn_pr = ImprovedAttention(attn_dim, cfg.attention.num_heads, cfg.attention.dropout)
        self.cross_attn_en = ImprovedAttention(attn_dim, cfg.attention.num_heads, cfg.attention.dropout)
        self.cross_attn_pr = ImprovedAttention(attn_dim, cfg.attention.num_heads, cfg.attention.dropout)
        self.attn_gen = ImprovedAttention(attn_dim, cfg.attention.num_heads, cfg.attention.dropout)

        # Pre-GRU cross-attention operates on conv-feature dim (may differ from attn_dim).
        if cfg.attention.placement in ("pre_gru", "dual"):
            conv_out_dim = cfg.enhancer_encoder.conv_out_channels
            if conv_out_dim != cfg.promoter_encoder.conv_out_channels:
                raise ValueError("pre_gru placement requires matching enhancer/promoter conv_out_channels")
            self.pre_cross_attn_en = ImprovedAttention(conv_out_dim, cfg.attention.num_heads, cfg.attention.dropout)
            self.pre_cross_attn_pr = ImprovedAttention(conv_out_dim, cfg.attention.num_heads, cfg.attention.dropout)

        # Gene-feature projection: (B, gene_input_dim) → (B, hidden_dim)
        self.gene_linear = nn.Linear(cfg.gene_data.input_dim, cfg.gene_data.hidden_dim)
        if cfg.gene_data.hidden_dim != attn_dim:
            raise ValueError(
                f"gene_data.hidden_dim ({cfg.gene_data.hidden_dim}) must equal attention dim ({attn_dim})"
            )

        # Final layers
        self.layer_norm = nn.LayerNorm(emb.dim)
        self.batchnorm1d = nn.BatchNorm1d(attn_dim)
        self.dropout = nn.Dropout(p=cfg.head.dropout)
        # FC head input size depends on total time steps; built lazily on first forward.
        self.head: Optional[ClassificationHead] = None

        self.criterion = nn.BCELoss()

    def _build_head_if_needed(self, total_time_steps: int) -> None:
        if self.head is None:
            attn_dim = _attention_dim(self.cfg)
            self._fc_input_dim = total_time_steps * attn_dim
            self.head = ClassificationHead(self._fc_input_dim, self.cfg.head)
            self.head.to(self.embedding_en.weight.device)

    def set_embedding_trainable(self, trainable: bool) -> None:
        self.embedding_en.weight.requires_grad = trainable
        self.embedding_pr.weight.requires_grad = trainable

    def forward(
        self,
        enhancer_ids: torch.Tensor,
        promoter_ids: torch.Tensor,
        gene_data: torch.Tensor,
    ):
        attn_cfg = self.cfg.attention

        enh_emb = self.embedding_en(enhancer_ids)
        pro_emb = self.embedding_pr(promoter_ids)

        gene = self.gene_linear(gene_data)
        gene = gene.unsqueeze(0)  # (1, B, hidden)

        # Stage 1: conv. Time-major (T', B, C_out).
        enh_conv = self.enhancer_encoder.forward_conv(enh_emb)
        pro_conv = self.promoter_encoder.forward_conv(pro_emb)

        # Optional pre-GRU cross-attention (placement = "pre_gru" or "dual")
        pre_cross_enh_w = None
        pre_cross_pro_w = None
        if attn_cfg.use_cross_attn and attn_cfg.placement in ("pre_gru", "dual"):
            enh_pre, pre_cross_enh_w = self.pre_cross_attn_en(enh_conv, pro_conv, pro_conv)
            pro_pre, pre_cross_pro_w = self.pre_cross_attn_pr(pro_conv, enh_conv, enh_conv)
            if attn_cfg.use_residual:
                enh_conv = enh_conv + enh_pre
                pro_conv = pro_conv + pro_pre
            else:
                enh_conv, pro_conv = enh_pre, pro_pre

        # Stage 2: GRU. (T', B, 2*hidden).
        enh_out = self.enhancer_encoder.forward_gru(enh_conv)
        pro_out = self.promoter_encoder.forward_gru(pro_conv)

        # Track attention weights only if needed (always returned for visualization)
        self_enh_w = None
        self_pro_w = None
        cross_enh_w = None
        cross_pro_w = None
        gen_w = None

        if attn_cfg.use_self_attn:
            enh_self, self_enh_w = self.self_attn_en(enh_out, enh_out, enh_out)
            pro_self, self_pro_w = self.self_attn_pr(pro_out, pro_out, pro_out)
        else:
            enh_self = torch.zeros_like(enh_out)
            pro_self = torch.zeros_like(pro_out)

        # Post-GRU cross-attention runs unless placement is "pre_gru" (exclusive)
        if attn_cfg.use_cross_attn and attn_cfg.placement in ("post_gru", "dual"):
            enh_cross, cross_enh_w = self.cross_attn_en(enh_out, pro_out, pro_out)
            pro_cross, cross_pro_w = self.cross_attn_pr(pro_out, enh_out, enh_out)
        else:
            enh_cross = torch.zeros_like(enh_out)
            pro_cross = torch.zeros_like(pro_out)

        if attn_cfg.use_residual:
            enh_combined = enh_out + enh_self + enh_cross
            pro_combined = pro_out + pro_self + pro_cross
        else:
            # No residual: replace baseline with attention sum (or just attn output if both flags on)
            enh_combined = enh_self + enh_cross if (attn_cfg.use_self_attn or attn_cfg.use_cross_attn) else enh_out
            pro_combined = pro_self + pro_cross if (attn_cfg.use_self_attn or attn_cfg.use_cross_attn) else pro_out

        gene, gen_w = self.attn_gen(gene, gene, gene)

        stacked = torch.cat((enh_combined, pro_combined), dim=0)
        stacked = torch.cat((stacked, gene), dim=0).permute(1, 2, 0)  # (B, attn_dim, T)

        self._build_head_if_needed(stacked.shape[-1])

        out = self.batchnorm1d(stacked)
        out = self.dropout(out)
        logits = self.head(out.flatten(start_dim=1))

        return torch.sigmoid(logits), {
            "self_enh": self_enh_w,
            "self_pro": self_pro_w,
            "cross_enh": cross_enh_w,
            "cross_pro": cross_pro_w,
            "pre_cross_enh": pre_cross_enh_w,
            "pre_cross_pro": pre_cross_pro_w,
            "gen": gen_w,
        }


def build_model(cfg: ModelConfig, embeddings_path: Optional[str | Path] = None) -> EPIModel:
    return EPIModel(cfg, embeddings_path=embeddings_path)


def load_checkpoint_with_remapping(
    model: EPIModel, checkpoint_path: str | Path, device: torch.device
) -> Dict[str, Any]:
    """Load a checkpoint that may use legacy key names (l1GRU/l2GRU) or 4096-row embeddings."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    remapped = {}
    for k, v in state_dict.items():
        # Legacy GRU names
        if k.startswith("l1GRU."):
            k = k.replace("l1GRU.", "enhancer_encoder.gru.")
        elif k.startswith("l2GRU."):
            k = k.replace("l2GRU.", "promoter_encoder.gru.")
        elif k.startswith("gru1."):
            k = k.replace("gru1.", "enhancer_encoder.gru.")
        elif k.startswith("gru2."):
            k = k.replace("gru2.", "promoter_encoder.gru.")
        # Legacy sequential conv block names
        elif k.startswith("enhancer_sequential."):
            k = k.replace("enhancer_sequential.", "enhancer_encoder.conv.net.")
        elif k.startswith("promoter_sequential."):
            k = k.replace("promoter_sequential.", "promoter_encoder.conv.net.")
        # Legacy attention names
        elif k.startswith("self_attn_cr_en."):
            k = k.replace("self_attn_cr_en.", "cross_attn_en.")
        elif k.startswith("self_attn_cr_pr."):
            k = k.replace("self_attn_cr_pr.", "cross_attn_pr.")
        elif k.startswith("self_attn_gen."):
            k = k.replace("self_attn_gen.", "attn_gen.")
        # Legacy linear/head names
        elif k.startswith("linear_layer."):
            k = k.replace("linear_layer.", "gene_linear.")
        elif k.startswith("fc."):
            # Old model used a single nn.Sequential `fc`; new model has `head.fc`.
            k = k.replace("fc.", "head.fc.")

        # Embedding size mismatch: pad [4096, D] → [4097, D]
        if k in ("embedding_en.weight", "embedding_pr.weight") and v.shape[0] == model.cfg.embedding.vocab_size - 1:
            v = torch.cat([v, torch.zeros(1, v.shape[1])], dim=0)

        remapped[k] = v

    # Ensure head exists before loading head weights (lazy build)
    if any(k.startswith("head.") for k in remapped):
        # Build head with the legacy default total_time_steps (245) if not yet built
        if model.head is None:
            # Best-effort: peek at head.fc.0.weight shape to deduce in_features
            head_w = next((v for k, v in remapped.items() if k.endswith("head.fc.0.weight")), None)
            if head_w is not None:
                in_features = head_w.shape[1]
                attn_dim = model.cfg.gru.hidden_size * (2 if model.cfg.gru.bidirectional else 1)
                total_time_steps = in_features // attn_dim
                model._build_head_if_needed(total_time_steps)

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    return {"missing": list(missing), "unexpected": list(unexpected)}
