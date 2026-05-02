"""Cross-attention placement variants: post_gru (default), pre_gru, dual."""

import pytest
import torch

from epintlm.config import Config
from epintlm.models.epintlm import build_model


def _tiny_cfg() -> Config:
    cfg = Config()
    cfg.model.embedding.dim = 64
    cfg.model.embedding.vocab_size = 4097
    cfg.model.enhancer_encoder.conv_out_channels = 32
    cfg.model.enhancer_encoder.conv_kernel = 8
    cfg.model.enhancer_encoder.pool_kernel = 4
    cfg.model.enhancer_encoder.pool_stride = 4
    cfg.model.promoter_encoder.conv_out_channels = 32
    cfg.model.promoter_encoder.conv_kernel = 8
    cfg.model.promoter_encoder.pool_kernel = 4
    cfg.model.promoter_encoder.pool_stride = 4
    cfg.model.gru.hidden_size = 16  # 2*16=32
    cfg.model.attention.embed_dim = 32
    cfg.model.attention.num_heads = 4
    cfg.model.gene_data.input_dim = 16
    cfg.model.gene_data.hidden_dim = 32
    return cfg


@pytest.mark.parametrize("placement", ["post_gru", "pre_gru", "dual"])
def test_placement_variant_forward(placement):
    cfg = _tiny_cfg()
    cfg.model.attention.placement = placement
    model = build_model(cfg.model, embeddings_path=None)
    model.eval()

    B = 2
    enh = torch.randint(0, cfg.model.embedding.vocab_size, (B, 100))
    pro = torch.randint(0, cfg.model.embedding.vocab_size, (B, 80))
    gene = torch.zeros(B, cfg.model.gene_data.input_dim)
    with torch.no_grad():
        out, attn = model(enh, pro, gene)
    assert out.shape == (B, 1)

    # pre_gru / dual must produce non-None pre_cross weights; post_gru must not.
    if placement in ("pre_gru", "dual"):
        assert attn["pre_cross_enh"] is not None
        assert attn["pre_cross_pro"] is not None
    else:
        assert attn["pre_cross_enh"] is None

    # post_gru / dual must produce non-None post-GRU cross weights; pre_gru must not.
    if placement in ("post_gru", "dual"):
        assert attn["cross_enh"] is not None
    else:
        assert attn["cross_enh"] is None


def test_unknown_placement_rejected():
    cfg = _tiny_cfg()
    cfg.model.attention.placement = "no_such_placement"
    with pytest.raises(ValueError, match="Unknown attention.placement"):
        build_model(cfg.model, embeddings_path=None)
