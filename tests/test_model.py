import torch

from epintlm.config import Config
from epintlm.models.epintlm import build_model


def test_model_forward_smoke():
    """Forward pass on tiny dummy batch — verifies all blocks compose without shape errors."""
    cfg = Config()
    # Tiny model variant for fast test (still satisfies dim invariants: gru*2 == attention.embed_dim)
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

    model = build_model(cfg.model, embeddings_path=None)
    model.eval()

    B, T_enh, T_pro = 2, 100, 80
    enh = torch.randint(0, cfg.model.embedding.vocab_size, (B, T_enh))
    pro = torch.randint(0, cfg.model.embedding.vocab_size, (B, T_pro))
    gene = torch.zeros(B, cfg.model.gene_data.input_dim)

    with torch.no_grad():
        out, attn = model(enh, pro, gene)

    assert out.shape == (B, 1)
    assert (out >= 0).all() and (out <= 1).all()
    expected_keys = {"self_enh", "self_pro", "cross_enh", "cross_pro",
                     "pre_cross_enh", "pre_cross_pro", "gen"}
    assert set(attn.keys()) == expected_keys


def test_ablation_toggles_compose():
    """Toggling cross-attn / residual off should still produce valid output."""
    cfg = Config()
    cfg.model.embedding.dim = 64
    cfg.model.enhancer_encoder.conv_out_channels = 32
    cfg.model.enhancer_encoder.conv_kernel = 8
    cfg.model.enhancer_encoder.pool_kernel = 4
    cfg.model.enhancer_encoder.pool_stride = 4
    cfg.model.promoter_encoder.conv_out_channels = 32
    cfg.model.promoter_encoder.conv_kernel = 8
    cfg.model.promoter_encoder.pool_kernel = 4
    cfg.model.promoter_encoder.pool_stride = 4
    cfg.model.gru.hidden_size = 16
    cfg.model.attention.embed_dim = 32
    cfg.model.attention.num_heads = 4
    cfg.model.attention.use_cross_attn = False
    cfg.model.attention.use_residual = False
    cfg.model.gene_data.input_dim = 16
    cfg.model.gene_data.hidden_dim = 32

    model = build_model(cfg.model, embeddings_path=None)
    model.eval()
    B = 2
    enh = torch.randint(0, cfg.model.embedding.vocab_size, (B, 80))
    pro = torch.randint(0, cfg.model.embedding.vocab_size, (B, 60))
    gene = torch.zeros(B, 16)
    with torch.no_grad():
        out, _ = model(enh, pro, gene)
    assert out.shape == (B, 1)
