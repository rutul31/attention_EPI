"""Soft-averaging ensemble across N synthetic checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from epintlm.config import Config
from epintlm.data.dataset import SeqGenDataset
from epintlm.data.dummy_features import DummyGenomicFeatures
from epintlm.eval.ensemble import ensemble_predict
from epintlm.models.epintlm import build_model


def _tiny_cfg() -> Config:
    cfg = Config()
    cfg.run.device = "cpu"
    cfg.data.embeddings_path = "/nonexistent_path"  # skip embedding load (tiny model dim != prod)
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
    cfg.model.gene_data.input_dim = 16
    cfg.model.gene_data.hidden_dim = 32
    return cfg


def _make_loader(cfg: Config, n: int = 32) -> DataLoader:
    enh = torch.randint(0, cfg.model.embedding.vocab_size, (n, 100), dtype=torch.long)
    pro = torch.randint(0, cfg.model.embedding.vocab_size, (n, 80), dtype=torch.long)
    gene = DummyGenomicFeatures(size=n, feat_dim=cfg.model.gene_data.input_dim)
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=n).astype(np.int64)
    ds = SeqGenDataset(enh, pro, gene, labels)
    return DataLoader(ds, batch_size=16, shuffle=False)


def _save_random_checkpoint(cfg: Config, path: Path, seed: int) -> None:
    torch.manual_seed(seed)
    model = build_model(cfg.model, embeddings_path=None)
    model.eval()
    # Force lazy head to materialize so the checkpoint is loadable.
    B = 2
    enh = torch.randint(0, cfg.model.embedding.vocab_size, (B, 100))
    pro = torch.randint(0, cfg.model.embedding.vocab_size, (B, 80))
    gene = torch.zeros(B, cfg.model.gene_data.input_dim)
    with torch.no_grad():
        model(enh, pro, gene)
    torch.save({"model_state_dict": model.state_dict()}, path)


def test_ensemble_averages_multiple_checkpoints(tmp_path: Path):
    cfg = _tiny_cfg()

    ckpts = []
    for seed in (1, 2, 3):
        p = tmp_path / f"member_{seed}.pt"
        _save_random_checkpoint(cfg, p, seed=seed)
        ckpts.append(p)

    loader = _make_loader(cfg)
    metrics = ensemble_predict(cfg, ckpts, loader, tmp_path / "preds", label="cellX")

    assert metrics["n_members"] == 3
    assert (tmp_path / "preds" / "cellX_ensemble_y_true.npy").exists()
    assert (tmp_path / "preds" / "cellX_ensemble_y_score.npy").exists()
    # Sanity: ensemble metrics are within plausible ranges.
    for k in ("ensemble_auc", "ensemble_aupr", "ensemble_f1", "ensemble_accuracy"):
        assert 0.0 <= metrics[k] <= 1.0
    # Member std should be ≥ 0.
    assert metrics["member_auc_std"] >= 0.0
