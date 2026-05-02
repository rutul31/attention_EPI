"""Tests for the motif-feature extraction tool."""

from __future__ import annotations

from pathlib import Path

import torch

from epintlm.tools.motif_features import compute_motif_features


def _write_fasta(path: Path, sequences: list[str]) -> None:
    with path.open("w") as f:
        for i, s in enumerate(sequences):
            f.write(f">seq{i}\n{s}\n")


def test_motif_features_shape_matches_target_dim(tmp_path: Path):
    enh = tmp_path / "enh.fa"
    pro = tmp_path / "pro.fa"
    out = tmp_path / "motifs.pt"

    # 3 short sequences, all containing CTCF-like + MYC E-box motifs to ensure non-zero scores.
    seqs = [
        "ACCCTAGTGGCCACCAGGGGGCAGCACGTG",
        "TGACTCATGGGCAGCACGTGGCGCGGGGCG",
        "CACGTGAGATAACCGGAATTCAGCACCNNN",
    ]
    _write_fasta(enh, seqs)
    _write_fasta(pro, seqs)

    feats = compute_motif_features(enh, pro, out, target_dim=55)
    assert feats.shape == (3, 55)
    assert feats.dtype == torch.float32

    # Some entries should be non-zero given the planted motif sites (consensus fallback at minimum).
    assert (feats != 0).any(), "expected non-zero motif scores from planted CTCF/E-box sites"


def test_motif_features_caches_to_disk(tmp_path: Path):
    enh = tmp_path / "enh.fa"
    pro = tmp_path / "pro.fa"
    out = tmp_path / "motifs.pt"
    _write_fasta(enh, ["ACGT" * 10] * 2)
    _write_fasta(pro, ["TGCA" * 10] * 2)
    compute_motif_features(enh, pro, out, target_dim=55)
    assert out.exists()
    loaded = torch.load(out, weights_only=False)
    assert loaded.shape == (2, 55)
