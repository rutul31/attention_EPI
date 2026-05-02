"""Tests for the TargetFinder pairs.csv → BED + label converter."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from epintlm.tools.targetfinder_to_bed import _expand_around_midpoint, _resolve_columns, main


def _write_pairs_csv(path: Path, n_pos: int = 30, n_neg: int = 70) -> None:
    rows = []
    for i in range(n_pos + n_neg):
        chrom = f"chr{(i % 22) + 1}"
        e_mid = 100_000 + i * 50_000
        p_mid = e_mid + 10_000
        rows.append({
            "enhancer_chrom": chrom, "enhancer_start": e_mid - 100, "enhancer_end": e_mid + 100,
            "promoter_chrom": chrom, "promoter_start": p_mid - 100, "promoter_end": p_mid + 100,
            "label": 1 if i < n_pos else 0,
            "bin": i % 10,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_expand_around_midpoint_basic():
    s, e = _expand_around_midpoint(1_000_000, 1_000_200, 1000)
    assert e - s == 1000
    # mid = (1_000_000 + 1_000_200) // 2 = 1_000_100; expand ±500
    assert s == 1_000_100 - 500
    assert e == 1_000_100 + 500


def test_expand_around_midpoint_clamps_at_zero():
    s, e = _expand_around_midpoint(10, 20, 1000)
    assert s == 0
    assert e == 1000


def test_resolve_columns_handles_synonyms():
    df = pd.DataFrame(columns=[
        "enhancer_chrom", "enhancer_start", "enhancer_end",
        "promoter_chrom", "promoter_start", "promoter_end",
        "label",
    ])
    cols = _resolve_columns(df)
    assert cols["enh_chrom"] == "enhancer_chrom"
    assert cols["label"] == "label"
    assert cols["bin"] is None  # optional


def test_resolve_columns_missing_required_raises():
    df = pd.DataFrame(columns=["enhancer_chrom"])
    with pytest.raises(KeyError):
        _resolve_columns(df)


def test_bin_split_is_default(tmp_path: Path, capsys):
    """Bin-based split is the new default; pairs in the same bin go to the same split."""
    pairs = tmp_path / "pairs.csv"
    out_dir = tmp_path / "out"
    _write_pairs_csv(pairs, n_pos=30, n_neg=70)

    rc = main([
        "--pairs", str(pairs),
        "--cell", "TestCell",
        "--out-dir", str(out_dir),
        "--test-frac", "0.20",
        "--seed", "42",
    ])
    assert rc == 0
    output = capsys.readouterr().out
    assert "split=bin" in output, f"expected bin-based split in output, got:\n{output}"


def test_main_end_to_end(tmp_path: Path):
    pairs = tmp_path / "pairs.csv"
    out_dir = tmp_path / "out"
    _write_pairs_csv(pairs, n_pos=30, n_neg=70)

    rc = main([
        "--pairs", str(pairs),
        "--cell", "TestCell",
        "--out-dir", str(out_dir),
        "--enh-seq-len", "1000",
        "--pro-seq-len", "800",
        "--test-frac", "0.20",
        "--seed", "42",
        "--split", "stratified",   # synthetic data has no real bin structure; force stratified
    ])
    assert rc == 0

    bed_dir = out_dir / "bed_files"
    seq_dir = out_dir / "sequence_data"

    for split in ("train", "test"):
        for kind, length in (("enhancer", 1000), ("promoter", 800)):
            bed = bed_dir / f"TestCell_{kind}_{split}.bed"
            assert bed.exists(), f"missing {bed}"
            with bed.open() as f:
                first = f.readline().strip().split("\t")
                assert len(first) == 4
                lbl, chrom, s, e = first
                assert lbl in {"0", "1"}
                assert chrom.startswith("chr")
                assert int(e) - int(s) == length

        labels = (seq_dir / f"TestCell_label_{split}.txt").read_text().splitlines()
        assert len(labels) > 0
        assert all(v in {"0", "1"} for v in labels)

    # Stratified split: roughly 20% in test
    test_count = sum(1 for _ in (bed_dir / "TestCell_enhancer_test.bed").open())
    train_count = sum(1 for _ in (bed_dir / "TestCell_enhancer_train.bed").open())
    assert test_count + train_count == 100
    assert 15 <= test_count <= 25
