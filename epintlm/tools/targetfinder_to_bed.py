"""Convert a TargetFinder pairs.csv into the EPINTLM-format BED + label files.

Input:  pairs.csv with columns including
          enhancer_chrom, enhancer_start, enhancer_end,
          promoter_chrom, promoter_start, promoter_end,
          label
        (additional columns are ignored).

Output (per cell line × split ∈ {train, test}):
  {out_dir}/bed_files/{cell}_enhancer_{split}.bed
  {out_dir}/bed_files/{cell}_promoter_{split}.bed
  {out_dir}/sequence_data/{cell}_label_{split}.txt

The BED format here is the EPINTLM custom 4-column format used by genomic_features.py:
  label \\t chrom \\t start \\t end
where [start, end) is the FIXED-LENGTH window (default 3000 bp enhancer / 2000 bp promoter)
centered on the midpoint of the original element. These BEDs feed bedtools getfasta to
produce the matching {enhancer,promoter}.fasta files.

Train/test split (DEFAULT: bin-based when available — matches paper Table 1 best):
  - Bin-based split holds out ~test_frac of bin IDs as test. Bins are chromosomally-coherent
    intervals from the original TargetFinder pairs.csv, so this prevents the same
    enhancer/promoter LOCUS from appearing in both train and test (locus leakage).
  - --split=stratified falls back to a stratified random split when the 'bin' column is
    missing or you explicitly want it.

Usage:
  python -m epintlm.tools.targetfinder_to_bed \\
    --pairs data/raw/targetfinder/pairs/HeLa-S3_pairs.csv \\
    --cell HeLa-S3 \\
    --out-dir data/raw/targetfinder \\
    [--enh-seq-len 3000] [--pro-seq-len 2000] \\
    [--test-frac 0.10] [--seed 2025] \\
    [--split bin|stratified]   # default: bin if column present, else stratified
    [--augment-train]          # oversample train positives 1:1 with negatives
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _expand_around_midpoint(start: int, end: int, length: int) -> Tuple[int, int]:
    """Return half-open [s, e) of width `length` centered on (start+end)//2; clamped at 0."""
    mid = (int(start) + int(end)) // 2
    s = max(0, mid - length // 2)
    e = s + length
    return s, e


def _write_bed(path: Path, rows: pd.DataFrame, chrom_col: str, start_col: str, end_col: str,
               label_col: str, length: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for _, r in rows.iterrows():
            s, e = _expand_around_midpoint(r[start_col], r[end_col], length)
            f.write(f"{int(r[label_col])}\t{r[chrom_col]}\t{s}\t{e}\n")


def _write_labels(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for v in labels:
            f.write(f"{int(v)}\n")


def _augment_to_balance(df: pd.DataFrame, label_col: str, seed: int) -> pd.DataFrame:
    """Oversample minority class with replacement so class counts match. Used for train only."""
    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return df
    rng = np.random.RandomState(seed)
    if len(pos) < len(neg):
        idx = rng.choice(len(pos), size=len(neg), replace=True)
        pos_aug = pos.iloc[idx]
        out = pd.concat([pos_aug, neg], ignore_index=True)
    elif len(neg) < len(pos):
        idx = rng.choice(len(neg), size=len(pos), replace=True)
        neg_aug = neg.iloc[idx]
        out = pd.concat([pos, neg_aug], ignore_index=True)
    else:
        out = df.copy()
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map our canonical names → df's actual column names. Tolerates minor variations."""
    candidates = {
        "enh_chrom": ["enhancer_chrom", "enh_chrom"],
        "enh_start": ["enhancer_start", "enh_start"],
        "enh_end":   ["enhancer_end", "enh_end"],
        "pro_chrom": ["promoter_chrom", "pro_chrom"],
        "pro_start": ["promoter_start", "pro_start"],
        "pro_end":   ["promoter_end", "pro_end"],
        "label":     ["label", "labels", "interaction"],
        "bin":       ["bin", "fold"],  # optional
    }
    resolved = {}
    for canonical, options in candidates.items():
        match = next((c for c in options if c in df.columns), None)
        if match is None and canonical != "bin":
            raise KeyError(f"pairs.csv missing required column for {canonical}; tried {options}. "
                           f"Found: {list(df.columns)}")
        resolved[canonical] = match
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", required=True, type=Path, help="Path to pairs.csv")
    parser.add_argument("--cell", required=True, help="Cell line name (used in output filenames)")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Root output dir (writes bed_files/ and sequence_data/ underneath)")
    parser.add_argument("--enh-seq-len", type=int, default=3000)
    parser.add_argument("--pro-seq-len", type=int, default=2000)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--augment-train", action="store_true",
                        help="Oversample minority class in TRAIN to balance (TargetFinder-style).")
    parser.add_argument("--split", choices=["bin", "stratified"], default="bin",
                        help="bin = hold out test_frac of bins (default; preserves locus boundaries). "
                             "stratified = stratified random split (allows locus leakage).")
    args = parser.parse_args(argv)

    if not args.pairs.exists():
        print(f"ERROR: pairs file not found: {args.pairs}", file=sys.stderr)
        return 1

    print(f"Reading {args.pairs} ...")
    df = pd.read_csv(args.pairs)
    print(f"  rows={len(df)}, columns={list(df.columns)}")

    cols = _resolve_columns(df)

    # Split — default is bin-based (locus-respecting). Fall back to stratified only if
    # the bin column is missing OR the user explicitly asks for it.
    use_bin = (args.split == "bin") and (cols["bin"] is not None)
    if args.split == "bin" and cols["bin"] is None:
        print("  WARNING: --split=bin requested but pairs.csv has no 'bin' column; "
              "falling back to stratified random.")
    if use_bin:
        bins = df[cols["bin"]].unique()
        rng = np.random.RandomState(args.seed)
        rng.shuffle(bins)
        n_test = max(1, int(round(len(bins) * args.test_frac)))
        test_bins = set(bins[:n_test])
        is_test = df[cols["bin"]].isin(test_bins)
        train_df = df[~is_test].reset_index(drop=True)
        test_df = df[is_test].reset_index(drop=True)
        print(f"  split=bin: total_bins={len(bins)} train_bins={len(bins) - n_test} test_bins={n_test}")
    else:
        train_df, test_df = train_test_split(
            df, test_size=args.test_frac, stratify=df[cols["label"]], random_state=args.seed
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        print(f"  split=stratified-random: train={len(train_df)} test={len(test_df)}")

    # Diagnostic: print pos/neg counts vs paper Table 1 expected ranges
    for split_name, sdf in (("train", train_df), ("test", test_df)):
        n_pos = int((sdf[cols["label"]] == 1).sum())
        n_neg = int((sdf[cols["label"]] == 0).sum())
        print(f"  {split_name} pre-augmentation: total={len(sdf)} pos={n_pos} neg={n_neg} "
              f"ratio=1:{n_neg / max(1, n_pos):.1f}")

    if args.augment_train:
        before = len(train_df)
        train_df = _augment_to_balance(train_df, cols["label"], seed=args.seed)
        print(f"  augmentation: train {before} → {len(train_df)} (balanced)")

    out_dir = args.out_dir
    bed_dir = out_dir / "bed_files"
    seq_dir = out_dir / "sequence_data"
    bed_dir.mkdir(parents=True, exist_ok=True)
    seq_dir.mkdir(parents=True, exist_ok=True)

    for split, sdf in (("train", train_df), ("test", test_df)):
        _write_bed(bed_dir / f"{args.cell}_enhancer_{split}.bed",
                   sdf, cols["enh_chrom"], cols["enh_start"], cols["enh_end"],
                   cols["label"], args.enh_seq_len)
        _write_bed(bed_dir / f"{args.cell}_promoter_{split}.bed",
                   sdf, cols["pro_chrom"], cols["pro_start"], cols["pro_end"],
                   cols["label"], args.pro_seq_len)
        _write_labels(seq_dir / f"{args.cell}_label_{split}.txt",
                      sdf[cols["label"]].to_numpy())

        n_pos = int((sdf[cols["label"]] == 1).sum())
        n_neg = int((sdf[cols["label"]] == 0).sum())
        print(f"  {split}: total={len(sdf)} pos={n_pos} neg={n_neg}")

    print("✅ BED + label files written.")
    print(f"   Next: extract FASTAs via scripts/extract_fastas.sh "
          f"(needs hg19.fa + bedtools).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
