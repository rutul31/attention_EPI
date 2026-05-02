"""End-to-end preprocessing: raw FASTA/BED/labels → cached SeqGenDataset.pt with split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from ..config import Config
from ..logging_utils import get_logger
from .dataset import SeqGenDataset
from .dummy_features import DummyGenomicFeatures
from .fasta import load_labels, tokenize_fasta
from .tokenizer import create_tokenizer

logger = get_logger(__name__)


def _raw_paths(cfg: Config, split: str) -> dict[str, Path]:
    raw = Path(cfg.data.raw_dir)
    cell = cfg.data.cell_line
    return {
        "enh_fasta": raw / "sequence_data" / f"{cell}_enhancer_{split}.fasta",
        "pro_fasta": raw / "sequence_data" / f"{cell}_promoter_{split}.fasta",
        "labels":    raw / "sequence_data" / f"{cell}_label_{split}.txt",
        "enh_bed":   raw / "bed_files" / f"{cell}_enhancer_{split}.bed",
        "pro_bed":   raw / "bed_files" / f"{cell}_promoter_{split}.bed",
    }


def _processed_path(cfg: Config, split: str) -> Path:
    return Path(cfg.data.processed_dir) / f"{cfg.data.cell_line}_combined_{split}.pt"


def _cache_key(cfg: Config) -> str:
    payload = {
        "cell": cfg.data.cell_line,
        "kmer_size": cfg.data.preprocessing.kmer_size,
        "enh_seq_len": cfg.data.preprocessing.enh_seq_len,
        "pro_seq_len": cfg.data.preprocessing.pro_seq_len,
        "bin_size": cfg.data.preprocessing.bin_size,
        "chromatin_enabled": cfg.data.chromatin_features.enabled,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _build_split(cfg: Config, split: str) -> SeqGenDataset:
    paths = _raw_paths(cfg, split)
    missing = [name for name, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw inputs for split={split}: {missing}. "
            f"Acquire data via scripts/download_targetfinder.sh."
        )

    tokenizer = create_tokenizer(k=cfg.data.preprocessing.kmer_size)
    logger.info("Tokenizing %s...", paths["enh_fasta"])
    enh_ids = tokenize_fasta(paths["enh_fasta"], tokenizer, cfg.data.preprocessing.num_workers)
    logger.info("Tokenizing %s...", paths["pro_fasta"])
    pro_ids = tokenize_fasta(paths["pro_fasta"], tokenizer, cfg.data.preprocessing.num_workers)
    labels = load_labels(paths["labels"])

    if cfg.data.chromatin_features.enabled:
        from .genomic_features import GenomicFeatures
        logger.info("Loading real chromatin features for %s...", cfg.data.cell_line)
        gene_data = GenomicFeatures(
            enh_bed=paths["enh_bed"],
            pro_bed=paths["pro_bed"],
            feats_config_path=cfg.data.chromatin_features.config_path,
            feats_order=cfg.data.chromatin_features.feats_order,
            cell=cfg.data.cell_line,
            enh_seq_len=cfg.data.preprocessing.enh_seq_len,
            pro_seq_len=cfg.data.preprocessing.pro_seq_len,
            bin_size=cfg.data.preprocessing.bin_size,
        )
    elif cfg.data.motif_features.enabled:
        gene_data = _load_or_compute_motif_features(cfg, split, paths, n_samples=len(labels))
    else:
        logger.info("Using DummyGenomicFeatures (chromatin_features.enabled=false).")
        gene_data = DummyGenomicFeatures(size=len(labels))

    return SeqGenDataset(enh_ids, pro_ids, gene_data, labels)


def _load_or_compute_motif_features(cfg: Config, split: str, paths: dict[str, Path], n_samples: int):
    """Cache motif feature tensors at {motif_features.cache_dir}/{cell}_motifs_{split}.pt."""
    from ..tools.motif_features import compute_motif_features

    cache_dir = Path(cfg.data.motif_features.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cfg.data.cell_line}_motifs_{split}.pt"

    if cache_path.exists():
        logger.info("Motif feature cache hit: %s", cache_path)
        tensor = torch.load(cache_path, weights_only=False)
    else:
        logger.info("Computing motif features for %s/%s → %s", cfg.data.cell_line, split, cache_path)
        tensor = compute_motif_features(
            enh_fasta=paths["enh_fasta"],
            pro_fasta=paths["pro_fasta"],
            out_path=cache_path,
            target_dim=cfg.data.motif_features.target_dim,
        )

    if tensor.shape[0] != n_samples:
        raise RuntimeError(
            f"Motif feature row count {tensor.shape[0]} != label count {n_samples} for {split}; "
            f"delete {cache_path} to recompute."
        )

    return _TensorDataset(tensor)


class _TensorDataset:
    """Tiny Dataset-like wrapper exposing __len__/__getitem__ for a precomputed (N, D) tensor."""
    def __init__(self, t: torch.Tensor):
        self._t = t

    def __len__(self) -> int:
        return self._t.shape[0]

    def __getitem__(self, idx) -> torch.Tensor:
        return self._t[idx]


def _subset_gene_data(gene_data, ids: np.ndarray, cfg: Config):
    """Index gene_data by `ids`, dispatching by type."""
    if isinstance(gene_data, _TensorDataset):
        return _TensorDataset(gene_data._t[ids])
    if isinstance(gene_data, DummyGenomicFeatures):
        return DummyGenomicFeatures(size=len(ids), feat_dim=gene_data.feat_dim)
    # Real chromatin features (or generic dataset): index element-wise.
    return [gene_data[int(i)] for i in ids]


def _split_dataset(
    full: SeqGenDataset, cfg: Config
) -> Tuple[SeqGenDataset, SeqGenDataset, SeqGenDataset]:
    """Split a single combined dataset into train/val/test honoring the configured fractions."""
    n = len(full)
    s = cfg.data.splits
    if abs((s.train_frac + s.val_frac + s.test_frac) - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0; got {s.train_frac + s.val_frac + s.test_frac}")

    indices = np.arange(n)
    labels_arr = full.labels if isinstance(full.labels, np.ndarray) else np.array(full.labels)

    rest_idx, test_idx = train_test_split(
        indices, test_size=s.test_frac, stratify=labels_arr, random_state=cfg.run.seed
    )
    val_relative = s.val_frac / (s.train_frac + s.val_frac)
    train_idx, val_idx = train_test_split(
        rest_idx, test_size=val_relative, stratify=labels_arr[rest_idx], random_state=cfg.run.seed
    )

    def subset(idx_array: np.ndarray) -> SeqGenDataset:
        return SeqGenDataset(
            enhancer_ids=full.enhancer_ids[idx_array],
            promoter_ids=full.promoter_ids[idx_array],
            gene_data_all=_subset_gene_data(full.gene_data_all, idx_array, cfg),
            labels=labels_arr[idx_array],
        )

    return subset(train_idx), subset(val_idx), subset(test_idx)


def preprocess(cfg: Config, force: bool = False) -> dict[str, Path]:
    """Build (or load from cache) train/val/test datasets. Returns dict of paths."""
    Path(cfg.data.processed_dir).mkdir(parents=True, exist_ok=True)

    targets = {split: _processed_path(cfg, split) for split in ("train", "val", "test")}
    cache_marker = Path(cfg.data.processed_dir) / f"{cfg.data.cell_line}_cache.json"

    if not force and all(p.exists() for p in targets.values()) and cache_marker.exists():
        with cache_marker.open("r") as f:
            existing_key = json.load(f).get("cache_key")
        if existing_key == _cache_key(cfg):
            logger.info("Cache hit for %s (key=%s); skipping preprocessing.", cfg.data.cell_line, existing_key)
            return targets
        logger.info("Cache key changed; rebuilding.")

    # Try the simple path first: separate train/test files exist
    train_paths_exist = all(_raw_paths(cfg, "train")[k].exists() for k in _raw_paths(cfg, "train"))
    test_paths_exist = all(_raw_paths(cfg, "test")[k].exists() for k in _raw_paths(cfg, "test"))

    if train_paths_exist and test_paths_exist:
        logger.info("Building train + test directly from raw files; carving val from train.")
        train_full = _build_split(cfg, "train")
        test_ds = _build_split(cfg, "test")

        # Carve val out of train using configured val_frac (relative to train+val)
        val_rel = cfg.data.splits.val_frac / (cfg.data.splits.train_frac + cfg.data.splits.val_frac)
        labels_arr = train_full.labels if isinstance(train_full.labels, np.ndarray) else np.array(train_full.labels)
        idx = np.arange(len(train_full))
        tr_idx, val_idx = train_test_split(
            idx, test_size=val_rel, stratify=labels_arr, random_state=cfg.run.seed
        )

        def subset(full: SeqGenDataset, ids: np.ndarray) -> SeqGenDataset:
            return SeqGenDataset(
                enhancer_ids=full.enhancer_ids[ids],
                promoter_ids=full.promoter_ids[ids],
                gene_data_all=_subset_gene_data(full.gene_data_all, ids, cfg),
                labels=(full.labels[ids] if isinstance(full.labels, np.ndarray) else np.array(full.labels)[ids]),
            )

        train_ds = subset(train_full, tr_idx)
        val_ds = subset(train_full, val_idx)
    elif test_paths_exist:
        logger.warning("Only test files found — splitting test 81/9/10 (train/val/test). "
                       "For real training, acquire train data via scripts/download_targetfinder.sh.")
        full = _build_split(cfg, "test")
        train_ds, val_ds, test_ds = _split_dataset(full, cfg)
    else:
        raise FileNotFoundError(
            f"No raw data found for cell line {cfg.data.cell_line}. "
            f"Run scripts/download_targetfinder.sh first."
        )

    for split, ds in zip(("train", "val", "test"), (train_ds, val_ds, test_ds)):
        torch.save(ds, targets[split])
        logger.info("Saved %s split (%d samples) → %s", split, len(ds), targets[split])

    with cache_marker.open("w") as f:
        json.dump({"cache_key": _cache_key(cfg), "cell_line": cfg.data.cell_line}, f, indent=2)

    return targets
