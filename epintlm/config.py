"""YAML config loader with dataclass schema and CLI dot-notation overrides."""

from __future__ import annotations

import argparse
import copy
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, List, Optional

import yaml


# ── Dataclass schema ─────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    name: str = "default"
    seed: int = 2025
    device: str = "cuda"
    output_dir: str = "runs"


@dataclass
class ChromatinFeaturesConfig:
    enabled: bool = False
    config_path: str = "data/chromatin_features/CTCF_DNase_6histone.500.json"
    feats_order: List[str] = field(
        default_factory=lambda: ["CTCF", "DNase", "H3K27ac", "H3K4me1", "H3K4me3"]
    )


@dataclass
class MotifFeaturesConfig:
    enabled: bool = False
    cache_dir: str = "data/processed"
    target_dim: int = 55


@dataclass
class SplitsConfig:
    train_frac: float = 0.81
    val_frac: float = 0.09
    test_frac: float = 0.10


@dataclass
class PreprocessingConfig:
    kmer_size: int = 6
    enh_seq_len: int = 3000
    pro_seq_len: int = 2500
    bin_size: int = 500
    num_workers: int = 8


@dataclass
class DataConfig:
    cell_line: str = "HeLa-S3"
    raw_dir: str = "data/raw/targetfinder"
    processed_dir: str = "data/processed"
    chromatin_features: ChromatinFeaturesConfig = field(default_factory=ChromatinFeaturesConfig)
    motif_features: MotifFeaturesConfig = field(default_factory=MotifFeaturesConfig)
    embeddings_path: str = "data/embeddings/nucleotide_transformer_6_kmer_embedding.npy"
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)


@dataclass
class EmbeddingConfig:
    vocab_size: int = 4097
    dim: int = 1280
    freeze_until_epoch: int = 2


@dataclass
class EncoderConfig:
    conv_out_channels: int = 64
    conv_kernel: int = 40
    pool_kernel: int = 20
    pool_stride: int = 20
    dropout: float = 0.5


@dataclass
class GRUConfig:
    hidden_size: int = 32
    num_layers: int = 2
    bidirectional: bool = True


@dataclass
class AttentionConfig:
    embed_dim: int = 64
    num_heads: int = 8
    dropout: float = 0.05
    use_self_attn: bool = True
    use_cross_attn: bool = True
    use_residual: bool = True
    # Where cross-attention is applied along the encoder pipeline.
    #   "post_gru" (default): after Conv→GRU. Original EPINTLM placement.
    #   "pre_gru":  between Conv and GRU. Cross-attends conv-feature time series.
    #   "dual":     applies cross-attn at BOTH pre_gru and post_gru positions.
    placement: str = "post_gru"


@dataclass
class GeneDataConfig:
    input_dim: int = 55
    hidden_dim: int = 64


@dataclass
class HeadConfig:
    hidden_dim: int = 128
    dropout: float = 0.5


@dataclass
class ModelConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    enhancer_encoder: EncoderConfig = field(default_factory=EncoderConfig)
    promoter_encoder: EncoderConfig = field(default_factory=EncoderConfig)
    gru: GRUConfig = field(default_factory=GRUConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    gene_data: GeneDataConfig = field(default_factory=GeneDataConfig)
    head: HeadConfig = field(default_factory=HeadConfig)


@dataclass
class OptimizerConfig:
    type: str = "adam"
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-3


@dataclass
class FineTuneConfig:
    lr: float = 1.0e-4
    unfreeze_epoch: int = 2


@dataclass
class SchedulerConfig:
    type: str = "multistep"
    milestones: List[int] = field(default_factory=lambda: [25])
    gamma: float = 0.1


@dataclass
class EarlyStoppingConfig:
    patience: int = 40
    min_delta: float = 1.0e-4


@dataclass
class TrainingConfig:
    batch_size: int = 256
    num_epochs: int = 40
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    fine_tune: FineTuneConfig = field(default_factory=FineTuneConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    grad_clip_max_norm: float = 1.0
    checkpoint_freq: int = 1
    plot_freq: int = 5


@dataclass
class EvalConfig:
    checkpoint: Optional[str] = None
    cell_lines: List[str] = field(
        default_factory=lambda: ["HeLa-S3", "GM12878", "HUVEC", "IMR90"]
    )
    batch_size: int = 512
    generate_plots: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s [%(run_id)s] %(levelname)s %(name)s: %(message)s"
    console: bool = True
    file: bool = True


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ── Loader ───────────────────────────────────────────────────────────────────


def _from_dict(cls: type, raw: dict) -> Any:
    """Recursively instantiate a dataclass from a dict, preserving defaults for missing keys."""
    if not is_dataclass(cls):
        return raw
    # Resolve string annotations (PEP 563) into real types.
    type_hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name in raw:
            value = raw[f.name]
            field_type = type_hints.get(f.name, f.type)
            if is_dataclass(field_type) and isinstance(value, dict):
                kwargs[f.name] = _from_dict(field_type, value)
            else:
                kwargs[f.name] = value
    return cls(**kwargs)


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Deep-merge overlay into base; overlay keys win."""
    result = copy.deepcopy(base)
    for k, v in overlay.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _set_dotpath(d: dict, dotpath: str, value: Any) -> None:
    parts = dotpath.split(".")
    cursor = d
    for p in parts[:-1]:
        if p not in cursor or not isinstance(cursor[p], dict):
            cursor[p] = {}
        cursor = cursor[p]
    cursor[parts[-1]] = value


def _coerce(value: str) -> Any:
    """Best-effort coercion of CLI override strings to typed Python values via YAML."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def parse_overrides(override_args: List[str]) -> dict:
    """Parse CLI overrides like ['training.batch_size=128', 'run.seed=42'] into nested dict."""
    out: dict = {}
    for arg in override_args:
        if "=" not in arg:
            raise ValueError(f"Override must be key=value, got: {arg!r}")
        key, value = arg.split("=", 1)
        _set_dotpath(out, key.strip(), _coerce(value.strip()))
    return out


def load_config(path: str | Path, overrides: Optional[List[str]] = None) -> Config:
    """Load YAML config, deep-merge over default schema, apply CLI overrides, return Config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    # Start from the dataclass defaults expressed as a dict, then overlay file, then CLI
    base = asdict(Config())
    merged = _deep_merge(base, raw)
    if overrides:
        merged = _deep_merge(merged, parse_overrides(overrides))

    return _from_dict(Config, merged)


def config_to_dict(cfg: Config) -> dict:
    return asdict(cfg)


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False, default_flow_style=False)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add --config and trailing override positional args to a parser."""
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dot-notation overrides, e.g. training.batch_size=128 run.seed=42",
    )
