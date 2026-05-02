"""CLI entry point: train a model from a YAML config.

Usage:
  python -m epintlm.cli.train --config configs/cell_lines/HeLa-S3.yaml \
    [training.num_epochs=40] [run.seed=42] [run.name=my_run]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .._env_check import require

require("torch", "numpy", "yaml", "sklearn")

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ..config import add_config_args, load_config, save_config  # noqa: E402
from ..data.dataset import register_safe_globals  # noqa: E402
from ..data.preprocess import preprocess  # noqa: E402
from ..logging_utils import configure_logging, get_logger, set_run_id  # noqa: E402
from ..manifest import write_manifest  # noqa: E402
from ..training.trainer import Trainer  # noqa: E402

logger = get_logger("epintlm.cli.train")


def _build_run_dir(cfg) -> Path:
    name = cfg.run.name
    if name == "default" or not name:
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    elif "_" not in name or not name[:10].replace("-", "").isdigit():
        # Prefix with timestamp unless caller already supplied one
        name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{name}"
    run_dir = Path(cfg.run.output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _make_loader(path: Path, batch_size: int, shuffle: bool, num_workers: int = 2) -> DataLoader:
    register_safe_globals()
    dataset = torch.load(str(path), weights_only=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train EPINTLM model.")
    add_config_args(parser)
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing step (assume cached files exist).")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    run_dir = _build_run_dir(cfg)
    set_run_id(run_dir.name)

    log_file = run_dir / "train.log" if cfg.logging.file else None
    configure_logging(level=cfg.logging.level, fmt=cfg.logging.format,
                      console=cfg.logging.console, log_file=log_file)

    logger.info("Run dir: %s", run_dir)
    save_config(cfg, run_dir / "config.yaml")
    write_manifest(run_dir, cfg)

    if not args.skip_preprocess:
        logger.info("Preprocessing (cached if available)...")
        preprocess(cfg)

    processed_dir = Path(cfg.data.processed_dir)
    train_path = processed_dir / f"{cfg.data.cell_line}_combined_train.pt"
    val_path = processed_dir / f"{cfg.data.cell_line}_combined_val.pt"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Missing processed splits. Expected: {train_path}, {val_path}. "
            f"Run preprocessing first or remove --skip-preprocess."
        )

    logger.info("Loading train loader from %s", train_path)
    train_loader = _make_loader(train_path, cfg.training.batch_size, shuffle=True)
    logger.info("Loading val loader from %s", val_path)
    val_loader = _make_loader(val_path, cfg.training.batch_size, shuffle=False)

    trainer = Trainer(cfg, run_dir, train_loader, val_loader)
    result = trainer.fit()

    logger.info("Training finished: %s", result)
    logger.info("Best checkpoint: %s", run_dir / "checkpoints" / "best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
