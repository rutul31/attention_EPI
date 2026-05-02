"""CLI entry point: preprocess raw inputs into cached SeqGenDataset .pt files.

Usage:
  python -m epintlm.cli.preprocess --config configs/cell_lines/HeLa-S3.yaml \
    [data.preprocessing.num_workers=4] [--force]
"""

from __future__ import annotations

import argparse
import sys

from .._env_check import require

require("torch", "numpy", "yaml", "sklearn")

from ..config import add_config_args, load_config  # noqa: E402
from ..data.preprocess import preprocess  # noqa: E402
from ..logging_utils import configure_logging, get_logger, set_run_id  # noqa: E402

logger = get_logger("epintlm.cli.preprocess")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess raw FASTA/BED into cached .pt datasets.")
    add_config_args(parser)
    parser.add_argument("--force", action="store_true", help="Force rebuild even if cache hit.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    set_run_id(f"preprocess:{cfg.data.cell_line}")
    configure_logging(level=cfg.logging.level, fmt=cfg.logging.format, console=cfg.logging.console)

    logger.info("Cell line: %s", cfg.data.cell_line)
    logger.info("Raw dir: %s", cfg.data.raw_dir)
    logger.info("Processed dir: %s", cfg.data.processed_dir)

    paths = preprocess(cfg, force=args.force)
    for split, p in paths.items():
        logger.info("  %s → %s", split, p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
