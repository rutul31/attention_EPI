"""CLI entry point: evaluate a trained checkpoint (or a list of pretrained checkpoints).

Usage:
  # Single checkpoint (default: {RUN_DIR}/checkpoints/best.pt)
  python -m epintlm.cli.eval --config configs/cell_lines/HeLa-S3.yaml --run-dir runs/my_run

  # Override checkpoint
  python -m epintlm.cli.eval --config configs/default.yaml --checkpoint checkpoints/L1_HELA.pt --run-dir runs/eval_only

  # Multi-checkpoint mode (one per cell line listed in eval.cell_lines), looked up from a directory
  python -m epintlm.cli.eval --config configs/default.yaml --checkpoints-dir checkpoints --run-dir runs/eval_all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .._env_check import require

require("torch", "numpy", "yaml", "sklearn")

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ..config import add_config_args, load_config, save_config  # noqa: E402
from ..data.dataset import register_safe_globals  # noqa: E402
from ..eval.ensemble import ensemble_predict  # noqa: E402
from ..eval.evaluator import evaluate  # noqa: E402
from ..eval.plotting import generate_all_plots  # noqa: E402
from ..logging_utils import configure_logging, get_logger, set_run_id  # noqa: E402
from ..manifest import write_manifest  # noqa: E402

logger = get_logger("epintlm.cli.eval")

# Maps cell line → expected pretrained checkpoint filename in --checkpoints-dir
DEFAULT_CHECKPOINT_NAMES = {
    "HeLa-S3":  "L1_HELA.pt",
    "GM12878":  "L1_NU_GM12878.pt",
    "HUVEC":    "L1_NU_HUVEC.pt",
    "IMR90":    "L1_NU_IMR90.pt",
}


def _make_test_loader(processed_dir: Path, cell_line: str, batch_size: int) -> DataLoader:
    register_safe_globals()
    test_path = processed_dir / f"{cell_line}_combined_test.pt"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing processed test split: {test_path}")
    dataset = torch.load(str(test_path), weights_only=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate EPINTLM checkpoint(s).")
    add_config_args(parser)
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Output directory for eval results (a runs/ subdirectory).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint path (overrides config eval.checkpoint).")
    parser.add_argument("--checkpoints-dir", type=str, default=None,
                        help="Multi-checkpoint mode: directory containing per-cell-line checkpoints.")
    parser.add_argument("--ensemble", nargs="+", default=None, metavar="CHECKPOINT",
                        help="Ensemble mode: average sigmoid outputs across the listed checkpoints "
                             "(soft averaging; expects ≥2 paths). Mutually exclusive with --checkpoint.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_run_id(run_dir.name)

    log_file = run_dir / "eval.log" if cfg.logging.file else None
    configure_logging(level=cfg.logging.level, fmt=cfg.logging.format,
                      console=cfg.logging.console, log_file=log_file)

    save_config(cfg, run_dir / "config.yaml")
    write_manifest(run_dir, cfg)

    test_loader = _make_test_loader(Path(cfg.data.processed_dir), cfg.data.cell_line, cfg.eval.batch_size)

    multi: dict[str, Path] | None = None
    if args.checkpoints_dir:
        ckpt_dir = Path(args.checkpoints_dir)
        multi = {}
        for cell in cfg.eval.cell_lines:
            name = DEFAULT_CHECKPOINT_NAMES.get(cell)
            if name is None:
                logger.warning("No default checkpoint name registered for %s; skipping.", cell)
                continue
            path = ckpt_dir / name
            if not path.exists():
                logger.warning("Checkpoint not found: %s; skipping %s.", path, cell)
                continue
            multi[cell] = path
        if not multi:
            raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")

    if args.ensemble:
        if args.checkpoint:
            raise SystemExit("--ensemble and --checkpoint are mutually exclusive.")
        ckpts = [Path(c) for c in args.ensemble]
        out_dir = run_dir / "eval"
        em = ensemble_predict(cfg, ckpts, test_loader, out_dir / "predictions",
                              label=cfg.data.cell_line)
        with (out_dir / "ensemble_metrics.json").open("w") as f:
            import json as _json
            _json.dump(em, f, indent=2)
        logger.info("[ensemble n=%d] member AUC=%.4f±%.4f → ensemble AUC=%.4f",
                    em["n_members"], em["member_auc_mean"], em["member_auc_std"], em["ensemble_auc"])
        logger.info("[ensemble n=%d] member AUPR=%.4f±%.4f → ensemble AUPR=%.4f",
                    em["n_members"], em["member_aupr_mean"], em["member_aupr_std"], em["ensemble_aupr"])
        return 0

    results = evaluate(cfg, test_loader, run_dir=run_dir,
                       checkpoint_override=args.checkpoint, multi_checkpoints=multi)

    for label, m in results.items():
        logger.info("[%s] AUC=%.4f AUPR=%.4f F1=%.4f Acc=%.4f", label, m["auc"], m["aupr"], m["f1"], m["accuracy"])

    if cfg.eval.generate_plots:
        plot_paths = generate_all_plots(run_dir / "eval" / "predictions", run_dir / "eval" / "plots")
        for name, path in plot_paths.items():
            if path:
                logger.info("Plot %s: %s", name, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
