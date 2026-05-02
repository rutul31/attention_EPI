"""Evaluator — runs inference on one or more checkpoints, saves predictions + metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import register_safe_globals
from ..logging_utils import get_logger
from ..models.epintlm import EPIModel, load_checkpoint_with_remapping

logger = get_logger(__name__)


def _device_from_cfg(cfg: Config) -> torch.device:
    if cfg.run.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_checkpoint(cfg: Config, run_dir: Path | None, override: str | None) -> Path:
    if override:
        return Path(override)
    if cfg.eval.checkpoint:
        return Path(cfg.eval.checkpoint)
    if run_dir is None:
        raise ValueError("No checkpoint specified and no run_dir provided.")
    return Path(run_dir) / "checkpoints" / "best.pt"


@torch.no_grad()
def _run_inference(model: EPIModel, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for enh, pro, gene, labels in loader:
        enh = enh.to(device)
        pro = pro.to(device)
        gene = gene.to(device)
        labels = labels.to(device).float().view(-1, 1)

        outputs, _ = model(enh, pro, gene)
        outputs = outputs.view(-1, 1)
        loss = model.criterion(outputs, labels)
        total_loss += loss.item()

        all_preds.append(outputs.view(-1).cpu().numpy())
        all_labels.append(labels.view(-1).cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        total_loss / max(1, len(loader)),
    )


def evaluate_checkpoint(
    cfg: Config,
    checkpoint_path: Path,
    test_loader: DataLoader,
    output_dir: Path,
    label: str = "model",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_cfg(cfg)

    register_safe_globals()
    model = EPIModel(cfg.model, embeddings_path=cfg.data.embeddings_path).to(device)
    info = load_checkpoint_with_remapping(model, checkpoint_path, device)
    if info["missing"] or info["unexpected"]:
        logger.info("Checkpoint load: missing=%d unexpected=%d", len(info["missing"]), len(info["unexpected"]))

    y_score, y_true, avg_loss = _run_inference(model, test_loader, device)
    pred_bin = (y_score >= 0.5).astype(int)

    metrics = {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "loss": float(avg_loss),
        "auc": float(roc_auc_score(y_true, y_score)),
        "aupr": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true.astype(int), pred_bin)),
        "accuracy": float(accuracy_score(y_true.astype(int), pred_bin)),
        "positive_preds": int(pred_bin.sum()),
        "total_positives": int(y_true.sum()),
        "n_samples": int(len(y_true)),
    }

    np.save(output_dir / f"{label}_y_true.npy", y_true)
    np.save(output_dir / f"{label}_y_score.npy", y_score)
    return metrics


def evaluate(
    cfg: Config,
    test_loader: DataLoader,
    run_dir: Path | None = None,
    checkpoint_override: str | None = None,
    multi_checkpoints: Dict[str, Path] | None = None,
) -> Dict[str, dict]:
    """If multi_checkpoints provided, evaluate each and return per-label metrics; else single checkpoint."""
    out_root = (run_dir / "eval") if run_dir else Path("eval")
    pred_dir = out_root / "predictions"

    results: Dict[str, dict] = {}
    if multi_checkpoints:
        for label, ckpt in multi_checkpoints.items():
            logger.info("Evaluating %s using %s", label, ckpt)
            results[label] = evaluate_checkpoint(cfg, ckpt, test_loader, pred_dir, label=label)
    else:
        ckpt = _resolve_checkpoint(cfg, run_dir, checkpoint_override)
        label = cfg.data.cell_line
        logger.info("Evaluating using %s", ckpt)
        results[label] = evaluate_checkpoint(cfg, ckpt, test_loader, pred_dir, label=label)

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "metrics.json").open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote metrics → %s", out_root / "metrics.json")
    return results
