"""Multi-seed ensemble inference: average sigmoid outputs from N independently-trained checkpoints.

The ensemble is "soft" averaging — we take the mean of post-sigmoid probabilities across
checkpoints. This typically lifts AUPR on imbalanced binary classification by 0.005–0.015
without any hyperparameter tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

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


@torch.no_grad()
def _predict_one(cfg: Config, ckpt: Path, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    register_safe_globals()
    model = EPIModel(cfg.model, embeddings_path=cfg.data.embeddings_path).to(device)
    info = load_checkpoint_with_remapping(model, ckpt, device)
    if info["missing"] or info["unexpected"]:
        logger.info("Checkpoint %s: missing=%d unexpected=%d", ckpt, len(info["missing"]), len(info["unexpected"]))
    model.eval()

    preds, labels = [], []
    for enh, pro, gene, lbl in loader:
        enh, pro, gene = enh.to(device), pro.to(device), gene.to(device)
        out, _ = model(enh, pro, gene)
        preds.append(out.view(-1).cpu().numpy())
        labels.append(lbl.view(-1).numpy())
    return np.concatenate(preds), np.concatenate(labels)


def ensemble_predict(
    cfg: Config,
    checkpoints: Iterable[Path],
    test_loader: DataLoader,
    output_dir: Path,
    label: str,
) -> dict:
    """Soft-average sigmoid outputs across `checkpoints`. Saves predictions + per-member detail."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device_from_cfg(cfg)

    checkpoints = list(checkpoints)
    if not checkpoints:
        raise ValueError("ensemble_predict needs at least one checkpoint")

    member_aucs: List[float] = []
    member_auprs: List[float] = []
    avg_score: np.ndarray | None = None
    y_true: np.ndarray | None = None

    for i, ckpt in enumerate(checkpoints):
        logger.info("Ensemble member %d/%d: %s", i + 1, len(checkpoints), ckpt)
        scores, labels = _predict_one(cfg, ckpt, test_loader, device)
        if y_true is None:
            y_true = labels
            avg_score = np.zeros_like(scores)
        elif not np.array_equal(y_true, labels):
            raise RuntimeError("Test loader produced different labels across members; check shuffle=False.")
        avg_score = avg_score + scores
        member_aucs.append(float(roc_auc_score(labels, scores)))
        member_auprs.append(float(average_precision_score(labels, scores)))

    avg_score = avg_score / len(checkpoints)
    pred_bin = (avg_score >= 0.5).astype(int)

    metrics = {
        "label": label,
        "n_members": len(checkpoints),
        "members": [str(c) for c in checkpoints],
        "member_auc_mean": float(np.mean(member_aucs)),
        "member_auc_std": float(np.std(member_aucs)),
        "member_aupr_mean": float(np.mean(member_auprs)),
        "member_aupr_std": float(np.std(member_auprs)),
        "ensemble_auc": float(roc_auc_score(y_true, avg_score)),
        "ensemble_aupr": float(average_precision_score(y_true, avg_score)),
        "ensemble_f1": float(f1_score(y_true.astype(int), pred_bin)),
        "ensemble_accuracy": float(accuracy_score(y_true.astype(int), pred_bin)),
    }

    np.save(output_dir / f"{label}_ensemble_y_true.npy", y_true)
    np.save(output_dir / f"{label}_ensemble_y_score.npy", avg_score)
    return metrics
