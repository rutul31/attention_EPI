"""Plot generation — ROC, PR, model-comparison bar charts, ablation, repro vs paper."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ..logging_utils import get_logger

logger = get_logger(__name__)

CELL_LINES = ["HeLa-S3", "GM12878", "HUVEC", "IMR90"]
COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]

# Paper Table 2 reference values (AUC, AUPR) — last entry per cell line is filled at runtime.
MODELS = ["EPIANN", "SIMCNN", "PEP-WORD", "SPEID", "EPIPDLF", "EPINTLM"]
TABLE2_AUC = {
    "HeLa-S3": [0.924, 0.949, 0.843, 0.923, 0.964, None],
    "GM12878": [0.919, 0.941, 0.842, 0.916, 0.939, None],
    "HUVEC":   [0.918, 0.933, 0.845, 0.904, 0.935, None],
    "IMR90":   [0.945, 0.951, 0.898, 0.915, 0.936, None],
}
TABLE2_AUPR = {
    "HeLa-S3": [0.702, 0.737, 0.803, 0.797, 0.849, None],
    "GM12878": [0.723, 0.706, 0.807, 0.773, 0.788, None],
    "HUVEC":   [0.616, 0.640, 0.760, 0.523, 0.730, None],
    "IMR90":   [0.770, 0.737, 0.868, 0.732, 0.779, None],
}

# Paper Tables 3 & 4 — ablation mean ± std
ABLATION_VARIANTS = ["w/o Residual\n+ Cross-Attn", "w/o\nResidual", "EPINTLM\n(Full)"]
ABLATION_AUC_MEAN = {
    "HeLa-S3": [0.969, 0.968, 0.970], "GM12878": [0.943, 0.939, 0.949],
    "HUVEC":   [0.928, 0.938, 0.935], "IMR90":   [0.907, 0.908, 0.909],
}
ABLATION_AUC_STD = {
    "HeLa-S3": [0.008, 0.005, 0.004], "GM12878": [0.005, 0.006, 0.002],
    "HUVEC":   [0.012, 0.006, 0.000], "IMR90":   [0.006, 0.008, 0.003],
}
ABLATION_AUPR_MEAN = {
    "HeLa-S3": [0.861, 0.863, 0.865], "GM12878": [0.787, 0.780, 0.779],
    "HUVEC":   [0.700, 0.748, 0.741], "IMR90":   [0.724, 0.733, 0.727],
}
ABLATION_AUPR_STD = {
    "HeLa-S3": [0.015, 0.010, 0.011], "GM12878": [0.005, 0.012, 0.001],
    "HUVEC":   [0.027, 0.008, 0.006], "IMR90":   [0.012, 0.008, 0.007],
}
PAPER_AUC = {"HeLa-S3": 0.970, "GM12878": 0.949, "HUVEC": 0.935, "IMR90": 0.909}
PAPER_AUPR = {"HeLa-S3": 0.865, "GM12878": 0.779, "HUVEC": 0.741, "IMR90": 0.727}


def _load_predictions(pred_dir: Path, cell_line: str):
    y_true = np.load(pred_dir / f"{cell_line}_y_true.npy")
    y_score = np.load(pred_dir / f"{cell_line}_y_score.npy")
    return y_true, y_score


def _available_cell_lines(pred_dir: Path) -> List[str]:
    return [c for c in CELL_LINES if (pred_dir / f"{c}_y_true.npy").exists()]


def plot_roc(pred_dir: Path, out_dir: Path) -> Path | None:
    cells = _available_cell_lines(pred_dir)
    if not cells:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, xlim, ylim, title in [
        (axes[0], [0, 1], [0, 1.02], "ROC Curves — EPINTLM (All Cell Lines)"),
        (axes[1], [0, 0.2], [0.5, 1.02], "ROC Curves — Zoomed (FPR ≤ 0.2)"),
    ]:
        for cl, color in zip(cells, COLORS):
            y_true, y_score = _load_predictions(pred_dir, cl)
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{cl} (AUC = {auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(title); ax.legend(loc="lower right", fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "A_roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    logger.info("Saved %s", path)
    return path


def plot_pr(pred_dir: Path, out_dir: Path) -> Path | None:
    cells = _available_cell_lines(pred_dir)
    if not cells:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    baseline_used = False
    for cl, color in zip(cells, COLORS):
        y_true, y_score = _load_predictions(pred_dir, cl)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax.plot(recall, precision, color=color, lw=2, label=f"{cl} (AUPR = {auc(recall, precision):.3f})")
        if not baseline_used:
            ax.axhline(y=y_true.mean(), color="gray", linestyle="--", lw=1, alpha=0.6,
                       label=f"Random baseline ({y_true.mean():.3f})")
            baseline_used = True
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — EPINTLM (All Cell Lines)")
    ax.legend(loc="upper right", fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "B_pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    logger.info("Saved %s", path)
    return path


def plot_model_comparison(pred_dir: Path, out_dir: Path) -> Path | None:
    cells = _available_cell_lines(pred_dir)
    if not cells:
        return None
    # Fill EPINTLM column with reproduced numbers
    for cl in cells:
        y_true, y_score = _load_predictions(pred_dir, cl)
        TABLE2_AUC[cl][-1] = round(roc_auc_score(y_true, y_score), 4)
        TABLE2_AUPR[cl][-1] = round(average_precision_score(y_true, y_score), 4)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    bar_colors = ["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500", "#e63946"]
    x = np.arange(len(cells))
    width = 0.13

    for ax, table, ylabel, title in [
        (axes[0], TABLE2_AUC, "AUROC", "Model Comparison — AUROC (Table 2)"),
        (axes[1], TABLE2_AUPR, "AUPR", "Model Comparison — AUPR (Table 2)"),
    ]:
        for i, (model_name, color) in enumerate(zip(MODELS, bar_colors)):
            vals = [table[cl][i] for cl in cells]
            offset = (i - len(MODELS) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=model_name, color=color, alpha=0.88)
            if model_name == "EPINTLM":
                for bar in bars:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(1.5)
        ax.set_xticks(x); ax.set_xticklabels(cells, fontsize=11)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8, loc="lower right"); ax.set_ylim([0.6, 1.05])
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    path = out_dir / "CD_model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    logger.info("Saved %s", path)
    return path


def plot_ablation(out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(CELL_LINES))
    width = 0.25
    colors = ["#adb5bd", "#6c757d", "#e63946"]

    for ax, mean_dict, std_dict, ylabel, title in [
        (axes[0], ABLATION_AUC_MEAN, ABLATION_AUC_STD, "AUROC", "Ablation Study — AUROC (Table 3)"),
        (axes[1], ABLATION_AUPR_MEAN, ABLATION_AUPR_STD, "AUPR", "Ablation Study — AUPR (Table 4)"),
    ]:
        for i, (variant, color) in enumerate(zip(ABLATION_VARIANTS, colors)):
            means = [mean_dict[cl][i] for cl in CELL_LINES]
            stds = [std_dict[cl][i] for cl in CELL_LINES]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, means, width, yerr=stds, label=variant, color=color, alpha=0.88, capsize=4)
            if variant.startswith("EPINTLM"):
                for bar in bars:
                    bar.set_edgecolor("black"); bar.set_linewidth(1.5)
        ax.set_xticks(x); ax.set_xticklabels(CELL_LINES, fontsize=11)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=9); ax.set_ylim([0.65, 1.02])
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    path = out_dir / "E_ablation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    logger.info("Saved %s", path)
    return path


def plot_reproduced_vs_paper(pred_dir: Path, out_dir: Path) -> Path | None:
    cells = _available_cell_lines(pred_dir)
    if not cells:
        return None
    repro_auc, repro_aupr = {}, {}
    for cl in cells:
        y_true, y_score = _load_predictions(pred_dir, cl)
        repro_auc[cl] = roc_auc_score(y_true, y_score)
        repro_aupr[cl] = average_precision_score(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(cells))
    width = 0.35
    for ax, paper, repro, ylabel, title in [
        (axes[0], PAPER_AUC, repro_auc, "AUROC", "Reproduced vs Paper — AUROC"),
        (axes[1], PAPER_AUPR, repro_aupr, "AUPR", "Reproduced vs Paper — AUPR"),
    ]:
        pv = [paper[cl] for cl in cells]
        rv = [repro[cl] for cl in cells]
        ax.bar(x - width / 2, pv, width, label="Paper (reported)", color="#457b9d", alpha=0.85)
        ax.bar(x + width / 2, rv, width, label="Reproduced",       color="#e63946", alpha=0.85)
        for xi, (p, r) in enumerate(zip(pv, rv)):
            diff = r - p
            sign = "+" if diff >= 0 else ""
            ax.text(xi + width / 2, r + 0.003, f"{sign}{diff:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(cells, fontsize=11)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=10); ax.set_ylim([0.65, 1.05])
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    path = out_dir / "F_reproduced_vs_paper.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    logger.info("Saved %s", path)
    return path


def generate_all_plots(pred_dir: Path, out_dir: Path) -> Dict[str, Path | None]:
    pred_dir = Path(pred_dir); out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "roc": plot_roc(pred_dir, out_dir),
        "pr":  plot_pr(pred_dir, out_dir),
        "compare": plot_model_comparison(pred_dir, out_dir),
        "ablation": plot_ablation(out_dir),
        "repro": plot_reproduced_vs_paper(pred_dir, out_dir),
    }
