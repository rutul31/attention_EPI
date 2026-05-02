"""Records attention statistics across epochs and renders evolution + heatmap plots."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)


class AttentionMonitor:
    def __init__(self, save_dir: str | Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.attention_stats: Dict = {}
        self.sample_weights: Dict[str, list] = {}
        self.epoch_metrics: List[dict] = []

    def record(self, attention_dict: dict, epoch: int, batch_idx: int = 0) -> dict:
        stats = {}
        for key, attn in attention_dict.items():
            if attn is None:
                continue
            t = attn.detach().cpu()
            if t.numel() == 0:
                continue

            stats[key] = {
                "mean": float(t.mean()),
                "std": float(t.std()),
                "max": float(t.max()),
                "min": float(t.min()),
                "sparsity": float((t < 0.01).float().mean()),
                "shape": str(tuple(t.shape)),
            }

            if batch_idx == 0:
                sample = t[0] if t.dim() > 0 else t
                self.sample_weights.setdefault(key, []).append(
                    {"epoch": epoch, "weight": sample.numpy()}
                )
        return stats

    def update_epoch_stats(self, epoch: int, stats: dict) -> None:
        row: Dict[str, float | int] = {"epoch": epoch}
        for key, vals in stats.items():
            for stat_name, value in vals.items():
                if stat_name == "shape":
                    continue
                row[f"{key}_{stat_name}"] = value
        self.epoch_metrics.append(row)

    def plot_evolution(self) -> Optional[Path]:
        if not self.epoch_metrics:
            return None
        df = pd.DataFrame(self.epoch_metrics)
        attention_types: set[str] = set()
        for col in df.columns:
            if col != "epoch" and "_" in col:
                attention_types.add(col.rsplit("_", 1)[0])
        if not attention_types:
            return None

        types_list = sorted(attention_types)
        fig, axes = plt.subplots(len(types_list), 1, figsize=(12, 4 * len(types_list)))
        if len(types_list) == 1:
            axes = [axes]
        for ax, attn_type in zip(axes, types_list):
            for metric in ("mean", "std", "sparsity"):
                col = f"{attn_type}_{metric}"
                if col in df.columns:
                    ax.plot(df["epoch"], df[col], label=metric, marker="o", markersize=3)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.set_title(f"{attn_type} Evolution")
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = self.save_dir / "attention_evolution.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Saved attention evolution → %s", path)
        return path

    def plot_heatmaps(self) -> None:
        for attn_type, weights_list in self.sample_weights.items():
            if not weights_list:
                continue
            n = len(weights_list)
            indices = [0, n // 3, 2 * n // 3, n - 1] if n >= 4 else list(range(n))
            indices = sorted(set(indices))

            fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 5))
            if len(indices) == 1:
                axes = [axes]
            fig.suptitle(f"{attn_type} Attention Heatmaps", fontsize=14)

            for ax, sample_idx in zip(axes, indices):
                w = weights_list[sample_idx]["weight"]
                if w.ndim > 2:
                    w = w.mean(axis=0)
                elif w.ndim == 1:
                    side = int(np.sqrt(len(w)))
                    if side * side == len(w):
                        w = w.reshape(side, side)
                    else:
                        w = w.reshape(-1, 1)
                im = ax.imshow(w, cmap="hot", aspect="auto", interpolation="nearest")
                ax.set_title(f"Epoch {weights_list[sample_idx]['epoch']}")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            path = self.save_dir / f"{attn_type}_heatmaps.png"
            plt.savefig(path, dpi=150)
            plt.close()
            logger.info("Saved heatmap → %s", path)

    def save_summary(self) -> Path:
        path = self.save_dir / "attention_summary.json"
        with path.open("w") as f:
            json.dump({"total_epochs": len(self.epoch_metrics), "metrics": self.epoch_metrics}, f, indent=2)
        return path
