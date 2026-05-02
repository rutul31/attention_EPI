"""JSON-backed metrics tracker with append + save semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class MetricsTracker:
    def __init__(self, fields: List[str] | None = None):
        defaults = ["epoch", "train_loss", "val_loss", "val_aupr", "val_auc", "learning_rate"]
        self.fields = fields or defaults
        self.metrics: Dict[str, list] = {f: [] for f in self.fields}

    def update(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self.metrics:
                self.metrics[k].append(v)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w") as f:
            json.dump(self.metrics, f, indent=2)
