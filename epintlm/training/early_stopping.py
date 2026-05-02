"""Patience-based early stopping callback. Stops when monitored score plateaus."""

from __future__ import annotations

from ..logging_utils import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False

    def __call__(self, score: float) -> None:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return
        self.counter += 1
        logger.info("EarlyStopping counter: %d/%d", self.counter, self.patience)
        if self.counter >= self.patience:
            self.early_stop = True
            logger.info("Early stopping triggered.")
