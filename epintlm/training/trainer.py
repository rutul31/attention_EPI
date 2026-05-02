"""Trainer — config-driven training loop. Replaces trainepintlm.py."""

from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import register_safe_globals
from ..logging_utils import get_logger
from ..models.epintlm import EPIModel
from .attention_monitor import AttentionMonitor
from .early_stopping import EarlyStopping
from .metrics import MetricsTracker

logger = get_logger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _build_optimizer(cfg: Config, model: EPIModel) -> torch.optim.Optimizer:
    params = filter(lambda p: p.requires_grad, model.parameters())
    if cfg.training.optimizer.type.lower() != "adam":
        raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer.type}")
    return torch.optim.Adam(
        params, lr=cfg.training.optimizer.lr, weight_decay=cfg.training.optimizer.weight_decay
    )


def _build_scheduler(cfg: Config, optimizer: torch.optim.Optimizer):
    if cfg.training.scheduler.type.lower() != "multistep":
        raise ValueError(f"Unsupported scheduler: {cfg.training.scheduler.type}")
    return MultiStepLR(optimizer, milestones=cfg.training.scheduler.milestones, gamma=cfg.training.scheduler.gamma)


class Trainer:
    def __init__(
        self,
        cfg: Config,
        run_dir: Path,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(cfg.run.device if (cfg.run.device == "cpu" or torch.cuda.is_available()) else "cpu")
        if cfg.run.device == "cuda" and self.device.type == "cpu":
            logger.warning("CUDA requested but unavailable; falling back to CPU.")

        register_safe_globals()
        _set_seed(cfg.run.seed)

        self.model = EPIModel(cfg.model, embeddings_path=cfg.data.embeddings_path).to(self.device)
        self.model.set_embedding_trainable(False)  # frozen until unfreeze_epoch

        self.attention_monitor = AttentionMonitor(self.run_dir / "plots")
        self.metrics_tracker = MetricsTracker()
        self.early_stopping = EarlyStopping(
            patience=cfg.training.early_stopping.patience,
            min_delta=cfg.training.early_stopping.min_delta,
        )

        self.optimizer = _build_optimizer(cfg, self.model)
        self.scheduler = _build_scheduler(cfg, self.optimizer)

        self.best_aupr = 0.0
        self.best_auc = 0.0
        self.best_epoch = 0

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        all_probs, all_labels = [], []

        for batch_idx, (enh, pro, gene, labels) in enumerate(self.train_loader):
            enh = enh.to(self.device)
            pro = pro.to(self.device)
            gene = gene.to(self.device)
            labels = labels.to(self.device).float().view(-1, 1)

            self.optimizer.zero_grad()
            outputs, attention_dict = self.model(enh, pro, gene)

            if batch_idx == 0:
                stats = self.attention_monitor.record(attention_dict, epoch, batch_idx)
                if stats:
                    self.attention_monitor.update_epoch_stats(epoch, stats)

            outputs = outputs.view(-1, 1)
            loss = self.model.criterion(outputs, labels)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.training.grad_clip_max_norm)
            self.optimizer.step()

            with torch.no_grad():
                all_probs.append(outputs.view(-1).detach().cpu())
                all_labels.append(labels.view(-1).detach().cpu())

            if batch_idx % 50 == 0:
                _clear_cuda()

        probs = torch.cat(all_probs).numpy()
        lbls = torch.cat(all_labels).numpy()
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        avg_loss = total_loss / max(1, len(self.train_loader))
        try:
            auc = roc_auc_score(lbls, probs)
        except ValueError:
            auc = float("nan")
        aupr = average_precision_score(lbls, probs)
        return avg_loss, auc, aupr

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch_idx, (enh, pro, gene, labels) in enumerate(self.val_loader):
            enh = enh.to(self.device)
            pro = pro.to(self.device)
            gene = gene.to(self.device)
            labels = labels.to(self.device).float().view(-1, 1)

            outputs, _ = self.model(enh, pro, gene)
            outputs = outputs.view(-1, 1)
            loss = self.model.criterion(outputs, labels)
            total_loss += loss.item()

            all_preds.append(outputs.view(-1).cpu().numpy())
            all_labels.append(labels.view(-1).cpu().numpy())

        preds = np.concatenate(all_preds)
        lbls = np.concatenate(all_labels)
        avg_loss = total_loss / max(1, len(self.val_loader))
        aupr = average_precision_score(lbls, preds)
        auc = roc_auc_score(lbls, preds)
        acc = accuracy_score((lbls >= 0.5).astype(int), (preds >= 0.5).astype(int))
        return avg_loss, aupr, auc, acc

    def _save_checkpoint(self, epoch: int, val_aupr: float, val_auc: float, name: str) -> Path:
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_aupr": val_aupr,
                "val_auc": val_auc,
            },
            path,
        )
        return path

    def fit(self) -> dict:
        cfg = self.cfg
        for epoch in range(cfg.training.num_epochs):
            logger.info("Epoch %d/%d", epoch, cfg.training.num_epochs - 1)

            if epoch == cfg.training.fine_tune.unfreeze_epoch:
                logger.info("Unfreezing embeddings; switching to fine-tune LR.")
                self.model.set_embedding_trainable(True)
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=cfg.training.fine_tune.lr,
                    weight_decay=cfg.training.optimizer.weight_decay,
                )
                self.scheduler = _build_scheduler(cfg, self.optimizer)

            train_loss, train_auc, train_aupr = self.train_epoch(epoch)
            val_loss, val_aupr, val_auc, val_acc = self.validate(epoch)
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Train loss=%.4f auc=%.4f aupr=%.4f | Val loss=%.4f auc=%.4f aupr=%.4f acc=%.4f | lr=%.6f",
                train_loss, train_auc, train_aupr, val_loss, val_auc, val_aupr, val_acc, current_lr,
            )

            self.metrics_tracker.update(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_aupr=val_aupr,
                val_auc=val_auc,
                learning_rate=current_lr,
            )

            if epoch % cfg.training.checkpoint_freq == 0:
                self._save_checkpoint(epoch, val_aupr, val_auc, f"epoch_{epoch}.pt")
            self._save_checkpoint(epoch, val_aupr, val_auc, "last.pt")

            improved = (val_auc > self.best_auc) or (
                val_auc == self.best_auc and val_aupr >= self.best_aupr
            )
            if improved:
                self.best_auc = val_auc
                self.best_aupr = val_aupr
                self.best_epoch = epoch
                self._save_checkpoint(epoch, val_aupr, val_auc, "best.pt")
                logger.info("New best at epoch %d: AUC=%.4f AUPR=%.4f", epoch, val_auc, val_aupr)

            self.early_stopping(val_aupr)
            if self.early_stopping.early_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break

            if (epoch + 1) % cfg.training.plot_freq == 0:
                self.attention_monitor.plot_evolution()
                self.attention_monitor.plot_heatmaps()

            _clear_cuda()

        # Finalize
        self.attention_monitor.plot_evolution()
        self.attention_monitor.plot_heatmaps()
        self.attention_monitor.save_summary()
        self.metrics_tracker.save(self.run_dir / "metrics" / "training_metrics.json")

        logger.info("Training complete. Best AUC=%.4f AUPR=%.4f at epoch %d",
                    self.best_auc, self.best_aupr, self.best_epoch)
        return {"best_auc": self.best_auc, "best_aupr": self.best_aupr, "best_epoch": self.best_epoch}
