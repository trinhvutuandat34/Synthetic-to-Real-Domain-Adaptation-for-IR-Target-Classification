# training/trainer.py
# ─────────────────────────────────────────────────────────────
# Training loop with:
#   - W&B experiment tracking
#   - Checkpoint saving to Google Drive
#   - Early stopping
#   - GPU augmentation via Kornia
#   - Learning rate scheduling
# ─────────────────────────────────────────────────────────────

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Trainer] W&B not installed — logging to console only.")

from utils.config import cfg
from data.augmentation import build_augmentation_pipeline, IRNoiseLayer


class Trainer:
    """
    Handles the full training lifecycle for one model.

    Usage:
        trainer = Trainer(model, run_name="baseline")
        best_ckpt = trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model:           nn.Module,
        run_name:        str          = cfg.run_name,
        learning_rate:   float        = cfg.learning_rate,
        weight_decay:    float        = cfg.weight_decay,
        epochs:          int          = cfg.epochs,
        patience:        int          = cfg.early_stop_patience,
        aug_level:       str          = "baseline",   # "baseline" | "extended" | "aggressive"
        checkpoint_dir:  str          = cfg.checkpoint_dir,
        use_wandb:       bool         = True,
    ):
        self.model          = model
        self.run_name       = run_name
        self.epochs         = epochs
        self.patience       = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb      = use_wandb and WANDB_AVAILABLE
        self.device         = next(model.parameters()).device

        # Optimiser: AdamW — same as the paper
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate, weight_decay=weight_decay
        )

        # Cosine annealing LR schedule
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        self.criterion = nn.CrossEntropyLoss()

        # GPU augmentation pipeline (Kornia)
        self.aug       = build_augmentation_pipeline(aug_level).to(self.device)
        self.ir_noise  = IRNoiseLayer().to(self.device)

    # ── Public API ────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> str:
        """
        Train the model. Returns path to best checkpoint.
        """
        if self.use_wandb:
            wandb.init(
                project=cfg.project_name,
                name=self.run_name,
                config={
                    "epochs":        self.epochs,
                    "lr":            self.optimizer.param_groups[0]["lr"],
                    "batch_size":    train_loader.batch_size,
                    "model":         cfg.model_name,
                    "aug_level":     "see trainer",
                    "num_classes":   cfg.num_classes,
                }
            )

        best_val_acc  = 0.0
        no_improve    = 0
        best_ckpt     = str(self.checkpoint_dir / f"{self.run_name}_best.pt")

        print(f"\n[Trainer] Starting '{self.run_name}' — {self.epochs} epochs")
        print(f"  train={len(train_loader.dataset)} | "
              f"val={len(val_loader.dataset)}")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            # ── Train ─────────────────────────────────────────
            train_loss, train_acc = self._train_epoch(train_loader)

            # ── Validate ──────────────────────────────────────
            val_loss,   val_acc   = self._val_epoch(val_loader)

            self.scheduler.step()
            elapsed = time.time() - t0

            # ── Log ───────────────────────────────────────────
            metrics = {
                "epoch":      epoch,
                "train/loss": round(train_loss, 4),
                "train/acc":  round(train_acc,  4),
                "val/loss":   round(val_loss,   4),
                "val/acc":    round(val_acc,    4),
                "lr":         self.optimizer.param_groups[0]["lr"],
            }
            if self.use_wandb:
                wandb.log(metrics)

            print(
                f"  Epoch {epoch:3d}/{self.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
                f"{elapsed:.1f}s"
            )

            # ── Checkpoint best model ─────────────────────────
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve   = 0
                torch.save(self.model.state_dict(), best_ckpt)
                print(f"  ✓ New best val_acc={best_val_acc:.4f} → saved")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"\n[Trainer] Early stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs)")
                    break

        if self.use_wandb:
            wandb.finish()

        print(f"\n[Trainer] Done. Best val_acc={best_val_acc:.4f}")
        print(f"  Checkpoint: {best_ckpt}")
        return best_ckpt

    # ── Private helpers ───────────────────────────────────────

    def _train_epoch(
        self, loader: DataLoader
    ) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(loader, desc="  train", leave=False):
            x, y = batch[0].to(self.device), batch[1].to(self.device)

            # GPU augmentation
            x = self.aug(x)
            x = self.ir_noise(x)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss   = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)

        return total_loss / len(loader), correct / total

    def _val_epoch(
        self, loader: DataLoader
    ) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="  val  ", leave=False):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(x)
                loss   = self.criterion(logits, y)

                total_loss += loss.item()
                correct    += (logits.argmax(1) == y).sum().item()
                total      += y.size(0)

        return total_loss / len(loader), correct / total
