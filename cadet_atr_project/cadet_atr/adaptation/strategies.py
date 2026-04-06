# adaptation/strategies.py
# ─────────────────────────────────────────────────────────────
# All four domain adaptation strategies for the Cadet ATR project:
#   Strategy 1 — Histogram matching
#   Strategy 2 — Domain randomisation (handled in augmentation.py;
#                BackgroundSwapDataset left as a stub here)
#   Strategy 3 — Fine-tuning on real IR data (RealDataFinetuner)
#   Strategy 4 — DANN adversarial adaptation (DANNTrainer + DANNModel)
# ─────────────────────────────────────────────────────────────

import os
import glob
import itertools
import math
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from skimage import exposure
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as tv_models

from utils.config import cfg
from models.convnext import build_model

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
# STRATEGY 1 — Histogram Matching
# ══════════════════════════════════════════════════════════════
#
# The idea: real IR sensors have characteristic intensity
# distributions that differ from Stable Diffusion outputs.
# By matching synthetic pixel distributions to the real
# reference histogram, we close part of the gap before training.
#
# build_reference_histogram() builds one "target" distribution
# from ALL real images combined (more samples → more reliable).
# apply_histogram_matching() then remaps every synthetic image
# to that distribution using skimage's histogram specification.

def build_reference_histogram(real_dir: str) -> np.ndarray:
    """
    Compute a cumulative intensity histogram from all real IR images.

    We pool all images into one big pixel set before computing the
    histogram — this avoids any single image dominating the reference.

    Args:
        real_dir: root directory; scans all subdirectories for PNG/JPG.

    Returns:
        hist: float32 ndarray of shape (256,), normalised so sum == 1.
    """
    pixel_counts = np.zeros(256, dtype=np.float64)
    total_images = 0

    # Walk every class subdirectory under real_dir
    pattern_png = os.path.join(real_dir, "**", "*.png")
    pattern_jpg = os.path.join(real_dir, "**", "*.jpg")
    all_paths   = glob.glob(pattern_png, recursive=True) + \
                  glob.glob(pattern_jpg, recursive=True)

    if not all_paths:
        raise FileNotFoundError(
            f"No PNG/JPG images found under '{real_dir}'. "
            "Check cfg.real_dir and make sure real images are in place."
        )

    for path in tqdm(all_paths, desc="[Histogram] Scanning real images"):
        img = np.array(Image.open(path).convert("L"))  # grayscale uint8
        counts, _ = np.histogram(img, bins=256, range=(0, 255))
        pixel_counts += counts
        total_images += 1

    # Normalise so it sums to 1 — this is the probability distribution
    # that skimage.exposure.match_histograms will use as its target.
    hist = (pixel_counts / pixel_counts.sum()).astype(np.float32)
    print(f"[Histogram] Reference built from {total_images} real images.")
    return hist


def apply_histogram_matching(
    synth_img_dir: str,
    output_dir:    str,
    reference:     np.ndarray,
) -> None:
    """
    Remap every synthetic image's intensity distribution to match
    the real-image reference histogram.

    Directory structure is preserved:
        synth_img_dir/aircraft/00001.png
            → output_dir/aircraft/00001.png

    Args:
        synth_img_dir: source directory with class subdirectories.
        output_dir:    destination directory (created if absent).
        reference:     float32 (256,) histogram from build_reference_histogram.
    """
    pattern_png = os.path.join(synth_img_dir, "**", "*.png")
    pattern_jpg = os.path.join(synth_img_dir, "**", "*.jpg")
    all_paths   = glob.glob(pattern_png, recursive=True) + \
                  glob.glob(pattern_jpg, recursive=True)

    if not all_paths:
        raise FileNotFoundError(
            f"No synthetic images found under '{synth_img_dir}'."
        )

    n_matched = 0
    for src_path in tqdm(all_paths, desc="[Histogram] Matching images"):
        # Mirror the subdirectory structure under output_dir
        rel_path = os.path.relpath(src_path, synth_img_dir)
        dst_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Load as grayscale uint8
        img = np.array(Image.open(src_path).convert("L"))

        # skimage's match_histograms takes multichannel=False for grayscale.
        # It computes the CDF of img and maps it to match reference's CDF.
        matched = exposure.match_histograms(img, reference)
        matched = np.clip(matched, 0, 255).astype(np.uint8)

        Image.fromarray(matched).save(dst_path)
        n_matched += 1

    print(f"[Histogram] Matched {n_matched} images → {output_dir}")


# ══════════════════════════════════════════════════════════════
# STRATEGY 3 — Fine-tuning on Real IR Data
# ══════════════════════════════════════════════════════════════
#
# The key insight: after synthetic pre-training the backbone
# already knows what "an aircraft shape" looks like; we only
# need to adapt the final feature representations to real sensor
# noise. Head-only fine-tuning with a small real dataset
# (< 200 images/class) is much less prone to catastrophic
# forgetting than full-model fine-tuning.

class RealDataFinetuner:
    """
    Fine-tunes a pre-trained ConvNeXt on real IR images.

    Three modes (controlled by `finetune()` argument):
        head_only   — freeze backbone, train classifier head only.
                      Best for very small real datasets (< 200/class).
        full        — unfreeze everything with a tiny LR (finetune_lr_full).
                      Risk of catastrophic forgetting; use with caution.
        layer_wise  — starts head-only, progressively unfreezes earlier
                      stages every 5 epochs. Best of both worlds.
    """

    def __init__(
        self,
        model:           nn.Module,
        checkpoint_path: str,
        save_path:       str,
        epochs:          int = cfg.finetune_epochs,
        patience:        int = 10,
    ):
        self.model     = model
        self.save_path = save_path
        self.epochs    = epochs
        self.patience  = patience
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.criterion = nn.CrossEntropyLoss()

        # Load the pre-trained checkpoint (e.g. baseline or domain_random best)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device)
        print(f"[Finetune] Loaded checkpoint: {checkpoint_path}")

    # ── Parameter control helpers ─────────────────────────────

    def _freeze_all(self) -> None:
        """Freeze every parameter so we can selectively unfreeze."""
        for p in self.model.parameters():
            p.requires_grad = False

    def _unfreeze_head(self) -> None:
        """Unfreeze the classifier head only (ConvNeXt: model.classifier)."""
        for p in self.model.classifier.parameters():
            p.requires_grad = True

    def _unfreeze_all(self) -> None:
        """Unfreeze everything — used for full fine-tuning mode."""
        for p in self.model.parameters():
            p.requires_grad = True

    def _unfreeze_next_block(self, epoch: int) -> None:
        """
        Progressive unfreezing: every 5 epochs, expose one earlier
        ConvNeXt stage to gradient updates.

        ConvNeXt-tiny has 4 stages in model.features (indices 0–3,
        with some interleaved downsampling layers). We count from the
        end so that the richest semantic stages train first.
        """
        stage_idx = -(epoch // 5)  # e.g. epoch 5 → -1 (last stage)
        try:
            block = self.model.features[stage_idx]
            for p in block.parameters():
                p.requires_grad = True
            print(f"  [Finetune] Unfrozen stage index {stage_idx}")
        except IndexError:
            # All stages already unfrozen — nothing to do
            pass

    def _quick_eval(self, loader: DataLoader) -> float:
        """Return accuracy on a loader without computing gradients."""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                preds = self.model(x).argmax(1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        return correct / total if total > 0 else 0.0

    # ── Core fine-tuning loop (from approved patch — verbatim) ──

    def finetune(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        mode:         str = "head_only",
    ) -> nn.Module:
        """
        Fine-tune the model on real IR training data.
        Returns the adapted model with best validation accuracy.
        """
        print(f"\n[Finetune] Mode: {mode} | "
              f"train={len(train_loader.dataset)} real images")

        if mode == "head_only":
            self._freeze_all()
            self._unfreeze_head()
            lr = cfg.finetune_lr

        elif mode == "full":
            self._unfreeze_all()
            lr = cfg.finetune_lr_full

        elif mode == "layer_wise":
            self._freeze_all()
            self._unfreeze_head()
            lr = cfg.finetune_lr

        else:
            raise ValueError(f"Unknown fine-tune mode: {mode}")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        best_val_acc = 0.0
        no_improve   = 0

        for epoch in range(1, self.epochs + 1):

            if mode == "layer_wise" and epoch % 5 == 0:
                self._unfreeze_next_block(epoch)
                # Rebuild optimizer so newly unfrozen params are included
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=lr * 0.1, weight_decay=cfg.weight_decay
                )

            train_loss, train_acc = self._train_epoch(train_loader, optimizer)
            scheduler.step()

            val_acc = self._quick_eval(val_loader)
            print(f"  Epoch {epoch:2d} | val_acc={val_acc:.4f} | "
                  f"train_acc={train_acc:.4f} | train_loss={train_loss:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve   = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"  ✓ Saved best → {self.save_path}")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"  Early stop at epoch {epoch}")
                    break

        self.model.load_state_dict(
            torch.load(self.save_path, map_location=self.device)
        )
        print(f"\n[Finetune] Done. Best val_acc={best_val_acc:.4f}")
        return self.model

    def _train_epoch(
        self,
        loader:    DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> tuple:
        """
        Single training epoch.

        FIX: logits are computed ONCE per batch and reused for both
        loss calculation and accuracy tracking.  The original code
        called self.model(x) a second time inside the accuracy line,
        doubling the forward-pass compute and potentially giving
        inconsistent results when dropout is active.
        """
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(loader, desc="  finetune", leave=False):
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)

            optimizer.zero_grad()

            # ── Single forward pass — reuse logits for both loss and acc ──
            logits = self.model(x)
            loss   = self.criterion(logits, y)
            loss.backward()
            optimizer.step()

            # No second self.model(x) here — use the logits we already have
            total_loss += loss.item()
            correct    += (logits.detach().argmax(1) == y).sum().item()
            total      += y.size(0)

        return total_loss / len(loader), correct / total


# ══════════════════════════════════════════════════════════════
# STRATEGY 4 — DANN Adversarial Domain Adaptation
# ══════════════════════════════════════════════════════════════
#
# Theory in one paragraph:
# DANN (Ganin et al., 2016) adds a domain discriminator branch
# to the backbone and trains it adversarially via the Gradient
# Reversal Layer (GRL). The class branch sees normal gradients.
# The domain branch also sees normal gradients — but the GRL
# flips the sign before they reach the backbone. This means:
#   - The domain branch LEARNS to tell synth from real.
#   - The backbone UNLEARNS domain-specific features (because
#     reversed gradients push it toward domain confusion).
# At inference, only the class branch runs — zero overhead.


# ── Gradient Reversal Layer ───────────────────────────────────

class _GRLFunction(torch.autograd.Function):
    """
    Custom autograd Function implementing gradient reversal.

    Forward:  identity  (x → x, no change to values)
    Backward: multiply incoming gradient by -lambda_val

    We use autograd.Function (not nn.Module) because we need to
    customise the backward pass — nn.Module doesn't expose that.
    lambda_val is passed via the `ctx` context object so the
    backward step can read it without a global variable.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        # Save lambda for use in backward
        ctx.lambda_val = lambda_val
        # forward is a pure identity — return x unchanged
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Flip sign and scale — this is the "reversal"
        # We must return a gradient for each input to forward():
        #   grad for x         → -lambda * grad_output
        #   grad for lambda_val → None  (it's a float, not a tensor)
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Thin nn.Module wrapper around _GRLFunction so it can sit
    inside an nn.Sequential without special handling.
    """

    def __init__(self, lambda_val: float = 1.0):
        super().__init__()
        # lambda is mutable so the trainer can ramp it up over epochs
        self.lambda_val = lambda_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFunction.apply(x, self.lambda_val)

    def set_lambda(self, lambda_val: float) -> None:
        self.lambda_val = lambda_val


# ── DANN Model ────────────────────────────────────────────────

class DANNModel(nn.Module):
    """
    Domain-Adversarial Neural Network built on ConvNeXt-tiny.

    Architecture:
        input (B, 3, 224, 224)
            ↓
        backbone: ConvNeXt features → AdaptiveAvgPool2d → Flatten
            → (B, 768)
            ↓          ↓
        class_head   domain_head (GRL → Linear → ReLU → Dropout → Linear)
        (B, num_classes)   (B, 2)   ← only active during training

    In inference mode (return_domain=False), the domain head is
    never called — identical cost to a plain ConvNeXt.
    """

    def __init__(self, num_classes: int = cfg.num_classes):
        super().__init__()

        # Build a fresh ConvNeXt-tiny to steal its feature extractor
        _base = tv_models.convnext_tiny(weights="IMAGENET1K_V1")

        # backbone: features + pool + flatten  →  768-d vector
        # We wrap these three operations in a Sequential so that
        # _extract_features() in visualise.py can hook model.backbone
        # directly and get the flat 768-d output (see utils/visualise.py).
        self.backbone = nn.Sequential(
            _base.features,                  # ConvNeXt stages 0-7
            nn.AdaptiveAvgPool2d((1, 1)),    # (B, 768, 1, 1)
            nn.Flatten(),                    # (B, 768)
        )

        # Class prediction head — standard cross-entropy target
        self.class_classifier = nn.Linear(768, num_classes)

        # Domain prediction head — binary: 0=synthetic, 1=real
        # The GRL sits at the START of this branch so that gradients
        # flowing back from the domain loss are reversed before they
        # reach self.backbone.
        self.grl = GradientReversalLayer(lambda_val=1.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )

    def forward(
        self,
        x:             torch.Tensor,
        return_domain: bool  = False,
        lambda_val:    float = 1.0,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:             input images, shape (B, 3, H, W)
            return_domain: if True, return (class_logits, domain_logits)
                           if False (inference), return class_logits only
            lambda_val:    GRL reversal strength; caller ramps this up
                           over training epochs via the Ganin schedule

        Returns:
            class_logits alone  (B, num_classes)  — when return_domain=False
            or
            (class_logits, domain_logits)          — when return_domain=True
        """
        # Always run the backbone — this is shared by both heads
        features = self.backbone(x)  # (B, 768)

        class_logits = self.class_classifier(features)

        if not return_domain:
            # Pure inference path: domain head never executes
            return class_logits

        # Training path: apply GRL then domain head
        self.grl.set_lambda(lambda_val)
        reversed_features = self.grl(features)
        domain_logits     = self.domain_classifier(reversed_features)

        return class_logits, domain_logits


# ── DANN Trainer ──────────────────────────────────────────────

class DANNTrainer:
    """
    Orchestrates DANN adversarial training.

    Each epoch:
      1. Zip synth and real batches (cycling the shorter loader).
      2. Forward: class loss on synth + domain loss on both.
      3. Backward: gradients flow normally to both heads, but the GRL
         reverses domain gradients before they reach the backbone.
      4. Lambda ramp: Ganin 2016 schedule ramps λ from 0 → lambda_max
         so the domain signal is weak early (when class loss dominates)
         and strong later (when features are almost domain-invariant).

    Checkpoint: best model by synthetic validation accuracy.
    """

    def __init__(
        self,
        backbone_checkpoint: str,
        synth_loader:        DataLoader,
        real_loader:         DataLoader,
        val_loader:          DataLoader,
        save_path:           str,
        wandb_run_name:      str = "dann",
    ):
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"
        self.synth_loader = synth_loader
        self.real_loader  = real_loader
        self.val_loader   = val_loader
        self.save_path    = save_path

        # Build DANN model
        self.model = DANNModel(num_classes=cfg.num_classes).to(self.device)

        # Load only backbone weights from the pre-trained checkpoint.
        # strict=False allows us to ignore the class/domain head keys
        # that aren't present in the saved ConvNeXt checkpoint.
        checkpoint_state = torch.load(backbone_checkpoint, map_location=self.device)

        # The checkpoint may come from a plain ConvNeXt (keys like
        # "features.0.weight") or from a previous DANNModel
        # (keys like "backbone.0.0.weight").  We remap if needed.
        remapped = {}
        for k, v in checkpoint_state.items():
            if k.startswith("backbone."):
                remapped[k] = v           # already namespaced correctly
            elif k.startswith("features."):
                # plain ConvNeXt checkpoint → prepend "backbone.0."
                remapped["backbone.0." + k[len("features."):]] = v
            # classifier/fc keys are intentionally dropped (strict=False)

        missing, unexpected = self.model.load_state_dict(remapped, strict=False)
        print(f"[DANN] Backbone loaded. Missing keys: {len(missing)}, "
              f"Unexpected keys: {len(unexpected)}")

        # Two optimizers: one per logical branch.
        # Sharing a single optimizer across both would work, but
        # separate ones let us set different LRs easily in future.
        self.class_optimizer  = torch.optim.AdamW(
            list(self.model.backbone.parameters()) +
            list(self.model.class_classifier.parameters()),
            lr=cfg.finetune_lr, weight_decay=cfg.weight_decay,
        )
        self.domain_optimizer = torch.optim.AdamW(
            self.model.domain_classifier.parameters(),
            lr=cfg.finetune_lr, weight_decay=cfg.weight_decay,
        )

        self.criterion = nn.CrossEntropyLoss()

        # Optional W&B run
        self.use_wandb = WANDB_AVAILABLE
        if self.use_wandb:
            try:
                wandb.init(
                    project=cfg.project_name,
                    name=wandb_run_name,
                    config={
                        "dann_epochs":    cfg.dann_epochs,
                        "lambda_max":     cfg.dann_lambda_max,
                        "finetune_lr":    cfg.finetune_lr,
                        "num_classes":    cfg.num_classes,
                    }
                )
            except Exception:
                self.use_wandb = False

    @staticmethod
    def _ganin_lambda(epoch: int, total_epochs: int, lambda_max: float) -> float:
        """
        Ganin et al. (2016) λ schedule.

        p goes from 0 → 1 over training.
        lambda = lambda_max * (2 / (1 + exp(-10 * p)) - 1)

        This S-curve ramp means:
          - Early epochs: λ ≈ 0  (class loss dominates, backbone stabilises)
          - Late epochs:  λ → lambda_max  (domain adversary has full strength)
        """
        p      = epoch / total_epochs
        return lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

    def _eval_accuracy(self) -> float:
        """Quick accuracy check on the synthetic validation set."""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                preds = self.model(x, return_domain=False).argmax(1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        return correct / total if total > 0 else 0.0

    def train(self) -> nn.Module:
        """
        Run DANN training and return the best model.

        Returns:
            self.model with best validation checkpoint loaded.
        """
        total_epochs = cfg.dann_epochs
        best_val_acc = 0.0
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)

        print(f"\n[DANN] Starting training — {total_epochs} epochs")
        print(f"  synth={len(self.synth_loader.dataset)} | "
              f"real={len(self.real_loader.dataset)}")

        for epoch in range(1, total_epochs + 1):
            self.model.train()

            # Compute this epoch's lambda (ramps 0 → lambda_max)
            lambda_val = self._ganin_lambda(epoch, total_epochs, cfg.dann_lambda_max)

            total_class_loss  = 0.0
            total_domain_loss = 0.0
            correct_class     = 0
            total_samples     = 0
            n_batches         = 0

            # cycle() on the shorter loader so we always have a pair.
            # itertools.cycle re-iterates infinitely; zip stops at the
            # length of the longer (non-cycled) loader.
            if len(self.synth_loader) >= len(self.real_loader):
                paired = zip(
                    self.synth_loader,
                    itertools.cycle(self.real_loader),
                )
            else:
                paired = zip(
                    itertools.cycle(self.synth_loader),
                    self.real_loader,
                )

            for synth_batch, real_batch in tqdm(
                paired,
                total=max(len(self.synth_loader), len(self.real_loader)),
                desc=f"  DANN epoch {epoch:2d}",
                leave=False,
            ):
                x_s = synth_batch[0].to(self.device)  # synth images
                y_s = synth_batch[1].to(self.device)  # class labels (synth)
                x_r = real_batch[0].to(self.device)   # real images
                # real images have no class labels in domain adaptation;
                # only domain labels (label=1) are used for real data.

                # ── Forward pass ──────────────────────────────────
                # Synth: need both class AND domain logits
                cls_logits_s, dom_logits_s = self.model(
                    x_s, return_domain=True, lambda_val=lambda_val
                )

                # Real: only domain logits needed (no class supervision)
                _, dom_logits_r = self.model(
                    x_r, return_domain=True, lambda_val=lambda_val
                )

                # ── Losses ────────────────────────────────────────
                # Classification loss: only on synthetic (labelled) images
                class_loss = self.criterion(cls_logits_s, y_s)

                # Domain loss: synth=0, real=1
                # Concatenate both domain predictions and their labels
                dom_preds  = torch.cat([dom_logits_s, dom_logits_r], dim=0)
                dom_labels = torch.cat([
                    torch.zeros(x_s.size(0), dtype=torch.long, device=self.device),
                    torch.ones( x_r.size(0), dtype=torch.long, device=self.device),
                ], dim=0)
                domain_loss = self.criterion(dom_preds, dom_labels)

                # Total loss — domain loss contributes via GRL-reversed grads
                total_loss = class_loss + domain_loss

                # ── Backward ──────────────────────────────────────
                self.class_optimizer.zero_grad()
                self.domain_optimizer.zero_grad()
                total_loss.backward()
                self.class_optimizer.step()
                self.domain_optimizer.step()

                # ── Metrics accumulation ──────────────────────────
                total_class_loss  += class_loss.item()
                total_domain_loss += domain_loss.item()
                correct_class     += (cls_logits_s.detach().argmax(1) == y_s).sum().item()
                total_samples     += y_s.size(0)
                n_batches         += 1

            # ── Epoch summary ─────────────────────────────────────
            train_acc  = correct_class / total_samples if total_samples else 0.0
            avg_cls    = total_class_loss  / n_batches
            avg_dom    = total_domain_loss / n_batches
            val_acc    = self._eval_accuracy()

            print(
                f"  Epoch {epoch:2d}/{total_epochs} | λ={lambda_val:.3f} | "
                f"cls_loss={avg_cls:.4f} | dom_loss={avg_dom:.4f} | "
                f"train_acc={train_acc:.3f} | val_acc={val_acc:.4f}"
            )

            if self.use_wandb:
                try:
                    wandb.log({
                        "epoch":      epoch,
                        "lambda":     lambda_val,
                        "cls_loss":   avg_cls,
                        "dom_loss":   avg_dom,
                        "train_acc":  train_acc,
                        "val_acc":    val_acc,
                    })
                except Exception:
                    pass

            # ── Checkpoint ────────────────────────────────────────
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.save_path)
                print(f"  ✓ Saved best → {self.save_path} (val_acc={val_acc:.4f})")

        # Load best weights before returning
        self.model.load_state_dict(
            torch.load(self.save_path, map_location=self.device)
        )
        print(f"\n[DANN] Training complete. Best val_acc={best_val_acc:.4f}")

        if self.use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass

        return self.model
