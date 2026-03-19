# adaptation/strategies.py
# ─────────────────────────────────────────────────────────────
# All four domain adaptation strategies in one file:
#
#   Strategy 1 — Histogram matching
#   Strategy 2 — Domain randomisation (extended augmentation)
#   Strategy 3 — Fine-tuning on real images
#   Strategy 4 — DANN (Domain-Adversarial Neural Network)
#
# Run them in order. Record the gap before and after each.
# Hand Strategy 4 checkpoint to 이성제 for gap evaluation.
# ─────────────────────────────────────────────────────────────

import os
import glob
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from skimage.exposure import match_histograms
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from utils.config import cfg
from data.augmentation import build_augmentation_pipeline, IRNoiseLayer


# ═══════════════════════════════════════════════════════════════
# STRATEGY 1 — Histogram Matching
# ═══════════════════════════════════════════════════════════════

def build_reference_histogram(
    real_img_dir: str,
    n_samples:    int = 100,
) -> np.ndarray:
    """
    Build a reference pixel distribution from real IR images.
    This is what we'll match synthetic images to.

    Args:
        real_img_dir: path to any real IR image folder
        n_samples:    how many real images to average over

    Returns:
        reference_image: a representative real IR image array
                         (used by skimage.match_histograms)
    """
    paths = glob.glob(os.path.join(real_img_dir, "**", "*.png"), recursive=True)
    paths += glob.glob(os.path.join(real_img_dir, "**", "*.jpg"), recursive=True)

    if not paths:
        raise FileNotFoundError(f"No images found in {real_img_dir}")

    paths = paths[:n_samples]
    arrays = [np.array(Image.open(p).convert("L")) for p in paths]

    # Average pixel values as the reference distribution
    reference = np.mean(arrays, axis=0).astype(np.uint8)
    print(f"[HistMatch] Built reference from {len(arrays)} real images.")
    return reference


def apply_histogram_matching(
    synth_img_dir:  str,
    output_dir:     str,
    reference:      np.ndarray,
) -> None:
    """
    Apply histogram matching to all synthetic images.
    Matched images are saved to output_dir.

    After running this, re-point your DataLoader to output_dir
    and retrain from scratch. Then re-measure the gap.

    Args:
        synth_img_dir: root of your synthetic dataset
        output_dir:    where to save matched images
        reference:     reference array from build_reference_histogram()
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("Run: pip install scikit-image")

    paths = glob.glob(
        os.path.join(synth_img_dir, "**", "*.png"), recursive=True
    )
    paths += glob.glob(
        os.path.join(synth_img_dir, "**", "*.jpg"), recursive=True
    )

    print(f"[HistMatch] Matching {len(paths)} synthetic images...")

    for path in tqdm(paths):
        img     = np.array(Image.open(path).convert("L"))
        matched = match_histograms(img, reference, channel_axis=None)
        matched = matched.astype(np.uint8)

        # Mirror directory structure in output
        rel_path = os.path.relpath(path, synth_img_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(matched).save(out_path)

    print(f"[HistMatch] Done. Matched images → {output_dir}")
    print("  Next: retrain your model using the matched images as training data.")
    print("  Then call measure_domain_gap() again to see the improvement.")


class HistogramMatchTransform:
    """
    On-the-fly histogram matching as a PyTorch transform.
    More convenient than pre-processing all files, but slower.

    Usage:
        reference = build_reference_histogram("data/real/")
        transform = transforms.Compose([
            ...
            HistogramMatchTransform(reference),
            transforms.ToTensor(),
            ...
        ])
    """

    def __init__(self, reference: np.ndarray):
        if not SKIMAGE_AVAILABLE:
            raise ImportError("Run: pip install scikit-image")
        self.reference = reference

    def __call__(self, img: Image.Image) -> Image.Image:
        arr     = np.array(img.convert("L"))
        matched = match_histograms(arr, self.reference, channel_axis=None)
        return Image.fromarray(matched.astype(np.uint8))


# ═══════════════════════════════════════════════════════════════
# STRATEGY 2 — Domain Randomisation
# ═══════════════════════════════════════════════════════════════
#
# Implementation is in data/augmentation.py — use:
#     aug = build_augmentation_pipeline("extended")  or "aggressive"
#
# The functions below handle background replacement, which is
# the most effective domain randomisation technique for IR.

def paste_target_on_real_background(
    synth_img:   np.ndarray,
    bg_img:      np.ndarray,
    threshold:   int = 30,
) -> np.ndarray:
    """
    Cut-and-paste augmentation: paste the synthetic target
    onto a real IR background.

    Assumes target pixels are brighter than threshold
    (typical for IR — hot objects on cooler background).

    Args:
        synth_img:  H×W grayscale synthetic image
        bg_img:     H×W grayscale real IR background crop
        threshold:  pixel value above which = target (not background)

    Returns:
        composite: H×W grayscale image
    """
    bg = np.array(Image.fromarray(bg_img).resize(
        (synth_img.shape[1], synth_img.shape[0])
    ))
    target_mask = synth_img > threshold
    composite   = bg.copy()
    composite[target_mask] = synth_img[target_mask]
    return composite


class BackgroundSwapDataset(torch.utils.data.Dataset):
    """
    Wraps a synthetic dataset and randomly swaps backgrounds
    with crops from real IR images.

    This is the most powerful domain randomisation technique —
    it removes the "clean background" shortcut the model would
    otherwise learn.

    Usage:
        real_bg_paths = glob.glob("data/real/**/*.png", recursive=True)
        ds = BackgroundSwapDataset(synth_dataset, real_bg_paths)
    """

    def __init__(
        self,
        synth_dataset:   torch.utils.data.Dataset,
        real_bg_paths:   List[str],
        swap_prob:       float = 0.5,
        threshold:       int   = 30,
    ):
        self.synth      = synth_dataset
        self.bg_paths   = real_bg_paths
        self.swap_prob  = swap_prob
        self.threshold  = threshold

    def __len__(self) -> int:
        return len(self.synth)

    def __getitem__(self, idx: int):
        sample = self.synth[idx]
        image  = sample[0]   # tensor [3, H, W]
        label  = sample[1]

        if torch.rand(1).item() < self.swap_prob and self.bg_paths:
            bg_path = self.bg_paths[torch.randint(len(self.bg_paths), (1,)).item()]
            try:
                bg    = np.array(Image.open(bg_path).convert("L"))
                synth = (image[0].numpy() * 255).astype(np.uint8)
                comp  = paste_target_on_real_background(synth, bg, self.threshold)
                comp  = np.stack([comp, comp, comp], axis=0).astype(np.float32) / 255.
                image = torch.tensor(comp)
            except Exception:
                pass   # fall back to original if anything fails

        return (image,) + sample[1:]


# ═══════════════════════════════════════════════════════════════
# STRATEGY 3 — Fine-Tuning on Real Images
# ═══════════════════════════════════════════════════════════════

class RealDataFinetuner:
    """
    Fine-tune a pre-trained synthetic model on real IR images.

    This is the most effective adaptation strategy.
    Even 50–100 real images per class can close 60–70% of the gap.

    Three modes:
        head_only   — freeze backbone, train classifier only (safest for small sets)
        full        — fine-tune all layers with very low lr
        layer_wise  — progressively unfreeze from top to bottom

    Usage:
        finetuner = RealDataFinetuner(model, checkpoint_path)
        adapted_model = finetuner.finetune(
            real_train_loader, real_val_loader, mode="head_only"
        )
    """

    def __init__(
        self,
        model:           nn.Module,
        checkpoint_path: str,
        save_path:       str = "checkpoints/finetuned_best.pt",
        epochs:          int = cfg.finetune_epochs,
        patience:        int = 5,
    ):
        self.model     = model
        self.save_path = save_path
        self.epochs    = epochs
        self.patience  = patience
        self.device    = next(model.parameters()).device
        self.criterion = nn.CrossEntropyLoss()

        # Load pre-trained weights from synthetic training
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        print(f"[Finetune] Loaded checkpoint: {checkpoint_path}")

    def finetune(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        mode:         str = "head_only",   # "head_only" | "full" | "layer_wise"
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
            lr = cfg.finetune_lr_full    # much lower to avoid forgetting

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

            # Layer-wise: unfreeze one more block every 5 epochs
            if mode == "layer_wise" and epoch % 5 == 0:
                self._unfreeze_next_block(epoch)
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=lr * 0.1, weight_decay=cfg.weight_decay
                )

            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            for batch in tqdm(train_loader, desc=f"  finetune ep{epoch}", leave=False):
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct    += (self.model(x).argmax(1) == y).sum().item()
                total      += y.size(0)

            scheduler.step()

            val_acc = self._quick_eval(val_loader)
            print(f"  Epoch {epoch:2d} | val_acc={val_acc:.4f} | "
                  f"train_acc={correct/total:.4f}")

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

    def _freeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

    def _unfreeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = True

    def _unfreeze_head(self) -> None:
        for name, p in self.model.named_parameters():
            if any(k in name for k in ["classifier", "head", "fc"]):
                p.requires_grad = True

    def _unfreeze_next_block(self, epoch: int) -> None:
        """Progressively unfreeze from the top of the network downward."""
        all_names = [n for n, _ in self.model.named_parameters()]
        block_idx = min(epoch // 5, 4)
        n         = len(all_names)
        cutoff    = n - (block_idx + 1) * (n // 5)
        for i, (name, p) in enumerate(self.model.named_parameters()):
            if i >= cutoff:
                p.requires_grad = True
        n_trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
        print(f"  Layer-wise: unfroze block {block_idx}, "
              f"trainable params = {n_trainable}")

    def _quick_eval(self, loader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                correct += (self.model(x).argmax(1) == y).sum().item()
                total   += y.size(0)
        return correct / total


# ═══════════════════════════════════════════════════════════════
# STRATEGY 4 — DANN (Domain-Adversarial Neural Network)
# ═══════════════════════════════════════════════════════════════
#
# How it works:
#   The ConvNeXt backbone is split into two heads:
#     1. Class classifier  → trained normally     (minimise Lc)
#     2. Domain classifier → trained adversarially via GRL
#                            (minimise Ld for domain head;
#                             MAXIMISE Ld for backbone via GRL)
#
#   The Gradient Reversal Layer (GRL) sits between the backbone
#   and the domain classifier. Forward pass: identity.
#   Backward pass: multiplies gradient by −λ, forcing the backbone
#   to produce domain-invariant IR features.
#
#   Loss optimised end-to-end:
#       L = Lc  −  λ · Ld
#       Lc = CrossEntropy(class_logits,  target_labels)   [synth only]
#       Ld = CrossEntropy(domain_logits, domain_labels)   [synth + real]
#
#   λ is annealed from 0 → 1 using the schedule from Ganin et al.
#   (2016) to prevent instability while the domain head is cold.
#
# Reference:
#   Ganin et al., "Domain-Adversarial Training of Neural Networks"
#   JMLR 2016. https://arxiv.org/abs/1505.07818
#
# Usage (in run_experiment.py — Phase 3c):
#
#   from adaptation.strategies import DANNTrainer
#
#   dann = DANNTrainer(
#       backbone_checkpoint = ckpt_dr,       # start from Strategy 2/3 best
#       synth_loader        = train_loader,
#       real_loader         = real_train_loader,
#       val_loader          = val_loader,
#   )
#   adapted_model = dann.train()
#   # → saves checkpoints/dann_best.pt  (hand to 이성제 for evaluation)
#
#   gap_dann = measure_domain_gap(
#       adapted_model, synth_test_loader, real_test_loader,
#       save_path=f"{cfg.results_dir}confusion_dann.png"
#   )
# ─────────────────────────────────────────────────────────────

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from models.convnext import build_model


# ── Gradient Reversal Layer ───────────────────────────────────

class _GradientReversalFunction(Function):
    """
    Custom autograd function implementing the GRL.

    Forward:  identity  →  output = input
    Backward: negation  →  grad_input = −λ · grad_output

    ctx stores lambda between the two passes.
    The second return value in backward() covers the lambda_
    argument from forward(), which has no gradient (float).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Thin nn.Module wrapper around the GRL function.

    λ starts at 0.0 and is updated each epoch by DANNTrainer
    via set_lambda(). Placing this between the backbone and the
    domain classifier is the entire DANN trick.
    """

    def __init__(self, lambda_: float = 0.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = lambda_


# ── DANN Model  (backbone + two heads) ───────────────────────

class DANNModel(nn.Module):
    """
    ConvNeXt backbone with two classifier heads:

        backbone ──┬──> class_classifier    (ATR: num_classes outputs)
                   │
                   └──> GRL ──> domain_classifier  (synth=0 / real=1)

    The backbone is split at the penultimate feature layer.
    For ConvNeXt-tiny the feature dimension is 768.

    At inference time, only class_classifier is used —
    the domain head adds zero computational overhead in deployment.

    Args:
        num_classes:  ATR target classes (3 in this project)
        feature_dim:  ConvNeXt-tiny penultimate feature dim (768)
    """

    def __init__(
        self,
        num_classes:  int = cfg.num_classes,
        feature_dim:  int = 768,
    ):
        super().__init__()

        # ── Load ConvNeXt and strip its classifier head ───────
        # build_model() returns the full ConvNeXt with a replaced
        # Linear head. We keep features + avgpool, detach the head.
        base = build_model(num_classes=num_classes)

        self.backbone = nn.Sequential(
            base.features,   # ConvNeXt feature extractor blocks
            base.avgpool,    # AdaptiveAvgPool2d → (B, 768, 1, 1)
            nn.Flatten(),    # → (B, 768)
        )

        # ── Head 1: ATR target class classifier ───────────────
        # Mirrors the original ConvNeXt head (LayerNorm + Linear)
        self.class_classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes),
        )

        # ── Head 2: domain classifier (synth vs real) ─────────
        # Binary head — simpler than the class head is fine.
        # GRL is the first layer so its gradient reversal applies
        # to the full domain branch during backprop.
        self.grl = GradientReversalLayer(lambda_=0.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),   # 2 outputs: synthetic=0, real=1
        )

    def forward(
        self,
        x:             torch.Tensor,
        return_domain: bool = False,
    ):
        """
        Args:
            x:             input image tensor  (B, C, H, W)
            return_domain: True during training to get domain logits;
                           False at inference (deployment mode)

        Returns:
            class_logits:   (B, num_classes)           always
            domain_logits:  (B, 2)   only if return_domain=True
        """
        features     = self.backbone(x)
        class_logits = self.class_classifier(features)

        if return_domain:
            reversed_feat = self.grl(features)
            domain_logits = self.domain_classifier(reversed_feat)
            return class_logits, domain_logits

        return class_logits

    def set_lambda(self, lambda_: float) -> None:
        """Propagate updated λ from the trainer each epoch."""
        self.grl.set_lambda(lambda_)


# ── DANN Trainer ─────────────────────────────────────────────

class DANNTrainer:
    """
    Trains DANNModel end-to-end using synthetic + real IR data.

    Training loop per batch:
        1. Forward synthetic batch  → class_logits + domain_logits
           Lc       = CrossEntropy(class_logits, target_labels)
           Ld_synth = CrossEntropy(domain_logits, zeros)   ← synth=0
        2. Forward real batch (labels ignored) → domain_logits only
           Ld_real  = CrossEntropy(domain_logits, ones)    ← real=1
        3. Total loss = Lc + λ·(Ld_synth + Ld_real)
        4. Single backward + optimizer step

    λ annealing (Ganin et al. 2016):
        λ(p) = 2 / (1 + exp(−10·p)) − 1,  p = epoch / total_epochs
        Ramps smoothly from ~0 → ~1, preventing instability while
        the domain classifier is still cold.

    W&B logs: lambda, class_loss, domain_loss, class_acc, domain_acc,
              val_acc — giving 이성제 a complete gap-reduction chart.

    Args:
        backbone_checkpoint: path to a Strategy 1/2/3 .pt checkpoint.
                             Start from your best so far — DANN adapts
                             on top of existing transfer learning.
        synth_loader:        DataLoader of synthetic IR training images
        real_loader:         DataLoader of real IR images (FLIR ADAS).
                             Class labels are NOT used — domain signal only.
        val_loader:          Validation DataLoader (synth or mixed)
        epochs:              training epochs (cfg.finetune_epochs default)
        save_path:           output checkpoint path
        wandb_run_name:      W&B run name for 이성제's comparison chart
    """

    DOMAIN_SYNTH = 0
    DOMAIN_REAL  = 1

    def __init__(
        self,
        backbone_checkpoint: str,
        synth_loader:        DataLoader,
        real_loader:         DataLoader,
        val_loader:          DataLoader,
        epochs:              int = cfg.finetune_epochs,
        save_path:           str = "checkpoints/dann_best.pt",
        wandb_run_name:      str = "dann_strategy4",
    ):
        self.synth_loader = synth_loader
        self.real_loader  = real_loader
        self.val_loader   = val_loader
        self.epochs       = epochs
        self.save_path    = save_path
        self.device       = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── Build model and transfer backbone weights ─────────
        self.model = DANNModel(num_classes=cfg.num_classes).to(self.device)
        self._load_backbone_weights(backbone_checkpoint)

        # ── Loss functions ────────────────────────────────────
        self.class_criterion  = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

        # ── Single optimizer covers all parameters ────────────
        # AdamW at finetune_lr — adapting, not training from scratch.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr           = cfg.finetune_lr,
            weight_decay = cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )

        # ── W&B ───────────────────────────────────────────────
        if WANDB_AVAILABLE:
            import wandb
            wandb.init(
                project = cfg.project_name,
                name    = wandb_run_name,
                config  = {
                    "strategy":    "DANN",
                    "epochs":      self.epochs,
                    "lr":          cfg.finetune_lr,
                    "num_classes": cfg.num_classes,
                    "backbone":    cfg.model_name,
                }
            )

        print(f"[DANN] Device: {self.device} | "
              f"synth batches={len(synth_loader)} | "
              f"real batches={len(real_loader)}")

    # ── Public entry point ────────────────────────────────────

    def train(self) -> DANNModel:
        """
        Run the full DANN training loop.
        Returns the best model (highest val accuracy).
        Saves checkpoint to self.save_path for 이성제's evaluation.
        """
        best_val_acc   = 0.0
        patience_count = 0
        patience       = cfg.early_stop_patience

        for epoch in range(1, self.epochs + 1):

            # ── Update λ via DANN annealing schedule ──────────
            p       = epoch / self.epochs
            lambda_ = self._dann_lambda(p)
            self.model.set_lambda(lambda_)

            # ── Training step ─────────────────────────────────
            metrics = self._train_epoch(epoch, lambda_)

            # ── Validation ────────────────────────────────────
            val_acc = self._validate()

            # ── W&B logging ───────────────────────────────────
            if WANDB_AVAILABLE:
                import wandb
                wandb.log({
                    "epoch":             epoch,
                    "lambda":            lambda_,
                    "train/class_loss":  metrics["class_loss"],
                    "train/domain_loss": metrics["domain_loss"],
                    "train/total_loss":  metrics["total_loss"],
                    "train/class_acc":   metrics["class_acc"],
                    "train/domain_acc":  metrics["domain_acc"],
                    "val/class_acc":     val_acc,
                })

            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"λ={lambda_:.3f} | "
                f"Lc={metrics['class_loss']:.4f} | "
                f"Ld={metrics['domain_loss']:.4f} | "
                f"cls_acc={metrics['class_acc']:.4f} | "
                f"dom_acc={metrics['domain_acc']:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

            # ── Checkpoint ────────────────────────────────────
            if val_acc > best_val_acc:
                best_val_acc   = val_acc
                patience_count = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"  ✓ Best saved → {self.save_path}")
                if WANDB_AVAILABLE:
                    import wandb
                    wandb.run.summary["best_val_acc"] = best_val_acc
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"  Early stop at epoch {epoch} "
                          f"(no improvement for {patience} epochs)")
                    break

            self.scheduler.step()

        # Return model with best weights loaded
        self.model.load_state_dict(
            torch.load(self.save_path, map_location=self.device)
        )
        print(f"\n[DANN] Done. Best val_acc={best_val_acc:.4f}")
        print(f"  → Checkpoint for 이성제: {self.save_path}")
        if WANDB_AVAILABLE:
            import wandb
            wandb.finish()
        return self.model

    # ── Private helpers ───────────────────────────────────────

    def _train_epoch(self, epoch: int, lambda_: float) -> dict:
        """One training epoch iterating over both loaders."""
        self.model.train()

        total_class_loss  = 0.0
        total_domain_loss = 0.0
        class_correct     = 0
        domain_correct    = 0
        n_class_samples   = 0
        n_domain_samples  = 0

        # Cycle the real loader if it's shorter than synth
        real_iter = iter(self.real_loader)

        for synth_batch in tqdm(
            self.synth_loader, desc=f"  DANN ep{epoch}", leave=False
        ):
            # ── Synthetic batch ───────────────────────────────
            synth_x, synth_y = (
                synth_batch[0].to(self.device),
                synth_batch[1].to(self.device),
            )
            synth_domain_labels = torch.full(
                (synth_x.size(0),), self.DOMAIN_SYNTH,
                dtype=torch.long, device=self.device
            )

            # ── Real batch (class labels ignored) ─────────────
            try:
                real_batch = next(real_iter)
            except StopIteration:
                real_iter  = iter(self.real_loader)
                real_batch = next(real_iter)

            real_x = real_batch[0].to(self.device)
            real_domain_labels = torch.full(
                (real_x.size(0),), self.DOMAIN_REAL,
                dtype=torch.long, device=self.device
            )

            self.optimizer.zero_grad()

            # ── Forward ───────────────────────────────────────
            class_logits, synth_domain_logits = self.model(
                synth_x, return_domain=True
            )
            _, real_domain_logits = self.model(
                real_x, return_domain=True
            )

            # ── Losses ────────────────────────────────────────
            Lc       = self.class_criterion(class_logits, synth_y)
            Ld_synth = self.domain_criterion(
                synth_domain_logits, synth_domain_labels
            )
            Ld_real  = self.domain_criterion(
                real_domain_logits, real_domain_labels
            )
            Ld   = Ld_synth + Ld_real
            loss = Lc + lambda_ * Ld

            loss.backward()
            self.optimizer.step()

            # ── Metrics accumulation ──────────────────────────
            total_class_loss  += Lc.item()
            total_domain_loss += Ld.item()
            n_class_samples   += synth_y.size(0)
            n_domain_samples  += synth_x.size(0) + real_x.size(0)

            class_correct += (
                class_logits.argmax(1) == synth_y
            ).sum().item()

            domain_preds   = torch.cat([
                synth_domain_logits.argmax(1),
                real_domain_logits.argmax(1),
            ])
            domain_targets = torch.cat([
                synth_domain_labels, real_domain_labels
            ])
            domain_correct += (domain_preds == domain_targets).sum().item()

        n_batches = len(self.synth_loader)
        return {
            "class_loss":  total_class_loss  / n_batches,
            "domain_loss": total_domain_loss / n_batches,
            "total_loss":  (total_class_loss + total_domain_loss) / n_batches,
            "class_acc":   class_correct  / max(n_class_samples,  1),
            "domain_acc":  domain_correct / max(n_domain_samples, 1),
        }

    def _validate(self) -> float:
        """Accuracy on the validation set using the class head only."""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(x, return_domain=False)
                correct += (logits.argmax(1) == y).sum().item()
                total   += y.size(0)
        return correct / max(total, 1)

    def _load_backbone_weights(self, checkpoint_path: str) -> None:
        """
        Transfer weights from a plain ConvNeXt checkpoint into the
        DANN backbone + class head.
        The domain classifier intentionally starts from random init.
        """
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        tmp = build_model(num_classes=cfg.num_classes).to(self.device)
        tmp.load_state_dict(state_dict)

        # Transfer backbone (features + avgpool)
        self.model.backbone[0].load_state_dict(tmp.features.state_dict())
        self.model.backbone[1].load_state_dict(tmp.avgpool.state_dict())

        # Transfer class head  (ConvNeXt classifier = [LayerNorm, Flatten, Linear])
        # Our class_classifier  =                     [LayerNorm,          Linear]
        self.model.class_classifier[0].load_state_dict(
            tmp.classifier[0].state_dict()   # LayerNorm  at index 0
        )
        self.model.class_classifier[1].load_state_dict(
            tmp.classifier[2].state_dict()   # Linear     at index 2 (skip Flatten)
        )

        del tmp
        print(f"[DANN] Backbone + class head loaded from: {checkpoint_path}")
        print("       Domain classifier starts from random init (correct).")

    @staticmethod
    def _dann_lambda(p: float) -> float:
        """
        λ annealing schedule from Ganin et al. (2016).
        p ∈ [0, 1] = epoch / total_epochs.

        p=0.0 → λ≈0.00  (domain head has no pull at the start)
        p=0.5 → λ≈0.82
        p=1.0 → λ≈1.00  (full adversarial pressure at the end)
        """
        return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


# ═══════════════════════════════════════════════════════════════
# QUICK COLAB SMOKE TEST
# ═══════════════════════════════════════════════════════════════
# Paste into a Colab cell to verify Strategy 4 loads correctly:
#
#   from adaptation.strategies import DANNModel
#   import torch
#
#   model = DANNModel(num_classes=3)
#   dummy = torch.randn(4, 3, 224, 224)
#
#   # Inference mode
#   cls_logits = model(dummy)
#   assert cls_logits.shape == (4, 3), "Class head shape wrong"
#
#   # Training mode
#   cls_logits, dom_logits = model(dummy, return_domain=True)
#   assert dom_logits.shape == (4, 2), "Domain head shape wrong"
#
#   # GRL gradient check
#   loss = dom_logits.sum()
#   loss.backward()
#   print("✓ All shapes correct. GRL gradients flow.")
