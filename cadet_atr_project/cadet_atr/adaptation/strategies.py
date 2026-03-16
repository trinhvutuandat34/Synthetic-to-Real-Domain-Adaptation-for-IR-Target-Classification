# adaptation/finetune.py
# adaptation/histogram.py
# adaptation/domain_random.py
#
# ─────────────────────────────────────────────────────────────
# All three domain adaptation strategies in one file:
#
#   Strategy 1 — Histogram matching
#   Strategy 2 — Domain randomisation (extended augmentation)
#   Strategy 3 — Fine-tuning on real images
#
# Run them in order. Record the gap before and after each.
# ─────────────────────────────────────────────────────────────

import os
import glob
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
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
        img  = np.array(Image.open(path).convert("L"))
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
        self.synth       = synth_dataset
        self.bg_paths    = real_bg_paths
        self.swap_prob   = swap_prob
        self.threshold   = threshold

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
                synth = (image[0].numpy() * 255).astype(np.uint8)  # first channel
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
        model:              nn.Module,
        checkpoint_path:    str,
        save_path:          str          = "checkpoints/finetuned_best.pt",
        epochs:             int          = cfg.finetune_epochs,
        patience:           int          = 5,
    ):
        self.model            = model
        self.save_path        = save_path
        self.epochs           = epochs
        self.patience         = patience
        self.device           = next(model.parameters()).device
        self.criterion        = nn.CrossEntropyLoss()

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
            # Start frozen, progressively unfreeze every N epochs
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
                # Re-create optimizer to include new parameters
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=lr * 0.1, weight_decay=cfg.weight_decay
                )

            # Train epoch
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

            # Validate
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

        # Load best weights back
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
        # Estimate which block to unfreeze based on epoch
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
