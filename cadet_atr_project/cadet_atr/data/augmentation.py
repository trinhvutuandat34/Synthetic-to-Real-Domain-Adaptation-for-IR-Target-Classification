# data/augmentation.py
# ─────────────────────────────────────────────────────────────
# IR-specific augmentation pipeline using Kornia.
# Runs on GPU (much faster than CPU-based torchvision transforms
# for large batches).
#
# Three pipeline levels:
#   1. baseline  — replicates the paper exactly
#   2. extended  — adds IR noise for domain randomisation
#   3. aggressive— maximum randomisation for adaptation training
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import kornia.augmentation as K

from utils.config import cfg


def build_augmentation_pipeline(level: str = "baseline") -> nn.Sequential:
    """
    Build a Kornia GPU augmentation pipeline.

    Args:
        level: "baseline" | "extended" | "aggressive"

            baseline   — reproduces the paper's augmentation (Table 1)
            extended   — adds IR-specific noise types for domain randomisation
            aggressive — maximises diversity; best for closing the domain gap

    Returns:
        nn.Sequential of Kornia transforms.
        Call pipeline(batch_tensor) inside your training loop AFTER
        moving the batch to GPU.

    Usage:
        aug = build_augmentation_pipeline("extended")
        for x, y, _ in train_loader:
            x = x.cuda()
            x = aug(x)     # augment on GPU
            out = model(x)
    """
    p = cfg.aug_prob   # default 0.5 per transform — matches the paper

    # ── Paper baseline (Table 1) ──────────────────────────────
    baseline_transforms = [
        K.RandomHorizontalFlip(p=p),
        K.RandomVerticalFlip(p=p),
        K.RandomRotation(degrees=90.0, p=p),
        K.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), p=p
        ),
        K.RandomPerspective(distortion_scale=0.2, p=p),
        K.RandomResizedCrop(
            size=(cfg.input_size, cfg.input_size),
            scale=(0.8, 1.0), p=p
        ),
        K.RandomBrightness(brightness=cfg.aug_brightness, p=p),
        K.RandomContrast(contrast=cfg.aug_contrast, p=p),
        K.RandomEqualize(p=p * 0.5),                     # less frequent
        K.RandomGaussianNoise(mean=0.0, std=cfg.aug_noise_std, p=p),
    ]

    if level == "baseline":
        return nn.Sequential(*baseline_transforms)

    # ── Extended: adds IR-specific noise ─────────────────────
    extended_transforms = baseline_transforms + [
        # Simulate sensor non-uniformity: random smooth gain map
        K.RandomGaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0), p=p * 0.6),
        # Simulate atmospheric scattering / haze
        K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(1.0, 3.0), p=p * 0.3),
        # Simulate dead pixels (occlusion / damage)
        K.RandomErasing(
            scale=(0.01, 0.05), ratio=(0.5, 2.0), p=p * 0.4
        ),
        # Sharpness variation (sensor focus changes)
        K.RandomSharpness(sharpness=0.5, p=p * 0.4),
    ]

    if level == "extended":
        return nn.Sequential(*extended_transforms)

    # ── Aggressive: maximum domain randomisation ──────────────
    aggressive_transforms = extended_transforms + [
        K.RandomBrightness(brightness=(0.4, 1.6), p=p),    # wider range
        K.RandomContrast(contrast=(0.4, 1.6), p=p),
        K.RandomGaussianNoise(mean=0.0, std=0.12, p=p),    # stronger noise
        K.RandomErasing(
            scale=(0.05, 0.2), ratio=(0.3, 3.0), p=p * 0.5
        ),
    ]

    return nn.Sequential(*aggressive_transforms)


class IRNoiseLayer(nn.Module):
    """
    Simulates real IR sensor noise patterns not captured by
    standard augmentation libraries.

    Applies:
        - Fixed-pattern noise (FPN): column-wise additive bias
        - Non-uniformity (NU):       smooth multiplicative gain map

    These are the two most common sources of domain gap between
    synthetic and real IR images.

    Usage:
        noise = IRNoiseLayer(fpn_strength=0.02, nu_strength=0.01)
        x = noise(x)   # apply after standard augmentation
    """

    def __init__(
        self,
        fpn_strength: float = 0.02,
        nu_strength:  float = 0.01,
        apply_prob:   float = 0.5,
    ):
        super().__init__()
        self.fpn_strength = fpn_strength
        self.nu_strength  = nu_strength
        self.apply_prob   = apply_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.apply_prob:
            return x

        B, C, H, W = x.shape

        # Fixed-pattern noise: column-wise bias (same pattern per batch)
        fpn = torch.randn(1, 1, 1, W, device=x.device) * self.fpn_strength
        fpn = fpn.expand(B, C, H, W)

        # Non-uniformity: smooth spatially-varying gain
        # Generate at low resolution then upsample for smoothness
        nu_low = torch.randn(B, 1, H // 16 + 1, W // 16 + 1, device=x.device)
        nu = nn.functional.interpolate(
            nu_low, size=(H, W), mode="bilinear", align_corners=False
        )
        nu = (1.0 + nu * self.nu_strength).expand(B, C, H, W)

        return (x + fpn) * nu
