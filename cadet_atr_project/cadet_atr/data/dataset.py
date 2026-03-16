# data/dataset.py
# ─────────────────────────────────────────────────────────────
# PyTorch Dataset classes for:
#   1. Synthetic IR images (Stable Diffusion generated)
#   2. Real IR images (FLIR / KAIST / OTCBVS)
#   3. A combined dataset for domain adaptation experiments
# ─────────────────────────────────────────────────────────────

import os
import random
from pathlib import Path
from typing import Optional, Tuple, Callable

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils.config import cfg


# ── Base transforms ───────────────────────────────────────────

def get_base_transform(input_size: int = 224) -> transforms.Compose:
    """
    Minimal transform applied to ALL images (train + val + test).
    ImageNet normalisation is used because we start from ImageNet
    pretrained weights — same as the paper.
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.Grayscale(num_output_channels=3),   # IR is grayscale; expand to 3ch
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet stats
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_train_transform(input_size: int = 224) -> transforms.Compose:
    """
    Training transform — includes basic geometric augmentation.
    Kornia augmentation is applied separately in the training loop
    so it runs on GPU (much faster for large batches).
    """
    return transforms.Compose([
        transforms.Resize((int(input_size * 1.1), int(input_size * 1.1))),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ── Synthetic IR Dataset ──────────────────────────────────────

class SyntheticIRDataset(ImageFolder):
    """
    Wraps torchvision's ImageFolder for the synthetic dataset.
    Expects directory structure:
        data/synthetic/
            aircraft/  ← 200+ PNG images
            vessel/
            vehicle/

    The `domain` label (0 = synthetic) is added to each sample
    for use in domain adaptation experiments.
    """

    def __init__(
        self,
        root: str = cfg.synth_dir,
        transform: Optional[Callable] = None,
    ):
        if transform is None:
            transform = get_base_transform(cfg.input_size)
        super().__init__(root, transform=transform)
        self.domain = 0   # 0 = synthetic

    def __getitem__(self, index: int) -> Tuple:
        image, class_label = super().__getitem__(index)
        return image, class_label, self.domain


# ── Real IR Dataset ───────────────────────────────────────────

class RealIRDataset(ImageFolder):
    """
    Real IR images from FLIR ADAS, KAIST, or similar datasets.
    Expects the same directory structure as SyntheticIRDataset:
        data/real/
            aircraft/
            vessel/
            vehicle/

    The `domain` label (1 = real) distinguishes these from
    synthetic images in domain adaptation experiments.
    """

    def __init__(
        self,
        root: str = cfg.real_dir,
        transform: Optional[Callable] = None,
    ):
        if transform is None:
            transform = get_base_transform(cfg.input_size)
        super().__init__(root, transform=transform)
        self.domain = 1   # 1 = real

    def __getitem__(self, index: int) -> Tuple:
        image, class_label = super().__getitem__(index)
        return image, class_label, self.domain


# ── Combined Dataset (for DANN) ───────────────────────────────

class CombinedDataset(Dataset):
    """
    Combines synthetic + real datasets into one.
    Used by DANN (adversarial domain adaptation) which needs
    samples from both domains in every training batch.

    Returns: (image, class_label, domain_label)
        domain_label: 0 = synthetic, 1 = real
    """

    def __init__(
        self,
        synth_dataset: SyntheticIRDataset,
        real_dataset:  RealIRDataset,
    ):
        self.synth = synth_dataset
        self.real  = real_dataset

    def __len__(self) -> int:
        # Length = larger dataset; smaller dataset wraps around
        return max(len(self.synth), len(self.real))

    def __getitem__(self, index: int) -> Tuple:
        s_img, s_cls, s_dom = self.synth[index % len(self.synth)]
        r_img, r_cls, r_dom = self.real[index  % len(self.real)]
        return s_img, s_cls, s_dom, r_img, r_cls, r_dom


# ── DataLoader factory ────────────────────────────────────────

def make_loaders(
    dataset_type: str = "synthetic",
    batch_size:   int = cfg.batch_size,
    num_workers:  int = 2,
    seed:         int = cfg.seed,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from a single dataset root.

    Args:
        dataset_type: "synthetic" or "real"
        batch_size:   samples per batch
        num_workers:  parallel loading workers
        seed:         random seed for reproducible splits

    Returns:
        train_loader, val_loader, test_loader

    IMPORTANT: call this once at the start of the project and
    save the split indices. Never change the splits between
    experiments — it invalidates your comparison table.
    """
    torch.manual_seed(seed)

    if dataset_type == "synthetic":
        ds = SyntheticIRDataset(transform=get_train_transform(cfg.input_size))
        ds_val = SyntheticIRDataset(transform=get_base_transform(cfg.input_size))
    else:
        ds = RealIRDataset(transform=get_train_transform(cfg.input_size))
        ds_val = RealIRDataset(transform=get_base_transform(cfg.input_size))

    n = len(ds)
    n_train = int(cfg.train_frac * n)
    n_val   = int(cfg.val_frac   * n)
    n_test  = n - n_train - n_val

    # Fixed generator ensures same split every run
    gen = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = random_split(
        range(n), [n_train, n_val, n_test], generator=gen
    )

    # Apply different transforms to val/test (no augmentation)
    from torch.utils.data import Subset
    train_ds = Subset(ds,     list(train_idx))
    val_ds   = Subset(ds_val, list(val_idx))
    test_ds  = Subset(ds_val, list(test_idx))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"[Dataset] {dataset_type}: "
          f"train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    return train_loader, val_loader, test_loader


def make_real_loader(
    root:        str = cfg.real_dir,
    batch_size:  int = cfg.batch_size,
    num_workers: int = 2,
) -> DataLoader:
    """
    Single DataLoader for the entire real test set.
    Used in gap measurement — no train/val split needed.
    """
    ds = RealIRDataset(root=root, transform=get_base_transform(cfg.input_size))
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
