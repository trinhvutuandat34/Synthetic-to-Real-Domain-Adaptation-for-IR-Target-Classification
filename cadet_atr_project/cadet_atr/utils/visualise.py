# utils/visualise.py
# ─────────────────────────────────────────────────────────────
# All project visualisations:
#   1. t-SNE — feature space (before + after adaptation)
#   2. Pixel intensity histogram — shows distribution shift
#   3. Gap reduction bar chart — your main results figure
#   4. Training curves — from W&B or local logs
# ─────────────────────────────────────────────────────────────

import os
import glob
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from utils.config import cfg


# ── 1. t-SNE feature space ────────────────────────────────────

def plot_tsne(
    model:        nn.Module,
    synth_loader: DataLoader,
    real_loader:  DataLoader,
    class_names:  Optional[List[str]] = None,
    title:        str = "Feature space — before adaptation",
    save_path:    Optional[str] = None,
    device:       str = "cuda",
) -> None:
    """
    Plot two t-SNE views side by side:
        Left:  coloured by CLASS  — shows whether classes are separable
        Right: coloured by DOMAIN — shows whether the gap exists

    BEFORE adaptation: right plot shows two blobs (synth/real separate)
    AFTER adaptation:  right plot shows mixed colours = gap closing

    This is one of your key report figures.
    """
    if class_names is None:
        class_names = cfg.class_names

    print("[t-SNE] Extracting features...")
    f_s, l_s = _extract_features(model, synth_loader, device)
    f_r, l_r = _extract_features(model, real_loader,  device)

    feats   = np.concatenate([f_s, f_r], axis=0)
    labels  = np.concatenate([l_s, l_r], axis=0)
    domains = np.array([0] * len(f_s) + [1] * len(f_r))

    print(f"[t-SNE] Running dimensionality reduction on {len(feats)} points...")
    emb = TSNE(
        n_components=2, perplexity=30,
        n_iter=1000, random_state=cfg.seed
    ).fit_transform(feats)

    # Colour palettes
    class_colours  = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    domain_colours = ["#4fc3f7", "#e05c5c"]   # blue=synth, red=real

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13, y=1.01)

    # Left: by class
    for i, (cls, col) in enumerate(zip(class_names, class_colours)):
        mask = labels == i
        # Distinguish domain by marker shape
        for dom, marker, alpha in [(0, "o", 0.5), (1, "^", 0.7)]:
            dm = mask & (domains == dom)
            ax1.scatter(emb[dm, 0], emb[dm, 1],
                        c=[col], marker=marker, alpha=alpha,
                        s=20, label=f"{cls} ({'synth' if dom==0 else 'real'})")
    ax1.set_title("Coloured by class\n○=synthetic  △=real")
    ax1.legend(fontsize=8, markerscale=1.5, loc="best")
    ax1.axis("off")

    # Right: by domain — this directly shows the gap
    for dom, col, lbl in [(0, "#4fc3f7", "Synthetic"), (1, "#e05c5c", "Real")]:
        mask = domains == dom
        ax2.scatter(emb[mask, 0], emb[mask, 1],
                    c=col, alpha=0.5, s=15, label=lbl)
    ax2.set_title("Coloured by DOMAIN ← look for mixing here\n"
                  "Well-adapted: colours interspersed")
    ax2.legend(fontsize=9)
    ax2.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  t-SNE saved → {save_path}")
    plt.show()


def _extract_features(
    model:  nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer features using a forward hook."""
    model.eval()
    features, labels = [], []
    hook_storage = []

    def hook(module, input, output):
        out = output.detach().cpu()
        if out.dim() > 2:
            out = out.mean([-2, -1])   # global average pool
        hook_storage.append(out)

    # Register hook on the layer before the classifier
    handle = None
    for name, module in model.named_modules():
        if any(k in name for k in ["classifier", "avgpool", "head"]):
            last_module = module
    handle = last_module.register_forward_hook(hook)

    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            model(x)
            features.append(hook_storage.pop())
            labels.extend(y.numpy())

    handle.remove()
    return (
        torch.cat(features).numpy(),
        np.array(labels)
    )


# ── 2. Pixel intensity histogram ──────────────────────────────

def plot_intensity_histograms(
    synth_dir:  str,
    real_dir:   str,
    n_samples:  int = 200,
    save_path:  Optional[str] = None,
) -> None:
    """
    Compare pixel intensity distributions of synthetic vs real images.
    If the distributions are different shapes → histogram matching will help.
    If they overlap well → the gap comes from texture/noise, not brightness.
    """
    def collect_pixels(img_dir, n):
        paths = glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True)
        paths += glob.glob(os.path.join(img_dir, "**", "*.jpg"), recursive=True)
        paths = paths[:n]
        arrays = [np.array(Image.open(p).convert("L")).flatten() for p in paths]
        return np.concatenate(arrays)

    print("[Histogram] Collecting pixel values...")
    synth_px = collect_pixels(synth_dir, n_samples)
    real_px  = collect_pixels(real_dir,  n_samples)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, px, col, lbl in [
        (axes[0], synth_px, "#7F77DD", "Synthetic"),
        (axes[0], real_px,  "#1D9E75", "Real"),
    ]:
        ax.hist(px, bins=50, alpha=0.6, color=col, label=lbl, density=True)
    axes[0].set_title("Pixel intensity distribution overlay")
    axes[0].set_xlabel("Pixel value (0–255)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # CDF comparison
    for px, col, lbl in [(synth_px, "#7F77DD", "Synthetic"),
                          (real_px,  "#1D9E75", "Real")]:
        sorted_px = np.sort(px)
        cdf = np.linspace(0, 1, len(sorted_px))
        axes[1].plot(sorted_px[::1000], cdf[::1000], color=col, label=lbl, lw=2)
    axes[1].set_title("Cumulative distribution\n(similar shape = low histogram gap)")
    axes[1].set_xlabel("Pixel value")
    axes[1].set_ylabel("CDF")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Histogram plot saved → {save_path}")
    plt.show()


# ── 3. Gap reduction bar chart ────────────────────────────────

def plot_gap_reduction(
    results: List[dict],
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart showing real-image accuracy at each stage.
    This is your primary results figure for the report.

    Args:
        results: list of dicts, each with:
            {"name": str, "synth_acc": float, "real_acc": float}

    Example:
        plot_gap_reduction([
            {"name": "Paper baseline",      "synth_acc": 0.90, "real_acc": None},
            {"name": "Synthetic baseline",  "synth_acc": 0.88, "real_acc": 0.52},
            {"name": "+ Hist. matching",    "synth_acc": 0.87, "real_acc": 0.61},
            {"name": "+ Domain random.",    "synth_acc": 0.86, "real_acc": 0.70},
            {"name": "+ Fine-tuning",       "synth_acc": 0.85, "real_acc": 0.81},
        ])
    """
    names      = [r["name"] for r in results]
    synth_accs = [r["synth_acc"] if r["synth_acc"] else 0 for r in results]
    real_accs  = [r["real_acc"]  if r["real_acc"]  else 0 for r in results]

    x  = np.arange(len(names))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))

    bars_s = ax.bar(x - w/2, synth_accs, w, label="Synthetic test",
                    color="#7F77DD", alpha=0.8, edgecolor="white", lw=0.5)
    bars_r = ax.bar(x + w/2, real_accs,  w, label="Real test",
                    color="#1D9E75", alpha=0.8, edgecolor="white", lw=0.5)

    # Gap annotation arrows
    for i, (s, r) in enumerate(zip(synth_accs, real_accs)):
        if s > 0 and r > 0:
            gap = s - r
            ax.annotate(
                f"gap\n{gap:.1%}",
                xy=(x[i], min(s, r) - 0.02),
                ha="center", va="top",
                fontsize=8, color="#888",
                arrowprops=dict(arrowstyle="->", color="#888"),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Domain gap reduction — real-image accuracy at each stage")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Gap chart saved → {save_path}")
    plt.show()
