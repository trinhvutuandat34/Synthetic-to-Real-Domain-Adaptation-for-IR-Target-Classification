# ─────────────────────────────────────────────────────────────
# ADDITIONS / FIXES for utils/visualise.py
#
# 1. plot_gradcam()         — Grad-CAM saliency maps for any model
#                             in the zoo (ConvNeXt, ResNet, ViT, Swin)
# 2. _extract_features()    — Fixed hook that works on both plain
#                             ConvNeXt AND DANNModel
#
# Drop these into visualise.py, replacing the original
# _extract_features() and adding plot_gradcam() at the bottom.
# ─────────────────────────────────────────────────────────────

import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ══════════════════════════════════════════════════════════════
# 1. GRAD-CAM
# ══════════════════════════════════════════════════════════════
#
# Grad-CAM (Selvaraju et al., 2017) answers: "which spatial
# regions of the IR image drove the ATR classification decision?"
#
# Algorithm:
#   1. Register a hook on the last convolutional feature map.
#   2. Forward pass → class logits.
#   3. Backprop the logit of the predicted (or specified) class
#      back to the feature map.
#   4. Global-average-pool the gradients over spatial dims → weights.
#   5. Weight the feature map channels → ReLU → upsample → overlay.
#
# For CMS operators, this shows WHICH part of the IR target
# (engine exhaust plume, hull silhouette, rotor disk) the model
# is using — critical for building trust in operational use.
#
# Usage:
#   plot_gradcam(
#       model,
#       image_paths=["data/real/aircraft/00001.png", ...],
#       class_names=cfg.class_names,
#       save_dir="results/",
#   )
# ──────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM for any model in the ConvNeXt / ResNet / ViT / Swin zoo.

    Automatically selects the correct target layer for each architecture:
      - ConvNeXt / Swin: last stage of model.features  (stage index -1)
      - ResNet / ResNeXt: layer4
      - VGG:              features[-1]  (last MaxPool)
      - ViT:              encoder.layers[-1]  (last transformer block)

    For DANNModel, targets model.backbone[0][-1]  (last ConvNeXt stage).
    """

    def __init__(self, model: nn.Module, model_name: str = "convnext_tiny"):
        self.model      = model
        self.model_name = model_name
        self._gradients = None
        self._activations = None
        self._hook_handles: List = []

        self._register_hooks(model, model_name)

    def _register_hooks(self, model: nn.Module, model_name: str) -> None:
        """
        Register forward + backward hooks on the target layer.
        The target layer is the last spatial feature map before the
        global average pool — this is where spatial resolution is
        still present but semantic content is richest.
        """
        target_layer = self._get_target_layer(model, model_name)

        def forward_hook(module, input, output):
            # output shape: (B, C, H, W)  for CNN, (B, T, C) for ViT
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] shape matches output shape
            self._gradients = grad_output[0].detach()

        self._hook_handles.append(
            target_layer.register_forward_hook(forward_hook)
        )
        self._hook_handles.append(
            target_layer.register_full_backward_hook(backward_hook)
        )

    def _get_target_layer(self, model: nn.Module, model_name: str) -> nn.Module:
        """Return the last spatial feature layer for each architecture."""
        # DANNModel: backbone[0] is the ConvNeXt features block
        if hasattr(model, "backbone") and hasattr(model.backbone, "__getitem__"):
            features = model.backbone[0]
            return features[-1]  # last ConvNeXt stage

        if "convnext" in model_name or "swin" in model_name:
            return model.features[-1]

        elif "resnet" in model_name or "resnext" in model_name:
            return model.layer4

        elif "vgg" in model_name:
            return model.features[-1]

        elif "vit" in model_name:
            # ViT: last transformer encoder block
            return model.encoder.layers[-1]

        else:
            raise ValueError(
                f"Unknown model_name '{model_name}' for Grad-CAM. "
                "Add your architecture's target layer here."
            )

    def generate(
        self,
        x:          torch.Tensor,   # shape (1, C, H, W)  — single image
        class_idx:  Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Run Grad-CAM for one image.

        Args:
            x:          preprocessed image tensor on the model's device
            class_idx:  which class logit to backprop.
                        None → uses the argmax (predicted class).

        Returns:
            cam:       H×W float32 array in [0,1], same spatial size as x
            class_idx: the class that was visualised
        """
        self.model.eval()
        self._gradients  = None
        self._activations = None

        # Forward pass
        # DANNModel in eval mode returns class_logits only
        output = self.model(x)
        if isinstance(output, tuple):
            output = output[0]  # (class_logits, domain_logits)

        if class_idx is None:
            class_idx = output.argmax(1).item()

        # Backprop the target class logit
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Global-average-pool gradients over spatial dimensions
        grads = self._gradients   # (1, C, H', W') or (1, T, C) for ViT
        acts  = self._activations # same shape

        # ViT: tokens instead of spatial feature map
        if grads is not None and grads.dim() == 3:
            # (1, T, C) → treat token dim as spatial, pool over T
            weights = grads.mean(dim=1, keepdim=True)  # (1, 1, C)
            cam = (weights * acts).sum(dim=2).squeeze()  # (T,)
            # For ViT, T = (H/16)^2 + 1 (incl. cls token) — reshape if square-ish
            T = cam.shape[0] - 1  # drop cls token
            s = int(T**0.5)
            if s * s == T:
                cam = cam[1:].reshape(s, s)
            else:
                cam = cam[1:]
        else:
            # CNN: (1, C, H', W')
            weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            cam = (weights * acts).sum(dim=1).squeeze()     # (H', W')

        # ReLU + normalise
        cam = F.relu(torch.tensor(cam)).numpy() if not isinstance(cam, np.ndarray) else cam
        if isinstance(cam, torch.Tensor):
            cam = cam.numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam /= cam.max()

        # Upsample to original image size
        H, W = x.shape[2], x.shape[3]
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
        cam_pil = cam_pil.resize((W, H), Image.BILINEAR)
        cam = np.array(cam_pil).astype(np.float32) / 255.0

        return cam, class_idx

    def remove_hooks(self) -> None:
        """Clean up hooks when done."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def plot_gradcam(
    model:        nn.Module,
    image_paths:  List[str],
    class_names:  List[str],
    model_name:   str = "convnext_tiny",
    transform             = None,
    n_images:     int  = 8,
    save_dir:     Optional[str] = None,
    device:       str  = "cuda",
) -> None:
    """
    Generate and save Grad-CAM overlays for a batch of IR images.

    For the ATR project, this answers: "which part of the thermal
    image (engine plume, hull shape, rotor signature) is the
    ConvNeXt model using to classify each target?"

    Args:
        model:       trained model (ConvNeXt, DANN, or any MODEL_ZOO key)
        image_paths: list of image file paths (PNG/JPG)
        class_names: list of class name strings
        model_name:  key from MODEL_ZOO — controls which layer is targeted
        transform:   preprocessing transform. Defaults to base_transform.
        n_images:    how many images to visualise (max)
        save_dir:    directory to write gradcam_*.png files
        device:      "cuda" or "cpu"

    Output files: save_dir/gradcam_CLASSNAME_IMGINDEX.png
    Each file shows: original IR image | heatmap | overlay
    """
    if transform is None:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    model  = model.to(device)
    gradcam = GradCAM(model, model_name)

    paths_to_show = image_paths[:n_images]
    n_cols = 3   # original | heatmap | overlay
    n_rows = len(paths_to_show)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # keep 2-D indexing

    fig.suptitle(
        "Grad-CAM — IR target saliency map\n"
        "(bright = regions that drove the ATR decision)",
        fontsize=12
    )

    # Mean/std for denormalisation
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for row_idx, img_path in enumerate(paths_to_show):
        # Load and preprocess
        pil_img = Image.open(img_path).convert("L")
        x       = transform(pil_img).unsqueeze(0).to(device)

        # Grad-CAM
        cam, pred_class = gradcam.generate(x)

        # Denormalise tensor → displayable numpy image
        x_show  = (x.squeeze().cpu() * std + mean)
        x_show  = x_show.permute(1, 2, 0).numpy().clip(0, 1)
        gray_img = x_show.mean(axis=2)   # collapse to grayscale for display

        # Colour heatmap
        heatmap = cm.jet(cam)[..., :3]   # RGB from colormap

        # Overlay: weighted sum of grayscale + heatmap
        gray_3ch = np.stack([gray_img] * 3, axis=-1)
        overlay  = 0.55 * gray_3ch + 0.45 * heatmap
        overlay  = overlay.clip(0, 1)

        pred_name = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
        img_stem  = os.path.splitext(os.path.basename(img_path))[0]

        # Plot row
        axes[row_idx, 0].imshow(gray_img, cmap="gray")
        axes[row_idx, 0].set_title(f"{img_stem}\npred: {pred_name}", fontsize=9)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(heatmap)
        axes[row_idx, 1].set_title("Grad-CAM heatmap", fontsize=9)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title("Overlay", fontsize=9)
        axes[row_idx, 2].axis("off")

        # Save individual overlay image for the report/CMS briefing
        if save_dir:
            out_path = os.path.join(
                save_dir, f"gradcam_{pred_name}_{img_stem}.png"
            )
            Image.fromarray((overlay * 255).astype(np.uint8)).save(out_path)

    plt.tight_layout()

    if save_dir:
        summary_path = os.path.join(save_dir, "gradcam_summary.png")
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        print(f"[Grad-CAM] Summary saved → {summary_path}")
        print(f"[Grad-CAM] Individual overlays → {save_dir}gradcam_*.png")

    plt.show()
    gradcam.remove_hooks()


def plot_gradcam_per_class(
    model:        nn.Module,
    real_dir:     str,
    class_names:  List[str],
    model_name:   str = "convnext_tiny",
    n_per_class:  int = 2,
    save_dir:     Optional[str] = None,
    device:       str = "cuda",
) -> None:
    """
    Convenience wrapper: run Grad-CAM on N images per class from real_dir.

    This is the typical usage for 이성제's report figures — one grid
    showing representative saliency maps for all 6 ATR classes.

    Usage:
        from utils.visualise import plot_gradcam_per_class
        plot_gradcam_per_class(
            model       = adapted_model,
            real_dir    = cfg.real_dir,
            class_names = cfg.class_names,
            n_per_class = 2,
            save_dir    = "results/",
        )
    """
    all_paths = []
    for cls_name in class_names:
        cls_dir = os.path.join(real_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"[Grad-CAM] Warning: class directory not found: {cls_dir}")
            continue
        paths = sorted(
            glob.glob(os.path.join(cls_dir, "*.png")) +
            glob.glob(os.path.join(cls_dir, "*.jpg"))
        )[:n_per_class]
        all_paths.extend(paths)

    if not all_paths:
        print("[Grad-CAM] No images found in real_dir. Check cfg.real_dir.")
        return

    print(f"[Grad-CAM] Running on {len(all_paths)} images "
          f"({n_per_class} × {len(class_names)} classes)")

    plot_gradcam(
        model       = model,
        image_paths = all_paths,
        class_names = class_names,
        model_name  = model_name,
        n_images    = len(all_paths),
        save_dir    = save_dir,
        device      = device,
    )


# ══════════════════════════════════════════════════════════════
# 2. FIXED _extract_features()
# ══════════════════════════════════════════════════════════════
#
# Original bug: iterates all named modules and stores the last one
# whose name contains "classifier", "avgpool", or "head".
# On DANNModel, the last such module is domain_classifier[-1],
# NOT the backbone's avgpool → wrong features extracted for t-SNE.
#
# Fix: explicitly handle DANNModel by checking for the .backbone
# attribute, and fall back to architecture-specific hooks for
# plain ConvNeXt, ResNet, ViT, and VGG.

def _extract_features(
    model:  nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract penultimate-layer features using a forward hook.

    Works on:
      - ConvNeXt, Swin:       model.features[-1] → avgpool
      - ResNet, ResNeXt:      model.avgpool
      - VGG:                  model.classifier[-3] (second-to-last ReLU)
      - ViT:                  model.encoder (last transformer block output)
      - DANNModel:            model.backbone (full backbone incl. flatten)
                              → extracts 768-d feature vector directly

    Returns:
        features: (N, D) float32 numpy array
        labels:   (N,)   int64  numpy array
    """
    model.eval()
    hook_storage = []
    handle = None

    def hook(module, input, output):
        out = output.detach().cpu()
        # Flatten spatial dims if still 4-D (B, C, H, W)
        if out.dim() == 4:
            out = out.mean(dim=[2, 3])   # global average pool → (B, C)
        # ViT encoder returns (B, T, C) — take the CLS token
        elif out.dim() == 3:
            out = out[:, 0, :]           # CLS token → (B, C)
        hook_storage.append(out)

    # ── Architecture-aware hook registration ─────────────────
    if hasattr(model, "backbone"):
        # DANNModel: backbone = Sequential(features, avgpool, Flatten)
        # Hooking the full backbone gives us the 768-d flat feature vector
        handle = model.backbone.register_forward_hook(hook)

    elif hasattr(model, "features") and hasattr(model, "avgpool"):
        # ConvNeXt / Swin — hook avgpool (output is (B, C, 1, 1))
        handle = model.avgpool.register_forward_hook(hook)

    elif hasattr(model, "layer4"):
        # ResNet / ResNeXt — hook avgpool after layer4
        handle = model.avgpool.register_forward_hook(hook)

    elif hasattr(model, "encoder"):
        # ViT — hook the full encoder; take CLS token in hook above
        handle = model.encoder.register_forward_hook(hook)

    elif hasattr(model, "classifier"):
        # VGG — hook second-to-last layer (ReLU before final Linear)
        # classifier = [..., Linear, ReLU, Dropout, Linear]
        #                              ↑ index -3
        handle = model.classifier[-3].register_forward_hook(hook)

    else:
        raise ValueError(
            "Cannot find a suitable feature layer for t-SNE hook. "
            "Add your model's architecture here."
        )

    features, labels = [], []

    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            model(x)
            if hook_storage:
                features.append(hook_storage.pop())
            labels.extend(y.numpy())

    handle.remove()

    return (
        torch.cat(features, dim=0).float().numpy(),
        np.array(labels),
    )

# ══════════════════════════════════════════════════════════════
# 3. PLOT_TSNE
# ══════════════════════════════════════════════════════════════

def plot_tsne(
    model:        nn.Module,
    synth_loader: DataLoader,
    real_loader:  DataLoader,
    title:        str = "",
    save_path:    Optional[str] = None,
    device:       str = "cuda",
) -> None:
    """
    2-D t-SNE of penultimate-layer features — shows whether the ATR model
    has learned domain-invariant representations before/after adaptation.
    """
    # Import here so the module-level import list stays unchanged,
    # which matters because sklearn is a heavy import.
    from sklearn.manifold import TSNE

    print("[t-SNE] Extracting features from synthetic loader...")
    synth_feats, synth_labels = _extract_features(model, synth_loader, device)

    print("[t-SNE] Extracting features from real loader...")
    real_feats,  real_labels  = _extract_features(model, real_loader,  device)

    # Build a single matrix for t-SNE.
    # t-SNE's embedding is only meaningful within a single run,
    # so both domains MUST be embedded together.
    all_feats   = np.concatenate([synth_feats, real_feats], axis=0)   # (N_s+N_r, D)
    all_labels  = np.concatenate([synth_labels, real_labels], axis=0) # class labels
    domain_ids  = np.array(
        [0] * len(synth_labels) + [1] * len(real_labels)
    )  # 0 = synthetic, 1 = real

    print(f"[t-SNE] Running TSNE on {len(all_feats)} feature vectors "
          f"(dim={all_feats.shape[1]})...")
    tsne    = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embedded = tsne.fit_transform(all_feats)  # (N, 2)

    # Split back into synth / real using the domain_ids we built above
    synth_emb = embedded[domain_ids == 0]
    real_emb  = embedded[domain_ids == 1]
    synth_cls = all_labels[domain_ids == 0]
    real_cls  = all_labels[domain_ids == 1]

    class_names = cfg.class_names
    n_classes   = len(class_names)

    # Use a colormap with enough distinct colours for 6 classes
    cmap   = plt.cm.get_cmap("tab10", n_classes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, emb, cls_arr, domain_label in [
        (axes[0], synth_emb, synth_cls, "Synthetic"),
        (axes[1], real_emb,  real_cls,  "Real"),
    ]:
        for c_idx in range(n_classes):
            mask = cls_arr == c_idx
            if mask.sum() == 0:
                continue
            cls_name = class_names[c_idx] if c_idx < len(class_names) else str(c_idx)
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=[cmap(c_idx)], label=cls_name,
                s=18, alpha=0.7, linewidths=0,
            )
        ax.set_title(f"{domain_label} features")
        ax.legend(fontsize=8, markerscale=1.5, loc="best")
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.grid(True, linewidth=0.3, alpha=0.5)

    fig.suptitle(title or "Feature Space (t-SNE)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[t-SNE] Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 4. PLOT_INTENSITY_HISTOGRAMS
# ══════════════════════════════════════════════════════════════

def plot_intensity_histograms(
    synth_dir: str,
    real_dir:  str,
    save_path: Optional[str] = None,
) -> None:
    """
    Pixel-intensity histogram comparison — visualises the low-level domain
    gap between Stable Diffusion synthetic IR images and real FLIR imagery.
    """
    def _collect_pixels(root: str, max_images: int = 200) -> np.ndarray:
        """Walk all class subdirectories and collect flattened pixel values."""
        all_paths = []
        # Collect PNG and JPG from every class subfolder
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            all_paths.extend(
                glob.glob(os.path.join(root, "**", ext), recursive=True)
            )

        if len(all_paths) == 0:
            print(f"[Histogram] Warning: no images found under {root}")
            return np.array([128.0])  # fallback so the plot doesn't crash

        # Randomly subsample to keep memory reasonable
        random.shuffle(all_paths)
        selected = all_paths[:max_images]

        pixels = []
        for path in selected:
            try:
                arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
                pixels.append(arr.ravel())  # flatten spatial dims
            except Exception:
                pass  # skip corrupt files silently

        return np.concatenate(pixels) if pixels else np.array([128.0])

    print("[Histogram] Collecting synthetic pixel values...")
    synth_pixels = _collect_pixels(synth_dir)
    print("[Histogram] Collecting real pixel values...")
    real_pixels  = _collect_pixels(real_dir)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Plot both distributions on the same axes for easy visual comparison
    ax.hist(
        synth_pixels, bins=128, range=(0, 255),
        color="steelblue", alpha=0.65, density=True, label="Synthetic (SD)"
    )
    ax.hist(
        real_pixels, bins=128, range=(0, 255),
        color="darkorange", alpha=0.65, density=True, label="Real (FLIR)"
    )

    ax.set_xlabel("Pixel Intensity", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Pixel Intensity Distribution: Synthetic vs Real", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # Annotate the means — useful for your report ("real images are ~X brighter")
    ax.axvline(synth_pixels.mean(), color="steelblue",   linestyle="--",
               linewidth=1.2, label=f"Synth mean={synth_pixels.mean():.1f}")
    ax.axvline(real_pixels.mean(),  color="darkorange",  linestyle="--",
               linewidth=1.2, label=f"Real  mean={real_pixels.mean():.1f}")
    ax.legend(fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Histogram] Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# 5. PLOT_GAP_REDUCTION
# ══════════════════════════════════════════════════════════════

def plot_gap_reduction(
    results:   list,
    save_path: Optional[str] = None,
) -> None:
    """
    Grouped bar chart of synthetic vs real accuracy for each adaptation
    strategy — the headline results figure for the ATR cadet report.

    Each element of `results` must be a dict with keys:
        "name"      : str            — experiment label
        "synth_acc" : float or None  — accuracy on synthetic test set
        "real_acc"  : float or None  — accuracy on real test set
    """
    n = len(results)
    if n == 0:
        print("[GapReduction] Empty results list — nothing to plot.")
        return

    names      = [r["name"]      for r in results]
    synth_accs = [r.get("synth_acc") or 0.0 for r in results]
    real_accs  = [r.get("real_acc")  or 0.0 for r in results]

    x     = np.arange(n)
    width = 0.35   # width of each bar in the pair

    fig, ax = plt.subplots(figsize=(max(8, n * 1.8), 6))

    bars_s = ax.bar(x - width / 2, synth_accs, width,
                    label="Synthetic acc", color="steelblue",  alpha=0.85)
    bars_r = ax.bar(x + width / 2, real_accs,  width,
                    label="Real acc",      color="darkorange", alpha=0.85)

    def _label_bars(bars, accs, source_results, key):
        """Add numeric labels above each bar; skip None entries."""
        for bar, orig_val in zip(bars, [r.get(key) for r in source_results]):
            if orig_val is None:
                continue  # bar is at 0 due to the `or 0.0` above; don't label it
            height = bar.get_height()
            ax.annotate(
                f"{height:.1%}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),  # 4pt vertical offset
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5,
            )

    _label_bars(bars_s, synth_accs, results, "synth_acc")
    _label_bars(bars_r, real_accs,  results, "real_acc")

    # Draw the gap as a thin red line between synth and real bars for each exp
    for i, r in enumerate(results):
        s = r.get("synth_acc")
        re = r.get("real_acc")
        if s is not None and re is not None:
            ax.plot(
                [x[i] - width / 2, x[i] + width / 2], [s, re],
                color="crimson", linewidth=1.2, linestyle="--", alpha=0.7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0.0, 1.08)  # leave headroom above 100% for labels
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Domain Gap Reduction Across Strategies", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[GapReduction] Saved → {save_path}")
    plt.show()
