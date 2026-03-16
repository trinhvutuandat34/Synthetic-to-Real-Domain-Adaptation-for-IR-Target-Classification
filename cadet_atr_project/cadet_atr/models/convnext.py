# models/convnext.py
# ─────────────────────────────────────────────────────────────
# ConvNeXt model setup for IR target classification.
#
# Mirrors the paper's approach:
#   - Pre-trained on ImageNet (1,000 classes)
#   - Final Linear layer replaced: 768 → num_classes
#   - Optional: freeze backbone for head-only fine-tuning
#
# Also provides a multi-model builder for ensemble experiments.
# ─────────────────────────────────────────────────────────────

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models

from utils.config import cfg


# ── Available models (matching the paper's 6) ────────────────

MODEL_ZOO: Dict[str, callable] = {
    "convnext_tiny":       lambda: models.convnext_tiny(weights="IMAGENET1K_V1"),
    "resnet18":            lambda: models.resnet18(weights="IMAGENET1K_V1"),
    "vgg16":               lambda: models.vgg16(weights="IMAGENET1K_V1"),
    "resnext50_32x4d":     lambda: models.resnext50_32x4d(weights="IMAGENET1K_V1"),
    "vit_b_16":            lambda: models.vit_b_16(weights="IMAGENET1K_V1"),
    "swin_t":              lambda: models.swin_t(weights="IMAGENET1K_V1"),
}

# Feature dimensions of each model's penultimate layer
FEATURE_DIMS: Dict[str, int] = {
    "convnext_tiny":   768,
    "resnet18":        512,
    "vgg16":           4096,
    "resnext50_32x4d": 2048,
    "vit_b_16":        768,
    "swin_t":          768,
}


def build_model(
    model_name:      str  = cfg.model_name,
    num_classes:     int  = cfg.num_classes,
    freeze_backbone: bool = False,
    pretrained:      bool = cfg.pretrained,
) -> nn.Module:
    """
    Build a single classification model with transfer learning.

    Args:
        model_name:      key from MODEL_ZOO
        num_classes:     output classes (3 for your project, 6 for the paper)
        freeze_backbone: if True, only the classifier head trains.
                         Use for small real datasets (<200 images/class).
        pretrained:      load ImageNet weights (always True for transfer learning)

    Returns:
        model on CUDA (if available)
    """
    if model_name not in MODEL_ZOO:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_ZOO.keys())}"
        )

    model = MODEL_ZOO[model_name]()

    # ── Replace classifier head ───────────────────────────────
    feat_dim = FEATURE_DIMS[model_name]
    _replace_head(model, model_name, feat_dim, num_classes)

    # ── Optionally freeze backbone ────────────────────────────
    if freeze_backbone:
        for name, param in model.named_parameters():
            # Keep classifier trainable; freeze everything else
            if "classifier" not in name and "head" not in name and "fc" not in name:
                param.requires_grad = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    n_params       = sum(p.numel() for p in model.parameters())
    n_trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_name} | "
          f"total={n_params/1e6:.1f}M | "
          f"trainable={n_trainable/1e6:.1f}M | "
          f"freeze_backbone={freeze_backbone}")
    return model


def _replace_head(
    model:       nn.Module,
    model_name:  str,
    feat_dim:    int,
    num_classes: int,
) -> None:
    """
    Replace the final classification layer in-place.
    Each architecture stores its head differently.
    """
    new_head = nn.Linear(feat_dim, num_classes)

    if "convnext" in model_name or "swin" in model_name:
        # ConvNeXt and Swin: model.classifier[-1]  or  model.head
        if hasattr(model, "classifier"):
            model.classifier[-1] = new_head
        else:
            model.head = new_head

    elif "vit" in model_name:
        model.heads.head = new_head

    elif "resnet" in model_name or "resnext" in model_name:
        model.fc = new_head

    elif "vgg" in model_name:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)


# ── Ensemble builder ──────────────────────────────────────────

def build_ensemble(
    model_names: Optional[List[str]] = None,
    num_classes: int = cfg.num_classes,
    checkpoint_paths: Optional[Dict[str, str]] = None,
) -> List[nn.Module]:
    """
    Build a list of models for ensemble inference.

    Args:
        model_names:       list of model keys from MODEL_ZOO.
                           Defaults to all 6 (matching the paper).
        num_classes:       output classes
        checkpoint_paths:  {model_name: path_to_checkpoint}
                           If provided, loads saved weights.

    Returns:
        List of models on CUDA, all in eval() mode.
    """
    if model_names is None:
        model_names = list(MODEL_ZOO.keys())

    ensemble = []
    for name in model_names:
        model = build_model(name, num_classes=num_classes)
        if checkpoint_paths and name in checkpoint_paths:
            state = torch.load(checkpoint_paths[name], map_location="cuda")
            model.load_state_dict(state)
            print(f"[Ensemble] Loaded checkpoint for {name}")
        model.eval()
        ensemble.append(model)

    print(f"[Ensemble] Built {len(ensemble)} models: {model_names}")
    return ensemble


# ── Feature extractor (for t-SNE) ────────────────────────────

class FeatureExtractor(nn.Module):
    """
    Wraps a model to return penultimate-layer features
    instead of class logits. Used for t-SNE visualisation.

    Usage:
        extractor = FeatureExtractor(model, "convnext_tiny")
        features = extractor(batch)   # shape: [B, feat_dim]
    """

    def __init__(self, model: nn.Module, model_name: str):
        super().__init__()
        self.model      = model
        self.model_name = model_name
        self._features  = []

        # Register a forward hook on the layer before the classifier
        self._register_hook(model, model_name)

    def _register_hook(self, model: nn.Module, model_name: str) -> None:
        def hook(module, input, output):
            # Flatten spatial dimensions if needed
            if output.dim() > 2:
                self._features.append(output.mean([-2, -1]).detach().cpu())
            else:
                self._features.append(output.detach().cpu())

        if "convnext" in model_name or "swin" in model_name:
            if hasattr(model, "classifier"):
                # Hook the layer before the final Linear
                model.classifier[-2].register_forward_hook(hook)
            else:
                model.avgpool.register_forward_hook(hook)
        elif "vit" in model_name:
            model.encoder.register_forward_hook(hook)
        elif "resnet" in model_name or "resnext" in model_name:
            model.avgpool.register_forward_hook(hook)
        elif "vgg" in model_name:
            model.classifier[-2].register_forward_hook(hook)

    def extract(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run loader through model and return (features, labels).
        """
        self.model.eval()
        self._features.clear()
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                x, y = batch[0], batch[1]
                self.model(x.cuda())
                all_labels.extend(y.numpy())

        features = torch.cat(self._features, dim=0)
        labels   = torch.tensor(all_labels)
        self._features.clear()
        return features, labels
