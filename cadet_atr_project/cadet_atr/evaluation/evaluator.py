# evaluation/evaluator.py
# ─────────────────────────────────────────────────────────────
# All evaluation logic in one place:
#   - Per-class accuracy, F1, precision, recall
#   - Domain gap measurement (synthetic acc vs real acc)
#   - Confusion matrix plotting
#   - Ensemble inference (Geometric TTA, Averaging, Voting)
# ─────────────────────────────────────────────────────────────

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from utils.config import cfg


# ── Single model evaluation ───────────────────────────────────

def evaluate(
    model:       nn.Module,
    loader:      DataLoader,
    class_names: Optional[List[str]] = None,
    device:      str = "cuda",
) -> Dict:
    """
    Evaluate a single model on a DataLoader.

    Returns dict with:
        accuracy, per-class precision/recall/F1,
        confusion_matrix, all_preds, all_labels
    """
    if class_names is None:
        class_names = cfg.class_names

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval", leave=False):
            x, y = batch[0].to(device), batch[1]
            logits = model(x)
            probs  = F.softmax(logits, dim=1).cpu()
            preds  = logits.argmax(1).cpu()

            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class AUC (one-vs-rest)
    try:
        auc = roc_auc_score(
            all_labels, all_probs,
            multi_class="ovr", average=None
        )
    except Exception:
        auc = [None] * len(class_names)

    return {
        "accuracy":         report["accuracy"],
        "report":           report,
        "confusion_matrix": cm,
        "all_preds":        all_preds,
        "all_labels":       all_labels,
        "all_probs":        all_probs,
        "auc_per_class":    auc,
    }


# ── Domain gap measurement ────────────────────────────────────

def measure_domain_gap(
    model:        nn.Module,
    synth_loader: DataLoader,
    real_loader:  DataLoader,
    class_names:  Optional[List[str]] = None,
    save_path:    Optional[str] = None,
) -> Dict:
    """
    Core experiment: evaluate model on BOTH synthetic and real
    test sets and compute the domain gap.

    This is the project's headline result.

    Returns:
        {
            "synth_acc":    float,
            "real_acc":     float,
            "domain_gap":   float,   # synth_acc - real_acc
            "synth_results": dict,
            "real_results":  dict,
        }
    """
    if class_names is None:
        class_names = cfg.class_names

    print("[Gap Measurement] Evaluating on synthetic test set...")
    synth_results = evaluate(model, synth_loader, class_names)

    print("[Gap Measurement] Evaluating on real test set...")
    real_results  = evaluate(model, real_loader,  class_names)

    acc_s = synth_results["accuracy"]
    acc_r = real_results["accuracy"]
    gap   = acc_s - acc_r

    print("\n" + "─" * 50)
    print(f"  Synthetic accuracy : {acc_s:.1%}")
    print(f"  Real accuracy      : {acc_r:.1%}")
    print(f"  Domain gap         : {gap:.1%}  ← headline result")
    print("─" * 50)

    # Per-class breakdown
    print("\nPer-class accuracy breakdown:")
    print(f"  {'Class':<15} {'Synthetic':>10} {'Real':>10} {'Gap':>10}")
    for cls in class_names:
        s = synth_results["report"][cls]["recall"]
        r = real_results["report"][cls]["recall"]
        print(f"  {cls:<15} {s:>10.1%} {r:>10.1%} {s-r:>10.1%}")

    # Plot confusion matrices side by side
    _plot_confusion_matrices(
        synth_results["confusion_matrix"],
        real_results["confusion_matrix"],
        class_names, acc_s, acc_r, save_path
    )

    return {
        "synth_acc":     acc_s,
        "real_acc":      acc_r,
        "domain_gap":    gap,
        "synth_results": synth_results,
        "real_results":  real_results,
    }


def _plot_confusion_matrices(
    cm_synth:    np.ndarray,
    cm_real:     np.ndarray,
    class_names: List[str],
    acc_s:       float,
    acc_r:       float,
    save_path:   Optional[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, cm, title, acc in zip(
        axes,
        [cm_synth,     cm_real],
        ["Synthetic test set", "Real test set"],
        [acc_s,        acc_r],
    ):
        # Normalise rows to show percentages
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm, annot=cm, fmt="d", ax=ax, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            vmin=0, vmax=1
        )
        ax.set_title(f"{title}  (accuracy = {acc:.1%})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Confusion matrices saved → {save_path}")
    plt.show()


# ── Ensemble inference ────────────────────────────────────────

def ensemble_predict(
    models:  List[nn.Module],
    x:       torch.Tensor,
    method:  str = "averaging",   # "averaging" | "voting" | "geometric"
) -> torch.Tensor:
    """
    Run ensemble inference on a batch tensor x.

    Methods:
        averaging: mean of softmax probabilities across all models (best)
        voting:    hard majority vote across all models
        geometric: Test-Time Augmentation on a single model (models[0])

    Returns:
        Predicted class indices — shape [B]
    """
    if method == "averaging":
        probs = [F.softmax(m(x), dim=1) for m in models]
        return torch.stack(probs).mean(0).argmax(1)

    elif method == "voting":
        preds = [m(x).argmax(1) for m in models]
        preds = torch.stack(preds, dim=1)   # [B, num_models]
        # Mode along model dimension
        return torch.mode(preds, dim=1).values

    elif method == "geometric":
        # TTA on models[0]: 8 variants (original + flips + rotations)
        model = models[0]
        variants = _generate_tta_variants(x)
        probs = [F.softmax(model(v), dim=1) for v in variants]
        return torch.stack(probs).mean(0).argmax(1)

    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def _generate_tta_variants(x: torch.Tensor) -> List[torch.Tensor]:
    """
    Generate 8 TTA variants: original + flips + 4 rotations.
    Matches the paper's Geometric ensemble exactly.
    """
    variants = [x]
    # Horizontal flip
    x_hflip = torch.flip(x, dims=[3])
    variants.append(x_hflip)
    # 4 rotation angles: 0°, 90°, 180°, 270°
    for k in range(1, 4):
        variants.append(torch.rot90(x,       k=k, dims=[2, 3]))
        variants.append(torch.rot90(x_hflip, k=k, dims=[2, 3]))
    return variants[:8]   # exactly 8 variants


def evaluate_ensemble(
    models:      List[nn.Module],
    loader:      DataLoader,
    method:      str = "averaging",
    class_names: Optional[List[str]] = None,
    device:      str = "cuda",
) -> Dict:
    """
    Evaluate an ensemble on a full DataLoader.
    """
    if class_names is None:
        class_names = cfg.class_names

    all_preds, all_labels = [], []
    for model in models:
        model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  ensemble({method})", leave=False):
            x, y = batch[0].to(device), batch[1]
            preds = ensemble_predict(models, x, method=method)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True, zero_division=0
    )
    return {
        "accuracy": report["accuracy"],
        "report":   report,
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
    }
