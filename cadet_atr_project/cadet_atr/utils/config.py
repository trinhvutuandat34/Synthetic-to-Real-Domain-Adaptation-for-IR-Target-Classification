# utils/config.py
# ─────────────────────────────────────────────────────────────
# Central configuration — change values HERE, not scattered
# across training scripts.
# ─────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:

    # ── Project ───────────────────────────────────────────────
    project_name: str = "cadet-atr"
    run_name:     str = "baseline"          # override per experiment
    seed:         int = 42

    # ── Classes ───────────────────────────────────────────────
    # Simplified from paper's 6 military classes to 3 broader
    # categories — easier to source real data for.
    class_names: List[str] = field(
        default_factory=lambda: ["aircraft", "vessel", "vehicle"]
    )

    # ── Paths ─────────────────────────────────────────────────
    synth_dir:      str = "data/synthetic/"     # generated IR images
    real_dir:       str = "data/real/"          # FLIR / KAIST images
    checkpoint_dir: str = "checkpoints/"
    results_dir:    str = "results/"

    # ── Model ─────────────────────────────────────────────────
    model_name:      str   = "convnext_tiny"    # matches paper
    pretrained:      bool  = True               # ImageNet weights
    num_classes:     int   = 3
    input_size:      int   = 224                # matches paper

    # ── Training ──────────────────────────────────────────────
    epochs:          int   = 50
    batch_size:      int   = 32                 # lower than paper (128)
                                                # for Colab memory safety
    learning_rate:   float = 1e-4
    weight_decay:    float = 1e-2
    early_stop_patience: int = 7

    # ── Data splits ───────────────────────────────────────────
    train_frac: float = 0.70
    val_frac:   float = 0.15
    # test_frac  = 1 - train_frac - val_frac = 0.15

    # ── Augmentation ──────────────────────────────────────────
    aug_prob:       float = 0.5    # probability per transform (paper used 0.5)
    aug_noise_std:  float = 0.05   # Gaussian noise std
    aug_brightness: tuple = (0.7, 1.3)
    aug_contrast:   tuple = (0.7, 1.3)

    # ── Fine-tuning (adaptation phase) ────────────────────────
    finetune_epochs:    int   = 20
    finetune_lr:        float = 1e-4   # head-only
    finetune_lr_full:   float = 5e-6   # full fine-tune (lower to avoid forgetting)
    finetune_freeze:    bool  = True   # True = head-only, False = full


# ── Singleton — import this everywhere ───────────────────────
cfg = Config()
