# Cadet ATR Project — Synthetic-to-Real Domain Adaptation

### IR Target Classification · Based on Hanwha Systems (JKSCI 2025)

---

## Team

| Member | Role | Responsibilities | Future Plan |
|--------|------|-----------------|-------------|
| 뚜언닷 | Data & Training lead | Synthetic IR generation (Stable Diffusion), DANN (strategies.py Strategy 4), EO/IR dual-band fusion, airborne class expansion, trainer.py, W&B logging | Expand and harden the synthetic-to-real ATR pipeline for multi-domain air and naval operations — adding fixed-wing / rotary-wing / UAV classes and adversarial domain adaptation (DANN) targeting Air Force IRST and EOTS sensors |
| 이성제 | Eval & Report lead | ONNX/TorchScript export, 4-strategy domain gap analysis, t-SNE / Grad-CAM, evaluator.py, final report | Evaluate, deploy, and disseminate 뚜언닷's expanded ATR system as an edge-deployable intelligence asset — benchmarking on Jetson Nano, producing a Grad-CAM interpretability analysis for CMS operators, and targeting JKSCI / KIISE student track submission |

---

## Source paper

> **A Study on Deep Learning-based Automatic Target Recognition System in IR Image for Intelligent Combat Management System**
> Gyu-Seok Do, Ju-Mi Park, Won-Seok Jang, Young-Sub Yang, Ji-Seok Yoon
> Naval R&D Center, Hanwha Systems, Pangyo, Korea
> *Journal of The Korea Society of Computer and Information (JKSCI)*, Vol. 30 No. 1, pp. 33–40, January 2025
> DOI: `10.9708/jksci.2025.30.01.033`

Key result: ConvNeXt-tiny **90.25%** (single model) → **92%** (6-model softmax ensemble) across 6 anti-air / anti-ship target classes.

---

## Project structure

```
cadet_atr/
├── data/
│   ├── dataset.py              # PyTorch Dataset classes for synthetic + real IR
│   └── augmentation.py         # IR-optimised augmentation pipeline (Kornia)
├── models/
│   └── convnext.py             # ConvNeXt model setup with transfer learning
├── training/
│   └── trainer.py              # Training loop with W&B logging + checkpointing
├── evaluation/
│   └── evaluator.py            # Gap measurement, confusion matrix, per-class metrics
├── adaptation/
│   └── strategies.py           # All 4 domain adaptation strategies:
│                               #   Strategy 1 — Histogram matching
│                               #   Strategy 2 — Domain randomisation (BackgroundSwapDataset)
│                               #   Strategy 3 — Fine-tuning on real IR (RealDataFinetuner)
│                               #   Strategy 4 — DANN / GRL (DANNTrainer)  ← 뚜언닷
├── utils/
│   ├── visualise.py            # t-SNE plots, intensity histograms, Grad-CAM  ← 이성제
│   └── config.py               # All hyperparameters in one place
├── notebooks/
│   └── 00_full_pipeline.ipynb  # Single Colab notebook — runs everything
├── generate_synthetic.py       # Stable Diffusion synthetic IR image generator
├── run_experiment.py           # Main entry point — runs full experiment pipeline
└── README.md
```

---

## Target classes

| # | Class | Type | Branch relevance |
|---|-------|------|-----------------|
| 1 | `aircraft` (fixed-wing) | Airborne | Air Force IRST / EOTS |
| 2 | `aircraft` (rotary-wing) | Airborne | Air Force / Naval helicopter |
| 3 | `uav` | Airborne | Air Force / multi-domain |
| 4 | `vessel` | Surface | Naval CMS / EOTS |
| 5 | `vehicle` (ground) | Ground | Army / joint ops |
| 6 | `vehicle` (wheeled APC) | Ground | Army / joint ops |

Classes 1–3 added by 뚜언닷 in the airborne expansion phase (Plan 2).
Classes 4–6 carried over from the original 3-class baseline.

---

## Quickstart (Google Colab)

```python
# 1. Clone repo
!git clone https://github.com/YOUR_USERNAME/cadet_atr.git
%cd cadet_atr

# 2. Install dependencies
!pip install -q torch torchvision timm kornia scikit-learn scikit-image \
             wandb diffusers accelerate grad-cam pandas seaborn tqdm

# 3. Generate synthetic data — expanded 6-class set (뚜언닷)
!python generate_synthetic.py \
    --classes fixed_wing rotary_wing uav vessel vehicle_ground vehicle_apc \
    --n 200

# 4. Run full experiment (all 4 strategies)
!python run_experiment.py --mode full
```

---

## Running individual strategies

```python
# Strategy 1 — Histogram matching
!python run_experiment.py --mode adapt --strategy histogram

# Strategy 2 — Domain randomisation
!python run_experiment.py --mode adapt --strategy domain_random

# Strategy 3 — Fine-tuning on real IR (FLIR ADAS)
!python run_experiment.py --mode adapt --strategy finetune

# Strategy 4 — DANN adversarial adaptation (뚜언닷)
!python run_experiment.py --mode adapt --strategy dann

# Evaluate gap on a saved checkpoint (이성제)
!python run_experiment.py --mode gap_only --checkpoint checkpoints/dann_best.pt
```

---

## DANN quick smoke test (Colab cell)

```python
# Verify Strategy 4 loads and gradients flow correctly
from adaptation.strategies import DANNModel
import torch

model = DANNModel(num_classes=6)          # 6 classes after expansion
dummy = torch.randn(4, 3, 224, 224)

# Inference mode — domain head inactive, zero deployment overhead
cls_logits = model(dummy)
assert cls_logits.shape == (4, 6), "Class head shape wrong"

# Training mode — both heads active
cls_logits, dom_logits = model(dummy, return_domain=True)
assert dom_logits.shape == (4, 2), "Domain head shape wrong"

# GRL gradient check
dom_logits.sum().backward()
print("✓ All shapes correct. GRL gradients flow.")
```

---

## Results table (fill in as you go)

| Experiment | Synth acc | Real acc | Gap |
|---|---|---|---|
| Paper baseline (Hanwha 2025) | ~90.25% | — | ? |
| Synthetic baseline (3 classes) | TBD | TBD | TBD |
| + Histogram matching | TBD | TBD ↑ | TBD |
| + Domain randomisation | TBD | TBD ↑ | TBD |
| + Fine-tuning on real data | TBD | TBD ↑ | TBD |
| + DANN — Strategy 4 (뚜언닷) | TBD | TBD ↑ | TBD |
| Expanded baseline (6 classes) | TBD | TBD | TBD |
| Final best model (6 classes) | TBD | TBD ↑ | smallest ✓ |

Gap = Synth acc − Real acc. Target: gap < 10% on the 6-class set.

---

## Key config values (`utils/config.py`)

```python
# Classes — update after 뚜언닷's airborne expansion
class_names = ["fixed_wing", "rotary_wing", "uav",
               "vessel", "vehicle_ground", "vehicle_apc"]
num_classes = 6

# DANN hyperparameters — Strategy 4
dann_epochs     = 20
finetune_lr     = 1e-4    # shared with Strategy 3 and DANN
finetune_lr_full = 5e-6
```

---

## Checkpoint handoff (뚜언닷 → 이성제)

| File | Produced by | Used by |
|------|------------|---------|
| `checkpoints/baseline_best.pt` | 뚜언닷 (Phase 1) | 이성제 — gap baseline |
| `checkpoints/dann_best.pt` | 뚜언닷 (Phase 3c) | 이성제 — DANN gap eval, ONNX export |
| `results/tsne_before.png` | 뚜언닷 | 이성제 — report figure |
| `results/tsne_after_dann.png` | 이성제 (evaluator.py) | Final report |
| `results/gradcam_*.png` | 이성제 (visualise.py) | CMS operator briefing, report |

---

## Deployment target (이성제 — Phase 3d)

```python
# Export best DANN model to ONNX for edge deployment benchmark
import torch
from adaptation.strategies import DANNModel

model = DANNModel(num_classes=6)
model.load_state_dict(torch.load("checkpoints/dann_best.pt"))
model.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy,
    "checkpoints/dann_best.onnx",
    input_names  = ["ir_image"],
    output_names = ["class_logits"],
    opset_version = 17,
)
print("ONNX export complete — ready for Jetson Nano latency benchmark.")
```

---

## References

- Do, G.-S., Park, J.-M., Jang, W.-S., Yang, Y.-S., & Yoon, J.-S. (2025). A Study on Deep Learning-based Automatic Target Recognition System in IR Image for Intelligent Combat Management System. *Journal of The Korea Society of Computer and Information*, 30(1), 33–40. https://doi.org/10.9708/jksci.2025.30.01.033
- Ganin, Y., et al. (2016). Domain-Adversarial Training of Neural Networks. *JMLR*, 17(59), 1–35. https://arxiv.org/abs/1505.07818
- Liu, Z., et al. (2022). A ConvNet for the 2020s. *CVPR*. https://arxiv.org/abs/2201.03545
