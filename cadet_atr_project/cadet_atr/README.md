# Cadet ATR Project — Synthetic-to-Real Domain Adaptation
### IR Target Classification · Based on Hanwha Systems (JKSCI 2025)

---

## Project structure

```
cadet_atr/
├── data/
│   ├── dataset.py          # PyTorch Dataset classes for synthetic + real IR
│   └── augmentation.py     # IR-optimised augmentation pipeline (Kornia)
├── models/
│   └── convnext.py         # ConvNeXt model setup with transfer learning
├── training/
│   └── trainer.py          # Training loop with W&B logging + checkpointing
├── evaluation/
│   └── evaluator.py        # Gap measurement, confusion matrix, per-class metrics
├── adaptation/
│   ├── histogram.py        # Histogram matching (synthetic → real distribution)
│   ├── domain_random.py    # Domain randomisation augmentation
│   └── finetune.py         # Fine-tuning strategies (head-only, full, layerwise)
├── utils/
│   ├── visualise.py        # t-SNE plots, intensity histograms, grad-CAM
│   └── config.py           # All hyperparameters in one place
├── notebooks/
│   └── 00_full_pipeline.ipynb  # Single Colab notebook — runs everything
├── generate_synthetic.py   # Generate synthetic IR images with Stable Diffusion
├── run_experiment.py       # Main entry point — runs full experiment pipeline
└── README.md
```

## Quickstart (Google Colab)

```python
# 1. Clone repo
!git clone https://github.com/YOUR_USERNAME/cadet_atr.git
%cd cadet_atr

# 2. Install dependencies
!pip install -q torch torchvision timm kornia scikit-learn scikit-image \
             wandb diffusers accelerate grad-cam pandas seaborn tqdm

# 3. Generate synthetic data
!python generate_synthetic.py --classes aircraft vessel vehicle --n 200

# 4. Run full experiment
!python run_experiment.py --mode full
```

## Results table (fill in as you go)

| Experiment                   | Synth acc | Real acc | Gap   |
|------------------------------|-----------|----------|-------|
| Paper baseline (Hanwha 2025) | ~90.25%   | —        | ?     |
| Your synthetic baseline      | TBD       | TBD      | TBD   |
| + Histogram matching         | TBD       | TBD ↑    | TBD   |
| + Domain randomisation       | TBD       | TBD ↑    | TBD   |
| + Fine-tuning on real data   | TBD       | TBD ↑    | TBD   |
| Final best model             | TBD       | TBD ↑    | smallest ✓ |
