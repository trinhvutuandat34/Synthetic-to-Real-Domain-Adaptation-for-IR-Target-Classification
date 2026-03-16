# run_experiment.py
# ─────────────────────────────────────────────────────────────
# Main entry point. Runs the full experimental pipeline:
#
#   Phase 1 — Train baseline on synthetic data
#   Phase 2 — Measure domain gap (baseline)
#   Phase 3 — Apply adaptation strategies
#   Phase 4 — Measure gap after each strategy
#   Phase 5 — Generate all report figures
#
# Usage:
#   python run_experiment.py --mode full
#   python run_experiment.py --mode baseline_only
#   python run_experiment.py --mode gap_only   --checkpoint checkpoints/baseline_best.pt
#   python run_experiment.py --mode adapt      --checkpoint checkpoints/baseline_best.pt
# ─────────────────────────────────────────────────────────────

import argparse
import os
import json
from pathlib import Path

import torch

from utils.config import cfg
from models.convnext import build_model
from data.dataset import make_loaders, make_real_loader
from training.trainer import Trainer
from evaluation.evaluator import measure_domain_gap
from adaptation.strategies import (
    build_reference_histogram,
    apply_histogram_matching,
    RealDataFinetuner,
)
from utils.visualise import (
    plot_tsne,
    plot_intensity_histograms,
    plot_gap_reduction,
)


def run_full_pipeline(args) -> None:
    """
    Runs everything end-to-end.
    Best for: first time running the project.
    """
    print("=" * 60)
    print("  CADET ATR — Full Experimental Pipeline")
    print("=" * 60)

    # Create output directories
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    results_log = []   # will hold gap numbers at each stage

    # ── Phase 1: Train baseline ───────────────────────────────
    print("\n[Phase 1] Training synthetic baseline...")
    model = build_model(model_name=cfg.model_name, num_classes=cfg.num_classes)

    train_loader, val_loader, synth_test_loader = make_loaders(
        dataset_type="synthetic", batch_size=cfg.batch_size
    )
    real_test_loader = make_real_loader(batch_size=cfg.batch_size)

    trainer  = Trainer(model, run_name="baseline", aug_level="baseline")
    ckpt_path = trainer.fit(train_loader, val_loader)

    # ── Phase 2: Measure baseline gap ─────────────────────────
    print("\n[Phase 2] Measuring domain gap — baseline...")
    model.load_state_dict(torch.load(ckpt_path))

    gap_results = measure_domain_gap(
        model, synth_test_loader, real_test_loader,
        save_path=f"{cfg.results_dir}confusion_baseline.png"
    )
    results_log.append({
        "name":      "Synthetic baseline",
        "synth_acc": gap_results["synth_acc"],
        "real_acc":  gap_results["real_acc"],
    })
    _save_results(results_log, f"{cfg.results_dir}results.json")

    # t-SNE before adaptation
    plot_tsne(
        model, synth_test_loader, real_test_loader,
        title="Feature space — BEFORE adaptation",
        save_path=f"{cfg.results_dir}tsne_before.png"
    )

    # Intensity histograms
    plot_intensity_histograms(
        synth_dir=cfg.synth_dir, real_dir=cfg.real_dir,
        save_path=f"{cfg.results_dir}intensity_histograms.png"
    )

    # ── Phase 3a: Histogram matching ─────────────────────────
    print("\n[Phase 3a] Strategy 1 — Histogram matching...")
    reference = build_reference_histogram(cfg.real_dir)
    apply_histogram_matching(
        synth_img_dir = cfg.synth_dir,
        output_dir    = "data/synthetic_matched/",
        reference     = reference,
    )

    # Retrain on histogram-matched images
    from data.dataset import SyntheticIRDataset
    from data.augmentation import build_augmentation_pipeline

    model_hm = build_model()
    train_ldr_hm, val_ldr_hm, test_ldr_hm = make_loaders(
        dataset_type="synthetic", batch_size=cfg.batch_size
    )
    # Swap to matched directory (quick override)
    train_ldr_hm.dataset.dataset.root = "data/synthetic_matched/"

    trainer_hm = Trainer(
        model_hm, run_name="histmatch", aug_level="baseline"
    )
    ckpt_hm = trainer_hm.fit(train_ldr_hm, val_ldr_hm)
    model_hm.load_state_dict(torch.load(ckpt_hm))

    gap_hm = measure_domain_gap(
        model_hm, test_ldr_hm, real_test_loader,
        save_path=f"{cfg.results_dir}confusion_histmatch.png"
    )
    results_log.append({
        "name":      "+ Histogram matching",
        "synth_acc": gap_hm["synth_acc"],
        "real_acc":  gap_hm["real_acc"],
    })
    _save_results(results_log, f"{cfg.results_dir}results.json")

    # ── Phase 3b: Domain randomisation ───────────────────────
    print("\n[Phase 3b] Strategy 2 — Domain randomisation...")
    model_dr = build_model()
    trainer_dr = Trainer(
        model_dr, run_name="domain_random", aug_level="extended"
    )
    ckpt_dr = trainer_dr.fit(train_loader, val_loader)
    model_dr.load_state_dict(torch.load(ckpt_dr))

    gap_dr = measure_domain_gap(
        model_dr, synth_test_loader, real_test_loader,
        save_path=f"{cfg.results_dir}confusion_domainrandom.png"
    )
    results_log.append({
        "name":      "+ Domain randomisation",
        "synth_acc": gap_dr["synth_acc"],
        "real_acc":  gap_dr["real_acc"],
    })
    _save_results(results_log, f"{cfg.results_dir}results.json")

    # ── Phase 3c: Fine-tuning on real data ────────────────────
    print("\n[Phase 3c] Strategy 3 — Fine-tuning on real data...")
    real_train_loader, real_val_loader, real_test_loader2 = make_loaders(
        dataset_type="real", batch_size=cfg.batch_size
    )

    # Start from the domain-randomisation checkpoint (best so far)
    finetuner = RealDataFinetuner(
        model          = build_model(),
        checkpoint_path = ckpt_dr,
        save_path       = f"{cfg.checkpoint_dir}finetuned_best.pt",
    )
    adapted_model = finetuner.finetune(
        real_train_loader, real_val_loader, mode="head_only"
    )

    gap_ft = measure_domain_gap(
        adapted_model, synth_test_loader, real_test_loader,
        save_path=f"{cfg.results_dir}confusion_finetuned.png"
    )
    results_log.append({
        "name":      "+ Fine-tuning (head)",
        "synth_acc": gap_ft["synth_acc"],
        "real_acc":  gap_ft["real_acc"],
    })
    _save_results(results_log, f"{cfg.results_dir}results.json")

    # t-SNE after adaptation
    plot_tsne(
        adapted_model, synth_test_loader, real_test_loader,
        title="Feature space — AFTER adaptation",
        save_path=f"{cfg.results_dir}tsne_after.png"
    )

    # ── Phase 4: Final results chart ─────────────────────────
    print("\n[Phase 4] Generating results figures...")
    plot_gap_reduction(
        results_log,
        save_path=f"{cfg.results_dir}gap_reduction.png"
    )

    # ── Final summary ─────────────────────────────────────────
    _print_final_summary(results_log)


def run_baseline_only(args) -> None:
    """Train the baseline model only. Quick start for Week 5."""
    model = build_model()
    train_loader, val_loader, _ = make_loaders("synthetic", cfg.batch_size)
    trainer = Trainer(model, run_name="baseline", aug_level="baseline")
    ckpt = trainer.fit(train_loader, val_loader)
    print(f"\nBaseline checkpoint saved: {ckpt}")
    print("Next: run with --mode gap_only --checkpoint", ckpt)


def run_gap_only(args) -> None:
    """Measure the gap from an existing checkpoint."""
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for gap_only mode")

    model = build_model()
    model.load_state_dict(torch.load(args.checkpoint))
    _, _, synth_test_loader = make_loaders("synthetic", cfg.batch_size)
    real_loader = make_real_loader(cfg.batch_size)

    measure_domain_gap(
        model, synth_test_loader, real_loader,
        save_path=f"{cfg.results_dir}confusion_gap.png"
    )
    plot_tsne(
        model, synth_test_loader, real_loader,
        save_path=f"{cfg.results_dir}tsne_gap.png"
    )


def _save_results(results: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results logged → {path}")


def _print_final_summary(results: list) -> None:
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Experiment':<28} {'Synth':>7} {'Real':>7} {'Gap':>7}")
    print("  " + "─" * 54)
    for r in results:
        s = f"{r['synth_acc']:.1%}" if r['synth_acc'] else "—"
        re = f"{r['real_acc']:.1%}"  if r['real_acc']  else "—"
        g  = f"{(r['synth_acc'] - r['real_acc']):.1%}" if (
            r['synth_acc'] and r['real_acc']) else "—"
        print(f"  {r['name']:<28} {s:>7} {re:>7} {g:>7}")
    print("=" * 60)
    print(f"\n  All figures saved to: {cfg.results_dir}")
    print(f"  All checkpoints in:   {cfg.checkpoint_dir}")


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cadet ATR — domain adaptation experiment runner"
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "baseline_only", "gap_only", "adapt"],
        help="Which part of the pipeline to run"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (required for gap_only / adapt)"
    )
    args = parser.parse_args()

    dispatch = {
        "full":          run_full_pipeline,
        "baseline_only": run_baseline_only,
        "gap_only":      run_gap_only,
    }
    dispatch[args.mode](args)
