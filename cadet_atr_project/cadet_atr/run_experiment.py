# run_experiment.py  (fixed)
# ─────────────────────────────────────────────────────────────
# Fixes applied:
#   FIX-1  Added run_adapt_strategy() — the README promises
#          `--mode adapt --strategy X` but the original dispatch
#          dict had no "adapt" key, causing an immediate KeyError.
#   FIX-2  dispatch dict now maps "adapt" correctly.
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
    DANNTrainer,
)
from utils.visualise import (
    plot_tsne,
    plot_intensity_histograms,
    plot_gap_reduction,
)


# ── Helpers ───────────────────────────────────────────────────

def _save_results(results: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results logged → {path}")


def _print_final_summary(results: list) -> None:
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Experiment':<30} {'Synth':>7} {'Real':>7} {'Gap':>7}")
    print("  " + "─" * 56)
    for r in results:
        s  = f"{r['synth_acc']:.1%}" if r.get('synth_acc') else "—"
        re = f"{r['real_acc']:.1%}"  if r.get('real_acc')  else "—"
        g  = (
            f"{(r['synth_acc'] - r['real_acc']):.1%}"
            if (r.get('synth_acc') and r.get('real_acc')) else "—"
        )
        print(f"  {r['name']:<30} {s:>7} {re:>7} {g:>7}")
    print("=" * 60)
    print(f"\n  Figures → {cfg.results_dir}")
    print(f"  Checkpoints → {cfg.checkpoint_dir}")


# ── FIX-1: per-strategy adapt runner ─────────────────────────

def run_adapt_strategy(args) -> None:
    """
    Run a single adaptation strategy by name.

    Usage:
        python run_experiment.py --mode adapt --strategy histogram
        python run_experiment.py --mode adapt --strategy domain_random
        python run_experiment.py --mode adapt --strategy finetune
        python run_experiment.py --mode adapt --strategy dann

    Requires --checkpoint pointing at the baseline .pt file,
    EXCEPT for domain_random which retrains from ImageNet weights.
    """
    strategy = args.strategy
    if strategy is None:
        raise ValueError(
            "--strategy is required for --mode adapt.\n"
            "Choices: histogram | domain_random | finetune | dann"
        )

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, synth_test_loader = make_loaders(
        "synthetic", cfg.batch_size
    )
    real_test_loader = make_real_loader(cfg.batch_size)

    # ── Strategy 1: Histogram matching ───────────────────────
    if strategy == "histogram":
        print("\n[Adapt] Strategy 1 — Histogram matching")
        if not args.checkpoint:
            raise ValueError("--checkpoint required for histogram strategy")

        reference = build_reference_histogram(cfg.real_dir)
        apply_histogram_matching(
            synth_img_dir = cfg.synth_dir,
            output_dir    = "data/synthetic_matched/",
            reference     = reference,
        )

        model = build_model()
        train_ldr, val_ldr, test_ldr = make_loaders("synthetic", cfg.batch_size)
        # Redirect root to matched images
        train_ldr.dataset.dataset.root = "data/synthetic_matched/"

        trainer = Trainer(model, run_name="histmatch", aug_level="baseline")
        ckpt    = trainer.fit(train_ldr, val_ldr)
        model.load_state_dict(torch.load(ckpt))

        gap = measure_domain_gap(
            model, test_ldr, real_test_loader,
            save_path=f"{cfg.results_dir}confusion_histmatch.png"
        )
        print(f"\nHistogram matching — real acc: {gap['real_acc']:.1%}, "
              f"gap: {gap['domain_gap']:.1%}")

    # ── Strategy 2: Domain randomisation ─────────────────────
    elif strategy == "domain_random":
        print("\n[Adapt] Strategy 2 — Domain randomisation (extended aug)")
        model    = build_model()
        trainer  = Trainer(model, run_name="domain_random", aug_level="extended")
        ckpt     = trainer.fit(train_loader, val_loader)
        model.load_state_dict(torch.load(ckpt))

        gap = measure_domain_gap(
            model, synth_test_loader, real_test_loader,
            save_path=f"{cfg.results_dir}confusion_domainrandom.png"
        )
        print(f"\nDomain randomisation — real acc: {gap['real_acc']:.1%}, "
              f"gap: {gap['domain_gap']:.1%}")

    # ── Strategy 3: Fine-tuning on real data ─────────────────
    elif strategy == "finetune":
        print("\n[Adapt] Strategy 3 — Fine-tuning on real IR images")
        if not args.checkpoint:
            raise ValueError("--checkpoint required for finetune strategy")

        real_train_loader, real_val_loader, _ = make_loaders("real", cfg.batch_size)

        finetuner = RealDataFinetuner(
            model           = build_model(),
            checkpoint_path = args.checkpoint,
            save_path       = f"{cfg.checkpoint_dir}finetuned_best.pt",
        )
        model_ft = finetuner.finetune(
            real_train_loader, real_val_loader,
            mode = args.finetune_mode,    # "head_only" | "full" | "layer_wise"
        )

        gap = measure_domain_gap(
            model_ft, synth_test_loader, real_test_loader,
            save_path=f"{cfg.results_dir}confusion_finetuned.png"
        )
        print(f"\nFine-tuning — real acc: {gap['real_acc']:.1%}, "
              f"gap: {gap['domain_gap']:.1%}")

    # ── Strategy 4: DANN adversarial adaptation ───────────────
    elif strategy == "dann":
        print("\n[Adapt] Strategy 4 — DANN adversarial domain adaptation")
        if not args.checkpoint:
            raise ValueError("--checkpoint required for dann strategy")

        real_train_loader, real_val_loader, _ = make_loaders("real", cfg.batch_size)

        dann = DANNTrainer(
            backbone_checkpoint = args.checkpoint,
            synth_loader        = train_loader,
            real_loader         = real_train_loader,
            val_loader          = val_loader,
            save_path           = f"{cfg.checkpoint_dir}dann_best.pt",
            wandb_run_name      = "dann_strategy4",
        )
        model_dann = dann.train()

        gap = measure_domain_gap(
            model_dann, synth_test_loader, real_test_loader,
            save_path=f"{cfg.results_dir}confusion_dann.png"
        )
        plot_tsne(
            model_dann, synth_test_loader, real_test_loader,
            title     = "Feature space — AFTER DANN adaptation",
            save_path = f"{cfg.results_dir}tsne_after_dann.png",
        )
        print(f"\nDANN — real acc: {gap['real_acc']:.1%}, "
              f"gap: {gap['domain_gap']:.1%}")

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Choices: histogram | domain_random | finetune | dann"
        )


# ── Full pipeline (unchanged from original, shown for completeness) ──

def run_full_pipeline(args) -> None:
    """Run all four strategies end-to-end."""
    print("=" * 60)
    print("  CADET ATR — Full Experimental Pipeline")
    print("=" * 60)

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    results_log = []

    # Phase 1: Train baseline
    print("\n[Phase 1] Training synthetic baseline...")
    model = build_model(model_name=cfg.model_name, num_classes=cfg.num_classes)
    train_loader, val_loader, synth_test_loader = make_loaders("synthetic", cfg.batch_size)
    real_test_loader = make_real_loader(cfg.batch_size)

    trainer   = Trainer(model, run_name="baseline", aug_level="baseline")
    ckpt_path = trainer.fit(train_loader, val_loader)

    # Phase 2: Measure baseline gap
    model.load_state_dict(torch.load(ckpt_path))
    gap_baseline = measure_domain_gap(
        model, synth_test_loader, real_test_loader,
        save_path=f"{cfg.results_dir}confusion_baseline.png"
    )
    results_log.append({
        "name":      "Synthetic baseline",
        "synth_acc": gap_baseline["synth_acc"],
        "real_acc":  gap_baseline["real_acc"],
    })
    _save_results(results_log, f"{cfg.results_dir}results.json")

    plot_tsne(
        model, synth_test_loader, real_test_loader,
        title     = "Feature space — BEFORE adaptation",
        save_path = f"{cfg.results_dir}tsne_before.png",
    )
    plot_intensity_histograms(
        synth_dir = cfg.synth_dir,
        real_dir  = cfg.real_dir,
        save_path = f"{cfg.results_dir}intensity_histograms.png",
    )

    # Phase 3a: Histogram matching
    args.checkpoint   = ckpt_path
    args.strategy     = "histogram"
    args.finetune_mode = "head_only"
    run_adapt_strategy(args)

    # Phase 3b: Domain randomisation
    args.strategy = "domain_random"
    run_adapt_strategy(args)

    # Load the domain-random checkpoint as the base for strategies 3 & 4
    dr_ckpt   = f"{cfg.checkpoint_dir}domain_random_best.pt"
    args.checkpoint = dr_ckpt

    # Phase 3c: Fine-tuning
    args.strategy = "finetune"
    run_adapt_strategy(args)

    # Phase 3d: DANN
    args.strategy = "dann"
    run_adapt_strategy(args)

    _print_final_summary(results_log)


def run_baseline_only(args) -> None:
    model   = build_model()
    train_loader, val_loader, _ = make_loaders("synthetic", cfg.batch_size)
    trainer = Trainer(model, run_name="baseline", aug_level="baseline")
    ckpt    = trainer.fit(train_loader, val_loader)
    print(f"\nBaseline checkpoint saved: {ckpt}")
    print("Next: --mode gap_only --checkpoint", ckpt)


def run_gap_only(args) -> None:
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


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cadet ATR — domain adaptation experiment runner"
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "baseline_only", "gap_only", "adapt"],
        help="Which pipeline phase to run"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pt checkpoint (required for gap_only / adapt)"
    )
    # FIX-1: --strategy argument was missing from the original parser
    parser.add_argument(
        "--strategy", type=str, default=None,
        choices=["histogram", "domain_random", "finetune", "dann"],
        help="(--mode adapt only) Which adaptation strategy to run"
    )
    parser.add_argument(
        "--finetune_mode", type=str, default="head_only",
        choices=["head_only", "full", "layer_wise"],
        help="(--strategy finetune only) How to fine-tune the backbone"
    )
    args = parser.parse_args()

    # FIX-2: dispatch now includes "adapt" key
    dispatch = {
        "full":          run_full_pipeline,
        "baseline_only": run_baseline_only,
        "gap_only":      run_gap_only,
        "adapt":         run_adapt_strategy,   # ← was missing; KeyError before
    }
    dispatch[args.mode](args)
