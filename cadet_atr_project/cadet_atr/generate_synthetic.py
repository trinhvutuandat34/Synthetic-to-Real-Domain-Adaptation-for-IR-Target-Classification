# generate_synthetic.py
# ─────────────────────────────────────────────────────────────
# Generate synthetic IR images using Stable Diffusion.
# Run this ONCE at the start of your project to build your
# training dataset.
#
# Usage:
#   python generate_synthetic.py
#   python generate_synthetic.py --n 300 --classes aircraft vessel
#   python generate_synthetic.py --output data/synthetic_v2/
#
# Runtime: ~2–3 hours on Colab T4 for 600 images (200 × 3 classes)
# ─────────────────────────────────────────────────────────────

import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm


# ── IR-style prompts per class ────────────────────────────────
# These prompts are tuned to produce realistic-looking IR imagery.
# Tweak them if your generated images look too "photographic".

IR_PROMPTS = {
    "aircraft": [
        "infrared thermal image of military aircraft in flight, "
        "IR sensor view, grayscale thermal, hot engine glow, dark background",

        "thermal infrared image of jet fighter, EOTS sensor perspective, "
        "grayscale, glowing engine nozzle, night sky",

        "IR camera image of military helicopter, thermal signature, "
        "rotor heat, grayscale, overhead view",
    ],
    "vessel": [
        "infrared thermal image of naval vessel at sea, IR sensor, "
        "grayscale thermal, ship hull heat signature, dark ocean",

        "thermal infrared image of fast patrol boat, military, "
        "IR camera view, grayscale, engine heat, sea background",

        "EOTS infrared image of warship, thermal signature, "
        "grayscale, glowing engine exhausts, dark water",
    ],
    "vehicle": [
        "infrared thermal image of military tank, ground vehicle, "
        "IR sensor, grayscale, engine heat signature, dark terrain",

        "thermal camera image of armoured vehicle, military, "
        "IR grayscale, hot engine, overhead view",

        "infrared image of military truck convoy, thermal signature, "
        "IR camera, grayscale, engine heat patterns",
    ],
}

# Negative prompt: reduces photographic / unrealistic elements
NEGATIVE_PROMPT = (
    "color, rgb, photograph, realistic photo, hyperrealistic, "
    "text, watermark, logo, cartoon, painting"
)


def generate_images(
    output_dir: str = "data/synthetic/",
    classes:    list = None,
    n_per_class: int = 200,
    model_id:   str = "runwayml/stable-diffusion-v1-5",
    steps:      int = 30,
    seed:       int = 42,
) -> None:
    """
    Generate synthetic IR images for each class.

    Args:
        output_dir:   where to save images
        classes:      list of class names (default: all in IR_PROMPTS)
        n_per_class:  images per class
        model_id:     HuggingFace model ID
        steps:        diffusion steps (more = better quality, slower)
        seed:         for reproducibility
    """
    if classes is None:
        classes = list(IR_PROMPTS.keys())

    # Validate requested classes
    for cls in classes:
        if cls not in IR_PROMPTS:
            raise ValueError(
                f"Unknown class '{cls}'. "
                f"Available: {list(IR_PROMPTS.keys())}"
            )

    # Import here so the script fails fast if diffusers not installed
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError:
        raise ImportError(
            "Run: pip install diffusers accelerate\n"
            "Then restart the Colab runtime."
        )

    print(f"[Generate] Loading Stable Diffusion: {model_id}")
    print("  This downloads ~4GB on first run — it caches after that.")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.safety_checker = None    # disable — no NSFW risk for IR military imagery
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator().manual_seed(seed)

    total = len(classes) * n_per_class
    print(f"\n[Generate] Generating {total} images "
          f"({n_per_class} × {len(classes)} classes)")
    print(f"  Output → {output_dir}\n")

    for cls in classes:
        cls_dir = Path(output_dir) / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        prompts = IR_PROMPTS[cls]
        existing = len(list(cls_dir.glob("*.png")))

        if existing >= n_per_class:
            print(f"  {cls}: already has {existing} images — skipping")
            continue

        n_needed = n_per_class - existing
        print(f"  Generating {n_needed} images for class '{cls}'...")

        for i in tqdm(range(n_needed), desc=f"  {cls}"):
            # Rotate through the prompts for diversity
            prompt = prompts[i % len(prompts)]

            image = pipe(
                prompt          = prompt,
                negative_prompt = NEGATIVE_PROMPT,
                num_inference_steps = steps,
                guidance_scale  = 7.5,
                generator       = generator,
                width           = 256,
                height          = 256,
            ).images[0]

            # Convert to grayscale (IR is single-channel)
            image = image.convert("L")
            image.save(cls_dir / f"{existing + i:05d}.png")

    print(f"\n[Generate] Done.")
    _print_dataset_summary(output_dir)


def _print_dataset_summary(root: str) -> None:
    print("\n── Dataset summary ──────────────────────────")
    total = 0
    for cls_dir in sorted(Path(root).iterdir()):
        if cls_dir.is_dir():
            n = len(list(cls_dir.glob("*.png")))
            print(f"  {cls_dir.name:<15} {n:>5} images")
            total += n
    print(f"  {'TOTAL':<15} {total:>5} images")
    print("─────────────────────────────────────────────")


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic IR images with Stable Diffusion"
    )
    parser.add_argument(
        "--output", type=str, default="data/synthetic/",
        help="Output directory (default: data/synthetic/)"
    )
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Classes to generate (default: aircraft vessel vehicle)"
    )
    parser.add_argument(
        "--n", type=int, default=200,
        help="Images per class (default: 200)"
    )
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Diffusion steps — more = better quality (default: 30)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    generate_images(
        output_dir  = args.output,
        classes     = args.classes,
        n_per_class = args.n,
        steps       = args.steps,
        seed        = args.seed,
    )
