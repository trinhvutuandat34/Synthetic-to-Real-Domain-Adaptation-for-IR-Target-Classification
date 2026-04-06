# utils/config.py  (updated for 6-class expansion)
# ─────────────────────────────────────────────────────────────
# CHANGES from original:
#   - num_classes: 3 → 6
#   - class_names: 3-class list → 6-class military list
#   - synth_dir: updated to reflect 6-class folder layout
#   - dann_lambda_max added (was missing from original)
# ─────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:

    # ── Project ───────────────────────────────────────────────
    project_name: str = "cadet-atr"
    run_name:     str = "baseline"
    seed:         int = 42

    # ── Classes ───────────────────────────────────────────────
    # 6-class layout matching:
    #   Classes 1-3: airborne targets (뚜언닷 expansion)
    #   Classes 4-6: surface/ground targets (original paper scope)
    #
    # Maps directly to subdirectory names under synth_dir / real_dir:
    #   data/synthetic/fixed_wing/
    #   data/synthetic/rotary_wing/
    #   data/synthetic/uav/
    #   data/synthetic/vessel/
    #   data/synthetic/vehicle_ground/
    #   data/synthetic/vehicle_apc/
    class_names: List[str] = field(
        default_factory=lambda: [
            "fixed_wing",      # F-16 analogue — Air Force IRST target
            "rotary_wing",     # helicopter — Naval / Air Force
            "uav",             # unmanned aerial vehicle — multi-domain
            "vessel",          # surface ship — Naval CMS / EOTS
            "vehicle_ground",  # tank / IFV — Army joint ops
            "vehicle_apc",     # wheeled APC — Army joint ops
        ]
    )

    # ── Paths ─────────────────────────────────────────────────
    synth_dir:      str = "data/synthetic/"
    real_dir:       str = "data/real/"
    checkpoint_dir: str = "checkpoints/"
    results_dir:    str = "results/"

    # ── Model ─────────────────────────────────────────────────
    model_name:  str  = "convnext_tiny"   # best single model in the paper
    pretrained:  bool = True
    input_size:  int  = 224

    # ── Training ──────────────────────────────────────────────
    epochs:              int   = 50
    batch_size:          int   = 32       # safe for Colab T4 (paper used 128)
    learning_rate:       float = 1e-4
    weight_decay:        float = 1e-2
    early_stop_patience: int   = 7

    # ── Data splits ───────────────────────────────────────────
    train_frac: float = 0.70
    val_frac:   float = 0.15
    # test_frac = 0.15 (implicit)

    # ── Augmentation ──────────────────────────────────────────
    aug_prob:       float = 0.50
    aug_noise_std:  float = 0.05
    aug_brightness: tuple = (0.7, 1.3)
    aug_contrast:   tuple = (0.7, 1.3)

    # ── Fine-tuning & DANN ────────────────────────────────────
    finetune_epochs:   int   = 20
    finetune_lr:       float = 1e-4
    finetune_lr_full:  float = 5e-6
    finetune_freeze:   bool  = True
    dann_epochs:       int   = 20
    dann_lambda_max:   float = 1.0

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


cfg = Config()


# ─────────────────────────────────────────────────────────────
# generate_synthetic.py  — updated IR_PROMPTS for 6 classes
# ─────────────────────────────────────────────────────────────
#
# Replace the IR_PROMPTS dict in generate_synthetic.py with this.
# Prompts are tuned to produce realistic grayscale thermal signatures
# for each class, minimising "photographic" artifacts.

IR_PROMPTS_6CLASS = {
    "fixed_wing": [
        "infrared thermal image of fighter jet in flight, "
        "EOTS sensor view, grayscale thermal, hot engine nozzle, dark sky",

        "thermal IR image of military F-16 aircraft, airborne, "
        "grayscale, glowing dual engine exhaust, dark background",

        "IR camera image of supersonic jet, engine heat signature, "
        "grayscale thermal, exhaust plume, overhead perspective",
    ],
    "rotary_wing": [
        "infrared thermal image of military helicopter in flight, "
        "IR sensor, grayscale thermal, rotor disc heat, engine exhaust",

        "thermal camera image of attack helicopter, EOTS view, "
        "grayscale, engine exhaust plume, rotor wash, dark sky",

        "IR image of naval helicopter, thermal signature, "
        "grayscale, spinning rotor heat, turbine exhaust",
    ],
    "uav": [
        "infrared thermal image of military unmanned aerial vehicle, "
        "small UAV drone, IR sensor, grayscale thermal, engine heat, dark sky",

        "thermal IR image of surveillance drone, MALE UAV, "
        "grayscale, wing heat, propulsion exhaust, dark background",

        "IR camera image of tactical UAV, small aircraft thermal signature, "
        "grayscale, hot engine, dark background, overhead view",
    ],
    "vessel": [
        "infrared thermal image of naval warship at sea, "
        "IR sensor, grayscale thermal, ship hull heat, engine exhaust",

        "thermal IR image of fast patrol boat, military vessel, "
        "IR camera, grayscale, engine plume, dark ocean background",

        "EOTS infrared image of destroyer, warship thermal signature, "
        "grayscale, smokestack heat, dark water",
    ],
    "vehicle_ground": [
        "infrared thermal image of main battle tank, ground vehicle, "
        "IR sensor, grayscale, engine heat signature, dark terrain",

        "thermal camera image of armoured fighting vehicle, military, "
        "IR grayscale, hot engine compartment, dark ground",

        "infrared image of military tank moving, thermal signature, "
        "IR camera, grayscale, engine and exhaust heat",
    ],
    "vehicle_apc": [
        "infrared thermal image of wheeled armoured personnel carrier, "
        "APC, IR sensor, grayscale, engine heat, dark road background",

        "thermal camera image of military APC BTR, wheeled vehicle, "
        "IR grayscale, engine compartment heat, dark terrain",

        "infrared image of military wheeled APC, troop carrier, "
        "thermal signature, grayscale, hot engine",
    ],
}

NEGATIVE_PROMPT = (
    "color, rgb, photograph, realistic photo, hyperrealistic, "
    "text, watermark, logo, cartoon, painting, blue sky, grass, trees"
)
