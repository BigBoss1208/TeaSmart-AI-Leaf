"""Defaults for the audited TeaSmart dataset and the MobileNetV2 experiment."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR.parent / "Dataset" / "tea-leaf-clean-v1"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
WEIGHTS_CACHE = PROJECT_DIR / ".torch-cache"
WEIGHTS_NAME = "IMAGENET1K_V2"

CLASS_NAMES = (
    "1. Tea algal leaf spot",
    "2. Brown Blight",
    "3. Gray Blight",
    "4. Helopeltis",
    "5. Red spider",
    "6. Green mirid bug",
    "7. Healthy leaf",
)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
EXPECTED_COUNTS = {
    "train": dict(zip(CLASS_NAMES, (296, 335, 721, 401, 357, 890, 647), strict=True)),
    "valid": dict(zip(CLASS_NAMES, (91, 109, 183, 130, 103, 247, 188), strict=True)),
}


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    freeze_epochs: int = 5
    batch_size: int = 8
    num_workers: int = 0
    seed: int = 42
    image_size: int = 224
    imagenet_mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, ...] = (0.229, 0.224, 0.225)
    train_flip_probability: float = 0.5
    train_rotation_degrees: float = 8.0
    train_brightness_jitter: float = 0.08
    train_contrast_jitter: float = 0.08
    finetune_blocks: int = 4
    warmup_classifier_lr: float = 1e-3
    finetune_classifier_lr: float = 3e-4
    finetune_backbone_lr: float = 3e-5
    weight_decay: float = 1e-4
    patience: int = 8  # Count only fine-tuning epochs. Set to 0 to disable.
    amp: bool = True
