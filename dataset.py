"""Read the audited Train and Valid records, leaving Test outside the loaders."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import CLASS_TO_IDX, EXPECTED_COUNTS, TrainingConfig


def build_transforms(config: TrainingConfig, *, training: bool) -> transforms.Compose:
    mean_fill = tuple(round(channel * 255) for channel in config.imagenet_mean)
    operations = []
    if training:
        operations += [
            transforms.RandomHorizontalFlip(config.train_flip_probability),
            # A small rotation can expand the canvas so the leaf edges remain visible.
            transforms.RandomRotation(
                config.train_rotation_degrees,
                interpolation=transforms.InterpolationMode.BILINEAR,
                expand=True,
                fill=mean_fill,
            ),
            transforms.ColorJitter(
                brightness=config.train_brightness_jitter,
                contrast=config.train_contrast_jitter,
            ),
        ]
    operations += [
        transforms.Resize(
            (config.image_size, config.image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(config.imagenet_mean, config.imagenet_std),
    ]
    return transforms.Compose(operations)


class LeafDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, target


def read_audited_splits(root: Path) -> tuple[dict[str, list[tuple[Path, int]]], str]:
    selected: dict[str, dict[str, tuple[Path, str, str]]] = {
        split: {} for split in EXPECTED_COUNTS
    }
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"relative_path", "split", "label", "sha256"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Dataset manifest is missing required columns.")
        for row in reader:
            split = row["split"]
            if split == "test":
                continue
            if split not in selected:
                raise ValueError(f"Unknown split in the manifest: {split!r}")
            relative_path, label = row["relative_path"], row["label"]
            parts = PurePosixPath(relative_path).parts
            if (
                label not in CLASS_TO_IDX or len(parts) != 3 or parts[:2] != (split, label)
                or "\\" in relative_path or ":" in relative_path
                or parts[2] == ".." or not parts[2].endswith(".jpg")
                or "/".join(parts) != relative_path or relative_path in selected[split]
            ):
                raise ValueError(f"Invalid or duplicate manifest path: {relative_path!r}")
            selected[split][relative_path] = (root.joinpath(*parts), label, row["sha256"])

    dataset_hash = hashlib.sha256()
    samples: dict[str, list[tuple[Path, int]]] = {}
    for split, entries in selected.items():
        folders = {path.name for path in (root / split).iterdir() if path.is_dir()}
        if folders != set(CLASS_TO_IDX):
            raise ValueError(f"Class folders for {split} differ from the audited seven labels.")
        files = {
            path.relative_to(root).as_posix()
            for label in folders
            for path in (root / split / label).iterdir()
            if path.is_file() and path.suffix.lower() == ".jpg"
        }
        if files != entries.keys():
            raise ValueError(f"Manifest and disk differ for {split}.")
        counts = Counter(label for _, label, _ in entries.values())
        if counts != EXPECTED_COUNTS[split]:
            raise ValueError(f"Unexpected class counts for {split}: {dict(counts)}")

        samples[split] = []
        for relative_path, (image_path, label, expected_hash) in sorted(entries.items()):
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"Image SHA-256 differs from manifest: {relative_path}")
            dataset_hash.update(json.dumps([relative_path, expected_hash]).encode("utf-8"))
            samples[split].append((image_path, CLASS_TO_IDX[label]))
    return samples, dataset_hash.hexdigest()


def make_loaders(
    samples: dict[str, list[tuple[Path, int]]], config: TrainingConfig
) -> tuple[DataLoader, DataLoader, torch.Generator]:
    generator = torch.Generator().manual_seed(config.seed)
    train = DataLoader(
        LeafDataset(samples["train"], build_transforms(config, training=True)),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
        pin_memory=True,
    )
    valid = DataLoader(
        LeafDataset(samples["valid"], build_transforms(config, training=False)),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train, valid, generator
