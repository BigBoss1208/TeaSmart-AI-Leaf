"""Verify the selected MobileNetV2 or explicitly evaluate its frozen Test split."""

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path, PurePosixPath, PureWindowsPath

import torch
from torch.utils.data import DataLoader

from config import CLASS_NAMES, CLASS_TO_IDX, DATASET_DIR, OUTPUTS_DIR, TrainingConfig
from dataset import LeafDataset, build_transforms
from model import build_model

RUN_NAME = "mobilenetv2-30epochs-20261119-215540"
RUN_DIR = OUTPUTS_DIR / RUN_NAME
EVALUATION_DIR = OUTPUTS_DIR / f"{RUN_NAME}-test-evaluation"
BEST_EPOCH = 20
BEST_SHA256 = "6c3ed706fee56e481d8820c1b74a207ab98b53c4615b5b55a8b381f593d5d10e"
TEST_SUPPORT = dict(zip(CLASS_NAMES, (31, 51, 103, 63, 52, 133, 97), strict=True))
LIMITATIONS = (
    "Independence by physical leaf, plant, garden, and capture session is not fully verified.",
    "The clean dataset retains 165 lower-similarity cross-split candidate pairs "
    "(46 touching Test) with insufficient evidence to call them the same leaf.",
)


def load_selected_model():
    checkpoint_path = RUN_DIR / "best_mobilenetv2.pth"
    run_spec = json.loads((RUN_DIR / "config.json").read_text(encoding="utf-8"))
    mapping = json.loads((RUN_DIR / "class_to_idx.json").read_text(encoding="utf-8"))
    if any(
        type(index) is not int for index in mapping.values()
    ) or mapping != CLASS_TO_IDX or run_spec["class_to_idx"] != mapping:
        raise ValueError("Class mapping differs from the fixed indices 0 through 6.")
    if run_spec["model"] != "mobilenet_v2" or PureWindowsPath(run_spec["dataset_root"]).name != DATASET_DIR.name:
        raise ValueError("Training config does not refer to the selected model and dataset.")

    with checkpoint_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != BEST_SHA256:
            raise ValueError("SHA-256 does not match the Validation-selected epoch 20 checkpoint.")
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    if (
        checkpoint["epoch"] != BEST_EPOCH or checkpoint["best_epoch"] != BEST_EPOCH
        or checkpoint["format"] != run_spec["checkpoint_format"]
        or checkpoint["class_to_idx"] != mapping
        or checkpoint["train_valid_sha256"] != run_spec["train_valid_sha256"]
    ):
        raise ValueError("Selected checkpoint metadata does not match the training run.")

    validation = checkpoint["history"][-1]
    if validation["epoch"] != BEST_EPOCH or validation["valid_macro_f1"] != checkpoint["best_f1"]:
        raise ValueError("Best checkpoint and Validation history disagree.")
    config = TrainingConfig(**run_spec["config"])
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval(), config, run_spec, {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": digest,
        "epoch": BEST_EPOCH,
        "validation_accuracy": validation["valid_accuracy"],
        "validation_macro_f1": validation["valid_macro_f1"],
    }


def collect_test_samples(root: Path) -> tuple[list[tuple[str, Path, int]], str]:
    records = {}
    support = Counter()
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"relative_path", "split", "label", "sha256"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Test manifest is missing required fields.")
        for row in reader:
            if row["split"] != "test":
                continue
            relative, label = row["relative_path"], row["label"]
            sha256 = row["sha256"]
            parts = PurePosixPath(relative).parts
            if (
                label not in CLASS_TO_IDX or len(parts) != 3 or parts[:2] != ("test", label)
                or any(c in relative for c in ("\\", ":")) or not parts[2].endswith(".jpg")
                or "/".join(parts) != relative or relative in records
                or not re.fullmatch(r"[0-9a-f]{64}", sha256)
            ):
                raise ValueError(f"Invalid or duplicate Test manifest record: {relative!r}")
            records[relative] = (root.joinpath(*parts), CLASS_TO_IDX[label], sha256)
            support[label] += 1
    if support != TEST_SUPPORT:
        raise ValueError(f"Unexpected Test support: {dict(support)}")
    test_dir = root / "test"
    folders = {entry.name for entry in test_dir.iterdir() if entry.is_dir()}
    if folders != set(CLASS_NAMES):
        raise ValueError("Test directory does not contain the seven trained class names.")
    disk = {
        entry.relative_to(root).as_posix()
        for folder in folders for entry in (test_dir / folder).iterdir()
        if entry.is_file() and entry.suffix.lower() == ".jpg"
    }
    if disk != records.keys():
        raise ValueError("Test manifest paths and on-disk images differ.")

    samples = []
    fingerprint = hashlib.sha256()
    for relative, (path, target, expected_hash) in sorted(records.items()):
        with path.open("rb") as image:
            if hashlib.file_digest(image, "sha256").hexdigest() != expected_hash:
                raise ValueError(f"Test image SHA-256 differs from manifest: {relative}")
        fingerprint.update(json.dumps([relative, expected_hash]).encode("utf-8"))
        samples.append((relative, path, target))
    return samples, fingerprint.hexdigest()


def infer(model, config: TrainingConfig, samples, device: torch.device) -> list[dict]:
    dataset = LeafDataset(
        [(path, target) for _, path, target in samples],
        build_transforms(config, training=False),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    rows = []
    with torch.inference_mode():
        for images, targets in loader:
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=config.amp):
                logits = model(images.to(device))
            if logits.shape != (len(targets), len(CLASS_NAMES)) or not bool(logits.isfinite().all()):
                raise RuntimeError("Unexpected shape or non-finite Test logits.")
            probabilities, predictions = logits.float().softmax(dim=1).max(dim=1)
            offset = len(rows)
            for target, prediction, probability in zip(
                targets.tolist(), predictions.tolist(), probabilities.tolist(), strict=True
            ):
                relative, _, expected_target = samples[offset]
                if target != expected_target:
                    raise RuntimeError("Test sample order or labels changed during inference.")
                rows.append({
                    "relative_path": relative,
                    "true_index": target,
                    "true_label": CLASS_NAMES[target],
                    "predicted_index": prediction,
                    "predicted_label": CLASS_NAMES[prediction],
                    "predicted_probability": probability,
                    "correct": target == prediction,
                })
                offset += 1
    if len(rows) != len(samples):
        raise RuntimeError("Test inference did not produce exactly one prediction per sample.")
    return rows


def summarize_predictions(rows: list[dict]) -> tuple[dict, list[dict], list[list[int]]]:
    count = len(CLASS_NAMES)
    matrix = [[0 for _ in range(count)] for _ in range(count)]
    for row in rows:
        matrix[row["true_index"]][row["predicted_index"]] += 1

    by_class = []
    for index, label in enumerate(CLASS_NAMES):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        by_class.append({"class_index": index, "label": label, "support": support,
                         "precision": precision, "recall": recall, "f1": f1})

    total = len(rows)
    if not total:
        raise ValueError("No predictions to summarize.")
    aggregate = {
        "samples": total,
        "accuracy": sum(matrix[i][i] for i in range(count)) / total,
        "macro_precision": sum(row["precision"] for row in by_class) / count,
        "macro_recall": sum(row["recall"] for row in by_class) / count,
        "macro_f1": sum(row["f1"] for row in by_class) / count,
        "weighted_f1": sum(row["f1"] * row["support"] for row in by_class) / total,
        "macro_classes": count,
        "zero_division": 0,
        "confusion_matrix_orientation": "rows=true_labels, columns=predicted_labels",
        "limitations": LIMITATIONS,
    }
    return aggregate, by_class, matrix


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_once(model, config, run_spec, selected: dict, device: torch.device) -> None:
    samples, fingerprint = collect_test_samples(DATASET_DIR)
    # Reserve a single output for this frozen model before the first real prediction.
    EVALUATION_DIR.mkdir(exist_ok=False)
    write_json(EVALUATION_DIR / "evaluation_config.json", {
        **selected,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(DATASET_DIR),
        "test_manifest_sha256": fingerprint,
        "expected_support": TEST_SUPPORT,
        "class_to_idx": CLASS_TO_IDX,
        "training_config": run_spec["config"],
        "preprocessing": {
            "color": "RGB", "resize": [config.image_size, config.image_size],
            "interpolation": "bilinear", "antialias": True, "crop": False,
            "augmentation": False, "mean": config.imagenet_mean, "std": config.imagenet_std,
        },
        "batch_size": config.batch_size, "num_workers": 0, "shuffle": False,
        "inference_dtype": "CUDA autocast float16" if config.amp else "float32",
        "environment": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": version("torchvision"), "pillow": version("Pillow"),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
            "cudnn": torch.backends.cudnn.version(),
        },
        "limitations": LIMITATIONS,
    })
    predictions = infer(model, config, samples, device)
    summary, by_class, matrix = summarize_predictions(predictions)
    if {row["label"]: row["support"] for row in by_class} != TEST_SUPPORT:
        raise RuntimeError("Predicted sample support differs from the audited Test split.")
    write_rows(EVALUATION_DIR / "predictions.csv", predictions)
    write_rows(EVALUATION_DIR / "per_class_metrics.csv", by_class)
    with (EVALUATION_DIR / "confusion_matrix.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_label / predicted_label", *CLASS_NAMES])
        writer.writerows([label, *row] for label, row in zip(CLASS_NAMES, matrix, strict=True))
    write_json(EVALUATION_DIR / "metrics.json", summary)
    write_json(EVALUATION_DIR / "completed.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples": summary["samples"],
    })
    print(f"Test evaluation saved to {EVALUATION_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "test"), help="Only 'test' reads the Test split.")
    args = parser.parse_args()
    try:
        if args.mode == "test" and EVALUATION_DIR.exists():
            raise FileExistsError(f"Test output already exists: {EVALUATION_DIR}")
        model, config, run_spec, selected = load_selected_model()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in AI-Leaf.venv.")
        device = torch.device("cuda:0")
        model.to(device)
        if args.mode == "verify":
            with torch.inference_mode(), torch.amp.autocast(
                "cuda", dtype=torch.float16, enabled=config.amp
            ):
                dummy = torch.zeros((1, 3, config.image_size, config.image_size), device=device)
                logits = model(dummy)
            torch.cuda.synchronize(device)
            if logits.shape != (1, len(CLASS_NAMES)) or not bool(logits.isfinite().all()):
                raise RuntimeError("Checkpoint verification forward pass failed.")
            print(json.dumps({**selected, "strict_state_dict": True, "device": str(logits.device),
                              "synthetic_output_shape": list(logits.shape), "test_images_read": 0},
                             ensure_ascii=True))
        else:
            evaluate_once(model, config, run_spec, selected, device)
        return 0
    except torch.cuda.OutOfMemoryError:
        print("CUDA ran out of memory. No completed Test evaluation was published.", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite an existing evaluation: {exc}", file=sys.stderr)
        return 3
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
