"""Train only on the audited Train/Valid splits, or check a batch on CUDA."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import torch
from filelock import FileLock, Timeout
from torch import nn

from config import CLASS_TO_IDX, DATASET_DIR, OUTPUTS_DIR, WEIGHTS_NAME, TrainingConfig
from dataset import make_loaders, read_audited_splits
from metrics import EpochMetrics
from model import build_model, build_optimizer, configure_stage

CHECKPOINT_FORMAT = 1


def parse_args() -> argparse.Namespace:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Inspect one batch per split. No epoch or run files.")
    parser.add_argument("--run-name", help="Name under outputs/. Required for a training run.")
    parser.add_argument("--epochs", type=int, default=defaults.epochs, help="Total epochs, capped at 30.")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--resume", action="store_true", help="Continue the named run from its latest completed epoch.")
    parser.add_argument("--no-amp", action="store_true", help="Use float32 instead of CUDA AMP.")
    args = parser.parse_args()
    if not 1 <= args.epochs <= defaults.epochs or args.batch_size < 1:
        parser.error("Use 1–30 epochs and a positive batch size.")
    if args.check and (args.run_name or args.resume):
        parser.error("--check uses no run name or checkpoint.")
    if not args.check and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", args.run_name or ""):
        parser.error("Specify --run-name with 1–64 letters, numbers, hyphens or underscores.")
    return args


def prepare_cuda(seed: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Verify the AI-Leaf.venv PyTorch CUDA installation.")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda:0")


def describe_run(config: TrainingConfig, dataset_hash: str, device: torch.device) -> dict:
    return {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "model": "mobilenet_v2",
        "initial_weights": WEIGHTS_NAME,
        "dataset_root": str(DATASET_DIR),
        "train_valid_sha256": dataset_hash,
        "class_to_idx": CLASS_TO_IDX,
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "torch": torch.__version__,
            "torchvision": version("torchvision"),
            "pillow": version("Pillow"),
            "filelock": version("filelock"),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device),
            "os": platform.platform(),
        },
    }


def check_batches(loaders: tuple, config: TrainingConfig, device: torch.device) -> None:
    free, total = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = build_model(pretrained=True).to(device).eval()
    with torch.inference_mode():
        for name, loader in zip(("train", "valid"), loaders, strict=True):
            images, labels = next(iter(loader))
            expected_input = (labels.numel(), 3, config.image_size, config.image_size)
            if tuple(images.shape) != expected_input or labels.min() < 0 or labels.max() >= len(CLASS_TO_IDX):
                raise ValueError(f"Unexpected {name} batch shape or class indices.")
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=config.amp):
                logits = model(images.to(device, non_blocking=True))
            torch.cuda.synchronize(device)
            expected_output = (labels.numel(), len(CLASS_TO_IDX))
            if tuple(logits.shape) != expected_output or not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"Unexpected {name} output shape or non-finite logits.")
            print(f"{name}: input={tuple(images.shape)} output={tuple(logits.shape)} "
                  f"device={logits.device} dtype={logits.dtype} finite=True")
    mib = 1024 ** 2
    print(f"VRAM before model: {free / mib:.1f}/{total / mib:.1f} MiB free")
    print(f"CUDA peak: {torch.cuda.max_memory_allocated(device) / mib:.1f} MiB allocated, "
          f"{torch.cuda.max_memory_reserved(device) / mib:.1f} MiB reserved")
    print("Batch and forward checks passed.")


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    config: TrainingConfig,
    optimizer: torch.optim.AdamW | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    if not training:
        model.eval()
    metrics = EpochMetrics(len(CLASS_TO_IDX), device)
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=config.amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            metrics.update(logits, targets, loss)
    summary = metrics.summarize()
    if not math.isfinite(summary["loss"]):
        raise RuntimeError("Non-finite epoch loss. The latest completed checkpoint is preserved.")
    return summary


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_atomic(path: Path, write_content, *, replace_existing: bool = False) -> None:
    # Publish a complete file on the same volume, without overwriting any checkpoint.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            write_content(stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        if replace_existing:
            os.replace(temporary, path)
        elif os.name == "nt":
            # On Windows, rename fails atomically when the destination exists.
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_history(path: Path, rows: list[dict]) -> None:
    def write_content(stream) -> None:
        csv_stream = io.StringIO(newline="")
        writer = csv.DictWriter(csv_stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.write(csv_stream.getvalue().encode("utf-8"))

    write_atomic(path, write_content, replace_existing=True)


def finalize_run(run: Path, checkpoints: Path, state: dict) -> None:
    for filename, epoch in (
        ("best_mobilenetv2.pth", state["best_epoch"]),
        ("last_checkpoint.pth", state["epoch"]),
    ):
        path = run / filename
        source = checkpoints / f"epoch-{epoch:03d}.pth"
        with source.open("rb") as checkpoint:
            if path.exists():
                with path.open("rb") as existing:
                    matches = (hashlib.file_digest(existing, "sha256").digest()
                               == hashlib.file_digest(checkpoint, "sha256").digest())
                if not matches:
                    raise FileExistsError(f"Existing checkpoint will not be overwritten: {path}")
            else:
                write_atomic(path, lambda stream: shutil.copyfileobj(checkpoint, stream))


def train(
    args: argparse.Namespace,
    config: TrainingConfig,
    device: torch.device,
    loaders: tuple,
    generator: torch.Generator,
    spec: dict,
) -> None:
    run = OUTPUTS_DIR / args.run_name
    if args.resume:
        if not run.is_dir():
            raise FileNotFoundError(f"Run directory for resume not found: {run}")
    else:
        run.mkdir(parents=True, exist_ok=False)

    with FileLock(run / ".training.lock", timeout=0):
        config_path = run / "config.json"
        mapping_path = run / "class_to_idx.json"
        checkpoints = run / "checkpoints"
        if args.resume:
            if config_path.read_bytes() != json_bytes(spec):
                raise ValueError("Resume requires the original arguments, Python environment and audited Train/Valid data.")
            if mapping_path.exists() and mapping_path.read_bytes() != json_bytes(CLASS_TO_IDX):
                raise ValueError("Saved class mapping differs from the audited class indices.")
        else:
            write_atomic(config_path, lambda stream: stream.write(json_bytes(spec)))
        if not mapping_path.exists():
            write_atomic(mapping_path, lambda stream: stream.write(json_bytes(CLASS_TO_IDX)))
        checkpoints.mkdir(exist_ok=True)

        completed = sorted(checkpoints.glob("epoch-*.pth"))
        state = None
        if completed:
            state = torch.load(completed[-1], map_location="cpu", weights_only=True)
            if (
                state["format"] != CHECKPOINT_FORMAT
                or state["class_to_idx"] != CLASS_TO_IDX
                or state["train_valid_sha256"] != spec["train_valid_sha256"]
                or completed[-1].name != f"epoch-{state['epoch']:03d}.pth"
                or state["epoch"] != len(state["history"])
            ):
                raise ValueError("Latest checkpoint metadata is inconsistent.")
            write_history(run / "training_history.csv", state["history"])
            if state["finish_reason"]:
                finalize_run(run, checkpoints, state)
                print(f"Run is already complete: {state['finish_reason']}.")
                return

        model = build_model(pretrained=state is None).to(device)
        optimizer = build_optimizer(model, config)
        scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
        criterion = nn.CrossEntropyLoss()
        if state is None:
            history = []
            best_epoch, best_f1 = 0, -1.0
            finetune_best, stalled_epochs = -1.0, 0
            start_epoch = 1
        else:
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scaler.load_state_dict(state["scaler_state_dict"])
            random.setstate(state["python_rng_state"])
            torch.set_rng_state(state["torch_rng_state"])
            torch.cuda.set_rng_state(state["cuda_rng_state"], device)
            generator.set_state(state["train_generator_state"])
            history = state["history"]
            best_epoch, best_f1 = state["best_epoch"], state["best_f1"]
            finetune_best, stalled_epochs = state["finetune_best"], state["stalled_epochs"]
            start_epoch = state["epoch"] + 1
            del state

        train_loader, valid_loader = loaders
        for epoch in range(start_epoch, config.epochs + 1):
            stage = configure_stage(model, optimizer, epoch, config)
            train_result = run_epoch(model, train_loader, criterion, device, config, optimizer, scaler)
            valid_result = run_epoch(model, valid_loader, criterion, device, config)
            score = valid_result["macro_f1"]
            if score > best_f1:
                best_epoch, best_f1 = epoch, score
            if stage == "finetune":
                if score > finetune_best:
                    finetune_best, stalled_epochs = score, 0
                else:
                    stalled_epochs += 1

            row = {
                "epoch": epoch,
                "stage": stage,
                "train_loss": train_result["loss"],
                "train_accuracy": train_result["accuracy"],
                "valid_loss": valid_result["loss"],
                "valid_accuracy": valid_result["accuracy"],
                "valid_macro_f1": score,
                "classifier_lr": optimizer.param_groups[0]["lr"],
                "backbone_lr": optimizer.param_groups[1]["lr"],
            }
            history.append(row)
            finish_reason = ""
            if epoch == config.epochs:
                finish_reason = "epoch_limit"
            elif config.patience and stalled_epochs >= config.patience:
                finish_reason = "early_stopping"
            state = {
                "format": CHECKPOINT_FORMAT,
                "class_to_idx": CLASS_TO_IDX,
                "train_valid_sha256": spec["train_valid_sha256"],
                "epoch": epoch,
                "finish_reason": finish_reason,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "python_rng_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device),
                "train_generator_state": generator.get_state(),
                "history": history,
                "best_epoch": best_epoch,
                "best_f1": best_f1,
                "finetune_best": finetune_best,
                "stalled_epochs": stalled_epochs,
            }
            # The immutable checkpoint is the source of truth if the process stops before writing the CSV.
            path = checkpoints / f"epoch-{epoch:03d}.pth"
            write_atomic(path, lambda stream: torch.save(state, stream))
            write_history(run / "training_history.csv", history)
            print(f"epoch {epoch:02d}/{config.epochs} {stage}  "
                  f"train loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f}  "
                  f"valid loss={row['valid_loss']:.4f} acc={row['valid_accuracy']:.4f} "
                  f"macro-F1={score:.4f}", flush=True)
            if finish_reason:
                finalize_run(run, checkpoints, state)
                print(f"Best validation macro-F1 {best_f1:.4f} at epoch {best_epoch}; {finish_reason}.")
                return


def main() -> int:
    args = parse_args()
    config = replace(TrainingConfig(), epochs=args.epochs, batch_size=args.batch_size, amp=not args.no_amp)
    try:
        device = prepare_cuda(config.seed)
        samples, dataset_hash = read_audited_splits(DATASET_DIR)
        train_loader, valid_loader, generator = make_loaders(samples, config)
        spec = describe_run(config, dataset_hash, device)
        print(f"Dataset {DATASET_DIR}: train={len(train_loader.dataset)}, valid={len(valid_loader.dataset)}, "
              f"classes={len(CLASS_TO_IDX)}", flush=True)
        print(json.dumps(spec["environment"], ensure_ascii=True), flush=True)
        loaders = (train_loader, valid_loader)
        if args.check:
            check_batches(loaders, config, device)
        else:
            train(args, config, device, loaders, generator, spec)
        return 0
    except torch.cuda.OutOfMemoryError:
        print(f"CUDA out of memory at batch size {config.batch_size}. Use a smaller --batch-size. "
              "To change batch size after an interrupted training run, start a new run name. "
              "Existing completed checkpoints are preserved.", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Existing output will not be overwritten. Use another --run-name or --resume. {exc}", file=sys.stderr)
        return 3
    except Timeout:
        print("This run is already in use by another process.", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("Stopped. Resume the same run to continue from the latest completed epoch.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
