"""Render the completed run's Train/Validation history using matplotlib."""

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path


def read_history(path: Path) -> list[dict]:
    fields = ("train_loss", "valid_loss", "train_accuracy", "valid_accuracy", "valid_macro_f1")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [{"epoch": int(row["epoch"]), **{key: float(row[key]) for key in fields}}
                for row in csv.DictReader(stream)]
    if not rows or [row["epoch"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Expected contiguous epochs beginning at 1.")
    for row in rows:
        for key in fields:
            value = row[key]
            if not math.isfinite(value) or value < 0 or ("loss" not in key and value > 1):
                raise ValueError(f"Invalid metric {key} at epoch {row['epoch']}.")
    return rows


def render(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    epochs = [row["epoch"] for row in rows]
    best = max(rows, key=lambda row: row["valid_macro_f1"])
    train_color = "#176b54"
    valid_color = "#285a94"
    accent = "#ac552b"

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9), sharex=True)
    fig.subplots_adjust(left=0.119, right=0.955, bottom=0.09, top=0.83, hspace=0.45)
    fig.suptitle("MobileNetV2 · Lịch sử huấn luyện TeaSmart", x=0.119, y=0.966,
                 ha="left", fontsize=17, fontweight="bold", color="#1b2d31")
    fig.text(0.119, 0.907,
             f"{len(epochs)} epoch  |  Chọn epoch {best['epoch']} theo Validation Macro-F1"
             f" ({best['valid_macro_f1']:.4f})",
             fontsize=10, color="#3e4d57")

    plots = (
        ("Cross-entropy loss", (("train_loss", "Train", train_color), ("valid_loss", "Validation", valid_color))),
        ("Accuracy", (("train_accuracy", "Train", train_color), ("valid_accuracy", "Validation", valid_color))),
        ("Macro-F1", (("valid_macro_f1", "Validation", valid_color),)),
    )
    for axis, (title, series) in zip(axes, plots, strict=True):
        axis.set_ylabel(title, labelpad=9)
        axis.grid(axis="y", color="#dce5e8", linewidth=0.75)
        axis.set_axisbelow(True)
        axis.spines[["right", "top"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#a9b9bf")
        for field, label, color in series:
            axis.plot(epochs, [row[field] for row in rows], label=label, color=color, linewidth=2.0)
        axis.axvline(best["epoch"], color=accent, linestyle="--", linewidth=1.35,
                    alpha=0.85, label="Epoch được chọn" if axis is axes[0] else None)
        axis.scatter(best["epoch"], best[series[-1][0]], color=accent, zorder=4, s=26)
        axis.legend(loc="best", frameon=False, fontsize=9, ncol=3 if axis is axes[0] else 2)
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axes[-1].set_xlabel("Epoch")
    ticks = sorted({1, epochs[-1], best["epoch"]} | set(range(5, epochs[-1], 5)))
    axes[-1].set_xticks(ticks)
    axes[-1].set_xlim(0.6, epochs[-1] + 0.4)
    axes[0].set_ylim(bottom=0)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing training_history.csv")
    args = parser.parse_args()
    run = args.run_dir.resolve(strict=True)
    rows = read_history(run / "training_history.csv")
    output = run.with_name(f"{run.name}-training-plots")
    output.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="tea-leaf-mpl-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        render(rows, output / "training_curves.png")
    print(output / "training_curves.png")


if __name__ == "__main__":
    main()
