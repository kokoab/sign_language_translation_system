#!/usr/bin/env python3
"""Render the measured figures used by the Capstone 1 revision checklist."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "charts"
CHARTS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 180,
})


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(CHARTS / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def grouped_accuracy() -> None:
    labels = ["Validation A", "Validation B", "Familiar-signer\ndiagnostic",
              "Equal phrase\nsegments", "Activity phrase\ncrops"]
    before = np.array([96.30, 89.06, 97.10, 54.44, 59.65])
    after = np.array([96.03, 89.16, 97.03, 66.41, 69.79])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    width = 0.36
    ax.bar(x - width / 2, before, width, label="Before adaptation", color="#7b8794")
    ax.bar(x + width / 2, after, width, label="After adaptation", color="#1167b1")
    for index, value in enumerate(before):
        ax.text(index - width / 2, value + 1.1, f"{value:.2f}", ha="center", fontsize=8)
    for index, value in enumerate(after):
        ax.text(index + width / 2, value + 1.1, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 106)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, ncols=2, loc="lower center")
    ax.set_title("Phrase/activity adaptation improves contextual sign crops")
    save(fig, "phrase-activity-adaptation")


def extractor_comparison() -> None:
    quality_labels = ["Output frames\n(median)", "Pre-trim\ndetection",
                      "Post-trim\ndetection"]
    apple = np.array([87.50, 42.65, 85.24])
    alternative = np.array([87.50, 38.54, 84.73])
    x = np.arange(len(quality_labels))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5))
    width = 0.36
    axes[0].bar(x - width / 2, apple, width, label="Apple Vision", color="#1167b1")
    axes[0].bar(x + width / 2, alternative, width, label="MediaPipe", color="#d97706")
    axes[0].set_xticks(x, quality_labels)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Coverage (%)")
    axes[0].set_title("Observed active-hand coverage\n(same 300 clips)")
    axes[0].legend(frameon=False, fontsize=8)
    for index, (apple_value, alternative_value) in enumerate(zip(apple, alternative)):
        axes[0].text(index - width / 2, apple_value + 1.3, f"{apple_value:.2f}",
                     ha="center", fontsize=7)
        axes[0].text(index + width / 2, alternative_value + 1.3,
                     f"{alternative_value:.2f}", ha="center", fontsize=7)

    times = [0.6780, 1.2302]
    axes[1].bar([0, 1], times, color=["#1167b1", "#d97706"])
    axes[1].set_xticks([0, 1], ["Apple Vision", "MediaPipe"])
    axes[1].set_ylabel("Median seconds per clip")
    axes[1].set_ylim(0, 1.4)
    axes[1].set_title("Extractor time")
    for index, value in enumerate(times):
        axes[1].text(index, value + 0.04, f"{value:.3f}", ha="center")

    top1 = [93.12, 89.95]
    axes[2].bar([0, 1], top1, color=["#1167b1", "#d97706"])
    axes[2].set_xticks([0, 1], ["Apple Vision", "MediaPipe"])
    axes[2].set_ylabel("Top-1 accuracy (%)")
    axes[2].set_ylim(80, 96)
    axes[2].set_title("Matched 378-clip classifier check")
    for index, value in enumerate(top1):
        axes[2].text(index, value + 0.35, f"{value:.2f}", ha="center")
    save(fig, "extractor-comparison")


def legacy_extractor_screening() -> None:
    labels = ["Apple Vision", "MediaPipe\n(optimized)", "RTMW-XL"]
    speed_ms = [5.0, 28.0, 447.6]
    frame_output = [80.6, 75.9, 100.0]
    colors = ["#1167b1", "#d97706", "#7b8794"]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))

    speed_bars = axes[0].barh(y, speed_ms, color=colors)
    axes[0].set_xscale("log")
    axes[0].set_xlim(3, 700)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Milliseconds per sampled frame (log scale)")
    axes[0].set_title("Extractor speed")
    for bar, value in zip(speed_bars, speed_ms):
        axes[0].text(value * 1.08, bar.get_y() + bar.get_height() / 2,
                     f"{value:.1f} ms", va="center", fontsize=8)

    output_bars = axes[1].barh(y, frame_output, color=colors)
    axes[1].set_xlim(0, 108)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Frames with emitted hand output (%)")
    axes[1].set_title("Observed frame output")
    for index, (bar, value) in enumerate(zip(output_bars, frame_output)):
        suffix = "*" if index == 2 else "%"
        label = f"{value:.1f}{suffix}"
        axes[1].text(value + 1.2, bar.get_y() + bar.get_height() / 2,
                     label, va="center", fontsize=8)

    fig.suptitle("Historical extractor screening on the same 500-video sample")
    fig.text(
        0.5,
        -0.01,
        "* RTMW-XL always emits both-hand pose estimates; 100% output is not verified hand detection.",
        ha="center",
        fontsize=8,
    )
    save(fig, "legacy-extractor-screening")


def modality_comparison() -> None:
    labels = ["Skeletal\nlandmarks", "RGB hand-crop\nfeatures", "Learned\nfusion"]
    top1 = [95.50, 80.69, 96.30]
    top5 = [98.68, 94.71, 99.21]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.bar(x - width / 2, top1, width, label="Top-1", color="#1167b1")
    ax.bar(x + width / 2, top5, width, label="Top-5", color="#7b8794")
    for index, value in enumerate(top1):
        ax.text(index - width / 2, value + 0.45, f"{value:.2f}", ha="center", fontsize=8)
    for index, value in enumerate(top5):
        ax.text(index + width / 2, value + 0.45, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_ylim(70, 102)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, ncols=2)
    ax.set_title("Current modality components on the same 378 clips")
    save(fig, "modality-comparison")


def architecture_comparison() -> None:
    labels = ["Graph-part\nreplacement", "Wider flat\nSqueezeformer",
              "Flat\nSqueezeformer", "Part-wise + global\nSqueezeformer"]
    top1 = [78.31, 95.24, 95.77, 96.83]
    params = [6.48, 14.34, 6.47, 6.79]
    latency = [11.42, 7.35, 4.90, 6.50]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    colors = ["#d97706", "#7b8794", "#4d8f73", "#1167b1"]
    bars = axes[0].bar(labels, top1, color=colors)
    axes[0].set_ylim(70, 100)
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].set_title("Accuracy on the same 378 clips")
    for bar, accuracy, parameter_count in zip(bars, top1, params):
        axes[0].text(bar.get_x() + bar.get_width() / 2, accuracy + 0.6,
                     f"{accuracy:.2f}%\n{parameter_count:.2f}M", ha="center", fontsize=8)
    speed_bars = axes[1].bar(labels, latency, color=colors)
    axes[1].set_ylim(0, 13)
    axes[1].set_ylabel("Median model latency (ms)")
    axes[1].set_title("Matched batch-1 CPU recognition benchmark")
    for bar, value in zip(speed_bars, latency):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.25,
                     f"{value:.2f} ms", ha="center", fontsize=8)
    fig.suptitle("Controlled v17 architecture accuracy and recognition speed")
    save(fig, "architecture-comparison")


def architecture_family_comparison() -> None:
    result_path = ROOT / "stage1_family_benchmark/result.json"
    results = json.loads(result_path.read_text(encoding="utf-8"))["results"]
    labels = [
        item["display_name"].replace("Part-wise + global ", "Part-wise + global\n")
        for item in results
    ]
    labels = [label.replace("ST-GCN", "ST-GCN\n(compact)") for label in labels]
    top1 = [item["validation"]["top1"] for item in results]
    latency = [item["latency"]["median_ms"] for item in results]
    params = [item["parameters"] / 1_000_000 for item in results]
    family_colors = {
        "transformer": "#4d8f73",
        "compact_transformer": "#4d8f73",
        "conv_transformer": "#4d8f73",
        "anatomical_token_transformer": "#4d8f73",
        "partwise_transformer": "#4d8f73",
        "squeezeformer": "#1167b1",
    }
    colors = [family_colors.get(item["family"], "#7b8794") for item in results]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 7.0), sharey=True)

    bars = axes[0].barh(y, top1, color=colors)
    axes[0].set_xlim(0, 100)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Validation top-1 (%)")
    axes[0].set_title("Recognition accuracy")
    for bar, value in zip(bars, top1):
        axes[0].text(value + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{value:.2f}%", va="center", fontsize=8)

    bars = axes[1].barh(y, latency, color=colors)
    axes[1].set_xlim(0, 7.2)
    axes[1].set_xlabel("Median latency (ms)")
    axes[1].set_title("Batch-1 CPU inference")
    for bar, value in zip(bars, latency):
        axes[1].text(value + 0.10, bar.get_y() + bar.get_height() / 2,
                     f"{value:.2f}", va="center", fontsize=8)

    bars = axes[2].barh(y, params, color=colors)
    axes[2].set_xlim(0, 7.6)
    axes[2].set_xlabel("Parameters (millions)")
    axes[2].set_title("Model size")
    for bar, value in zip(bars, params):
        axes[2].text(value + 0.10, bar.get_y() + bar.get_height() / 2,
                     f"{value:.2f}M", va="center", fontsize=8)

    fig.suptitle("Matched Stage 1 architecture-family comparison")
    save(fig, "stage1-family-comparison")


def stage1_training() -> None:
    history_path = (
        ROOT.parents[1]
        / "generated/kaggle_stage1_partwise_kokoab_pull_v1"
        / "stage1_v17_partwise_v2/history.json"
    )
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = np.array([item["epoch"] for item in history])
    selected_epoch = 100
    selected = history[selected_epoch - 1]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))

    axes[0].plot(epochs, [item["train_loss"] for item in history],
                 color="#1167b1", label="Training loss")
    axes[0].plot(epochs, [item["loss"] for item in history],
                 color="#d97706", label="Validation loss")
    axes[0].axvline(selected_epoch, color="#d62728", linestyle="--",
                    label=f"Selected epoch ({selected_epoch})")
    axes[0].scatter([selected_epoch], [selected["loss"]], color="#d62728", zorder=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and validation loss")
    axes[0].legend(frameon=False, fontsize=8)

    top1 = np.array([item["top1"] for item in history])
    top5 = np.array([item["top5"] for item in history])
    axes[1].fill_between(epochs, top1, color="#4d8f73", alpha=0.55,
                         label="Validation top-1")
    axes[1].plot(epochs, top5, color="#6f42c1", linewidth=2,
                 label="Validation top-5")
    axes[1].axvline(selected_epoch, color="#d62728", linestyle="--")
    axes[1].scatter([selected_epoch], [selected["top1"]], color="#d62728", zorder=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 102)
    axes[1].set_title("Validation accuracy")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    axes[1].text(
        0.04,
        0.34,
        f"Selected validation: {selected['top1']:.2f}%\n"
        "Frozen held-out test: 87.57%\n"
        f"Validation top-5: {selected['top5']:.2f}%",
        transform=axes[1].transAxes,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff4cc",
              "edgecolor": "#d97706", "alpha": 0.95},
    )

    axes[2].plot(epochs, [item["lr"] for item in history], color="#6f42c1")
    axes[2].axvline(selected_epoch, color="#d62728", linestyle="--")
    axes[2].scatter([selected_epoch], [selected["lr"]], color="#d62728",
                    zorder=3, label=f"Selected epoch ({selected_epoch})")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_title("Learning-rate schedule")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("Stage 1 training (v17 — part-wise + global Squeezeformer, d=256)")
    save(fig, "stage1-training")


def stage1_final_adaptation() -> None:
    result_path = (
        ROOT.parents[1]
        / "models/stage1_v17_unified_phrase_activity_adapt_reel_v2/result.json"
    )
    history = json.loads(result_path.read_text(encoding="utf-8"))["history"]
    epochs = np.array([item["epoch"] for item in history])
    selected_epoch = 26
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))

    axes[0].plot(epochs, [item["loss"] for item in history], color="#1167b1",
                 label="Adaptation loss")
    axes[0].axvline(selected_epoch, color="#d62728", linestyle="--",
                    label=f"Selected epoch ({selected_epoch})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[0].set_title("Final fusion-head adaptation loss")
    axes[0].legend(frameon=False, fontsize=8)

    series = (
        ("Validation A", "citizen", "#1167b1"),
        ("Validation B", "semlex", "#7b8794"),
        ("Equal phrase segments", "phrase", "#4d8f73"),
        ("Activity phrase crops", "phrase_activity", "#d97706"),
    )
    for label, key, color in series:
        axes[1].plot(epochs, [item["domains"][key]["top1"] for item in history],
                     label=label, color=color)
    axes[1].axvline(selected_epoch, color="#d62728", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Top-1 accuracy (%)")
    axes[1].set_ylim(50, 100)
    axes[1].set_title("Validation and contextual-crop accuracy")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")

    axes[2].plot(epochs, [item["learning_rate"] for item in history],
                 color="#6f42c1")
    selected = history[selected_epoch - 1]
    axes[2].scatter([selected_epoch], [selected["learning_rate"]], color="#d62728",
                    zorder=3, label=f"Selected epoch ({selected_epoch})")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_title("Learning-rate schedule")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Current Stage 1 final adaptation (v17 — frozen encoders, fusion head)"
    )
    save(fig, "stage1-final-adaptation")


def stage3_performance() -> None:
    labels = ["Overall", "100-gloss\nscope", "5+ glosses", "Controlled\n5+ glosses"]
    exact = [93.36, 94.00, 91.90, 100.00]
    chrf = [98.60, 99.02, 98.47, 99.57]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - width / 2, exact, width, label="Normalized exact", color="#1167b1")
    ax.bar(x + width / 2, chrf, width, label="chrF++", color="#4d8f73")
    ax.set_ylim(85, 102)
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, ncols=2)
    ax.set_title("English rephrasing performance on the held-out text test")
    for index, value in enumerate(exact):
        ax.text(index - width / 2, value + 0.35, f"{value:.2f}", ha="center", fontsize=8)
    for index, value in enumerate(chrf):
        ax.text(index + width / 2, value + 0.35, f"{value:.2f}", ha="center", fontsize=8)
    save(fig, "stage3-performance")


if __name__ == "__main__":
    grouped_accuracy()
    extractor_comparison()
    legacy_extractor_screening()
    modality_comparison()
    architecture_comparison()
    architecture_family_comparison()
    stage1_training()
    stage1_final_adaptation()
    stage3_performance()
    expected = {f"{name}.{suffix}" for name in (
        "phrase-activity-adaptation", "extractor-comparison",
        "legacy-extractor-screening", "modality-comparison",
        "architecture-comparison", "stage1-family-comparison", "stage1-training",
        "stage1-final-adaptation", "stage3-performance"
    ) for suffix in ("png", "svg")}
    written = {path.name for path in CHARTS.iterdir() if path.is_file()}
    assert expected <= written
