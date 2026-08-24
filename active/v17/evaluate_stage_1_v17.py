#!/usr/bin/env python3
"""Evaluate a frozen v17 Stage 1 checkpoint and relate errors to hand coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
    from active.v17.train_stage_1_v17 import (
        EXTRACTORS,
        Citizen100V17Dataset,
        extractor_schema_fingerprint,
        select_device,
    )
else:
    from .model_v17 import SLTStage1V17, Stage1V17Config
    from .train_stage_1_v17 import (
        EXTRACTORS,
        Citizen100V17Dataset,
        extractor_schema_fingerprint,
        select_device,
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_quality(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["feature_path"]: float(row["hand_active_output_frames"])
            for row in csv.DictReader(handle)
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data/local/citizen100_v17/landmarks"))
    parser.add_argument("--extractor", choices=EXTRACTORS, default="apple")
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument("--rejections", type=Path, default=Path("data/local/citizen100_v17/rejections.csv"))
    parser.add_argument("--quality-csv", type=Path, default=Path("artifacts/reports/citizen100_v17_landmark_quality.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/stage1_v17_validation"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("test evaluation requires explicit --allow-test")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("not a v17 Stage 1 checkpoint")
    if checkpoint["manifest_sha256"] != file_sha256(args.manifest):
        raise ValueError("checkpoint manifest does not match the current manifest")
    expected_schema = extractor_schema_fingerprint(args.extractor)
    if checkpoint["schema_fingerprint"] != expected_schema:
        raise ValueError("checkpoint extractor schema does not match v17")

    dataset = Citizen100V17Dataset(
        args.data_root, args.split, args.manifest, args.rejections,
        cache=True, expected_schema=expected_schema,
    )
    if checkpoint["label_to_index"] != dataset.label_to_index:
        raise ValueError("checkpoint label mapping does not match the dataset")
    device = select_device(args.device)
    model = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    predictions: list[int] = []
    top5_predictions: list[list[int]] = []
    all_logits: list[np.ndarray] = []
    with torch.no_grad():
        for features, _ in loader:
            logits = model(features.to(device, non_blocking=device.type == "cuda"))
            if device.type == "mps":
                torch.mps.synchronize()
            all_logits.append(logits.float().cpu().numpy())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            top5_predictions.extend(logits.topk(5, dim=1).indices.cpu().tolist())
    targets = dataset.targets.numpy()
    predicted = np.asarray(predictions, dtype=np.int64)
    top5 = np.asarray(top5_predictions, dtype=np.int64)
    correct = predicted == targets
    top5_correct = (top5 == targets[:, None]).any(axis=1)

    classes = dataset.num_classes
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    quality = load_quality(args.quality_csv)
    relative_paths = [str(path.relative_to(args.data_root)) for path in dataset.files]
    coverage = np.asarray([quality[path] for path in relative_paths])
    bins = (
        ("[0.00, 0.50)", 0.00, 0.50),
        ("[0.50, 0.75)", 0.50, 0.75),
        ("[0.75, 0.90)", 0.75, 0.90),
        ("[0.90, 1.01]", 0.90, 1.01),
    )
    coverage_rows = []
    for name, lower, upper in bins:
        selected = (coverage >= lower) & (coverage < upper)
        coverage_rows.append(
            {
                "bin": name,
                "clips": int(selected.sum()),
                "top1": float(100 * correct[selected].mean()) if selected.any() else None,
                "top5": float(100 * top5_correct[selected].mean()) if selected.any() else None,
            }
        )

    mistakes = []
    for true_index in range(classes):
        for predicted_index in range(classes):
            if true_index != predicted_index and confusion[true_index, predicted_index]:
                mistakes.append(
                    (
                        int(confusion[true_index, predicted_index]),
                        dataset.index_to_label[true_index],
                        dataset.index_to_label[predicted_index],
                    )
                )
    mistakes.sort(reverse=True)
    result = {
        "split": args.split,
        "samples": len(dataset),
        "top1": float(100 * correct.mean()),
        "top5": float(100 * top5_correct.mean()),
        "macro_f1": float(100 * f1.mean()),
        "coverage_bins": coverage_rows,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_metrics": checkpoint["validation_metrics"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "logits.npz",
        logits=np.concatenate(all_logits, axis=0).astype(np.float32),
        targets=targets,
        item_ids=np.asarray(
            [str(Path(path).with_suffix("").with_suffix("")) for path in relative_paths]
        ),
    )
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("feature_path", "true_label", "predicted_label", "correct", "hand_active_output_frames"))
        for index, path in enumerate(relative_paths):
            writer.writerow(
                (
                    path,
                    dataset.index_to_label[int(targets[index])],
                    dataset.index_to_label[int(predicted[index])],
                    bool(correct[index]),
                    float(coverage[index]),
                )
            )
    lines = [
        f"# v17 Stage 1 {args.split} evaluation",
        "",
        f"- Checkpoint: `{args.checkpoint}` (epoch {checkpoint['epoch']})",
        f"- Samples: {len(dataset)}",
        f"- Top-1: {result['top1']:.2f}%",
        f"- Top-5: {result['top5']:.2f}%",
        f"- Macro F1: {result['macro_f1']:.2f}%",
        "",
        "## Accuracy by archived hand-active coverage",
        "",
        "| Coverage | Clips | Top-1 | Top-5 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in coverage_rows:
        top1_text = "n/a" if row["top1"] is None else f"{row['top1']:.2f}%"
        top5_text = "n/a" if row["top5"] is None else f"{row['top5']:.2f}%"
        lines.append(f"| {row['bin']} | {row['clips']} | {top1_text} | {top5_text} |")
    lines.extend(
        [
            "",
            "## Most frequent confusions",
            "",
            "| Count | True | Predicted |",
            "| ---: | --- | --- |",
        ]
    )
    lines.extend(f"| {count} | {true} | {prediction} |" for count, true, prediction in mistakes[:20])
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
