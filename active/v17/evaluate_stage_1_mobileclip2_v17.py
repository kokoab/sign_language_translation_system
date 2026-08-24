#!/usr/bin/env python3
"""Evaluate the frozen v17 MobileCLIP2 temporal checkpoint on validation only."""

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
    from active.v17.model_mobileclip2_v17 import MobileCLIP2Stage1Config, MobileCLIP2Stage1V17
    from active.v17.schema_mobileclip2_v17 import MobileCLIP2V17Config, schema_fingerprint
    from active.v17.train_stage_1_mobileclip2_v17 import MobileCLIP2CitizenDataset, select_device
else:
    from .model_mobileclip2_v17 import MobileCLIP2Stage1Config, MobileCLIP2Stage1V17
    from .schema_mobileclip2_v17 import MobileCLIP2V17Config, schema_fingerprint
    from .train_stage_1_mobileclip2_v17 import MobileCLIP2CitizenDataset, select_device


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics_from_logits(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    predicted = logits.argmax(1)
    top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
    classes = logits.shape[1]
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    tp = np.diag(confusion).astype(float)
    recall = tp / np.maximum(confusion.sum(1), 1)
    precision = tp / np.maximum(confusion.sum(0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "samples": int(len(targets)),
        "top1": float(100 * (predicted == targets).mean()),
        "top5": float(100 * (top5 == targets[:, None]).any(1).mean()),
        "macro_f1": float(100 * f1.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/local/citizen100_v17/mobileclip2_s0"))
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument("--rejections", type=Path, default=Path("data/local/citizen100_v17/rejections.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/stage1_v17_mobileclip2_s0_validation"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_mobileclip2_v17":
        raise ValueError("not a v17 MobileCLIP2 Stage 1 checkpoint")
    if checkpoint["manifest_sha256"] != file_sha256(args.manifest):
        raise ValueError("checkpoint manifest mismatch")
    expected_schema = schema_fingerprint(MobileCLIP2V17Config())
    if checkpoint["schema_fingerprint"] != expected_schema:
        raise ValueError("checkpoint RGB schema mismatch")
    dataset = MobileCLIP2CitizenDataset(
        args.data_root, "val", args.manifest, args.rejections,
        expected_schema=expected_schema
    )
    if checkpoint["label_to_index"] != dataset.label_to_index:
        raise ValueError("checkpoint label map mismatch")
    device = select_device(args.device)
    model = MobileCLIP2Stage1V17(MobileCLIP2Stage1Config(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    logits_batches = []
    with torch.inference_mode():
        for embeddings, _ in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
            logits = model(embeddings.to(device))
            if device.type == "mps":
                torch.mps.synchronize()
            logits_batches.append(logits.float().cpu().numpy())
    logits = np.concatenate(logits_batches)
    targets = dataset.targets.numpy()
    metrics = metrics_from_logits(logits, targets)
    metrics.update({
        "split": "val",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_evaluated": False,
    })
    predicted = logits.argmax(1)
    relative_paths = [str(path.relative_to(args.data_root)) for path in dataset.files]
    item_ids = [str(Path(path).with_suffix("").with_suffix("")) for path in relative_paths]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "logits.npz", logits=logits.astype(np.float32),
        targets=targets, item_ids=np.asarray(item_ids),
    )
    index_to_label = {value: key for key, value in dataset.label_to_index.items()}
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("feature_path", "true_label", "predicted_label", "correct"))
        for index, path in enumerate(relative_paths):
            writer.writerow((
                path, index_to_label[int(targets[index])],
                index_to_label[int(predicted[index])], bool(predicted[index] == targets[index]),
            ))
    report = [
        "# v17 MobileCLIP2-S0 Stage 1 validation", "",
        f"- Checkpoint: `{args.checkpoint}` (epoch {checkpoint['epoch']})",
        f"- Samples: {metrics['samples']}",
        f"- Top-1: {metrics['top1']:.2f}%",
        f"- Top-5: {metrics['top5']:.2f}%",
        f"- Macro F1: {metrics['macro_f1']:.2f}%",
        "- Citizen test split accessed: no", "",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
