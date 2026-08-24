#!/usr/bin/env python3
"""Evaluate a v17 hand-MobileCLIP2 checkpoint on Citizen validation only."""

from __future__ import annotations

import argparse
import csv
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

from active.v17.evaluate_stage_1_mobileclip2_v17 import file_sha256, metrics_from_logits
from active.v17.model_hand_mobileclip2_v17 import (
    HandMobileCLIP2Stage1Config,
    HandMobileCLIP2Stage1V17,
)
from active.v17.schema_hand_mobileclip2_v17 import (
    HandMobileCLIP2V17Config,
    schema_fingerprint,
)
from active.v17.train_stage_1_hand_mobileclip2_v17 import HandMobileCLIP2Dataset
from active.v17.train_stage_1_mobileclip2_v17 import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("data/local/citizen100_v17/hand_mobileclip2_s0"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("active/v17/citizen100_manifest.json"),
    )
    parser.add_argument(
        "--rejections", type=Path,
        default=Path("data/local/citizen100_v17/rejections.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_hand_mobileclip2_v17":
        raise ValueError("not a v17 hand-MobileCLIP2 checkpoint")
    if checkpoint["manifest_sha256"] != file_sha256(args.manifest):
        raise ValueError("checkpoint manifest mismatch")
    expected_schema = schema_fingerprint(HandMobileCLIP2V17Config())
    if checkpoint["schema_fingerprint"] != expected_schema:
        raise ValueError("checkpoint hand embedding schema mismatch")
    dataset = HandMobileCLIP2Dataset(
        args.data_root, "val", args.manifest, args.rejections
    )
    if checkpoint["label_to_index"] != dataset.label_to_index:
        raise ValueError("checkpoint label map mismatch")
    device = select_device(args.device)
    model = HandMobileCLIP2Stage1V17(
        HandMobileCLIP2Stage1Config(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    logits_batches = []
    with torch.inference_mode():
        for embeddings, valid, boxes, _ in DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False
        ):
            logits = model(
                embeddings.to(device), valid.to(device), boxes.to(device)
            )
            if device.type == "mps":
                torch.mps.synchronize()
            logits_batches.append(logits.float().cpu().numpy())
    logits = np.concatenate(logits_batches)
    targets = dataset.targets.numpy()
    metrics = {
        **metrics_from_logits(logits, targets), "split": "val",
        "checkpoint_epoch": int(checkpoint["epoch"]), "test_evaluated": False,
    }
    item_ids = np.asarray([
        f"val/{path.parent.name}/{path.name}" for path in dataset.files
    ])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "logits.npz", logits=logits.astype(np.float32),
        targets=targets, item_ids=item_ids,
    )
    predicted = logits.argmax(1)
    index_to_label = {value: key for key, value in dataset.label_to_index.items()}
    with (args.output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("feature_path", "true_label", "predicted_label", "correct"))
        for index, path in enumerate(dataset.files):
            writer.writerow((
                str(path.relative_to(args.data_root)),
                index_to_label[int(targets[index])],
                index_to_label[int(predicted[index])],
                bool(predicted[index] == targets[index]),
            ))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
