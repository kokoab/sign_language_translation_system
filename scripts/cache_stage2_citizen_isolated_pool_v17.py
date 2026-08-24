#!/usr/bin/env python3
"""Cache Citizen-train-only frozen temporal tokens for synthetic Stage-2 replay."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")

import argparse
import csv
import gc
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_stage2_v17 import (
    FROZEN_TEMPORAL_FEATURE_DIM,
    FrozenUnifiedTemporalEncoderV17,
    load_frozen_unified_stage1,
)


LOG = logging.getLogger("cache_stage2_citizen_isolated_pool_v17")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(args: argparse.Namespace, label_to_index: dict[str, int]):
    with args.rejections.open(newline="", encoding="utf-8") as handle:
        rejected = {
            (row["canonical_label"], row["video"])
            for row in csv.DictReader(handle)
            if row["split"] == "train"
        }
    output = []
    for label, target in sorted(label_to_index.items(), key=lambda item: item[1]):
        for landmark in sorted((args.landmark_root / "train" / label).glob("*.v17.npz")):
            stem = landmark.name.removesuffix(".v17.npz")
            if (label, f"{stem}.mp4") in rejected:
                continue
            hand = args.hand_root / "train" / label / f"{stem}.hand_mobileclip2_v17.npz"
            if hand.exists():
                output.append((label, target, stem, landmark, hand))
    return output


def run(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
    label_to_index = {str(k): int(v) for k, v in checkpoint["label_to_index"].items()}
    if sorted(label_to_index.values()) != list(range(100)):
        raise ValueError("selected Stage-1 label map is not the locked 100 classes")
    rows = records(args, label_to_index)
    counts = {label: sum(row[0] == label for row in rows) for label in label_to_index}
    if min(counts.values()) < 1:
        raise ValueError(f"Citizen train-only pool lacks classes: {counts}")
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    )
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    landmark_model, hand_model, fusion, _ = load_frozen_unified_stage1(args.stage1_checkpoint)
    encoder = FrozenUnifiedTemporalEncoderV17(landmark_model, hand_model, fusion).to(device).eval()
    features = np.zeros((len(rows), 32, FROZEN_TEMPORAL_FEATURE_DIM), dtype=np.float16)
    targets = np.zeros(len(rows), dtype=np.int64)
    item_ids = []
    started = time.monotonic()
    peak_driver = 0
    for index, (label, target, stem, landmark_path, hand_path) in enumerate(rows):
        with np.load(landmark_path, allow_pickle=False) as payload:
            landmarks = payload["features"].astype(np.float32)
        with np.load(hand_path, allow_pickle=False) as payload:
            embeddings = payload["embeddings"].astype(np.float32)
            valid = payload["valid"].astype(np.bool_)
            boxes = payload["boxes_normalized"].astype(np.float32)
        with torch.inference_mode():
            value = encoder(
                torch.from_numpy(landmarks).reshape(1, 1, 32, 61, 5).to(device),
                torch.from_numpy(embeddings).reshape(1, 1, 16, 3, 512).to(device),
                torch.from_numpy(valid).reshape(1, 1, 16, 3).to(device),
                torch.from_numpy(boxes).reshape(1, 1, 16, 3, 4).to(device),
            )[0, 0].float().cpu().numpy()
            if device.type == "mps":
                torch.mps.synchronize()
        features[index] = value.astype(np.float16)
        targets[index] = target
        item_ids.append(f"{label}/{stem}")
        del landmarks, embeddings, valid, boxes, value
        if (index + 1) % 16 == 0:
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
        if device.type == "mps":
            peak_driver = max(peak_driver, int(torch.mps.driver_allocated_memory()))
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(rows):
            LOG.info("%d/%d elapsed=%.1fs", index + 1, len(rows), time.monotonic() - started)
    metadata = {
        "format": "slt_stage2_citizen_train_isolated_pool_v17",
        "format_version": 1,
        "stage1_checkpoint": args.stage1_checkpoint.as_posix(),
        "stage1_checkpoint_sha256": sha256(args.stage1_checkpoint),
        "source_split": "citizen_official_train_only",
        "items": len(rows),
        "class_counts": counts,
        "item_ids": item_ids,
        "feature_dim": FROZEN_TEMPORAL_FEATURE_DIM,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        frozen_features=features,
        target_indices=targets,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(args.output)
    result = {
        "output": args.output.as_posix(),
        "output_sha256": sha256(args.output),
        "items": len(rows),
        "classes": len(counts),
        "minimum_class_items": min(counts.values()),
        "maximum_class_items": max(counts.values()),
        "device": str(device),
        "peak_mps_driver_bytes": peak_driver,
        "seconds": time.monotonic() - started,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landmark-root", type=Path, default=Path("data/local/citizen100_v17/landmarks"))
    parser.add_argument("--hand-root", type=Path, default=Path("data/local/citizen100_v17/hand_mobileclip2_s0"))
    parser.add_argument("--rejections", type=Path, default=Path("data/local/citizen100_v17/rejections.csv"))
    parser.add_argument(
        "--stage1-checkpoint", type=Path,
        default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/local/stage2_v17_synthetic/citizen_train_isolated_pool.npz"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_synthetic/citizen_pool.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.12)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
