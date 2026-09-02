#!/usr/bin/env python3
"""Cache train/validation isolated clips as frozen Stage-2 temporal features."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.15")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.07")
import torch

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from active.v17.model_stage2_v17 import (
    FROZEN_TEMPORAL_FEATURE_DIM,
    FrozenUnifiedTemporalEncoderV17,
    load_frozen_unified_stage1,
)
from active.v17.schema_hand_mobileclip2_v17 import (
    HandMobileCLIP2V17Config,
    schema_fingerprint as hand_schema_fingerprint,
)
from active.v17.schema_v17 import V17Config, schema_fingerprint
from active.v17.train_stage_1_v17 import load_v17_archive, mask_mouth_nodes_v17
from active.v17.train_unified_multimodal_student_v17 import (
    PairRecord,
    citizen_records,
    label_map,
    load_hand,
    semlex_validation_records,
    supplement_records,
)


LOG = logging.getLogger("cache_stage2_isolated_pool_v17")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(args: argparse.Namespace, labels: dict[str, int]) -> list[PairRecord]:
    if args.source == "citizen":
        split = "train" if args.role == "train" else "val"
        return citizen_records(
            args.citizen_landmarks, args.citizen_hand, split, labels,
            args.citizen_rejections,
        )
    if args.source == "semlex":
        if args.role == "train":
            return supplement_records(
                args.semlex_train_manifest, "semlex", args.supplement_hand, labels
            )
        return semlex_validation_records(
            args.semlex_val_manifest, args.semlex_val_hand, labels
        )
    source = "local_deep_clean" if args.role == "train" else "local_deep_clean_val"
    manifest = args.local_train_manifest if args.role == "train" else args.local_val_manifest
    return supplement_records(manifest, source, args.local_hand, labels)


@torch.inference_mode()
def encode_batch(
    rows: list[PairRecord], encoder: FrozenUnifiedTemporalEncoderV17,
    device: torch.device,
) -> np.ndarray:
    landmark_schema = schema_fingerprint(V17Config())
    hand_schema = hand_schema_fingerprint(HandMobileCLIP2V17Config())
    landmarks, embeddings, valid, boxes = [], [], [], []
    for row in rows:
        landmark = load_v17_archive(row.landmark_path, landmark_schema)
        if row.mask_mouth:
            landmark = mask_mouth_nodes_v17(landmark)
        hand, hand_valid, hand_boxes = load_hand(row.hand_path, hand_schema)
        landmarks.append(landmark)
        embeddings.append(hand)
        valid.append(hand_valid)
        boxes.append(hand_boxes)
    value = encoder(
        torch.stack(landmarks).unsqueeze(1).to(device),
        torch.stack(embeddings).unsqueeze(1).to(device),
        torch.stack(valid).unsqueeze(1).to(device),
        torch.stack(boxes).unsqueeze(1).to(device),
    )[:, 0]
    return value.float().cpu().numpy()


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.role not in {"train", "validation"}:
        raise ValueError("isolated cache role must be train or validation")
    if not 0 < args.mps_memory_fraction <= 0.25:
        raise ValueError("MPS memory fraction must be in (0, 0.25]")
    labels = label_map(args.manifest)
    rows = records(args, labels)
    if not rows:
        raise ValueError("no isolated rows selected")
    counts = Counter(row.label for row in rows)
    if args.source == "citizen" and set(counts) != set(labels):
        raise ValueError(f"{args.source} {args.role} does not cover all locked classes")

    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else args.device
    )
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    landmark, hand, fusion, checkpoint = load_frozen_unified_stage1(args.stage1_checkpoint)
    encoder = FrozenUnifiedTemporalEncoderV17(landmark, hand, fusion).to(device).eval()
    features = np.empty(
        (len(rows), 32, FROZEN_TEMPORAL_FEATURE_DIM), dtype=np.float16
    )
    started = time.monotonic()
    peak_driver = 0
    for start in range(0, len(rows), args.batch_size):
        stop = min(len(rows), start + args.batch_size)
        features[start:stop] = encode_batch(rows[start:stop], encoder, device).astype(np.float16)
        if device.type == "mps":
            torch.mps.synchronize()
            peak_driver = max(peak_driver, int(torch.mps.driver_allocated_memory()))
        if stop % 256 == 0 or stop == len(rows):
            LOG.info("%d/%d elapsed=%.1fs", stop, len(rows), time.monotonic() - started)
        if stop % 1024 == 0:
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()

    metadata = {
        "format": "slt_stage2_isolated_pool_v17",
        "format_version": 1,
        "source": args.source,
        "role": args.role,
        "items": len(rows),
        "classes": len(counts),
        "class_counts": dict(sorted(counts.items())),
        "item_ids": [row.item_id for row in rows],
        "stage1_checkpoint": args.stage1_checkpoint.as_posix(),
        "stage1_checkpoint_sha256": sha256(args.stage1_checkpoint),
        "local_mouth_policy": "zero_only_four_lip_points" if args.source == "local" else "full_face",
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        frozen_features=features,
        target_indices=np.asarray([row.target for row in rows], dtype=np.int64),
        item_ids=np.asarray([row.item_id for row in rows]),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(args.output)
    result = {
        "output": args.output.as_posix(),
        "output_sha256": sha256(args.output),
        "source": args.source,
        "role": args.role,
        "items": len(rows),
        "classes": len(counts),
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
    parser.add_argument("--source", choices=("citizen", "semlex", "local"), required=True)
    parser.add_argument("--role", choices=("train", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument("--citizen-landmarks", type=Path, default=Path("data/local/citizen100_v17/landmarks"))
    parser.add_argument("--citizen-hand", type=Path, default=Path("data/local/citizen100_v17/hand_mobileclip2_s0"))
    parser.add_argument("--citizen-rejections", type=Path, default=Path("data/local/citizen100_v17/rejections.csv"))
    parser.add_argument("--semlex-train-manifest", type=Path, default=Path("data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json"))
    parser.add_argument("--semlex-val-manifest", type=Path, default=Path("data/local/semlex_citizen100_val_audit/selection_plan.json"))
    parser.add_argument("--semlex-val-hand", type=Path, default=Path("data/local/semlex_citizen100_val_audit/hand_mobileclip2_s0"))
    parser.add_argument("--supplement-hand", type=Path, default=Path("data/local/hand_mobileclip2_supplements_v17"))
    parser.add_argument("--local-train-manifest", type=Path, default=Path("data/local/local_deep_clean_v17/train_final_manifest.json"))
    parser.add_argument("--local-val-manifest", type=Path, default=Path("data/local/local_deep_clean_v17/val_final_manifest.json"))
    parser.add_argument("--local-hand", type=Path, default=Path("data/local/local_deep_clean_v17/hand_mobileclip2_s0"))
    parser.add_argument("--stage1-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
