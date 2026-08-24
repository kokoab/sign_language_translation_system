#!/usr/bin/env python3
"""Audit every windowed Stage-2 MobileCLIP2 hand embedding archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.schema_stage2_hand_mobileclip2_v17 import (
    Stage2HandMobileCLIP2V17Config,
    schema_fingerprint,
)
from scripts.encode_stage2_hand_mobileclip2_v17 import output_path


def run(args: argparse.Namespace):
    crops = sorted(args.crop_root.glob("*/*/*.stage2_rgb_v17.npz"))
    expected = {output_path(args.hand_root, args.crop_root, path): path for path in crops}
    actual = set(args.hand_root.glob("*/*/*.stage2_hand_mobileclip2_v17.npz"))
    failures = []
    windows = valid_views = 0
    expected_schema = schema_fingerprint(Stage2HandMobileCLIP2V17Config())
    for hand_path, crop_path in expected.items():
        errors = []
        if not hand_path.exists():
            failures.append({"path": hand_path.as_posix(), "errors": ["missing"]})
            continue
        with np.load(crop_path, allow_pickle=False) as crop:
            crop_metadata = json.loads(str(crop["metadata_json"]))
            crop_valid = crop["hand_valid"].astype(np.bool_)
            crop_boxes = crop["hand_boxes_normalized"].astype(np.float16)
        with np.load(hand_path, allow_pickle=False) as hand:
            metadata = json.loads(str(hand["metadata_json"]))
            embeddings = hand["embeddings"]
            valid = hand["valid"].astype(np.bool_)
            boxes = hand["boxes_normalized"].astype(np.float16)
        expected_shape = crop_valid.shape + (512,)
        if embeddings.shape != expected_shape:
            errors.append(f"embedding shape {embeddings.shape} != {expected_shape}")
        if not np.array_equal(valid, crop_valid) or not np.array_equal(boxes, crop_boxes):
            errors.append("valid/box arrays differ from crop source")
        if not np.isfinite(embeddings).all():
            errors.append("non-finite embedding")
        if np.any(embeddings[~valid] != 0):
            errors.append("invalid views have nonzero embeddings")
        if valid.any():
            norms = np.linalg.norm(embeddings[valid].astype(np.float32), axis=-1)
            if float(np.max(np.abs(norms - 1.0))) > 0.01:
                errors.append("valid embeddings are not unit normalized")
        for key in ("source_item_id", "source", "role", "video_sha256", "window_count"):
            if metadata.get(key) != crop_metadata.get(key):
                errors.append(f"metadata {key} mismatch")
        if metadata.get("schema_fingerprint") != expected_schema:
            errors.append("schema fingerprint mismatch")
        if errors:
            failures.append({"path": hand_path.as_posix(), "errors": errors})
        windows += crop_valid.shape[0]
        valid_views += int(valid.sum())
    unexpected = sorted(path.as_posix() for path in actual - set(expected))
    report = {
        "format": "slt_stage2_hand_mobileclip2_v17_audit",
        "version": 1,
        "expected_archives": len(expected),
        "actual_archives": len(actual),
        "total_windows": windows,
        "valid_views": valid_views,
        "schema_fingerprint": expected_schema,
        "failures": failures,
        "unexpected": unexpected,
        "integrity_pass": not failures and not unexpected and len(actual) == len(expected),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-root", type=Path, default=Path("data/local/stage2_v17_multimodal"))
    parser.add_argument("--hand-root", type=Path, default=Path("data/local/stage2_v17_hand_mobileclip2"))
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_hand_mobileclip2/audit.json"),
    )
    return parser


def main():
    report = run(build_parser().parse_args())
    print(json.dumps(report, indent=2))
    if not report["integrity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
