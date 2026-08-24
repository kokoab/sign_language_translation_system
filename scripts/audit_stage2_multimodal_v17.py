#!/usr/bin/env python3
"""Fail-closed audit for windowed v17 Stage-2 landmark/hand-RGB archives."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.schema_stage2_features_v17 import Stage2FeatureV17Config, schema_fingerprint
from active.v17.schema_v17 import MOUTH_END, MOUTH_START
from scripts.extract_stage2_multimodal_v17 import ACTIVE_ROLES, safe_name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_archive(
    path: Path, row: dict[str, Any], manifest_sha: str, expected_schema: str
) -> dict[str, Any]:
    errors = []
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        landmarks = payload["landmarks"]
        landmark_valid = payload["landmark_window_valid"]
        offsets = payload["hand_jpeg_offsets"]
        blob = payload["hand_jpeg_blob"]
        hand_valid = payload["hand_valid"]
        boxes = payload["hand_boxes_normalized"]
        ranges = payload["window_source_ranges"]
        targets = payload["target_indices"]
        windows = landmarks.shape[0]
        expected_shapes = {
            "landmarks": (windows, 32, 61, 5),
            "landmark_window_valid": (windows,),
            "hand_jpeg_offsets": (windows, 16, 3, 2),
            "hand_valid": (windows, 16, 3),
            "hand_boxes_normalized": (windows, 16, 3, 4),
            "window_source_ranges": (windows, 2),
        }
        values = {
            "landmarks": landmarks,
            "landmark_window_valid": landmark_valid,
            "hand_jpeg_offsets": offsets,
            "hand_valid": hand_valid,
            "hand_boxes_normalized": boxes,
            "window_source_ranges": ranges,
        }
        for name, expected in expected_shapes.items():
            if values[name].shape != expected:
                errors.append(f"{name} shape {values[name].shape} != {expected}")
        if windows < 1:
            errors.append("no temporal windows")
        if not np.isfinite(landmarks).all() or not np.isfinite(boxes).all():
            errors.append("non-finite features")
        if not landmark_valid.any():
            errors.append("no valid landmark window")
        if not hand_valid.any():
            errors.append("no valid hand crop")
        starts = offsets[..., 0]
        lengths = offsets[..., 1]
        present = starts >= 0
        if np.any(starts[present] + lengths[present] > len(blob)):
            errors.append("JPEG offset exceeds blob")
        if np.any(hand_valid != present):
            errors.append("hand valid mask and JPEG offsets disagree")
        if list(targets.astype(int)) != row["target_indices"]:
            errors.append("target indices disagree with manifest")
        if row.get("zero_lip_nodes") and np.any(landmarks[:, :, MOUTH_START:MOUTH_END]):
            errors.append("local unavailable lip nodes are not zero")
        checks = {
            "schema_fingerprint": expected_schema,
            "training_manifest_sha256": manifest_sha,
            "source_item_id": row["source_item_id"],
            "video_sha256": row["video_sha256"],
            "role": row["role"],
            "source": row["source"],
        }
        for key, expected in checks.items():
            if metadata.get(key) != expected:
                errors.append(f"metadata {key} mismatch")
        return {
            "path": path.as_posix(),
            "errors": errors,
            "windows": windows,
            "invalid_landmark_windows": int((~landmark_valid).sum()),
            "hand_valid_fraction": float(hand_valid.mean()),
            "bytes": path.stat().st_size,
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_sha = sha256(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    rows = [row for row in manifest["rows"] if row["role"] in ACTIVE_ROLES]
    expected_schema = schema_fingerprint(
        Stage2FeatureV17Config(maximum_source_frames=args.maximum_source_frames)
    )
    expected_paths = {
        (
            args.root / row["role"] / row["source"] /
            f"{safe_name(row)}.stage2_rgb_v17.npz"
        ): row
        for row in rows
    }
    actual_paths = set(args.root.glob("*/*/*.stage2_rgb_v17.npz"))
    missing = sorted(path.as_posix() for path in set(expected_paths) - actual_paths)
    unexpected = sorted(path.as_posix() for path in actual_paths - set(expected_paths))
    details = []
    for path, row in sorted(expected_paths.items(), key=lambda item: item[0].as_posix()):
        if path.exists():
            details.append(audit_archive(path, row, manifest_sha, expected_schema))
    failures = [detail for detail in details if detail["errors"]]
    counts = Counter((row["source"], row["role"]) for row in rows)
    report = {
        "format": "slt_v17_stage2_multimodal_extraction_audit",
        "version": 1,
        "manifest": args.manifest.as_posix(),
        "manifest_sha256": manifest_sha,
        "root": args.root.as_posix(),
        "schema_fingerprint": expected_schema,
        "expected_archives": len(rows),
        "actual_archives": len(actual_paths),
        "audited_archives": len(details),
        "missing": missing,
        "unexpected": unexpected,
        "archive_failures": failures,
        "archive_failure_count": len(failures),
        "source_role_counts": {
            f"{source}:{role}": count
            for (source, role), count in sorted(counts.items())
        },
        "total_windows": sum(item["windows"] for item in details),
        "invalid_landmark_windows": sum(item["invalid_landmark_windows"] for item in details),
        "mean_hand_valid_fraction": (
            float(np.mean([item["hand_valid_fraction"] for item in details]))
            if details else 0.0
        ),
        "archive_bytes": sum(item["bytes"] for item in details),
        "integrity_pass": not missing and not unexpected and not failures,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("active/v17/stage2_training_manifest_v17.json"),
    )
    parser.add_argument("--root", type=Path, default=Path("data/local/stage2_v17_multimodal"))
    parser.add_argument("--maximum-source-frames", type=int, default=256)
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_multimodal_extraction/audit.json"),
    )
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps(report, indent=2))
    if not report["integrity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
