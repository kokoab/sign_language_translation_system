#!/usr/bin/env python3
"""Fail-closed audit for 2M-Flores frozen Stage-1 temporal feature archives."""

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

from active.v17.model_stage2_v17 import FROZEN_TEMPORAL_FEATURE_DIM
from scripts.cache_stage2_frozen_features_v17 import cache_path, sha256


def run(args: argparse.Namespace) -> dict[str, object]:
    vocabulary = json.loads(args.vocabulary.read_text())
    manifest = json.loads(args.manifest.read_text())
    class_count = int(vocabulary["expanded_vocabulary_count"])
    if class_count != len(vocabulary["classes"]):
        raise ValueError("auxiliary vocabulary class count mismatch")
    if int(vocabulary["locked_prefix_count"]) != 100:
        raise ValueError("locked 100-class prefix changed")
    crop_paths = sorted(args.crop_root.glob("*/*/*.stage2_rgb_v17.npz"))
    expected = {cache_path(args.cache_root, args.crop_root, path): path for path in crop_paths}
    actual = set(args.cache_root.glob("*/*/*.stage2_frozen_v17.npz"))
    failures = []
    total_windows = total_targets = 0
    for frozen_path, crop_path in expected.items():
        errors = []
        if not frozen_path.exists():
            failures.append({"path": frozen_path.as_posix(), "errors": ["missing"]})
            continue
        with np.load(crop_path, allow_pickle=False) as crop:
            crop_metadata = json.loads(str(crop["metadata_json"]))
            crop_targets = crop["target_indices"].astype(np.int64)
            windows = int(crop["landmarks"].shape[0])
        with np.load(frozen_path, allow_pickle=False) as frozen:
            metadata = json.loads(str(frozen["metadata_json"]))
            features = frozen["frozen_features"]
            mask = frozen["window_mask"].astype(np.bool_)
            targets = frozen["target_indices"].astype(np.int64)
        if features.shape != (windows, 32, FROZEN_TEMPORAL_FEATURE_DIM):
            errors.append(f"feature shape {features.shape}")
        if mask.shape != (windows,) or not mask.all():
            errors.append("invalid window mask")
        if not np.array_equal(targets, crop_targets):
            errors.append("target mismatch")
        if len(targets) > windows * 8:
            errors.append("target length exceeds CTC steps")
        if np.any(targets < 0) or np.any(targets >= class_count):
            errors.append("target outside auxiliary vocabulary")
        if not np.isfinite(features).all():
            errors.append("non-finite frozen features")
        for key in ("source_item_id", "source", "role", "video_sha256", "window_count"):
            if metadata.get(key) != crop_metadata.get(key):
                errors.append(f"metadata {key} mismatch")
        if metadata.get("stage1_checkpoint_sha256") != args.stage1_checkpoint_sha256:
            errors.append("Stage-1 checkpoint mismatch")
        if metadata.get("source") != "two_m_flores_asl" or metadata.get("role") != "train":
            errors.append("source/role is not 2M-Flores training-only")
        if errors:
            failures.append({"path": frozen_path.as_posix(), "errors": errors})
        total_windows += windows
        total_targets += len(targets)
    unexpected = sorted(path.as_posix() for path in actual - set(expected))
    manifest_rows = len(manifest["rows"])
    report = {
        "format": "slt_stage2_2m_flores_frozen_features_audit_v17",
        "version": 1,
        "manifest": args.manifest.as_posix(),
        "manifest_sha256": sha256(args.manifest),
        "vocabulary": args.vocabulary.as_posix(),
        "vocabulary_sha256": sha256(args.vocabulary),
        "expected_archives": len(expected),
        "actual_archives": len(actual),
        "manifest_rows": manifest_rows,
        "total_windows": total_windows,
        "total_target_tokens": total_targets,
        "auxiliary_classes": class_count,
        "stage1_checkpoint_sha256": args.stage1_checkpoint_sha256,
        "failures": failures,
        "unexpected": unexpected,
        "integrity_pass": (
            not failures and not unexpected and len(expected) == len(actual) == manifest_rows
        ),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-root", type=Path, default=Path("data/local/stage2_v17_2m_flores_multimodal"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/local/stage2_v17_2m_flores_frozen_features"))
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/stage2_2m_flores_training_manifest_v17.json"))
    parser.add_argument("--vocabulary", type=Path, default=Path("active/v17/stage2_2m_flores_vocabulary_v17.json"))
    parser.add_argument(
        "--stage1-checkpoint-sha256",
        default="1caeadf4b3ca620aa9fef00b35c012b39d7c093f67da1ee2f6987d2c2297906b",
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_2m_flores_frozen_features/audit.json"),
    )
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps(report, indent=2))
    if not report["integrity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
