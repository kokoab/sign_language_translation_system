#!/usr/bin/env python3
"""Audit cached real and synthetic frozen-feature inputs before Stage-2 training."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.cache_stage2_frozen_features_v17 import cache_path, sha256


def run(args: argparse.Namespace):
    crop_paths = sorted(args.crop_root.glob("*/*/*.stage2_rgb_v17.npz"))
    expected = {cache_path(args.cache_root, args.crop_root, path): path for path in crop_paths}
    actual = set(args.cache_root.glob("*/*/*.stage2_frozen_v17.npz"))
    failures = []
    source_roles = Counter()
    total_windows = 0
    for frozen_path, crop_path in expected.items():
        errors = []
        if not frozen_path.exists():
            failures.append({"path": frozen_path.as_posix(), "errors": ["missing"]})
            continue
        with np.load(crop_path, allow_pickle=False) as crop:
            crop_metadata = json.loads(str(crop["metadata_json"]))
            target_indices = crop["target_indices"].astype(np.int64)
            windows = crop["landmarks"].shape[0]
        with np.load(frozen_path, allow_pickle=False) as frozen:
            metadata = json.loads(str(frozen["metadata_json"]))
            features = frozen["frozen_features"]
            mask = frozen["window_mask"].astype(np.bool_)
            frozen_targets = frozen["target_indices"].astype(np.int64)
        if features.shape != (windows, 32, 612):
            errors.append(f"feature shape {features.shape}")
        if mask.shape != (windows,) or not mask.all():
            errors.append("invalid real window mask")
        if not np.array_equal(target_indices, frozen_targets):
            errors.append("target mismatch")
        if not np.isfinite(features).all():
            errors.append("non-finite features")
        for key in ("source_item_id", "source", "role", "video_sha256", "window_count"):
            if metadata.get(key) != crop_metadata.get(key):
                errors.append(f"metadata {key} mismatch")
        if errors:
            failures.append({"path": frozen_path.as_posix(), "errors": errors})
        source_roles[(metadata["source"], metadata["role"])] += 1
        total_windows += windows

    with np.load(args.synthetic_pool, allow_pickle=False) as pool:
        pool_features = pool["frozen_features"]
        pool_targets = pool["target_indices"].astype(np.int64)
        pool_metadata = json.loads(str(pool["metadata_json"]))
    plan = json.loads(args.synthetic_plan.read_text())
    synthetic_errors = []
    if pool_features.shape != (1475, 32, 612) or not np.isfinite(pool_features).all():
        synthetic_errors.append(f"invalid pool features {pool_features.shape}")
    if sorted(set(pool_targets.tolist())) != list(range(100)):
        synthetic_errors.append("pool lacks locked classes")
    if plan["pool_sha256"] != sha256(args.synthetic_pool):
        synthetic_errors.append("plan/pool hash mismatch")
    if pool_metadata.get("source_split") != "citizen_official_train_only":
        synthetic_errors.append("pool source is not Citizen train-only")
    for row in plan["rows"]:
        indices = np.asarray(row["pool_indices"], dtype=np.int64)
        targets = np.asarray(row["target_indices"], dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(pool_targets)):
            synthetic_errors.append(f"{row['sequence_id']}: pool index out of range")
            break
        if not np.array_equal(pool_targets[indices], targets):
            synthetic_errors.append(f"{row['sequence_id']}: target mismatch")
            break
    unexpected = sorted(path.as_posix() for path in actual - set(expected))
    report = {
        "format": "slt_stage2_frozen_training_inputs_audit",
        "version": 1,
        "expected_real_archives": len(expected),
        "actual_real_archives": len(actual),
        "real_total_windows": total_windows,
        "real_source_role_counts": {
            f"{source}:{role}": count
            for (source, role), count in sorted(source_roles.items())
        },
        "real_failures": failures,
        "unexpected_real_archives": unexpected,
        "synthetic_pool_items": int(pool_features.shape[0]),
        "synthetic_pool_classes": len(set(pool_targets.tolist())),
        "synthetic_plan_sequences": len(plan["rows"]),
        "synthetic_errors": synthetic_errors,
        "integrity_pass": not failures and not unexpected and not synthetic_errors and len(actual) == len(expected),
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
    parser.add_argument("--cache-root", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    parser.add_argument(
        "--synthetic-pool", type=Path,
        default=Path("data/local/stage2_v17_synthetic/citizen_train_isolated_pool.npz"),
    )
    parser.add_argument(
        "--synthetic-plan", type=Path,
        default=Path("active/v17/stage2_synthetic_plan_v17.json"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_frozen_features/audit.json"),
    )
    return parser


def main():
    report = run(build_parser().parse_args())
    print(json.dumps(report, indent=2))
    if not report["integrity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
