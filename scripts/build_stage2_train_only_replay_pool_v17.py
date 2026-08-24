#!/usr/bin/env python3
"""Combine Citizen and contextual ASLLRP train-only frozen sign tokens."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    with np.load(args.citizen_pool, allow_pickle=False) as payload:
        citizen_features = payload["frozen_features"].astype(np.float16)
        citizen_targets = payload["target_indices"].astype(np.int64)
        citizen_metadata = json.loads(str(payload["metadata_json"]))
    if citizen_metadata.get("source_split") != "citizen_official_train_only":
        raise ValueError("Citizen source pool is not official-training-only")
    if citizen_features.ndim != 3 or citizen_features.shape[1:] != (32, 612):
        raise ValueError(f"unexpected Citizen feature shape: {citizen_features.shape}")

    contextual_features: list[np.ndarray] = []
    contextual_targets: list[int] = []
    contextual_ids: list[str] = []
    excluded_multiwindow: list[str] = []
    checkpoint_hashes: set[str] = set()
    for path in sorted(args.contextual_root.rglob("*.stage2_frozen_v17.npz")):
        with np.load(path, allow_pickle=False) as payload:
            features = payload["frozen_features"].astype(np.float16)
            targets = payload["target_indices"].astype(np.int64)
            metadata = json.loads(str(payload["metadata_json"]))
        if metadata.get("role") != "train" or metadata.get("source") != "asllrp_segmented_train":
            raise ValueError(f"{path}: contextual source/role mismatch")
        if targets.shape != (1,):
            raise ValueError(f"{path}: contextual target is not isolated")
        checkpoint_hashes.add(str(metadata["stage1_checkpoint_sha256"]))
        if features.shape != (1, 32, 612):
            if features.ndim == 3 and features.shape[1:] == (32, 612):
                excluded_multiwindow.append(str(metadata["source_item_id"]))
                continue
            raise ValueError(f"{path}: unexpected contextual feature shape {features.shape}")
        contextual_features.append(features[0])
        contextual_targets.append(int(targets[0]))
        contextual_ids.append(str(metadata["source_item_id"]))
    if not contextual_features:
        raise ValueError("no one-window contextual train features found")
    expected_checkpoint = str(citizen_metadata["stage1_checkpoint_sha256"])
    if checkpoint_hashes != {expected_checkpoint}:
        raise ValueError("Citizen/contextual Stage-1 checkpoint mismatch")

    contextual_array = np.stack(contextual_features).astype(np.float16)
    contextual_target_array = np.asarray(contextual_targets, dtype=np.int64)
    features = np.concatenate([citizen_features, contextual_array], axis=0)
    targets = np.concatenate([citizen_targets, contextual_target_array], axis=0)
    source_codes = np.concatenate([
        np.zeros(len(citizen_targets), dtype=np.uint8),
        np.ones(len(contextual_target_array), dtype=np.uint8),
    ])
    citizen_counts = Counter(int(value) for value in citizen_targets)
    contextual_counts = Counter(int(value) for value in contextual_target_array)
    if sorted(citizen_counts) != list(range(100)):
        raise ValueError("Citizen replay no longer covers the locked 100 classes")

    metadata = {
        "format": "slt_stage2_train_only_replay_pool_v17",
        "format_version": 1,
        "source_split": "citizen_asllrp_train_only_replay",
        "source_code_map": {"0": "citizen_official_train_only", "1": "asllrp_contextual_train_only"},
        "citizen_pool": args.citizen_pool.as_posix(),
        "citizen_pool_sha256": sha256(args.citizen_pool),
        "contextual_root": args.contextual_root.as_posix(),
        "stage1_checkpoint_sha256": expected_checkpoint,
        "items": int(len(targets)),
        "citizen_items": int(len(citizen_targets)),
        "asllrp_contextual_items": int(len(contextual_target_array)),
        "asllrp_contextual_classes": sorted(contextual_counts),
        "asllrp_contextual_item_ids": contextual_ids,
        "excluded_multiwindow_item_ids": excluded_multiwindow,
        "citizen_class_counts": {str(key): citizen_counts[key] for key in range(100)},
        "asllrp_contextual_class_counts": {
            str(key): contextual_counts[key] for key in sorted(contextual_counts)
        },
        "feature_dim": 612,
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
        source_codes=source_codes,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(args.output)
    result = {
        "output": args.output.as_posix(),
        "output_sha256": sha256(args.output),
        "items": int(len(targets)),
        "citizen_items": int(len(citizen_targets)),
        "asllrp_contextual_items": int(len(contextual_target_array)),
        "asllrp_contextual_classes": len(contextual_counts),
        "excluded_multiwindow_items": len(excluded_multiwindow),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--citizen-pool", type=Path,
        default=Path("data/local/stage2_v17_synthetic/citizen_train_isolated_pool.npz"),
    )
    parser.add_argument(
        "--contextual-root", type=Path,
        default=Path("data/local/stage2_v17_asllrp_segmented_train_frozen_features"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/local/stage2_v17_synthetic/train_only_replay_pool_v2.npz"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_synthetic/train_only_replay_pool_v2.json"),
    )
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
