#!/usr/bin/env python3
"""Merge Citizen, SemLex, and contextual ASLLRP train-only signer trajectories."""

from __future__ import annotations

import argparse
import csv
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
    with np.load(args.base_pool, allow_pickle=False) as payload:
        base_features = payload["frozen_features"].astype(np.float16)
        base_targets = payload["target_indices"].astype(np.int64)
        base_sources = payload["source_codes"].astype(np.uint8)
        base_metadata = json.loads(str(payload["metadata_json"]))
    if base_metadata.get("source_split") != "citizen_asllrp_train_only_replay":
        raise ValueError("base pool is not the frozen Citizen/ASLLRP train-only pool")
    with np.load(args.semlex_pool, allow_pickle=False) as payload:
        semlex_features = payload["frozen_features"].astype(np.float16)
        semlex_targets = payload["target_indices"].astype(np.int64)
        semlex_signers = payload["signer_ids"].astype(str)
        semlex_frames = payload["observed_frames"].astype(np.int16)
        semlex_metadata = json.loads(str(payload["metadata_json"]))
    if semlex_metadata.get("source_split") != "semlex_official_train_only":
        raise ValueError("SemLex pool is not official-training-only")
    if semlex_metadata["stage1_checkpoint_sha256"] != base_metadata["stage1_checkpoint_sha256"]:
        raise ValueError("Stage-1 checkpoint mismatch across pools")

    provenance = {
        row["video"].removesuffix(".mp4"): row
        for row in csv.DictReader(args.citizen_provenance.open(newline="", encoding="utf-8"))
        if row["split"] == "train"
    }
    citizen_ids = list(json.loads(str(np.load(args.citizen_pool, allow_pickle=False)["metadata_json"]))["item_ids"])
    citizen_signers = []
    citizen_frames = []
    for item_id in citizen_ids:
        label, stem = str(item_id).split("/", 1)
        row = provenance.get(stem)
        if row is None:
            raise ValueError(f"missing Citizen signer provenance for {item_id}")
        citizen_signers.append(f"citizen:{row['participant']}")
        landmark_path = args.citizen_landmark_root / "train" / label / f"{stem}.v17.npz"
        with np.load(landmark_path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
        citizen_frames.append(max(4, min(32, int(metadata.get("source_frames_processed", 32)))))

    contextual_manifest = json.loads(args.contextual_manifest.read_text())
    row_by_item = {str(row["source_item_id"]): row for row in contextual_manifest["rows"]}
    duration_by_item = {}
    for path in sorted(args.contextual_multimodal_root.rglob("*.stage2_rgb_v17.npz")):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
            ranges = payload["window_source_ranges"].astype(np.int64)
        if len(ranges) == 1:
            duration_by_item[str(metadata["source_item_id"])] = int(ranges[0, 1] - ranges[0, 0])
    contextual_ids = list(base_metadata["asllrp_contextual_item_ids"])
    contextual_signers = []
    contextual_frames = []
    for item_id in contextual_ids:
        row = row_by_item.get(str(item_id))
        if row is None or row.get("role") != "train" or str(item_id) not in duration_by_item:
            raise ValueError(f"missing ASLLRP train-only signer/timing for {item_id}")
        contextual_signers.append(f"asllrp:{row['signer_id']}")
        contextual_frames.append(max(4, min(32, duration_by_item[str(item_id)])))

    citizen_count = int(np.sum(base_sources == 0))
    contextual_count = int(np.sum(base_sources == 1))
    if citizen_count != len(citizen_ids) or contextual_count != len(contextual_ids):
        raise ValueError("base pool source ordering changed")
    signer_ids = np.asarray(citizen_signers + contextual_signers + semlex_signers.tolist())
    observed_frames = np.asarray(citizen_frames + contextual_frames, dtype=np.int16)
    observed_frames = np.concatenate((observed_frames, semlex_frames))
    source_codes = np.concatenate((base_sources, np.full(len(semlex_targets), 2, dtype=np.uint8)))
    features = np.concatenate((base_features, semlex_features), axis=0)
    targets = np.concatenate((base_targets, semlex_targets), axis=0)
    if not (len(features) == len(targets) == len(source_codes) == len(signer_ids) == len(observed_frames)):
        raise ValueError("multi-voice arrays are not aligned")

    metadata = {
        "format": "slt_stage2_multivoice_train_only_pool_v17",
        "format_version": 1,
        "source_split": "citizen_semlex_asllrp_train_only_replay",
        "source_code_map": {
            "0": "citizen_official_train_only",
            "1": "asllrp_contextual_train_only",
            "2": "semlex_official_train_only",
        },
        "transition_scale_source_code": 1,
        "stage1_checkpoint_sha256": base_metadata["stage1_checkpoint_sha256"],
        "base_pool": args.base_pool.as_posix(),
        "base_pool_sha256": sha256(args.base_pool),
        "semlex_pool": args.semlex_pool.as_posix(),
        "semlex_pool_sha256": sha256(args.semlex_pool),
        "items": len(targets),
        "source_items": {
            "citizen": citizen_count,
            "asllrp": contextual_count,
            "semlex": len(semlex_targets),
        },
        "signer_voices": len(set(signer_ids.tolist())),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        frozen_features=features,
        target_indices=targets,
        source_codes=source_codes,
        signer_ids=signer_ids,
        observed_frames=observed_frames,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(args.output)
    result = {
        "output": args.output.as_posix(),
        "output_sha256": sha256(args.output),
        **{key: metadata[key] for key in ("items", "source_items", "signer_voices")},
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pool", type=Path, default=Path("data/local/stage2_v17_synthetic/train_only_replay_pool_v2.npz"))
    parser.add_argument("--citizen-pool", type=Path, default=Path("data/local/stage2_v17_synthetic/citizen_train_isolated_pool.npz"))
    parser.add_argument("--semlex-pool", type=Path, default=Path("data/local/stage2_v17_synthetic/semlex_train_isolated_pool.npz"))
    parser.add_argument("--citizen-provenance", type=Path, default=Path("data/local/citizen100_v17/provenance.csv"))
    parser.add_argument("--citizen-landmark-root", type=Path, default=Path("data/local/citizen100_v17/landmarks"))
    parser.add_argument("--contextual-manifest", type=Path, default=Path("active/v17/stage2_asllrp_segmented_train_manifest_v17.json"))
    parser.add_argument("--contextual-multimodal-root", type=Path, default=Path("data/local/stage2_v17_asllrp_segmented_train_multimodal"))
    parser.add_argument("--output", type=Path, default=Path("data/local/stage2_v17_synthetic/train_only_multivoice_pool_v3.npz"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_multivoice/pool.json"))
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
