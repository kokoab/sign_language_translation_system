#!/usr/bin/env python3
"""Materialize the train-only 67-voice pool in raw v17 landmark space."""

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

from active.v17.train_transition_inpainter_v17 import sha256


def feature_row(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        value = payload["features"].astype(np.float16)
    if value.shape != (32, 61, 5) or not np.isfinite(value).all():
        raise ValueError(f"invalid landmark archive: {path}")
    return value


def run(args: argparse.Namespace) -> dict[str, object]:
    with np.load(args.aligned_pool, allow_pickle=False) as payload:
        targets = payload["target_indices"].astype(np.int64)
        sources = payload["source_codes"].astype(np.uint8)
        signers = payload["signer_ids"].astype(str)
        observed_frames = payload["observed_frames"].astype(np.int16)
        aligned_metadata = json.loads(str(payload["metadata_json"]))
    if aligned_metadata.get("source_split") != "citizen_semlex_asllrp_train_only_replay":
        raise ValueError("input is not the locked train-only aligned pool")

    with np.load(args.citizen_pool, allow_pickle=False) as payload:
        citizen_metadata = json.loads(str(payload["metadata_json"]))
    citizen_items = [str(value) for value in citizen_metadata["item_ids"]]
    with np.load(args.base_pool, allow_pickle=False) as payload:
        base_metadata = json.loads(str(payload["metadata_json"]))
    asllrp_items = [str(value) for value in base_metadata["asllrp_contextual_item_ids"]]
    with np.load(args.semlex_pool, allow_pickle=False) as payload:
        semlex_metadata = json.loads(str(payload["metadata_json"]))
    semlex_items = [str(value) for value in semlex_metadata["item_ids"]]

    citizen_count = int((sources == 0).sum())
    asllrp_count = int((sources == 1).sum())
    semlex_count = int((sources == 2).sum())
    if (citizen_count, asllrp_count, semlex_count) != (
        len(citizen_items), len(asllrp_items), len(semlex_items)
    ):
        raise ValueError("source counts no longer align with item inventories")

    contextual = {}
    for path in sorted(args.asllrp_root.rglob("*.stage2_rgb_v17.npz")):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
            landmarks = payload["landmarks"]
            valid = payload["landmark_window_valid"]
        if len(landmarks) == 1 and bool(valid[0]):
            contextual[str(metadata["source_item_id"])] = path
    if set(asllrp_items) - set(contextual):
        missing = sorted(set(asllrp_items) - set(contextual))[:5]
        raise ValueError(f"missing ASLLRP landmark items: {missing}")

    landmarks = []
    item_ids = []
    archive_paths = []
    for item in citizen_items:
        label, stem = item.split("/", 1)
        path = args.citizen_root / label / f"{stem}.v17.npz"
        landmarks.append(feature_row(path))
        item_ids.append(f"citizen:{item}")
        archive_paths.append(path.as_posix())
    for item in asllrp_items:
        path = contextual[item]
        with np.load(path, allow_pickle=False) as payload:
            value = payload["landmarks"][0].astype(np.float16)
        if value.shape != (32, 61, 5) or not np.isfinite(value).all():
            raise ValueError(f"invalid ASLLRP landmarks: {path}")
        landmarks.append(value)
        item_ids.append(f"asllrp:{item}")
        archive_paths.append(path.as_posix())
    for item in semlex_items:
        label, stem = item.split("/", 1)
        path = args.semlex_root / label / f"{stem}.v17.npz"
        landmarks.append(feature_row(path))
        item_ids.append(f"semlex:{item}")
        archive_paths.append(path.as_posix())

    values = np.stack(landmarks).astype(np.float16)
    if not (
        len(values) == len(targets) == len(sources) == len(signers)
        == len(observed_frames) == len(item_ids)
    ):
        raise ValueError("raw signing-voice arrays are not aligned")
    if not np.array_equal((values[..., 3] > 0), (values[..., 3] == 1)):
        raise ValueError("presence channel is not binary")

    metadata = {
        "format": "slt_signing_voice_train_only_landmark_pool_v17",
        "version": 1,
        "source_split": "citizen_semlex_asllrp_train_only",
        "aligned_pool": args.aligned_pool.as_posix(),
        "aligned_pool_sha256": sha256(args.aligned_pool),
        "items": len(values),
        "voices": len(set(signers.tolist())),
        "eligible_multiclass_voices": sum(
            len(set(targets[signers == signer].tolist())) >= 2
            for signer in set(signers.tolist())
        ),
        "source_items": {
            "citizen": citizen_count,
            "asllrp": asllrp_count,
            "semlex": semlex_count,
        },
        "archive_paths": archive_paths,
        "class_count": len(set(targets.tolist())),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        landmarks=values,
        target_indices=targets,
        source_codes=sources,
        signer_ids=signers,
        observed_frames=observed_frames,
        item_ids=np.asarray(item_ids),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(args.output)
    report = {
        "output": args.output.as_posix(),
        "output_sha256": sha256(args.output),
        **{key: metadata[key] for key in (
            "items", "voices", "eligible_multiclass_voices", "source_items", "class_count",
            "citizen_test_accessed", "semlex_test_accessed", "local_test_accessed",
            "held_out_validation_signer_accessed",
        )},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aligned-pool", type=Path,
        default=Path("data/local/stage2_v17_synthetic/train_only_multivoice_pool_v3.npz"),
    )
    parser.add_argument(
        "--base-pool", type=Path,
        default=Path("data/local/stage2_v17_synthetic/train_only_replay_pool_v2.npz"),
    )
    parser.add_argument(
        "--citizen-pool", type=Path,
        default=Path("data/local/stage2_v17_synthetic/citizen_train_isolated_pool.npz"),
    )
    parser.add_argument(
        "--semlex-pool", type=Path,
        default=Path("data/local/stage2_v17_synthetic/semlex_train_isolated_pool.npz"),
    )
    parser.add_argument(
        "--citizen-root", type=Path,
        default=Path("data/local/citizen100_v17/landmarks/train"),
    )
    parser.add_argument(
        "--semlex-root", type=Path,
        default=Path("data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17"),
    )
    parser.add_argument(
        "--asllrp-root", type=Path,
        default=Path("data/local/stage2_v17_asllrp_segmented_train_multimodal"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/local/signing_voice_v17/train_only_landmark_pool.npz"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/signing_voice_v17/landmark_pool.json"),
    )
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
