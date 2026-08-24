#!/usr/bin/env python3
"""Build train-only, signer-coherent, timing-aware Stage-2 compositions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
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
    with np.load(args.pool, allow_pickle=False) as payload:
        targets = payload["target_indices"].astype(np.int64)
        sources = payload["source_codes"].astype(np.uint8)
        metadata = json.loads(str(payload["metadata_json"]))
    if metadata.get("source_split") != "citizen_asllrp_train_only_replay":
        raise ValueError("signer voice synthesis requires the locked mixed replay pool")
    manifest = json.loads(args.contextual_manifest.read_text())
    row_by_item = {str(row["source_item_id"]): row for row in manifest["rows"]}
    contextual_ids = list(metadata["asllrp_contextual_item_ids"])
    contextual_pool_indices = np.flatnonzero(sources == 1).tolist()
    if len(contextual_ids) != len(contextual_pool_indices):
        raise ValueError("contextual pool metadata is no longer aligned")

    decoded_duration_by_item: dict[str, int] = {}
    for path in sorted(args.contextual_multimodal_root.rglob("*.stage2_rgb_v17.npz")):
        with np.load(path, allow_pickle=False) as payload:
            item_metadata = json.loads(str(payload["metadata_json"]))
            ranges = payload["window_source_ranges"].astype(np.int64)
        item_id = str(item_metadata["source_item_id"])
        if len(ranges) == 1:
            decoded_duration_by_item[item_id] = int(ranges[0, 1] - ranges[0, 0])

    by_signer_class: dict[str, dict[int, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    signer_pool_indices: dict[str, list[int]] = defaultdict(list)
    for pool_index, item_id in zip(contextual_pool_indices, contextual_ids):
        row = row_by_item.get(str(item_id))
        if row is None or row.get("role") != "train":
            raise ValueError(f"missing train-only signer metadata: {item_id}")
        signer = str(row["signer_id"])
        label = int(targets[pool_index])
        if str(item_id) not in decoded_duration_by_item:
            raise ValueError(f"{item_id}: missing authoritative one-window decoded duration")
        duration = decoded_duration_by_item[str(item_id)]
        if not 4 <= duration <= 32:
            raise ValueError(f"{item_id}: unexpected one-window duration {duration}")
        by_signer_class[signer][label].append((pool_index, duration))
        signer_pool_indices[signer].append(pool_index)
    if len(by_signer_class) < 3:
        raise ValueError("at least three train-only signer voices are required")

    citizen_by_class: dict[int, list[int]] = defaultdict(list)
    for index in np.flatnonzero(sources == 0):
        citizen_by_class[int(targets[index])].append(int(index))
    if sorted(citizen_by_class) != list(range(100)):
        raise ValueError("Citizen vocabulary coverage changed")

    rng = np.random.default_rng(args.seed)
    rows = []
    signers = sorted(by_signer_class)
    signer_occurrences = Counter()
    class_occurrences = Counter()
    for sequence_index in range(args.asllrp_sequences):
        signer = signers[sequence_index % len(signers)]
        available = sorted(by_signer_class[signer])
        length = 3 if rng.random() < args.three_sign_probability else 2
        labels = [available[(sequence_index // len(signers)) % len(available)]]
        while len(labels) < length:
            candidate = int(rng.choice(available))
            if candidate != labels[-1]:
                labels.append(candidate)
        selected = [by_signer_class[signer][label][int(rng.integers(
            len(by_signer_class[signer][label])
        ))] for label in labels]
        pool_indices = [value[0] for value in selected]
        durations = [value[1] for value in selected]
        rows.append({
            "sequence_id": f"synthetic_asllrp_signer_voice_{sequence_index:05d}",
            "source": "synthetic_asllrp_contextual_train",
            "target_indices": labels,
            "pool_indices": pool_indices,
            "signer_voice_synthesis": {
                "signer_id": signer,
                "token_duration_frames": durations,
                "context_frames": args.context_frames,
                "bridge_frames": args.bridge_frames,
                "max_trim_frames": args.max_trim_frames,
                "minimum_keep_frames": args.minimum_keep_frames,
            },
        })
        signer_occurrences[signer] += 1
        class_occurrences.update(labels)

    for sequence_index in range(args.citizen_sequences):
        length = int(rng.integers(2, 6))
        labels = [sequence_index % 100]
        while len(labels) < length:
            candidate = int(rng.integers(100))
            if candidate != labels[-1]:
                labels.append(candidate)
        pool_indices = [int(rng.choice(citizen_by_class[label])) for label in labels]
        rows.append({
            "sequence_id": f"synthetic_citizen_train_{sequence_index:05d}",
            "source": "synthetic_citizen_train",
            "target_indices": labels,
            "pool_indices": pool_indices,
        })
    rng.shuffle(rows)

    output = {
        "format": "slt_stage2_signer_voice_composition_plan_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pool": args.pool.as_posix(),
        "pool_sha256": sha256(args.pool),
        "contextual_manifest": args.contextual_manifest.as_posix(),
        "contextual_manifest_sha256": sha256(args.contextual_manifest),
        "contextual_multimodal_root": args.contextual_multimodal_root.as_posix(),
        "seed": args.seed,
        "sequence_count": len(rows),
        "sequence_counts": {
            "synthetic_citizen_train": args.citizen_sequences,
            "synthetic_asllrp_contextual_train": args.asllrp_sequences,
        },
        "signer_pool_indices": {key: value for key, value in sorted(signer_pool_indices.items())},
        "signer_sequence_counts": dict(sorted(signer_occurrences.items())),
        "asllrp_class_occurrences": {
            str(key): value for key, value in sorted(class_occurrences.items())
        },
        "synthesis_contract": (
            "AI-selected locked gloss sequences; every ASLLRP sequence uses real trajectories "
            "from exactly one training signer; observed token timing is restored; monotonic "
            "boundary frame selection plus a short interpolated bridge approximates coarticulation; "
            "signer-specific endpoint statistics provide outer context; no held-out signer data"
        ),
        "research_basis": [
            "Saunders et al. CVPR 2022 FS-Net monotonic frame selection for coarticulation",
            "Joshi et al. EMNLP 2025 PoseStitch-SLT",
            "Yang et al. CVIU 2024 CombSLR feature-level composition",
        ],
        "rows": rows,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
        "two_m_flores_devtest_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    report = {
        "output": args.output.as_posix(),
        "output_sha256": sha256(args.output),
        "sequences": len(rows),
        "signer_sequence_counts": output["signer_sequence_counts"],
        "asllrp_classes": len(class_occurrences),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool", type=Path,
        default=Path("data/local/stage2_v17_synthetic/train_only_replay_pool_v2.npz"),
    )
    parser.add_argument(
        "--contextual-manifest", type=Path,
        default=Path("active/v17/stage2_asllrp_segmented_train_manifest_v17.json"),
    )
    parser.add_argument(
        "--contextual-multimodal-root", type=Path,
        default=Path("data/local/stage2_v17_asllrp_segmented_train_multimodal"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("active/v17/stage2_signer_voice_plan_v17.json")
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_signer_voice/plan.json"),
    )
    parser.add_argument("--citizen-sequences", type=int, default=6000)
    parser.add_argument("--asllrp-sequences", type=int, default=6000)
    parser.add_argument("--three-sign-probability", type=float, default=0.10)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--bridge-frames", type=int, default=2)
    parser.add_argument("--max-trim-frames", type=int, default=3)
    parser.add_argument("--minimum-keep-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3701)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
