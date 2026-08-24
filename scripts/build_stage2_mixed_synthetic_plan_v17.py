#!/usr/bin/env python3
"""Build deterministic Citizen/ASLLRP train-only composition sequences."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCES = {0: "synthetic_citizen_train", 1: "synthetic_asllrp_contextual_train"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    with np.load(args.pool, allow_pickle=False) as payload:
        targets = payload["target_indices"].astype(int)
        source_codes = payload["source_codes"].astype(int)
        metadata = json.loads(str(payload["metadata_json"]))
    if metadata.get("source_split") != "citizen_asllrp_train_only_replay":
        raise ValueError("mixed composition requires the locked train-only replay pool")
    if targets.shape != source_codes.shape:
        raise ValueError("pool target/source shapes differ")
    by_source_class: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (source, target) in enumerate(zip(source_codes, targets)):
        if int(source) not in SOURCES:
            raise ValueError(f"unsupported source code {source}")
        by_source_class[int(source)][int(target)].append(index)
    if sorted(by_source_class[0]) != list(range(100)):
        raise ValueError("Citizen part does not cover all locked classes")

    rng = np.random.default_rng(args.seed)
    rows = []
    occurrences: dict[str, Counter] = {value: Counter() for value in SOURCES.values()}
    sequence_counts = {0: args.citizen_sequences, 1: args.asllrp_sequences}
    for source_code, count in sequence_counts.items():
        available = sorted(by_source_class[source_code])
        source_name = SOURCES[source_code]
        for sequence_index in range(count):
            length = int(rng.integers(args.minimum_length, args.maximum_length + 1))
            labels = [available[sequence_index % len(available)]]
            while len(labels) < length:
                candidate = int(rng.choice(available))
                if candidate != labels[-1]:
                    labels.append(candidate)
            pool_indices = [
                int(rng.choice(by_source_class[source_code][label])) for label in labels
            ]
            rows.append({
                "sequence_id": f"{source_name}_{sequence_index:05d}",
                "source": source_name,
                "target_indices": labels,
                "pool_indices": pool_indices,
                **({"leading_padding_frames": int(rng.integers(0, 32))}
                   if args.window_phase_augmentation else {}),
            })
            occurrences[source_name].update(labels)
    rng.shuffle(rows)
    payload = {
        "format": "slt_stage2_mixed_synthetic_composition_plan_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pool": args.pool.as_posix(),
        "pool_sha256": sha256(args.pool),
        "seed": args.seed,
        "sequence_count": len(rows),
        "sequence_counts": {SOURCES[key]: value for key, value in sequence_counts.items()},
        "length_range": [args.minimum_length, args.maximum_length],
        "class_occurrences": {
            source: {str(label): count for label, count in sorted(values.items())}
            for source, values in occurrences.items()
        },
        "composition_contract": (
            "concatenate frozen isolated temporal windows within one train-only source domain; "
            "optionally shift the concatenated stream across arbitrary 32-frame boundaries; "
            "no validation/test rows and no pixel synthesis"
        ),
        "window_phase_augmentation": bool(args.window_phase_augmentation),
        "rows": rows,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return {
        "output": args.output.as_posix(),
        "output_sha256": sha256(args.output),
        "sequences": len(rows),
        "sequence_counts": payload["sequence_counts"],
        "citizen_minimum_class_occurrences": min(occurrences[SOURCES[0]].values()),
        "asllrp_minimum_class_occurrences": min(occurrences[SOURCES[1]].values()),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool", type=Path,
        default=Path("data/local/stage2_v17_synthetic/train_only_replay_pool_v2.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("active/v17/stage2_mixed_synthetic_plan_v17.json"),
    )
    parser.add_argument("--citizen-sequences", type=int, default=6000)
    parser.add_argument("--asllrp-sequences", type=int, default=6000)
    parser.add_argument("--minimum-length", type=int, default=2)
    parser.add_argument("--maximum-length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--window-phase-augmentation", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
