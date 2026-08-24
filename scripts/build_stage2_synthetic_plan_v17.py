#!/usr/bin/env python3
"""Build deterministic train-only synthetic composition rows from frozen sign tokens."""

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
        targets = payload["target_indices"].astype(int)
        metadata = json.loads(str(payload["metadata_json"]))
    if metadata.get("source_split") != "citizen_official_train_only":
        raise ValueError("synthetic composition requires the Citizen training-only pool")
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)
    if sorted(by_class) != list(range(100)):
        raise ValueError("isolated pool does not cover all 100 classes")
    rng = np.random.default_rng(args.seed)
    rows = []
    class_occurrences = Counter()
    for sequence_index in range(args.sequences):
        length = int(rng.integers(args.minimum_length, args.maximum_length + 1))
        first = sequence_index % 100
        labels = [first]
        while len(labels) < length:
            candidate = int(rng.integers(0, 100))
            if candidate != labels[-1]:
                labels.append(candidate)
        pool_indices = [int(rng.choice(by_class[label])) for label in labels]
        rows.append({
            "sequence_id": f"synthetic_{sequence_index:05d}",
            "target_indices": labels,
            "pool_indices": pool_indices,
        })
        class_occurrences.update(labels)
    payload = {
        "format": "slt_stage2_synthetic_composition_plan_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pool": args.pool.as_posix(),
        "pool_sha256": sha256(args.pool),
        "seed": args.seed,
        "sequence_count": len(rows),
        "length_range": [args.minimum_length, args.maximum_length],
        "class_occurrences": {str(index): class_occurrences[index] for index in range(100)},
        "composition_contract": "concatenate frozen Citizen-train-only isolated temporal windows; no pixel synthesis",
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
        "minimum_class_occurrences": min(class_occurrences.values()),
        "maximum_class_occurrences": max(class_occurrences.values()),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool", type=Path,
        default=Path("data/local/stage2_v17_synthetic/citizen_train_isolated_pool.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("active/v17/stage2_synthetic_plan_v17.json"),
    )
    parser.add_argument("--sequences", type=int, default=10000)
    parser.add_argument("--minimum-length", type=int, default=2)
    parser.add_argument("--maximum-length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1701)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
