#!/usr/bin/env python3
"""Build a train-only unlabeled transition manifest from acquired How2Sign clips."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = json.loads((args.root / "selection_plan.json").read_text())
    completed_payload = json.loads((args.root / "completed_files.json").read_text())
    completed = {
        str(row["sentence_id"]): row for row in completed_payload["files"]
    }
    if len(completed) != int(plan["row_count"]):
        raise ValueError(
            f"acquisition incomplete: {len(completed)}/{plan['row_count']} clips"
        )
    rows = []
    for source in plan["rows"]:
        sentence_id = str(source["sentence_id"])
        acquired = completed[sentence_id]
        rows.append({
            "source_item_id": f"how2sign:{sentence_id}",
            "source": "how2sign_unlabeled_continuous",
            "role": "train",
            "video_path": (args.root / acquired["path"]).as_posix(),
            "video_sha256": str(acquired["sha256"]),
            "source_group": f"how2sign:signer_{source['signer_id']}",
            "signer_id": f"how2sign:{source['signer_id']}",
            "target_indices": [],
            "target_sequence": [],
            "sentence": str(source["sentence"]),
            "duration_seconds": float(source["duration"]),
            "zero_lip_nodes": False,
            "lip_supervision": "genuine_video_observed",
            "license": plan["source_license"],
            "source_repo": plan["source_repo"],
            "source_revision": plan["source_revision"],
        })
    payload = {
        "format": "how2sign_unlabeled_transition_manifest_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "self-supervised masked transition learning; never Stage-2 CTC targets"
        ),
        "source_plan": (args.root / "selection_plan.json").as_posix(),
        "source_completed_files": (args.root / "completed_files.json").as_posix(),
        "row_count": len(rows),
        "signer_counts": dict(sorted(Counter(row["signer_id"] for row in rows).items())),
        "duration_hours": sum(row["duration_seconds"] for row in rows) / 3600.0,
        "rows": rows,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "consumed_rit_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("data/local/how2sign_transition_subset_v17"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("active/v17/how2sign_transition_manifest_v17.json"),
    )
    return parser


def main() -> None:
    payload = run(build_parser().parse_args())
    print(json.dumps({
        "row_count": payload["row_count"],
        "signer_counts": payload["signer_counts"],
        "duration_hours": payload["duration_hours"],
    }, indent=2))


if __name__ == "__main__":
    main()
