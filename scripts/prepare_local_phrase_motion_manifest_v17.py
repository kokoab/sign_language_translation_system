#!/usr/bin/env python3
"""Prepare all nine local phrase families for unlabeled v17 motion learning."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.prepare_stage2_training_manifest_v17 import (
    LOCAL_CAPTURE_BATCH_SIZE,
    birth_time,
    sha256,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    with args.audit_csv.open(newline="", encoding="utf-8") as handle:
        audited = list(csv.DictReader(handle))
    by_phrase: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audited:
        by_phrase[row["phrase"]].append(row)

    rows = []
    for phrase, phrase_rows in sorted(by_phrase.items()):
        if len(phrase_rows) % LOCAL_CAPTURE_BATCH_SIZE:
            raise ValueError(f"{phrase}: incomplete 20-recording capture batch")
        phrase_rows.sort(key=lambda row: (birth_time(Path(row["path"])), row["path"]))
        for ordinal, row in enumerate(phrase_rows):
            path = Path(row["path"])
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise ValueError(f"{path}: missing or changed after source audit")
            recording = ordinal % LOCAL_CAPTURE_BATCH_SIZE
            role = "validation" if recording % 5 == 4 else "train"
            batch = ordinal // LOCAL_CAPTURE_BATCH_SIZE
            rows.append({
                "source_item_id": f"local_motion:{phrase}:{path.stem}",
                "source": "local_phrase_unlabeled_continuous",
                "role": role,
                "video_path": path.as_posix(),
                "video_sha256": row["sha256"],
                "source_group": f"local:{phrase}:capture_batch_{batch:02d}",
                "signer_id": "local:unknown",
                "target_indices": [],
                "target_sequence": [],
                "phrase_prompt": phrase,
                "duration_seconds": float(row["duration_seconds"]),
                "license": "project-owned local recordings",
            })
    counts = Counter((row["phrase_prompt"], row["role"]) for row in rows)
    payload = {
        "format": "continuous_unlabeled_transition_manifest_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "motion-only continuous learning; phrase labels are provenance, not CTC targets",
        "source_audit_csv": args.audit_csv.as_posix(),
        "source_audit_csv_sha256": sha256(args.audit_csv),
        "split_contract": "20-recording capture batches; every fifth repetition is validation; signer identity is unavailable",
        "row_count": len(rows),
        "phrase_role_counts": {
            f"{phrase}:{role}": count
            for (phrase, role), count in sorted(counts.items())
        },
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
    parser.add_argument("--audit-csv", type=Path, default=Path("artifacts/reports/stage2_v17_data_audit/local_videos.csv"))
    parser.add_argument("--output", type=Path, default=Path("active/v17/local_phrase_motion_manifest_v17.json"))
    return parser


if __name__ == "__main__":
    payload = run(build_parser().parse_args())
    print(json.dumps({"rows": payload["row_count"], "counts": payload["phrase_role_counts"]}, indent=2))
