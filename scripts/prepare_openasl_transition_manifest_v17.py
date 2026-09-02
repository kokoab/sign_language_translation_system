#!/usr/bin/env python3
"""Convert the retained OpenASL acquisition state into a v17 motion manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def run(args: argparse.Namespace) -> dict[str, object]:
    state = json.loads(args.state.read_text())
    rows = []
    for item in state["completed"].values():
        rows.append({
            "source_item_id": f"openasl:{item['vid']}",
            "source": "openasl_unlabeled_continuous",
            "role": item["role"],
            "video_path": item["path"],
            "video_sha256": item["sha256"],
            "source_group": item["channel_id"],
            "signer_id": item["voice_proxy"],
            "target_indices": [],
            "target_sequence": [],
            "duration_seconds": item["duration"],
            "license": "OpenASL research data",
        })
    payload = {
        "format": "continuous_unlabeled_transition_manifest_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "motion-only continuous domain reference",
        "row_count": len(rows),
        "rows": sorted(rows, key=lambda row: row["source_item_id"]),
        "openasl_test_video_accessed": False,
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
    parser.add_argument("--state", type=Path, default=Path("data/local/openasl_transition_subset_v17/acquisition_state.json"))
    parser.add_argument("--output", type=Path, default=Path("active/v17/openasl_transition_manifest_v17.json"))
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2))
