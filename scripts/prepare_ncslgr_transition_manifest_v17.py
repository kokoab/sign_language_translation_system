#!/usr/bin/env python3
"""Prepare the acquired NCSLGR utterances for genuine transition self-supervision."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any


def duration_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path.as_posix(),
        ],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.source_manifest.read_text())
    if source.get("format") != "slt_v17_ncslgr_continuous_source_manifest":
        raise ValueError("unexpected NCSLGR source manifest")
    rows = []
    for item in source["items"]:
        video = Path(item["video_path"])
        if not video.is_file():
            raise FileNotFoundError(video)
        signer = f"ncslgr:{item['participant_id']}"
        rows.append({
            "source_item_id": f"ncslgr:{item['collection']}:{item['source_id']}",
            "source": "ncslgr_public_continuous",
            "role": "train",
            "video_path": video.as_posix(),
            "video_sha256": item["video_sha256"],
            "source_group": f"ncslgr:{item['collection']}",
            "signer_id": signer,
            "target_indices": [],
            "target_sequence": [],
            "duration_seconds": duration_seconds(video),
            "zero_lip_nodes": False,
            "lip_supervision": "genuine_video_observed",
            "license": "ASLLRP Sign Bank Terms of Use",
            "source_page": item["page_url"],
            "annotation_path": item["annotation_path"],
            "annotation_sha256": item["annotation_sha256"],
        })
    payload = {
        "format": "continuous_unlabeled_transition_manifest_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "self-supervised masked transition learning; no CTC targets",
        "source_manifest": args.source_manifest.as_posix(),
        "row_count": len(rows),
        "signer_counts": dict(sorted(Counter(row["signer_id"] for row in rows).items())),
        "duration_hours": sum(row["duration_seconds"] for row in rows) / 3600.0,
        "rows": rows,
        "test_evaluated": False,
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
        "--source-manifest", type=Path,
        default=Path("data/local/ncslgr_continuous_v17_source/manifest.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("active/v17/ncslgr_transition_manifest_v17.json"),
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
