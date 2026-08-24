#!/usr/bin/env python3
"""Freeze the acquired channel-diverse YouTube-ASL clips into a v17 manifest."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def signer_id(channel: str) -> str:
    return "youtube_asl_channel:" + hashlib.sha256(channel.encode()).hexdigest()[:16]


def run(args: argparse.Namespace) -> dict[str, object]:
    state = json.loads(args.state.read_text())
    if state.get("format") != "slt_youtube_asl_transition_voice_acquisition_v17":
        raise ValueError("unexpected acquisition state")
    rows = []
    for channel, acquired in state["completed"].items():
        video_path = Path(acquired["path"])
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        if sha256(video_path) != acquired["sha256"]:
            raise ValueError(f"hash mismatch: {video_path}")
        role = acquired.get("role")
        if role not in {"train", "validation"}:
            raise ValueError(f"unfrozen role for {channel}")
        rows.append({
            "source_item_id": acquired["video_id"],
            "source": "youtube_asl",
            "role": role,
            "signer_id": signer_id(channel),
            "source_channel": channel,
            "source_channel_name": acquired.get("channel"),
            "source_group": acquired["video_id"],
            "video_path": video_path.as_posix(),
            "video_sha256": acquired["sha256"],
            "duration_seconds": acquired["probe"]["duration"],
            "license": (
                "YouTube-ASL official video-ID release; video copyright remains "
                "with each source owner"
            ),
            "segment_start": acquired["segment_start"],
            "segment_end": acquired["segment_end"],
        })
    rows.sort(key=lambda row: (row["role"], row["signer_id"]))
    counts = Counter(str(row["role"]) for row in rows)
    if counts["train"] != state["train_voice_proxies"]:
        raise ValueError("train voice count mismatch")
    if counts["validation"] != state["validation_voice_proxies"]:
        raise ValueError("validation voice count mismatch")
    manifest = {
        "format": "continuous_unlabeled_transition_manifest_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "YouTube-ASL official human-filtered video ID list",
        "source_generation": state["source_generation"],
        "acquisition_state": args.state.as_posix(),
        "acquisition_state_sha256": sha256(args.state),
        "split_policy": state["split_policy"],
        "rows": rows,
        "row_count": len(rows),
        "train_voice_proxies": counts["train"],
        "validation_voice_proxies": counts["validation"],
        "one_clip_per_channel": True,
        "channel_is_signer_proxy_not_verified_identity": True,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "consumed_rit_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state", type=Path,
        default=Path("data/local/youtube_asl_transition_subset_v17/acquisition_state.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("active/v17/youtube_asl_transition_manifest_v17.json"),
    )
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps({
        "manifest": "active/v17/youtube_asl_transition_manifest_v17.json",
        "rows": result["row_count"],
        "train_voices": result["train_voice_proxies"],
        "validation_voices": result["validation_voice_proxies"],
    }, indent=2))


if __name__ == "__main__":
    main()
