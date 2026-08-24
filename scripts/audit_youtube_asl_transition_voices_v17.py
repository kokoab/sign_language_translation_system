#!/usr/bin/env python3
"""Audit usable channel-diverse YouTube-ASL v17 transition trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.train_transition_inpainter_v17 import landmark_tree_fingerprint, sha256


def run(args: argparse.Namespace) -> dict[str, object]:
    manifest = json.loads(args.manifest.read_text())
    expected = {str(row["signer_id"]): row for row in manifest["rows"]}
    archives = {}
    role_windows: Counter[str] = Counter()
    role_archives: Counter[str] = Counter()
    for path in sorted(args.landmark_root.glob("*/*.transition_landmarks_v17.npz")):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
            valid = payload["window_valid"].astype(np.bool_)
        signer = str(metadata["signer_id"])
        if signer not in expected:
            raise ValueError(f"archive signer absent from manifest: {signer}")
        if signer in archives:
            raise ValueError(f"multiple archives for one channel voice: {signer}")
        row = expected[signer]
        if metadata["video_sha256"] != row["video_sha256"]:
            raise ValueError(f"video hash mismatch: {path}")
        if metadata["role"] != row["role"]:
            raise ValueError(f"role mismatch: {path}")
        archives[signer] = path
        role_archives[row["role"]] += 1
        role_windows[row["role"]] += int(valid.sum())
    archive_count, tree_hash = landmark_tree_fingerprint(args.landmark_root)
    expected_roles = Counter(str(row["role"]) for row in expected.values())
    missing = sorted(set(expected) - set(archives))
    meets_voice_floor = (
        role_archives["train"] >= args.minimum_train_voices
        and role_archives["validation"] >= args.minimum_validation_voices
    )
    report = {
        "format": "youtube_asl_transition_voice_audit_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": args.manifest.as_posix(),
        "manifest_sha256": sha256(args.manifest),
        "landmark_root": args.landmark_root.as_posix(),
        "landmark_archive_count": archive_count,
        "landmark_tree_sha256": tree_hash,
        "expected_voice_proxies": dict(expected_roles),
        "usable_voice_proxies": dict(role_archives),
        "valid_windows": dict(role_windows),
        "minimum_usable_voice_proxies": {
            "train": args.minimum_train_voices,
            "validation": args.minimum_validation_voices,
        },
        "meets_usable_voice_floor": meets_voice_floor,
        "missing_voice_proxy_count": len(missing),
        "missing_voice_proxies": missing,
        "one_archive_per_usable_channel": True,
        "channel_is_signer_proxy_not_verified_identity": True,
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "consumed_rit_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if not meets_voice_floor:
        raise RuntimeError(
            "usable YouTube-ASL voice proxy count is below the generalization floor"
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("active/v17/youtube_asl_transition_manifest_v17.json"),
    )
    parser.add_argument(
        "--landmark-root", type=Path,
        default=Path("data/local/youtube_asl_transition_landmarks_v17"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/youtube_asl_transition_voice_audit_v17.json"),
    )
    parser.add_argument("--minimum-train-voices", type=int, default=80)
    parser.add_argument("--minimum-validation-voices", type=int, default=16)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
