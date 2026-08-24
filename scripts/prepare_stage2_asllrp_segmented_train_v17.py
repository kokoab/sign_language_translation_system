#!/usr/bin/env python3
"""Prepare ASLLRP train-signer contextual sign segments for Stage-2 replay only."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


HELD_OUT_SIGNERS = {"JONATHAN"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace):
    source = json.loads(args.source_manifest.read_text())
    vocabulary = json.loads(args.vocabulary_manifest.read_text())
    label_to_index = {
        str(row["canonical_label"]): int(row["class_index"])
        for row in vocabulary["classes"]
    }
    rows = []
    exclusions = Counter()
    for item in source["videos"]:
        if item["split_role"] != "train_candidate":
            exclusions["external_evaluation_reserved"] += 1
            continue
        if item["signer_id"] in HELD_OUT_SIGNERS:
            exclusions["held_out_validation_signer"] += 1
            continue
        if int(item["frame_count"]) < 4:
            exclusions["fewer_than_four_frames"] += 1
            continue
        path = Path(item["path"])
        if not path.exists() or sha256(path) != item["sha256"]:
            raise ValueError(f"{path}: missing or changed")
        label = item["canonical_label"]
        if label not in label_to_index:
            raise ValueError(f"unexpected label {label}")
        rows.append({
            "source": "asllrp_segmented_train",
            "role": "train",
            "source_item_id": f"asllrp_segmented:{item['sign_video_filename']}",
            "video_path": item["path"],
            "video_sha256": item["sha256"],
            "parent_video_sha256": None,
            "source_group": f"asllrp:signer:{item['signer_id']}",
            "signer_id": item["signer_id"],
            "signer_metadata_available": True,
            "zero_lip_nodes": False,
            "lip_supervision": "available",
            "target_sequence": [label],
            "target_indices": [label_to_index[label]],
            "target_token_count": 1,
            "frame_count": int(item["frame_count"]),
            "duration_seconds": float(item["duration_seconds"]),
        })
    payload = {
        "format": "slt_v17_stage2_asllrp_segmented_train_manifest",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": args.source_manifest.as_posix(),
        "source_manifest_sha256": sha256(args.source_manifest),
        "vocabulary_manifest": args.vocabulary_manifest.as_posix(),
        "vocabulary_manifest_sha256": sha256(args.vocabulary_manifest),
        "held_out_signers": sorted(HELD_OUT_SIGNERS),
        "rows": rows,
        "exclusions": dict(exclusions),
        "classes": len(set(row["target_sequence"][0] for row in rows)),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    result = {
        "output": args.output.as_posix(),
        "output_sha256": sha256(args.output),
        "rows": len(rows),
        "classes": payload["classes"],
        "signers": sorted(set(row["signer_id"] for row in rows)),
        "exclusions": payload["exclusions"],
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest", type=Path,
        default=Path("data/local/asllrp_segmented_citizen100_v17/manifest.json"),
    )
    parser.add_argument(
        "--vocabulary-manifest", type=Path,
        default=Path("active/v17/citizen100_manifest.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("active/v17/stage2_asllrp_segmented_train_manifest_v17.json"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_asllrp_segmented_train/manifest.json"),
    )
    return parser


def main():
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
