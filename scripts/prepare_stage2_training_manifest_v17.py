#!/usr/bin/env python3
"""Lock leakage-aware v17 Stage-2 train/validation rows.

This script does not decode video.  It joins the audited local phrase inventory and
the already verified ASLLRP contiguous-span manifest, applies the locked 100-class
vocabulary, and writes a hash-pinned preprocessing manifest.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


LOCAL_TARGETS = {
    "GOOD_MORNING": ("GOOD", "MORNING"),
    "HELLO_HOW_YOU": ("HELLO", "HOW", "YOU"),
    "MY_NAME": ("MY", "NAME"),
    "PLEASE_HELP_ME": ("PLEASE", "HELP", "I"),
    "THANKYOU_FRIEND": ("THANKYOU", "FRIEND"),
    "TOMORROW_SCHOOL_GO": ("TOMORROW", "SCHOOL", "GO"),
}
ASLLRP_VALIDATION_SIGNERS = {"JONATHAN"}
MINIMUM_SOURCE_FRAMES_PER_TARGET = 8
LOCAL_CAPTURE_BATCH_SIZE = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def birth_time(path: Path) -> float:
    stat = path.stat()
    return float(getattr(stat, "st_birthtime", stat.st_mtime))


def load_label_map(path: Path) -> tuple[dict[str, int], str]:
    payload = json.loads(path.read_text())
    labels = [str(item["canonical_label"]).upper() for item in payload["classes"]]
    if len(labels) != 100 or len(labels) != len(set(labels)):
        raise ValueError("the locked vocabulary must contain 100 unique labels")
    return {label: index for index, label in enumerate(labels)}, sha256(path)


def local_rows(audit_csv: Path, label_to_index: dict[str, int]) -> list[dict[str, Any]]:
    with audit_csv.open(newline="", encoding="utf-8") as handle:
        audited = list(csv.DictReader(handle))
    by_phrase: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audited:
        if row["phrase"] in LOCAL_TARGETS:
            by_phrase[row["phrase"]].append(row)

    output: list[dict[str, Any]] = []
    for phrase, targets in sorted(LOCAL_TARGETS.items()):
        rows = by_phrase[phrase]
        if not rows or len(rows) % LOCAL_CAPTURE_BATCH_SIZE:
            raise ValueError(f"{phrase}: expected complete 20-recording capture batches")
        rows.sort(key=lambda row: (birth_time(Path(row["path"])), row["path"]))
        for ordinal, row in enumerate(rows):
            path = Path(row["path"])
            if not path.exists() or sha256(path) != row["sha256"]:
                raise ValueError(f"{path}: missing or changed since the source audit")
            capture_batch = ordinal // LOCAL_CAPTURE_BATCH_SIZE
            recording_in_batch = ordinal % LOCAL_CAPTURE_BATCH_SIZE
            enough_frames = int(row["frame_count"]) >= (
                len(targets) * MINIMUM_SOURCE_FRAMES_PER_TARGET
            )
            # Every fifth independently recorded repetition is validation. The same
            # capture batch never has exact duplicate hashes across the split.
            role = (
                "excluded_too_short"
                if not enough_frames
                else "validation"
                if recording_in_batch % 5 == 4
                else "train"
            )
            output.append(
                {
                    "source": "local_phrases",
                    "role": role,
                    "source_item_id": f"local:{phrase}:{path.stem}",
                    "video_path": path.as_posix(),
                    "video_sha256": row["sha256"],
                    "source_group": f"local:{phrase}:capture_batch_{capture_batch:02d}",
                    "signer_id": None,
                    "signer_metadata_available": False,
                    "zero_lip_nodes": True,
                    "lip_supervision": "unavailable_zero_only_four_lip_nodes",
                    "target_sequence": list(targets),
                    "target_indices": [label_to_index[label] for label in targets],
                    "target_token_count": len(targets),
                    "frame_count": int(row["frame_count"]),
                    "duration_seconds": float(row["duration_seconds"]),
                    "capture_batch": capture_batch,
                    "recording_in_batch": recording_in_batch,
                    "minimum_source_frames_required": (
                        len(targets) * MINIMUM_SOURCE_FRAMES_PER_TARGET
                    ),
                }
            )
    return output


def asllrp_rows(path: Path, label_to_index: dict[str, int]) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text())
    output = []
    for span in manifest["spans"]:
        targets = [str(label).upper() for label in span["target_sequence"]]
        missing = [label for label in targets if label not in label_to_index]
        if missing:
            raise ValueError(f"{span['path']}: labels outside locked vocabulary: {missing}")
        source_role = span["split_role"]
        signer = span["signer_id"]
        if source_role == "external_evaluation_reserved":
            role = "external_evaluation_reserved"
        elif signer in ASLLRP_VALIDATION_SIGNERS:
            role = "validation"
        else:
            role = "train"
        output.append(
            {
                "source": "asllrp_contiguous",
                "role": role,
                "source_item_id": (
                    f"asllrp:{span['utterance_video_filename']}:"
                    f"span{int(span['span_index_in_utterance']):02d}"
                ),
                "video_path": span["path"],
                "video_sha256": span["sha256"],
                "parent_video_sha256": span["parent_sha256"],
                "source_group": f"asllrp:signer:{signer}",
                "signer_id": signer,
                "signer_metadata_available": True,
                "zero_lip_nodes": False,
                "lip_supervision": "available",
                "target_sequence": targets,
                "target_indices": [label_to_index[label] for label in targets],
                "target_token_count": len(targets),
                "frame_count": int(span["frames"]),
                "duration_seconds": float(span["duration_seconds"]),
                "source_split_role": source_role,
            }
        )
    return output


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [row["source_item_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source_item_id")
    hashes_by_role: dict[str, set[str]] = defaultdict(set)
    parent_roles: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        hashes_by_role[row["role"]].add(row["video_sha256"])
        parent = row.get("parent_video_sha256")
        if parent:
            parent_roles[parent].add(row["role"])
    active_roles = ("train", "validation", "external_evaluation_reserved")
    for first_index, first in enumerate(active_roles):
        for second in active_roles[first_index + 1 :]:
            overlap = hashes_by_role[first] & hashes_by_role[second]
            if overlap:
                raise ValueError(f"video hash leakage between {first} and {second}")
    leaking_parents = {
        parent: sorted(roles)
        for parent, roles in parent_roles.items()
        if len(roles & set(active_roles)) > 1
    }
    if leaking_parents:
        raise ValueError(f"parent utterance leakage: {leaking_parents}")
    counts = Counter((row["source"], row["role"]) for row in rows)
    tokens = Counter(
        (row["source"], row["role"])
        for row in rows
        for _ in row["target_sequence"]
    )
    return {
        "row_counts": {
            f"{source}:{role}": count
            for (source, role), count in sorted(counts.items())
        },
        "target_token_counts": {
            f"{source}:{role}": count
            for (source, role), count in sorted(tokens.items())
        },
        "active_video_hash_overlap": 0,
        "active_parent_utterance_overlap": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    label_to_index, vocabulary_sha = load_label_map(args.vocabulary_manifest)
    rows = local_rows(args.local_audit_csv, label_to_index)
    rows.extend(asllrp_rows(args.asllrp_manifest, label_to_index))
    audit = validate(rows)
    payload = {
        "format": "slt_v17_stage2_training_manifest",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "vocabulary_manifest": args.vocabulary_manifest.as_posix(),
        "vocabulary_manifest_sha256": vocabulary_sha,
        "class_count": len(label_to_index),
        "label_to_index": label_to_index,
        "local_source_audit_csv": args.local_audit_csv.as_posix(),
        "local_source_audit_csv_sha256": sha256(args.local_audit_csv),
        "asllrp_contiguous_manifest": args.asllrp_manifest.as_posix(),
        "asllrp_contiguous_manifest_sha256": sha256(args.asllrp_manifest),
        "split_contract": {
            "local": (
                "20-recording capture batches; every fifth adequate-length repetition "
                "is validation; signer metadata is unavailable and signer overlap is allowed"
            ),
            "asllrp": (
                "JONATHAN is signer-held-out validation; other ASLLRP train candidates train; "
                "all RIT rows remain external-evaluation reserved"
            ),
            "minimum_source_frames_per_target": MINIMUM_SOURCE_FRAMES_PER_TARGET,
            "food_to_eat_approved": False,
        },
        "audit": audit,
        "rows": rows,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    report = {
        "manifest": args.output.as_posix(),
        "manifest_sha256": sha256(args.output),
        **audit,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vocabulary-manifest", type=Path,
        default=Path("active/v17/citizen100_manifest.json"),
    )
    parser.add_argument(
        "--local-audit-csv", type=Path,
        default=Path("artifacts/reports/stage2_v17_data_audit/local_videos.csv"),
    )
    parser.add_argument(
        "--asllrp-manifest", type=Path,
        default=Path("data/local/asllrp_contiguous_phrases_v17/manifest.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("active/v17/stage2_training_manifest_v17.json"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_training_manifest/audit.json"),
    )
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
