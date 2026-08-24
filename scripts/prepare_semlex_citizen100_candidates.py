#!/usr/bin/env python3
"""Build an exact ASL-LEX-linked, signer-diverse SemLex train-only plan.

This script reads metadata only.  It never treats English-name similarity as lexical
equivalence: Citizen's pinned ASL-LEX code is resolved through the official ASL-LEX
2.0 table and then matched to SemLex's ``asllex`` label.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_asllex_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def choose_distinct_signers(rows: list[dict[str, str]], cap: int) -> list[dict[str, str]]:
    """Choose one clip per official SemLex signer, preferring ordinary durations."""
    unique_videos: dict[str, dict[str, str]] = {}
    for row in rows:
        unique_videos.setdefault(row["video_id"], row)
    ranked = sorted(
        unique_videos.values(),
        key=lambda row: (
            abs(float(row["duration"]) - 1600.0),
            row["signer_id"],
            row["video_id"],
        ),
    )
    selected: list[dict[str, str]] = []
    seen_signers: set[str] = set()
    for row in ranked:
        signer = row["signer_id"]
        if signer in seen_signers:
            continue
        selected.append(row)
        seen_signers.add(signer)
        if cap and len(selected) == cap:
            break
    return selected


def build_selection(
    manifest: dict[str, object],
    asllex_rows: list[dict[str, str]],
    semlex_rows: list[dict[str, str]],
    cap_per_class: int,
    split: str = "train",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    asllex_by_code = {row["Code"]: row for row in asllex_rows}
    semlex_by_label: dict[str, list[dict[str, str]]] = {}
    for row in semlex_rows:
        if row["split"] != split or row["label_type"] != "asllex":
            continue
        duration = float(row["duration"])
        if not 400.0 <= duration <= 6000.0:
            continue
        semlex_by_label.setdefault(normalize_asllex_label(row["label"]), []).append(row)

    videos: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    claimed_video_ids: dict[str, str] = {}
    for item in manifest["classes"]:  # type: ignore[index]
        canonical = str(item["canonical_label"])
        code = str(item["citizen_asl_lex_code"])
        asllex = asllex_by_code.get(code)
        if asllex is None:
            raise ValueError(f"Citizen ASL-LEX code absent from official table: {code}")
        entry_id = asllex["EntryID"]
        candidates = semlex_by_label.get(normalize_asllex_label(entry_id), [])
        selected = choose_distinct_signers(candidates, cap_per_class)
        class_rows.append(
            {
                "canonical_label": canonical,
                "citizen_raw_gloss": item["citizen_raw_gloss"],
                "citizen_asl_lex_code": code,
                "asllex_entry_id": entry_id,
                "available_unique_videos": len({row["video_id"] for row in candidates}),
                "available_signers": len({row["signer_id"] for row in candidates}),
                "selected_clips": len(selected),
                "selected_signers": len({row["signer_id"] for row in selected}),
            }
        )
        for row in selected:
            previous = claimed_video_ids.setdefault(row["video_id"], canonical)
            if previous != canonical:
                raise ValueError(
                    f"SemLex video {row['video_id']} maps to both {previous} and {canonical}"
                )
            videos.append(
                {
                    "canonical_label": canonical,
                    "citizen_raw_gloss": item["citizen_raw_gloss"],
                    "citizen_asl_lex_code": code,
                    "asllex_entry_id": entry_id,
                    "semlex_video_id": row["video_id"],
                    "semlex_signer_id": row["signer_id"],
                    "duration_ms": float(row["duration"]),
                    "semlex_split": split,
                    "semlex_label_type": "asllex",
                    "semlex_label": row["label"],
                    "archive_member": f"./{split}/{row['video_id']}.webm",
                    "training_eligible": False,
                }
            )
    return videos, class_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument(
        "--semlex-metadata",
        type=Path,
        default=Path("data/local/dataset_metadata/semlex_official/semlex_metadata.csv"),
    )
    parser.add_argument(
        "--asllex-metadata",
        type=Path,
        default=Path("data/local/dataset_metadata/asllex2_official/signdata.csv"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/local/semlex_citizen100_train_audit")
    )
    parser.add_argument("--cap-per-class", type=int, default=5)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    args = parser.parse_args()
    if args.cap_per_class < 0:
        raise ValueError("cap-per-class must be non-negative")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with args.asllex_metadata.open(encoding="latin-1", newline="") as handle:
        asllex_rows = list(csv.DictReader(handle))
    with args.semlex_metadata.open(encoding="utf-8-sig", newline="") as handle:
        semlex_rows = list(csv.DictReader(handle))
    videos, classes = build_selection(
        manifest, asllex_rows, semlex_rows, args.cap_per_class, args.split
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": f"exact-ASL-LEX SemLex {args.split} acquisition plan",
        "training_eligible": False,
        "split": args.split,
        "split_eligibility": (
            "train_only_after_decode_v17_quality_and_mismatch_review"
            if args.split == "train"
            else "evaluation_only_never_training"
        ),
        "license": "SemLex CC BY-NC-SA; links supplied after named-user access acceptance",
        "manifest_sha256": sha256(args.manifest),
        "semlex_metadata_sha256": sha256(args.semlex_metadata),
        "asllex_metadata_sha256": sha256(args.asllex_metadata),
        "cap_per_class": args.cap_per_class or None,
        "selected_clips": len(videos),
        "selected_classes": sum(row["selected_clips"] > 0 for row in classes),
        "selected_signers": len({row["semlex_signer_id"] for row in videos}),
        "classes": classes,
        "videos": videos,
    }
    (args.output_root / "selection_plan.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    members = "\n".join(str(row["archive_member"]) for row in videos) + "\n"
    (args.output_root / "archive_members.txt").write_text(members, encoding="utf-8")
    print(
        json.dumps(
            {
                "clips": payload["selected_clips"],
                "classes": payload["selected_classes"],
                "signers": payload["selected_signers"],
                "missing_classes": [
                    row["canonical_label"] for row in classes if row["selected_clips"] == 0
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
