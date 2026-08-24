#!/usr/bin/env python3
"""Selectively download a small ASL Citizen audit subset.

The official ASL Citizen archive is approximately 46 GB. This script uses HTTP
byte ranges to read only chosen ZIP members, records their provenance and
checksums, and leaves official train/validation/test assignments unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from build_ios100_dataset_coverage import (
    ASL_CITIZEN_SIZE,
    ASL_CITIZEN_URL,
    HTTPRangeReader,
)


DEFAULT_SIGNS = (
    "HELLO",
    "THANKYOU",
    "GOODBYE",
    "YOU",
    "WHAT",
    "HELP",
    "LOVE",
    "COME",
    "GOOD",
    "SCHOOL",
    "HOSPITAL",
    "DRINK",
)
SPLITS = ("train", "val", "test")


def normalize_gloss(value: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]", "", value.upper())
    return re.sub(r"(?<=[A-Z])\d+$", "", normalized)


def stable_key(*values: str) -> str:
    return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    labels = {
        label
        for values in config["canonical_labels"].values()
        for label in values
    }
    config["_labels"] = labels
    return config


def citizen_source_keys(config: dict[str, object], canonical: str) -> set[str]:
    aliases = (
        config.get("source_aliases", {})
        .get(canonical, {})
        .get("asl_citizen", [])
    )
    return {normalize_gloss(canonical), *(normalize_gloss(item) for item in aliases)}


def load_rows(cache_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in SPLITS:
        path = cache_dir / f"asl_citizen_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/build_ios100_dataset_coverage.py first."
            )
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "split": split,
                        "participant": row["Participant ID"].strip(),
                        "video": row["Video file"].strip(),
                        "raw_gloss": row["Gloss"].strip(),
                        "lex_code": row["ASL-LEX Code"].strip(),
                        "normalized_gloss": normalize_gloss(row["Gloss"]),
                    }
                )
    return rows


def choose_rows(
    rows: list[dict[str, str]],
    config: dict[str, object],
    signs: list[str],
    per_split: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for canonical in signs:
        if canonical not in config["_labels"]:
            raise ValueError(f"{canonical} is not in the proposed iOS-100 vocabulary")
        source_keys = citizen_source_keys(config, canonical)
        for split in SPLITS:
            candidates = [
                row for row in rows
                if row["split"] == split
                and row["normalized_gloss"] in source_keys
            ]
            if not candidates:
                raise ValueError(f"No ASL Citizen candidates for {canonical}/{split}")

            # Round-robin across raw labels first, then choose distinct people.
            # This exposes lexical-variant problems in a small audit sample.
            by_raw: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in candidates:
                by_raw[row["raw_gloss"]].append(row)
            for raw_gloss, values in by_raw.items():
                values.sort(
                    key=lambda row: stable_key(
                        canonical,
                        split,
                        raw_gloss,
                        row["participant"],
                        row["video"],
                    )
                )

            chosen: list[dict[str, str]] = []
            participants: set[str] = set()
            raw_names = sorted(by_raw, key=lambda raw: stable_key(canonical, split, raw))
            while len(chosen) < per_split:
                added = False
                for raw_gloss in raw_names:
                    for row in by_raw[raw_gloss]:
                        if row["participant"] in participants:
                            continue
                        chosen.append({**row, "canonical_gloss": canonical})
                        participants.add(row["participant"])
                        added = True
                        break
                    if len(chosen) == per_split:
                        break
                if not added:
                    break
            if len(chosen) < per_split:
                raise ValueError(
                    f"Only {len(chosen)} distinct participants available for "
                    f"{canonical}/{split}; requested {per_split}"
                )
            selected.extend(chosen)
    return selected


def build_plan(
    selected: list[dict[str, str]],
    zip_infos: dict[str, zipfile.ZipInfo],
    output_dir: Path,
) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    for row in selected:
        member = f"ASL_Citizen/videos/{row['video']}"
        if member not in zip_infos:
            raise KeyError(f"Archive member not found: {member}")
        info = zip_infos[member]
        destination = (
            output_dir
            / "asl_citizen"
            / row["split"]
            / row["canonical_gloss"]
            / row["video"]
        )
        plan.append(
            {
                **row,
                "source_url": ASL_CITIZEN_URL,
                "archive_member": member,
                "destination": str(destination),
                "uncompressed_bytes": info.file_size,
                "compressed_bytes": info.compress_size,
            }
        )
    return plan


def write_provenance(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("active/v16/ios100_vocabulary_proposal.json"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/local/dataset_metadata"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/local/ios100_audit"),
    )
    parser.add_argument("--per-split", type=int, default=2)
    parser.add_argument("--signs", nargs="+", default=list(DEFAULT_SIGNS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.per_split < 1:
        parser.error("--per-split must be positive")

    signs = [normalize_gloss(sign) for sign in args.signs]
    config = load_config(args.config)
    selected = choose_rows(load_rows(args.cache_dir), config, signs, args.per_split)

    with zipfile.ZipFile(HTTPRangeReader(ASL_CITIZEN_URL, ASL_CITIZEN_SIZE)) as archive:
        zip_infos = {info.filename: info for info in archive.infolist()}
        plan = build_plan(selected, zip_infos, args.output_dir)
        total_compressed = sum(int(row["compressed_bytes"]) for row in plan)
        print(
            f"Planned {len(plan)} videos across {len(signs)} signs; "
            f"compressed transfer is approximately {total_compressed / 1_000_000:.1f} MB"
        )
        if args.dry_run:
            for row in plan:
                print(
                    f"{row['split']:5} {row['canonical_gloss']:10} "
                    f"{row['participant']:>4} {row['raw_gloss']:12} "
                    f"{int(row['compressed_bytes']) / 1_000_000:5.2f} MB"
                )
            return

        completed: list[dict[str, object]] = []
        for index, row in enumerate(plan, start=1):
            destination = Path(str(row["destination"]))
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.stat().st_size == row["uncompressed_bytes"]:
                payload = destination.read_bytes()
                status = "existing"
            else:
                payload = archive.read(str(row["archive_member"]))
                destination.write_bytes(payload)
                status = "downloaded"
            completed.append(
                {
                    **row,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "status": status,
                }
            )
            print(
                f"[{index:02}/{len(plan)}] {status:10} "
                f"{row['split']}/{row['canonical_gloss']}/{row['video']}"
            )

    provenance = args.output_dir / "asl_citizen" / "provenance.csv"
    write_provenance(provenance, completed)
    print(f"Wrote {provenance}")


if __name__ == "__main__":
    main()
