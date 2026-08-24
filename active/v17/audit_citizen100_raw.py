#!/usr/bin/env python3
"""Audit the downloaded Citizen100 raw corpus and official split contract."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import cv2


SPLITS = ("train", "val", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json")
    )
    parser.add_argument(
        "--provenance", type=Path, default=Path("data/local/citizen100_v17/provenance.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/reports/CITIZEN100_RAW_AUDIT.md")
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = {
        (str(item["canonical_label"]), split): int(item["video_counts"][split])
        for item in manifest["classes"]
        for split in SPLITS
    }
    expected_total = sum(expected.values())
    with args.provenance.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    errors: list[str] = []
    counts: Counter[tuple[str, str]] = Counter()
    participants: dict[str, set[str]] = defaultdict(set)
    dimensions: Counter[str] = Counter()
    orientations: Counter[str] = Counter()
    decoded = 0
    for row in rows:
        split = row["split"]
        label = row["canonical_label"]
        counts[(label, split)] += 1
        participants[split].add(row["participant"])
        path = Path(row["destination"])
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        if path.stat().st_size != int(row["uncompressed_bytes"]):
            errors.append(f"size mismatch: {path}")
            continue
        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        rotation = int(round(capture.get(cv2.CAP_PROP_ORIENTATION_META)))
        capture.release()
        if not ok or frame is None:
            errors.append(f"decode failure: {path}")
            continue
        decoded += 1
        height, width = frame.shape[:2]
        dimensions[f"{width}x{height}"] += 1
        orientations[
            "square" if width == height else "portrait" if height > width else "landscape"
        ] += 1
        if rotation:
            dimensions[f"rotation_metadata_{rotation}"] += 1

    for key, count in expected.items():
        if counts[key] != count:
            errors.append(f"count mismatch {key[0]}/{key[1]}: {counts[key]} != {count}")
    overlap = {
        "train_val": sorted(participants["train"] & participants["val"]),
        "train_test": sorted(participants["train"] & participants["test"]),
        "val_test": sorted(participants["val"] & participants["test"]),
    }
    for name, values in overlap.items():
        if values:
            errors.append(f"participant overlap {name}: {values}")
    status = (
        "PASS"
        if not errors and len(rows) == expected_total and decoded == len(rows)
        else "FAIL"
    )
    lines = [
        "# Citizen100 raw dataset audit",
        "",
        f"**Status: {status}**",
        "",
        f"- Provenance rows: {len(rows)}",
        f"- First-frame decode success: {decoded}/{len(rows)}",
        f"- Classes: {len({row['canonical_label'] for row in rows})}",
        f"- Videos by split: `{json.dumps(Counter(row['split'] for row in rows), sort_keys=True)}`",
        f"- Unique participants by split: `{json.dumps({s: len(participants[s]) for s in SPLITS})}`",
        f"- Cross-split participant overlap: `{json.dumps(overlap, sort_keys=True)}`",
        f"- Decoded orientation: `{json.dumps(orientations, sort_keys=True)}`",
        f"- Decoded dimensions: `{json.dumps(dimensions, sort_keys=True)}`",
        f"- Errors: {len(errors)}",
        "",
        "Files were already checked against the official ZIP member size and CRC during",
        "download. This audit independently enforces manifest counts, first-frame decode,",
        "and signer-disjoint split membership.",
    ]
    if errors:
        lines.extend(["", "## Errors", ""] + [f"- {error}" for error in errors[:100]])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "videos": len(rows), "errors": len(errors)}, indent=2))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
