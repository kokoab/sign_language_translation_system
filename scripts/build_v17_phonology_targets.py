#!/usr/bin/env python3
"""Freeze ASL-LEX 2.0 phonological supervision for the pinned v17 vocabulary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ATTRIBUTES = (
    ("handshape", "Handshape.2.0"),
    ("selected_fingers", "SelectedFingers.2.0"),
    ("flexion", "Flexion.2.0"),
    ("sign_type", "SignType.2.0"),
    ("movement", "Movement.2.0"),
    ("major_location", "MajorLocation.2.0"),
    ("minor_location", "MinorLocation.2.0"),
    ("contact", "Contact.2.0"),
    ("repeated_movement", "RepeatedMovement.2.0"),
    ("wrist_twist", "UlnarRotation.2.0"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_value(value: str) -> str | None:
    cleaned = value.strip()
    return None if not cleaned or cleaned.casefold() in {"na", "n/a", "nan"} else cleaned


def build_payload(
    manifest: dict[str, object], asllex_rows: list[dict[str, str]], *,
    manifest_sha256: str, asllex_sha256: str,
) -> dict[str, object]:
    rows_by_code = {row["Code"]: row for row in asllex_rows}
    classes = sorted(manifest["classes"], key=lambda item: int(item["class_index"]))  # type: ignore[index]
    expected_indices = list(range(len(classes)))
    if [int(item["class_index"]) for item in classes] != expected_indices:
        raise ValueError("manifest class indices must be contiguous")

    resolved = []
    for item in classes:
        code = str(item["citizen_asl_lex_code"])
        row = rows_by_code.get(code)
        if row is None:
            raise ValueError(f"ASL-LEX code is absent from official metadata: {code}")
        resolved.append(row)

    attributes = []
    for name, column in ATTRIBUTES:
        raw_targets = [normalized_value(row[column]) for row in resolved]
        values = sorted({value for value in raw_targets if value is not None})
        if len(values) < 2:
            raise ValueError(f"phonological attribute {name} has fewer than two values")
        value_to_index = {value: index for index, value in enumerate(values)}
        targets = [
            -100 if value is None else value_to_index[value] for value in raw_targets
        ]
        attributes.append(
            {
                "name": name,
                "source_column": column,
                "values": values,
                "targets_by_class_index": targets,
                "annotated_classes": sum(target != -100 for target in targets),
            }
        )

    return {
        "format": "slt_v17_asllex_phonology_targets",
        "version": 1,
        "manifest_sha256": manifest_sha256,
        "asllex_metadata_sha256": asllex_sha256,
        "class_count": len(classes),
        "classes": [
            {
                "class_index": int(item["class_index"]),
                "canonical_label": item["canonical_label"],
                "citizen_asl_lex_code": item["citizen_asl_lex_code"],
                "asllex_entry_id": resolved[index]["EntryID"],
            }
            for index, item in enumerate(classes)
        ],
        "attributes": attributes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json")
    )
    parser.add_argument(
        "--asllex-metadata", type=Path,
        default=Path("data/local/dataset_metadata/asllex2_official/signdata.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("active/v17/citizen100_phonology.json")
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with args.asllex_metadata.open(encoding="latin-1", newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload = build_payload(
        manifest,
        rows,
        manifest_sha256=sha256(args.manifest),
        asllex_sha256=sha256(args.asllex_metadata),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "classes": payload["class_count"],
                "attributes": {
                    item["name"]: {
                        "values": len(item["values"]),
                        "annotated_classes": item["annotated_classes"],
                    }
                    for item in payload["attributes"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
