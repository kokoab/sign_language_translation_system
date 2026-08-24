#!/usr/bin/env python3
"""Prove local train/validation raw clips do not duplicate development corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-root", type=Path, default=Path("data/local/local_deep_clean_v17")
    )
    parser.add_argument(
        "--citizen-root", type=Path, default=Path("data/local/citizen100_v17/raw")
    )
    parser.add_argument(
        "--semlex-train-provenance",
        type=Path,
        default=Path(
            "data/local/semlex_citizen100_train_audit/download_provenance.json"
        ),
    )
    parser.add_argument(
        "--semlex-validation-provenance",
        type=Path,
        default=Path("data/local/semlex_citizen100_val_audit/download_provenance.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/reports/local_deep_clean_v17/raw_hash_overlap_audit.json"
        ),
    )
    args = parser.parse_args()
    citizen_hashes: dict[str, str] = {}
    for split in ("train", "val"):
        for path in sorted((args.citizen_root / split).glob("*/*")):
            if path.is_file():
                citizen_hashes[sha256_file(path)] = f"{split}:{path}"
    semlex_hashes: dict[str, str] = {}
    for split, path in (
        ("train", args.semlex_train_provenance),
        ("val", args.semlex_validation_provenance),
    ):
        provenance = json.loads(path.read_text(encoding="utf-8"))
        if any(row.get("semlex_split") != split for row in provenance["videos"]):
            raise ValueError(f"mixed SemLex split in {path}")
        for row in provenance["videos"]:
            semlex_hashes[str(row["sha256"])] = f"{split}:{row['raw_path']}"

    splits: dict[str, object] = {}
    total_overlaps = 0
    for split in ("train", "val"):
        manifest_path = args.local_root / f"{split}_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("split") != split
            or manifest.get("citizen_test_accessed") is not False
            or manifest.get("semlex_test_accessed") is not False
        ):
            raise ValueError(f"invalid local {split} manifest")
        citizen_matches = [
            {
                "canonical_label": row["canonical_label"],
                "item_id": row["item_id"],
                "matched_development_path": citizen_hashes[row["raw_sha256"]],
            }
            for row in manifest["videos"]
            if row["raw_sha256"] in citizen_hashes
        ]
        semlex_matches = [
            {
                "canonical_label": row["canonical_label"],
                "item_id": row["item_id"],
                "matched_development_path": semlex_hashes[row["raw_sha256"]],
            }
            for row in manifest["videos"]
            if row["raw_sha256"] in semlex_hashes
        ]
        total_overlaps += len(citizen_matches) + len(semlex_matches)
        splits[split] = {
            "local_clips": len(manifest["videos"]),
            "citizen_trainval_exact_raw_matches": citizen_matches,
            "semlex_trainval_exact_raw_matches": semlex_matches,
        }
    result = {
        "format": "slt_v17_local_deep_clean_raw_overlap_audit_v1",
        "status": "PASS" if total_overlaps == 0 else "FAIL",
        "citizen_trainval_unique_raw_hashes": len(citizen_hashes),
        "semlex_trainval_unique_raw_hashes": len(semlex_hashes),
        "exact_raw_overlaps": total_overlaps,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if total_overlaps == 0 else 1)


if __name__ == "__main__":
    main()
