#!/usr/bin/env python3
"""Finalize local deep-clean manifests after fresh Apple Vision v17 extraction."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.schema_v17 import V17Config, schema_fingerprint
from active.v17.train_stage_1_v17 import load_v17_archive


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize_manifest(
    manifest_path: Path,
    landmark_root: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = str(manifest.get("split", ""))
    if (
        manifest.get("format") != "slt_v17_local_deep_clean_v1"
        or split not in ("train", "val")
        or manifest.get("citizen_test_accessed") is not False
        or manifest.get("semlex_test_accessed") is not False
    ):
        raise ValueError("invalid pre-extraction local manifest")
    rows = manifest.get("videos")
    if not isinstance(rows, list) or int(manifest.get("selected_clips", -1)) != len(rows):
        raise ValueError("local manifest row count mismatch")
    expected_schema = schema_fingerprint(V17Config())
    retained: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for row in rows:
        label = str(row.get("canonical_label", ""))
        item_id = str(row.get("item_id", ""))
        feature_path = landmark_root / label / f"{item_id}.v17.npz"
        reason = None
        if not feature_path.is_file():
            reason = "missing_v17_archive"
        else:
            try:
                load_v17_archive(feature_path, expected_schema)
            except (ValueError, KeyError, OSError) as error:
                reason = f"invalid_v17_archive:{type(error).__name__}"
        if reason is None:
            retained.append({**row, "feature_path": str(feature_path)})
        else:
            rejected.append({**row, "extraction_rejection_reason": reason})
    class_counts = dict(
        sorted(Counter(str(row["canonical_label"]) for row in retained).items())
    )
    expected_classes = int(manifest.get("selected_classes", -1))
    if len(class_counts) != expected_classes:
        missing_classes = sorted(
            set(str(label) for label in manifest.get("class_counts", {}))
            - set(class_counts)
        )
        raise ValueError(
            "fresh extraction lost one or more local classes: "
            f"expected={expected_classes}, retained={len(class_counts)}, "
            f"missing={missing_classes}"
        )
    finalized = {
        **manifest,
        "format": "slt_v17_local_deep_clean_final_v1",
        "pre_extraction_manifest": str(manifest_path),
        "pre_extraction_manifest_sha256": sha256_file(manifest_path),
        "fresh_extractor": "active/v17/extract_v17.py",
        "extractor_schema_fingerprint": expected_schema,
        "extraction_complete": True,
        "selected_clips": len(retained),
        "selected_classes": len(class_counts),
        "class_counts": class_counts,
        "extraction_rejections": len(rejected),
        "videos": retained,
    }
    return finalized, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("data/local/local_deep_clean_v17")
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val"),
        default=("train", "val"),
        help="Finalize one or both extracted development splits",
    )
    args = parser.parse_args()
    summary: dict[str, object] = {
        "format": "slt_v17_local_deep_clean_finalization_v1",
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "splits": {},
    }
    all_rejections: list[dict[str, object]] = []
    for split in args.splits:
        source = args.root / f"{split}_manifest.json"
        output = args.root / f"{split}_final_manifest.json"
        finalized, rejected = finalize_manifest(
            source, args.root / "landmarks" / split
        )
        output.write_text(json.dumps(finalized, indent=2) + "\n", encoding="utf-8")
        all_rejections.extend(rejected)
        summary["splits"][split] = {
            "selected_clips": finalized["selected_clips"],
            "selected_classes": finalized["selected_classes"],
            "extraction_rejections": len(rejected),
            "manifest": str(output),
            "manifest_sha256": sha256_file(output),
        }
    rejection_path = args.root / "extraction_rejections.json"
    rejection_path.write_text(
        json.dumps({"count": len(all_rejections), "videos": all_rejections}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    summary["total_extraction_rejections"] = len(all_rejections)
    (args.root / "finalization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
