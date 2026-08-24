#!/usr/bin/env python3
"""Extract hand crops for the frozen SemLex validation diagnostic only."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.extract_hand_rgb_v17 import extract_clip, save_archive
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.schema_hand_rgb_v17 import HandRGBV17Config, schema_fingerprint
from active.v17.train_stage_1_visual_speech_v17 import sha256_file


LOG = logging.getLogger("hand_rgb_semlex_val_v17")
SOURCE = "semlex_val"
SPLIT = "val_domain_diagnostic"


@dataclass(frozen=True)
class SemLexValidationItem:
    label: str
    item_id: str
    raw_path: Path
    landmark_path: Path


def validation_items(manifest_path: Path) -> tuple[list[SemLexValidationItem], dict[str, object]]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("split") != "val"
        or manifest.get("training_eligible") is not False
        or manifest.get("split_eligibility") != "evaluation_only_never_training"
    ):
        raise ValueError("SemLex selection is not the frozen validation-only diagnostic")
    rows = manifest.get("videos")
    if not isinstance(rows, list) or len(rows) != int(manifest.get("selected_clips", -1)):
        raise ValueError("SemLex validation selection count mismatch")
    raw_root = manifest_path.parent / "raw"
    landmark_root = manifest_path.parent / "landmarks_v17"
    items: list[SemLexValidationItem] = []
    missing: list[str] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("semlex_split") != "val" or row.get("training_eligible") is not False:
            raise ValueError("SemLex validation row violates the evaluation-only contract")
        label = str(row.get("canonical_label", ""))
        item_id = str(row.get("semlex_video_id", ""))
        if not label or not item_id or Path(item_id).name != item_id:
            raise ValueError("unsafe SemLex validation label or item ID")
        key = (label, item_id)
        if key in seen:
            raise ValueError(f"duplicate SemLex validation item: {label}/{item_id}")
        seen.add(key)
        raw_path = raw_root / label / f"{item_id}.webm"
        landmark_path = landmark_root / label / f"{item_id}.v17.npz"
        if raw_path.is_file() != landmark_path.is_file():
            raise ValueError(f"partial SemLex validation pair: {label}/{item_id}")
        if not raw_path.is_file():
            missing.append(f"{label}/{item_id}")
            continue
        items.append(SemLexValidationItem(label, item_id, raw_path, landmark_path))
    actual_raw = {
        (path.parent.name, path.stem) for path in raw_root.glob("*/*.webm")
    }
    if actual_raw != {(item.label, item.item_id) for item in items}:
        raise ValueError("SemLex validation raw directory contains unplanned clips")
    if len(items) != 978 or len(missing) != 6:
        raise ValueError(
            f"expected 978 retained and 6 quarantined clips, got {len(items)} and {len(missing)}"
        )
    return items, manifest


def run(args: argparse.Namespace) -> dict[str, object]:
    items, manifest = validation_items(args.selection_manifest)
    if args.limit:
        items = items[:args.limit]
    config = HandRGBV17Config()
    config.validate()
    fingerprint = schema_fingerprint(config)
    detector = AppleVisionDetector(args.minimum_confidence)
    written = skipped = 0
    started = time.monotonic()
    for index, item in enumerate(items, start=1):
        output_path = args.output_root / item.label / f"{item.item_id}.hand_rgb_v17.npz"
        if output_path.exists() and not args.overwrite:
            with np.load(output_path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
            if (
                metadata.get("schema_fingerprint") != fingerprint
                or metadata.get("source") != SOURCE
                or metadata.get("source_item_id") != item.item_id
                or metadata.get("split") != SPLIT
                or metadata.get("training_eligible") is not False
            ):
                raise ValueError(f"existing SemLex validation crop mismatch: {output_path}")
            skipped += 1
            continue
        arrays, metadata, diagnostics = extract_clip(
            item.raw_path, item.landmark_path, detector, config
        )
        metadata.update({
            "source": SOURCE,
            "source_item_id": item.item_id,
            "canonical_label": item.label,
            "selection_manifest": str(args.selection_manifest),
            "selection_manifest_sha256": sha256_file(args.selection_manifest),
            "split": SPLIT,
            "training_eligible": False,
            "test_accessed": False,
        })
        save_archive(output_path, arrays, metadata, diagnostics, config)
        written += 1
        if index == 1 or index % 25 == 0 or index == len(items):
            LOG.info(
                "%d/%d written=%d skipped=%d elapsed=%.1fs",
                index, len(items), written, skipped, time.monotonic() - started,
            )
    result = {
        "source": SOURCE,
        "split": SPLIT,
        "clips": len(items),
        "written": written,
        "skipped": skipped,
        "classes": len({item.label for item in items}),
        "selection_manifest": str(args.selection_manifest),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "selection_declares_training_eligible": manifest.get("training_eligible"),
        "training_eligible": False,
        "schema_fingerprint": fingerprint,
        "seconds": time.monotonic() - started,
        "test_accessed": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "extraction_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-manifest", type=Path,
        default=Path("data/local/semlex_citizen100_val_audit/selection_plan.json"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/local/semlex_citizen100_val_audit/hand_rgb_v17"),
    )
    parser.add_argument("--minimum-confidence", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
