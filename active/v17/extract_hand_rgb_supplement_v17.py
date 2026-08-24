#!/usr/bin/env python3
"""Extract hand RGB crops from frozen train-only SemLex/local selections."""

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
from active.v17.schema_hand_rgb_v17 import (
    HandRGBV17Config,
    schema_fingerprint,
)
from active.v17.train_stage_1_visual_speech_v17 import sha256_file


LOG = logging.getLogger("hand_rgb_supplement_v17")
ALLOWED_LOCAL_TIERS = ("tier_a_dual_top1",)
DEEP_CLEAN_FORMAT = "slt_v17_local_deep_clean_final_v1"


@dataclass(frozen=True)
class SupplementItem:
    source: str
    label: str
    item_id: str
    raw_path: Path
    landmark_path: Path
    source_row: dict[str, object]


def selection_items(
    manifest_path: Path, source: str
) -> tuple[list[SupplementItem], dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("videos")
    if not isinstance(rows, list) or not rows:
        raise ValueError("supplement selection has no videos")
    if source == "semlex":
        if manifest.get("split") != "train_only":
            raise ValueError("SemLex RGB selection must be train_only")
        rows = [row for row in rows if row.get("semlex_split") == "train"]
        expected = int(manifest.get("selected_clips", -1))
        if len(rows) != expected:
            raise ValueError("SemLex train selection count mismatch")
        def paths(row):
            return (
                str(row["semlex_video_id"]), Path(str(row["raw_path"])),
                Path(str(row["feature_path"])),
            )
    elif source == "local_tier_a":
        if manifest.get("split_eligibility") not in (
            "train_only_after_human_review",
            "train_only_after_ASL_fluent_exact_variant_review",
        ):
            raise ValueError("local RGB selection is not train-only review data")
        rows = [
            row for row in rows
            if row.get("consensus_tier") in ALLOWED_LOCAL_TIERS
        ]
        if not rows:
            raise ValueError("local RGB selection has no Tier-A clips")
        def paths(row):
            raw = Path(str(row["raw_path"]))
            return raw.stem, raw, Path(str(row["feature_path"]))
    elif source in ("local_deep_clean", "local_deep_clean_val"):
        split = "train" if source == "local_deep_clean" else "val"
        if (
            manifest.get("format") != DEEP_CLEAN_FORMAT
            or manifest.get("split") != split
            or manifest.get("extraction_complete") is not True
            or int(manifest.get("selected_classes", -1)) != 94
            or manifest.get("signer_disjoint") is not False
            or manifest.get("signer_overlap_user_approved") is not True
            or manifest.get("citizen_test_accessed") is not False
            or manifest.get("semlex_test_accessed") is not False
        ):
            raise ValueError(f"invalid finalized local deep-clean {split} manifest")
        if int(manifest.get("selected_clips", -1)) != len(rows):
            raise ValueError("local deep-clean selected_clips mismatch")
        for row in rows:
            eligible = (
                row.get("training_eligible") is True
                and row.get("validation_eligible") is False
                if split == "train"
                else row.get("training_eligible") is False
                and row.get("validation_eligible") is True
            )
            if row.get("local_split") != split or not eligible:
                raise ValueError("local deep-clean row violates split contract")

        def paths(row):
            return (
                str(row["item_id"]),
                Path(str(row["raw_path"])),
                Path(str(row["feature_path"])),
            )
    else:
        raise ValueError(f"unsupported supplement source: {source}")

    items: list[SupplementItem] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        label = str(row.get("canonical_label", ""))
        item_id, raw_path, landmark_path = paths(row)
        if not label or not item_id or Path(item_id).name != item_id:
            raise ValueError("unsafe supplement label or item ID")
        key = (label, item_id)
        if key in seen:
            raise ValueError(f"duplicate supplement item: {label}/{item_id}")
        seen.add(key)
        if not raw_path.is_file():
            raise FileNotFoundError(f"missing supplement video: {raw_path}")
        if not landmark_path.is_file():
            raise FileNotFoundError(f"missing supplement landmark: {landmark_path}")
        items.append(SupplementItem(
            source=source, label=label, item_id=item_id, raw_path=raw_path,
            landmark_path=landmark_path, source_row=row,
        ))
    return items, manifest


def run(args: argparse.Namespace) -> dict[str, object]:
    items, manifest = selection_items(args.selection_manifest, args.source)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    items = items[args.shard_index :: args.shard_count]
    if args.limit:
        items = items[:args.limit]
    config = HandRGBV17Config()
    config.validate()
    fingerprint = schema_fingerprint(config)
    detector = AppleVisionDetector(args.minimum_confidence)
    is_validation = args.source == "local_deep_clean_val"
    expected_training_eligible = not is_validation
    split_contract = (
        "validation_nonsigner_disjoint_user_approved"
        if is_validation
        else "train_only"
    )
    written = skipped = 0
    started = time.monotonic()
    for index, item in enumerate(items, start=1):
        output_path = (
            args.output_root / args.source / item.label
            / f"{item.item_id}.hand_rgb_v17.npz"
        )
        if output_path.exists() and not args.overwrite:
            with np.load(output_path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
            if (
                metadata.get("schema_fingerprint") != fingerprint
                or metadata.get("source") != args.source
                or metadata.get("source_item_id") != item.item_id
                or metadata.get("split") != split_contract
                or metadata.get("training_eligible") is not expected_training_eligible
                or metadata.get("test_accessed") is not False
            ):
                raise ValueError(f"existing supplement RGB mismatch: {output_path}")
            skipped += 1
            continue
        arrays, metadata, diagnostics = extract_clip(
            item.raw_path, item.landmark_path, detector, config
        )
        metadata.update({
            "source": args.source,
            "source_item_id": item.item_id,
            "canonical_label": item.label,
            "selection_manifest": str(args.selection_manifest),
            "selection_manifest_sha256": sha256_file(args.selection_manifest),
            "split": split_contract,
            "training_eligible": expected_training_eligible,
            "test_accessed": False,
        })
        diagnostics["selection_quality_score"] = float(
            item.source_row.get("quality_score", 0.0)
        )
        save_archive(output_path, arrays, metadata, diagnostics, config)
        written += 1
        if index == 1 or index % 25 == 0 or index == len(items):
            LOG.info(
                "%s %d/%d written=%d skipped=%d elapsed=%.1fs",
                args.source, index, len(items), written, skipped,
                time.monotonic() - started,
            )
    result = {
        "source": args.source,
        "clips": len(items),
        "written": written,
        "skipped": skipped,
        "classes": len({item.label for item in items}),
        "selection_manifest": str(args.selection_manifest),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "selection_declares_training_eligible": manifest.get("training_eligible"),
        "schema_fingerprint": fingerprint,
        "seconds": time.monotonic() - started,
        "split": split_contract,
        "training_eligible": expected_training_eligible,
        "test_accessed": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    result_name = (
        f"{args.source}_extraction_result.json"
        if args.shard_count == 1
        else f"{args.source}_extraction_shard_{args.shard_index}_of_{args.shard_count}.json"
    )
    result.update({"shard_index": args.shard_index, "shard_count": args.shard_count})
    (args.output_root / result_name).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=(
            "semlex",
            "local_tier_a",
            "local_deep_clean",
            "local_deep_clean_val",
        ),
        required=True,
    )
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/local/hand_rgb_supplements_v17"),
    )
    parser.add_argument("--minimum-confidence", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
