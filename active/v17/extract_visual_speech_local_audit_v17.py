#!/usr/bin/env python3
"""Extract evaluation-only face views for the frozen local audit pool."""

from __future__ import annotations

import argparse
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

from active.v17.extract_visual_speech_v17 import extract_clip, save_archive
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.local_multimodal_audit_v17 import SOURCE, SPLIT, local_audit_items
from active.v17.schema_visual_speech_v17 import VisualSpeechV17Config, schema_fingerprint
from active.v17.train_stage_1_visual_speech_v17 import sha256_file


LOG = logging.getLogger("visual_speech_local_audit_v17")


def run(args: argparse.Namespace) -> dict[str, object]:
    items, _ = local_audit_items(args.selection_manifest)
    if args.limit:
        items = items[:args.limit]
    config = VisualSpeechV17Config()
    detector = AppleVisionDetector(config.minimum_face_confidence)
    fingerprint = schema_fingerprint(config)
    written = skipped = 0
    started = time.monotonic()
    for index, item in enumerate(items, start=1):
        output = args.output_root / item.label / f"{item.item_id}.visual_speech_v17.npz"
        if output.exists() and not args.overwrite:
            with np.load(output, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
            if (
                metadata.get("schema_fingerprint") != fingerprint
                or metadata.get("source") != SOURCE
                or metadata.get("split") != SPLIT
                or metadata.get("training_eligible") is not False
                or metadata.get("audio_accessed") is not False
            ):
                raise ValueError(f"existing local visual-speech mismatch: {output}")
            skipped += 1
            continue
        arrays, metadata, diagnostics = extract_clip(item.raw_path, detector, config)
        metadata.update({
            "source": SOURCE, "source_item_id": item.item_id,
            "canonical_label": item.label, "split": SPLIT,
            "selection_manifest": str(args.selection_manifest),
            "selection_manifest_sha256": sha256_file(args.selection_manifest),
            "training_eligible": False, "audio_accessed": False,
            "test_accessed": False,
        })
        save_archive(output, arrays, metadata, diagnostics, config)
        written += 1
        if index == 1 or index % 25 == 0 or index == len(items):
            LOG.info(
                "%d/%d written=%d skipped=%d elapsed=%.1fs",
                index, len(items), written, skipped, time.monotonic() - started,
            )
    result = {
        "source": SOURCE, "split": SPLIT, "clips": len(items),
        "written": written, "skipped": skipped,
        "classes": len({item.label for item in items}),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "training_eligible": False, "schema_fingerprint": fingerprint,
        "audio_accessed": False, "test_accessed": False,
        "seconds": time.monotonic() - started,
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
        default=Path("data/local/local_citizen100_quality_audit_q82_cap14_exact/candidate_selection.json"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/local/local_citizen100_quality_audit_q82_cap14_exact/visual_speech_rgb"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
