#!/usr/bin/env python3
"""Extract one globally normalized v17 trajectory per labeled source video."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.extract_v17 import AppleVisionDetector, extract_video_v17
from active.v17.schema_v17 import V17Config, schema_fingerprint, schema_payload
from scripts.extract_how2sign_transition_landmarks_v17 import safe_name, save, sha256


LOG = logging.getLogger("extract_full_trajectory_landmarks_v17")


def selected_rows(manifest: dict[str, object], args: argparse.Namespace):
    rows = [
        row for row in manifest["rows"]
        if (not args.roles or row["role"] in args.roles)
        and (not args.sources or row["source"] in args.sources)
    ]
    if args.limit:
        rows = rows[:args.limit]
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0, shard count)")
    return [row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]


def destination_for(root: Path, row: dict[str, object]) -> Path:
    return (
        root / str(row["role"]) / safe_name(str(row["source"]))
        / f"{safe_name(str(row['source_item_id']))}.full_trajectory_v17.npz"
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("format") != "slt_full_trajectory_generation_manifest_v17":
        raise ValueError("unexpected full-trajectory manifest")
    manifest_sha = sha256(args.manifest)
    config = V17Config(
        target_frames=args.target_frames,
        maximum_source_frames=args.maximum_source_frames,
        trim_to_hand_activity=False,
    )
    expected_schema = schema_fingerprint(config)
    detector = AppleVisionDetector(config.minimum_point_confidence)
    rows = selected_rows(manifest, args)
    counts: Counter[str] = Counter()
    failures = []
    started = time.monotonic()
    for index, row in enumerate(rows, start=1):
        destination = destination_for(args.output_root, row)
        if destination.exists() and not args.overwrite:
            with np.load(destination, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
            if (
                metadata.get("manifest_sha256") != manifest_sha
                or metadata.get("video_sha256") != row["video_sha256"]
                or metadata.get("schema_fingerprint") != expected_schema
            ):
                raise ValueError(f"stale full-trajectory archive: {destination}")
            counts["skipped"] += 1
            continue
        try:
            result = extract_video_v17(
                row["video_path"], config, detector=detector,
                rotation="auto", input_mirrored=False,
            )
            if result is None:
                raise RuntimeError("no usable hand detections")
            features = result.features.astype(np.float16)
            present = features[..., 3] > 0
            video_metadata = result.metadata
            fps = float(video_metadata.get("fps", 0) or 0)
            reported = int(video_metadata.get("reported_frame_count", 0) or 0)
            duration = row.get("duration_seconds")
            if duration is None and fps > 0 and reported > 0:
                duration = reported / fps
            metadata = {
                "format": "slt_full_trajectory_landmarks_v17",
                "version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "manifest_sha256": manifest_sha,
                "schema_fingerprint": expected_schema,
                "schema": schema_payload(config),
                "source_item_id": row["source_item_id"],
                "source": row["source"],
                "role": row["role"],
                "source_group": row.get("source_group"),
                "signer_id": row.get("signer_id"),
                "video_path": row["video_path"],
                "video_sha256": row["video_sha256"],
                "target_sequence": row["target_sequence"],
                "target_token_ids": row["target_token_ids"],
                "duration_seconds": duration,
                "source_video_metadata": video_metadata,
                "extraction_diagnostics": result.diagnostics,
                "observation_presence_fraction": float(present.mean()),
                "hand_participation": [
                    bool(present[:, :21].any()), bool(present[:, 21:42].any())
                ],
                "whole_utterance_normalization": True,
                "generated_motion": False,
                "citizen_test_accessed": False,
                "semlex_test_accessed": False,
                "local_test_accessed": False,
            }
            save(destination, {
                "observation_features": features,
                "target_token_ids": np.asarray(row["target_token_ids"], dtype=np.int64),
            }, metadata)
            counts["written"] += 1
        except Exception as error:
            counts["failed"] += 1
            failures.append({
                "source_item_id": row["source_item_id"],
                "error": f"{type(error).__name__}: {error}",
            })
            LOG.exception("failed %s", row["source_item_id"])
        finally:
            gc.collect()
        if index == 1 or index % 25 == 0 or index == len(rows):
            LOG.info(
                "%d/%d written=%d skipped=%d failed=%d elapsed=%.1fs",
                index, len(rows), counts["written"], counts["skipped"],
                counts["failed"], time.monotonic() - started,
            )
    report = {
        "format": "slt_full_trajectory_extraction_report_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": args.manifest.as_posix(),
        "manifest_sha256": manifest_sha,
        "output_root": args.output_root.as_posix(),
        "selected_rows": len(rows),
        "counts": dict(counts),
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
        "schema_fingerprint": expected_schema,
        "config": schema_payload(config)["config"],
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, default=Path("active/v17/full_trajectory_generation_manifest_v17.json"))
    value.add_argument("--output-root", type=Path, default=Path("data/local/full_trajectory_landmarks_v17"))
    value.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_full_trajectory_generation_v1/extraction.json"))
    value.add_argument("--target-frames", type=int, default=128)
    value.add_argument("--maximum-source-frames", type=int, default=256)
    value.add_argument("--roles", nargs="*", default=[])
    value.add_argument("--sources", nargs="*", default=[])
    value.add_argument("--limit", type=int, default=0)
    value.add_argument("--shard-count", type=int, default=1)
    value.add_argument("--shard-index", type=int, default=0)
    value.add_argument("--overwrite", action="store_true")
    return value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run(parser().parse_args()), indent=2))
