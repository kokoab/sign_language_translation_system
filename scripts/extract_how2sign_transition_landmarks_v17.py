#!/usr/bin/env python3
"""Extract genuine continuous How2Sign windows with the fixed v17 Apple schema."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import time

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.extract_v17 import (
    AppleVisionDetector,
    choose_coarse_orientation_v17,
    extract_frames_v17,
    read_video_frames,
    rotate_frame_clockwise,
)
from active.v17.schema_stage2_features_v17 import landmark_config
from active.v17.schema_v17 import (
    NUM_CHANNELS,
    NUM_NODES,
    schema_fingerprint,
    schema_payload,
)
from scripts.extract_stage2_multimodal_v17 import window_ranges


LOG = logging.getLogger("extract_how2sign_transition_landmarks_v17")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def save(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(path)


def selected_rows(
    manifest: dict[str, object], per_signer: int, limit: int,
    shard_count: int = 1, shard_index: int = 0,
):
    rows = list(manifest["rows"])
    if per_signer:
        counts: Counter[str] = Counter()
        balanced = []
        for row in rows:
            signer = str(row["signer_id"])
            if counts[signer] < per_signer:
                balanced.append(row)
                counts[signer] += 1
        rows = balanced
    if limit:
        rows = rows[:limit]
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard index must be in [0, shard count)")
    rows = [row for index, row in enumerate(rows) if index % shard_count == shard_index]
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    manifest_sha = sha256(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("format") not in {
        "how2sign_unlabeled_transition_manifest_v17",
        "continuous_unlabeled_transition_manifest_v17",
    }:
        raise ValueError("unexpected continuous transition manifest")
    rows = selected_rows(
        manifest, args.per_signer, args.limit, args.shard_count, args.shard_index
    )
    config = landmark_config()
    detector = AppleVisionDetector(config.minimum_point_confidence)
    expected_schema = schema_fingerprint(config)
    written = skipped = failed = 0
    failures = []
    started = time.monotonic()
    for index, row in enumerate(rows, start=1):
        destination = (
            args.output_root / str(row["signer_id"]).replace(":", "_")
            / f"{safe_name(str(row['source_item_id']))}.transition_landmarks_v17.npz"
        )
        if destination.exists() and not args.overwrite:
            with np.load(destination, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
            if (
                metadata.get("video_sha256") != row["video_sha256"]
                or metadata.get("manifest_sha256") != manifest_sha
                or metadata.get("schema_fingerprint") != expected_schema
            ):
                raise ValueError(f"{destination}: stale extracted archive")
            skipped += 1
            continue
        try:
            frames, video_metadata = read_video_frames(
                row["video_path"], args.maximum_source_frames,
                args.maximum_image_side, rotation="auto", input_mirrored=False,
            )
            correction, orientation_scores = choose_coarse_orientation_v17(
                frames, detector
            )
            if correction:
                frames = [rotate_frame_clockwise(frame, correction) for frame in frames]
            ranges = window_ranges(len(frames), 32, 4)
            if not ranges:
                raise RuntimeError("fewer than four sampled frames")
            landmarks = np.zeros(
                (len(ranges), 32, NUM_NODES, NUM_CHANNELS), dtype=np.float16
            )
            valid = np.zeros(len(ranges), dtype=np.bool_)
            diagnostics = []
            for window, (start, end) in enumerate(ranges):
                result = extract_frames_v17(
                    frames[start:end], config, detector=detector
                )
                if result is not None:
                    landmarks[window] = result.features
                    valid[window] = True
                    diagnostics.append(result.diagnostics)
                else:
                    diagnostics.append({"no_usable_hand_detections": True})
            if not valid.any():
                raise RuntimeError("no valid v17 landmark windows")
            metadata = {
                "format": "continuous_transition_landmarks_v17",
                "version": 1,
                "schema_fingerprint": expected_schema,
                "schema": schema_payload(config),
                "manifest_sha256": manifest_sha,
                "source_item_id": row["source_item_id"],
                "source": row["source"],
                "role": row.get("role", "train"),
                "signer_id": row["signer_id"],
                "source_group": row["source_group"],
                "video_sha256": row["video_sha256"],
                "duration_seconds": row["duration_seconds"],
                "license": row["license"],
                "window_count": len(ranges),
                "valid_windows": int(valid.sum()),
                "sampled_source_frames": len(frames),
                "vision_coarse_rotation_clockwise": correction,
                "vision_orientation_scores": orientation_scores,
                "video_metadata": video_metadata,
                "window_diagnostics": diagnostics,
                "lip_nodes_preserved": True,
                "citizen_test_accessed": False,
                "semlex_test_accessed": False,
                "local_test_accessed": False,
                "how2sign_validation_accessed": False,
                "how2sign_test_accessed": False,
            }
            save(destination, {
                "landmarks": landmarks,
                "window_valid": valid,
                "window_source_ranges": np.asarray(ranges, dtype=np.int64),
            }, metadata)
            written += 1
        except Exception as error:
            failed += 1
            failures.append({
                "source_item_id": row["source_item_id"],
                "error": f"{type(error).__name__}: {error}",
            })
            LOG.exception("failed %s", row["source_item_id"])
        finally:
            gc.collect()
        if index == 1 or index % 10 == 0 or index == len(rows):
            LOG.info(
                "%d/%d written=%d skipped=%d failed=%d elapsed=%.1fs",
                index, len(rows), written, skipped, failed, time.monotonic() - started,
            )
    result = {
        "format": "continuous_transition_landmark_extraction_report_v17",
        "source": manifest.get("source"),
        "manifest": args.manifest.as_posix(),
        "manifest_sha256": manifest_sha,
        "selected_rows": len(rows),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "failures": failures,
        "schema_fingerprint": expected_schema,
        "maximum_source_frames": args.maximum_source_frames,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "seconds": time.monotonic() - started,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("active/v17/how2sign_transition_manifest_v17.json"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/local/how2sign_transition_landmarks_v17"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path(
            "artifacts/reports/how2sign_transition_landmarks_v17/extraction.json"
        ),
    )
    parser.add_argument("--maximum-source-frames", type=int, default=256)
    parser.add_argument("--maximum-image-side", type=int, default=960)
    parser.add_argument("--per-signer", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
