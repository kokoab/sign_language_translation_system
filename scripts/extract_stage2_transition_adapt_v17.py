#!/usr/bin/env python3
"""Extract timestamp-bounded STEM intervals through the shared v17 Stage-2 path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts import extract_stage2_multimodal_v17 as base
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.schema_hand_rgb_v17 import HandRGBV17Config
from active.v17.schema_stage2_features_v17 import Stage2FeatureV17Config


def interval_sample_indices(start_frame: int, end_frame_inclusive: int, source_fps: float, *, with_timestamps: bool = False):
    """Map a verified inclusive interval to nearest source frames at 30 fps."""
    if start_frame < 0 or end_frame_inclusive < start_frame or source_fps <= 0:
        raise ValueError("invalid inclusive interval or source fps")
    duration = (end_frame_inclusive - start_frame + 1) / source_fps
    count = max(1, int(np.ceil(duration * 30.0 - 1e-12)))
    timestamps = start_frame / source_fps + np.arange(count, dtype=np.float64) / 30.0
    indices = np.clip(np.rint(timestamps * source_fps).astype(np.int64), start_frame, end_frame_inclusive)
    return (indices, timestamps) if with_timestamps else indices


def read_interval_frames(video_path: Path, start_frame: int, end_frame_inclusive: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Seek only verified source frames; no crop, resize, or aspect-ratio change."""
    capture = cv2.VideoCapture(video_path.as_posix())
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    indices = interval_sample_indices(start_frame, end_frame_inclusive, fps)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"could not decode source frame {index}: {video_path}")
        frames.append(frame)
    capture.release()
    height, width = frames[0].shape[:2]
    return frames, {
        "fps": fps,
        "source_frame_indices": indices.tolist(),
        "source_start_frame_inclusive": start_frame,
        "source_end_frame_inclusive": end_frame_inclusive,
        "start_seconds": start_frame / fps,
        "end_seconds_exclusive": (end_frame_inclusive + 1) / fps,
        "sample_rate_fps": 30.0,
        "source_width": width,
        "source_height": height,
        "geometry_transform": "none",
    }


def extract_interval_row(row: dict[str, Any], manifest_sha256: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    start, end = (int(value) for value in row["interval"])
    original_reader = base.read_video_frames

    def interval_reader(path: Path, *_: Any, **__: Any) -> tuple[list[np.ndarray], dict[str, Any]]:
        return read_interval_frames(Path(path), start, end)

    base.read_video_frames = interval_reader
    try:
        arrays, metadata = base.extract_row(
            row,
            AppleVisionDetector(0.15),
            AppleVisionDetector(0.15),
            Stage2FeatureV17Config(),
            HandRGBV17Config(),
            manifest_sha256,
        )
    finally:
        base.read_video_frames = original_reader
    metadata["interval_sampling"] = metadata["video_metadata"]
    return arrays, metadata


def output_path(root: Path, row: dict[str, Any]) -> Path:
    return root / row["role"] / row["source"] / f"{row['source_item_id'].replace(':', '_')}.stage2_rgb_v17.npz"


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_sha = base.sha256(args.manifest)
    rows = json.loads(args.manifest.read_text())["rows"]
    rows = [row for row in rows if row["role"] == args.role]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("no rows selected")
    written = 0
    for row in rows:
        arrays, metadata = extract_interval_row(row, manifest_sha)
        destination = output_path(args.output_root, row)
        base.save_archive(destination, arrays, metadata)
        written += 1
    result = {"manifest": args.manifest.as_posix(), "selected_rows": len(rows), "written": written, "output_root": args.output_root.as_posix(), "citizen_test_accessed": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/reports/stage2_v17_transition_adapt_v1/manifest.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/local/stage2_v17_transition_adapt_v1/multimodal"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_transition_adapt_v1/extraction.json"))
    parser.add_argument("--role", choices=("train", "validation"), default="train")
    parser.add_argument("--limit", type=int, default=0)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2))
