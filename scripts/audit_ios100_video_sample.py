#!/usr/bin/env python3
"""Audit decoding, dimensions, orientation and optional v16 extraction."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def inspect_video(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded_frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded_frames += 1
    capture.release()
    orientation = "square" if width == height else ("portrait" if height > width else "landscape")
    return {
        "width": width,
        "height": height,
        "orientation": orientation,
        "fps": round(fps, 4),
        "reported_frames": reported_frames,
        "decoded_frames": decoded_frames,
        "decode_complete": decoded_frames > 0 and (
            reported_frames <= 0 or decoded_frames == reported_frames
        ),
        "duration_seconds": round(decoded_frames / fps, 4) if fps > 0 else 0.0,
        "file_bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/local/ios100_audit/asl_citizen/provenance.csv"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("artifacts/reports/ios100_video_audit.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/reports/IOS100_VIDEO_AUDIT.md"),
    )
    parser.add_argument(
        "--landmark-dir",
        type=Path,
        default=Path("data/local/ios100_audit/landmarks_aspect_correct"),
    )
    parser.add_argument("--extract", action="store_true")
    parser.add_argument(
        "--body3d-interval",
        type=int,
        default=8,
        help="Run body detection every N frames during optional extraction",
    )
    parser.add_argument(
        "--legacy-coordinates",
        action="store_true",
        help="Extract the legacy anisotropic schema for checkpoint diagnostics",
    )
    args = parser.parse_args()

    with args.provenance.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    extractor = None
    vision_available = False
    if args.extract:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from active.v16.extract_v16 import VISION_AVAILABLE, extract_video_v16

        vision_available = VISION_AVAILABLE
        if not vision_available:
            raise RuntimeError(
                "Apple Vision Python bridge is unavailable. Run without --extract "
                "for the media audit or install PyObjC Vision/Quartz in a project venv."
            )
        extractor = extract_video_v16

    results: list[dict[str, object]] = []
    for index, row in enumerate(source_rows, start=1):
        path = Path(row["destination"])
        media = inspect_video(path)
        extraction_status = "not_requested"
        hand_mask_fraction = ""
        if extractor is not None:
            features = extractor(
                str(path),
                body_3d_interval=args.body3d_interval,
                aspect_correct=not args.legacy_coordinates,
            )
            if features is None:
                extraction_status = "no_hands"
            else:
                output_path = (
                    args.landmark_dir
                    / row["split"]
                    / row["canonical_gloss"]
                    / f"{path.stem}.npy"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(output_path, features)
                hand_mask = features[:, :42, 3] > 0.5
                hand_mask_fraction = round(float(hand_mask.mean()), 6)
                extraction_status = "ok"
        result = {
            "split": row["split"],
            "canonical_gloss": row["canonical_gloss"],
            "raw_gloss": row["raw_gloss"],
            "lex_code": row["lex_code"],
            "participant": row["participant"],
            "video": row["video"],
            **media,
            "extraction_status": extraction_status,
            "hand_mask_fraction": hand_mask_fraction,
        }
        results.append(result)
        print(
            f"[{index:02}/{len(source_rows)}] {row['canonical_gloss']:10} "
            f"{media['width']}x{media['height']} {media['orientation']:9} "
            f"decode={media['decoded_frames']} extraction={extraction_status}"
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    orientations = Counter(str(row["orientation"]) for row in results)
    resolutions = Counter(f"{row['width']}x{row['height']}" for row in results)
    decode_failures = [row for row in results if not row["decode_complete"]]
    extraction_counts = Counter(str(row["extraction_status"]) for row in results)
    durations = [float(row["duration_seconds"]) for row in results]
    frames = [int(row["decoded_frames"]) for row in results]
    summary = {
        "videos": len(results),
        "signs": len({str(row["canonical_gloss"]) for row in results}),
        "participants": len({str(row["participant"]) for row in results}),
        "orientations": dict(sorted(orientations.items())),
        "resolutions": dict(resolutions.most_common()),
        "decode_failures": len(decode_failures),
        "duration_seconds_min": min(durations),
        "duration_seconds_median": float(np.median(durations)),
        "duration_seconds_max": max(durations),
        "decoded_frames_min": min(frames),
        "decoded_frames_median": float(np.median(frames)),
        "decoded_frames_max": max(frames),
        "vision_extraction_requested": args.extract,
        "vision_available": vision_available,
        "body3d_interval": args.body3d_interval if args.extract else None,
        "coordinate_schema": (
            "legacy_anisotropic"
            if args.extract and args.legacy_coordinates
            else "aspect_correct_isotropic"
            if args.extract
            else None
        ),
        "extraction_status": dict(sorted(extraction_counts.items())),
    }
    lines = [
        "# iOS-100 video audit",
        "",
        "**Dataset:** ASL Citizen selective audit subset  ",
        f"**Videos:** {summary['videos']} across {summary['signs']} signs and "
        f"{summary['participants']} unique public participant IDs",
        "",
        "## Media findings",
        "",
        f"- Orientation counts: `{json.dumps(summary['orientations'], sort_keys=True)}`",
        f"- Resolution counts: `{json.dumps(summary['resolutions'])}`",
        f"- Decode failures: **{summary['decode_failures']}**",
        f"- Duration seconds, min/median/max: **{summary['duration_seconds_min']:.2f} / "
        f"{summary['duration_seconds_median']:.2f} / {summary['duration_seconds_max']:.2f}**",
        f"- Decoded frames, min/median/max: **{summary['decoded_frames_min']} / "
        f"{summary['decoded_frames_median']:.1f} / {summary['decoded_frames_max']}**",
        "",
        "## Apple Vision extraction",
        "",
        f"- Requested: **{summary['vision_extraction_requested']}**",
        f"- Python bridge available: **{summary['vision_available']}**",
        f"- Body detection interval: **{summary['body3d_interval']}**",
        f"- Coordinate schema: **{summary['coordinate_schema']}**",
        f"- Status counts: `{json.dumps(summary['extraction_status'], sort_keys=True)}`",
        "",
        "The detailed per-video measurements are in "
        "`artifacts/reports/ios100_video_audit.csv`.",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
