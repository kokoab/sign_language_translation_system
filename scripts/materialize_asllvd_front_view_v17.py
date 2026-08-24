#!/usr/bin/env python3
"""Materialize the ASLLVD front-camera half without aspect distortion."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from prepare_asllvd_asllex_supplement_v17 import probe_video, sha256_file


def crop_front(
    source: Path,
    destination: Path,
    width: int,
    height: int,
    annotated_start: int,
    annotated_end: int,
) -> dict[str, object]:
    if height != 656 or height % 2:
        raise ValueError(f"unexpected ASLLVD composite dimensions: {width}x{height}")
    front_height = height // 2
    sign_frames = annotated_end - annotated_start + 1
    if sign_frames < 4:
        raise ValueError(f"invalid annotated sign interval: {annotated_start}..{annotated_end}")
    if destination.is_file():
        media = probe_video(destination)
        if (
            media["width"] == width
            and media["height"] == front_height
            and media["frames"] == sign_frames
        ):
            return {**media, "bytes": destination.stat().st_size, "sha256": sha256_file(destination)}
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-i", str(source),
                "-map", "0:v:0", "-an", "-vf",
                (
                    f"crop=iw:{front_height}:0:0,"
                    f"trim=start_frame=50:end_frame={50 + sign_frames},"
                    "setpts=PTS-STARTPTS"
                ),
                "-c:v", "libx264", "-crf", "0", "-preset", "veryfast",
                "-pix_fmt", "yuv444p", "-movflags", "+faststart", str(temporary),
            ],
            check=True,
        )
        media = probe_video(temporary)
        if (
            media["width"] != width
            or media["height"] != front_height
            or media["frames"] != sign_frames
        ):
            raise ValueError(f"front crop geometry mismatch: {temporary}")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {**media, "bytes": destination.stat().st_size, "sha256": sha256_file(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/local/asllvd_asllex_v17/exact_variant_manifest.json"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/local/asllvd_asllex_v17/raw_front"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")
    if args.workers < 1:
        raise ValueError("workers must be positive")

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if payload.get("format") != "slt_v17_asllvd_asllex_exact_supplement":
        raise ValueError("not an ASLLVD exact-variant manifest")
    videos = payload.get("videos")
    if not isinstance(videos, list) or len(videos) != 175:
        raise ValueError("expected the frozen 175-clip ASLLVD selection")

    def materialize(row: dict[str, object]) -> str:
        current = Path(str(row["raw_path"]))
        source = Path(str(row.get("source_composite_raw_path", current)))
        source_sha = str(row.get("source_composite_sha256", row["sha256"]))
        if not source.is_file() or sha256_file(source) != source_sha:
            raise ValueError(f"composite source mismatch: {source}")
        label = str(row["canonical_label"])
        destination = args.output_root / label / source.name
        source_width = int(row.get("source_composite_width", row["width"]))
        source_height = int(row.get("source_composite_height", row["height"]))
        annotated_start = int(row["start_frame"])
        annotated_end = int(row["end_frame"])
        media = crop_front(
            source,
            destination,
            source_width,
            source_height,
            annotated_start,
            annotated_end,
        )
        row.update(
            {
                "source_composite_raw_path": str(source),
                "source_composite_sha256": source_sha,
                "source_composite_bytes": int(
                    row.get("source_composite_bytes", source.stat().st_size)
                ),
                "source_composite_width": source_width,
                "source_composite_height": source_height,
                "raw_path": str(destination),
                "sha256": media["sha256"],
                "bytes": media["bytes"],
                "width": media["width"],
                "height": media["height"],
                "fps": media["fps"],
                "frames": media["frames"],
                "front_view_crop": {
                    "source_layout": "top_front_camera_over_bottom_side_camera",
                    "region": {"x": 0, "y": 0, "width": source_width, "height": source_height // 2},
                    "aspect_policy": "exact_pixel_crop_no_resize_no_stretch",
                    "temporal_policy": "inclusive workbook Start..End interval",
                    "source_start_offset_frames": 50,
                    "annotated_start_frame": annotated_start,
                    "annotated_end_frame": annotated_end,
                    "retained_frames": annotated_end - annotated_start + 1,
                    "video_encoding": "lossless_h264_yuv444p",
                },
                "feature_path": str(
                    args.manifest.parent / "landmarks" / label / f"{destination.stem}.v17.npz"
                ),
            }
        )
        return f"{label}/{destination.name}"

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, value in enumerate(executor.map(materialize, videos), start=1):
            print(f"[{index}/{len(videos)}] {value}", flush=True)
    payload["front_view_materialized_utc"] = datetime.now(timezone.utc).isoformat()
    payload["extraction_view"] = "top_front_camera_exact_annotated_sign_interval"
    payload["extraction_aspect_policy"] = "exact_pixel_crop_no_resize_no_stretch"
    payload["extraction_temporal_policy"] = (
        "official extended movie offset 50; inclusive workbook Start..End only"
    )
    payload.pop("feature_finalized_utc", None)
    payload.pop("feature_schema_fingerprint", None)
    payload.pop("training_eligible_clips", None)
    payload.pop("feature_rejected_clips", None)
    payload.pop("extraction_rejected_clips", None)
    payload.pop("feature_errors", None)
    for row in videos:
        row["training_eligible"] = True
        row["consensus_tier"] = "official_asllex_signbank_exact"
        row.pop("feature_sha256", None)
        row.pop("feature_rejection_reason", None)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"front_view_clips": len(videos), "output_root": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
