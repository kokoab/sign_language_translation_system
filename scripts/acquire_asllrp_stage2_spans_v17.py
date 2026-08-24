#!/usr/bin/env python3
"""Acquire and crop exact contiguous Citizen100 phrase spans from ASLLRP utterances."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
import urllib.request


USER_AGENT = "SLT-v17-ASLLRP-stage2-span-acquisition/1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    frames = stream.get("nb_frames")
    result = {
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream.get("avg_frame_rate"),
        "frames": int(frames) if frames not in {None, "", "N/A"} else None,
        "duration_seconds": float(payload["format"]["duration"]),
    }
    if result["width"] <= 0 or result["height"] <= 0 or result["duration_seconds"] <= 0:
        raise ValueError(f"invalid video: {path}")
    return result


def read_spans(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"span CSV is empty: {path}")
    required = {
        "source",
        "split_role",
        "signer_id",
        "utterance_video_filename",
        "utterance_video_url",
        "span_index_in_utterance",
        "target_sequence",
        "crop_start_frame_local",
        "crop_end_frame_local",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"span CSV is missing fields: {sorted(missing)}")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        filename = Path(row["utterance_video_filename"]).name
        if filename != row["utterance_video_filename"] or not filename.lower().endswith(".mp4"):
            raise ValueError(f"unsafe utterance filename: {row['utterance_video_filename']!r}")
        sequence = json.loads(row["target_sequence"])
        start = int(row["crop_start_frame_local"])
        end = int(row["crop_end_frame_local"])
        if not isinstance(sequence, list) or len(sequence) < 2 or start < 0 or end < start:
            raise ValueError(f"invalid span row: {row}")
        parsed.append(
            {
                **row,
                "target_sequence": [str(value) for value in sequence],
                "target_variants": json.loads(row["target_variants"]),
                "span_index_in_utterance": int(row["span_index_in_utterance"]),
                "target_token_count": int(row["target_token_count"]),
                "crop_start_frame_local": start,
                "crop_end_frame_local": end,
            }
        )
    return parsed


def acquire_parent(row: dict[str, Any], transport: Path, timeout: int, retries: int) -> dict[str, Any]:
    destination = transport / row["source"] / row["utterance_video_filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    status = "existing"
    if not destination.exists():
        error: Exception | None = None
        for attempt in range(retries):
            temporary = destination.with_suffix(destination.suffix + ".part")
            try:
                request = urllib.request.Request(
                    row["utterance_video_url"], headers={"User-Agent": USER_AGENT}
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    content_type = response.headers.get_content_type()
                    if not content_type.startswith("video/"):
                        raise ValueError(f"unexpected content type: {content_type}")
                    with temporary.open("wb") as handle:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            handle.write(block)
                os.replace(temporary, destination)
                status = "downloaded"
                error = None
                break
            except Exception as exc:
                error = exc
                if temporary.exists():
                    temporary.unlink()
                if attempt + 1 < retries:
                    time.sleep(1.5 * (attempt + 1))
        if error is not None:
            raise RuntimeError(f"failed to download {row['utterance_video_url']}: {error}") from error
    media = probe(destination)
    return {
        "source": row["source"],
        "filename": row["utterance_video_filename"],
        "url": row["utterance_video_url"],
        "status": status,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        **media,
    }


def sequence_slug(sequence: list[str]) -> str:
    return "_".join(re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_") for value in sequence)


def crop_span(row: dict[str, Any], parent: dict[str, Any], output_root: Path) -> dict[str, Any]:
    start = int(row["crop_start_frame_local"])
    end = int(row["crop_end_frame_local"])
    frames = parent["frames"]
    if frames is not None and end >= int(frames):
        raise ValueError(
            f"crop exceeds parent frames: {row['utterance_video_filename']} {start}-{end}/{frames}"
        )
    basename = Path(row["utterance_video_filename"]).stem
    destination = (
        output_root
        / row["split_role"]
        / row["source"]
        / sequence_slug(row["target_sequence"])
        / f"{basename}_span{int(row['span_index_in_utterance']):02d}.mp4"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    status = "existing"
    if not destination.exists():
        temporary = destination.with_suffix(".part.mp4")
        expression = f"select=between(n\\,{start}\\,{end}),setpts=N/FRAME_RATE/TB"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                parent["path"],
                "-vf",
                expression,
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(temporary),
            ],
            check=True,
        )
        os.replace(temporary, destination)
        status = "created"
    media = probe(destination)
    expected = end - start + 1
    if media["frames"] is not None and int(media["frames"]) != expected:
        raise ValueError(
            f"cropped frame mismatch: {destination}: {media['frames']} != {expected}"
        )
    return {
        **row,
        "status": status,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "parent_sha256": parent["sha256"],
        **media,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spans",
        type=Path,
        default=Path(
            "artifacts/reports/asllrp_continuous_citizen100_v17/"
            "stage2_contiguous_target_spans.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/local/asllrp_contiguous_phrases_v17"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    spans = read_spans(args.spans)
    unique_parents: dict[tuple[str, str], dict[str, Any]] = {}
    for row in spans:
        unique_parents.setdefault((row["source"], row["utterance_video_filename"]), row)
    parents: dict[tuple[str, str], dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    transport = args.output_root / "transport"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(acquire_parent, row, transport, args.timeout, args.retries): key
            for key, row in unique_parents.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                parents[key] = future.result()
            except Exception as exc:
                failures.append({"source": key[0], "filename": key[1], "error": str(exc)})
    completed: list[dict[str, Any]] = []
    if not failures:
        for row in spans:
            key = (row["source"], row["utterance_video_filename"])
            try:
                completed.append(crop_span(row, parents[key], args.output_root / "spans"))
            except Exception as exc:
                failures.append(
                    {
                        "source": row["source"],
                        "filename": row["utterance_video_filename"],
                        "error": str(exc),
                    }
                )
                break
    manifest = {
        "format": "slt_v17_asllrp_contiguous_target_spans",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_span_csv": str(args.spans),
        "source_span_csv_sha256": sha256_file(args.spans),
        "variant_and_crop_contract": (
            "official ASL-LEX exact variants; every non-target visible annotation breaks a span; "
            "manual first/last sign bounds plus five context frames"
        ),
        "expected_parent_videos": len(unique_parents),
        "verified_parent_videos": len(parents),
        "expected_spans": len(spans),
        "verified_spans": len(completed),
        "failures": failures,
        "parent_bytes": sum(int(row["bytes"]) for row in parents.values()),
        "span_bytes": sum(int(row["bytes"]) for row in completed),
        "span_duration_seconds": sum(float(row["duration_seconds"]) for row in completed),
        "split_counts": dict(Counter(row["split_role"] for row in completed)),
        "participants": sorted({row["signer_id"] for row in completed}),
        "parents": sorted(parents.values(), key=lambda row: (row["source"], row["filename"])),
        "spans": completed,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(output),
                "parents": len(parents),
                "spans": len(completed),
                "failures": len(failures),
            },
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
