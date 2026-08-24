#!/usr/bin/env python3
"""Acquire a bounded, channel-diverse OpenASL train-only transition subset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import urllib.request
import urllib.parse

import cv2
OPENASL_REVISION = "c7d2350b22f344c5a6669ad37518b493c8f78822"
RAW_ROOT = "https://raw.githubusercontent.com/chevalierNoir/OpenASL"
TSV_URL = f"{RAW_ROOT}/{OPENASL_REVISION}/data/openasl-v1.0.tsv"
BBOX_URL = f"{RAW_ROOT}/{OPENASL_REVISION}/data/bbox-v1.0.json"
YT_DLP_VERSION = "2026.08.19"
YT_DLP_URL = (
    "https://github.com/yt-dlp/yt-dlp/releases/download/"
    f"{YT_DLP_VERSION}/yt-dlp_macos"
)
YT_DLP_SHA256 = "0f192b7ec147ab6288885d6351d9ab67367640029b4377576ef46dd79cf7b202"
YT_DLP_JS_RUNTIME = "node:/opt/homebrew/bin/node"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def download_metadata(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    temporary.replace(path)


def ensure_yt_dlp(path: Path) -> None:
    if path.exists() and sha256(path) == YT_DLP_SHA256:
        os.chmod(path, 0o755)
        return
    download_metadata(YT_DLP_URL, path)
    if sha256(path) != YT_DLP_SHA256:
        raise RuntimeError("yt-dlp binary hash mismatch")
    os.chmod(path, 0o755)


def timestamp_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def load_candidates(
    tsv_path: Path,
    bboxes: dict[str, list[float]],
    minimum_duration: float,
    maximum_duration: float,
    seed: int,
) -> list[dict[str, object]]:
    by_video: dict[str, list[dict[str, str]]] = {}
    with tsv_path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["split"] != "train" or row["vid"] not in bboxes:
                continue
            start = timestamp_seconds(row["start"])
            stop = timestamp_seconds(row["end"])
            duration = stop - start
            if minimum_duration <= duration <= maximum_duration:
                by_video.setdefault(row["yid"], []).append(row)
    candidates = []
    for video_id, rows in by_video.items():
        row = min(
            rows,
            key=lambda item: hashlib.sha256(
                f"{seed}:{item['vid']}".encode()
            ).hexdigest(),
        )
        candidates.append({
            "vid": row["vid"],
            "video_id": video_id,
            "start": timestamp_seconds(row["start"]),
            "end": timestamp_seconds(row["end"]),
            "duration": timestamp_seconds(row["end"]) - timestamp_seconds(row["start"]),
            "raw_text": row["raw-text"],
            "bbox": bboxes[row["vid"]],
        })
    random.Random(seed).shuffle(candidates)
    return candidates


def inspect_video(video_id: str, timeout: int, yt_dlp: Path) -> dict[str, object]:
    del yt_dlp
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    url = (
        "https://www.youtube.com/oembed?url="
        + urllib.parse.quote(watch_url, safe="")
        + "&format=json"
    )
    with urllib.request.urlopen(url, timeout=timeout) as response:
        info = json.load(response)
    return {
        "channel_id": info.get("author_url") or info.get("author_name"),
        "channel": info.get("author_name"),
        "availability": "public_oembed",
        "live_status": "not_live",
        "source_duration": None,
    }


def probe(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if width < 1 or height < 1 or fps <= 0 or frames < 2:
        raise RuntimeError(f"invalid downloaded video: {path}")
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration": frames / fps,
    }


def square_crop_geometry(
    width: int, height: int, normalized: list[float]
) -> tuple[int, int, int, int, int, int, int]:
    x0, y0, x1, y1 = (
        normalized[0] * width,
        normalized[1] * height,
        normalized[2] * width,
        normalized[3] * height,
    )
    side = max(x1 - x0, y1 - y0)
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    left = math.floor(center_x - side / 2)
    top = math.floor(center_y - side / 2)
    right = math.ceil(center_x + side / 2)
    bottom = math.ceil(center_y + side / 2)
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)
    size = max(right - left, bottom - top)
    return pad_left, pad_top, pad_right, pad_bottom, left + pad_left, top + pad_top, size


def download_and_crop(
    row: dict[str, object], output: Path, maximum_side: int, timeout: int,
    yt_dlp: Path,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="openasl_v17_") as directory:
        temporary = Path(directory)
        template = temporary / "source.%(ext)s"
        command = [
            str(yt_dlp),
            "--quiet",
            "--no-warnings",
            "--no-playlist",
            "--socket-timeout",
            str(timeout),
            "--retries",
            "2",
            "--fragment-retries",
            "2",
            "--js-runtimes",
            YT_DLP_JS_RUNTIME,
            "--format",
            "bv*[height<=720]+ba/b[height<=720]/b",
            "--download-sections",
            f"*{row['start']:.3f}-{row['end']:.3f}",
            "--force-keyframes-at-cuts",
            "--merge-output-format",
            "mp4",
            "--output",
            str(template),
            f"https://www.youtube.com/watch?v={row['video_id']}",
        ]
        subprocess.run(command, check=True, timeout=timeout * 8)
        sources = [path for path in temporary.glob("source.*") if path.is_file()]
        if len(sources) != 1:
            raise RuntimeError(f"expected one downloaded source, found {sources}")
        source = sources[0]
        source_probe = probe(source)
        geometry = square_crop_geometry(
            int(source_probe["width"]), int(source_probe["height"]), row["bbox"]
        )
        pad_left, pad_top, pad_right, pad_bottom, crop_x, crop_y, crop_size = geometry
        scale = min(maximum_side, crop_size)
        scale -= scale % 2
        video_filter = (
            f"pad=iw+{pad_left + pad_right}:ih+{pad_top + pad_bottom}:"
            f"{pad_left}:{pad_top}:black,"
            f"crop={crop_size}:{crop_size}:{crop_x}:{crop_y},"
            f"scale={scale}:{scale}:flags=lanczos,fps=30,format=yuv420p"
        )
        normalized = temporary / "normalized.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source), "-vf", video_filter, "-an",
                "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
                "-movflags", "+faststart", str(normalized),
            ],
            check=True,
            timeout=timeout * 4,
        )
        normalized_probe = probe(normalized)
        if normalized_probe["duration"] < 2.0:
            raise RuntimeError("normalized clip is too short")
        normalized.replace(output)
    return {
        "path": output.as_posix(),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "probe": normalized_probe,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if not 0 <= args.validation_voices < args.target_voices:
        raise ValueError("validation voices must be non-negative and below target voices")
    args.root.mkdir(parents=True, exist_ok=True)
    ensure_yt_dlp(args.yt_dlp)
    metadata_root = args.root / "metadata"
    tsv_path = metadata_root / "openasl-v1.0.tsv"
    bbox_path = metadata_root / "bbox-v1.0.json"
    download_metadata(TSV_URL, tsv_path)
    download_metadata(BBOX_URL, bbox_path)
    bboxes = json.loads(bbox_path.read_text())
    candidates = load_candidates(
        tsv_path, bboxes, args.minimum_duration, args.maximum_duration, args.seed
    )

    state_path = args.root / "acquisition_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {
            "format": "slt_openasl_transition_voice_acquisition_v17",
            "version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": OPENASL_REVISION,
            "source_split": "train",
            "openasl_validation_video_accessed": False,
            "openasl_test_video_accessed": False,
            "target_channel_voice_proxies": args.target_voices,
            "attempts": {},
            "completed": {},
        }
    completed = state["completed"]
    attempts = state["attempts"]
    state["target_channel_voice_proxies"] = args.target_voices
    if any(not key.startswith("https://") for key in completed):
        migrated = {}
        for old_key, row in completed.items():
            try:
                info = inspect_video(str(row["video_id"]), args.timeout, args.yt_dlp)
                channel_key = str(info["channel_id"])
                row.update(info)
                row["previous_channel_id"] = old_key
            except Exception:
                channel_key = old_key
            if channel_key not in migrated:
                migrated[channel_key] = row
            else:
                state.setdefault("migrated_duplicate_downloads", []).append(row)
        completed.clear()
        completed.update(migrated)
    selected_channels = set(completed)

    pending = []
    for row in candidates[:args.maximum_candidates]:
        video_id = str(row["video_id"])
        if video_id in attempts and not (
            args.retry_failed and attempts[video_id].get("status") == "failed"
        ):
            continue
        pending.append(row)
    batch_size = args.discovery_workers * 2
    for batch_start in range(0, len(pending), batch_size):
        if len(selected_channels) >= args.target_voices:
            break
        batch = pending[batch_start:batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=args.discovery_workers) as executor:
            futures = [
                executor.submit(
                    inspect_video, str(row["video_id"]), args.timeout, args.yt_dlp
                )
                for row in batch
            ]
            discovered = []
            for row, future in zip(batch, futures):
                try:
                    discovered.append((row, future.result(), None))
                except Exception as error:
                    discovered.append((row, None, error))
        for row, info, discovery_error in discovered:
            video_id = str(row["video_id"])
            attempt = {"vid": row["vid"], "video_id": video_id}
            if video_id in attempts:
                attempt["prior_attempt"] = attempts[video_id]
            try:
                if discovery_error is not None:
                    raise discovery_error
                attempt.update(info)
                channel_id = info["channel_id"]
                if not channel_id:
                    raise RuntimeError("source has no channel identity")
                if channel_id in selected_channels:
                    attempt["status"] = "duplicate_channel"
                elif len(selected_channels) >= args.target_voices:
                    attempt["status"] = "target_already_reached"
                elif info["live_status"] not in (None, "not_live"):
                    attempt["status"] = "non_static_video"
                else:
                    output = args.root / "clips" / f"{row['vid']}.mp4"
                    acquired = download_and_crop(
                        row, output, args.maximum_side, args.timeout, args.yt_dlp
                    )
                    role = (
                        "validation"
                        if len(selected_channels) < args.validation_voices
                        else "train"
                    )
                    completed[channel_id] = {
                        **row,
                        **info,
                        **acquired,
                        "role": role,
                        "voice_proxy": f"openasl_channel:{channel_id}",
                    }
                    selected_channels.add(channel_id)
                    attempt["status"] = "downloaded"
            except Exception as error:
                attempt["status"] = "failed"
                attempt["error"] = f"{type(error).__name__}: {error}"
            attempts[video_id] = attempt
            state["updated_utc"] = datetime.now(timezone.utc).isoformat()
            state["completed_count"] = len(completed)
            atomic_json(state_path, state)

    if len(completed) < args.target_voices:
        raise RuntimeError(
            f"only acquired {len(completed)}/{args.target_voices} channel voices"
        )
    # Freeze role allocation by channel hash after acquisition; this is independent of
    # video content and guarantees the requested validation count exactly.
    held_out = set(sorted(
        completed,
        key=lambda channel: hashlib.sha256(
            f"{args.seed}:holdout:{channel}".encode()
        ).hexdigest(),
    )[:args.validation_voices])
    for channel, row in completed.items():
        row["role"] = "validation" if channel in held_out else "train"
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    state["completed_count"] = len(completed)
    state["train_voice_proxies"] = len(completed) - len(held_out)
    state["validation_voice_proxies"] = len(held_out)
    state["metadata"] = {
        "tsv_path": tsv_path.as_posix(),
        "tsv_sha256": sha256(tsv_path),
        "bbox_path": bbox_path.as_posix(),
        "bbox_sha256": sha256(bbox_path),
    }
    atomic_json(state_path, state)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("data/local/openasl_transition_subset_v17"),
    )
    parser.add_argument("--target-voices", type=int, default=96)
    parser.add_argument("--validation-voices", type=int, default=16)
    parser.add_argument("--maximum-candidates", type=int, default=600)
    parser.add_argument("--minimum-duration", type=float, default=4.0)
    parser.add_argument("--maximum-duration", type=float, default=12.0)
    parser.add_argument("--maximum-side", type=int, default=720)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--seed", type=int, default=21701)
    parser.add_argument(
        "--yt-dlp", type=Path, default=Path("data/local/tools/yt-dlp_macos")
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--discovery-workers", type=int, default=6)
    return parser


def main() -> None:
    state = run(build_parser().parse_args())
    print(json.dumps({
        "completed_count": state["completed_count"],
        "train_voice_proxies": state["train_voice_proxies"],
        "validation_voice_proxies": state["validation_voice_proxies"],
        "state": "data/local/openasl_transition_subset_v17/acquisition_state.json",
    }, indent=2))


if __name__ == "__main__":
    main()
