#!/usr/bin/env python3
"""Acquire a bounded channel-diverse subset from the official YouTube-ASL IDs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.acquire_openasl_transition_voices_v17 import (
    YT_DLP_JS_RUNTIME,
    atomic_json,
    download_metadata,
    ensure_yt_dlp,
    inspect_video,
    probe,
    sha256,
)


VIDEO_IDS_URL = (
    "https://storage.googleapis.com/download/storage/v1/b/gresearch/o/"
    "youtube-asl%2Fyoutube_asl_video_ids.txt?generation=1686155837105688&alt=media"
)
VIDEO_IDS_GENERATION = "1686155837105688"


def normalized_dimensions(width: int, height: int, maximum_side: int) -> tuple[int, int]:
    ratio = min(1.0, maximum_side / max(width, height))
    output_width = max(2, int(math.floor(width * ratio / 2) * 2))
    output_height = max(2, int(math.floor(height * ratio / 2) * 2))
    return output_width, output_height


def download_segment(
    video_id: str,
    output: Path,
    yt_dlp: Path,
    start: float,
    end: float,
    maximum_side: int,
    timeout: int,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="youtube_asl_v17_") as directory:
        temporary = Path(directory)
        template = temporary / "source.%(ext)s"
        subprocess.run(
            [
                str(yt_dlp), "--quiet", "--no-warnings", "--no-playlist",
                "--socket-timeout", str(timeout), "--retries", "2",
                "--fragment-retries", "2", "--js-runtimes", YT_DLP_JS_RUNTIME,
                "--format", "bv*[height<=720]+ba/b[height<=720]/b",
                "--download-sections", f"*{start:.3f}-{end:.3f}",
                "--force-keyframes-at-cuts", "--merge-output-format", "mp4",
                "--output", str(template),
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            check=True,
            timeout=timeout * 10,
        )
        sources = [path for path in temporary.glob("source.*") if path.is_file()]
        if len(sources) != 1:
            raise RuntimeError(f"expected one source artifact, found {sources}")
        source = sources[0]
        source_probe = probe(source)
        width, height = normalized_dimensions(
            int(source_probe["width"]), int(source_probe["height"]), maximum_side
        )
        normalized = temporary / "normalized.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source), "-vf",
                f"scale={width}:{height}:flags=lanczos,fps=30,format=yuv420p",
                "-an", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
                "-movflags", "+faststart", str(normalized),
            ],
            check=True,
            timeout=timeout * 5,
        )
        normalized_probe = probe(normalized)
        if normalized_probe["duration"] < 4.0:
            raise RuntimeError("downloaded signing segment is shorter than four seconds")
        normalized.replace(output)
    return {
        "path": output.as_posix(),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "probe": normalized_probe,
        "source_probe": source_probe,
        "segment_start": start,
        "segment_end": end,
    }


def discover_channels(
    video_ids: list[str],
    attempts: dict[str, object],
    target: int,
    workers: int,
    timeout: int,
    yt_dlp: Path,
    retry_failed: bool,
    state_path: Path,
    state: dict[str, object],
) -> dict[str, list[dict[str, object]]]:
    channels: dict[str, list[dict[str, object]]] = {}
    for row in state.get("discovered_channels", {}).values():
        channels.setdefault(row["channel_id"], []).extend(row["videos"])
    pending = []
    for video_id in video_ids:
        previous = attempts.get(video_id)
        if previous and not (retry_failed and previous.get("status") == "failed"):
            continue
        pending.append(video_id)
    batch_size = workers * 3
    for start in range(0, len(pending), batch_size):
        if len(channels) >= target:
            break
        batch = pending[start:start + batch_size]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(inspect_video, video_id, timeout, yt_dlp)
                for video_id in batch
            ]
            for video_id, future in zip(batch, futures):
                attempt = {"video_id": video_id}
                try:
                    info = future.result()
                    attempt.update(info)
                    channel_id = str(info["channel_id"])
                    channel_rows = channels.setdefault(channel_id, [])
                    if len(channel_rows) < 3:
                        channel_rows.append({"video_id": video_id, **info})
                    attempt["status"] = "discovered"
                except Exception as error:
                    attempt["status"] = "failed"
                    attempt["error"] = f"{type(error).__name__}: {error}"
                attempts[video_id] = attempt
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        state["discovered_channel_count"] = len(channels)
        state["discovered_channels"] = {
            channel: {
                "channel_id": channel,
                "channel": rows[0]["channel"],
                "videos": rows,
            }
            for channel, rows in channels.items()
        }
        atomic_json(state_path, state)
    return channels


def run(args: argparse.Namespace) -> dict[str, object]:
    if not 0 <= args.validation_voices < args.target_voices:
        raise ValueError("validation voices must be non-negative and below target voices")
    args.root.mkdir(parents=True, exist_ok=True)
    ensure_yt_dlp(args.yt_dlp)
    id_path = args.root / "metadata" / "youtube_asl_video_ids.txt"
    download_metadata(VIDEO_IDS_URL, id_path)
    video_ids = [line.strip() for line in id_path.read_text().splitlines() if line.strip()]
    if len(video_ids) != len(set(video_ids)):
        raise RuntimeError("official YouTube-ASL ID list contains duplicates")
    random.Random(args.seed).shuffle(video_ids)

    state_path = args.root / "acquisition_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {
            "format": "slt_youtube_asl_transition_voice_acquisition_v17",
            "version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": "official human-filtered YouTube-ASL video ID list",
            "source_generation": VIDEO_IDS_GENERATION,
            "official_split_available": False,
            "split_policy": "deterministic channel-disjoint internal train/validation",
            "attempts": {},
            "completed": {},
        }
    state["target_channel_voice_proxies"] = args.target_voices
    state["metadata"] = {
        "video_ids_path": id_path.as_posix(),
        "video_ids_sha256": sha256(id_path),
        "video_id_count": len(video_ids),
    }
    attempts = state["attempts"]
    completed = state["completed"]
    discovery_target = args.target_voices + args.discovery_reserve
    channels = discover_channels(
        video_ids,
        attempts,
        discovery_target,
        args.discovery_workers,
        args.timeout,
        args.yt_dlp,
        args.retry_failed,
        state_path,
        state,
    )
    if len(channels) < args.target_voices:
        raise RuntimeError(
            f"only discovered {len(channels)}/{args.target_voices} channel voices"
        )

    for channel_id, row in list(completed.items()):
        if (
            row.get("segment_start") == args.segment_start
            and row.get("segment_end") == args.segment_end
        ):
            continue
        video_id = str(row["video_id"])
        try:
            refreshed = download_segment(
                video_id,
                Path(row["path"]),
                args.yt_dlp,
                args.segment_start,
                args.segment_end,
                args.maximum_side,
                args.timeout,
            )
            row.update(refreshed)
        except Exception as error:
            state.setdefault("refresh_failures", {})[channel_id] = {
                "video_id": video_id,
                "error": f"{type(error).__name__}: {error}",
            }
            completed.pop(channel_id)
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        state["completed_count"] = len(completed)
        atomic_json(state_path, state)

    ordered_channels = sorted(
        channels,
        key=lambda channel: hashlib.sha256(
            f"{args.seed}:download:{channel}".encode()
        ).hexdigest(),
    )
    def acquire_channel(channel_id: str):
        failures = []
        for row in channels[channel_id]:
            video_id = str(row["video_id"])
            output = args.root / "clips" / f"{video_id}.mp4"
            try:
                acquired = download_segment(
                    video_id,
                    output,
                    args.yt_dlp,
                    args.segment_start,
                    args.segment_end,
                    args.maximum_side,
                    args.timeout,
                )
                return {
                    **row,
                    **acquired,
                    "voice_proxy": f"youtube_asl_channel:{channel_id}",
                }, failures
            except Exception as error:
                failures.append({
                    "video_id": video_id,
                    "error": f"{type(error).__name__}: {error}",
                })
        return None, failures

    pending_channels = [
        channel for channel in ordered_channels if channel not in completed
    ]
    for start in range(0, len(pending_channels), args.download_workers):
        if len(completed) >= args.target_voices:
            break
        count = min(args.download_workers, args.target_voices - len(completed))
        batch = pending_channels[start:start + count]
        with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
            futures = [executor.submit(acquire_channel, channel) for channel in batch]
            for channel_id, future in zip(batch, futures):
                acquired, failures = future.result()
                if acquired is None:
                    state.setdefault("download_failures", {})[channel_id] = failures
                else:
                    completed[channel_id] = acquired
                    state.setdefault("download_failures", {}).pop(channel_id, None)
                state["updated_utc"] = datetime.now(timezone.utc).isoformat()
                state["completed_count"] = len(completed)
                atomic_json(state_path, state)

    if len(completed) < args.target_voices:
        raise RuntimeError(
            f"only downloaded {len(completed)}/{args.target_voices} channel voices"
        )
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
    atomic_json(state_path, state)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("data/local/youtube_asl_transition_subset_v17"),
    )
    parser.add_argument("--target-voices", type=int, default=96)
    parser.add_argument("--validation-voices", type=int, default=16)
    parser.add_argument("--discovery-reserve", type=int, default=48)
    parser.add_argument("--discovery-workers", type=int, default=8)
    parser.add_argument("--download-workers", type=int, default=2)
    parser.add_argument("--segment-start", type=float, default=30.0)
    parser.add_argument("--segment-end", type=float, default=38.0)
    parser.add_argument("--maximum-side", type=int, default=720)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--seed", type=int, default=31701)
    parser.add_argument(
        "--yt-dlp", type=Path, default=Path("data/local/tools/yt-dlp_macos")
    )
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def main() -> None:
    state = run(build_parser().parse_args())
    print(json.dumps({
        "completed_count": state["completed_count"],
        "train_voice_proxies": state["train_voice_proxies"],
        "validation_voice_proxies": state["validation_voice_proxies"],
        "state": "data/local/youtube_asl_transition_subset_v17/acquisition_state.json",
    }, indent=2))


if __name__ == "__main__":
    main()
