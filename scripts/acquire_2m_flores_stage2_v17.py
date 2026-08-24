#!/usr/bin/env python3
"""Resumably acquire only the selected 2M-Flores videos and transcode one at a time."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


GIB = 1024 ** 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def probe(path: Path) -> dict[str, object]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,codec_name,duration:format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    if not payload.get("streams"):
        raise ValueError(f"no video stream: {path}")
    stream = payload["streams"][0]
    stream_duration = stream.get("duration")
    container_duration = float(payload["format"]["duration"])
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": str(stream.get("avg_frame_rate", "")),
        "codec_name": str(stream.get("codec_name", "")),
        "duration_seconds": float(stream_duration) if stream_duration is not None else container_duration,
        "duration_basis": "video_stream" if stream_duration is not None else "container_fallback",
        "container_duration_seconds": container_duration,
    }


def transcode(source: Path, destination_part: Path, encoder: str) -> None:
    destination_part.parent.mkdir(parents=True, exist_ok=True)
    destination_part.unlink(missing_ok=True)
    video_filter = (
        "scale='min(1280,iw)':'min(720,ih)':"
        "force_original_aspect_ratio=decrease:force_divisible_by=2,fps=30"
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-map", "0:v:0", "-an", "-vf", video_filter,
    ]
    if encoder == "h264_videotoolbox":
        command.extend([
            "-c:v", encoder, "-realtime", "true", "-b:v", "4000k",
            "-maxrate", "6000k", "-bufsize", "12000k",
        ])
    else:
        command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])
    command.extend(["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination_part)])
    subprocess.run(command, check=True)


def verify_derived(source_probe: dict[str, object], derived: Path) -> dict[str, object]:
    derived_probe = probe(derived)
    if int(derived_probe["width"]) > 1280 or int(derived_probe["height"]) > 720:
        raise ValueError(f"derived dimensions exceed contract: {derived_probe}")
    duration_delta = abs(
        float(source_probe["duration_seconds"]) - float(derived_probe["duration_seconds"])
    )
    if duration_delta > 0.20:
        raise ValueError(f"duration changed by {duration_delta:.3f}s")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(derived), "-f", "null", "-"],
        check=True,
    )
    return {**derived_probe, "duration_delta_seconds": duration_delta}


def initial_state(selection_path: Path, selection: dict[str, object]) -> dict[str, object]:
    return {
        "format": "slt_stage2_2m_flores_asl_acquisition_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_manifest": selection_path.as_posix(),
        "selection_manifest_sha256": sha256(selection_path),
        "source_split": "dev",
        "reserved_devtest_accessed": False,
        "encoder": None,
        "completed_rows": {},
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }


def run(args: argparse.Namespace) -> None:
    for program in ("curl", "ffmpeg", "ffprobe"):
        if shutil.which(program) is None:
            raise RuntimeError(f"required executable not found: {program}")
    selection = json.loads(args.selection.read_text())
    if selection.get("source_split") != "dev" or selection.get("reserved_devtest_accessed") is not False:
        raise ValueError("selection must be dev-only and keep devtest reserved")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    temporary_root = root / ".source_tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    state_path = root / "acquisition_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else initial_state(args.selection, selection)
    if state["selection_manifest_sha256"] != sha256(args.selection):
        raise ValueError("selection manifest changed after acquisition started")
    state["encoder"] = args.encoder
    rows = selection["rows"]
    pending = [row for row in rows if str(row["id"]) not in state["completed_rows"]]
    if args.max_rows:
        pending = pending[:args.max_rows]
    print(
        f"selected={len(rows)} complete={len(state['completed_rows'])} "
        f"this_run={len(pending)} encoder={args.encoder}", flush=True
    )
    for position, row in enumerate(pending, start=1):
        row_id = str(row["id"])
        source_bytes = int(row["source_bytes"])
        free_bytes = shutil.disk_usage(root).free
        required_free = source_bytes + int(args.reserve_gib * GIB)
        if free_bytes < required_free:
            raise RuntimeError(
                f"disk safety stop before row {row_id}: free={free_bytes / GIB:.2f} GiB, "
                f"required={required_free / GIB:.2f} GiB"
            )
        source_suffix = Path(str(row["source_path"])).suffix
        source_part = temporary_root / f"{row_id}{source_suffix}.part"
        derived_relative = Path(str(row["derived_relative_path"]))
        destination = root / derived_relative
        destination_part = destination.with_suffix(".part.mp4")
        print(
            f"[{position}/{len(pending)}] row={row_id} "
            f"source={source_bytes / (1024 ** 2):.1f} MiB", flush=True
        )
        subprocess.run([
            "curl", "--silent", "--show-error", "--location", "--fail",
            "--retry", "5", "--retry-all-errors",
            "--continue-at", "-", "--output", str(source_part), str(row["video_url"]),
        ], check=True)
        actual_source_sha = sha256(source_part)
        if actual_source_sha != row["source_sha256"]:
            raise ValueError(
                f"source hash mismatch for row {row_id}: "
                f"{actual_source_sha} != {row['source_sha256']}"
            )
        source_probe = probe(source_part)
        transcode(source_part, destination_part, args.encoder)
        derived_probe = verify_derived(source_probe, destination_part)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(destination_part, destination)
        derived_sha = sha256(destination)
        state["completed_rows"][row_id] = {
            "id": int(row["id"]),
            "source_path": row["source_path"],
            "source_bytes": source_bytes,
            "source_sha256": actual_source_sha,
            "derived_path": destination.as_posix(),
            "derived_bytes": destination.stat().st_size,
            "derived_sha256": derived_sha,
            "source_probe": source_probe,
            "derived_probe": derived_probe,
            "gloss": row["gloss"],
            "matched_locked_labels": row["matched_locked_labels"],
        }
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(state_path, state)
        source_part.unlink()
        print(
            f"  complete derived={destination.stat().st_size / (1024 ** 2):.1f} MiB "
            f"total_complete={len(state['completed_rows'])}/{len(rows)}", flush=True
        )
    print(json.dumps({
        "state": state_path.as_posix(),
        "completed_rows": len(state["completed_rows"]),
        "selected_rows": len(rows),
        "complete": len(state["completed_rows"]) == len(rows),
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", type=Path,
        default=Path("data/local/dataset_metadata/2m_flores_asl/dev_selected_v17.json"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/local/2m_flores_asl_stage2_v17"),
    )
    parser.add_argument("--encoder", choices=("h264_videotoolbox", "libx264"), default="h264_videotoolbox")
    parser.add_argument("--reserve-gib", type=float, default=12.0)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.reserve_gib < 8:
        raise ValueError("--reserve-gib must be at least 8")
    if args.max_rows < 0:
        raise ValueError("--max-rows cannot be negative")
    run(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted; partial source and completed state are resumable", file=sys.stderr)
        raise
