#!/usr/bin/env python3
"""Select and resumably acquire a bounded multi-signer How2Sign train subset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import pyarrow.parquet as pq


REPO_ID = "martinctl/how2sign-asl-clips"
REVISION = "cfe9b6482aa34d6f6bda1974a7b7cae822c16613"
METADATA_URL = (
    f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/metadata.parquet"
)
SIGNER_PATTERN = re.compile(r"-(\d+)-rgb_front$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def download_file(url: str, destination: Path, timeout: int, retries: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(retries + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "SLT-v17-bounded-acquisition/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = Request(url, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                append = offset > 0 and status == 206
                if offset and not append:
                    offset = 0
                with partial.open("ab" if append else "wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            partial.replace(destination)
            return destination.stat().st_size
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError("unreachable")


def ensure_metadata(root: Path, args: argparse.Namespace) -> Path:
    path = root / "metadata.parquet"
    if not path.exists():
        download_file(METADATA_URL, path, args.timeout, args.retries)
    return path


def signer_id(row: dict[str, Any]) -> str:
    match = SIGNER_PATTERN.search(str(row["sentence_name"]))
    if not match:
        raise ValueError(f"cannot infer signer from {row['sentence_name']!r}")
    return match.group(1)


def select_rows(metadata: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = pq.read_table(metadata).to_pylist()
    if {str(row["split"]) for row in rows} != {"train"}:
        raise ValueError("mirror metadata unexpectedly contains a non-train split")
    eligible = [
        row for row in rows
        if args.minimum_duration <= float(row["duration"]) <= args.maximum_duration
    ]
    by_signer_video: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in eligible:
        by_signer_video[signer_id(row)][str(row["video_id"])].append(row)

    selected = []
    for signer in sorted(by_signer_video, key=lambda value: int(value)):
        rng = random.Random(args.seed + int(signer))
        videos = sorted(by_signer_video[signer])
        rng.shuffle(videos)
        signer_rows = []
        depth = 0
        while len(signer_rows) < args.target_per_signer:
            added = False
            for video in videos:
                candidates = sorted(
                    by_signer_video[signer][video], key=lambda row: row["sentence_id"]
                )
                if depth < min(args.maximum_per_video, len(candidates)):
                    signer_rows.append(candidates[depth])
                    added = True
                    if len(signer_rows) >= args.target_per_signer:
                        break
            if not added:
                break
            depth += 1
        for row in signer_rows:
            selected.append({
                "sentence_id": str(row["sentence_id"]),
                "sentence_name": str(row["sentence_name"]),
                "signer_id": signer,
                "video_id": str(row["video_id"]),
                "duration": float(row["duration"]),
                "sentence": str(row["sentence"]),
                "split": "train",
                "repo_path": str(row["file_name"]),
                "local_path": str(Path("clips") / Path(str(row["file_name"])).name),
            })
    return selected


def validate_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,duration",
        "-of", "json", path.as_posix(),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    streams = payload.get("streams", [])
    if len(streams) != 1 or int(streams[0].get("width", 0)) <= 0:
        raise ValueError(f"{path}: missing valid video stream")
    return streams[0]


def clip_url(repo_path: str) -> str:
    encoded = quote(repo_path, safe="/")
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{encoded}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    metadata = ensure_metadata(root, args)
    rows = select_rows(metadata, args)
    plan = {
        "format": "how2sign_transition_subset_plan_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_repo": REPO_ID,
        "source_revision": REVISION,
        "source_license": "CC BY-NC 4.0",
        "source_card": f"https://huggingface.co/datasets/{REPO_ID}",
        "official_dataset": "https://how2sign.github.io/",
        "metadata_sha256": sha256(metadata),
        "selection": {
            "split": "train only",
            "seed": args.seed,
            "minimum_duration": args.minimum_duration,
            "maximum_duration": args.maximum_duration,
            "target_per_signer": args.target_per_signer,
            "maximum_per_source_video": args.maximum_per_video,
        },
        "rows": rows,
        "row_count": len(rows),
        "signer_counts": dict(sorted(Counter(row["signer_id"] for row in rows).items())),
        "source_video_count": len({row["video_id"] for row in rows}),
        "duration_hours": sum(row["duration"] for row in rows) / 3600.0,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
    }
    atomic_json(root / "selection_plan.json", plan)
    if args.plan_only:
        return plan

    lock = threading.Lock()
    progress = {
        "format": "how2sign_transition_subset_acquisition_state_v17",
        "source_revision": REVISION,
        "planned": len(rows),
        "completed": 0,
        "failed": 0,
        "bytes": 0,
        "failures": [],
    }

    def acquire(row: dict[str, Any]) -> dict[str, Any]:
        path = root / row["local_path"]
        if not path.exists():
            download_file(clip_url(row["repo_path"]), path, args.timeout, args.retries)
        probe = validate_video(path)
        return {
            "sentence_id": row["sentence_id"],
            "path": row["local_path"],
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "probe": probe,
        }

    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(acquire, row): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
                completed.append(result)
                with lock:
                    progress["completed"] += 1
                    progress["bytes"] += int(result["bytes"])
            except Exception as error:
                with lock:
                    progress["failed"] += 1
                    progress["failures"].append({
                        "sentence_id": row["sentence_id"],
                        "error": f"{type(error).__name__}: {error}",
                    })
            progress["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(root / "acquisition_state.json", progress)
    completed.sort(key=lambda row: row["sentence_id"])
    atomic_json(root / "completed_files.json", {
        "files": completed,
        "file_count": len(completed),
        "total_bytes": sum(int(row["bytes"]) for row in completed),
    })
    if progress["failed"]:
        raise RuntimeError(
            f"{progress['failed']} How2Sign clips failed; rerun resumes completed files"
        )
    return {**plan, "acquisition": progress}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/local/how2sign_transition_subset_v17"),
    )
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--minimum-duration", type=float, default=2.0)
    parser.add_argument("--maximum-duration", type=float, default=12.0)
    parser.add_argument("--target-per-signer", type=int, default=400)
    parser.add_argument("--maximum-per-video", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps({
        "row_count": result["row_count"],
        "signer_counts": result["signer_counts"],
        "source_video_count": result["source_video_count"],
        "duration_hours": result["duration_hours"],
        "acquisition": result.get("acquisition"),
    }, indent=2))


if __name__ == "__main__":
    main()
