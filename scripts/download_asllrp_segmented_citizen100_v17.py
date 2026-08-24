#!/usr/bin/env python3
"""Download and verify exact-variant ASLLRP segmented sign candidates.

The candidate CSVs must be produced by
``prepare_asllrp_continuous_citizen100_v17.py``.  Files are streamed to temporary
paths, atomically installed, decoded, hashed, and recorded.  RIT stays physically
separate as a held-out external evaluation source.
"""

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
import subprocess
import time
from typing import Any
import urllib.request


USER_AGENT = "SLT-v17-ASLLRP-segmented-acquisition/1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, Any]:
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
    if not payload.get("streams"):
        raise ValueError(f"no video stream: {path}")
    stream = payload["streams"][0]
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    duration = float(payload.get("format", {}).get("duration", 0))
    if width <= 0 or height <= 0 or duration <= 0:
        raise ValueError(f"invalid decoded video metadata: {path}")
    return {
        "codec": stream.get("codec_name"),
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "frame_rate": stream.get("avg_frame_rate"),
        "frame_count": (
            int(stream["nb_frames"])
            if stream.get("nb_frames") not in {None, "", "N/A"}
            else None
        ),
    }


def read_candidates(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            current = list(csv.DictReader(handle))
        if not current:
            raise ValueError(f"candidate CSV is empty: {path}")
        required = {
            "source",
            "split_role",
            "canonical_label",
            "sign_video_filename",
            "sign_video_url",
            "asllrp_video_id",
        }
        missing = required - set(current[0])
        if missing:
            raise ValueError(f"candidate CSV is missing {sorted(missing)}: {path}")
        rows.extend(current)
    keys = [(row["source"], row["sign_video_filename"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("candidate CSVs contain duplicate source/video pairs")
    return rows


def download(
    row: dict[str, str], output_root: Path, timeout: int, retries: int
) -> dict[str, Any]:
    split = row["split_role"]
    if split not in {"train_candidate", "external_evaluation_reserved"}:
        raise ValueError(f"unexpected split role: {split}")
    filename = Path(row["sign_video_filename"]).name
    if filename != row["sign_video_filename"] or not filename.lower().endswith(".mp4"):
        raise ValueError(f"unsafe sign filename: {row['sign_video_filename']!r}")
    destination = (
        output_root
        / split
        / row["source"]
        / row["canonical_label"]
        / filename
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    status = "existing"
    if not destination.exists():
        error: Exception | None = None
        for attempt in range(retries):
            temporary = destination.with_suffix(destination.suffix + ".part")
            try:
                request = urllib.request.Request(
                    row["sign_video_url"], headers={"User-Agent": USER_AGENT}
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    content_type = response.headers.get_content_type()
                    if not content_type.startswith("video/"):
                        raise ValueError(
                            f"unexpected content type {content_type}: {row['sign_video_url']}"
                        )
                    with temporary.open("wb") as handle:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            handle.write(block)
                if temporary.stat().st_size <= 0:
                    raise ValueError(f"empty download: {row['sign_video_url']}")
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
            raise RuntimeError(f"failed to download {row['sign_video_url']}: {error}") from error
    media = probe_video(destination)
    return {
        **row,
        "status": status,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        **media,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    report = Path("artifacts/reports/asllrp_continuous_citizen100_v17")
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        action="append",
        default=None,
        help="May be repeated; defaults to ASLLRP train candidates plus RIT evaluation reserve.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/local/asllrp_segmented_citizen100_v17"),
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 1:
        parser.error("workers and retries must be positive")
    paths = args.candidate_csv or [
        report / "stage1_asllrp_train_candidates.csv",
        report / "stage1_rit_external_eval_candidates.csv",
    ]
    candidates = read_candidates(paths)
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download, row, args.output_root, args.timeout, args.retries): row
            for row in candidates
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                failures.append(
                    {
                        "source": row["source"],
                        "filename": row["sign_video_filename"],
                        "error": str(exc),
                    }
                )
            if index % 100 == 0 or index == len(futures):
                print(
                    f"processed {index}/{len(futures)}; failures={len(failures)}",
                    flush=True,
                )
    completed.sort(
        key=lambda row: (
            int(row["class_index"]),
            str(row["split_role"]),
            str(row["source"]),
            str(row["sign_video_filename"]),
        )
    )
    manifest = {
        "format": "slt_v17_asllrp_segmented_sign_acquisition",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_csvs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
        "license": "ASLLRP research-only noncommercial terms; do not redistribute videos",
        "expected_videos": len(candidates),
        "verified_videos": len(completed),
        "failures": failures,
        "bytes": sum(int(row["bytes"]) for row in completed),
        "duration_seconds": sum(float(row["duration_seconds"]) for row in completed),
        "classes": len({row["canonical_label"] for row in completed}),
        "participants": sorted({row["signer_id"] for row in completed}),
        "split_counts": dict(Counter(row["split_role"] for row in completed)),
        "videos": completed,
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
                "verified": len(completed),
                "failures": len(failures),
                "bytes": manifest["bytes"],
            },
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
