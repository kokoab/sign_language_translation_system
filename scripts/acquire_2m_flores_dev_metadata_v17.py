#!/usr/bin/env python3
"""Acquire only 2M-Flores-ASL ``dev`` text metadata via the dataset server.

The response's video field is reduced to its source path/URL metadata.  No video
bytes and no ``devtest`` rows are requested or written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/local/dataset_metadata/2m_flores_asl/dev_all_metadata_v17.json"
ENDPOINT = "https://datasets-server.huggingface.co/rows"
DATASET = "facebook/2M-Flores-ASL"
CONFIG = "default"
SPLIT = "dev"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch_page(offset: int, length: int, attempts: int = 5) -> dict:
    params = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "offset": offset,
        "length": length,
    }
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(ENDPOINT, params=params, timeout=90)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch offset {offset}") from last_error


def compact_row(row: dict) -> dict:
    video = row.pop("video", None)
    if isinstance(video, dict):
        row["video_path"] = video.get("path")
        row["video_url"] = video.get("src") or video.get("url")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--page-size", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.page_size <= 100:
        raise ValueError("page size must be between 1 and 100")

    first = fetch_page(0, args.page_size)
    total = int(first["num_rows_total"])
    pages = [first]
    for offset in range(args.page_size, total, args.page_size):
        pages.append(fetch_page(offset, min(args.page_size, total - offset)))
        print(f"fetched {min(offset + args.page_size, total)}/{total}", flush=True)

    rows = []
    for page in pages:
        for wrapped in page["rows"]:
            row = compact_row(dict(wrapped["row"]))
            row["row_idx"] = int(wrapped["row_idx"])
            rows.append(row)
    if len(rows) != total or len({row["row_idx"] for row in rows}) != total:
        raise RuntimeError("incomplete or duplicate metadata response")
    if any(not row.get("gloss") or not row.get("sentence") for row in rows):
        raise RuntimeError("missing required gloss/sentence metadata")

    payload = {
        "format": "2m_flores_asl_dev_text_metadata_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "config": CONFIG,
        "source_split": SPLIT,
        "endpoint": ENDPOINT,
        "rows": rows,
        "row_count": total,
        "video_bytes_downloaded": False,
        "reserved_devtest_accessed": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(f"wrote {total} text rows to {args.output}")
    print(f"sha256 {sha256_bytes(encoded)}")


if __name__ == "__main__":
    main()
