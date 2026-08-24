#!/usr/bin/env python3
"""Finish Kaggle blob uploads in bounded GCS-resumable chunks.

The official Kaggle CLI streams the whole remaining file in one request. On the
current network, connection resets make requests/urllib3 reread the stream while its
progress bar advances, so the displayed percentage can greatly exceed bytes accepted
by the server. This helper reuses the official CLI's saved resumable URL, queries the
authoritative GCS offset, and sends 256 KiB chunks. It never prints the signed URL.
After it finishes, rerun ``kaggle datasets create`` so the official CLI creates the
dataset record from the completed upload token.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


CHUNK_BYTES = 256 * 1024
PART_TEMPLATE = (
    "artifacts/generated/kaggle_movinet_v17/dataset_part_{part}/"
    "movinet_v17_trainval.tar.part-{part}"
)


def parse_headers(text: str) -> tuple[int, int | None]:
    statuses = [int(value) for value in re.findall(r"^HTTP/\S+\s+(\d+)", text, re.MULTILINE)]
    if not statuses:
        raise RuntimeError("upload server returned no HTTP status")
    ranges = re.findall(r"^range:\s*bytes=0-(\d+)\s*$", text, re.MULTILINE | re.IGNORECASE)
    accepted = int(ranges[-1]) + 1 if ranges else None
    return statuses[-1], accepted


def find_state(upload_file: Path) -> tuple[Path, dict]:
    temp_root = Path(tempfile.gettempdir()) / ".kaggle/uploads"
    target = upload_file.resolve()
    for state_path in temp_root.glob("*.json"):
        try:
            payload = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        recorded = Path(payload.get("path", ""))
        if not recorded.is_absolute():
            recorded = (Path.cwd() / recorded).resolve()
        if recorded == target:
            return state_path, payload
    raise FileNotFoundError(f"no Kaggle resumable state for {upload_file}")


def curl_headers(command: list[str]) -> tuple[int, int | None]:
    with tempfile.NamedTemporaryFile() as headers:
        completed = subprocess.run(
            ["curl", "--silent", "--show-error", "--output", os.devnull,
             "--dump-header", headers.name, *command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=90,
            check=False,
        )
        headers.seek(0)
        text = headers.read().decode("latin-1")
    if not text:
        raise RuntimeError(f"curl failed before HTTP response: {completed.stderr.strip()}")
    return parse_headers(text)


def query_offset(url: str, total: int) -> tuple[bool, int]:
    status, accepted = curl_headers(
        ["--request", "PUT", "--header", "Content-Length: 0",
         "--header", f"Content-Range: bytes */{total}",
         "--connect-timeout", "20", "--max-time", "60", url]
    )
    if status in (200, 201):
        return True, total
    if status != 308:
        raise RuntimeError(f"unexpected resumable-query HTTP status {status}")
    return False, accepted or 0


def send_chunk(url: str, upload_file: Path, start: int, total: int) -> tuple[bool, int]:
    length = min(CHUNK_BYTES, total - start)
    end = start + length - 1
    with upload_file.open("rb") as source, tempfile.NamedTemporaryFile() as chunk:
        source.seek(start)
        payload = source.read(length)
        if len(payload) != length:
            raise RuntimeError(f"short read at {start}: {len(payload)} != {length}")
        chunk.write(payload)
        chunk.flush()
        status, accepted = curl_headers(
            ["--upload-file", chunk.name,
             "--header", f"Content-Length: {length}",
             "--header", f"Content-Range: bytes {start}-{end}/{total}",
             "--connect-timeout", "20", "--max-time", "60", url]
        )
    if status in (200, 201):
        return True, total
    if status != 308:
        raise RuntimeError(f"unexpected chunk HTTP status {status} at {start}")
    return False, accepted if accepted is not None else start + length


def finish_part(part: str, max_stalls: int) -> dict:
    upload_file = Path(PART_TEMPLATE.format(part=part))
    if not upload_file.is_file():
        raise FileNotFoundError(upload_file)
    _, state = find_state(upload_file)
    response = state.get("start_blob_upload_response") or {}
    url = response.get("createUrl")
    if not url:
        raise RuntimeError(f"part {part}: resumable state has no createUrl")
    total = upload_file.stat().st_size
    complete, offset = query_offset(url, total)
    initial_offset = offset
    print(f"part {part}: accepted={offset}/{total}", flush=True)
    stalls = 0
    last_report = offset
    while not complete:
        try:
            complete, new_offset = send_chunk(url, upload_file, offset, total)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            time.sleep(min(1.0 + stalls * 0.25, 5.0))
            complete, new_offset = query_offset(url, total)
            print(f"part {part}: recovered after {type(error).__name__}", flush=True)
        if new_offset < offset or new_offset > total:
            raise RuntimeError(f"part {part}: invalid server offset {new_offset}")
        if new_offset == offset:
            stalls += 1
            if stalls > max_stalls:
                raise RuntimeError(f"part {part}: no server progress after {stalls} attempts")
        else:
            stalls = 0
            offset = new_offset
        if offset - last_report >= 4 * 1024 * 1024 or complete:
            print(f"part {part}: accepted={offset}/{total} ({100 * offset / total:.1f}%)", flush=True)
            last_report = offset
    return {"part": part, "initial_offset": initial_offset, "total": total, "complete": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="+", help="two-digit part IDs, e.g. 00 01")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-stalls", type=int, default=20)
    args = parser.parse_args()
    if any(not re.fullmatch(r"\d{2}", part) for part in args.parts):
        parser.error("parts must be two digits")

    results = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(args.parts))) as executor:
        futures = {executor.submit(finish_part, part, args.max_stalls): part for part in args.parts}
        for future in as_completed(futures):
            results.append(future.result())
    print(json.dumps(sorted(results, key=lambda row: row["part"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
