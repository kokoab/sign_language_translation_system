#!/usr/bin/env python3
"""Download the frozen YouTube-ASL pilot manifest with HTTP ZIP ranges."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import time
import zipfile
from collections import defaultdict
from pathlib import Path

from build_ios100_dataset_coverage import HTTPRangeReader


def notify(message: str) -> None:
    script = f'display notification {json.dumps(message)} with title "SLT download"'
    subprocess.run(["osascript", "-e", script], check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bitstreams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open(newline="")))
    if len(rows) != 2000 or len({row["member"] for row in rows}) != 2000:
        raise ValueError("Manifest must contain exactly 2,000 unique members")

    bitstreams = json.load(args.bitstreams.open())["_embedded"]["bitstreams"]
    archives = {
        row["name"]: (row["_links"]["content"]["href"], int(row["sizeBytes"]))
        for row in bitstreams if row["name"].endswith(".zip")
    }
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[row["shard"]].append(row["member"])
    if set(grouped) - set(archives):
        raise ValueError(f"Missing archive metadata: {sorted(set(grouped) - set(archives))}")

    args.output.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    completed = sum((args.output / member).is_file() for row in rows for member in [row["member"]])

    def write_status(state: str, detail: str = "") -> None:
        temp = args.status.with_suffix(".tmp")
        temp.write_text(json.dumps({"state": state, "completed": completed,
                                    "total": len(rows), "detail": detail}, indent=2) + "\n")
        temp.replace(args.status)

    write_status("running")
    try:
        for shard in sorted(grouped):
            url, size = archives[shard]
            attempt = 0
            while True:
                try:
                    reader = HTTPRangeReader(url, size, timeout=30)
                    with io.BufferedReader(reader, buffer_size=1024 * 1024) as buffered, \
                            zipfile.ZipFile(buffered) as archive:
                        names = set(archive.namelist())
                        missing = set(grouped[shard]) - names
                        if missing:
                            raise ValueError(f"{shard} is missing {len(missing)} manifest members")
                        members = sorted(grouped[shard], key=lambda name: archive.getinfo(name).header_offset)
                        for member in members:
                            destination = args.output / member
                            if destination.is_file():
                                continue
                            partial = destination.with_suffix(destination.suffix + ".part")
                            partial.write_bytes(archive.read(member))
                            json.loads(partial.read_text())
                            partial.replace(destination)
                            completed += 1
                            write_status("running", shard)
                            time.sleep(args.delay)
                    break
                except (OSError, RuntimeError, zipfile.BadZipFile):
                    attempt += 1
                    delay = min(300, max(30, 2 ** min(attempt, 8)))
                    write_status("retrying", f"{shard}, network attempt {attempt + 1}, waiting {delay}s")
                    time.sleep(delay)
        write_status("complete")
        notify(f"YouTube-ASL pilot complete: {completed}/2000 clips")
    except Exception as exc:
        write_status("failed", f"{type(exc).__name__}: {exc}")
        notify(f"YouTube-ASL pilot failed after {completed}/2000 clips")
        raise


if __name__ == "__main__":
    main()
