#!/usr/bin/env python3
"""Download selected ZIP members with one persistent HTTP range request each."""

from __future__ import annotations

import argparse
import binascii
import csv
import io
import json
import struct
import time
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path

import requests

from build_ios100_dataset_coverage import HTTPRangeReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bitstreams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open(newline="")))
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[row["shard"]].append(row["member"])
    streams = json.load(args.bitstreams.open())["_embedded"]["bitstreams"]
    archives = {row["name"]: (row["_links"]["content"]["href"], int(row["sizeBytes"]))
                for row in streams if row["name"].endswith(".zip")}
    args.output.mkdir(parents=True, exist_ok=True)
    completed = sum((args.output / row["member"]).is_file() for row in rows)
    downloaded = 0

    def status(state: str, detail: str = "") -> None:
        temp = args.status.with_suffix(".tmp")
        temp.write_text(json.dumps({"state": state, "completed": completed,
                                    "total": len(rows), "detail": detail}, indent=2) + "\n")
        temp.replace(args.status)

    session = requests.Session()
    session.headers["User-Agent"] = "SLT-YouTube-ASL-pilot/1.0"
    status("running")
    for shard in sorted(grouped):
        url, size = archives[shard]
        reader = HTTPRangeReader(url, size, timeout=30)
        with io.BufferedReader(reader, buffer_size=1024 * 1024) as buffered, zipfile.ZipFile(buffered) as archive:
            infos = [archive.getinfo(name) for name in grouped[shard]]
        for info in sorted(infos, key=lambda value: value.header_offset):
            destination = args.output / info.filename
            if destination.is_file():
                continue
            if args.limit is not None and downloaded >= args.limit:
                status("limited", f"downloaded {downloaded} new files")
                return
            # Local ZIP extra fields can differ from the central directory. Fetch the
            # maximum possible extra field so the compressed payload is always present.
            end = info.header_offset + 30 + len(info.filename.encode("utf-8")) + 65535 + info.compress_size - 1
            for attempt in range(20):
                try:
                    response = session.get(url, headers={"Range": f"bytes={info.header_offset}-{end}"},
                                           timeout=(15, 90))
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise RuntimeError(f"range ignored: HTTP {response.status_code}")
                    payload = response.content
                    signature, _, _, compression, _, _, _, _, _, name_len, extra_len = struct.unpack(
                        "<IHHHHHIIIHH", payload[:30])
                    if signature != 0x04034B50:
                        raise RuntimeError("invalid local ZIP header")
                    start = 30 + name_len + extra_len
                    compressed = payload[start:start + info.compress_size]
                    if len(compressed) != info.compress_size:
                        raise RuntimeError("short compressed member")
                    if compression == zipfile.ZIP_STORED:
                        data = compressed
                    elif compression == zipfile.ZIP_DEFLATED:
                        data = zlib.decompress(compressed, -15)
                    else:
                        raise RuntimeError(f"unsupported ZIP compression {compression}")
                    if len(data) != info.file_size or binascii.crc32(data) & 0xFFFFFFFF != info.CRC:
                        raise RuntimeError("ZIP size or CRC mismatch")
                    json.loads(data)
                    partial = destination.with_suffix(destination.suffix + ".part")
                    partial.write_bytes(data)
                    partial.replace(destination)
                    completed += 1
                    downloaded += 1
                    status("running", shard)
                    break
                except (OSError, requests.RequestException, RuntimeError, zlib.error):
                    if attempt == 19:
                        raise
                    time.sleep(min(60, 2 ** min(attempt, 6)))
    status("complete")


if __name__ == "__main__":
    main()
