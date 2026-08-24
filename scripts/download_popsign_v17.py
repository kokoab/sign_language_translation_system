#!/usr/bin/env python3
"""Disk-safe, resumable, participant-aware PopSign v1.0 downloader.

PopSign publishes one tar archive per sign and split. This utility intentionally
downloads only one requested archive, keeps the official archive for provenance,
and can extract a deterministic participant-balanced subset for extractor audits.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import threading
from typing import BinaryIO
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2


ARCHIVE_URL = "https://signdata.cc.gatech.edu/data/popsign_v1_0/game/{split}/{sign}.tar"
SIGN_PAGE_URL = "https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/game/{sign}/index.html"
DATASET_PAGE_URL = "https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
CHUNK_BYTES = 8 * 1024 * 1024


def participant_from_filename(filename: str, sign: str) -> str:
    marker = f"-{sign.lower()}-"
    lower = filename.lower()
    if marker in lower:
        return filename[: lower.index(marker)]
    match = re.match(r"^(.*?)-[^-]+-20\d{2}_", filename)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot derive PopSign participant from {filename!r}")


def fetch_source_map(sign: str, timeout: int = 60) -> tuple[dict[str, object], bytes]:
    url = SIGN_PAGE_URL.format(sign=sign.lower())
    page = urlopen(url, timeout=timeout).read()
    decoded = page.decode("utf-8", "replace")
    match = re.search(r"sourceMap=JSON\.parse\('(.*?)'\),", decoded, re.DOTALL)
    if not match:
        raise RuntimeError(f"PopSign sourceMap not found at {url}")
    payload = html.unescape(match.group(1))
    return json.loads(payload), payload.encode("utf-8")


def remote_archive_info(url: str, timeout: int = 60) -> dict[str, object]:
    request = Request(url, method="HEAD")
    with urlopen(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        return {
            "content_length": int(length) if length else None,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "accept_ranges": response.headers.get("Accept-Ranges", "").lower()
            == "bytes",
            "content_type": response.headers.get("Content-Type"),
        }


def ensure_disk_space(path: Path, needed_bytes: int, reserve_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path.parent).free
    required = max(0, int(needed_bytes)) + max(0, int(reserve_bytes))
    if free < required:
        raise RuntimeError(
            f"Insufficient disk space: need {required / 2**30:.2f} GiB including "
            f"reserve, have {free / 2**30:.2f} GiB at {path.parent}"
        )


def download_resumable(
    url: str,
    destination: Path,
    expected_size: int | None,
    reserve_bytes: int,
    connections: int = 4,
    timeout: int = 120,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_size is None or destination.stat().st_size == expected_size:
            return destination
        raise RuntimeError(
            f"Existing archive has wrong size: {destination.stat().st_size} != {expected_size}"
        )

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if expected_size is not None and offset > expected_size:
        raise RuntimeError(f"Partial archive is larger than remote object: {partial}")
    remaining = (expected_size - offset) if expected_size is not None else 0
    assembly_bytes = expected_size if expected_size is not None and connections > 1 else 0
    ensure_disk_space(partial, remaining + assembly_bytes, reserve_bytes)

    if expected_size is not None and connections > 1:
        return _download_parallel_ranges(
            url, destination, partial, expected_size, connections, timeout
        )

    headers = {"User-Agent": "SLT-v17-PopSign-downloader/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as exc:
        raise RuntimeError(f"PopSign download failed with HTTP {exc.code}: {url}") from exc

    status = getattr(response, "status", response.getcode())
    if offset and status != 206:
        response.close()
        raise RuntimeError(
            "Server did not honor the resume Range request; remove only the named "
            f"partial file to restart: {partial}"
        )
    mode = "ab" if offset else "wb"
    downloaded = offset
    with response, partial.open(mode) as output:
        while True:
            chunk = response.read(CHUNK_BYTES)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if expected_size:
                print(
                    f"downloaded {downloaded / 2**20:.1f}/{expected_size / 2**20:.1f} MiB",
                    flush=True,
                )
            else:
                print(f"downloaded {downloaded / 2**20:.1f} MiB", flush=True)

    if expected_size is not None and downloaded != expected_size:
        raise RuntimeError(f"Incomplete download: {downloaded} != {expected_size} bytes")
    partial.replace(destination)
    return destination


def _download_one_range(
    url: str,
    path: Path,
    start: int,
    end: int,
    timeout: int,
    print_lock: threading.Lock,
) -> Path:
    expected = end - start + 1
    existing = path.stat().st_size if path.exists() else 0
    if existing > expected:
        raise RuntimeError(f"Range part is too large: {path}")
    if existing == expected:
        return path
    request_start = start + existing
    request = Request(
        url,
        headers={
            "User-Agent": "SLT-v17-PopSign-downloader/1.0",
            "Range": f"bytes={request_start}-{end}",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", response.getcode())
        if status != 206:
            raise RuntimeError(f"Server did not honor Range {request_start}-{end}")
        with path.open("ab") as output:
            downloaded = existing
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                with print_lock:
                    print(
                        f"range {start}-{end}: {downloaded / 2**20:.1f}/{expected / 2**20:.1f} MiB",
                        flush=True,
                    )
    if path.stat().st_size != expected:
        raise RuntimeError(
            f"Incomplete range {start}-{end}: {path.stat().st_size} != {expected}"
        )
    return path


def _download_parallel_ranges(
    url: str,
    destination: Path,
    prefix: Path,
    expected_size: int,
    connections: int,
    timeout: int,
) -> Path:
    """Resume an existing prefix and fetch the remainder using bounded ranges."""
    offset = prefix.stat().st_size if prefix.exists() else 0
    if offset == expected_size:
        prefix.replace(destination)
        return destination
    parts_dir = destination.with_suffix(destination.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    remaining = expected_size - offset
    part_size = max(1, (remaining + connections - 1) // connections)
    ranges: list[tuple[int, int, Path]] = []
    for index in range(connections):
        start = offset + index * part_size
        if start >= expected_size:
            break
        end = min(expected_size - 1, start + part_size - 1)
        ranges.append((start, end, parts_dir / f"{start}-{end}.part"))
    print_lock = threading.Lock()
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = {
            executor.submit(
                _download_one_range, url, path, start, end, timeout, print_lock
            ): (start, end)
            for start, end, path in ranges
        }
        for future in as_completed(futures):
            start, end = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{start}-{end}: {exc}")

    # Some public hosts serialize connections or time out idle range requests.
    # Preserve completed bytes and finish any incomplete parts one at a time.
    if failures:
        print("parallel range fallback: " + "; ".join(failures), flush=True)
    for start, end, path in ranges:
        if not path.exists() or path.stat().st_size != end - start + 1:
            _download_one_range(url, path, start, end, timeout, print_lock)

    assembling = destination.with_suffix(destination.suffix + ".assembling")
    with assembling.open("wb") as output:
        if prefix.exists():
            with prefix.open("rb") as source:
                shutil.copyfileobj(source, output, length=CHUNK_BYTES)
        for _, _, path in ranges:
            with path.open("rb") as source:
                shutil.copyfileobj(source, output, length=CHUNK_BYTES)
    if assembling.stat().st_size != expected_size:
        raise RuntimeError(
            f"Assembled archive has wrong size: {assembling.stat().st_size} != {expected_size}"
        )
    assembling.replace(destination)
    if prefix.exists():
        prefix.unlink()
    for _, _, path in ranges:
        path.unlink()
    parts_dir.rmdir()
    return destination


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def safe_video_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    videos: list[tarfile.TarInfo] = []
    seen: set[str] = set()
    for member in archive.getmembers():
        if not member.isfile() or member.issym() or member.islnk():
            continue
        basename = Path(member.name).name
        if not basename or basename in (".", ".."):
            continue
        if Path(basename).suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if basename in seen:
            raise RuntimeError(f"Duplicate video basename in archive: {basename}")
        seen.add(basename)
        videos.append(member)
    return videos


def source_records(source_map: dict[str, object], split: str, sign: str) -> dict[str, dict[str, str]]:
    split_map = source_map.get(split)
    if not isinstance(split_map, dict):
        raise RuntimeError(f"No {split!r} source map for PopSign {sign!r}")
    previous = split_map.get("prev_name", {})
    original = split_map.get("orig_name", {})
    if not isinstance(previous, dict) or not isinstance(original, dict):
        raise RuntimeError("Unexpected PopSign sourceMap format")
    records: dict[str, dict[str, str]] = {}
    for key, archive_name in previous.items():
        original_name = str(original[key])
        records[Path(str(archive_name)).name] = {
            "original_name": Path(original_name).name,
            "participant": participant_from_filename(original_name, sign),
        }
    return records


def select_balanced_members(
    members: list[tarfile.TarInfo],
    records: dict[str, dict[str, str]],
    samples_per_participant: int,
) -> list[tuple[tarfile.TarInfo, dict[str, str]]]:
    grouped: dict[str, list[tuple[tarfile.TarInfo, dict[str, str]]]] = defaultdict(list)
    for member in members:
        basename = Path(member.name).name
        if basename not in records:
            raise RuntimeError(f"Archive member absent from official sourceMap: {basename}")
        record = records[basename]
        grouped[record["participant"]].append((member, record))
    selected: list[tuple[tarfile.TarInfo, dict[str, str]]] = []
    for participant in sorted(grouped, key=str.casefold):
        choices = sorted(grouped[participant], key=lambda item: item[1]["original_name"])
        if samples_per_participant > 0:
            choices = choices[:samples_per_participant]
        selected.extend(choices)
    return selected


def _copy_stream(source: BinaryIO, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("wb") as output:
        shutil.copyfileobj(source, output, length=CHUNK_BYTES)
    temporary.replace(destination)


def extract_selection(
    archive_path: Path,
    output_dir: Path,
    records: dict[str, dict[str, str]],
    samples_per_participant: int,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, object]] = []
    with tarfile.open(archive_path, "r:*") as archive:
        members = safe_video_members(archive)
        selected = select_balanced_members(members, records, samples_per_participant)
        for member, record in selected:
            destination = output_dir / record["original_name"]
            if destination.exists() and destination.stat().st_size == member.size:
                pass
            else:
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Could not read archive member {member.name}")
                with source:
                    _copy_stream(source, destination)
            capture = cv2.VideoCapture(str(destination))
            ok, frame = capture.read()
            frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            rotation = int(round(capture.get(cv2.CAP_PROP_ORIENTATION_META)))
            capture.release()
            if not ok or frame is None:
                raise RuntimeError(f"Extracted video does not decode: {destination}")
            height, width = frame.shape[:2]
            extracted.append(
                {
                    "path": str(destination),
                    "participant": record["participant"],
                    "original_name": record["original_name"],
                    "archive_name": Path(member.name).name,
                    "bytes": member.size,
                    "frame_count": frame_count,
                    "fps": fps,
                    "decoded_width": width,
                    "decoded_height": height,
                    "rotation_metadata_degrees": rotation,
                }
            )
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one official PopSign v1.0 archive")
    parser.add_argument("sign", help="lowercase PopSign gloss, for example thankyou")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--archive-root", type=Path, default=Path("data/local/popsign_v17_archives")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/local/popsign_v17_raw")
    )
    parser.add_argument(
        "--samples-per-participant",
        type=int,
        default=2,
        help="0 extracts all clips; default 2 is intended for orientation audits",
    )
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=2.0,
        help="free-space reserve in addition to remaining archive bytes",
    )
    parser.add_argument(
        "--connections",
        type=int,
        default=4,
        help="bounded parallel HTTP ranges; set 1 for a single resumable stream",
    )
    args = parser.parse_args()
    sign = args.sign.strip().lower()
    if not re.fullmatch(r"[a-z0-9_]+", sign):
        raise SystemExit("sign must contain only lowercase letters, numbers, or underscore")
    if args.samples_per_participant < 0:
        raise SystemExit("--samples-per-participant cannot be negative")
    if not 1 <= args.connections <= 8:
        raise SystemExit("--connections must be between 1 and 8")

    url = ARCHIVE_URL.format(split=args.split, sign=sign)
    info = remote_archive_info(url)
    expected_size = info["content_length"]
    archive_path = args.archive_root / args.split / f"{sign}.tar"
    source_map, source_map_bytes = fetch_source_map(sign)
    records = source_records(source_map, args.split, sign)
    archive_path = download_resumable(
        url,
        archive_path,
        int(expected_size) if expected_size is not None else None,
        int(args.reserve_gib * 2**30),
        connections=args.connections,
    )
    archive_sha256 = sha256_file(archive_path)

    extracted: list[dict[str, object]] = []
    if not args.download_only:
        output_dir = args.output_root / args.split / sign.upper()
        extracted = extract_selection(
            archive_path, output_dir, records, args.samples_per_participant
        )
    participants = sorted({str(row["participant"]) for row in extracted})
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "PopSign v1.0 game dataset",
        "dataset_page": DATASET_PAGE_URL,
        "license": "CC BY 4.0",
        "license_url": LICENSE_URL,
        "sign": sign,
        "split": args.split,
        "archive_url": url,
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "remote": info,
        "source_map_url": SIGN_PAGE_URL.format(sign=sign),
        "source_map_sha256": hashlib.sha256(source_map_bytes).hexdigest(),
        "official_source_map_video_count": len(records),
        "samples_per_participant": args.samples_per_participant,
        "extracted_video_count": len(extracted),
        "extracted_participant_count": len(participants),
        "participants": participants,
        "videos": extracted,
    }
    provenance_path = (
        args.output_root / args.split / sign.upper() / "_popsign_provenance.json"
        if not args.download_only
        else archive_path.with_suffix(".provenance.json")
    )
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "archive": str(archive_path),
                "archive_bytes": archive_path.stat().st_size,
                "archive_sha256": archive_sha256,
                "extracted_videos": len(extracted),
                "participants": len(participants),
                "provenance": str(provenance_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
