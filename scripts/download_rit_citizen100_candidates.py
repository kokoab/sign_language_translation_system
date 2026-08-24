#!/usr/bin/env python3
"""Download exact-name RIT Sign Bank candidates for the frozen Citizen100.

The ASLLRP/RIT collection and ASL Citizen use different lexical identifiers.
This downloader therefore quarantines every acquired clip. A pinned raw-gloss
match is stronger than a canonical-label-only match, but neither becomes
training-eligible until exact lexical-variant review is complete.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import time
from urllib.request import Request, urlopen
import zipfile
import zlib

import cv2


METADATA_URL = (
    "https://dai.cs.rutgers.edu/asllvd/signbank/rit_signs_2024_06_27.csv"
)
ARCHIVES = (
    {
        "name": "batch_signs_v1_1.zip",
        "url": "https://dai.cs.rutgers.edu/asllvd/signbank/rit/batch_signs_v1_1.zip",
        "size": 1_999_858_147,
    },
    {
        "name": "batch_signs_v1_2.zip",
        "url": "https://dai.cs.rutgers.edu/asllvd/signbank/rit/batch_signs_v1_2.zip",
        "size": 1_773_522_775,
    },
)
USER_AGENT = "SLT-v17-RIT-Citizen100-audit/1.0"


def fetch(
    url: str,
    *,
    byte_range: tuple[int, int] | None = None,
    timeout: int = 120,
    retries: int = 4,
) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if byte_range is not None:
                expected = byte_range[1] - byte_range[0] + 1
                if len(payload) != expected:
                    raise RuntimeError(
                        f"range returned {len(payload)} bytes, expected {expected}"
                    )
            return payload
        except Exception as exc:  # Network failures are retried with bounded delay.
            error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}: {error}") from error


class HTTPRangeReader(io.RawIOBase):
    """Minimal seekable HTTP reader for ZIP central-directory inspection."""

    def __init__(self, url: str, size: int, timeout: int):
        self.url = url
        self.size = size
        self.timeout = timeout
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        if size == 0 or self.position >= self.size:
            return b""
        end = min(self.size - 1, self.position + size - 1)
        payload = fetch(
            self.url,
            byte_range=(self.position, end),
            timeout=self.timeout,
        )
        self.position += len(payload)
        return payload


def parse_metadata(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    rows = list(csv.DictReader(text.splitlines()))
    required = {
        "Video ID number",
        "main entry gloss label",
        "entry/variant gloss label",
        "occurrence label",
        "full video filename",
        "Sign clip video filename",
        "sign type",
    }
    if not rows or not required.issubset(rows[0]):
        missing = sorted(required - (set(rows[0]) if rows else set()))
        raise ValueError(f"RIT metadata is missing required columns: {missing}")
    return rows


def participant_id(row: dict[str, str]) -> str:
    match = re.match(r"^(P\d+)_", row["full video filename"].strip())
    if not match:
        raise ValueError(
            f"could not derive participant from {row['full video filename']!r}"
        )
    return match.group(1)


def select_candidates(
    manifest: dict[str, object], rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    raw_lookup: dict[str, dict[str, object]] = {}
    canonical_lookup: dict[str, dict[str, object]] = {}
    for item in manifest["classes"]:
        raw_key = str(item["citizen_raw_gloss"]).strip().casefold()
        canonical_key = str(item["canonical_label"]).strip().casefold()
        if raw_key in raw_lookup or canonical_key in canonical_lookup:
            raise ValueError("frozen manifest contains duplicate exact gloss labels")
        raw_lookup[raw_key] = item
        canonical_lookup[canonical_key] = item

    selected: list[dict[str, object]] = []
    seen_clips: set[str] = set()
    for row in rows:
        variant = row["entry/variant gloss label"].strip()
        key = variant.casefold()
        item = raw_lookup.get(key)
        match_tier = "pinned_raw_gloss_exact"
        if item is None:
            item = canonical_lookup.get(key)
            match_tier = "canonical_label_only"
        if item is None:
            continue
        clip = row["Sign clip video filename"].strip()
        if not clip.lower().endswith(".mp4"):
            raise ValueError(f"unexpected RIT clip filename: {clip!r}")
        if clip in seen_clips:
            raise ValueError(f"RIT clip selected more than once: {clip}")
        seen_clips.add(clip)
        selected.append(
            {
                "class_index": int(item["class_index"]),
                "canonical_label": str(item["canonical_label"]),
                "citizen_raw_gloss": str(item["citizen_raw_gloss"]),
                "citizen_asl_lex_code": str(item["citizen_asl_lex_code"]),
                "rit_main_entry": row["main entry gloss label"].strip(),
                "rit_entry_variant": variant,
                "rit_occurrence": row["occurrence label"].strip(),
                "rit_video_id": row["Video ID number"].strip(),
                "rit_sign_type": row["sign type"].strip(),
                "participant": participant_id(row),
                "full_video_filename": row["full video filename"].strip(),
                "clip_filename": clip,
                "match_tier": match_tier,
                "training_eligible": False,
            }
        )
    return sorted(
        selected,
        key=lambda row: (
            int(row["class_index"]),
            str(row["participant"]),
            str(row["clip_filename"]),
        ),
    )


def build_archive_index(timeout: int) -> dict[str, tuple[dict[str, object], zipfile.ZipInfo]]:
    index: dict[str, tuple[dict[str, object], zipfile.ZipInfo]] = {}
    for archive in ARCHIVES:
        with zipfile.ZipFile(
            HTTPRangeReader(str(archive["url"]), int(archive["size"]), timeout)
        ) as handle:
            for info in handle.infolist():
                if info.is_dir():
                    continue
                if info.filename in index:
                    raise ValueError(f"duplicate member across RIT archives: {info.filename}")
                index[info.filename] = (archive, info)
    return index


def read_remote_member(
    archive: dict[str, object], info: zipfile.ZipInfo, timeout: int
) -> bytes:
    # Fetch the local header plus a bounded extra-field allowance and compressed data.
    allowance = 4096
    start = info.header_offset
    end = start + zipfile.sizeFileHeader + len(info.filename.encode("utf-8")) + allowance + info.compress_size - 1
    end = min(int(archive["size"]) - 1, end)
    block = fetch(str(archive["url"]), byte_range=(start, end), timeout=timeout)
    header = struct.unpack(zipfile.structFileHeader, block[: zipfile.sizeFileHeader])
    if header[zipfile._FH_SIGNATURE] != zipfile.stringFileHeader:
        raise zipfile.BadZipFile(f"invalid local header for {info.filename}")
    data_start = (
        zipfile.sizeFileHeader
        + header[zipfile._FH_FILENAME_LENGTH]
        + header[zipfile._FH_EXTRA_FIELD_LENGTH]
    )
    compressed = block[data_start : data_start + info.compress_size]
    if len(compressed) != info.compress_size:
        absolute_start = info.header_offset + data_start
        compressed = fetch(
            str(archive["url"]),
            byte_range=(absolute_start, absolute_start + info.compress_size - 1),
            timeout=timeout,
        )
    if info.compress_type == zipfile.ZIP_STORED:
        payload = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        payload = zlib.decompress(compressed, -15)
    else:
        raise ValueError(
            f"unsupported ZIP compression {info.compress_type}: {info.filename}"
        )
    if len(payload) != info.file_size:
        raise RuntimeError(
            f"size mismatch for {info.filename}: {len(payload)} != {info.file_size}"
        )
    if (zlib.crc32(payload) & 0xFFFFFFFF) != info.CRC:
        raise RuntimeError(f"CRC mismatch for {info.filename}")
    return payload


def validate_video(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if not ok or frame is None or frames < 1:
        raise RuntimeError(f"video did not decode: {path}")
    height, width = frame.shape[:2]
    return {
        "decoded_width": width,
        "decoded_height": height,
        "frames": frames,
        "fps": fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json")
    )
    parser.add_argument(
        "--metadata-cache",
        type=Path,
        default=Path("data/local/dataset_metadata/rit_signs_2024_06_27.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/local/rit_citizen100_variant_audit/raw"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--refresh-metadata", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    if args.refresh_metadata or not args.metadata_cache.exists():
        metadata_payload = fetch(METADATA_URL, timeout=args.timeout)
        args.metadata_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.metadata_cache.with_suffix(args.metadata_cache.suffix + ".part")
        temporary.write_bytes(metadata_payload)
        temporary.replace(args.metadata_cache)
    else:
        metadata_payload = args.metadata_cache.read_bytes()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    candidates = select_candidates(manifest, parse_metadata(metadata_payload))
    archive_index = build_archive_index(args.timeout)
    missing = sorted(
        str(row["clip_filename"])
        for row in candidates
        if str(row["clip_filename"]) not in archive_index
    )
    if missing:
        raise KeyError(f"{len(missing)} selected RIT clips are absent from archives: {missing[:5]}")

    tier_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    for row in candidates:
        tier = str(row["match_tier"])
        label = str(row["canonical_label"])
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        class_counts[label] = class_counts.get(label, 0) + 1
    print(
        json.dumps(
            {
                "candidate_clips": len(candidates),
                "candidate_classes": len(class_counts),
                "tier_counts": tier_counts,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    def acquire(row: dict[str, object]) -> dict[str, object]:
        clip = str(row["clip_filename"])
        archive, info = archive_index[clip]
        destination = args.output_root / str(row["canonical_label"]) / clip
        if destination.exists():
            payload = destination.read_bytes()
            status = "existing"
            if len(payload) != info.file_size or (zlib.crc32(payload) & 0xFFFFFFFF) != info.CRC:
                raise RuntimeError(f"existing file does not match ZIP metadata: {destination}")
        else:
            payload = read_remote_member(archive, info, args.timeout)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            status = "downloaded"
        return {
            **row,
            "status": status,
            "destination": str(destination),
            "source_archive": str(archive["name"]),
            "source_archive_url": str(archive["url"]),
            "compressed_bytes": info.compress_size,
            "bytes": len(payload),
            "zip_crc32": f"{info.CRC:08x}",
            "sha256": hashlib.sha256(payload).hexdigest(),
            **validate_video(destination),
        }

    completed: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(acquire, row): row for row in candidates}
        for count, future in enumerate(as_completed(futures), start=1):
            completed.append(future.result())
            if count % 25 == 0 or count == len(candidates):
                print(f"acquired {count}/{len(candidates)} RIT candidates", flush=True)

    completed.sort(
        key=lambda row: (
            int(row["class_index"]),
            str(row["participant"]),
            str(row["clip_filename"]),
        )
    )
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "ASLLRP/RIT isolated signs",
        "source_metadata_url": METADATA_URL,
        "source_metadata_sha256": hashlib.sha256(metadata_payload).hexdigest(),
        "license_warning": (
            "ASLLRP Sign Bank data are research-only, noncommercial, and "
            "non-redistributable under the source terms."
        ),
        "training_eligible": False,
        "eligibility_warning": (
            "Exact text matches do not prove ASL-LEX variant identity. Confirm each "
            "RIT entry/variant against the pinned Citizen raw gloss and ASL-LEX code."
        ),
        "candidate_class_count": len(class_counts),
        "candidate_video_count": len(completed),
        "tier_counts": tier_counts,
        "class_counts": dict(sorted(class_counts.items())),
        "videos": completed,
    }
    provenance_path = args.output_root.parent / "candidate_provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"provenance": str(provenance_path), "videos": len(completed)}, indent=2))


if __name__ == "__main__":
    main()
