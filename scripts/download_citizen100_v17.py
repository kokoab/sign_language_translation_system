#!/usr/bin/env python3
"""Selectively download the exact ASL Citizen members in the v17 manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import threading
import time
import zipfile
import zlib

try:
    from .build_ios100_dataset_coverage import (
        ASL_CITIZEN_SIZE,
        ASL_CITIZEN_URL,
        HTTPRangeReader,
        fetch,
    )
except ImportError:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from build_ios100_dataset_coverage import (  # type: ignore
        ASL_CITIZEN_SIZE,
        ASL_CITIZEN_URL,
        HTTPRangeReader,
        fetch,
    )


SPLITS = ("train", "val", "test")
LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
LOCAL_FILE_SIGNATURE = 0x04034B50


def load_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("name") != "citizen100_v17_manifest":
        raise ValueError(f"Unexpected manifest type in {path}")
    classes = manifest.get("classes", [])
    if len(classes) != 100:
        raise ValueError(f"Expected 100 manifest classes, got {len(classes)}")
    return manifest


def load_selected_rows(
    manifest: dict[str, object], cache_dir: Path
) -> list[dict[str, object]]:
    source_to_class = {
        (item["citizen_raw_gloss"], item["citizen_asl_lex_code"]): item
        for item in manifest["classes"]
    }
    rows: list[dict[str, object]] = []
    seen_videos: set[str] = set()
    for split in SPLITS:
        path = cache_dir / f"asl_citizen_{split}.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["Gloss"].strip(), row["ASL-LEX Code"].strip())
                selected_class = source_to_class.get(key)
                if selected_class is None:
                    continue
                video = row["Video file"].strip()
                if video in seen_videos:
                    raise ValueError(f"Video occurs more than once in selected data: {video}")
                seen_videos.add(video)
                rows.append(
                    {
                        "split": split,
                        "class_index": selected_class["class_index"],
                        "canonical_label": selected_class["canonical_label"],
                        "raw_gloss": key[0],
                        "asl_lex_code": key[1],
                        "participant": row["Participant ID"].strip(),
                        "video": video,
                        "archive_member": f"ASL_Citizen/videos/{video}",
                    }
                )

    actual: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["canonical_label"]), str(row["split"]))
        actual[key] = actual.get(key, 0) + 1
    for item in manifest["classes"]:
        for split in SPLITS:
            expected = int(item["video_counts"][split])
            count = actual.get((str(item["canonical_label"]), split), 0)
            if count != expected:
                raise ValueError(
                    f"Metadata count mismatch for {item['canonical_label']}/{split}: "
                    f"{count} != {expected}"
                )
    return sorted(
        rows,
        key=lambda item: (SPLITS.index(str(item["split"])), int(item["class_index"]), str(item["video"])),
    )


def attach_zip_info(
    rows: list[dict[str, object]], output_root: Path
) -> tuple[list[dict[str, object]], dict[str, zipfile.ZipInfo]]:
    with zipfile.ZipFile(HTTPRangeReader(ASL_CITIZEN_URL, ASL_CITIZEN_SIZE)) as archive:
        infos = {info.filename: info for info in archive.infolist()}
    plan: list[dict[str, object]] = []
    selected_infos: dict[str, zipfile.ZipInfo] = {}
    for row in rows:
        member = str(row["archive_member"])
        info = infos.get(member)
        if info is None:
            raise KeyError(f"Official archive member not found: {member}")
        if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise ValueError(f"Unsupported ZIP compression {info.compress_type}: {member}")
        selected_infos[member] = info
        destination = (
            output_root
            / str(row["split"])
            / str(row["canonical_label"])
            / str(row["video"])
        )
        plan.append(
            {
                **row,
                "destination": str(destination),
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
                "zip_crc32": f"{info.CRC:08x}",
                "zip_header_offset": info.header_offset,
                "zip_compression": info.compress_type,
            }
        )
    return plan, selected_infos


def compressed_data_range(info: zipfile.ZipInfo) -> tuple[int, int]:
    header = fetch(
        ASL_CITIZEN_URL,
        byte_range=(info.header_offset, info.header_offset + LOCAL_FILE_HEADER.size - 1),
    )
    if len(header) != LOCAL_FILE_HEADER.size:
        raise RuntimeError(f"Incomplete local ZIP header for {info.filename}")
    fields = LOCAL_FILE_HEADER.unpack(header)
    if fields[0] != LOCAL_FILE_SIGNATURE:
        raise RuntimeError(f"Invalid local ZIP signature for {info.filename}")
    filename_length, extra_length = fields[-2], fields[-1]
    start = info.header_offset + LOCAL_FILE_HEADER.size + filename_length + extra_length
    return start, start + info.compress_size - 1


def decode_member(info: zipfile.ZipInfo, compressed: bytes) -> bytes:
    if len(compressed) != info.compress_size:
        raise RuntimeError(
            f"Compressed size mismatch for {info.filename}: "
            f"{len(compressed)} != {info.compress_size}"
        )
    if info.compress_type == zipfile.ZIP_STORED:
        payload = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"Unsupported compression type: {info.compress_type}")
    if len(payload) != info.file_size:
        raise RuntimeError(
            f"Uncompressed size mismatch for {info.filename}: {len(payload)} != {info.file_size}"
        )
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    if crc != info.CRC:
        raise RuntimeError(f"CRC mismatch for {info.filename}: {crc:08x} != {info.CRC:08x}")
    return payload


def download_member(
    row: dict[str, object], info: zipfile.ZipInfo
) -> dict[str, object]:
    destination = Path(str(row["destination"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != info.file_size:
            raise RuntimeError(
                f"Existing destination has wrong size; refusing to overwrite: {destination}"
            )
        payload = destination.read_bytes()
        if (zlib.crc32(payload) & 0xFFFFFFFF) != info.CRC:
            raise RuntimeError(
                f"Existing destination has wrong CRC; refusing to overwrite: {destination}"
            )
        status = "existing"
    else:
        start, end = compressed_data_range(info)
        compressed = fetch(ASL_CITIZEN_URL, byte_range=(start, end))
        payload = decode_member(info, compressed)
        temporary = destination.with_suffix(
            destination.suffix + f".part.{threading.get_ident()}"
        )
        temporary.write_bytes(payload)
        temporary.replace(destination)
        status = "downloaded"
    return {
        **row,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "status": status,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/local/dataset_metadata")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/local/citizen100_v17/raw")
    )
    parser.add_argument(
        "--plan", type=Path, default=Path("artifacts/reports/citizen100_v17_download_plan.csv")
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reserve-gib", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")

    manifest = load_manifest(args.manifest)
    selected = load_selected_rows(manifest, args.cache_dir)
    plan, infos = attach_zip_info(selected, args.output_root)
    write_csv(args.plan, plan)
    compressed = sum(int(row["compressed_bytes"]) for row in plan)
    uncompressed = sum(int(row["uncompressed_bytes"]) for row in plan)
    existing = sum(
        int(row["uncompressed_bytes"])
        for row in plan
        if Path(str(row["destination"])).exists()
        and Path(str(row["destination"])).stat().st_size == int(row["uncompressed_bytes"])
    )
    summary = {
        "videos": len(plan),
        "compressed_transfer_gib": compressed / 2**30,
        "uncompressed_output_gib": uncompressed / 2**30,
        "existing_output_gib": existing / 2**30,
        "remaining_output_gib": (uncompressed - existing) / 2**30,
        "plan": str(args.plan),
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.output_root).free
    needed = uncompressed - existing + int(args.reserve_gib * 2**30)
    if free < needed:
        raise SystemExit(
            f"Insufficient disk space: need {needed / 2**30:.2f} GiB including "
            f"reserve; have {free / 2**30:.2f} GiB"
        )

    started = time.perf_counter()
    completed: list[dict[str, object]] = []
    failures: list[str] = []
    progress_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_member, row, infos[str(row["archive_member"])]): row
            for row in plan
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                failures.append(f"{row['archive_member']}: {exc}")
            if index == 1 or index % 25 == 0 or index == len(plan):
                with progress_lock:
                    elapsed = time.perf_counter() - started
                    print(
                        f"[{index}/{len(plan)}] ok={len(completed)} failed={len(failures)} "
                        f"rate={index / max(elapsed, 1e-6):.2f}/s",
                        flush=True,
                    )

    completed.sort(
        key=lambda item: (SPLITS.index(str(item["split"])), int(item["class_index"]), str(item["video"]))
    )
    provenance = args.output_root.parent / "provenance.csv"
    if completed:
        write_csv(provenance, completed)
    result = {
        **summary,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": ASL_CITIZEN_URL,
        "manifest": str(args.manifest),
        "completed": len(completed),
        "failed": len(failures),
        "failures": failures,
        "provenance": str(provenance),
    }
    summary_path = args.output_root.parent / "download_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"completed": len(completed), "failed": len(failures)}, indent=2))
    raise SystemExit(0 if not failures and len(completed) == len(plan) else 1)


if __name__ == "__main__":
    main()
