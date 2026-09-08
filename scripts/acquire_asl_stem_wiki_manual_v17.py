#!/usr/bin/env python3
"""Range-download a provisional ASL STEM Wiki manual-review pool."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import threading
import urllib.request
import zlib


SOURCE_URL = (
    "https://download.microsoft.com/download/4/c/f/"
    "4cfec788-7478-4e47-9a15-ace9b6a96198/ASL_STEM_Wiki.zip"
)
SOURCE_SIZE = 201_193_250_549
PROVISIONAL_DOWNLOAD_PARTICIPANTS = frozenset(("P12", "P13", "P21", "P28", "P33", "P35"))
CENTRAL_HEADER = struct.Struct("<4s6H3L5H2L")
LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")


def build_candidates(manual_rows, metadata, raw_to_canonical, candidate_participants):
    """Keep one unambiguous target-bearing sentence per provisional participant."""
    grouped = defaultdict(list)
    rejected = []
    for row in manual_rows:
        filename = row.get("filename", "").strip()
        if not filename:
            rejected.append({"reason": "missing filename"})
            continue
        source = metadata.get(filename)
        if source is None:
            raise ValueError(f"manual filename absent from source metadata: {filename}")
        participant = source["participant"]
        if participant not in candidate_participants:
            rejected.append({"filename": filename, "reason": "participant outside provisional download pool"})
            continue
        gloss = row.get("GLOSSED SENTENCE", "")
        depth = 0
        for char in gloss:
            depth += (char == "(") - (char == ")")
            if depth < 0:
                break
        if depth:
            rejected.append({"filename": filename, "reason": "unbalanced gloss annotation"})
            continue
        tokens = gloss.split()
        if not tokens:
            rejected.append({"filename": filename, "reason": "empty gloss sequence"})
            continue
        grouped[filename].append((row, source, tokens))

    candidates = []
    for filename, rows in sorted(grouped.items()):
        sequences = {tuple(tokens) for _, _, tokens in rows}
        if len(sequences) != 1:
            rejected.append({"filename": filename, "reason": "conflicting duplicate annotation"})
            continue
        row, source, tokens = rows[0]
        ctc = [raw_to_canonical.get(token, "OTHER") for token in tokens]
        target_count = sum(token != "OTHER" for token in ctc)
        if not target_count:
            rejected.append({"filename": filename, "reason": "no locked raw-gloss token"})
            continue
        candidates.append(
            {
                "filename": filename,
                "participant": source["participant"],
                "source_user": row.get("user", ""),
                "article": row.get("articleName", ""),
                "section_index": row.get("sectionIndex", ""),
                "sentence_index": row.get("sentenceIndex", ""),
                "duration_seconds": float(source["video length"]),
                "english_sentence": row.get("sentence", ""),
                "gloss_tokens": tokens,
                "ctc_sequence": ctc,
                "locked_raw_gloss_tokens": target_count,
                "source_path": f"ASL_STEM_Wiki/videos/{filename}",
                "role": "candidate_verification",
                "training_eligible": False,
                "signer_quality_verified": False,
                "variant_verified": False,
            }
        )
    return candidates, rejected


def parse_central_directory(path):
    payload = Path(path).read_bytes()
    position = 0
    result = {}
    while position < len(payload):
        fields = CENTRAL_HEADER.unpack_from(payload, position)
        if fields[0] != b"PK\x01\x02":
            raise ValueError(f"invalid central-directory signature at {position}")
        filename_length, extra_length, comment_length = fields[10:13]
        start = position + CENTRAL_HEADER.size
        filename = payload[start : start + filename_length].decode("utf-8")
        extra = payload[start + filename_length : start + filename_length + extra_length]
        compressed_size, uncompressed_size, offset = fields[8], fields[9], fields[16]
        if 0xFFFFFFFF in (compressed_size, uncompressed_size, offset):
            cursor = 0
            while cursor + 4 <= len(extra):
                tag, length = struct.unpack_from("<HH", extra, cursor)
                data = extra[cursor + 4 : cursor + 4 + length]
                cursor += 4 + length
                if tag != 1:
                    continue
                values = iter(struct.unpack("<" + "Q" * (len(data) // 8), data))
                if uncompressed_size == 0xFFFFFFFF:
                    uncompressed_size = next(values)
                if compressed_size == 0xFFFFFFFF:
                    compressed_size = next(values)
                if offset == 0xFFFFFFFF:
                    offset = next(values)
                break
        result[filename] = {
            "compression": fields[4],
            "crc32": fields[7],
            "compressed_bytes": compressed_size,
            "uncompressed_bytes": uncompressed_size,
            "header_offset": offset,
        }
        position = start + filename_length + extra_length + comment_length
    return result


def request_range(start, end):
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "SLT-v17-ASL-STEM/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def local_data_range(info):
    header = request_range(info["header_offset"], info["header_offset"] + LOCAL_HEADER.size - 1)
    fields = LOCAL_HEADER.unpack(header)
    if fields[0] != 0x04034B50:
        raise ValueError("invalid local ZIP header")
    start = info["header_offset"] + LOCAL_HEADER.size + fields[-2] + fields[-1]
    return start, start + info["compressed_bytes"] - 1


def decode_member(info, compressed):
    if len(compressed) != info["compressed_bytes"]:
        raise ValueError("compressed member size mismatch")
    if info["compression"] == 0:
        decoded = compressed
    elif info["compression"] == 8:
        decoded = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"unsupported ZIP compression: {info['compression']}")
    if len(decoded) != info["uncompressed_bytes"]:
        raise ValueError("decoded member size mismatch")
    if zlib.crc32(decoded) & 0xFFFFFFFF != info["crc32"]:
        raise ValueError("decoded member CRC mismatch")
    return decoded


def download_candidate(candidate, info, output_root):
    destination = output_root / "raw" / candidate["participant"] / candidate["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        payload = destination.read_bytes()
        if len(payload) != info["uncompressed_bytes"] or zlib.crc32(payload) & 0xFFFFFFFF != info["crc32"]:
            raise ValueError(f"existing output checksum mismatch: {destination}")
        status = "existing"
    else:
        start, end = local_data_range(info)
        payload = decode_member(info, request_range(start, end))
        temporary = destination.with_suffix(destination.suffix + f".part.{threading.get_ident()}")
        temporary.write_bytes(payload)
        temporary.replace(destination)
        status = "downloaded"
    return {**candidate, **info, "path": destination.as_posix(), "sha256": hashlib.sha256(payload).hexdigest(), "status": status}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/local/asl_stem_wiki_bootstrap_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument("--participants", help="comma-separated participant IDs")
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be 1 through 4")

    with (args.root / "42553_supp" / "ASL_STEM_Wiki_ManualAnnotations_CVPR-Final.csv").open(newline="", encoding="utf-8-sig") as handle:
        manual_rows = list(csv.DictReader(handle))
    with (args.root / "videos.csv").open(newline="", encoding="utf-8-sig") as handle:
        metadata = {row["video filename"]: row for row in csv.DictReader(handle)}
    classes = json.loads(args.manifest.read_text())["classes"]
    raw_to_canonical = {row["citizen_raw_gloss"]: row["canonical_label"] for row in classes}
    if len(raw_to_canonical) != len(classes):
        raise ValueError("Citizen raw glosses are not unique")
    participants = (
        frozenset(value.strip() for value in args.participants.split(",") if value.strip())
        if args.participants else PROVISIONAL_DOWNLOAD_PARTICIPANTS
    )
    candidates, rejected = build_candidates(
        manual_rows, metadata, raw_to_canonical, participants
    )
    central = parse_central_directory(args.root / "source_zip_central_directory.bin")
    for candidate in candidates:
        if candidate["source_path"] not in central:
            raise ValueError(f"candidate absent from source ZIP: {candidate['filename']}")

    downloaded = []
    if args.download:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(download_candidate, candidate, central[candidate["source_path"]], args.root): candidate
                for candidate in candidates
            }
            for future in as_completed(futures):
                downloaded.append(future.result())
        downloaded.sort(key=lambda row: (row["participant"], row["filename"]))

    target_counts = Counter(token for candidate in candidates for token in candidate["ctc_sequence"] if token != "OTHER")
    report = {
        "format": "slt_v17_asl_stem_wiki_manual_candidates",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL,
        "source_archive_bytes": SOURCE_SIZE,
        "manual_annotation_csv": "42553_supp/ASL_STEM_Wiki_ManualAnnotations_CVPR-Final.csv",
        "provisional_download_participants": sorted(participants),
        "signer_quality_verified": False,
        "variant_verified": False,
        "training_eligible": False,
        "selection_contract": "Downloaded review pool only. Signer quality, exact ASL-LEX variant, and sign boundaries require explicit verification before any training use.",
        "candidate_videos": len(candidates),
        "candidate_participants": dict(sorted(Counter(row["participant"] for row in candidates).items())),
        "locked_raw_gloss_classes": len(target_counts),
        "locked_raw_gloss_tokens": sum(target_counts.values()),
        "locked_raw_gloss_counts": dict(sorted(target_counts.items())),
        "rejected": rejected,
        "downloaded_videos": downloaded,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    destination = args.output_manifest or args.root / "manual_candidate_manifest.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"manifest": destination.as_posix(), "candidates": len(candidates), "downloaded": len(downloaded), "locked_classes": len(target_counts), "locked_tokens": sum(target_counts.values())}, indent=2))


if __name__ == "__main__":
    main()
