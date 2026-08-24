#!/usr/bin/env python3
"""Extract an official ASL-LEX/ASL SignBank exact-variant v17 supplement.

Admission is deliberately stronger than English-label agreement. Each Citizen class's
pinned ASL-LEX code is resolved through the official ASL-LEX 2.0 table to a nonempty
``SignBankAnnotationID``. That ID must exactly equal both ASLLRP's entry/variant gloss
and its occurrence gloss (apart from a trailing repetition marker). Only clips present
in the declared official ASLLRP archives are extracted.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

import cv2


ARCHIVE_URLS = {
    "batch_signs_video_v1_1.zip": (
        "https://dai.cs.rutgers.edu/asllvd/signbank/dsps/batch_signs_video_v1_1.zip"
    ),
    "batch_signs_video_v2_1.zip": (
        "https://dai.cs.rutgers.edu/asllvd/signbank/dsps/batch_signs_video_v2_1.zip"
    ),
    "batch_signs_video_v2_2.zip": (
        "https://dai.cs.rutgers.edu/asllvd/signbank/dsps/batch_signs_video_v2_2.zip"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, encoding: str) -> list[dict[str, str]]:
    with path.open(encoding=encoding, newline="") as handle:
        return list(csv.DictReader(handle))


def probe_video(path: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open extracted video: {path}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        frames += 1
    capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or frames < 4:
        raise ValueError(f"invalid extracted video metadata: {path}")
    return {"width": width, "height": height, "fps": fps, "frames": frames}


def archive_index(paths: list[Path]) -> tuple[dict[str, tuple[Path, str, int]], list[dict[str, object]]]:
    index: dict[str, tuple[Path, str, int]] = {}
    provenance: list[dict[str, object]] = []
    for path in paths:
        if path.name not in ARCHIVE_URLS:
            raise ValueError(f"undeclared ASLLRP archive: {path.name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"CRC failure in {path}: {bad}")
            for info in archive.infolist():
                member = PurePosixPath(info.filename)
                if info.is_dir():
                    continue
                if member.is_absolute() or ".." in member.parts:
                    raise ValueError(f"unsafe archive member in {path}: {info.filename}")
                basename = member.name
                if not basename.lower().endswith((".mp4", ".mov", ".m4v")):
                    continue
                if basename in index:
                    raise ValueError(f"duplicate clip basename across archives: {basename}")
                index[basename] = (path, info.filename, int(info.CRC))
        provenance.append(
            {
                "name": path.name,
                "url": ARCHIVE_URLS[path.name],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return index, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json")
    )
    parser.add_argument(
        "--asllex", type=Path,
        default=Path("data/local/dataset_metadata/asllex2_official/signdata.csv"),
    )
    parser.add_argument(
        "--asllrp", type=Path,
        default=Path("data/local/dataset_metadata/asllrp_signbank/dsp_signs_2024_06_27.csv"),
    )
    parser.add_argument(
        "--archive-dir", type=Path, default=Path("data/local/asllrp_asllex_v17/transport")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/local/asllrp_asllex_v17")
    )
    parser.add_argument("--cap-per-class", type=int, default=5)
    args = parser.parse_args()
    if args.cap_per_class < 1:
        raise ValueError("cap-per-class must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    asllex_rows = read_csv(args.asllex, "latin-1")
    asllrp_rows = read_csv(args.asllrp, "utf-8-sig")
    asllex_by_code = {row["Code"]: row for row in asllex_rows}
    archives = [args.archive_dir / name for name in ARCHIVE_URLS]
    members, archive_provenance = archive_index(archives)

    selected: list[dict[str, object]] = []
    class_summary: list[dict[str, object]] = []
    for item in sorted(manifest["classes"], key=lambda row: int(row["class_index"])):
        code = str(item["citizen_asl_lex_code"])
        asllex = asllex_by_code.get(code)
        if asllex is None:
            raise ValueError(f"ASL-LEX code is absent: {code}")
        annotation = asllex["SignBankAnnotationID"].strip()
        candidates = []
        if annotation:
            for row in asllrp_rows:
                occurrence = row["occurrence label"].strip()
                if (
                    row["entry/variant gloss label"].strip() == annotation
                    and occurrence.rstrip("+") == annotation
                    and row["Sign clip video filename"].strip() in members
                ):
                    candidates.append(row)
        # A full video identifies a source recording; take at most one occurrence from
        # each before admitting repeated tokens from the same recording.
        by_recording: dict[str, dict[str, str]] = {}
        for row in sorted(candidates, key=lambda value: int(value["Video ID number"])):
            by_recording.setdefault(row["full video filename"], row)
        chosen = list(by_recording.values())[: args.cap_per_class]
        class_summary.append(
            {
                "class_index": int(item["class_index"]),
                "canonical_label": item["canonical_label"],
                "citizen_raw_gloss": item["citizen_raw_gloss"],
                "citizen_asl_lex_code": code,
                "asllex_entry_id": asllex["EntryID"],
                "signbank_annotation_id": annotation,
                "exact_archive_candidates": len(candidates),
                "distinct_source_recordings": len(by_recording),
                "selected_clips": len(chosen),
            }
        )
        for row in chosen:
            clip = row["Sign clip video filename"].strip()
            selected.append(
                {
                    "class_index": int(item["class_index"]),
                    "canonical_label": item["canonical_label"],
                    "citizen_raw_gloss": item["citizen_raw_gloss"],
                    "citizen_asl_lex_code": code,
                    "asllex_entry_id": asllex["EntryID"],
                    "signbank_annotation_id": annotation,
                    "asllrp_video_id": row["Video ID number"],
                    "asllrp_entry_variant": row["entry/variant gloss label"],
                    "asllrp_occurrence": row["occurrence label"],
                    "asllrp_sign_type": row["sign type"],
                    "source_recording": row["full video filename"],
                    "clip_filename": clip,
                }
            )

    raw_root = args.output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    archive_handles: dict[Path, zipfile.ZipFile] = {}
    try:
        for row in selected:
            clip = str(row["clip_filename"])
            archive_path, member, crc = members[clip]
            destination = raw_root / str(row["canonical_label"]) / clip
            destination.parent.mkdir(parents=True, exist_ok=True)
            archive = archive_handles.setdefault(archive_path, zipfile.ZipFile(archive_path))
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=f".{clip}.", suffix=".tmp", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                with archive.open(member) as source:
                    shutil.copyfileobj(source, temporary)
            try:
                new_hash = sha256_file(temporary_path)
                if destination.exists():
                    if sha256_file(destination) != new_hash:
                        raise ValueError(f"refusing to replace changed clip: {destination}")
                    temporary_path.unlink()
                else:
                    os.replace(temporary_path, destination)
                media = probe_video(destination)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
            row.update(
                {
                    "raw_path": str(destination),
                    "feature_path": str(
                        args.output_root
                        / "landmarks"
                        / str(row["canonical_label"])
                        / f"{destination.stem}.v17.npz"
                    ),
                    "source_archive": archive_path.name,
                    "source_archive_url": ARCHIVE_URLS[archive_path.name],
                    "archive_member": member,
                    "zip_crc32": f"{crc:08x}",
                    "bytes": destination.stat().st_size,
                    "sha256": new_hash,
                    **media,
                    "consensus_tier": "official_asllex_signbank_exact",
                    "training_eligible": True,
                }
            )
    finally:
        for archive in archive_handles.values():
            archive.close()

    payload = {
        "format": "slt_v17_asllrp_asllex_exact_supplement",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "official-ASL-LEX-cross-referenced ASLLRP train-only supplement",
        "split_eligibility": "train_only_official_asllex_signbank_cross_reference",
        "license": "ASLLRP research-only noncommercial terms; do not redistribute clips",
        "manifest_sha256": sha256_file(args.manifest),
        "asllex_metadata_sha256": sha256_file(args.asllex),
        "asllrp_metadata_sha256": sha256_file(args.asllrp),
        "variant_contract": (
            "nonempty ASL-LEX SignBankAnnotationID equals ASLLRP entry/variant; "
            "occurrence differs only by optional trailing repetition marker"
        ),
        "cap_per_class": args.cap_per_class,
        "selected_clips": len(selected),
        "selected_classes": len({row["canonical_label"] for row in selected}),
        "source_recordings": len({row["source_recording"] for row in selected}),
        "archives": archive_provenance,
        "classes": class_summary,
        "videos": selected,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
    }
    output = args.output_root / "exact_variant_manifest.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "clips": payload["selected_clips"],
        "classes": payload["selected_classes"],
        "source_recordings": payload["source_recordings"],
    }, indent=2))


if __name__ == "__main__":
    main()
