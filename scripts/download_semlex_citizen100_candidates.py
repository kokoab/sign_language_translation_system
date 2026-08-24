#!/usr/bin/env python3
"""Resume-safe ranged download and selective extraction of SemLex train clips.

Google Drive publishes SemLex train video as one gzip-compressed tar archive, so
individual members cannot be fetched independently.  This downloader range-fetches
the transport archive in parallel, extracts only the exact preplanned members, then
can remove the transport archive so only the bounded Citizen100 subset is retained.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tarfile
import time
import urllib.request

import cv2


TRAIN_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1jiUasWSGv5lkrBUIRmtCXyMzliClCqXo&export=download&confirm=t"
)
TRAIN_ARCHIVE_SIZE = 23_673_462_199
CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def parse_content_range(value: str) -> tuple[int, int, int]:
    match = CONTENT_RANGE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"invalid Content-Range: {value!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def normalized_member(value: str) -> str:
    while value.startswith("./"):
        value = value[2:]
    return value


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fetch_range(
    url: str,
    fd: int,
    start: int,
    end: int,
    total: int,
    attempts: int,
) -> str:
    expected = end - start + 1
    for attempt in range(1, attempts + 1):
        digest = hashlib.sha256()
        written = 0
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "Mozilla/5.0 SLT-SemLex-bounded-downloader/1.0",
                },
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                if response.status != 206:
                    raise IOError(f"expected HTTP 206, received {response.status}")
                actual = parse_content_range(response.headers.get("Content-Range", ""))
                if actual != (start, end, total):
                    raise IOError(f"wrong response range {actual}, expected {(start, end, total)}")
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    os.pwrite(fd, block, start + written)
                    digest.update(block)
                    written += len(block)
            if written != expected:
                raise IOError(f"short range body: {written} != {expected}")
            return digest.hexdigest()
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 30))
    raise AssertionError("unreachable")


def download_archive(
    url: str,
    archive_path: Path,
    total: int,
    chunk_size: int,
    workers: int,
    attempts: int,
) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = archive_path.with_suffix(archive_path.suffix + ".ranges.json")
    chunks = math.ceil(total / chunk_size)
    state: dict[str, object]
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("total_bytes") != total or state.get("chunk_size") != chunk_size:
            raise ValueError("existing range state does not match requested archive geometry")
    else:
        if archive_path.exists() and archive_path.stat().st_size not in (0, total):
            raise ValueError("refusing to reuse archive without compatible range state")
        state = {
            "url": url,
            "total_bytes": total,
            "chunk_size": chunk_size,
            "completed": {},
        }
    completed = dict(state.get("completed", {}))
    fd = os.open(archive_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        os.ftruncate(fd, total)
        pending = [index for index in range(chunks) if str(index) not in completed]
        print(
            json.dumps(
                {
                    "archive": str(archive_path),
                    "total_bytes": total,
                    "chunks": chunks,
                    "already_complete": chunks - len(pending),
                    "pending": len(pending),
                    "workers": workers,
                }
            ),
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for index in pending:
                start = index * chunk_size
                end = min(total - 1, start + chunk_size - 1)
                future = executor.submit(fetch_range, url, fd, start, end, total, attempts)
                futures[future] = index
            for finished, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                completed[str(index)] = future.result()
                state["completed"] = completed
                state["updated_utc"] = datetime.now(timezone.utc).isoformat()
                atomic_json(state_path, state)
                if finished == 1 or finished % 8 == 0 or finished == len(pending):
                    print(
                        json.dumps(
                            {
                                "completed_chunks": len(completed),
                                "total_chunks": chunks,
                                "completed_gib": round(
                                    min(len(completed) * chunk_size, total) / 2**30, 3
                                ),
                            }
                        ),
                        flush=True,
                    )
        os.fsync(fd)
    finally:
        os.close(fd)
    if len(completed) != chunks:
        raise RuntimeError("archive range download incomplete")
    return state_path


def video_diagnostics(path: Path, expected_duration_ms: float) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"downloaded clip does not decode: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    container_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    container_fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded_frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded_frames += 1
    capture.release()
    if width <= 0 or height <= 0 or decoded_frames < 2 or expected_duration_ms <= 0:
        raise ValueError(f"downloaded clip has invalid video metadata: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "raw_path": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "decoded_frames": decoded_frames,
        "effective_fps_from_annotation": decoded_frames / (expected_duration_ms / 1000.0),
        "annotated_duration_seconds": expected_duration_ms / 1000.0,
        "container_frame_count": container_frames,
        "container_fps": container_fps,
    }


def extract_selected(
    archive_path: Path,
    selection: dict[str, object],
    raw_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected = {
        normalized_member(str(row["archive_member"])): row
        for row in selection["videos"]  # type: ignore[index]
    }
    found: set[str] = set()
    provenance: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            name = normalized_member(member.name)
            row = selected.get(name)
            if row is None:
                continue
            if not member.isfile():
                raise ValueError(f"selected archive member is not a file: {member.name}")
            video_id = str(row["semlex_video_id"])
            destination = raw_root / str(row["canonical_label"]) / f"{video_id}.webm"
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"could not read selected member: {member.name}")
            temporary = destination.with_suffix(destination.suffix + ".partial")
            with temporary.open("wb") as output:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
            temporary.replace(destination)
            try:
                diagnostics = video_diagnostics(destination, float(row["duration_ms"]))
            except Exception as error:
                rejected_path = (
                    raw_root.parent
                    / "rejected_raw"
                    / str(row["canonical_label"])
                    / destination.name
                )
                rejected_path.parent.mkdir(parents=True, exist_ok=True)
                destination.replace(rejected_path)
                rejected.append(
                    {
                        **row,
                        "raw_path": str(rejected_path),
                        "bytes": rejected_path.stat().st_size,
                        "sha256": hashlib.sha256(rejected_path.read_bytes()).hexdigest(),
                        "rejection_reason": f"decode_validation_failed: {error}",
                        "training_eligible": False,
                    }
                )
                found.add(name)
                continue
            provenance.append({**row, **diagnostics, "training_eligible": False})
            found.add(name)
            if len(found) % 25 == 0:
                print(json.dumps({"extracted": len(found), "expected": len(selected)}), flush=True)
    missing = sorted(set(selected) - found)
    for name in missing:
        row = selected[name]
        rejected.append(
            {
                **row,
                "raw_path": "",
                "bytes": 0,
                "sha256": "",
                "rejection_reason": "archive_member_missing",
                "training_eligible": False,
            }
        )
    return provenance, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("data/local/semlex_citizen100_train_audit/selection_plan.json"),
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=Path("data/local/semlex_citizen100_train_audit/transport/train.tar.gz"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/local/semlex_citizen100_train_audit/raw"),
    )
    parser.add_argument("--url", default=TRAIN_URL)
    parser.add_argument("--archive-size", type=int, default=TRAIN_ARCHIVE_SIZE)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use an already downloaded archive and make no network request.",
    )
    parser.add_argument("--remove-transport-after-success", action="store_true")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("training_eligible") is not False:
        raise ValueError("selection plan must explicitly be non-training-eligible")
    split = str(selection.get("split", "train"))
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported SemLex split: {split}")
    if any(row.get("semlex_split") != split for row in selection["videos"]):
        raise ValueError("selection contains mixed SemLex splits")

    state_path: Path | None = None
    if args.skip_download:
        if not args.archive_path.is_file():
            raise FileNotFoundError(args.archive_path)
        actual_size = args.archive_path.stat().st_size
        if actual_size != args.archive_size:
            raise ValueError(
                f"local archive size mismatch: {actual_size} != {args.archive_size}"
            )
    else:
        state_path = download_archive(
            args.url,
            args.archive_path,
            args.archive_size,
            args.chunk_mib * 1024 * 1024,
            args.workers,
            args.attempts,
        )
    videos, rejected = extract_selected(args.archive_path, selection, args.raw_root)
    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": f"decoded exact-ASL-LEX SemLex {split} subset",
        "training_eligible": False,
        "split": split,
        "split_eligibility": (
            "train_only_after_v17_quality_and_mismatch_review"
            if split == "train"
            else "evaluation_only_never_training"
        ),
        "selected_clips": len(videos),
        "selected_classes": len({row["canonical_label"] for row in videos}),
        "selected_signers": len({row["semlex_signer_id"] for row in videos}),
        "bytes": sum(int(row["bytes"]) for row in videos),
        "videos": videos,
        "requested_clips": len(selection["videos"]),
        "rejected_clips": len(rejected),
        "rejected_videos": rejected,
    }
    atomic_json(args.selection.parent / "download_provenance.json", output)
    if args.remove_transport_after_success:
        args.archive_path.unlink()
        if state_path is not None:
            state_path.unlink()
    print(
        json.dumps(
            {
                "clips": output["selected_clips"],
                "classes": output["selected_classes"],
                "signers": output["selected_signers"],
                "bytes": output["bytes"],
                "rejected_clips": output["rejected_clips"],
                "transport_removed": args.remove_transport_after_success,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
