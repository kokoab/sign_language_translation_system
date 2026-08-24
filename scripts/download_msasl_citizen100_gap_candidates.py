#!/usr/bin/env python3
"""Download a bounded official-train MS-ASL audit for Citizen100 coverage gaps.

Only exact canonical/pinned-raw text matches absent from the strict local review
shortlist are considered. Candidates are unique by annotated signer within class,
high-resolution rows are attempted first, and all outputs remain training-ineligible
until exact ASL-LEX review and model/extraction triage.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import parse_qs, urlparse

import cv2


def video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc == "www.youtube.com":
        return parse_qs(parsed.query).get("v", [""])[0]
    return ""


def build_candidates(
    annotations: list[dict[str, object]],
    manifest: dict[str, object],
    covered_classes: set[str],
    attempts_per_class: int,
) -> list[dict[str, object]]:
    items = {
        str(item["canonical_label"]): item for item in manifest["classes"]
    }
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in annotations:
        label = str(row.get("text", "")).upper()
        item = items.get(label)
        if (
            item is None
            or label in covered_classes
            or str(item["citizen_raw_gloss"]) != label
            or not video_id(str(row.get("url", "")))
        ):
            continue
        width = float(row.get("width") or 0)
        height = float(row.get("height") or 0)
        duration = float(row.get("end_time") or 0) - float(row.get("start_time") or 0)
        # Later offsets can force long source reads when servers ignore range seeks.
        # The <=120s bound preserves nearly all target classes while keeping every
        # attempted segment operationally small.
        if (
            width < 640
            or height < 360
            or duration < 0.4
            or duration > 8.0
            or float(row.get("start_time") or 0) > 120.0
        ):
            continue
        grouped.setdefault(label, []).append(row)

    candidates: list[dict[str, object]] = []
    for label in sorted(grouped):
        seen_signers: set[str] = set()
        selected = sorted(
            grouped[label],
            key=lambda row: (
                float(row.get("width") or 0) * float(row.get("height") or 0),
                -abs((float(row["end_time"]) - float(row["start_time"])) - 2.5),
            ),
            reverse=True,
        )
        for row in selected:
            signer = str(row["signer_id"])
            if signer in seen_signers:
                continue
            seen_signers.add(signer)
            item = items[label]
            candidates.append(
                {
                    "canonical_label": label,
                    "citizen_raw_gloss": item["citizen_raw_gloss"],
                    "citizen_asl_lex_code": item["citizen_asl_lex_code"],
                    "msasl_signer_id": signer,
                    "msasl_label": row["label"],
                    "url": row["url"],
                    "youtube_id": video_id(str(row["url"])),
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "annotation_width": row["width"],
                    "annotation_height": row["height"],
                    "annotation_fps": row["fps"],
                    "box": row["box"],
                    "training_eligible": False,
                }
            )
            if len(seen_signers) >= attempts_per_class:
                break
    return candidates


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_video(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("OpenCV could not open output")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    ok, _ = capture.read()
    capture.release()
    if not ok or width < 320 or height < 180 or frames <= 0 or fps <= 0:
        raise ValueError("invalid decoded video")
    return {
        "decoded_width": width,
        "decoded_height": height,
        "decoded_frames": frames,
        "decoded_fps": fps,
        "decoded_duration": frames / fps,
    }


def acquire_one(
    row: dict[str, object],
    output_root: Path,
    downloader_python: Path,
) -> dict[str, object]:
    label = str(row["canonical_label"])
    stem = (
        f"msasl_train_{row['msasl_label']}_{row['msasl_signer_id']}_"
        f"{row['youtube_id']}_{float(row['start_time']):.3f}"
    ).replace(".", "p")
    destination = output_root / label / f"{stem}.mp4"
    result = dict(row)
    result["destination"] = str(destination)
    if destination.exists():
        try:
            result.update(validate_video(destination))
            result.update(
                status="existing_verified",
                bytes=destination.stat().st_size,
                sha256=sha256_file(destination),
            )
            return result
        except Exception:
            result.update(status="failed", error="existing output failed decode")
            return result

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="msasl-segment-") as temporary:
        template = str(Path(temporary) / "segment.%(ext)s")
        command = [
            str(downloader_python),
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--no-warnings",
            "--js-runtimes",
            "node",
            "--download-sections",
            f"*{float(row['start_time']):.3f}-{float(row['end_time']):.3f}",
            "--force-keyframes-at-cuts",
            "--format",
            "bv*[vcodec^=avc][height>=360]/b[ext=mp4][height>=360]/b[height>=360]",
            "--remux-video",
            "mp4",
            "--socket-timeout",
            "15",
            "--retries",
            "2",
            "--fragment-retries",
            "2",
            "--concurrent-fragments",
            "4",
            "--output",
            template,
            str(row["url"]),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            result.update(status="failed", error="yt-dlp segment timeout after 60 seconds")
            return result
        outputs = sorted(Path(temporary).glob("segment.*"))
        if completed.returncode != 0 or not outputs:
            error = (completed.stderr or completed.stdout).strip().splitlines()
            result.update(status="failed", error=error[-1] if error else "yt-dlp failed")
            return result
        source = outputs[0]
        try:
            decoded = validate_video(source)
        except Exception as exc:
            result.update(status="failed", error=str(exc))
            return result
        source.replace(destination)
        result.update(decoded)
        result.update(
            status="downloaded",
            bytes=destination.stat().st_size,
            sha256=sha256_file(destination),
        )
        return result


def acquire_class(
    rows: list[dict[str, object]],
    output_root: Path,
    downloader_python: Path,
    target: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    successes = 0
    for row in rows:
        result = acquire_one(row, output_root, downloader_python)
        results.append(result)
        if result["status"] in {"downloaded", "existing_verified"}:
            successes += 1
            if successes >= target:
                break
    return results


def materialize_retained(rows: list[dict[str, object]], retained_root: Path) -> None:
    for row in rows:
        source = Path(str(row["destination"]))
        destination = retained_root / str(row["canonical_label"]) / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise ValueError(f"conflicting retained symlink: {destination}")
        elif destination.exists():
            raise ValueError(f"refusing to overwrite retained path: {destination}")
        else:
            destination.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/local/dataset_metadata/msasl_official/MS-ASL/MSASL_train.json"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json")
    )
    parser.add_argument(
        "--local-review",
        type=Path,
        default=Path("data/local/local_citizen100_quality_audit_q82/review_shortlist.json"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/local/msasl_citizen100_gap_audit/raw")
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/local/msasl_citizen100_gap_audit/candidate_provenance.json"),
    )
    parser.add_argument(
        "--retained-root",
        type=Path,
        default=Path("data/local/msasl_citizen100_gap_audit/retained_raw"),
    )
    parser.add_argument(
        "--downloader-python",
        type=Path,
        default=Path("artifacts/generated/msasl_download_env/bin/python"),
    )
    parser.add_argument("--target-per-class", type=int, default=3)
    parser.add_argument("--attempts-per-class", type=int, default=8)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    local_review = json.loads(args.local_review.read_text(encoding="utf-8"))
    covered = {str(row["canonical_label"]) for row in local_review["videos"]}
    candidates = build_candidates(
        annotations, manifest, covered, args.attempts_per_class
    )
    if args.dry_run:
        print(json.dumps({
            "candidate_attempts": len(candidates),
            "classes": len({row['canonical_label'] for row in candidates}),
        }, indent=2))
        return
    if not args.downloader_python.is_file():
        raise FileNotFoundError(args.downloader_python)

    results: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in candidates:
        grouped.setdefault(str(row["canonical_label"]), []).append(row)
    # Classes run in parallel; attempts inside each class are sequential and stop as
    # soon as the exact bounded target is reached. This prevents surplus downloads.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                acquire_class,
                rows,
                args.output_root,
                args.downloader_python,
                args.target_per_class,
            ): label
            for label, rows in grouped.items()
        }
        for future in as_completed(future_map):
            class_results = future.result()
            results.extend(class_results)
            successes = sum(
                row["status"] in {"downloaded", "existing_verified"}
                for row in class_results
            )
            print(
                f"{future_map[future]}: retained {successes}/{args.target_per_class} "
                f"after {len(class_results)} attempts",
                flush=True,
            )

    retained = [
        row for row in results if row["status"] in {"downloaded", "existing_verified"}
    ]
    materialize_retained(retained, args.retained_root)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "MS-ASL official train split",
        "source_annotations": str(args.annotations),
        "annotations_sha256": sha256_file(args.annotations),
        "license": "C-UDA-0.1",
        "training_eligible": False,
        "eligibility_warning": (
            "Exact English text does not prove the pinned ASL-LEX variant. "
            "ASL-fluent review and v17/model triage are required."
        ),
        "split": "train_only",
        "target_per_class": args.target_per_class,
        "retained_clips": len(retained),
        "retained_classes": len({row["canonical_label"] for row in retained}),
        "attempts": len(results),
        "videos": sorted(results, key=lambda row: (str(row["canonical_label"]), str(row["msasl_signer_id"]))),
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("retained_clips", "retained_classes", "attempts")}, indent=2))


if __name__ == "__main__":
    main()
