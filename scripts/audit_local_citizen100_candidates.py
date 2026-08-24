#!/usr/bin/env python3
"""Build a conservative, quality-first audit shortlist from the local raw corpus.

This does not approve local clips for training.  The old corpus has no trustworthy
signer IDs or lexical-variant identifiers, so the shortlist is deliberately small,
train-only, and selected for both image quality and appearance/session diversity.
Known scraped sources are excluded by filename instead of being relabeled as local.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import cv2
import numpy as np


LOCAL_HEX = re.compile(r"^[0-9a-f]{8}$", re.IGNORECASE)


@dataclass
class Candidate:
    canonical_label: str
    citizen_raw_gloss: str
    citizen_asl_lex_code: str
    lexical_tier: str
    source_kind: str
    raw_path: str
    crop_path: str
    width: int
    height: int
    frames: int
    fps: float
    duration_seconds: float
    brightness: float
    clipped_fraction: float
    sharpness: float
    quality_score: float
    appearance_min_distance: float | None = None
    selection_rank: int | None = None
    training_eligible: bool = False


def classify_source(path: Path, label: str) -> str:
    stem = path.stem
    if LOCAL_HEX.fullmatch(stem):
        return "local_hex_unknown_session"
    if re.fullmatch(re.escape(label) + r"_\d+", stem, re.IGNORECASE):
        return "local_numbered_single_session"
    lowered = stem.lower()
    for prefix in ("msasl_", "signasl_", "wlasl_"):
        if lowered.startswith(prefix):
            return prefix[:-1]
    return "unknown"


def is_exact_pinned_raw(manifest_item: dict[str, object]) -> bool:
    return str(manifest_item["canonical_label"]) == str(
        manifest_item["citizen_raw_gloss"]
    )


def appearance_descriptor(image: np.ndarray) -> np.ndarray:
    """Describe coarse room/clothing/face appearance, not signer identity."""
    small = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).ravel()
    hist = hist.astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)
    vector = np.concatenate([lab.ravel(), 4.0 * hist])
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(1.0 - float(np.dot(left, right)), 0.0, 2.0))


def select_diverse(
    candidates: list[Candidate],
    descriptors: dict[str, np.ndarray],
    cap: int,
    minimum_distance: float,
) -> list[Candidate]:
    if not candidates or cap <= 0:
        return []
    remaining = sorted(candidates, key=lambda item: (-item.quality_score, item.raw_path))
    selected = [remaining.pop(0)]
    selected[0].appearance_min_distance = None
    while remaining and len(selected) < cap:
        scored: list[tuple[float, float, Candidate]] = []
        for item in remaining:
            descriptor = descriptors[item.raw_path]
            distance = min(
                cosine_distance(descriptor, descriptors[chosen.raw_path])
                for chosen in selected
            )
            diversity = min(distance / 0.45, 1.0)
            score = 0.35 * item.quality_score + 0.65 * diversity
            scored.append((score, distance, item))
        _, distance, chosen = max(scored, key=lambda row: (row[0], row[1], row[2].raw_path))
        if distance < minimum_distance:
            break
        # Numbered clips are 25 repetitions from one visible recording session.
        if chosen.source_kind == "local_numbered_single_session" and any(
            item.source_kind == chosen.source_kind for item in selected
        ):
            remaining.remove(chosen)
            continue
        chosen.appearance_min_distance = distance
        selected.append(chosen)
        remaining.remove(chosen)
    for rank, item in enumerate(selected, start=1):
        item.selection_rank = rank
    return selected


def video_metadata(path: Path) -> tuple[int, int, int, float, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return 0, 0, 0, 0.0, 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    duration = frames / fps if frames > 0 and fps > 0 else 0.0
    return width, height, frames, fps, duration


def middle_frame_crop(path: Path, frame_count: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_count // 2, 0))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        return None
    height, width = frame.shape[:2]
    side = min(width, height)
    x0 = max((width - side) // 2, 0)
    y0 = max((height - side) // 2, 0)
    return cv2.resize(
        frame[y0 : y0 + side, x0 : x0 + side],
        (128, 128),
        interpolation=cv2.INTER_AREA,
    )


def inspect_candidate(
    path: Path,
    crop_path: Path,
    manifest_item: dict[str, object],
    source_kind: str,
    minimum_quality_score: float,
) -> tuple[Candidate | None, np.ndarray | None, str | None]:
    width, height, frames, fps, duration = video_metadata(path)
    if width < 320 or height < 180 or frames <= 0 or fps < 15:
        return None, None, "invalid_or_low_resolution_video"
    if duration < 0.50 or duration > 6.0:
        return None, None, "duration_outside_isolated_sign_range"
    image = (
        cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if crop_path.is_file()
        else middle_frame_crop(path, frames)
    )
    if image is None:
        return None, None, "missing_or_invalid_midframe"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    clipped = float(np.mean((gray <= 5) | (gray >= 250)))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    if brightness < 35 or brightness > 235 or clipped > 0.50 or sharpness < 5:
        return None, None, "poor_midframe_image_quality"
    sharpness_score = min(math.log1p(sharpness) / math.log1p(500.0), 1.0)
    exposure_score = math.exp(-((brightness - 128.0) / 95.0) ** 2) * (1.0 - clipped)
    resolution_score = min(width / 640.0, height / 480.0, 1.0)
    duration_score = math.exp(-((duration - 1.6) / 1.8) ** 2)
    quality = (
        0.35 * sharpness_score
        + 0.25 * exposure_score
        + 0.20 * resolution_score
        + 0.20 * duration_score
    )
    if quality < minimum_quality_score:
        return None, None, "below_conservative_quality_floor"
    canonical = str(manifest_item["canonical_label"])
    raw_gloss = str(manifest_item["citizen_raw_gloss"])
    lexical_tier = (
        "canonical_and_pinned_raw_text_equal"
        if canonical == raw_gloss
        else "canonical_only_variant_review_required"
    )
    return (
        Candidate(
            canonical_label=canonical,
            citizen_raw_gloss=raw_gloss,
            citizen_asl_lex_code=str(manifest_item["citizen_asl_lex_code"]),
            lexical_tier=lexical_tier,
            source_kind=source_kind,
            raw_path=str(path),
            crop_path=str(crop_path) if crop_path.is_file() else "",
            width=width,
            height=height,
            frames=frames,
            fps=fps,
            duration_seconds=duration,
            brightness=brightness,
            clipped_fraction=clipped,
            sharpness=sharpness,
            quality_score=quality,
        ),
        appearance_descriptor(image),
        None,
    )


def safe_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError(f"conflicting existing symlink: {destination}")
        return
    if destination.exists():
        raise ValueError(f"refusing to overwrite existing path: {destination}")
    destination.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/raw_videos/ASL VIDEOS")
    )
    parser.add_argument(
        "--crop-root", type=Path, default=Path("data/local/ASL_hand_crops_av")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json")
    )
    parser.add_argument(
        "--audit-root", type=Path, default=Path("data/local/local_citizen100_quality_audit_q82")
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("artifacts/reports/local_citizen100_quality_audit")
    )
    parser.add_argument("--cap-per-class", type=int, default=4)
    parser.add_argument("--minimum-appearance-distance", type=float, default=0.08)
    parser.add_argument("--minimum-quality-score", type=float, default=0.82)
    parser.add_argument(
        "--exact-pinned-raw-only",
        action="store_true",
        help="Quarantine classes whose canonical folder name differs from the pinned raw gloss",
    )
    parser.add_argument("--materialize-symlinks", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    manifest_items = {
        str(item["canonical_label"]): item for item in manifest["classes"]
    }
    exclusions: dict[str, int] = {}
    inventory: dict[str, dict[str, int]] = {}
    selected: list[Candidate] = []
    descriptors: dict[str, np.ndarray] = {}
    inspected = 0

    for label, item in sorted(manifest_items.items()):
        class_root = args.raw_root / label
        counts: dict[str, int] = {}
        inventory[label] = counts
        if not class_root.is_dir():
            counts["missing_exact_class_folder"] = 1
            continue
        if args.exact_pinned_raw_only and not is_exact_pinned_raw(item):
            quarantined = len(list(class_root.glob("*.mp4")))
            counts["non_exact_pinned_raw_class_quarantined"] = quarantined
            exclusions["non_exact_pinned_raw_class_quarantined"] = (
                exclusions.get("non_exact_pinned_raw_class_quarantined", 0)
                + quarantined
            )
            continue
        if label == "I":
            # Visual audit found both fingerspelled-I and self-reference productions.
            counts["mixed_variant_folder_quarantined"] = len(list(class_root.glob("*.mp4")))
            continue
        class_candidates: list[Candidate] = []
        for path in sorted(class_root.glob("*.mp4")):
            source_kind = classify_source(path, label)
            counts[source_kind] = counts.get(source_kind, 0) + 1
            if source_kind not in {
                "local_hex_unknown_session",
                "local_numbered_single_session",
            }:
                exclusions[source_kind] = exclusions.get(source_kind, 0) + 1
                continue
            crop_path = args.crop_root / f"{label}_{path.stem}.jpg"
            candidate, descriptor, reason = inspect_candidate(
                path, crop_path, item, source_kind, args.minimum_quality_score
            )
            inspected += 1
            if candidate is None or descriptor is None:
                reason = reason or "unknown_quality_failure"
                exclusions[reason] = exclusions.get(reason, 0) + 1
                continue
            descriptors[candidate.raw_path] = descriptor
            class_candidates.append(candidate)
        selected.extend(
            select_diverse(
                class_candidates,
                descriptors,
                args.cap_per_class,
                args.minimum_appearance_distance,
            )
        )

    if args.materialize_symlinks:
        for item in selected:
            source = Path(item.raw_path)
            safe_link(source, args.audit_root / "raw" / item.canonical_label / source.name)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "quality/diversity audit shortlist; not a training manifest",
        "manifest_sha256": manifest_sha256,
        "training_eligible": False,
        "split_eligibility": "train_only_after_exact_variant_review",
        "signer_warning": (
            "No trustworthy local signer IDs exist. Appearance distance is only a "
            "recording/session diversity heuristic and must not be reported as signer count."
        ),
        "mixed_variant_quarantine": {"I": "folder visibly mixes fingerspelled I and ME/self-reference"},
        "cap_per_class": args.cap_per_class,
        "minimum_appearance_distance": args.minimum_appearance_distance,
        "minimum_quality_score": args.minimum_quality_score,
        "exact_pinned_raw_only": args.exact_pinned_raw_only,
        "inspected_local_candidates": inspected,
        "selected_clips": len(selected),
        "selected_classes": len({item.canonical_label for item in selected}),
        "exclusions": exclusions,
        "inventory": inventory,
        "videos": [asdict(item) for item in selected],
    }
    args.audit_root.mkdir(parents=True, exist_ok=True)
    provenance_path = args.audit_root / "candidate_selection.json"
    provenance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    direct = sum(item.lexical_tier == "canonical_and_pinned_raw_text_equal" for item in selected)
    variant = len(selected) - direct
    args.report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Local Citizen100 quality audit",
        "",
        "**Status:** shortlist only; no local clip is automatically training-eligible.",
        "",
        f"- Inspected local-style clips: {inspected}",
        f"- Selected clips/classes: {len(selected)}/{len({item.canonical_label for item in selected})}",
        f"- Per-class cap: {args.cap_per_class}",
        f"- Exact pinned-raw classes only: {args.exact_pinned_raw_only}",
        f"- Pinned-raw text-equal / variant-review clips: {direct}/{variant}",
        "- Trustworthy signer IDs: none",
        "- The mixed local `I` folder is fully quarantined.",
        "",
        "Known `msasl_`, `signasl_`, and `wlasl_` files are excluded. Selection uses",
        "decode/resolution/duration/exposure/sharpness checks plus coarse appearance",
        "distance to avoid taking near-duplicate recording sessions. Appearance diversity",
        "is not signer identity. Frozen-model and v17 extraction triage are still required.",
        "",
        "| Class | Selected | Raw gloss | Tier |",
        "| --- | ---: | --- | --- |",
    ]
    for label in sorted({item.canonical_label for item in selected}):
        items = [item for item in selected if item.canonical_label == label]
        lines.append(
            f"| {label} | {len(items)} | {items[0].citizen_raw_gloss} | {items[0].lexical_tier} |"
        )
    (args.report_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "inspected_local_candidates", "selected_clips", "selected_classes", "exclusions"
    )}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
