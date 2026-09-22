#!/usr/bin/env python3
"""Pick one reference clip per locked-100 gloss and render it for the app gallery.

Project-owned local recordings are preferred.  The eligible pool is the exact-variant
audit shortlist, which is already free of scraped third-party clips and restricted to
classes whose canonical label and pinned raw gloss agree — the safeguard against showing
the wrong articulation for a word.  It reaches 77 of the 100 classes.

The remaining classes fall back to ASL Citizen.  Only its *train* split is eligible:
validation is reserved and the official test gate has been consumed once already, so
neither may be displayed or scanned here.  Citizen candidates are scored from their
extracted v17 landmark diagnostics, never the pixels, so ranking 1,476 of them costs a
few seconds and no decoding.

Rendering decodes only the 100 winners.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MANIFEST = REPO / "active/v17/citizen100_manifest.json"
CITIZEN = REPO / "data/local/citizen100_v17"
RAW_TRAIN = CITIZEN / "raw/train"
LANDMARKS_TRAIN = CITIZEN / "landmarks/train"
REJECTIONS = CITIZEN / "rejections.csv"
PROVENANCE = CITIZEN / "provenance.csv"
LOCAL_AUDIT = (
    REPO / "data/local/local_citizen100_quality_audit_q82_cap14_exact"
    / "candidate_selection.json"
)
DEFAULT_OUTPUT = REPO / "artifacts/app_assets/gloss_examples"

# A clip is shown to a person, so the weights favour a legible articulation over a
# merely well-tracked one.  Every component is stored in the catalogue, so a
# different balance can be re-derived without re-reading the corpus.
#
# The total ranks candidates *within* one gloss and means nothing across glosses:
# hand_presence counts both hands, so a one-handed sign floors near 0.5 however
# cleanly it is produced.
WEIGHTS = {
    "hand_presence": 0.40,
    "hand_frames": 0.25,
    "face_presence": 0.15,
    "edges": 0.10,
    "duration": 0.10,
}
IDEAL_SECONDS = (1.2, 2.6)


@dataclass
class Candidate:
    label: str
    video: Path
    landmarks: Path | None = None
    participant: str = ""
    sha256: str = ""
    source: str = "citizen"
    components: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    seconds: float = 0.0
    activity: tuple[int, int] = (0, 0)
    notes: list[str] = field(default_factory=list)


def provenance_rows() -> dict[str, dict[str, str]]:
    """Participant and hash per clip; the landmark sidecars do not carry them."""
    if not PROVENANCE.exists():
        return {}
    with PROVENANCE.open(newline="") as handle:
        return {row["video"]: row for row in csv.DictReader(handle)}


def local_candidates() -> dict[str, list[Candidate]]:
    """The exact-variant local shortlist, grouped by gloss and best-first.

    The audit already carries quality, duration and resolution per clip, so nothing
    here re-measures anything; it only orders what the audit approved.
    """
    try:
        rows = json.loads(LOCAL_AUDIT.read_text())["videos"]
    except (OSError, ValueError, KeyError):
        return {}
    grouped: dict[str, list[Candidate]] = {}
    for row in rows:
        video = REPO / row["raw_path"]
        if not video.is_file():
            continue
        seconds = float(row.get("duration_seconds") or 0.0)
        quality = float(row.get("quality_score") or 0.0)
        candidate = Candidate(
            label=row["canonical_label"], video=video, source="local",
            seconds=seconds,
            components={"quality": round(quality, 4), "duration": duration_fit(seconds)},
        )
        # The audit's own rank leads; duration only separates equals.
        candidate.score = quality + 0.05 * candidate.components["duration"]
        grouped.setdefault(candidate.label, []).append(candidate)
    for items in grouped.values():
        items.sort(key=lambda item: item.score, reverse=True)
    return grouped


def rejected_videos() -> set[str]:
    """Clips an earlier audit rejected by name; some still sit in the train pool."""
    if not REJECTIONS.exists():
        return set()
    with REJECTIONS.open(newline="") as handle:
        return {row["video"] for row in csv.DictReader(handle)}


def duration_fit(seconds: float) -> float:
    low, high = IDEAL_SECONDS
    if seconds <= 0:
        return 0.0
    if low <= seconds <= high:
        return 1.0
    edge = low - seconds if seconds < low else seconds - high
    return max(0.0, 1.0 - edge / 1.5)


def read_candidate(
    path: Path, label: str, provenance: dict[str, dict[str, str]]
) -> Candidate | None:
    """Score one clip from its landmark sidecar.  No pixels are touched."""
    try:
        with np.load(path, allow_pickle=False) as bundle:
            metadata = json.loads(str(bundle["metadata_json"].item()))
            diagnostics = json.loads(str(bundle["diagnostics_json"].item()))
    except (OSError, ValueError, KeyError):
        return None
    if not diagnostics.get("finite", False):
        return None

    video = REPO / metadata.get("video_path", "")
    if not video.is_file():
        return None

    fps = float(metadata.get("fps") or 30.0)
    start = int(metadata.get("hand_trim_start_frame", 0))
    end = int(metadata.get("hand_trim_end_frame_exclusive", 0))
    total = int(metadata.get("source_frames_before_hand_trim", 0)) or end
    seconds = max(0.0, (end - start) / fps) if fps > 0 else 0.0

    # Activity touching either edge means the articulation may be truncated.
    clipped = int(start <= 0) + int(end >= total)
    notes = []
    if start <= 0:
        notes.append("activity starts on the first frame")
    if end >= total:
        notes.append("activity runs to the last frame")

    components = {
        "hand_presence": float(diagnostics.get("hand_presence_fraction", 0.0)),
        "hand_frames": float(diagnostics.get("observed_hand_frame_fraction", 0.0)),
        "face_presence": float(diagnostics.get("face_presence_fraction", 0.0)),
        "edges": {0: 1.0, 1: 0.5}.get(clipped, 0.0),
        "duration": duration_fit(seconds),
    }
    source = provenance.get(video.name, {})
    candidate = Candidate(
        label=label, video=video, landmarks=path,
        participant=source.get("participant", ""), sha256=source.get("sha256", ""),
        components=components, seconds=seconds, activity=(start, end), notes=notes,
    )
    candidate.score = sum(WEIGHTS[key] * value for key, value in components.items())
    return candidate


def collect(
    label: str, skip: set[str], provenance: dict[str, dict[str, str]]
) -> list[Candidate]:
    folder = LANDMARKS_TRAIN / label
    if not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.glob("*.v17.npz")):
        candidate = read_candidate(path, label, provenance)
        if candidate is None or candidate.video.name in skip:
            continue
        # Belt and braces: the winner must come from the train split.
        if RAW_TRAIN not in candidate.video.parents:
            continue
        found.append(candidate)
    return found


def render(candidate: Candidate, output: Path, width: int, pad: float) -> dict:
    """Write the poster frame and a trimmed, downscaled loop for one gloss."""
    capture = cv2.VideoCapture(str(candidate.video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {candidate.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 1 else 30.0
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if candidate.source == "local":
        # No landmark window for these; the audit already bounded the duration.
        start, end = 0, count or 1
    else:
        margin = int(round(pad * fps))
        start, end = candidate.activity
        start = max(0, start - margin)
        end = min(count if count > 0 else end + margin, end + margin)
    if end <= start:
        start, end = 0, count or 1

    frames = []
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(end - start):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {candidate.video}")

    height = int(round(frames[0].shape[0] * width / frames[0].shape[1]))
    height += height % 2
    resized = [
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]

    output.mkdir(parents=True, exist_ok=True)
    loop = output / f"{candidate.label}.mp4"
    writer = cv2.VideoWriter(
        str(loop), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not write {loop}")
    for frame in resized:
        writer.write(frame)
    writer.release()

    poster = output / f"{candidate.label}.jpg"
    cv2.imwrite(str(poster), resized[len(resized) // 2], [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {
        "poster": poster.name, "loop": loop.name,
        "frames": len(resized), "fps": round(fps, 3),
        "width": width, "height": height,
    }


def build(args: argparse.Namespace) -> dict:
    classes = json.loads(MANIFEST.read_text())["classes"]
    skip = rejected_videos()
    provenance = provenance_rows()
    local = {} if args.citizen_only else local_candidates()
    catalog, missing, flagged = {}, [], []
    counts = {"local": 0, "citizen": 0}

    for entry in classes:
        label = entry["canonical_label"]
        # Local first; Citizen covers only what the exact-variant pool cannot.
        candidates = local.get(label) or collect(label, skip, provenance)
        if not candidates:
            missing.append(label)
            continue
        best = max(candidates, key=lambda item: item.score)
        if best.notes:
            flagged.append(label)
        counts[best.source] += 1
        row = {
            "canonical_label": label,
            "class_index": entry["class_index"],
            "category": entry["category"],
            "citizen_raw_gloss": entry["citizen_raw_gloss"],
            "citizen_asl_lex_code": entry["citizen_asl_lex_code"],
            "source": best.source,
            "participant": best.participant,
            "source_video": str(best.video.relative_to(REPO)),
            "source_sha256": best.sha256,
            "split": "train" if best.source == "citizen" else None,
            "seconds": round(best.seconds, 3),
            "score": round(best.score, 4),
            "components": {k: round(v, 4) for k, v in best.components.items()},
            "candidates_considered": len(candidates),
            "notes": best.notes,
        }
        if not args.catalog_only:
            row["assets"] = render(best, args.output, args.width, args.pad)
        catalog[label] = row

    summary = {
        "format": "slt_v17_app_gloss_examples",
        "version": 1,
        "manifest": str(MANIFEST.relative_to(REPO)),
        "preference": "project-owned local exact-variant pool, then ASL Citizen train",
        "source_split": "train",
        "val_accessed": False,
        "test_accessed": False,
        "selection": "landmark diagnostics only; no video decoded for scoring",
        "score_semantics": (
            "ranks candidates within one gloss only; not comparable across glosses "
            "because hand_presence counts both hands"
        ),
        "weights": WEIGHTS,
        "ideal_seconds": list(IDEAL_SECONDS),
        "rejected_videos_skipped": sorted(skip),
        "classes_covered": len(catalog),
        "classes_from_local": counts["local"],
        "classes_from_citizen": counts["citizen"],
        "local_pool": str(LOCAL_AUDIT.relative_to(REPO)),
        "classes_missing": missing,
        "classes_with_edge_notes": flagged,
        "examples": catalog,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "catalog.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--width", type=int, default=480)
    value.add_argument(
        "--pad", type=float, default=0.25,
        help="seconds of context kept either side of the hand-activity window",
    )
    value.add_argument(
        "--citizen-only", action="store_true",
        help="ignore the local pool and rank ASL Citizen train for every class",
    )
    value.add_argument(
        "--catalog-only", action="store_true",
        help="score and choose without decoding or writing any media",
    )
    return value


def main() -> None:
    args = parser().parse_args()
    summary = build(args)
    print(
        f"covered {summary['classes_covered']}/100 glosses "
        f"({summary['classes_from_local']} local, "
        f"{summary['classes_from_citizen']} citizen)"
    )
    if summary["classes_missing"]:
        print(f"missing: {', '.join(summary['classes_missing'])}")
    if summary["classes_with_edge_notes"]:
        print(f"edge-flagged: {len(summary['classes_with_edge_notes'])}")


if __name__ == "__main__":
    main()
