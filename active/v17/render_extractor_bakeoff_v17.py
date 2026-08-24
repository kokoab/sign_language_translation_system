#!/usr/bin/env python3
"""Render Apple and MediaPipe hands on the bakeoff's largest disagreements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.extract_mediapipe_v17 import DEFAULT_MODEL_PATH, MediaPipeHybridDetector
    from active.v17.extract_v17 import AppleVisionDetector, assign_hands, read_video_frames
    from active.v17.schema_mediapipe_v17 import MediaPipeV17Config
else:
    from .extract_mediapipe_v17 import DEFAULT_MODEL_PATH, MediaPipeHybridDetector
    from .extract_v17 import AppleVisionDetector, assign_hands, read_video_frames
    from .schema_mediapipe_v17 import MediaPipeV17Config


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)
COLORS = {"left": (55, 220, 55), "right": (220, 65, 220)}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_disagreements(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for metric in ("hand_active", "both_hands_active"):
        ordered = sorted(
            rows,
            key=lambda row: float(row[f"mediapipe_t50_{metric}"])
            - float(row[f"apple_{metric}"]),
        )
        extremes = ordered[:count] + list(reversed(ordered[-count:]))
        for row in extremes:
            if row["raw_path"] in seen:
                continue
            item = dict(row)
            item["selection_metric"] = metric
            item["selection_delta"] = str(
                float(row[f"mediapipe_t50_{metric}"])
                - float(row[f"apple_{metric}"])
            )
            selected.append(item)
            seen.add(row["raw_path"])
    return selected


def _point(xy: np.ndarray, index: int, width: int, height: int) -> tuple[int, int]:
    return (
        int(round(float(xy[index, 0]) * width)),
        int(round(float(xy[index, 1]) * height)),
    )


def _draw_hand(image: np.ndarray, xy: np.ndarray, confidence: np.ndarray, side: str) -> None:
    height, width = image.shape[:2]
    color = COLORS[side]
    for first, second in HAND_EDGES:
        if confidence[first] > 0 and confidence[second] > 0:
            cv2.line(
                image, _point(xy, first, width, height),
                _point(xy, second, width, height), color, 2, cv2.LINE_AA,
            )
    for index in np.flatnonzero(confidence > 0):
        cv2.circle(
            image, _point(xy, int(index), width, height),
            3, color, -1, cv2.LINE_AA,
        )


def _annotate(frame: np.ndarray, assigned: dict[str, object], backend: str) -> np.ndarray:
    image = frame.copy()
    labels = []
    for side in ("left", "right"):
        hand = assigned[side]
        if hand is not None:
            _draw_hand(image, hand.xy, hand.confidence, side)
            labels.append(f"{side[0].upper()} {hand.score:.2f}")
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        image, f"{backend}: {', '.join(labels) if labels else 'none'}",
        (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255),
        1, cv2.LINE_AA,
    )
    return cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)


def render_clip(
    row: dict[str, str],
    apple: AppleVisionDetector,
    mediapipe: MediaPipeHybridDetector,
    display_frames: int,
) -> np.ndarray:
    frames, _ = read_video_frames(
        Path(row["raw_path"]), maximum_frames=96, maximum_image_side=960
    )
    mediapipe.reset_sequence()
    previous_apple = {"left": None, "right": None}
    previous_mp = {"left": None, "right": None}
    targets = set(np.linspace(0, len(frames) - 1, display_frames).round().astype(int))
    tiles: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        apple_detection = apple.detect(
            frame, include_body=False, include_face=False, include_hands=True
        )
        mp_detection = mediapipe.detect(
            frame, include_body=False, include_face=False, include_hands=True
        )
        apple_assigned = assign_hands(apple_detection.hands, previous_apple)
        mp_assigned = assign_hands(mp_detection.hands, previous_mp)
        for assigned, previous in (
            (apple_assigned, previous_apple), (mp_assigned, previous_mp)
        ):
            for side in ("left", "right"):
                hand = assigned[side]
                if hand is not None and hand.confidence[0] > 0:
                    previous[side] = hand.xy[0].copy()
        if index in targets:
            tiles.extend(
                (_annotate(frame, apple_assigned, "Apple"),
                 _annotate(frame, mp_assigned, "MediaPipe 0.50"))
            )
    while len(tiles) < display_frames * 2:
        tiles.append(np.zeros((240, 320, 3), dtype=np.uint8))
    header = np.zeros((44, display_frames * 640, 3), dtype=np.uint8)
    caption = (
        f"{row['split']}/{row['label']} {Path(row['raw_path']).name} | "
        f"{row['selection_metric']} delta={float(row['selection_delta']):+.3f} | "
        "left=green right=magenta"
    )
    cv2.putText(
        header, caption, (7, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.53,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return np.vstack((header, np.hstack(tiles)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path,
        default=Path("artifacts/reports/extractor_bakeoff_v17.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/generated/v17_extractor_bakeoff"),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--extremes-per-direction", type=int, default=3)
    parser.add_argument("--display-frames", type=int, default=4)
    parser.add_argument("--rows-per-sheet", type=int, default=6)
    args = parser.parse_args()

    selected = select_disagreements(
        _read_rows(args.csv), args.extremes_per_direction
    )
    config = MediaPipeV17Config(
        minimum_hand_detection_confidence=0.50,
        minimum_hand_presence_confidence=0.50,
        minimum_hand_tracking_confidence=0.50,
        include_apple_auxiliary=False,
    )
    apple = AppleVisionDetector()
    mediapipe = MediaPipeHybridDetector(args.model, config)
    try:
        rendered = [
            render_clip(row, apple, mediapipe, args.display_frames)
            for row in selected
        ]
    finally:
        mediapipe.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for start in range(0, len(rendered), args.rows_per_sheet):
        canvas = np.vstack(rendered[start:start + args.rows_per_sheet])
        path = args.output_dir / f"disagreements_{start // args.rows_per_sheet + 1}.jpg"
        if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"could not write {path}")
        outputs.append(str(path))
    index = [
        {
            "raw_path": row["raw_path"],
            "label": row["label"],
            "split": row["split"],
            "metric": row["selection_metric"],
            "mediapipe_minus_apple": float(row["selection_delta"]),
        }
        for row in selected
    ]
    (args.output_dir / "index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"clips": len(selected), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
