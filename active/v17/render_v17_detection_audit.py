#!/usr/bin/env python3
"""Render Apple Vision detections on low/median/high-coverage Citizen clips."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.extract_v17 import (
        AppleVisionDetector,
        assign_hands,
        read_video_frames,
    )
else:
    from .extract_v17 import AppleVisionDetector, assign_hands, read_video_frames


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def point(xy: np.ndarray, index: int, width: int, height: int) -> tuple[int, int]:
    return int(round(float(xy[index, 0]) * width)), int(round(float(xy[index, 1]) * height))


def draw_hand(image: np.ndarray, xy: np.ndarray, confidence: np.ndarray, color) -> None:
    height, width = image.shape[:2]
    for first, second in HAND_EDGES:
        if confidence[first] > 0 and confidence[second] > 0:
            cv2.line(image, point(xy, first, width, height), point(xy, second, width, height), color, 2, cv2.LINE_AA)
    for index in np.flatnonzero(confidence > 0):
        cv2.circle(image, point(xy, int(index), width, height), 3, color, -1, cv2.LINE_AA)


def draw_points(image: np.ndarray, xy: np.ndarray, confidence: np.ndarray, color) -> None:
    height, width = image.shape[:2]
    for index in np.flatnonzero(confidence > 0):
        cv2.circle(image, point(xy, int(index), width, height), 3, color, -1, cv2.LINE_AA)


def raw_path(raw_root: Path, feature_path: str) -> Path:
    relative = Path(feature_path)
    filename = relative.name.removesuffix(".v17.npz") + ".mp4"
    return raw_root / relative.parent / filename


def select_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["hand_active_output_frames"]))
    return [rows[0], rows[len(rows) // 2], rows[-1]]


def render_row(
    row: dict[str, str],
    raw_root: Path,
    detector: AppleVisionDetector,
    columns: int,
) -> np.ndarray:
    video = raw_path(raw_root, row["feature_path"])
    frames, _ = read_video_frames(video, maximum_frames=columns, maximum_image_side=960)
    previous = {"left": None, "right": None}
    tiles = []
    for index, frame in enumerate(frames):
        detection = detector.detect(frame, include_body=True, include_face=True)
        assigned = assign_hands(detection.hands, previous)
        overlay = frame.copy()
        for side, color in (("left", (60, 220, 60)), ("right", (220, 70, 220))):
            hand = assigned[side]
            if hand is not None:
                draw_hand(overlay, hand.xy, hand.confidence, color)
                if hand.confidence[0] > 0:
                    previous[side] = hand.xy[0].copy()
        draw_points(overlay, detection.face_xy, detection.face_confidence, (0, 220, 255))
        draw_points(overlay, detection.body_xy, detection.body_confidence, (255, 220, 0))
        caption = (
            f"{row['split']}/{row['label']}  sample {index + 1}/{len(frames)}  "
            f"hands={len(detection.hands)}"
        )
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(overlay, caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tile = cv2.resize(overlay, (320, 240), interpolation=cv2.INTER_AREA)
        tiles.append(tile)
    while len(tiles) < columns:
        tiles.append(np.zeros((240, 320, 3), dtype=np.uint8))
    header = np.zeros((42, columns * 320, 3), dtype=np.uint8)
    text = (
        f"{row['feature_path']} | archived hand-active={float(row['hand_active_output_frames']):.3f} | "
        "LH green, RH magenta, face yellow, body cyan"
    )
    cv2.putText(header, text, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack((header, np.hstack(tiles)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-csv", type=Path, default=Path("artifacts/reports/citizen100_v17_landmark_quality.csv"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/local/citizen100_v17/raw"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/generated/v17_diagnostics/citizen_landmark_overlay_audit.jpg"))
    parser.add_argument("--columns", type=int, default=6)
    args = parser.parse_args()
    detector = AppleVisionDetector()
    rows = [render_row(row, args.raw_root, detector, args.columns) for row in select_rows(args.quality_csv)]
    canvas = np.vstack(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"could not write {args.output}")
    print(args.output)


if __name__ == "__main__":
    main()
