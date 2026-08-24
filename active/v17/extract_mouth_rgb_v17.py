#!/usr/bin/env python3
"""Extract real mouth/face pixels inside the frozen v17 hand-active interval."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.extract_v17 import AppleVisionDetector
    from active.v17.schema_mouth_rgb_v17 import (
        CROP_SIZE,
        SEQUENCE_LENGTH,
        MouthRGBV17Config,
        schema_fingerprint,
        schema_payload,
    )
    from active.v17.schema_v17 import V17Config, schema_fingerprint as v17_fingerprint
else:
    from .extract_v17 import AppleVisionDetector
    from .schema_mouth_rgb_v17 import (
        CROP_SIZE,
        SEQUENCE_LENGTH,
        MouthRGBV17Config,
        schema_fingerprint,
        schema_payload,
    )
    from .schema_v17 import V17Config, schema_fingerprint as v17_fingerprint


LOG = logging.getLogger("mouth_rgb_v17")
SPLITS = ("train", "val")


def selected_frame_indices(start: int, end_exclusive: int) -> np.ndarray:
    if start < 0 or end_exclusive <= start:
        raise ValueError("invalid frozen hand-active interval")
    return np.rint(
        np.linspace(start, end_exclusive - 1, SEQUENCE_LENGTH)
    ).astype(np.int64)


def decode_indices(path: Path, indices: np.ndarray) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    wanted = {int(value) for value in indices.tolist()}
    decoded: dict[int, np.ndarray] = {}
    frame_index = 0
    try:
        while wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                decoded[frame_index] = frame
                wanted.remove(frame_index)
            frame_index += 1
    finally:
        capture.release()
    if wanted:
        raise RuntimeError(f"video ended before selected frames {sorted(wanted)}: {path}")
    return [decoded[int(value)] for value in indices]


def mouth_square_box(
    face_xy: np.ndarray,
    face_confidence: np.ndarray,
    width: int,
    height: int,
    config: MouthRGBV17Config,
) -> np.ndarray | None:
    mouth = np.arange(7, 11)
    if not np.all(face_confidence[mouth] > 0):
        return None
    mouth_points = face_xy[mouth] * np.asarray([width, height], dtype=np.float32)
    center = mouth_points.mean(axis=0)
    mouth_width = float(mouth_points[:, 0].max() - mouth_points[:, 0].min())
    jaw = np.arange(11, 14)
    jaw_width = 0.0
    if np.all(face_confidence[jaw] > 0):
        jaw_points = face_xy[jaw] * np.asarray([width, height], dtype=np.float32)
        jaw_width = float(jaw_points[:, 0].max() - jaw_points[:, 0].min())
    side = max(
        mouth_width * config.mouth_width_scale,
        jaw_width * config.jaw_width_scale,
        float(config.minimum_crop_pixels),
    )
    x0, y0 = center - side / 2.0
    x1, y1 = center + side / 2.0
    return np.asarray((x0, y0, x1, y1), dtype=np.float32)


def crop_square(frame: np.ndarray, box: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = [int(round(value)) for value in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty mouth crop: {box.tolist()}")
    crop = frame[y0:y1, x0:x1]
    interpolation = cv2.INTER_AREA if max(crop.shape[:2]) > size else cv2.INTER_CUBIC
    return cv2.resize(crop, (size, size), interpolation=interpolation)


def pack_crops(crops: list[np.ndarray | None], quality: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.full((len(crops), 2), (-1, 0), dtype=np.int64)
    chunks: list[np.ndarray] = []
    cursor = 0
    for index, crop in enumerate(crops):
        if crop is None:
            continue
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("mouth JPEG encoding failed")
        value = encoded.reshape(-1).astype(np.uint8, copy=False)
        offsets[index] = (cursor, len(value))
        chunks.append(value)
        cursor += len(value)
    blob = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint8)
    return blob, offsets


def decode_packed_crops(blob: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    output = np.zeros((len(offsets), CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    for index, (start, length) in enumerate(offsets.tolist()):
        if start < 0:
            continue
        decoded = cv2.imdecode(blob[start : start + length], cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape != output[index].shape:
            raise ValueError("invalid packed mouth JPEG")
        output[index] = decoded
    return output


def load_landmark_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
    if metadata.get("schema_fingerprint") != v17_fingerprint(V17Config()):
        raise ValueError(f"landmark schema mismatch: {path}")
    return metadata


def extract_one(
    video_path: Path,
    landmark_path: Path,
    output_path: Path,
    split: str,
    label: str,
    detector: AppleVisionDetector,
    config: MouthRGBV17Config,
) -> dict[str, object]:
    metadata = load_landmark_metadata(landmark_path)
    indices = selected_frame_indices(
        int(metadata["hand_trim_start_frame"]),
        int(metadata["hand_trim_end_frame_exclusive"]),
    )
    frames = decode_indices(video_path, indices)
    crops: list[np.ndarray | None] = []
    boxes = np.zeros((SEQUENCE_LENGTH, 4), dtype=np.float32)
    valid = np.zeros(SEQUENCE_LENGTH, dtype=np.bool_)
    for index, frame in enumerate(frames):
        detection = detector.detect(
            frame, include_body=False, include_face=True, include_hands=False
        )
        box = mouth_square_box(
            detection.face_xy,
            detection.face_confidence,
            frame.shape[1],
            frame.shape[0],
            config,
        )
        if box is None:
            crops.append(None)
            continue
        crops.append(crop_square(frame, box, config.crop_size))
        boxes[index] = box / np.asarray(
            [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]],
            dtype=np.float32,
        )
        valid[index] = True
    blob, offsets = pack_crops(crops, config.jpeg_quality)
    output_metadata = {
        "schema_fingerprint": schema_fingerprint(config),
        "schema": schema_payload(config),
        "source_landmark_schema_fingerprint": metadata["schema_fingerprint"],
        "split": split,
        "canonical_label": label,
        "video_path": str(video_path),
        "landmark_path": str(landmark_path),
        "hand_trim_start_frame": int(metadata["hand_trim_start_frame"]),
        "hand_trim_end_frame_exclusive": int(metadata["hand_trim_end_frame_exclusive"]),
        "valid_frames": int(valid.sum()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            jpeg_blob=blob,
            jpeg_offsets=offsets,
            valid=valid,
            boxes=boxes.astype(np.float16),
            source_frame_indices=indices,
            metadata_json=np.array(json.dumps(output_metadata)),
        )
    temporary.replace(output_path)
    return output_metadata


def load_rejections(path: Path | None) -> set[tuple[str, str, str]]:
    if path is None or not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["split"], row["canonical_label"], row["video"])
            for row in csv.DictReader(handle)
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--raw-root", type=Path, default=Path("data/local/citizen100_v17/raw"))
    parser.add_argument("--landmark-root", type=Path, default=Path("data/local/citizen100_v17/landmarks"))
    parser.add_argument("--output-root", type=Path, default=Path("data/local/citizen100_v17/mouth_rgb"))
    parser.add_argument("--rejections", type=Path, default=Path("data/local/citizen100_v17/rejections.csv"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = MouthRGBV17Config()
    config.validate()
    rejected = load_rejections(args.rejections)
    videos = [
        path for path in sorted((args.raw_root / args.split).glob("*/*.mp4"))
        if (args.split, path.parent.name, path.name) not in rejected
    ]
    if args.limit:
        videos = videos[: args.limit]
    detector = AppleVisionDetector(config.minimum_face_confidence)
    completed = skipped = failed = 0
    for index, video_path in enumerate(videos, start=1):
        label = video_path.parent.name
        landmark_path = (
            args.landmark_root / args.split / label / f"{video_path.stem}.v17.npz"
        )
        output_path = (
            args.output_root / args.split / label / f"{video_path.stem}.mouth_rgb_v17.npz"
        )
        try:
            if output_path.is_file() and not args.overwrite:
                skipped += 1
            else:
                extract_one(
                    video_path, landmark_path, output_path, args.split, label,
                    detector, config,
                )
                completed += 1
        except Exception as error:
            failed += 1
            LOG.error("failed %s: %s", video_path, error)
        if index % 50 == 0 or index == len(videos):
            LOG.info(
                "progress=%d/%d completed=%d skipped=%d failed=%d",
                index, len(videos), completed, skipped, failed,
            )
    summary = {
        "split": args.split,
        "requested": len(videos),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "test_accessed": False,
        "schema_fingerprint": schema_fingerprint(config),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / f"{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    main()
