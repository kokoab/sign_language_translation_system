#!/usr/bin/env python3
"""Extract full-utterance, eye-aligned mouth/lower-face/full-face RGB views."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.extract_mobileclip2_v17 import decode_selected_frames, reference_indices
from active.v17.extract_hand_rgb_v17 import video_frame_count
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.schema_visual_speech_v17 import (
    CROP_SIZE,
    SEQUENCE_LENGTH,
    VIEW_NAMES,
    VisualSpeechV17Config,
    schema_fingerprint,
    schema_payload,
)


LOG = logging.getLogger("visual_speech_v17")
SPLITS = ("train", "val")


def load_rejections(path: Path | None) -> set[tuple[str, str, str]]:
    if path is None or not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["split"], row["canonical_label"], row["video"])
            for row in csv.DictReader(handle)
        }


def motion_interval(
    shapes: np.ndarray, valid: np.ndarray, config: VisualSpeechV17Config
) -> tuple[int, int, dict[str, object]]:
    """Find a conservative mouth-motion interval over reference-frame positions."""
    count = len(valid)
    if count < 2 or int(valid.sum()) < 3:
        return 0, count, {"mode": "full_utterance_fallback", "active_positions": 0}
    velocity = np.zeros(count, dtype=np.float32)
    pairs = valid[1:] & valid[:-1]
    velocity[1:][pairs] = np.linalg.norm(shapes[1:][pairs] - shapes[:-1][pairs], axis=1)
    positive = velocity[velocity > 0]
    if len(positive) < 2 or float(positive.max()) <= 1e-6:
        return 0, count, {"mode": "full_utterance_fallback", "active_positions": 0}
    threshold = float(np.quantile(positive, config.motion_quantile))
    active = np.flatnonzero(velocity >= threshold)
    if not len(active):
        return 0, count, {"mode": "full_utterance_fallback", "active_positions": 0}
    context = max(1, int(round(count * config.interval_context_fraction)))
    start = max(0, int(active[0]) - context)
    end = min(count, int(active[-1]) + context + 1)
    minimum = min(count, max(SEQUENCE_LENGTH, int(np.ceil(count * config.minimum_interval_fraction))))
    if end - start < minimum:
        center = 0.5 * (start + end - 1)
        start = max(0, int(round(center - (minimum - 1) / 2)))
        end = min(count, start + minimum)
        start = max(0, end - minimum)
    return start, end, {
        "mode": "mouth_motion",
        "active_positions": int(len(active)),
        "threshold": threshold,
        "reference_start": start,
        "reference_end_exclusive": end,
    }


def _affine_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points, np.ones((len(points), 1), dtype=np.float32)), axis=1)
    return homogeneous @ matrix.T


def _square_box(center: np.ndarray, side: float) -> np.ndarray:
    half = max(float(side), 1.0) / 2.0
    return np.asarray((center[0] - half, center[1] - half, center[0] + half, center[1] + half), dtype=np.float32)


def _crop_square(frame: np.ndarray, box: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = [int(round(value)) for value in box]
    left, top = max(0, -x0), max(0, -y0)
    right, bottom = max(0, x1 - width), max(0, y1 - height)
    if any((left, top, right, bottom)):
        frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        x0, x1, y0, y1 = x0 + left, x1 + left, y0 + top, y1 + top
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError(f"empty visual-speech crop: {box.tolist()}")
    interpolation = cv2.INTER_AREA if max(crop.shape[:2]) > size else cv2.INTER_CUBIC
    return cv2.resize(crop, (size, size), interpolation=interpolation)


def aligned_views(
    frame: np.ndarray,
    face_xy: np.ndarray,
    face_confidence: np.ndarray,
    config: VisualSpeechV17Config,
) -> tuple[list[np.ndarray | None], np.ndarray, np.ndarray | None]:
    required = np.asarray((0, 1, 7, 8, 9, 10, 11, 12, 13), dtype=np.int64)
    if not np.all(face_confidence[required] > 0):
        return [None] * len(VIEW_NAMES), np.zeros((len(VIEW_NAMES), 4), np.float32), None
    height, width = frame.shape[:2]
    points = face_xy * np.asarray((width, height), dtype=np.float32)
    left_eye, right_eye = points[0], points[1]
    eye_vector = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_vector))
    if eye_distance < 4.0:
        return [None] * len(VIEW_NAMES), np.zeros((len(VIEW_NAMES), 4), np.float32), None
    eye_center = 0.5 * (left_eye + right_eye)
    angle = float(np.degrees(np.arctan2(eye_vector[1], eye_vector[0])))
    matrix = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1.0).astype(np.float32)
    aligned = cv2.warpAffine(
        frame, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    points = _affine_points(points, matrix)
    mouth = points[7:11]
    mouth_center = mouth.mean(axis=0)
    mouth_width = float(np.ptp(mouth[:, 0]))
    jaw = points[11:14]
    chin = points[12]
    face_min = np.minimum(points[:14].min(axis=0), jaw.min(axis=0))
    face_max = np.maximum(points[:14].max(axis=0), jaw.max(axis=0))
    mouth_side = max(config.minimum_crop_pixels, mouth_width * config.mouth_width_scale, eye_distance * 1.05)
    lower_center = 0.58 * mouth_center + 0.42 * chin
    lower_side = max(config.minimum_crop_pixels, eye_distance * config.lower_face_eye_scale)
    full_center = 0.5 * (face_min + face_max)
    full_side = max(config.minimum_crop_pixels, float(np.max(face_max - face_min)) * config.full_face_scale)
    boxes = np.stack((
        _square_box(mouth_center, mouth_side),
        _square_box(lower_center, lower_side),
        _square_box(full_center, full_side),
    ))
    crops = [_crop_square(aligned, box, config.crop_size) for box in boxes]
    normalized = boxes / np.asarray((width, height, width, height), dtype=np.float32)
    shape = np.asarray(
        (
            float(np.linalg.norm(points[9] - points[10])) / eye_distance,
            mouth_width / eye_distance,
        ),
        dtype=np.float32,
    )
    return crops, normalized, shape


def pack_crops(crops: list[list[np.ndarray | None]], quality: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.full((len(crops), len(VIEW_NAMES), 2), (-1, 0), dtype=np.int64)
    chunks: list[np.ndarray] = []
    cursor = 0
    for frame_index, views in enumerate(crops):
        for view_index, crop in enumerate(views):
            if crop is None:
                continue
            ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError("visual-speech JPEG encoding failed")
            value = encoded.reshape(-1).astype(np.uint8, copy=False)
            offsets[frame_index, view_index] = (cursor, len(value))
            chunks.append(value)
            cursor += len(value)
    return (np.concatenate(chunks) if chunks else np.empty(0, np.uint8)), offsets


def decode_packed_crops(blob: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    output = np.zeros((len(offsets), len(VIEW_NAMES), CROP_SIZE, CROP_SIZE, 3), np.uint8)
    for frame in range(len(offsets)):
        for view in range(len(VIEW_NAMES)):
            start, length = offsets[frame, view]
            if start < 0:
                continue
            decoded = cv2.imdecode(blob[start:start + length], cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape != (CROP_SIZE, CROP_SIZE, 3):
                raise ValueError("invalid packed visual-speech JPEG")
            output[frame, view] = decoded
    return output


def extract_clip(
    video_path: Path, detector: AppleVisionDetector, config: VisualSpeechV17Config
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, object]]:
    frame_count = video_frame_count(video_path)
    reference = reference_indices(frame_count, config.maximum_reference_frames)
    frames, video_metadata = decode_selected_frames(video_path, reference)
    all_crops: list[list[np.ndarray | None]] = []
    all_boxes = np.zeros((len(frames), len(VIEW_NAMES), 4), np.float32)
    shapes = np.zeros((len(frames), 2), np.float32)
    detected = np.zeros(len(frames), np.bool_)
    for index, frame in enumerate(frames):
        observation = detector.detect(
            frame, include_body=False, include_face=True, include_hands=False
        )
        crops, boxes, shape = aligned_views(
            frame, observation.face_xy, observation.face_confidence, config
        )
        all_crops.append(crops)
        all_boxes[index] = boxes
        if shape is not None:
            shapes[index] = shape
            detected[index] = True
    start, end, selection = motion_interval(shapes, detected, config)
    positions = np.rint(np.linspace(start, max(start, end - 1), config.sequence_length)).astype(np.int64)
    selected_crops = [all_crops[int(index)] for index in positions]
    selected_boxes = all_boxes[positions]
    valid = np.asarray(
        [[crop is not None for crop in views] for views in selected_crops], dtype=np.bool_
    )
    blob, offsets = pack_crops(selected_crops, config.jpeg_quality)
    metadata = {
        **video_metadata,
        "video_path": str(video_path),
        "schema_fingerprint": schema_fingerprint(config),
        "view_names": list(VIEW_NAMES),
        "selection": selection,
        "audio_accessed": False,
    }
    diagnostics = {
        "reference_frames": int(len(reference)),
        "reference_face_detection_fraction": float(detected.mean()),
        "valid_fraction_by_view": {
            name: float(valid[:, index].mean()) for index, name in enumerate(VIEW_NAMES)
        },
        "selected_unique_frames": int(len(np.unique(reference[positions]))),
        "jpeg_bytes": int(len(blob)),
    }
    arrays = {
        "jpeg_blob": blob,
        "jpeg_offsets": offsets,
        "valid": valid,
        "boxes_normalized": selected_boxes.astype(np.float16),
        "selected_raw_frame_indices": reference[positions].astype(np.int64),
        "reference_raw_frame_indices": reference.astype(np.int64),
        "reference_face_detected": detected,
        "reference_mouth_shape": shapes.astype(np.float16),
    }
    return arrays, metadata, diagnostics


def save_archive(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
    diagnostics: dict[str, object],
    config: VisualSpeechV17Config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        diagnostics_json=np.array(json.dumps(diagnostics, sort_keys=True)),
        schema_json=np.array(json.dumps(schema_payload(config), sort_keys=True)),
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.split not in SPLITS:
        raise ValueError("visual-speech development accepts only train/val")
    config = VisualSpeechV17Config()
    config.validate()
    rejected = load_rejections(args.rejections)
    videos = [
        path for path in sorted((args.raw_root / args.split).glob("*/*.mp4"))
        if (args.split, path.parent.name, path.name) not in rejected
    ]
    if args.limit:
        videos = videos[:args.limit]
    detector = AppleVisionDetector(config.minimum_face_confidence)
    written = skipped = failed = 0
    started = time.monotonic()
    for index, video in enumerate(videos, start=1):
        relative = video.relative_to(args.raw_root / args.split)
        output = args.output_root / args.split / relative.parent / f"{video.stem}.visual_speech_v17.npz"
        try:
            if output.exists() and not args.overwrite:
                with np.load(output, allow_pickle=False) as payload:
                    metadata = json.loads(str(payload["metadata_json"]))
                if metadata.get("schema_fingerprint") != schema_fingerprint(config):
                    raise ValueError(f"existing schema mismatch: {output}")
                skipped += 1
            else:
                arrays, metadata, diagnostics = extract_clip(video, detector, config)
                metadata.update({"split": args.split, "canonical_label": video.parent.name})
                save_archive(output, arrays, metadata, diagnostics, config)
                written += 1
        except Exception as error:
            failed += 1
            LOG.error("failed %s: %s", video, error)
        if index == 1 or index % 25 == 0 or index == len(videos):
            LOG.info(
                "%s %d/%d written=%d skipped=%d failed=%d elapsed=%.1fs",
                args.split, index, len(videos), written, skipped, failed,
                time.monotonic() - started,
            )
    summary = {
        "split": args.split,
        "requested": len(videos),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "schema_fingerprint": schema_fingerprint(config),
        "audio_accessed": False,
        "test_accessed": False,
        "seconds": time.monotonic() - started,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / f"{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--raw-root", type=Path, default=Path("data/local/citizen100_v17/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/local/citizen100_v17/visual_speech_rgb"))
    parser.add_argument("--rejections", type=Path, default=Path("data/local/citizen100_v17/rejections.csv"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
