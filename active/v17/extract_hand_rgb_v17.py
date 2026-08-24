#!/usr/bin/env python3
"""Extract Apple-guided left/right/union RGB hand crops without fake pixels."""

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
    from active.v17.extract_mobileclip2_v17 import (
        decode_selected_frames,
        load_landmark_sampling_contract,
        temporal_sample_from_reference,
        validate_decoded_video_contract,
    )
    from active.v17.extract_v17 import AppleVisionDetector, HandDetection, assign_hands
    from active.v17.schema_hand_rgb_v17 import (
        VIEW_NAMES,
        HandRGBV17Config,
        schema_fingerprint,
        schema_payload,
    )
else:
    from .extract_mobileclip2_v17 import (
        decode_selected_frames,
        load_landmark_sampling_contract,
        temporal_sample_from_reference,
        validate_decoded_video_contract,
    )
    from .extract_v17 import AppleVisionDetector, HandDetection, assign_hands
    from .schema_hand_rgb_v17 import (
        VIEW_NAMES,
        HandRGBV17Config,
        schema_fingerprint,
        schema_payload,
    )


LOG = logging.getLogger("hand_rgb_v17")
ALLOWED_SPLITS = ("train", "val")


def video_frame_count(video_path: Path) -> int:
    """Return a trustworthy count when WebM container metadata is absent/garbage."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    reported = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if np.isfinite(reported) and 1 <= reported <= 10_000_000:
        capture.release()
        return int(round(reported))
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    if count < 1:
        raise RuntimeError(f"could not decode frames from {video_path}")
    return count


def load_rejected(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["split"], row["canonical_label"], row["video"])
            for row in csv.DictReader(handle)
        }


def hand_box(
    detection: HandDetection,
    image_width: int,
    image_height: int,
    config: HandRGBV17Config,
) -> np.ndarray | None:
    valid = detection.confidence > 0
    if int(valid.sum()) < config.minimum_joint_count:
        return None
    points = detection.xy[valid].astype(np.float64)
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    center_x = 0.5 * (x0 + x1) * image_width
    center_y = 0.5 * (y0 + y1) * image_height
    detected_side = max((x1 - x0) * image_width, (y1 - y0) * image_height)
    minimum_side = config.minimum_box_long_side_fraction * max(image_width, image_height)
    side = max(detected_side * config.hand_box_scale, minimum_side)
    return np.asarray(
        [center_x - side / 2, center_y - side / 2, center_x + side / 2, center_y + side / 2],
        dtype=np.float32,
    )


def union_box(
    boxes: list[np.ndarray],
    image_width: int,
    image_height: int,
    scale: float,
) -> np.ndarray | None:
    if not boxes:
        return None
    stacked = np.stack(boxes)
    x0, y0 = stacked[:, :2].min(axis=0)
    x1, y1 = stacked[:, 2:].max(axis=0)
    center_x, center_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    side = max(x1 - x0, y1 - y0) * scale
    side = max(side, 1.0)
    return np.asarray(
        [center_x - side / 2, center_y - side / 2, center_x + side / 2, center_y + side / 2],
        dtype=np.float32,
    )


def boxes_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / min(first_area, second_area) >= 0.05


def crop_square(frame: np.ndarray, box: np.ndarray, size: int) -> np.ndarray:
    """Crop a possibly out-of-frame square, padding only with real edge reflection."""
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = [int(round(value)) for value in box]
    pad_left, pad_top = max(0, -x0), max(0, -y0)
    pad_right, pad_bottom = max(0, x1 - width), max(0, y1 - height)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        frame = cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top
    crop = frame[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if crop.size == 0:
        raise ValueError(f"empty crop for box {box.tolist()}")
    interpolation = cv2.INTER_AREA if max(crop.shape[:2]) > size else cv2.INTER_CUBIC
    return cv2.resize(crop, (size, size), interpolation=interpolation)


def encode_jpeg(frame_bgr: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.reshape(-1)


def pack_crops(
    crops: list[list[np.ndarray | None]], quality: int
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.full((len(crops), len(VIEW_NAMES), 2), (-1, 0), dtype=np.int64)
    chunks: list[np.ndarray] = []
    cursor = 0
    for frame_index, frame_crops in enumerate(crops):
        if len(frame_crops) != len(VIEW_NAMES):
            raise ValueError("unexpected view count")
        for view_index, crop in enumerate(frame_crops):
            if crop is None:
                continue
            encoded = encode_jpeg(crop, quality)
            offsets[frame_index, view_index] = (cursor, len(encoded))
            chunks.append(encoded)
            cursor += len(encoded)
    blob = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint8)
    return blob.astype(np.uint8, copy=False), offsets


def decode_packed_crops(
    blob: np.ndarray, offsets: np.ndarray, crop_size: int
) -> np.ndarray:
    output = np.zeros(
        (offsets.shape[0], offsets.shape[1], crop_size, crop_size, 3), dtype=np.uint8
    )
    for frame_index in range(offsets.shape[0]):
        for view_index in range(offsets.shape[1]):
            start, length = offsets[frame_index, view_index]
            if start < 0:
                continue
            decoded = cv2.imdecode(blob[start:start + length], cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape != (crop_size, crop_size, 3):
                raise ValueError("invalid packed JPEG crop")
            output[frame_index, view_index] = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return output


def extract_clip(
    video_path: Path,
    landmark_path: Path,
    detector: AppleVisionDetector,
    config: HandRGBV17Config,
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, object]]:
    trim_start, trim_end, reference, landmark_metadata = load_landmark_sampling_contract(
        landmark_path, config.maximum_reference_frames
    )
    selected = temporal_sample_from_reference(
        reference, trim_start, trim_end, config.sequence_length,
    )
    frames, video_metadata = decode_selected_frames(video_path, selected)
    validate_decoded_video_contract(video_path, video_metadata, landmark_metadata)
    previous_wrists: dict[str, np.ndarray | None] = {"left": None, "right": None}
    crops: list[list[np.ndarray | None]] = []
    boxes = np.zeros((config.sequence_length, len(VIEW_NAMES), 4), dtype=np.float32)
    valid = np.zeros((config.sequence_length, len(VIEW_NAMES)), dtype=np.bool_)
    joint_counts = np.zeros((config.sequence_length, 2), dtype=np.uint8)
    contact = np.zeros(config.sequence_length, dtype=np.bool_)
    chirality_counts = {"left": 0, "right": 0, "unknown": 0}

    for frame_index, frame in enumerate(frames):
        height, width = frame.shape[:2]
        detection = detector.detect(frame, include_body=False, include_face=False)
        for hand in detection.hands:
            chirality_counts[hand.chirality] += 1
        assigned = assign_hands(detection.hands, previous_wrists)
        frame_boxes: list[np.ndarray | None] = []
        observed_boxes: list[np.ndarray] = []
        for view_index, slot in enumerate(("left", "right")):
            hand = assigned[slot]
            box = None if hand is None else hand_box(hand, width, height, config)
            if hand is not None:
                joint_counts[frame_index, view_index] = int((hand.confidence > 0).sum())
                if hand.confidence[0] > 0:
                    previous_wrists[slot] = hand.xy[0].copy()
            if box is not None:
                boxes[frame_index, view_index] = box / np.asarray([width, height, width, height])
                valid[frame_index, view_index] = True
                observed_boxes.append(box)
                frame_boxes.append(crop_square(frame, box, config.crop_size))
            else:
                frame_boxes.append(None)
        union = union_box(observed_boxes, width, height, config.union_box_scale)
        if union is not None:
            boxes[frame_index, 2] = union / np.asarray([width, height, width, height])
            valid[frame_index, 2] = True
            frame_boxes.append(crop_square(frame, union, config.crop_size))
        else:
            frame_boxes.append(None)
        if len(observed_boxes) == 2:
            contact[frame_index] = boxes_overlap(observed_boxes[0], observed_boxes[1])
        crops.append(frame_boxes)

    blob, offsets = pack_crops(crops, config.jpeg_quality)
    metadata = {
        **video_metadata,
        "schema_fingerprint": schema_fingerprint(config),
        "view_names": list(VIEW_NAMES),
        "video_path": str(video_path),
        "landmark_trim_source": str(landmark_path),
        "hand_trim_start_frame": trim_start,
        "hand_trim_end_frame_exclusive": trim_end,
    }
    diagnostics = {
        "valid_fraction_by_view": {
            name: float(valid[:, index].mean()) for index, name in enumerate(VIEW_NAMES)
        },
        "two_hand_frame_fraction": float((valid[:, 0] & valid[:, 1]).mean()),
        "contact_frame_fraction": float(contact.mean()),
        "mean_joints_per_detected_hand": float(joint_counts[joint_counts > 0].mean()) if (joint_counts > 0).any() else 0.0,
        "chirality_observation_counts": chirality_counts,
        "jpeg_bytes": int(len(blob)),
    }
    arrays = {
        "jpeg_blob": blob,
        "jpeg_offsets": offsets,
        "valid": valid,
        "boxes_normalized": boxes.astype(np.float16),
        "joint_counts": joint_counts,
        "contact": contact,
        "selected_raw_frame_indices": selected.astype(np.int64),
    }
    return arrays, metadata, diagnostics


def save_archive(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
    diagnostics: dict[str, object],
    config: HandRGBV17Config,
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
    if args.split not in ALLOWED_SPLITS:
        raise ValueError("the Citizen test split is sealed")
    config = HandRGBV17Config()
    config.validate()
    rejected = load_rejected(args.rejections)
    videos = [
        path for path in sorted((args.raw_root / args.split).glob("*/*.mp4"))
        if (args.split, path.parent.name, path.name) not in rejected
    ]
    if args.limit:
        videos = videos[:args.limit]
    detector = AppleVisionDetector(args.minimum_confidence)
    fingerprint = schema_fingerprint(config)
    written = skipped = 0
    started = time.monotonic()
    for index, video_path in enumerate(videos, start=1):
        relative = video_path.relative_to(args.raw_root / args.split)
        landmark_path = args.landmark_root / args.split / relative.parent / f"{video_path.stem}.v17.npz"
        output_path = args.output_root / args.split / relative.parent / f"{video_path.stem}.hand_rgb_v17.npz"
        if output_path.exists() and not args.overwrite:
            with np.load(output_path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
            if metadata.get("schema_fingerprint") != fingerprint:
                raise ValueError(f"{output_path}: existing schema mismatch")
            skipped += 1
            continue
        arrays, metadata, diagnostics = extract_clip(video_path, landmark_path, detector, config)
        save_archive(output_path, arrays, metadata, diagnostics, config)
        written += 1
        if index == 1 or index % 25 == 0 or index == len(videos):
            LOG.info(
                "%s %d/%d written=%d skipped=%d elapsed=%.1fs",
                args.split, index, len(videos), written, skipped, time.monotonic() - started,
            )
    return {
        "split": args.split,
        "clips": len(videos),
        "written": written,
        "skipped": skipped,
        "schema_fingerprint": fingerprint,
        "seconds": time.monotonic() - started,
        "test_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=ALLOWED_SPLITS)
    parser.add_argument("--raw-root", type=Path, default=Path("data/local/citizen100_v17/raw"))
    parser.add_argument("--landmark-root", type=Path, default=Path("data/local/citizen100_v17/landmarks"))
    parser.add_argument("--output-root", type=Path, default=Path("data/local/citizen100_v17/hand_rgb"))
    parser.add_argument("--rejections", type=Path, default=Path("data/local/citizen100_v17/rejections.csv"))
    parser.add_argument("--minimum-confidence", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
