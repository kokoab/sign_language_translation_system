#!/usr/bin/env python3
"""Extract bounded windowed landmarks and hand-RGB crops for v17 Stage 2."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.extract_hand_rgb_v17 import (
    boxes_overlap,
    crop_square,
    hand_box,
    pack_crops,
    union_box,
)
from active.v17.extract_v17 import (
    AppleVisionDetector,
    assign_hands,
    choose_coarse_orientation_v17,
    extract_frames_v17,
    read_video_frames,
    rotate_frame_clockwise,
)
from active.v17.schema_hand_rgb_v17 import HandRGBV17Config, VIEW_NAMES
from active.v17.schema_stage2_features_v17 import (
    Stage2FeatureV17Config,
    landmark_config,
    schema_fingerprint,
    schema_payload,
)
from active.v17.schema_v17 import MOUTH_END, MOUTH_START, NUM_CHANNELS, NUM_NODES


LOG = logging.getLogger("extract_stage2_multimodal_v17")
ACTIVE_ROLES = {"train", "validation"}
EXTERNAL_ROLE = "external_evaluation_reserved"


def window_ranges(
    frame_count: int, window_size: int = 32, minimum_tail_frames: int = 4
) -> list[tuple[int, int]]:
    if frame_count < minimum_tail_frames:
        return []
    ranges = []
    for start in range(0, frame_count, window_size):
        end = min(frame_count, start + window_size)
        if end - start >= minimum_tail_frames:
            ranges.append((start, end))
    return ranges


def sample_indices(frame_count: int, count: int) -> np.ndarray:
    if frame_count < 1 or count < 1:
        raise ValueError("positive frame and sample counts are required")
    return np.rint(np.linspace(0, frame_count - 1, count)).astype(np.int64)


def safe_name(row: dict[str, Any]) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", row["source_item_id"]).strip("_")
    return f"{stem}_{row['video_sha256'][:12]}"


def extract_hand_window(
    frames: list[np.ndarray],
    detector: AppleVisionDetector,
    config: HandRGBV17Config,
) -> tuple[list[list[np.ndarray | None]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = sample_indices(len(frames), config.sequence_length)
    boxes = np.zeros((config.sequence_length, len(VIEW_NAMES), 4), dtype=np.float32)
    valid = np.zeros((config.sequence_length, len(VIEW_NAMES)), dtype=np.bool_)
    joint_counts = np.zeros((config.sequence_length, 2), dtype=np.uint8)
    contact = np.zeros(config.sequence_length, dtype=np.bool_)
    crops: list[list[np.ndarray | None]] = []
    previous_wrists: dict[str, np.ndarray | None] = {"left": None, "right": None}
    for output_index, source_index in enumerate(selected):
        frame = frames[int(source_index)]
        height, width = frame.shape[:2]
        detection = detector.detect(frame, include_body=False, include_face=False)
        assigned = assign_hands(detection.hands, previous_wrists)
        frame_crops: list[np.ndarray | None] = []
        observed_boxes: list[np.ndarray] = []
        for view_index, slot in enumerate(("left", "right")):
            hand = assigned[slot]
            box = None if hand is None else hand_box(hand, width, height, config)
            if hand is not None:
                joint_counts[output_index, view_index] = int((hand.confidence > 0).sum())
                if hand.confidence[0] > 0:
                    previous_wrists[slot] = hand.xy[0].copy()
            if box is None:
                frame_crops.append(None)
                continue
            boxes[output_index, view_index] = box / np.asarray(
                [width, height, width, height], dtype=np.float32
            )
            valid[output_index, view_index] = True
            observed_boxes.append(box)
            frame_crops.append(crop_square(frame, box, config.crop_size))
        combined = union_box(observed_boxes, width, height, config.union_box_scale)
        if combined is None:
            frame_crops.append(None)
        else:
            boxes[output_index, 2] = combined / np.asarray(
                [width, height, width, height], dtype=np.float32
            )
            valid[output_index, 2] = True
            frame_crops.append(crop_square(frame, combined, config.crop_size))
        if len(observed_boxes) == 2:
            contact[output_index] = boxes_overlap(observed_boxes[0], observed_boxes[1])
        crops.append(frame_crops)
    return crops, valid, boxes, joint_counts, contact


def save_archive(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(path)


def extract_row(
    row: dict[str, Any],
    landmark_detector: AppleVisionDetector,
    hand_detector: AppleVisionDetector,
    stage2_config: Stage2FeatureV17Config,
    hand_config: HandRGBV17Config,
    manifest_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    video_path = Path(row["video_path"])
    frames, video_metadata = read_video_frames(
        video_path,
        stage2_config.maximum_source_frames,
        maximum_image_side=1280,
        rotation="auto",
        input_mirrored=False,
    )
    correction, scores = choose_coarse_orientation_v17(frames, landmark_detector)
    if correction:
        frames = [rotate_frame_clockwise(frame, correction) for frame in frames]
    ranges = window_ranges(
        len(frames), stage2_config.window_source_frames, stage2_config.minimum_tail_frames
    )
    if not ranges:
        raise RuntimeError(f"{video_path}: fewer than four usable source frames")

    landmarks = np.zeros((len(ranges), 32, NUM_NODES, NUM_CHANNELS), dtype=np.float16)
    landmark_valid = np.zeros(len(ranges), dtype=np.bool_)
    hand_valid = np.zeros((len(ranges), 16, len(VIEW_NAMES)), dtype=np.bool_)
    hand_boxes = np.zeros((len(ranges), 16, len(VIEW_NAMES), 4), dtype=np.float16)
    hand_joint_counts = np.zeros((len(ranges), 16, 2), dtype=np.uint8)
    hand_contact = np.zeros((len(ranges), 16), dtype=np.bool_)
    all_crops: list[list[np.ndarray | None]] = []
    window_diagnostics: list[dict[str, Any]] = []
    lconfig = landmark_config()
    for window_index, (start, end) in enumerate(ranges):
        window = frames[start:end]
        result = extract_frames_v17(window, lconfig, detector=landmark_detector)
        if result is not None:
            value = result.features.copy()
            if row.get("zero_lip_nodes", False):
                value[:, MOUTH_START:MOUTH_END] = 0
            landmarks[window_index] = value
            landmark_valid[window_index] = True
            window_diagnostics.append(result.diagnostics)
        else:
            window_diagnostics.append({"no_usable_hand_detections": True})
        crops, valid, boxes, joints, contact = extract_hand_window(
            window, hand_detector, hand_config
        )
        all_crops.extend(crops)
        hand_valid[window_index] = valid
        hand_boxes[window_index] = boxes.astype(np.float16)
        hand_joint_counts[window_index] = joints
        hand_contact[window_index] = contact

    blob, flat_offsets = pack_crops(all_crops, hand_config.jpeg_quality)
    offsets = flat_offsets.reshape(len(ranges), 16, len(VIEW_NAMES), 2)
    arrays = {
        "landmarks": landmarks,
        "landmark_window_valid": landmark_valid,
        "hand_jpeg_blob": blob,
        "hand_jpeg_offsets": offsets,
        "hand_valid": hand_valid,
        "hand_boxes_normalized": hand_boxes,
        "hand_joint_counts": hand_joint_counts,
        "hand_contact": hand_contact,
        "window_source_ranges": np.asarray(ranges, dtype=np.int64),
        "target_indices": np.asarray(row["target_indices"], dtype=np.int64),
    }
    metadata = {
        "schema_fingerprint": schema_fingerprint(stage2_config),
        "schema": schema_payload(stage2_config),
        "training_manifest_sha256": manifest_sha256,
        "source_item_id": row["source_item_id"],
        "source": row["source"],
        "role": row["role"],
        "video_path": row["video_path"],
        "video_sha256": row["video_sha256"],
        "source_group": row["source_group"],
        "signer_id": row.get("signer_id"),
        "target_sequence": row["target_sequence"],
        "zero_lip_nodes": bool(row.get("zero_lip_nodes", False)),
        "lip_supervision": row.get("lip_supervision"),
        "window_count": len(ranges),
        "sampled_source_frames": len(frames),
        "dropped_tail_frames": len(frames) - ranges[-1][1],
        "vision_coarse_rotation_clockwise": correction,
        "vision_orientation_scores": scores,
        "video_metadata": video_metadata,
        "window_diagnostics": window_diagnostics,
        "landmark_valid_windows": int(landmark_valid.sum()),
        "hand_valid_fraction": float(hand_valid.mean()),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    return arrays, metadata


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_sha = sha256(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    if args.external_evaluation:
        rows = [row for row in manifest["rows"] if row["role"] == EXTERNAL_ROLE]
    else:
        rows = [row for row in manifest["rows"] if row["role"] in ACTIVE_ROLES]
    if args.role:
        rows = [row for row in rows if row["role"] == args.role]
    if args.source:
        rows = [row for row in rows if row["source"] == args.source]
    if args.limit:
        rows = rows[: args.limit]
    stage2_config = Stage2FeatureV17Config(maximum_source_frames=args.maximum_source_frames)
    declared_too_long = [
        row["source_item_id"] for row in rows
        if int(row.get("frame_count", 0)) > stage2_config.maximum_source_frames
    ]
    if declared_too_long:
        raise ValueError(
            f"{len(declared_too_long)} manifest videos exceed --maximum-source-frames; "
            f"first={declared_too_long[0]}"
        )
    hand_config = HandRGBV17Config()
    landmark_detector = AppleVisionDetector(0.15)
    hand_detector = AppleVisionDetector(0.15)
    expected = schema_fingerprint(stage2_config)
    written = skipped = failed = 0
    failures = []
    started = time.monotonic()
    for index, row in enumerate(rows, start=1):
        destination = (
            args.output_root / row["role"] / row["source"] /
            f"{safe_name(row)}.stage2_rgb_v17.npz"
        )
        if destination.exists() and not args.overwrite:
            with np.load(destination, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
            if (
                metadata.get("schema_fingerprint") != expected
                or metadata.get("video_sha256") != row["video_sha256"]
                or metadata.get("training_manifest_sha256") != manifest_sha
            ):
                raise ValueError(f"{destination}: stale or incompatible existing archive")
            skipped += 1
            continue
        try:
            arrays, metadata = extract_row(
                row, landmark_detector, hand_detector, stage2_config, hand_config,
                manifest_sha,
            )
            save_archive(destination, arrays, metadata)
            written += 1
        except Exception as exc:
            failed += 1
            failures.append({"source_item_id": row["source_item_id"], "error": str(exc)})
            LOG.exception("failed %s", row["source_item_id"])
        finally:
            gc.collect()
        if index == 1 or index % 10 == 0 or index == len(rows):
            LOG.info(
                "%d/%d written=%d skipped=%d failed=%d elapsed=%.1fs",
                index, len(rows), written, skipped, failed, time.monotonic() - started,
            )
    summary = {
        "manifest": args.manifest.as_posix(),
        "manifest_sha256": manifest_sha,
        "output_root": args.output_root.as_posix(),
        "selected_rows": len(rows),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "failures": failures,
        "schema_fingerprint": expected,
        "seconds": time.monotonic() - started,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("active/v17/stage2_training_manifest_v17.json"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/local/stage2_v17_multimodal"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_multimodal_extraction/latest.json"),
    )
    parser.add_argument("--role", choices=sorted(ACTIVE_ROLES | {EXTERNAL_ROLE}))
    parser.add_argument(
        "--source", choices=("local_phrases", "asllrp_contiguous", "two_m_flores_asl")
    )
    parser.add_argument("--maximum-source-frames", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--external-evaluation", action="store_true",
        help="Process only the permanently reserved RIT external rows after model selection",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
