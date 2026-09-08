#!/usr/bin/env python3
"""Propose ASL STEM Wiki sign bounds with the frozen v17 models."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np


def ensure_repo_imports():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


ensure_repo_imports()


DURATION_SECONDS = (8/30, 12/30, 16/30, 20/30, 24/30, 28/30, 32/30, 40/30, 48/30, 64/30)


def candidate_spans(frame_count, fps, anchor, radius_seconds=5.0):
    """Return half-open candidate spans around an anchor."""
    radius = round(radius_seconds * fps)
    search_start = max(0, int(anchor) - radius)
    search_end = min(frame_count, int(anchor) + radius + 1)
    stride = max(1, round(fps / 10))
    spans = set()
    for seconds in DURATION_SECONDS:
        duration = max(4, round(seconds * fps))
        for start in range(search_start, search_end - duration + 1, stride):
            spans.add((start, start + duration))
    return sorted(spans)


def annotation_confidence(candidate):
    return float(np.clip(
        .45 * candidate["full_target_probability"]
        + .15 * candidate["coarse_target_probability"]
        + .15 * bool(candidate["full_top1_matches"])
        + .10 * bool(candidate["ctc_agrees"])
        + .10 * candidate["boundary_stability"]
        + .05 * candidate["motion_score"],
        0, 1,
    ))


def select_annotation(candidates):
    if not candidates:
        return {"confidence": 0.0, "has_target_evidence": False}
    scored = []
    for candidate in candidates:
        value = dict(candidate)
        value["has_target_evidence"] = bool(
            value["full_top1_matches"] or value["ctc_agrees"]
        )
        value["confidence"] = annotation_confidence(value)
        scored.append(value)
    return max(scored, key=lambda row: (row["confidence"], -row["end_frame"] + row["start_frame"]))


def calibrate_high_threshold(rows, minimum_successes=3):
    failures = [row["confidence"] for row in rows if not row["review_success"]]
    threshold = round(min(1.0, max(failures, default=.79) + .01), 6)
    successes = sum(
        row["review_success"] and row["confidence"] >= threshold for row in rows
    )
    return threshold if successes >= minimum_successes else None


def assign_confidence_tier(annotation, high_threshold):
    if not annotation.get("has_target_evidence"):
        return "abstain"
    if high_threshold is not None and annotation["confidence"] >= high_threshold:
        return "high"
    return "review"


def boundary_iou(first_start, first_end, second_start, second_end):
    intersection = max(0, min(first_end, second_end) - max(first_start, second_start) + 1)
    union = max(first_end, second_end) - min(first_start, second_start) + 1
    return intersection / union if union else 0.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def softmax(value):
    value = np.asarray(value, dtype=np.float64)
    scaled = np.exp(value - value.max())
    return scaled / scaled.sum()


def merged_ranges(ranges):
    output = []
    for start, end in sorted(ranges):
        if output and start <= output[-1][1]:
            output[-1] = (output[-1][0], max(end, output[-1][1]))
        else:
            output.append((start, end))
    return output


def requested_frame(frame_index, ranges):
    return any(start <= frame_index < end for start, end in ranges)


def video_shape(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frames < 4 or fps <= 0:
        raise ValueError(f"invalid video metadata: {path}")
    return frames, fps


def row_anchor(row, frame_count, token_counts):
    for key in ("proposed_frame", "proposed_start_frame"):
        if row.get(key) not in (None, ""):
            return int(float(row[key])), "pseudo_annotation"
    count = token_counts[row["filename"]]
    position = int(row["manual_token_index"])
    return round((position + .5) / count * frame_count), "manual_sequence_position"


def observe_ranges(path, ranges, detector, detection_side=640, maximum_side=1280):
    from active.v17.extract_v17 import (
        FrameDetection, assign_hands, limit_image_side, orient_frame,
    )
    from scripts.live_isolated_v17 import (
        ObservedFrame, observation_quality, wrist_motion,
    )

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    observations = {}
    frame_index = 0
    previous = {"left": None, "right": None}
    previous_requested = False
    while frame_index < ranges[-1][1]:
        ok, raw = capture.read()
        if not ok:
            break
        active = requested_frame(frame_index, ranges)
        if active and not previous_requested:
            previous = {"left": None, "right": None}
        if active:
            canonical = orient_frame(raw, 0, False)
            detection_frame = limit_image_side(canonical, detection_side)
            frame = limit_image_side(canonical, maximum_side)
            auxiliary = frame_index % 3 == 0
            detection = detector.detect(
                detection_frame, include_body=auxiliary,
                include_face=auxiliary, include_hands=True,
            )
            assigned = assign_hands(detection.hands, previous)
            model_detection = FrameDetection(
                detection.hands, detection.body_xy, detection.body_confidence,
                detection.face_xy, detection.face_confidence,
            )
            hand_quality, face_quality = observation_quality(detection)
            observations[frame_index] = ObservedFrame(
                frame, model_detection, assigned, frame_index / fps,
                wrist_motion(assigned, previous), hand_quality, face_quality,
                face_for_features=auxiliary,
            )
        previous_requested = active
        frame_index += 1
    capture.release()
    expected = sum(end - start for start, end in ranges)
    if len(observations) != expected:
        raise RuntimeError(f"decoded {len(observations)}/{expected} requested frames from {path}")
    return observations


def observation_slice(observations, start, end):
    try:
        return [observations[index] for index in range(start, end)]
    except KeyError:
        return []


def coarse_candidates(spans, observations, model, output_name, target_index):
    from active.v17.schema_v17 import V17Config
    from scripts.live_isolated_v17 import landmarks_from_observations

    rows = []
    for start, end in spans:
        selected = observation_slice(observations, start, end)
        if len(selected) < 4:
            continue
        try:
            features, _ = landmarks_from_observations(selected, V17Config())
        except ValueError:
            continue
        raw = np.asarray(model.predict({"landmarks": features[None]})[output_name]).reshape(-1)
        probabilities = softmax(raw[:100])
        rows.append({
            "start": start, "end": end,
            "coarse_target_probability": float(probabilities[target_index]),
            "coarse_top1_matches": int(probabilities.argmax()) == target_index,
        })
    for row in rows:
        nearby = [
            other for other in rows
            if abs(other["start"] - row["start"]) <= 4
            and abs(other["end"] - row["end"]) <= 4
            and other["coarse_top1_matches"]
        ]
        row["boundary_stability"] = min(1.0, len(nearby) / 3)
    return rows


def full_candidate(row, observations, stage2, target_index):
    selected = observation_slice(observations, row["start"], row["end"])
    frozen, ctc = stage2.classify_window(selected, [])
    result = dict(row)
    result.update({
        "start_frame": row["start"], "end_frame": row["end"] - 1,
        "motion_score": min(1.0, float(np.mean([item.motion for item in selected])) / .02),
        "ctc_hypothesis": ctc.get("hypothesis", []),
        "ctc_agrees": stage2.labels[target_index] in ctc.get("hypothesis", []),
        "stage2_diagnostics": ctc.get("diagnostics", {}),
    })
    if frozen is None:
        result.update({
            "full_target_probability": 0.0, "full_top1_matches": False,
            "full_top3": [],
        })
        return result
    probabilities = softmax(np.asarray(frozen)[:, -100:].mean(axis=0))
    order = np.argsort(probabilities)[::-1][:3]
    result.update({
        "full_target_probability": float(probabilities[target_index]),
        "full_top1_matches": int(order[0]) == target_index,
        "full_top3": [
            {"gloss": stage2.labels[int(index)], "probability": float(probabilities[index])}
            for index in order
        ],
    })
    return result


def scan_row(row, frame_count, fps, token_counts, observations, coarse_model, coarse_output, stage2):
    expected = row["canonical_label"]
    target_index = stage2.labels.index(expected)
    anchor, anchor_source = row_anchor(row, frame_count, token_counts)
    spans = candidate_spans(frame_count, fps, anchor)
    coarse = coarse_candidates(spans, observations, coarse_model, coarse_output, target_index)
    finalists = sorted(
        coarse, key=lambda value: value["coarse_target_probability"], reverse=True
    )[:5]
    full = [full_candidate(candidate, observations, stage2, target_index) for candidate in finalists]
    selected = select_annotation(full)
    selected.update({
        "queue_index": int(row["queue_index"]), "participant": row["participant"],
        "filename": row["filename"], "raw_gloss": row["raw_gloss"],
        "canonical_label": expected, "citizen_asl_lex_code": row["citizen_asl_lex_code"],
        "anchor_frame": anchor, "anchor_source": anchor_source,
        "search_start_frame": max(0, anchor - round(5 * fps)),
        "search_end_frame": min(frame_count - 1, anchor + round(5 * fps)),
        "coarse_candidates": len(coarse), "full_candidates": len(full),
        "human_review_present": bool(row["human_review_present"]),
    })
    return selected


def reviewed_success(annotation, row):
    if not row["human_review_present"]:
        return None
    human_positive = (
        row["signer_quality_decision"] == "yes"
        and row["variant_decision"] == "yes"
        and row["verified_start_frame"] and row["verified_end_frame"]
    )
    if not human_positive or "start_frame" not in annotation:
        return False
    start, end = int(row["verified_start_frame"]), int(row["verified_end_frame"])
    annotation["human_start_frame"] = start
    annotation["human_end_frame"] = end
    annotation["boundary_iou"] = boundary_iou(
        annotation["start_frame"], annotation["end_frame"], start, end
    )
    annotation["maximum_endpoint_error_frames"] = max(
        abs(annotation["start_frame"] - start), abs(annotation["end_frame"] - end)
    )
    return bool(
        annotation["has_target_evidence"]
        and annotation["boundary_iou"] >= .5
        and annotation["maximum_endpoint_error_frames"] <= 5
    )


def run(args):
    import coremltools as ct
    from active.v17.extract_v17 import AppleVisionDetector
    from scripts.live_reel_stage1_v17 import parser as reel_parser
    from scripts.live_stage2_ctc_v17 import LiveStage2CTC, _model_output_name, parser as ctc_parser

    with args.queue.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows):
        row["queue_index"] = index
        row["human_review_present"] = any(
            row[key] for key in (
                "signer_quality_decision", "variant_decision", "verified_start_frame",
                "verified_end_frame", "reviewer_notes",
            )
        )
    acquisition = json.loads(args.acquisition_manifest.read_text())
    token_counts = {
        row["filename"]: len(row["gloss_tokens"])
        for row in acquisition["downloaded_videos"]
    }
    by_video = {}
    for row in rows:
        by_video.setdefault(row["video_path"], []).append(row)

    reel_args = reel_parser().parse_args([])
    ctc_args = ctc_parser().parse_args([])
    coarse_model = ct.models.MLModel(
        str(reel_args.orientation_coreml), compute_units=ct.ComputeUnit.ALL
    )
    coarse_output = _model_output_name(coarse_model)
    stage2 = LiveStage2CTC(ctc_args)
    detector = AppleVisionDetector(.15)
    annotations = []
    started = time.monotonic()
    for video_number, (video_path, video_rows) in enumerate(by_video.items(), start=1):
        path = Path(video_path)
        frames, fps = video_shape(path)
        anchors = [row_anchor(row, frames, token_counts)[0] for row in video_rows]
        ranges = merged_ranges([
            (max(0, anchor - round(5 * fps)), min(frames, anchor + round(5 * fps) + 1))
            for anchor in anchors
        ])
        observations = observe_ranges(path, ranges, detector)
        for row in video_rows:
            annotation = scan_row(
                row, frames, fps, token_counts, observations,
                coarse_model, coarse_output, stage2,
            )
            annotation["review_success"] = reviewed_success(annotation, row)
            annotations.append(annotation)
        partial = {
            "format": "slt_v17_asl_stem_wiki_auto_annotation",
            "complete": False, "processed_videos": video_number,
            "total_videos": len(by_video), "annotations": annotations,
        }
        atomic_json(args.output, partial)
        print(
            f"{video_number}/{len(by_video)} videos | {len(annotations)}/{len(rows)} rows | "
            f"{time.monotonic() - started:.1f}s",
            flush=True,
        )

    reviewed = [row for row in annotations if row["review_success"] is not None]
    threshold = calibrate_high_threshold(reviewed)
    for annotation in annotations:
        annotation["confidence_tier"] = (
            "human_reviewed" if annotation["human_review_present"]
            else assign_confidence_tier(annotation, threshold)
        )
        annotation["automatic_training_eligible"] = False
    successful = sum(bool(row["review_success"]) for row in reviewed)
    report = {
        "format": "slt_v17_asl_stem_wiki_auto_annotation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "queue": args.queue.as_posix(), "queue_sha256_after_read": sha256(args.queue),
        "checkpoint_contract": {
            "coarse_landmark_model": str(reel_args.orientation_coreml),
            "full_multimodal_encoder": str(ctc_args.stage2_encoder),
            "ctc_primary": str(ctc_args.stage2_primary),
            "ctc_specialist": str(ctc_args.stage2_specialist),
        },
        "search_radius_seconds": 5.0,
        "reviewed_calibration_rows": len(reviewed),
        "review_successes": successful,
        "review_success_rate": successful / len(reviewed) if reviewed else 0.0,
        "high_confidence_threshold": threshold,
        "untouched_rows": sum(not row["human_review_present"] for row in rows),
        "confidence_tiers": {
            tier: sum(row["confidence_tier"] == tier for row in annotations)
            for tier in ("high", "review", "abstain", "human_reviewed")
        },
        "human_rows_overwritten": 0,
        "automatic_training_eligible": 0,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "rit_evaluation_accessed": False,
        "annotations": sorted(annotations, key=lambda row: row["queue_index"]),
    }
    atomic_json(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=Path(
        "artifacts/reports/asl_stem_wiki_manual_admission_v17/expert_review_queue.csv"
    ))
    parser.add_argument("--acquisition-manifest", type=Path, default=Path(
        "data/local/asl_stem_wiki_bootstrap_v1/manual_candidate_manifest.json"
    ))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/reports/asl_stem_wiki_auto_annotation_v17/annotations.json"
    ))
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        key: report[key] for key in (
            "reviewed_calibration_rows", "review_successes", "review_success_rate",
            "high_confidence_threshold", "untouched_rows", "confidence_tiers",
            "human_rows_overwritten", "automatic_training_eligible",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
