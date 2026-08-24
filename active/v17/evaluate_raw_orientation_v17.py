#!/usr/bin/env python3
"""Stress a frozen v17 model with pixel-space camera roll before Apple Vision.

This evaluator is deliberately restricted to Citizen train/validation clips. It
rotates decoded upright pixels on an expanded canvas, re-runs Apple Vision, and then
measures both detector coverage and classifier stability. The official test split is
rejected before any filesystem inventory is performed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.extract_v17 import (
    AppleVisionDetector,
    choose_coarse_orientation_v17,
    extract_frames_v17,
    read_video_frames,
    rotate_frame_clockwise,
)
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.schema_v17 import V17Config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_angles(value: str) -> tuple[float, ...]:
    angles = tuple(float(item.strip()) % 360.0 for item in value.split(","))
    if not angles or any(not math.isfinite(item) for item in angles):
        raise argparse.ArgumentTypeError("angles must be a comma-separated finite list")
    if len(set(angles)) != len(angles):
        raise argparse.ArgumentTypeError("angles must be unique modulo 360")
    if angles[0] != 0.0:
        raise argparse.ArgumentTypeError("angles must start with 0 as the stability reference")
    return angles


def load_class_map(manifest_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes = sorted(payload["classes"], key=lambda item: int(item["class_index"]))
    indices = [int(item["class_index"]) for item in classes]
    if indices != list(range(len(classes))):
        raise ValueError("class indices are not contiguous")
    forward = {str(item["canonical_label"]): int(item["class_index"]) for item in classes}
    return forward, {index: label for label, index in forward.items()}


def select_videos(
    raw_root: Path,
    feature_root: Path,
    split: str,
    label_to_index: dict[str, int],
    clips_per_class: int,
) -> list[tuple[str, int, Path]]:
    selected: list[tuple[str, int, Path]] = []
    extensions = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
    for label, target in sorted(label_to_index.items(), key=lambda item: item[1]):
        accepted_stems = {
            path.name.removesuffix(".v17.npz")
            for path in sorted((feature_root / split / label).glob("*.v17.npz"))
        }
        candidates = [
            path
            for path in sorted((raw_root / split / label).iterdir())
            if path.is_file()
            and path.suffix.lower() in extensions
            and path.stem in accepted_stems
        ]
        if len(candidates) < clips_per_class:
            raise ValueError(
                f"{label}: requested {clips_per_class} accepted clips, found {len(candidates)}"
            )
        selected.extend((label, target, path) for path in candidates[:clips_per_class])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--raw-root", type=Path, default=Path("data/local/citizen100_v17/raw"))
    parser.add_argument(
        "--feature-root", type=Path, default=Path("data/local/citizen100_v17/landmarks")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json")
    )
    parser.add_argument("--clips-per-class", type=int, default=1)
    parser.add_argument("--angles", type=parse_angles, default=parse_angles("0,17,37,73,90,123,180,270"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--vision-auto-orient",
        action="store_true",
        help="Select an upright lossless quadrant before the main Vision extraction",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/reports/stage1_v17_raw_orientation_robustness/metrics.json"),
    )
    args = parser.parse_args()
    if args.clips_per_class < 1:
        raise ValueError("clips-per-class must be positive")

    label_to_index, index_to_label = load_class_map(args.manifest)
    selected = select_videos(
        args.raw_root, args.feature_root, args.split, label_to_index, args.clips_per_class
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("not a v17 Stage-1 checkpoint")
    model = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    detector = AppleVisionDetector(V17Config().minimum_point_confidence)

    totals: dict[float, dict[str, float]] = {
        angle: defaultdict(float) for angle in args.angles
    }
    rows: list[dict[str, object]] = []
    reference_predictions: dict[str, int | None] = {}
    for item_index, (label, target, video_path) in enumerate(selected, start=1):
        frames, source_metadata = read_video_frames(
            video_path,
            V17Config().maximum_source_frames,
            V17Config().maximum_image_side,
            rotation="auto",
        )
        key = f"{label}/{video_path.name}"
        for angle in args.angles:
            rotated = frames if angle == 0.0 else [
                rotate_frame_clockwise(frame, angle) for frame in frames
            ]
            coarse_correction = 0.0
            orientation_scores = None
            if args.vision_auto_orient:
                coarse_correction, orientation_scores = choose_coarse_orientation_v17(
                    rotated, detector
                )
                if coarse_correction:
                    rotated = [
                        rotate_frame_clockwise(frame, coarse_correction)
                        for frame in rotated
                    ]
            result = extract_frames_v17(
                rotated,
                V17Config(),
                detector=detector,
                metadata={
                    "source_video": str(video_path),
                    "synthetic_pixel_roll_clockwise": angle,
                    "source_orientation": source_metadata["orientation"],
                    "vision_coarse_rotation_clockwise": coarse_correction,
                    "vision_orientation_scores": orientation_scores,
                },
            )
            bucket = totals[angle]
            bucket["attempted"] += 1
            prediction: int | None = None
            correct = False
            if result is not None:
                features = torch.from_numpy(result.features.astype(np.float32))[None].to(device)
                with torch.inference_mode():
                    logits = model(features)
                prediction = int(logits.argmax(dim=1).item())
                correct = prediction == target
                bucket["extracted"] += 1
                bucket["correct"] += int(correct)
                for name in (
                    "observed_hand_frame_fraction_before_trim",
                    "hand_presence_fraction",
                    "face_presence_fraction",
                    "body_presence_fraction",
                ):
                    bucket[name] += float(result.diagnostics[name])
            if angle == 0.0:
                reference_predictions[key] = prediction
            reference = reference_predictions.get(key)
            agrees = prediction is not None and reference is not None and prediction == reference
            bucket["reference_agreement"] += int(agrees)
            rows.append(
                {
                    "key": key,
                    "label": label,
                    "angle_degrees_clockwise": angle,
                    "vision_coarse_rotation_clockwise": coarse_correction,
                    "extracted": result is not None,
                    "prediction": index_to_label.get(prediction) if prediction is not None else None,
                    "correct": correct,
                    "upright_prediction_agreement": agrees,
                }
            )
        print(f"[{item_index}/{len(selected)}] {key}", flush=True)

    metrics: dict[str, dict[str, float | int]] = {}
    for angle in args.angles:
        bucket = totals[angle]
        attempted = int(bucket["attempted"])
        extracted = int(bucket["extracted"])
        entry: dict[str, float | int] = {
            "attempted": attempted,
            "extracted": extracted,
            "extraction_rate": extracted / max(attempted, 1),
            "correct": int(bucket["correct"]),
            "top1_over_attempted": bucket["correct"] / max(attempted, 1),
            "top1_over_extracted": bucket["correct"] / max(extracted, 1),
            "upright_prediction_agreement_over_attempted": (
                bucket["reference_agreement"] / max(attempted, 1)
            ),
        }
        for name in (
            "observed_hand_frame_fraction_before_trim",
            "hand_presence_fraction",
            "face_presence_fraction",
            "body_presence_fraction",
        ):
            entry[f"mean_{name}"] = bucket[name] / max(extracted, 1)
        metrics[str(angle)] = entry

    report = {
        "format": "slt_v17_raw_pixel_orientation_robustness",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "split": args.split,
        "test_accessed": False,
        "clips_per_class": args.clips_per_class,
        "selected_clips": len(selected),
        "angles_degrees_clockwise": list(args.angles),
        "pixel_transform": "expanded_canvas_no_crop_no_anisotropic_stretch",
        "vision_auto_orient": args.vision_auto_orient,
        "metrics": metrics,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
