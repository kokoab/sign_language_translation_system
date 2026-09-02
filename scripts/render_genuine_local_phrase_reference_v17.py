#!/usr/bin/env python3
"""Render all nine genuine local phrases through the safe observation/rig contract."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.extract_v17 import AppleVisionDetector, extract_video_v17
from active.v17.landmark_anatomy_v17 import complete_landmark_anatomy
from active.v17.schema_v17 import V17Config
from active.v17.train_signing_voice_v17 import sha256
from scripts.render_signing_voice_phrase_v17 import avatar_panel, put_text
from scripts.extract_how2sign_transition_landmarks_v17 import safe_name


def select_references(
    manifest: dict[str, object], archive_root: Path
) -> list[dict[str, object]]:
    candidates: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in manifest["rows"]:
        if row["role"] != "train":
            continue
        phrase = str(row["phrase_prompt"])
        archive = (
            archive_root / "local_unknown"
            / f"{safe_name(str(row['source_item_id']))}.transition_landmarks_v17.npz"
        )
        with np.load(archive, allow_pickle=False) as payload:
            valid = payload["window_valid"].astype(bool)
            features = payload["landmarks"][valid].astype(np.float32)
        if not len(features):
            continue
        present = features[..., 3] > 0
        candidate = dict(row)
        candidate["_detection_quality"] = float(
            0.5 * present.mean() + 0.5 * present[..., :42].any(axis=2).mean()
        )
        candidates[phrase].append(candidate)
    selected = {}
    for phrase, rows in candidates.items():
        durations = np.asarray([float(row["duration_seconds"]) for row in rows])
        qualities = np.asarray([float(row["_detection_quality"]) for row in rows])
        median_duration = float(np.median(durations))
        threshold = float(np.quantile(qualities, 0.75))
        high_quality = [row for row in rows if float(row["_detection_quality"]) >= threshold]
        selected[phrase] = min(
            high_quality,
            key=lambda row: (
                abs(float(row["duration_seconds"]) - median_duration),
                -float(row["_detection_quality"]),
            ),
        )
        selected[phrase]["_family_train_candidates"] = len(rows)
        selected[phrase]["_family_median_duration"] = median_duration
    if len(selected) != 9:
        raise ValueError(f"expected nine local phrase families, found {len(selected)}")
    return [selected[key] for key in sorted(selected)]


def coordinate_bounds(features: np.ndarray) -> tuple[np.ndarray, float]:
    points = features[..., :2][features[..., 3] > 0]
    low, high = np.percentile(points, (1, 99), axis=0)
    center = (low + high) / 2
    extent = max(float((high - low).max()) * 0.62, 1.0)
    return center, extent


def render_pair(
    observation: np.ndarray,
    rig: np.ndarray,
    frame: int,
    phrase: str,
    bounds: tuple[np.ndarray, float],
) -> np.ndarray:
    canvas = np.full((720, 1280, 3), 8, np.uint8)
    put_text(canvas, phrase.replace("_", " "), (28, 42), 0.85, (245, 247, 250), 2)
    put_text(canvas, "GENUINE LOCAL PERFORMANCE - NO MOTION GENERATION", (28, 74), 0.55, (158, 178, 205), 1)
    left = avatar_panel(observation, frame, (610, 560), *bounds, (56, 186, 255))
    right = avatar_panel(rig, frame, (610, 560), *bounds, (255, 170, 74))
    canvas[104:664, 20:630] = left
    canvas[104:664, 650:1260] = right
    put_text(canvas, "Detector observations", (34, 98), 0.61, (56, 186, 255), 2)
    put_text(canvas, "Render-only completed rig", (664, 98), 0.61, (255, 170, 74), 2)
    put_text(
        canvas,
        "Same real trajectory. Missing detections stay in the observation mask and never become recognizer ground truth.",
        (25, 701), 0.47, (190, 199, 215), 1,
    )
    return canvas


def run(args: argparse.Namespace) -> dict[str, object]:
    manifest = json.loads(args.manifest.read_text())
    rows = select_references(manifest, args.archive_root)
    with np.load(args.anatomy_package, allow_pickle=False) as payload:
        anatomy_metadata = json.loads(str(payload["metadata_json"]))
        template = {
            "absolute_xyz": payload["canonical_absolute_xyz"].astype(np.float32),
            "hand_shapes": payload["canonical_hand_shapes"].astype(np.float32),
            "wrist_from_elbow": payload["canonical_wrist_from_elbow"].astype(np.float32),
        }
    if anatomy_metadata.get("version") != 3:
        raise ValueError("genuine reference renderer requires the v3 hand-participation contract")

    args.output.mkdir(parents=True, exist_ok=True)
    detector = AppleVisionDetector(0.15)
    items = []
    rendered = []
    for row in rows:
        video = Path(row["video_path"])
        if sha256(video) != row["video_sha256"]:
            raise ValueError(f"source video changed: {video}")
        capture = cv2.VideoCapture(str(video))
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        capture.release()
        target_frames = max(4, min(reported_frames, args.maximum_frames))
        config = V17Config(
            target_frames=target_frames,
            maximum_source_frames=target_frames,
            trim_to_hand_activity=False,
        )
        result = extract_video_v17(video, config, detector=detector)
        if result is None:
            raise RuntimeError(f"no usable genuine landmarks: {video}")
        observation = result.features.astype(np.float32)
        rig, observed = complete_landmark_anatomy(observation, template)
        if not np.array_equal(observed, observation[..., 3] > 0):
            raise RuntimeError("render completion changed detector observations")
        observation_sides = np.asarray([
            observed[:, :21].any(), observed[:, 21:42].any()
        ])
        rig_sides = np.asarray([
            (rig[:, :21, 3] > 0).any(), (rig[:, 21:42, 3] > 0).any()
        ])
        if not np.array_equal(observation_sides, rig_sides):
            raise RuntimeError("render rig invented or removed a participating hand")
        phrase = str(row["phrase_prompt"])
        artifact = args.output / f"{phrase.lower()}.genuine_reference_v17.npz"
        metadata = {
            "format": "slt_genuine_local_phrase_reference_v17",
            "version": 1,
            "artifact_contract_version": 3,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "phrase_prompt": phrase,
            "source_item_id": row["source_item_id"],
            "source_role": "train",
            "source_video": video.as_posix(),
            "source_video_sha256": row["video_sha256"],
            "genuine_motion": True,
            "generated_motion": False,
            "reference_only": True,
            "training_eligible": False,
            "validation_eligible": False,
            "test_eligible": False,
        }
        np.savez_compressed(
            artifact,
            animation_rig_xyz=rig[..., :3].astype(np.float16),
            animation_rig_presence=(rig[..., 3] > 0),
            animation_rig_confidence=rig[..., 4].astype(np.float16),
            observation_xyz=observation[..., :3].astype(np.float16),
            observation_presence=observed,
            observation_confidence=observation[..., 4].astype(np.float16),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        item = {
            "phrase": phrase,
            "source_item_id": row["source_item_id"],
            "source_video": video.as_posix(),
            "source_video_sha256": row["video_sha256"],
            "frames": len(observation),
            "fps": fps,
            "observation_presence_fraction": float(observed.mean()),
            "frames_with_any_hand_observed": int(observed[:, :42].any(axis=1).sum()),
            "hand_participation": observation_sides.tolist(),
            "selection_detection_quality": float(row["_detection_quality"]),
            "family_train_candidates": int(row["_family_train_candidates"]),
            "family_median_duration_seconds": float(row["_family_median_duration"]),
            "artifact": artifact.as_posix(),
            "artifact_sha256": sha256(artifact),
        }
        items.append(item)
        rendered.append((phrase, observation, rig, coordinate_bounds(rig), fps))

    temporary = args.output / "genuine_local_phrase_reference.mp4v.mp4"
    video_path = args.output / "genuine_local_phrase_reference.mp4"
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), args.output_fps, (1280, 720)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the genuine reference video")
    for phrase, observation, rig, bounds, source_fps in rendered:
        first = render_pair(observation, rig, 0, phrase, bounds)
        for _ in range(args.output_fps // 2):
            writer.write(first)
        repeats = max(1, round(args.output_fps / source_fps))
        for frame in range(len(observation)):
            image = render_pair(observation, rig, frame, phrase, bounds)
            for _ in range(repeats):
                writer.write(image)
    writer.release()
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(temporary),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(video_path),
    ], check=True)
    temporary.unlink()
    preview = args.output / "preview.png"
    phrase, observation, rig, bounds, _ = rendered[0]
    cv2.imwrite(str(preview), render_pair(observation, rig, len(observation) // 2, phrase, bounds))
    thumbnails = []
    for phrase, observation, rig, bounds, _ in rendered:
        image = render_pair(observation, rig, len(observation) // 2, phrase, bounds)
        thumbnails.append(cv2.resize(image, (426, 240), interpolation=cv2.INTER_AREA))
    contact_sheet = args.output / "contact_sheet.png"
    cv2.imwrite(str(contact_sheet), np.vstack([
        np.hstack(thumbnails[start:start + 3]) for start in range(0, 9, 3)
    ]))
    report = {
        "format": "slt_genuine_local_phrase_reference_report_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": (
            "all train-role recordings audited; within the top detection-quality "
            "quartile, choose the duration closest to the family median"
        ),
        "items": items,
        "video": video_path.as_posix(),
        "video_sha256": sha256(video_path),
        "preview": preview.as_posix(),
        "preview_sha256": sha256(preview),
        "contact_sheet": contact_sheet.as_posix(),
        "contact_sheet_sha256": sha256(contact_sheet),
        "anatomy_package": args.anatomy_package.as_posix(),
        "anatomy_package_sha256": sha256(args.anatomy_package),
        "genuine_motion": True,
        "generated_motion": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, default=Path("active/v17/local_phrase_motion_manifest_v17.json"))
    value.add_argument("--anatomy-package", type=Path, default=Path("artifacts/models/signing_landmark_anatomy_v17_v3/anatomy.npz"))
    value.add_argument("--archive-root", type=Path, default=Path("data/local/local_phrase_motion_landmarks_v17"))
    value.add_argument("--output", type=Path, default=Path("artifacts/reports/stage2_v17_genuine_local_phrase_reference_v1"))
    value.add_argument("--maximum-frames", type=int, default=256)
    value.add_argument("--output-fps", type=int, default=30)
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"items": len(result["items"]), "video": result["video"]}, indent=2))
