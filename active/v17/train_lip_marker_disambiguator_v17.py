#!/usr/bin/env python3
"""Train a tiny GOOD/THANKYOU classifier from MediaPipe lip landmarks only.

Only the official Citizen train and validation directories are accepted.  The model
is saved as plain NumPy parameters so live inference does not import scikit-learn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from active.v17.lip_marker_v17 import (
    LABELS,
    LipMarkerTracker,
    lip_marker_features,
)


DEFAULT_RAW = Path("data/local/citizen100_v17/raw")
DEFAULT_OUTPUT = Path(
    "artifacts/models/lip_marker_good_thankyou_phrase_crops_v17/model.npz"
)
DEFAULT_PHRASE_MANIFEST = Path("active/v17/stage2_training_manifest_v17.json")


def extract_video(
    path: Path, processing_fps: float, *, end_fraction: float = 1.0,
) -> tuple[np.ndarray | None, dict, list[np.ndarray | None]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 1 else 30.0
    tracker = LipMarkerTracker()
    sequence: list[np.ndarray | None] = []
    frame = 0
    next_seconds = 0.0
    source_frames = max(1.0, capture.get(cv2.CAP_PROP_FRAME_COUNT))
    end_seconds = source_frames / fps * end_fraction
    started = time.perf_counter()
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            seconds = frame / fps
            frame += 1
            if seconds > end_seconds:
                break
            if seconds + 1e-6 < next_seconds:
                continue
            sequence.append(tracker.detect(image))
            next_seconds += 1.0 / processing_fps
    finally:
        tracker.close()
        capture.release()
    features = lip_marker_features(sequence)
    return features, {
        "path": str(path),
        "sampled_frames": len(sequence),
        "detected_frames": sum(value is not None for value in sequence),
        "elapsed_ms": 1000.0 * (time.perf_counter() - started),
        "end_fraction": end_fraction,
    }, sequence


def training_variants(sequence: list[np.ndarray | None]) -> list[np.ndarray]:
    """Approximate live activity crops without re-running MediaPipe."""
    output = []
    count = len(sequence)
    for start, end in (
        (0.0, 1.0), (0.15, 1.0), (0.30, 1.0),
        (0.20, 0.90), (0.35, 0.90), (0.40, 0.85),
    ):
        left = min(count - 1, int(round(start * count)))
        right = max(left + 4, min(count, int(round(end * count))))
        value = lip_marker_features(sequence[left:right])
        if value is not None:
            output.append(value)
    return output


def load_split(root: Path, split: str, processing_fps: float):
    if split not in {"train", "val", "validation"}:
        raise ValueError("only official train and validation splits are permitted")
    split_root = root / split
    features, targets, rows = [], [], []
    for target, label in enumerate(LABELS):
        for path in sorted((split_root / label).glob("*.mp4")):
            vector, row, sequence = extract_video(path, processing_fps)
            row.update({"split": split, "label": label, "usable": vector is not None})
            rows.append(row)
            if vector is not None:
                variants = training_variants(sequence) if split == "train" else [vector]
                features.extend(variants)
                targets.extend([target] * len(variants))
    if not features:
        raise RuntimeError(f"no usable videos in {split_root}")
    return np.stack(features), np.asarray(targets), rows


def load_phrase_split(
    manifest: Path, split: str, processing_fps: float,
):
    role = "validation" if split == "val" else "train"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("citizen_test_accessed") or payload.get("local_test_accessed"):
        raise ValueError("phrase manifest reports forbidden test access")
    features, targets, rows, seen = [], [], [], set()
    for item in payload["rows"]:
        sequence = item.get("target_sequence", [])
        path = Path(item.get("video_path", ""))
        if (
            item.get("role") != role or not sequence
            or sequence[0] not in LABELS or path in seen or not path.is_file()
        ):
            continue
        seen.add(path)
        # Only the first annotated gloss is supervised; later phrase signs are
        # deliberately excluded from this closed mouth-shape specialist.
        end_fraction = 1.0 / len(sequence)
        vector, row, points = extract_video(
            path, processing_fps, end_fraction=end_fraction
        )
        label = sequence[0]
        row.update({
            "split": split, "label": label, "usable": vector is not None,
            "source": "local_phrase_first_segment",
        })
        rows.append(row)
        if vector is not None:
            variants = training_variants(points) if split == "train" else [vector]
            features.extend(variants)
            targets.extend([LABELS.index(label)] * len(variants))
    if not features:
        raise RuntimeError(f"no usable {role} phrase videos in {manifest}")
    return np.stack(features), np.asarray(targets), rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--phrase-manifest", type=Path, default=DEFAULT_PHRASE_MANIFEST,
    )
    parser.add_argument("--processing-fps", type=float, default=15.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    forbidden = {"test", "sealed"}
    if forbidden.intersection(part.lower() for part in args.raw_root.parts):
        raise ValueError("test/sealed paths are forbidden")

    train_x, train_y, train_rows = load_split(
        args.raw_root, "train", args.processing_fps
    )
    val_x, val_y, val_rows = load_split(args.raw_root, "val", args.processing_fps)
    phrase_train_x, phrase_train_y, phrase_train_rows = load_phrase_split(
        args.phrase_manifest, "train", args.processing_fps
    )
    phrase_val_x, phrase_val_y, phrase_val_rows = load_phrase_split(
        args.phrase_manifest, "val", args.processing_fps
    )
    train_x = np.concatenate((train_x, phrase_train_x))
    train_y = np.concatenate((train_y, phrase_train_y))
    val_x = np.concatenate((val_x, phrase_val_x))
    val_y = np.concatenate((val_y, phrase_val_y))
    scaler = StandardScaler().fit(train_x)
    scaled_train = scaler.transform(train_x)
    scaled_val = scaler.transform(val_x)
    candidates = []
    for regularization in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        model = LogisticRegression(
            C=regularization,
            class_weight="balanced",
            max_iter=5000,
            random_state=17,
            solver="liblinear",
        ).fit(scaled_train, train_y)
        probability = model.predict_proba(scaled_val)[:, 1]
        prediction = (probability >= 0.5).astype(np.int64)
        accuracy = float((prediction == val_y).mean())
        correct_margin = float(
            np.mean(np.where(val_y == 1, probability - 0.5, 0.5 - probability))
        )
        candidates.append((accuracy, correct_margin, -regularization, model))
    accuracy, correct_margin, negative_c, model = max(
        candidates, key=lambda row: row[:3]
    )
    probability = model.predict_proba(scaled_val)[:, 1]
    prediction = (probability >= 0.5).astype(np.int64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        format=np.asarray("slt_lip_marker_binary_v17"),
        labels=np.asarray(LABELS),
        mean=scaler.mean_.astype(np.float32),
        scale=np.maximum(scaler.scale_, 1e-6).astype(np.float32),
        coefficient=model.coef_.reshape(-1).astype(np.float32),
        intercept=np.asarray(model.intercept_[0], np.float32),
        processing_fps=np.asarray(args.processing_fps, np.float32),
    )
    report = {
        "format": "slt_lip_marker_binary_training_v17",
        "scope": "closed GOOD-versus-THANKYOU specialist; never a general recognizer",
        "test_split_accessed": False,
        "train_examples": int(len(train_y)),
        "validation_examples": int(len(val_y)),
        "citizen_train_videos": int(len(train_rows)),
        "citizen_validation_examples": int(len(val_rows)),
        "phrase_train_videos": int(len(phrase_train_rows)),
        "phrase_validation_examples": int(len(phrase_val_rows)),
        "validation_accuracy": accuracy,
        "validation_correct_margin": correct_margin,
        "validation_confusion_good_thankyou": confusion_matrix(
            val_y, prediction, labels=(0, 1)
        ).tolist(),
        "selected_c": -negative_c,
        "model_bytes": args.output.stat().st_size,
        "model": str(args.output),
        "videos": train_rows + phrase_train_rows + [
            {**row, "prediction": LABELS[int(value)], "thankyou_probability": float(score)}
            for row, value, score in zip(
                val_rows + phrase_val_rows, prediction, probability
            )
        ],
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
