#!/usr/bin/env python3
"""Replay four user-described I/GO intervals with the familiar CTC candidate.

This is deliberately a diagnostic, not an accuracy evaluation: the saved video is
compressed and the interval names come from the user's live test, not annotations.
"""
from collections import deque
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from active.v17.continuous_vision_v17 import CausalVisionFeatures
from active.v17.extract_v17 import AppleVisionDetector, assign_hands, limit_image_side, orient_frame
from active.v17.familiar_live_v17 import FamiliarRecognizer
from active.v17.model_unified_streaming_ctc_v17 import load_unified_streaming_head
from active.v17.train_unified_streaming_ctc_v17 import rolling_windows
from scripts.train_local_familiar_ctc_v17 import ROOT


SESSION = ROOT / "artifacts/app_sessions/20260922_074918_286973"
RESULT = Path(__file__).with_name("replay_results.json")
INTERVALS = (
    ("I_1", "I", 26.53, 27.17),
    ("GO_1", "GO", 27.37, 29.54),
)
OLD_HEAD = ROOT / "artifacts/models/unified_streaming_aligned_grounded_v17_v1/best_model.pth"


def top3(probabilities, labels):
    return [
        {"token": int(index), "gloss": labels.get(int(index), "BLANK"), "probability": float(probabilities[index])}
        for index in np.argsort(probabilities)[-3:][::-1]
    ]


def interval_summary(steps, name, intended, start, end):
    chosen = [row for row in steps if start <= row["tick_seconds"] <= end]
    if not chosen:
        return {"interval": name, "user_described_gloss": intended, "start_seconds": start,
                "end_seconds": end, "steps": 0}
    def maximum(key):
        row = max(chosen, key=lambda value: value[key][intended])
        return {"maximum_probability": row[key][intended], "at_tick_seconds": row["tick_seconds"],
                "top3": row[key + "_top3"]}
    return {
        "interval": name, "user_described_gloss": intended, "start_seconds": start, "end_seconds": end,
        "steps": len(chosen),
        "base_classifier": maximum("base"),
        "familiar_ctc": maximum("familiar_ctc"),
        "old_ctc_head_same_evidence": maximum("old_ctc"),
        "familiar_blank_maximum": max(row["familiar_ctc"]["BLANK"] for row in chosen),
        "old_blank_maximum": max(row["old_ctc"]["BLANK"] for row in chosen),
    }


def main():
    torch.set_num_threads(2)
    history = json.loads((SESSION / "history.json").read_text())
    timestamps = history["video_source_timestamps_seconds"]
    if len(timestamps) != 1550:
        raise ValueError("unexpected recorded timestamp count")
    relevant_predictions = [
        {key: value for key, value in prediction.items() if key in ("start_seconds", "end_seconds", "gloss", "candidate_gloss", "committed_gloss", "model_score", "top3", "full_verifier")}
        for prediction in history["predictions"]
        if prediction["start_seconds"] <= 30 and prediction["end_seconds"] >= 26
    ]
    capture = cv2.VideoCapture(str(SESSION / "session_lowres.mp4"))
    if not capture.isOpened():
        RESULT.write_text(json.dumps({
            "format": "live_familiar_replay_v17", "state": "blocked_unreadable_recording",
            "session": str(SESSION), "recorded_timestamp_count": len(timestamps),
            "replay_requested_source_seconds": [24.5, 30.0],
            "reel_predictions_overlapping_26_to_30_seconds": relevant_predictions,
            "failure": "session_lowres.mp4 has no MP4 moov atom; OpenCV and ffprobe cannot decode frames",
            "limitation": "No candidate or old-head scores were inferred. The included Reel rows are model predictions, not annotations.",
        }, indent=2) + "\n")
        return
    recognizer = FamiliarRecognizer(device="cpu")
    old_payload = torch.load(OLD_HEAD, map_location="cpu", weights_only=False)
    old_head = load_unified_streaming_head(old_payload, device="cpu").eval()
    label_to_index = old_payload["label_to_index"]
    labels = {0: "BLANK", 101: "OTHER", **{int(index) + 1: name for name, index in label_to_index.items()}}
    if labels != recognizer.labels | {0: "BLANK"}:
        raise ValueError("candidate and old head label maps differ")
    detector, normalizer = AppleVisionDetector(), CausalVisionFeatures()
    previous = {"left": None, "right": None}; previous_seen = {"left": -float("inf"), "right": -float("inf")}
    old_evidence = deque(maxlen=old_head.config.receptive_field_steps)
    steps, tick_count, frame_count, next_tick, previous_features = [], 0, 0, None, None
    try:
        for frame_number, seconds in enumerate(timestamps):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"saved video ended at frame {frame_number}, expected {len(timestamps)}")
            # Two seconds of causal camera normalization before the first I/GO test.
            if seconds < 24.5 or seconds > 30.0:
                continue
            if next_tick is None:
                next_tick = seconds
            frame_count += 1
            frame = limit_image_side(orient_frame(frame, 0, False), 1280)
            detection = detector.detect(frame, include_body=True, include_face=True)
            for side in previous:
                if seconds - previous_seen[side] > .5:
                    previous[side] = None
            assigned = assign_hands(detection.hands, previous)
            for side, hand in assigned.items():
                if hand is not None and hand.confidence[0] > 0:
                    previous[side] = hand.xy[0].copy(); previous_seen[side] = seconds
            features = normalizer.add(detection, assigned, frame.shape[1], frame.shape[0])
            # Exact live rule: ticks before this camera observation use only the prior observation.
            while next_tick <= seconds + 1e-8:
                tick = features if next_tick >= seconds - 1e-8 or previous_features is None else previous_features
                recognizer.frames.append(tick.copy()); recognizer.frame_count += 1
                if recognizer.frame_count >= 8 and (recognizer.frame_count - 8) % 4 == 0:
                    window = rolling_windows(np.stack(recognizer.frames), stride=4, window_frames=8)[-1]
                    with torch.inference_mode():
                        base_logits, pooled = recognizer.encoder(torch.from_numpy(window).unsqueeze(0), return_embeddings=True)
                        evidence = torch.cat((pooled, base_logits), -1)[0]
                        recognizer.evidence.append(evidence)
                        old_evidence.append(evidence)
                        familiar_logits = recognizer.head(torch.stack(tuple(recognizer.evidence)).unsqueeze(0))[0, -1]
                        old_logits = old_head(torch.stack(tuple(old_evidence)).unsqueeze(0))[0, -1]
                    base = torch.softmax(base_logits[0], -1).numpy()
                    familiar = torch.softmax(familiar_logits, -1).numpy()
                    old = torch.softmax(old_logits, -1).numpy()
                    base_named = {name: float(base[index]) for name, index in label_to_index.items()}
                    familiar_named = {name: float(familiar[index]) for index, name in labels.items()}
                    old_named = {name: float(old[index]) for index, name in labels.items()}
                    steps.append({
                        "tick_seconds": float(next_tick), "source_observation_seconds": float(seconds),
                        "base": base_named, "base_top3": top3(np.r_[0.0, base, 0.0], labels),
                        "familiar_ctc": familiar_named, "familiar_ctc_top3": top3(familiar, labels),
                        "old_ctc": old_named, "old_ctc_top3": top3(old, labels),
                    })
                    tick_count += 1
                next_tick += 1 / recognizer.config.fps
            previous_features = features
    finally:
        capture.release()
    result = {
        "format": "live_familiar_replay_v17", "session": str(SESSION),
        "candidate_checkpoint": str(recognizer.DEFAULT) if hasattr(recognizer, "DEFAULT") else "default FamiliarRecognizer candidate",
        "old_head_checkpoint": str(OLD_HEAD), "recorded_timestamp_count": len(timestamps),
        "replayed_source_seconds": [24.5, 30.0], "replayed_frames": frame_count, "ctc_steps": tick_count,
        "sampling": "recorded source timestamps; 30Hz past-only duplicate rule from live_continuous_v17",
        "intervals": [interval_summary(steps, *item) for item in INTERVALS],
        "reel_predictions_overlapping_26_to_30_seconds": relevant_predictions,
        "limitations": [
            "saved low-resolution session video is compressed and may not match camera pixels exactly",
            "I/GO interval names come from the user's account; Reel predictions are model outputs, not ground truth",
            "old-head scores reuse familiar-candidate rich evidence, so they isolate head behavior but are not an old-pipeline replay",
            "this does not establish live accuracy or signer-independent performance",
        ],
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"replayed_frames": frame_count, "ctc_steps": tick_count, "intervals": result["intervals"]}, indent=2))


if __name__ == "__main__":
    main()
