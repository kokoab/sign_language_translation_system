#!/usr/bin/env python3
"""Experimental continuous camera/video signing with revisable partial text.

No isolated-sign boundary detector, neutral-pose requirement, or per-sign lock is
used. Enter finishes an utterance; signing continues while English is rendered.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.continuous_runtime_v17 import ContinuousRecognizer
from active.v17.continuous_vision_v17 import CausalVisionFeatures
from active.v17.extract_v17 import AppleVisionDetector, assign_hands, limit_image_side, orient_frame
from active.v17.train_streaming_tcn_ctc_v17 import refuse_protected
from scripts.live_isolated_v17 import TinyStage3Naturalizer, DEFAULT_STAGE3_TINY


def main(argv=None, *, shell=None, prebuilt=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/continuous_evidence_v17_v1/best_model.pth"))
    p.add_argument("--familiar-ctc", action="store_true", help="reviewed familiar CTC candidate; literal single-line transcript")
    p.add_argument("--context-checkpoint", type=Path, help="opt-in experimental motion-context correction at utterance end")
    p.add_argument("--context-proposals", action="store_true",
                   help="experimental motion proposals beyond the CTC beam, with exact CTC rescoring")
    p.add_argument("--video", type=Path)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "mps"), default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--stage3-checkpoint", type=Path, default=DEFAULT_STAGE3_TINY)
    p.add_argument("--stage3-device", choices=("cpu", "mps"), default="cpu")
    p.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    p.add_argument("--input-mirrored", action="store_true")
    p.add_argument("--maximum-side", type=int, default=640)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--speak", action="store_true", help="speak completed English utterances with macOS say")
    p.add_argument("--seconds", type=float, default=0)
    p.add_argument("--utterance-gap-seconds", type=float, default=1.2,
                   help="finish after this long with no observed hands; 0 disables automatic endings")
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    if a.context_proposals and a.context_checkpoint is None:
        p.error("--context-proposals requires --context-checkpoint")
    if not np.isfinite(a.utterance_gap_seconds) or a.utterance_gap_seconds < 0:
        p.error("--utterance-gap-seconds must be finite and nonnegative")
    if a.headless and a.video is None and a.seconds <= 0:
        p.error("headless camera runs need --seconds")
    if a.video is not None:
        refuse_protected((a.video,))
    output = a.output or Path("artifacts/reports/continuous_live_v17") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    if a.familiar_ctc:
        if a.context_checkpoint or a.context_proposals:
            p.error("familiar CTC does not use motion-context correction")
        from active.v17.familiar_live_v17 import FamiliarRecognizer
        recognizer = prebuilt or FamiliarRecognizer(a.checkpoint, device=a.device)
    else:
        recognizer = ContinuousRecognizer(a.checkpoint, device=a.device, context_checkpoint=a.context_checkpoint,
                                         context_proposals=a.context_proposals)
    normalizer = CausalVisionFeatures()
    detector = AppleVisionDetector()
    naturalizer = None if a.familiar_ctc else TinyStage3Naturalizer(a)
    capture = cv2.VideoCapture(str(a.video) if a.video else a.camera)
    if not capture.isOpened():
        raise RuntimeError("cannot open camera/video")
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    if not a.video:
        capture.set(cv2.CAP_PROP_FPS, recognizer.config.fps)
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = recognizer.config.fps
    previous = {"left": None, "right": None}
    previous_seen = {"left": -float("inf"), "right": -float("inf")}
    pending, completed, step_times = [], [], []
    utterance = 0
    latest, english = None, ""
    started = time.perf_counter()
    camera_origin = None
    previous_features = None
    last_hand_seconds = 0.
    frame_index, next_sample, feature_samples = 0, 0., 0
    logfile = (output / "events.jsonl").open("w")
    executor = ThreadPoolExecutor(max_workers=1)

    def record(event):
        logfile.write(json.dumps(event) + "\n"); logfile.flush()

    def render_english(glosses):
        if "OTHER" in glosses:
            return {"sentence": " ".join("[unrecognized sign]" if g == "OTHER" else g for g in glosses),
                    "rendering_mode": "unresolved_glosses", "input_glosses": glosses}
        if naturalizer is None:
            return {"sentence": " ".join(glosses), "rendering_mode": "literal", "input_glosses": glosses}
        return naturalizer.rephrase(glosses)

    def finish():
        nonlocal utterance, latest
        final = recognizer.finish()
        if final is not None:
            record(dict(type="final", utterance=utterance, **final))
            if final["glosses"]:
                pending.append((utterance, executor.submit(render_english, list(final["glosses"]))))
            utterance += 1
        recognizer.reset(); latest = None

    def collect(wait=False):
        nonlocal english
        for item in list(pending):
            index, future = item
            if not wait and not future.done():
                continue
            result = future.result()
            english = result["sentence"]
            row = dict(type="english", utterance=index, **result)
            record(row); completed.append(row); pending.remove(item)
            print(json.dumps(row), flush=True)
            if a.speak and result["rendering_mode"] != "unresolved_glosses":
                # Queue speech with the same worker so adjacent utterances cannot overlap.
                executor.submit(subprocess.run, ["say", "--", english], check=True)

    actions = []
    if shell is not None:
        shell.attach("Sign Language Translation", [1280, 720], started,
                     lambda action, seconds: actions.append(action))
    was_paused = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            now = time.perf_counter()
            if camera_origin is None:
                camera_origin = now
            seconds = frame_index / source_fps if a.video else now - camera_origin
            frame_index += 1
            if a.seconds and seconds >= a.seconds:
                break
            if seconds + 1e-8 < next_sample:
                continue
            if shell is not None and shell.paused:
                if not was_paused:
                    recognizer.reset(); normalizer.reset(); latest = None
                    previous_features = None
                was_paused = True
                next_sample = seconds
                key = shell.present(frame, ())
                if key in (ord('q'), ord('Q')): break
                continue
            if was_paused:
                next_sample = seconds
                previous = {"left": None, "right": None}
                previous_seen = {"left": -float("inf"), "right": -float("inf")}
                was_paused = False
            if actions:
                finish(); actions.clear()
            work_started = time.perf_counter()
            frame = limit_image_side(orient_frame(frame, a.rotation, a.input_mirrored), a.maximum_side)
            detection = detector.detect(frame, include_body=True, include_face=True)
            for side in previous:
                if seconds - previous_seen[side] > .5:
                    previous[side] = None
            assigned = assign_hands(detection.hands, previous)
            for side, hand in assigned.items():
                if hand is not None and hand.confidence[0] > 0:
                    previous[side] = hand.xy[0].copy(); previous_seen[side] = seconds
            features = normalizer.add(detection, assigned, frame.shape[1], frame.shape[0])
            if features[:42, 3].any():
                last_hand_seconds = seconds
            # Only an observation already available at a tick may fill that tick.
            # Late camera frames must never be backdated into missed observations.
            while next_sample <= seconds + 1e-8:
                tick_features = features if next_sample >= seconds - 1e-8 or previous_features is None else previous_features
                result = recognizer.add(tick_features)
                feature_samples += 1
                if result is not None:
                    latest = result
                    record(dict(type="partial", utterance=utterance, source_seconds=seconds, **result))
                next_sample += 1 / recognizer.config.fps
            previous_features = features
            if (a.utterance_gap_seconds and latest and latest["glosses"]
                    and seconds - last_hand_seconds >= a.utterance_gap_seconds):
                finish()
            step_times.append(1000 * (time.perf_counter() - work_started))
            collect()
            if not a.headless:
                canvas = cv2.copyMakeBorder(frame, 0, 160, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
                lines = ["Keep signing. Enter: finish sentence. Esc: exit.",
                         "Partial: " + " ".join((latest or {}).get("glosses", [])),
                         "Stable: " + " ".join((latest or {}).get("stable_glosses", [])), english]
                if a.familiar_ctc:
                    lines = ["Keep signing. Enter: clear transcript. Esc: exit.",
                             "Signs: " + " ".join((latest or {}).get("glosses", []))]
                for index, line in enumerate(lines):
                    cv2.putText(canvas, line[:100], (12, frame.shape[0] + 28 + index * 35),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 235, 235), 1, cv2.LINE_AA)
                if shell is None:
                    cv2.imshow("SLT continuous prototype", canvas)
                    key = cv2.waitKey(1) & 0xff
                else:
                    key = shell.present(canvas, (latest or {}).get("glosses", []))
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (10, 13, ord("r"), ord("f")):
                    finish()
        finish(); collect(wait=True)
    finally:
        capture.release()
        executor.shutdown(wait=True)
        logfile.close()
        if not a.headless and shell is None:
            cv2.destroyAllWindows()
    elapsed = time.perf_counter() - started
    report = dict(format="slt_continuous_live_v17", video=str(a.video) if a.video else None,
        checkpoint=str(a.checkpoint), familiar_ctc=a.familiar_ctc, context_checkpoint=str(a.context_checkpoint) if a.context_checkpoint else None,
        context_proposals=a.context_proposals,
        observed_frames=normalizer.frames_seen, feature_ticks=feature_samples, elapsed_seconds=elapsed,
        processing_ms_median=float(np.median(step_times)) if step_times else None,
        processing_ms_p95=float(np.percentile(step_times, 95)) if step_times else None,
        observed_frames_per_wall_second=normalizer.frames_seen / max(elapsed, 1e-6), completed=completed,
        per_sign_input_lock=False, normalization="past-only trailing 32 camera observations",
        utterance_gap_seconds=a.utterance_gap_seconds,
        limitations=["causal camera normalization differs from stored training clips",
                     "30-Hz duplicate ticks do not replace missing camera observations",
                     "context correction is experimental and only runs at utterance end",
                     "end-to-end accuracy and iPhone performance are unproven"], test_accessed=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
