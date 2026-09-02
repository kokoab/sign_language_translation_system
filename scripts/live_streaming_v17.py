#!/usr/bin/env python3
"""Experimental no-pause v17 recognition with overlapping isolated-sign windows.

This keeps the proven live_isolated_v17.py path unchanged. It continuously classifies
the newest fixed-duration window and stabilizes repeated predictions; it is not a CTC
decoder and cannot reliably separate two adjacent repetitions of the same sign.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from active.v17.extract_v17 import (
    AppleVisionDetector,
    assign_hands,
    limit_image_side,
    orient_frame,
)
from active.v17.schema_v17 import V17Config
from scripts.live_isolated_v17 import (
    MODE_BUTTONS,
    IsolatedClassifier,
    LiveSpeaker,
    ObservedFrame,
    SessionRecorder,
    atomic_json,
    clicked_control,
    clicked_mode,
    draw_detection,
    make_naturalizer,
    observation_quality,
    parser as isolated_parser,
    utc_now,
    wrist_motion,
)


WINDOW_NAME = "SLT v17 streaming-window experiment"


@dataclass
class StreamingStabilizer:
    required_hits: int = 2
    release_hits: int = 2
    candidate: str | None = None
    hits: int = 0
    gaps: int = 0
    emitted: bool = False

    def reset(self) -> None:
        self.candidate = None
        self.hits = 0
        self.gaps = 0
        self.emitted = False

    def update(self, result: dict[str, object]) -> str | None:
        label = str(result.get("gloss", "UNKNOWN"))
        valid = bool(result.get("accepted")) and label != "UNKNOWN"
        if not valid:
            self.gaps += 1
            if self.gaps >= self.release_hits:
                self.reset()
            return None
        self.gaps = 0
        if label != self.candidate:
            self.candidate = label
            self.hits = 1
            self.emitted = False
        else:
            self.hits += 1
        if self.hits < self.required_hits or self.emitted:
            return None
        self.emitted = True
        # ponytail: adjacent identical signs need a sequence model or explicit boundary.
        return label


def draw_stream_hud(
    frame: np.ndarray,
    latest: ObservedFrame | None,
    latest_result: dict[str, object] | None,
    stabilizer: StreamingStabilizer,
    pending: bool,
    fps: float,
    mode: str,
    window_fill: float,
    glosses: list[str],
    sentence: str,
    finish_pending: bool,
    speech_text: str | None,
) -> np.ndarray:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (min(width, 650), 205), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    state = "CLASSIFYING" if pending else "STREAMING"
    quality = "hand --  face --  motion --"
    if latest is not None:
        quality = (
            f"hand {latest.hand_quality:.2f}  face {latest.face_quality:.2f}  "
            f"motion {latest.motion:.3f}"
        )
    lines = [
        f"{state}   {fps:.1f} FPS   {mode.upper()}",
        quality,
        f"window {window_fill * 100:3.0f}%   stable {stabilizer.hits}/{stabilizer.required_hits}",
    ]
    if latest_result is not None:
        lines.append(
            f"candidate {latest_result['gloss']}   evidence "
            f"{float(latest_result['gate_score']):.3f}"
        )
    lines.append("No neutral pose required; overlapping-window Stage 1 experiment")
    for index, text in enumerate(lines):
        cv2.putText(frame, text, (14, 28 + 30 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 1, cv2.LINE_AA)
    for mode_name, (left, top, right, bottom) in MODE_BUTTONS.items():
        selected = mode == mode_name
        color = (55, 130, 70) if selected else (65, 65, 65)
        cv2.rectangle(frame, (left, top), (right, bottom), color, -1)
        cv2.putText(frame, "DEFAULT" if mode_name == "hybrid" else "CASCADE",
                    (left + 8, top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (255, 255, 255), 1, cv2.LINE_AA)

    panel_top = max(224, height - 92)
    cv2.rectangle(frame, (0, panel_top), (width, height), (20, 20, 20), -1)
    shown = "  ".join(glosses) if glosses else "(empty)"
    max_chars = max(18, (width - 36) // 11)
    if len(shown) > max_chars:
        shown = "..." + shown[-(max_chars - 3):]
    cv2.putText(frame, f"GLOSS BUFFER: {shown}", (14, panel_top + 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 1, cv2.LINE_AA)
    status = f"SPEAKING: {speech_text}" if speech_text else sentence
    if finish_pending:
        status = "Naturalizing finished glosses..."
    if status:
        cv2.putText(frame, status[:max_chars], (14, panel_top + 59),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.57, (225, 225, 225), 1,
                    cv2.LINE_AA)
    for action, (left, top, right, bottom) in {
        "reset": (14, height - 54, 134, height - 16),
        "finish": (146, height - 54, 286, height - 16),
    }.items():
        color = (55, 55, 155) if action == "reset" else (35, 145, 70)
        cv2.rectangle(frame, (left, top), (right, bottom), color, -1)
        cv2.rectangle(frame, (left, top), (right, bottom), (245, 245, 245), 1)
        cv2.putText(frame, action.upper(), (left + 18, top + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return frame


def run(args: argparse.Namespace) -> dict[str, object]:
    classifier = IsolatedClassifier(args)
    naturalizer = make_naturalizer(args)
    speaker = None if args.no_speech else LiveSpeaker()
    source = str(args.video) if args.video else f"camera:{args.camera}"
    recorder = SessionRecorder(args, classifier, source)
    recorder.data.update({
        "format": "slt_live_streaming_v17_session",
        "version": 1,
        "prototype_scope": (
            "overlapping-window isolated Stage-1 experiment; not continuous CTC Stage-2"
        ),
    })
    recorder.data["config"].update({
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "stability_hits": args.stability_hits,
        "release_hits": args.release_hits,
    })
    atomic_json(recorder.history_path, recorder.data)
    capture = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not capture.isOpened():
        recorder.close()
        raise RuntimeError(f"could not open {source}")
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    if not args.video:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_fps = source_fps if np.isfinite(source_fps) and source_fps > 1 else 30.0
    detector = AppleVisionDetector(args.minimum_point_confidence)
    window_frames = max(4, math.ceil(args.window_seconds * args.processing_fps))
    observations: deque[ObservedFrame] = deque(maxlen=window_frames)
    stabilizer = StreamingStabilizer(args.stability_hits, args.release_hits)
    previous_wrists = {"left": None, "right": None}
    classifier_executor = ThreadPoolExecutor(max_workers=1)
    language_executor = ThreadPoolExecutor(max_workers=1)
    warm_future = language_executor.submit(naturalizer.warm)
    future: Future | None = None
    naturalizer_future: Future | None = None
    naturalizer_context: dict[str, object] | None = None
    latest: ObservedFrame | None = None
    latest_result: dict[str, object] | None = None
    results: list[dict[str, object]] = []
    glosses: list[str] = []
    pending_speech: deque[dict[str, object]] = deque()
    sentence = ""
    finish_requested = False
    display_times = deque(maxlen=30)
    auxiliary_interval = V17Config().body_interval
    frame_index = processed = 0
    next_process = next_submit = 0.0
    wall_started = time.perf_counter()
    warm_logged = False
    ui_actions: deque[tuple[str, float]] = deque()
    display_size = [1280, 720]

    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME)

        def on_mouse(event, x, y, _flags, _parameter):
            if event != cv2.EVENT_LBUTTONUP:
                return
            mode = clicked_mode(x, y)
            if mode is not None:
                classifier.set_interactive_mode(mode)
                return
            action = clicked_control(x, y, display_size[0], display_size[1])
            if action is not None:
                ui_actions.append((action, time.perf_counter() - wall_started))

        cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    def collect_classifier(wait: bool = False) -> None:
        nonlocal future, latest_result
        if future is None or (not wait and not future.done()):
            return
        result = future.result()
        emitted = stabilizer.update(result)
        result.update({
            "stream_candidate": stabilizer.candidate,
            "stream_candidate_hits": stabilizer.hits,
            "stream_emitted_gloss": emitted,
        })
        latest_result = result
        results.append(result)
        recorder.add(result)
        print(json.dumps(result, indent=2))
        if emitted is not None:
            glosses.append(emitted)
            pending_speech.append({
                "text": emitted,
                "kind": "gloss",
                "reference": len(results) - 1,
            })
        future = None

    def service_language() -> None:
        nonlocal warm_logged, naturalizer_future, naturalizer_context, sentence
        nonlocal finish_requested
        if not warm_logged and warm_future.done():
            warm_logged = True
            recorder.add_event({"type": "naturalizer_warmup", **warm_future.result()})
        if finish_requested and future is None and naturalizer_future is None:
            finish_requested = False
            finished = list(glosses)
            glosses.clear()
            stabilizer.reset()
            if not finished:
                sentence = "No recognized signs to finish."
            else:
                utterance_counter_local = len(recorder.data["utterances"]) + 1
                naturalizer_context = {
                    "utterance_id": f"stream-{utterance_counter_local:04d}",
                    "requested_utc": utc_now(),
                    "glosses": finished,
                }
                naturalizer_future = language_executor.submit(
                    naturalizer.rephrase, finished
                )
        if naturalizer_future is not None and naturalizer_future.done():
            result = naturalizer_future.result()
            utterance = {
                **(naturalizer_context or {}), **result, "completed_utc": utc_now()
            }
            recorder.add_utterance(utterance)
            sentence = str(result["sentence"])
            pending_speech.append({
                "text": sentence,
                "kind": "finished_sentence",
                "reference": utterance.get("utterance_id"),
            })
            naturalizer_future = None
            naturalizer_context = None

    def service_speech() -> None:
        if speaker is None:
            pending_speech.clear()
            return
        while pending_speech:
            item = pending_speech.popleft()
            speaker.enqueue(str(item["text"]), str(item["kind"]), item["reference"])
            recorder.add_event({"type": "speech_queued", **item})
        started = speaker.update()
        if started is not None:
            recorder.add_event({"type": "speech_started", **started})

    def reset_display(seconds: float, source_name: str) -> None:
        nonlocal sentence, finish_requested
        cleared = list(glosses)
        glosses.clear()
        stabilizer.reset()
        sentence = ""
        finish_requested = False
        pending_speech.clear()
        if speaker is not None:
            speaker.clear()
        recorder.add_event({
            "type": "reset", "source": source_name, "seconds": seconds,
            "cleared_glosses": cleared,
        })

    try:
        while True:
            ok, raw = capture.read()
            if not ok:
                break
            seconds = frame_index / source_fps if args.video else time.perf_counter() - wall_started
            frame_index += 1
            canonical = orient_frame(raw, args.rotation, args.input_mirrored)
            recorder.write_frame(canonical, seconds)
            collect_classifier()
            service_language()
            if seconds + 1e-6 >= next_process:
                detection_frame = limit_image_side(canonical, args.detection_image_side)
                frame = limit_image_side(canonical, args.maximum_image_side)
                detection = detector.detect(
                    detection_frame,
                    include_body=processed % auxiliary_interval == 0,
                    include_face=True,
                    include_hands=True,
                )
                assigned = assign_hands(detection.hands, previous_wrists)
                latest = ObservedFrame(
                    frame, detection, assigned, seconds,
                    wrist_motion(assigned, previous_wrists),
                    *observation_quality(detection),
                    face_for_features=processed % auxiliary_interval == 0,
                )
                observations.append(latest)
                processed += 1
                next_process = max(next_process + 1.0 / args.processing_fps, seconds)
                if (
                    future is None
                    and len(observations) == window_frames
                    and seconds + 1e-6 >= next_submit
                ):
                    future = classifier_executor.submit(
                        classifier.classify, list(observations)
                    )
                    next_submit = seconds + args.stride_seconds

            if not args.no_display:
                now = time.perf_counter()
                display_times.append(now)
                fps = (
                    (len(display_times) - 1) / (display_times[-1] - display_times[0])
                    if len(display_times) > 1 else 0.0
                )
                shown = draw_detection(
                    canonical, latest, mirror=not args.no_mirror_display
                )
                display_size[:] = [shown.shape[1], shown.shape[0]]
                shown = draw_stream_hud(
                    shown, latest, latest_result, stabilizer, future is not None, fps,
                    classifier.mode, len(observations) / window_frames, glosses,
                    sentence, finish_requested or naturalizer_future is not None,
                    None if speaker is None else speaker.current_text,
                )
                cv2.imshow(WINDOW_NAME, shown)
                key = cv2.waitKey(1) & 0xFF
                service_speech()
                while ui_actions:
                    action, action_seconds = ui_actions.popleft()
                    if action == "reset":
                        reset_display(action_seconds, "button")
                    else:
                        finish_requested = True
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    reset_display(seconds, "keyboard")
                if key == ord("f"):
                    finish_requested = True
            else:
                service_speech()

        if future is not None:
            collect_classifier(wait=True)
        service_language()
        if naturalizer_future is not None:
            naturalizer_future.result()
            service_language()
        service_speech()
    finally:
        capture.release()
        classifier_executor.shutdown(wait=True)
        language_executor.shutdown(wait=True)
        recorder.close()
        cv2.destroyAllWindows()
    summary = {
        "session": str(recorder.root),
        "history": str(recorder.history_path),
        "video": str(recorder.video_path),
        "windows": len(results),
        "emitted": sum(row.get("stream_emitted_gloss") is not None for row in results),
        "utterances": len(recorder.data["utterances"]),
    }
    print(json.dumps(summary, indent=2))
    return summary


def parser() -> argparse.ArgumentParser:
    value = isolated_parser()
    value.description = __doc__
    value.set_defaults(
        mode="cascade",
        output_root=REPO / "artifacts/reports/live_streaming_v17",
    )
    value.add_argument("--window-seconds", type=float, default=1.2)
    value.add_argument("--stride-seconds", type=float, default=0.20)
    value.add_argument("--stability-hits", type=int, default=2)
    value.add_argument("--release-hits", type=int, default=2)
    return value


def main() -> None:
    args = parser().parse_args()
    if min(
        args.processing_fps, args.window_seconds, args.stride_seconds,
        args.stability_hits, args.release_hits,
    ) <= 0:
        raise ValueError("stream timing and stabilization values must be positive")
    if args.window_seconds > args.maximum_accept_seconds:
        raise ValueError("window-seconds cannot exceed maximum-accept-seconds")
    if args.expected_label:
        args.expected_label = args.expected_label.upper()
    run(args)


if __name__ == "__main__":
    main()
