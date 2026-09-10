#!/usr/bin/env python3
"""Live v17 Stage-2 CTC experiment using non-overlapping continuous windows.

The working isolated and overlapping-window scripts remain unchanged. This path uses
the accepted general CTC selector and its parity-validated Core ML packages.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from active.v17.extract_v17 import (
    AppleVisionDetector,
    assign_hands,
    limit_image_side,
    orient_frame,
)
from active.v17.geometry_v17 import (
    body_relative_normalize,
    image_normalized_to_isotropic,
    interpolate_scalar_short_gaps,
    interpolate_short_gaps,
    resample_features,
)
from active.v17.schema_v17 import (
    BODY_END,
    BODY_START,
    FACE_END,
    FACE_START,
    LHAND_END,
    LHAND_START,
    NUM_CHANNELS,
    NUM_NODES,
    RHAND_END,
    RHAND_START,
    V17Config,
)
from scripts.live_isolated_v17 import (
    LiveSpeaker,
    ObservedFrame,
    SessionRecorder,
    atomic_json,
    clicked_control,
    draw_detection,
    hand_inputs,
    make_naturalizer,
    observation_quality,
    parser as isolated_parser,
    sha256,
    utc_now,
    wrist_motion,
)


WINDOW_NAME = "SLT v17 live Stage-2 CTC experiment"
DEFAULT_ENCODER = REPO / "artifacts/coreml/Stage2FrozenEncoderV17FP32.mlpackage"
DEFAULT_PRIMARY = REPO / "artifacts/coreml/Stage2SelectorPrimaryV17FP32.mlpackage"
DEFAULT_SPECIALIST = REPO / "artifacts/coreml/Stage2SelectorSpecialistV17FP32.mlpackage"
DEFAULT_SELECTOR = (
    REPO / "artifacts/models/stage2_v17_general_ctc_selector_v1/model.pth"
)
DEFAULT_SELECTOR_REPORT = (
    REPO / "artifacts/reports/stage2_v17_general_ctc_selector_v1/validation.json"
)
DEFAULT_VOCABULARY = REPO / "active/v17/citizen100_manifest.json"
WINDOW_SOURCE_FRAMES = 32
MAXIMUM_WINDOWS = 8
TOKENS_PER_WINDOW = 8
FROZEN_FEATURE_DIM = 612
TRAINING_FPS = 30.0
WINDOW_SECONDS = WINDOW_SOURCE_FRAMES / TRAINING_FPS


def _model_output_name(model) -> str:
    return model.get_spec().description.output[0].name


def collapse_ctc_tokens(logits: np.ndarray, length: int) -> tuple[int, ...]:
    prediction = np.asarray(logits).reshape(-1, logits.shape[-1])[:length].argmax(-1)
    output: list[int] = []
    previous = -1
    for value in prediction:
        token = int(value)
        if token != 0 and token != previous:
            output.append(token)
        previous = token
    return tuple(output)


def supported_ctc_path(tokens, positions):
    """Remove auxiliary OTHER after CTC collapse, keeping genuine repetitions."""
    if len(tokens) != len(positions) or any(token < 1 or token > 101 for token in tokens):
        raise ValueError("invalid locked100 CTC path")
    pairs = [(token, position) for token, position in zip(tokens, positions) if token != 101]
    return tuple(token for token, _ in pairs), tuple(position for _, position in pairs)


def validate_preservation_runtime(args, payload, selector):
    expected = payload['accepted_checkpoint']['selector_config']
    if any(selector.get(key) != expected[key] for key in ('blend_weight', 'blank_bias', 'score_margin', 'minimum_tokens')):
        raise ValueError('OTHER preservation selector configuration changed')
    if sha256(args.stage2_selector) != payload['accepted_sha256']:
        raise ValueError('OTHER preservation requires its pinned accepted selector')
    from active.v17.train_stage_2_other_ctc_v17 import directory_sha256
    for name in ('primary', 'specialist', 'encoder', 'image_encoder'):
        path = args.image_encoder if name == 'image_encoder' else getattr(args, f'stage2_{name}')
        if directory_sha256(path) != payload[f'runtime_{name}_sha256']:
            raise ValueError(f'OTHER preservation {name} Core ML package changed')


def collapse_ctc_path(
    logits: np.ndarray, length: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    prediction = np.asarray(logits).reshape(-1, logits.shape[-1])[:length].argmax(-1)
    tokens: list[int] = []
    positions: list[int] = []
    previous = -1
    for position, value in enumerate(prediction):
        token = int(value)
        if token != 0 and token != previous:
            tokens.append(token)
            positions.append(position)
        previous = token
    return tuple(tokens), tuple(positions)


def roll_ctc_prefix(
    locked: list[str], hypothesis: list[str], positions: list[int]
) -> tuple[list[str], list[str], list[int]]:
    """Freeze emissions leaving the eight-window CTC context."""
    exiting = [
        gloss for gloss, position in zip(hypothesis, positions)
        if position < TOKENS_PER_WINDOW
    ]
    retained = [
        gloss for gloss, position in zip(hypothesis, positions)
        if position >= TOKENS_PER_WINDOW
    ]
    shifted = [
        position - TOKENS_PER_WINDOW for position in positions
        if position >= TOKENS_PER_WINDOW
    ]
    return [*locked, *exiting], retained, shifted


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    if not np.isfinite(maximum):
        return float("-inf")
    return float(maximum + math.log(sum(math.exp(value - maximum) for value in values)))


def ctc_sequence_log_probability(logits: np.ndarray, tokens: tuple[int, ...]) -> float:
    """Exact CTC sequence score used by the accepted general selector."""
    value = np.asarray(logits, dtype=np.float64)
    if value.ndim != 2 or not len(value):
        raise ValueError("expected a non-empty [time, classes] CTC stream")
    if any(token <= 0 or token >= value.shape[-1] for token in tokens):
        raise ValueError("invalid non-blank CTC token")
    maximum = value.max(axis=-1, keepdims=True)
    log_probabilities = value - maximum - np.log(
        np.exp(value - maximum).sum(axis=-1, keepdims=True)
    )
    extended = [0]
    for token in tokens:
        extended.extend((int(token), 0))
    states = np.full(len(extended), -np.inf, dtype=np.float64)
    states[0] = log_probabilities[0, 0]
    if tokens:
        states[1] = log_probabilities[0, extended[1]]
    for frame in range(1, len(log_probabilities)):
        following = np.full_like(states, -np.inf)
        for state, token in enumerate(extended):
            paths = [float(states[state])]
            if state >= 1:
                paths.append(float(states[state - 1]))
            if state >= 2 and token != 0 and token != extended[state - 2]:
                paths.append(float(states[state - 2]))
            following[state] = _logsumexp(paths) + log_probabilities[frame, token]
        states = following
    return float(states[0] if not tokens else _logsumexp(states[-2:].tolist()))


def select_general_ctc_logits(
    primary: np.ndarray,
    specialist: np.ndarray,
    length: int,
    *,
    blend_weight: float,
    blank_bias: float,
    score_margin: float,
    minimum_tokens: int,
) -> tuple[np.ndarray, bool, tuple[int, ...], tuple[int, ...]]:
    """Mirror Stage2GeneralCTCSelectorV17 without loading the PyTorch graph."""
    primary = np.asarray(primary).reshape(-1, primary.shape[-1])
    specialist = np.asarray(specialist).reshape(-1, specialist.shape[-1])
    calibrated = primary * (1.0 - blend_weight) + specialist * blend_weight
    calibrated = calibrated.copy()
    calibrated[:, 0] += blank_bias
    base_tokens = collapse_ctc_tokens(calibrated, length)
    specialist_tokens = collapse_ctc_tokens(specialist, length)
    selected = False
    if (
        base_tokens != specialist_tokens
        and len(base_tokens) >= minimum_tokens
        and len(base_tokens) == len(specialist_tokens)
    ):
        base_score = ctc_sequence_log_probability(specialist[:length], base_tokens)
        specialist_score = ctc_sequence_log_probability(
            specialist[:length], specialist_tokens
        )
        selected = specialist_score - base_score >= score_margin
    return specialist if selected else calibrated, selected, base_tokens, specialist_tokens


@dataclass
class StablePrefixSpeaker:
    """Speak only the prefix that survives a following CTC update."""

    previous: tuple[str, ...] = ()
    spoken: tuple[str, ...] = ()

    def reset(self) -> None:
        self.previous = ()
        self.spoken = ()

    def update(self, hypothesis: list[str]) -> list[str]:
        current = tuple(hypothesis)
        common = 0
        for left, right in zip(self.previous, current):
            if left != right:
                break
            common += 1
        emitted: list[str] = []
        if current[: len(self.spoken)] == self.spoken and common > len(self.spoken):
            emitted = list(current[len(self.spoken):common])
            self.spoken = current[:common]
        self.previous = current
        return emitted


@dataclass
class ElapsedWindowBuffer:
    """Partition observations by wall time, not by the extractor's achieved FPS."""

    period_seconds: float = WINDOW_SECONDS
    start_seconds: float | None = None
    frames: list[ObservedFrame] = field(default_factory=list)

    def reset(self) -> None:
        self.start_seconds = None
        self.frames.clear()

    def add(self, item: ObservedFrame) -> list[ObservedFrame] | None:
        if self.start_seconds is None:
            self.start_seconds = item.seconds
        self.frames.append(item)
        deadline = self.start_seconds + self.period_seconds
        if item.seconds < deadline:
            return None
        split = next(
            (index for index, frame in enumerate(self.frames) if frame.seconds >= deadline),
            len(self.frames),
        )
        window = self.frames[:split]
        self.frames = self.frames[split:]
        self.start_seconds = deadline
        if not self.frames and item.seconds >= deadline + self.period_seconds:
            self.start_seconds = item.seconds
        return window if len(window) >= 4 else None

    def finish(self) -> list[ObservedFrame] | None:
        window = list(self.frames)
        self.reset()
        return window if len(window) >= 4 else None


class LatestCamera:
    """Drain the webcam continuously and expose only its newest frame."""

    def __init__(self, capture: cv2.VideoCapture, started: float):
        self.capture = capture
        self.started = started
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.sequence = 0
        self.latest: tuple[int, float, np.ndarray] | None = None
        self.failed = False
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        while not self.stop.is_set():
            ok, frame = self.capture.read()
            if not ok:
                self.failed = True
                return
            seconds = time.perf_counter() - self.started
            with self.lock:
                self.sequence += 1
                self.latest = (self.sequence, seconds, frame)

    def after(self, sequence: int) -> tuple[int, float, np.ndarray] | None:
        with self.lock:
            if self.latest is None or self.latest[0] <= sequence:
                return None
            index, seconds, frame = self.latest
            return index, seconds, frame.copy()

    def close(self) -> None:
        self.stop.set()
        self.capture.release()
        self.thread.join(timeout=1.0)


class DisplayLipTracker:
    """MediaPipe lips for the overlay only; model features remain Apple Vision."""

    OUTER_LIP = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                 409, 270, 269, 267, 0, 37, 39, 40, 185)

    def __init__(self, enabled: bool):
        self.mesh = None
        if enabled:
            import mediapipe as mp
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        if self.mesh is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        points = result.multi_face_landmarks[0].landmark
        return np.asarray(
            [(points[index].x, points[index].y) for index in self.OUTER_LIP],
            dtype=np.float32,
        )

    def close(self) -> None:
        if self.mesh is not None:
            self.mesh.close()


def draw_display_lips(
    frame: np.ndarray, lips: np.ndarray | None, *, mirror: bool
) -> np.ndarray:
    if lips is None:
        return frame
    height, width = frame.shape[:2]
    points = lips.copy()
    if mirror:
        points[:, 0] = 1.0 - points[:, 0]
    pixels = [
        (int(round(point[0] * (width - 1))), int(round(point[1] * (height - 1))))
        for point in points
    ]
    for first, second in zip(pixels, pixels[1:] + pixels[:1]):
        cv2.line(frame, first, second, (0, 0, 0), 2, cv2.LINE_AA)
    for point in pixels:
        cv2.circle(frame, point, 2, (255, 255, 255), -1, cv2.LINE_AA)
    return frame


def validate_live_adapted_runtime(args, payload):
    from scripts.cache_stage2_live_matched_v17 import input_contract
    if not (getattr(args, "sequence_preview", False) or getattr(args, "revisable_transcript", False)) or getattr(args, "dense_model_auxiliary", False):
        raise ValueError("live-adapted checkpoint requires the matched sequence-preview input")
    if payload['input_contract'] != input_contract(args):
        raise ValueError("live-adapted checkpoint input contract changed")
    if args.stage2_other_preservation is None or sha256(args.stage2_other_preservation) != payload['design']['teacher_sha256']:
        raise ValueError("live-adapted checkpoint teacher provenance changed")


def observe_stage2_frame(frame, seconds, processed, detector, previous_wrists, args):
    """Shared live/training observation contract; input is already canonically oriented."""
    dense = getattr(args, "dense_model_auxiliary", False)
    face = dense or processed % V17Config().face_interval == 0
    body = dense or processed % V17Config().body_interval == 0
    detection = detector.detect(
        limit_image_side(frame, args.detection_image_side),
        include_body=body, include_face=face, include_hands=True,
    )
    assigned = assign_hands(detection.hands, previous_wrists)
    if not body:
        from active.v17.extract_v17 import FrameDetection
        detection = FrameDetection(
            detection.hands, np.zeros_like(detection.body_xy),
            np.zeros_like(detection.body_confidence), detection.face_xy,
            detection.face_confidence,
        )
    return ObservedFrame(
        limit_image_side(frame, args.maximum_image_side), detection, assigned,
        seconds, wrist_motion(assigned, previous_wrists),
        *observation_quality(detection), face_for_features=face,
    )


def stage2_landmarks_from_observations(
    observations: list[ObservedFrame],
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Build the non-trimmed Stage-2 landmark window from the live Vision pass."""
    if len(observations) < 4:
        return None, {"rejection_reason": "fewer_than_four_frames"}
    config = V17Config(
        target_frames=32,
        maximum_source_frames=32,
        trim_to_hand_activity=False,
    )
    height, width = observations[0].frame.shape[:2]
    if any(item.frame.shape[:2] != (height, width) for item in observations):
        raise ValueError("processed frame dimensions changed within a Stage-2 window")
    count = len(observations)
    xy = np.zeros((count, NUM_NODES, 2), np.float32)
    confidence = np.zeros((count, NUM_NODES), np.float32)
    detector_depth = np.zeros((count, NUM_NODES), np.float32)
    depth_confidence = np.zeros((count, NUM_NODES), np.float32)
    observed_hand_frames = 0
    for frame_index, item in enumerate(observations):
        any_hand = False
        for side, start, end in (
            ("left", LHAND_START, LHAND_END),
            ("right", RHAND_START, RHAND_END),
        ):
            hand = item.assigned[side]
            if hand is None:
                continue
            xy[frame_index, start:end] = hand.xy
            confidence[frame_index, start:end] = hand.confidence
            any_hand |= bool((hand.confidence > 0).any())
            if hand.world_xyz is not None:
                world = np.asarray(hand.world_xyz, dtype=np.float32)
                if world.shape != (21, 3):
                    raise ValueError(f"unexpected hand world shape {world.shape}")
                palm_scale = float(np.linalg.norm(world[9] - world[0]))
                if np.isfinite(palm_scale) and palm_scale > 1e-5:
                    valid = hand.confidence > 0
                    detector_depth[frame_index, start:end] = (
                        world[:, 2] - world[0, 2]
                    ) / palm_scale
                    depth_confidence[frame_index, start:end] = hand.confidence * valid
        observed_hand_frames += int(any_hand)
        if (item.detection.body_confidence > 0).any():
            xy[frame_index, BODY_START:BODY_END] = item.detection.body_xy
            confidence[frame_index, BODY_START:BODY_END] = item.detection.body_confidence
        if item.face_for_features and (item.detection.face_confidence > 0).any():
            xy[frame_index, FACE_START:FACE_END] = item.detection.face_xy
            confidence[frame_index, FACE_START:FACE_END] = item.detection.face_confidence
    if observed_hand_frames < config.minimum_detected_hand_frames:
        return None, {
            "rejection_reason": "insufficient_hand_evidence",
            "source_frames": count,
            "observed_hand_frames": observed_hand_frames,
        }

    valid = confidence > 0
    xy = image_normalized_to_isotropic(xy, width, height, valid)
    detector_depth[:, :RHAND_END], depth_confidence[:, :RHAND_END] = (
        interpolate_scalar_short_gaps(
            detector_depth[:, :RHAND_END],
            depth_confidence[:, :RHAND_END],
            config.hand_gap_frames,
        )
    )
    xy[:, :RHAND_END], confidence[:, :RHAND_END] = interpolate_short_gaps(
        xy[:, :RHAND_END], confidence[:, :RHAND_END], config.hand_gap_frames
    )
    xy[:, FACE_START:], confidence[:, FACE_START:] = interpolate_short_gaps(
        xy[:, FACE_START:], confidence[:, FACE_START:], config.auxiliary_gap_frames
    )
    normalized_xy, depth, normalization = body_relative_normalize(xy, confidence)
    direct_depth = depth_confidence > 0
    depth[direct_depth] = detector_depth[direct_depth]
    features = np.zeros((count, NUM_NODES, NUM_CHANNELS), np.float32)
    features[..., :2] = normalized_xy
    features[..., 2] = depth
    features[..., 3] = confidence > 0
    features[..., 4] = np.clip(confidence, 0, 1)
    features = resample_features(features, 32).astype(np.float32)
    return features, {
        **normalization,
        "source_frames": count,
        "observed_hand_frames": observed_hand_frames,
        "hand_presence_fraction": float(features[:, :RHAND_END, 3].mean()),
        "face_presence_fraction": float(features[:, FACE_START:FACE_END, 3].mean()),
        "body_presence_fraction": float(features[:, BODY_START:BODY_END, 3].mean()),
        "detector_world_depth_fraction": float(direct_depth[:, :RHAND_END].mean()),
    }


class LiveStage2CTC:
    def __init__(self, args: argparse.Namespace):
        import coremltools as ct

        self.args = args
        self.mode = "stage2-ctc"
        self.ct = ct
        compute = ct.ComputeUnit.ALL
        self.image_encoder = ct.models.MLModel(str(args.image_encoder), compute_units=compute)
        self.encoder = ct.models.MLModel(str(args.stage2_encoder), compute_units=compute)
        self.primary = ct.models.MLModel(str(args.stage2_primary), compute_units=compute)
        self.specialist = ct.models.MLModel(
            str(args.stage2_specialist), compute_units=compute
        )
        manifest = json.loads(args.vocabulary.read_text(encoding="utf-8"))
        self.labels = [
            row["canonical_label"]
            for row in sorted(manifest["classes"], key=lambda row: row["class_index"])
        ]
        report = json.loads(args.selector_report.read_text(encoding="utf-8"))
        self.selector = report["selector_config"]
        self.encoder_output = _model_output_name(self.encoder)
        self.primary_output = _model_output_name(self.primary)
        self.specialist_output = _model_output_name(self.specialist)
        self.other_preservation = None
        preservation_path = getattr(args, 'stage2_other_preservation', None)
        if preservation_path is not None:
            from active.v17.model_stage2_v17 import load_stage2_other_preserving
            self.other_preservation, payload = load_stage2_other_preserving(preservation_path)
            validate_preservation_runtime(args, payload, self.selector)
        self.live_adapted = None
        adapted_path = getattr(args, "stage2_live_checkpoint", None)
        if adapted_path is not None:
            from active.v17.model_stage2_live_adapt_v17 import load_stage2_live_adapted
            self.live_adapted, adapted_payload = load_stage2_live_adapted(adapted_path)
            validate_live_adapted_runtime(args, adapted_payload)
            self.live_adapted.eval()
        self._warm()

    def _warm(self) -> None:
        self.image_encoder.predict({"image": Image.fromarray(np.zeros((256, 256, 3), np.uint8))})
        landmarks = np.zeros((1, MAXIMUM_WINDOWS, 32, 61, 5), np.float32)
        embeddings = np.zeros((1, MAXIMUM_WINDOWS, 16, 3, 512), np.float32)
        valid = np.zeros((1, MAXIMUM_WINDOWS, 16, 3), np.float32)
        boxes = np.zeros((1, MAXIMUM_WINDOWS, 16, 3, 4), np.float32)
        mask = np.zeros((1, MAXIMUM_WINDOWS), np.float32)
        mask[0, 0] = 1
        frozen = np.asarray(self.encoder.predict({
            "landmarks": landmarks,
            "hand_embeddings": embeddings,
            "hand_valid": valid,
            "hand_boxes": boxes,
            "window_mask": mask,
        })[self.encoder_output]).reshape(1, MAXIMUM_WINDOWS, 32, FROZEN_FEATURE_DIM)
        provider = {"frozen_features": frozen, "window_mask": mask}
        self.primary.predict(provider)
        self.specialist.predict(provider)

    def provenance(self) -> dict[str, object]:
        paths = {
            "general_selector": self.args.stage2_selector,
            "selector_report": self.args.selector_report,
            "frozen_encoder_coreml": self.args.stage2_encoder,
            "primary_coreml": self.args.stage2_primary,
            "specialist_coreml": self.args.stage2_specialist,
            "image_encoder": self.args.image_encoder,
        }
        if getattr(self.args, 'stage2_live_checkpoint', None) is not None:
            paths['live_adapted'] = self.args.stage2_live_checkpoint
        if getattr(self.args, 'stage2_other_preservation', None) is not None:
            paths['other_preservation'] = self.args.stage2_other_preservation
        return {
            name: {
                "path": str(path),
                "sha256": sha256(path) if path.is_file() else "validated_mlpackage",
            }
            for name, path in paths.items()
        }

    def _encode_hands(
        self, crops: list[list[np.ndarray | None]], valid: np.ndarray
    ) -> np.ndarray:
        embeddings = np.zeros((16, 3, 512), np.float32)
        for frame in range(16):
            for view in range(3):
                crop = crops[frame][view]
                if crop is None or not valid[frame, view]:
                    continue
                image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                output = self.image_encoder.predict({"image": image})
                embeddings[frame, view] = np.asarray(output["embedding"]).reshape(512)
        return embeddings

    def decode_frozen_logits(
        self, windows: list[np.ndarray],
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Return uncollapsed CTC emissions for up to eight retained visual windows."""
        if not 1 <= len(windows) <= MAXIMUM_WINDOWS:
            raise ValueError(f"expected 1..{MAXIMUM_WINDOWS} frozen windows")
        features = np.zeros((1, MAXIMUM_WINDOWS, 32, FROZEN_FEATURE_DIM), np.float32)
        mask = np.zeros((1, MAXIMUM_WINDOWS), np.float32)
        features[0, :len(windows)] = np.asarray(windows, dtype=np.float32)
        mask[0, :len(windows)] = 1
        provider = {"frozen_features": features, "window_mask": mask}
        primary = np.asarray(self.primary.predict(provider)[self.primary_output])
        specialist = np.asarray(self.specialist.predict(provider)[self.specialist_output])
        length = len(windows) * TOKENS_PER_WINDOW
        chosen, selected, base_tokens, specialist_tokens = select_general_ctc_logits(
            primary, specialist, length,
            blend_weight=float(self.selector["blend_weight"]),
            blank_bias=float(self.selector["blank_bias"]),
            score_margin=float(self.selector["score_margin"]),
            minimum_tokens=int(self.selector["minimum_tokens"]),
        )
        if self.live_adapted is not None:
            import torch
            with torch.inference_mode():
                chosen, _ = self.live_adapted(
                    torch.from_numpy(features), torch.from_numpy(mask > 0.5),
                )
                chosen = chosen.numpy()
        elif self.other_preservation is not None:
            import torch
            from active.v17.model_stage2_v17 import preserve_ctc_emission_runs
            with torch.inference_mode():
                odds, lengths = self.other_preservation.other_log_odds(
                    torch.from_numpy(features), torch.from_numpy(mask > 0.5),
                )
                chosen = preserve_ctc_emission_runs(
                    torch.from_numpy(chosen.reshape(1, -1, 101).copy()), odds,
                    lengths, self.other_preservation.margin,
                ).numpy()
        return np.asarray(chosen).reshape(-1, chosen.shape[-1])[:length], {
            "primary_tokens": list(base_tokens),
            "specialist_tokens": list(specialist_tokens),
            "specialist_selected": bool(selected),
            "window_count": len(windows),
        }

    def decode_frozen_windows(self, windows: list[np.ndarray]) -> dict[str, object]:
        """Decode up to the model's real eight-window context from retained vision features."""
        chosen, metadata = self.decode_frozen_logits(windows)
        tokens, positions = collapse_ctc_path(chosen, len(chosen))
        other_rejections = tokens.count(101)
        tokens, positions = supported_ctc_path(tokens, positions)
        return {
            "hypothesis": [self.labels[token - 1] for token in tokens],
            "ctc_tokens": list(tokens),
            "token_positions": list(positions),
            "other_rejections": other_rejections,
            **metadata,
        }

    def classify_window(
        self, observations: list[ObservedFrame], prior: list[np.ndarray]
    ) -> tuple[np.ndarray | None, dict[str, object]]:
        started = time.perf_counter()
        landmarks, diagnostics = stage2_landmarks_from_observations(observations)
        landmark_done = time.perf_counter()
        if landmarks is None:
            return None, {
                "accepted": False,
                "hypothesis": [],
                "rejection_reasons": [str(diagnostics["rejection_reason"])],
                "frames": len(observations),
                "start_seconds": observations[0].seconds,
                "end_seconds": observations[-1].seconds,
                "diagnostics": diagnostics,
                "latency_ms": {"total": 1000 * (landmark_done - started)},
            }
        crops, valid, boxes = hand_inputs(observations, 0, len(observations))
        embeddings = self._encode_hands(crops, valid)
        hands_done = time.perf_counter()
        model_landmarks = np.zeros((1, MAXIMUM_WINDOWS, 32, 61, 5), np.float32)
        model_embeddings = np.zeros((1, MAXIMUM_WINDOWS, 16, 3, 512), np.float32)
        model_valid = np.zeros((1, MAXIMUM_WINDOWS, 16, 3), np.float32)
        model_boxes = np.zeros((1, MAXIMUM_WINDOWS, 16, 3, 4), np.float32)
        one_mask = np.zeros((1, MAXIMUM_WINDOWS), np.float32)
        model_landmarks[0, 0] = landmarks
        model_embeddings[0, 0] = embeddings
        model_valid[0, 0] = valid
        model_boxes[0, 0] = boxes
        one_mask[0, 0] = 1
        frozen = np.asarray(self.encoder.predict({
            "landmarks": model_landmarks,
            "hand_embeddings": model_embeddings,
            "hand_valid": model_valid,
            "hand_boxes": model_boxes,
            "window_mask": one_mask,
        })[self.encoder_output]).reshape(1, MAXIMUM_WINDOWS, 32, FROZEN_FEATURE_DIM)[0, 0]
        encoded = time.perf_counter()

        decoded = self.decode_frozen_windows([*prior, frozen])
        inferred = time.perf_counter()
        return frozen, {
            "accepted": True,
            **decoded,
            "frames": len(observations),
            "start_seconds": observations[0].seconds,
            "end_seconds": observations[-1].seconds,
            "diagnostics": diagnostics,
            "latency_ms": {
                "landmark_tensor": 1000 * (landmark_done - started),
                "hand_image_encoding": 1000 * (hands_done - landmark_done),
                "frozen_encoder": 1000 * (encoded - hands_done),
                "ctc_selector": 1000 * (inferred - encoded),
                "total": 1000 * (inferred - started),
            },
        }


def draw_stage2_hud(
    frame: np.ndarray,
    latest: ObservedFrame | None,
    latest_result: dict[str, object] | None,
    hypothesis: list[str],
    windows: int,
    buffered_frames: int,
    locked_glosses: int,
    pending: bool,
    fps: float,
    finishing_glosses: list[str],
    sentence: str,
    finish_pending: bool,
    speech_text: str | None,
) -> np.ndarray:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (min(width, 700), 205), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    quality = "hand --  face --  motion --"
    if latest is not None:
        quality = (
            f"hand {latest.hand_quality:.2f}  face {latest.face_quality:.2f}  "
            f"motion {latest.motion:.3f}"
        )
    lines = [
        f"{'ENCODING' if pending else 'LIVE CTC'}   {fps:.1f} FPS   STAGE 2",
        quality,
        f"rolling context {windows}/{MAXIMUM_WINDOWS}   next {buffered_frames} samples   locked {locked_glosses}",
        f"Time-normalized {WINDOW_SECONDS:.2f}s windows; FINISH closes the utterance",
    ]
    if latest_result is not None:
        lines.append(
            f"selector {'specialist' if latest_result.get('specialist_selected') else 'primary'}   "
            f"{float(latest_result['latency_ms']['total']):.0f} ms"
        )
    for index, text in enumerate(lines):
        cv2.putText(frame, text, (14, 28 + 30 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 1, cv2.LINE_AA)

    panel_top = max(224, height - 102)
    cv2.rectangle(frame, (0, panel_top), (width, height), (20, 20, 20), -1)
    shown_glosses = finishing_glosses if finish_pending and finishing_glosses else hypothesis
    shown = "  ".join(shown_glosses) if shown_glosses else "(empty)"
    max_chars = max(18, (width - 36) // 11)
    if len(shown) > max_chars:
        shown = "..." + shown[-(max_chars - 3):]
    cv2.putText(frame, f"CTC GLOSS BUFFER: {shown}", (14, panel_top + 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 1, cv2.LINE_AA)
    status = f"SPEAKING: {speech_text}" if speech_text else sentence
    if finish_pending:
        status = "Finishing CTC tail and naturalizing..."
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
    classifier = LiveStage2CTC(args)
    naturalizer = make_naturalizer(args)
    speaker = None if args.no_speech else LiveSpeaker()
    source = str(args.video) if args.video else f"camera:{args.camera}"
    recorder = SessionRecorder(args, classifier, source)
    recorder.data.update({
        "format": "slt_live_stage2_ctc_v17_session",
        "version": 1,
        "mode": classifier.mode,
        "prototype_scope": (
            "accepted v17 general CTC selector over non-overlapping continuous windows"
        ),
        "score_semantics": "greedy CTC sequence; no calibrated confidence claim",
    })
    recorder.data["config"].update({
        "window_source_frames": WINDOW_SOURCE_FRAMES,
        "window_seconds": args.window_seconds,
        "maximum_windows": MAXIMUM_WINDOWS,
        "tokens_per_window": TOKENS_PER_WINDOW,
        "live_face_display": "MediaPipe overlay every processed frame",
        "model_face_sampling": "every eighth processed frame, matching v17 training",
        "capture_policy": "latest-frame webcam; sequential saved video",
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
    lip_tracker = DisplayLipTracker(
        enabled=not args.no_display and not args.no_live_lips
    )
    previous_wrists = {"left": None, "right": None}
    window_buffer = ElapsedWindowBuffer(args.window_seconds)
    ready_windows: deque[list[ObservedFrame]] = deque()
    frozen_features: list[np.ndarray] = []
    locked_glosses: list[str] = []
    context_hypothesis: list[str] = []
    context_positions: list[int] = []
    hypothesis: list[str] = []
    finishing_glosses: list[str] = []
    results: list[dict[str, object]] = []
    stable_speaker = StablePrefixSpeaker()
    classifier_executor = ThreadPoolExecutor(max_workers=1)
    language_executor = ThreadPoolExecutor(max_workers=1)
    warm_future = language_executor.submit(naturalizer.warm)
    future: Future | None = None
    future_epoch = 0
    epoch = 0
    naturalizer_future: Future | None = None
    naturalizer_context: dict[str, object] | None = None
    pending_speech: deque[dict[str, object]] = deque()
    latest: ObservedFrame | None = None
    latest_result: dict[str, object] | None = None
    sentence = ""
    finish_requested = False
    warm_logged = False
    display_times = deque(maxlen=30)
    ui_actions: deque[tuple[str, float]] = deque()
    display_size = [1280, 720]
    frame_index = processed = 0
    next_process = 0.0
    wall_started = time.perf_counter()
    camera = None if args.video else LatestCamera(capture, wall_started)
    camera_sequence = 0
    dropped_camera_frames = 0
    latest_lips: np.ndarray | None = None

    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME)

        def on_mouse(event, x, y, _flags, _parameter):
            if event != cv2.EVENT_LBUTTONUP:
                return
            action = clicked_control(x, y, display_size[0], display_size[1])
            if action is not None:
                ui_actions.append((action, time.perf_counter() - wall_started))

        cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    def submit(observations: list[ObservedFrame]) -> None:
        nonlocal future, future_epoch, locked_glosses
        nonlocal context_hypothesis, context_positions, hypothesis
        if future is not None or len(observations) < 4:
            return
        if len(frozen_features) >= MAXIMUM_WINDOWS:
            locked_glosses, context_hypothesis, context_positions = roll_ctc_prefix(
                locked_glosses, context_hypothesis, context_positions
            )
            frozen_features.pop(0)
            hypothesis = [*locked_glosses, *context_hypothesis]
        future_epoch = epoch
        future = classifier_executor.submit(
            classifier.classify_window, list(observations), list(frozen_features)
        )

    def collect(wait: bool = False) -> None:
        nonlocal future, latest_result, hypothesis
        nonlocal context_hypothesis, context_positions
        if future is None or (not wait and not future.done()):
            return
        feature, result = future.result()
        result["stream_epoch"] = future_epoch
        result["ignored_after_reset"] = future_epoch != epoch
        if future_epoch == epoch and feature is not None:
            frozen_features.append(feature)
            context_hypothesis = list(result["hypothesis"])
            context_positions = [int(value) for value in result["token_positions"]]
            hypothesis = [*locked_glosses, *context_hypothesis]
            result["locked_prefix"] = list(locked_glosses)
            result["full_hypothesis"] = list(hypothesis)
            latest_result = result
            for gloss in stable_speaker.update(hypothesis):
                pending_speech.append({
                    "text": gloss,
                    "kind": "stable_ctc_gloss",
                    "reference": len(results),
                })
        recorder.add(result)
        results.append(result)
        print(json.dumps(result, indent=2))
        future = None

    def maybe_submit_ready_window() -> None:
        if future is None and ready_windows and not finish_requested:
            submit(ready_windows.popleft())

    def start_naturalizer() -> None:
        nonlocal finish_requested, naturalizer_future, naturalizer_context, sentence
        nonlocal locked_glosses, context_hypothesis, context_positions
        glosses = list(hypothesis)
        finishing_glosses[:] = glosses
        finish_requested = False
        window_buffer.reset()
        ready_windows.clear()
        frozen_features.clear()
        locked_glosses.clear()
        context_hypothesis.clear()
        context_positions.clear()
        hypothesis.clear()
        stable_speaker.reset()
        if not glosses:
            sentence = "No recognized signs to finish."
            recorder.add_event({"type": "finish_empty"})
            return
        naturalizer_context = {
            "utterance_id": f"ctc-{len(recorder.data['utterances']) + 1:04d}",
            "requested_utc": utc_now(),
            "glosses": glosses,
            "stream_epoch": epoch,
        }
        recorder.add_event({"type": "finish_submitted", **naturalizer_context})
        naturalizer_future = language_executor.submit(naturalizer.rephrase, glosses)

    def service_finish() -> None:
        if not finish_requested or future is not None or naturalizer_future is not None:
            return
        if ready_windows:
            submit(ready_windows.popleft())
            return
        start_naturalizer()

    def service_language() -> None:
        nonlocal warm_logged, naturalizer_future, naturalizer_context, sentence
        if not warm_logged and warm_future.done():
            warm_logged = True
            recorder.add_event({"type": "naturalizer_warmup", **warm_future.result()})
        if naturalizer_future is not None and naturalizer_future.done():
            result = naturalizer_future.result()
            utterance = {
                **(naturalizer_context or {}), **result, "completed_utc": utc_now()
            }
            visible = utterance.get("stream_epoch") == epoch
            utterance["displayed_and_spoken"] = visible
            recorder.add_utterance(utterance)
            if visible:
                sentence = str(result["sentence"])
                pending_speech.clear()
                if speaker is not None:
                    speaker.clear()
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
        nonlocal epoch, sentence, finish_requested
        cleared = list(hypothesis)
        epoch += 1
        window_buffer.reset()
        ready_windows.clear()
        frozen_features.clear()
        locked_glosses.clear()
        context_hypothesis.clear()
        context_positions.clear()
        hypothesis.clear()
        finishing_glosses.clear()
        stable_speaker.reset()
        sentence = ""
        finish_requested = False
        pending_speech.clear()
        if speaker is not None:
            speaker.clear()
        recorder.add_event({
            "type": "reset",
            "source": source_name,
            "seconds": seconds,
            "cleared_glosses": cleared,
            "classifier_pending": future is not None,
        })

    def request_finish(seconds: float, source_name: str) -> None:
        nonlocal finish_requested
        if finish_requested:
            return
        finish_requested = True
        tail = window_buffer.finish()
        if tail is not None:
            ready_windows.append(tail)
        recorder.add_event({
            "type": "finish_requested",
            "source": source_name,
            "seconds": seconds,
            "active_glosses": list(hypothesis),
            "buffered_frames": sum(len(chunk) for chunk in ready_windows),
        })
        service_finish()

    try:
        while True:
            if camera is None:
                ok, raw = capture.read()
                if not ok:
                    break
                seconds = frame_index / source_fps
                frame_index += 1
            else:
                packet = camera.after(camera_sequence)
                if packet is None:
                    if camera.failed:
                        break
                    time.sleep(0.001)
                    continue
                sequence, seconds, raw = packet
                dropped_camera_frames += max(0, sequence - camera_sequence - 1)
                camera_sequence = sequence
                frame_index += 1
            canonical = orient_frame(raw, args.rotation, args.input_mirrored)
            recorder.write_frame(canonical, seconds)
            collect()
            maybe_submit_ready_window()
            service_finish()
            service_language()
            if (
                not finish_requested
                and seconds + 1e-6 >= next_process
            ):
                detection_frame = limit_image_side(canonical, args.detection_image_side)
                frame = limit_image_side(canonical, args.maximum_image_side)
                feature_frame = processed % V17Config().face_interval == 0
                latest_lips = lip_tracker.detect(detection_frame)
                detection = detector.detect(
                    detection_frame,
                    include_body=processed % V17Config().body_interval == 0,
                    include_face=feature_frame,
                    include_hands=True,
                )
                assigned = assign_hands(detection.hands, previous_wrists)
                latest = ObservedFrame(
                    frame,
                    detection,
                    assigned,
                    seconds,
                    wrist_motion(assigned, previous_wrists),
                    *observation_quality(detection),
                    face_for_features=feature_frame,
                )
                chunk = window_buffer.add(latest)
                if chunk is not None:
                    ready_windows.append(chunk)
                processed += 1
                next_process = max(next_process + 1.0 / args.processing_fps, seconds)
                maybe_submit_ready_window()

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
                shown = draw_display_lips(
                    shown, latest_lips, mirror=not args.no_mirror_display
                )
                display_size[:] = [shown.shape[1], shown.shape[0]]
                shown = draw_stage2_hud(
                    shown,
                    latest,
                    latest_result,
                    hypothesis,
                    len(frozen_features),
                    len(window_buffer.frames),
                    len(locked_glosses),
                    future is not None,
                    fps,
                    finishing_glosses,
                    sentence,
                    finish_requested or naturalizer_future is not None,
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
                        request_finish(action_seconds, "button")
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    reset_display(seconds, "keyboard")
                if key == ord("f"):
                    request_finish(seconds, "keyboard")
            else:
                service_speech()

        tail = window_buffer.finish()
        if tail is not None:
            ready_windows.append(tail)
        while future is not None or ready_windows:
            if future is not None:
                collect(wait=True)
            elif ready_windows:
                submit(ready_windows.popleft())
        if args.finish_at_eof and hypothesis:
            finish_requested = True
            recorder.add_event({
                "type": "finish_requested",
                "source": "video_eof",
                "seconds": (
                    frame_index / source_fps if args.video
                    else time.perf_counter() - wall_started
                ),
                "active_glosses": list(hypothesis),
                "buffered_frames": 0,
            })
        service_finish()
        if future is not None:
            collect(wait=True)
            service_finish()
        if naturalizer_future is not None:
            naturalizer_future.result()
            service_language()
        service_speech()
    finally:
        if camera is None:
            capture.release()
        else:
            camera.close()
        lip_tracker.close()
        classifier_executor.shutdown(wait=True)
        language_executor.shutdown(wait=True)
        recorder.data["capture_stats"] = {
            "frames_processed": frame_index,
            "landmark_observations": processed,
            "stale_camera_frames_dropped": dropped_camera_frames,
        }
        recorder.close()
        cv2.destroyAllWindows()

    expected = [value.upper() for value in args.expected_sequence]
    final_hypothesis = list(finishing_glosses if args.finish_at_eof else hypothesis)
    summary = {
        "session": str(recorder.root),
        "history": str(recorder.history_path),
        "video": str(recorder.video_path),
        "windows": max(
            (int(row.get("window_count", 0)) for row in results), default=0
        ),
        "predictions": len(results),
        "locked_glosses": len(locked_glosses),
        "stale_camera_frames_dropped": dropped_camera_frames,
        "hypothesis": final_hypothesis,
        "expected": expected,
        "exact": final_hypothesis == expected if expected else None,
        "utterances": len(recorder.data["utterances"]),
    }
    print(json.dumps(summary, indent=2))
    return summary


def parser() -> argparse.ArgumentParser:
    value = isolated_parser()
    value.description = __doc__
    value.set_defaults(
        mode="fast",
        output_root=REPO / "artifacts/reports/live_stage2_ctc_v17",
    )
    value.add_argument("--stage2-encoder", type=Path, default=DEFAULT_ENCODER)
    value.add_argument("--stage2-primary", type=Path, default=DEFAULT_PRIMARY)
    value.add_argument("--stage2-specialist", type=Path, default=DEFAULT_SPECIALIST)
    value.add_argument("--stage2-selector", type=Path, default=DEFAULT_SELECTOR)
    value.add_argument("--stage2-other-preservation", type=Path)
    value.add_argument("--selector-report", type=Path, default=DEFAULT_SELECTOR_REPORT)
    value.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    value.add_argument(
        "--window-seconds", type=float, default=WINDOW_SECONDS,
        help="elapsed-time span resampled into each trained 32-frame CTC window",
    )
    value.add_argument(
        "--no-live-lips", action="store_true",
        help="disable the display-only MediaPipe lip overlay",
    )
    value.add_argument("--expected-sequence", nargs="*", default=())
    value.add_argument(
        "--finish-at-eof", action="store_true",
        help="exercise the FINISH/naturalizer path after a saved-video replay",
    )
    return value


def main() -> None:
    args = parser().parse_args()
    if (
        args.processing_fps <= 0
        or args.record_fps <= 0
        or args.ollama_timeout <= 0
        or args.window_seconds <= 0
    ):
        raise ValueError("FPS and timeout values must be positive")
    if args.detection_image_side < 320 or args.maximum_image_side < args.detection_image_side:
        raise ValueError("image sizes are too small")
    run(args)


if __name__ == "__main__":
    main()
