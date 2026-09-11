#!/usr/bin/env python3
"""Reel-style live Stage-1 recognition without a required neutral pose.

This is a separate experiment.  It grows one candidate from hand activity, displays
the provisional Stage-1 result immediately, and commits only after the same completed
clip prediction survives additional frames.  RESET clears the visible buffer but not
the JSON log; FINISH naturalizes and speaks the committed glosses.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
import sys
import tempfile
import time

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from active.v17.extract_v17 import (
    AppleVisionDetector,
    FrameDetection,
    HandDetection,
    assign_hands,
    limit_image_side,
    orient_frame,
)
from active.v17.lip_marker_v17 import (
    LipMarkerDisambiguator,
    LipMarkerTracker,
    draw_lip_markers,
)
from active.v17.schema_v17 import V17Config
from scripts.live_isolated_v17 import (
    BODY_CONNECTIONS,
    HAND_CONNECTIONS,
    IsolatedClassifier,
    LiveSpeaker,
    ObservedFrame,
    SessionRecorder,
    atomic_json,
    make_naturalizer,
    landmarks_from_observations,
    observation_quality,
    parser as isolated_parser,
    sha256,
    trim_to_motion,
    utc_now,
    wrist_motion,
)
from scripts.reel_hud_v17 import ReelHud, clicked_reel_control
from scripts.live_stage2_ctc_v17 import (
    DEFAULT_ENCODER,
    DEFAULT_PRIMARY,
    DEFAULT_SELECTOR,
    DEFAULT_SELECTOR_REPORT,
    DEFAULT_SPECIALIST,
    DEFAULT_VOCABULARY,
    ElapsedWindowBuffer,
    LatestCamera,
    LiveStage2CTC,
    MAXIMUM_WINDOWS,
    collapse_ctc_path,
    roll_ctc_prefix,
    observe_stage2_frame,
    supported_ctc_path,
)
from active.v17.streaming_ctc_prefix_v17 import StreamingCTCPrefix
from active.v17.stage1_window_v17 import (
    NO_EMIT,
    STRIDE_SECONDS as STAGE1_WINDOW_STRIDE_SECONDS,
    WINDOW_SECONDS as STAGE1_WINDOW_SECONDS,
    Stage1WindowTranscript,
    load_stage1_window_checkpoint,
    window_end_times,
)
from active.v17.stage1_window_v17 import raw_observation_features, normalize_time_window


WINDOW_NAME = "SLT v17 reel-style Stage 1"
DEFAULT_PHRASE_ADAPTED_COREML = (
    REPO / "artifacts/coreml/Stage1PhraseAdaptReelV17.mlpackage"
)
DEFAULT_UNIFIED_PHRASE_ADAPTED = (
    REPO / (
        "artifacts/models/stage1_v17_unified_phrase_activity_adapt_reel_v2/"
        "best_model.pth"
    )
)
DEFAULT_UNIFIED_PHRASE_ADAPTED_COREML = (
    REPO / "artifacts/coreml/Stage1UnifiedPhraseActivityAdaptReelV17FP16.mlpackage"
)
DEFAULT_LIP_MARKER_MODEL = (
    REPO / "artifacts/models/lip_marker_good_thankyou_phrase_crops_v17/model.npz"
)


class Stage1WindowClassifier:
    """Landmark-only classifier for the separately trained window candidate."""

    def __init__(self, args: argparse.Namespace):
        import torch

        self.path = args.stage1_window_checkpoint
        self.device = torch.device("cpu")
        torch.set_num_threads(2)
        self.model, self.labels, self.checkpoint = load_stage1_window_checkpoint(
            self.path, self.device
        )
        with torch.inference_mode():
            self.model(torch.zeros((1, 32, 61, 5), device=self.device))

    def provenance(self) -> dict[str, object]:
        return {
            "stage1_window_checkpoint": {
                "path": str(self.path), "sha256": sha256(self.path),
                "format": self.checkpoint["format"],
            },
            "stage1_window": self.checkpoint["stage1_window"],
            "landmark_only": True,
        }

    def classify_window(
        self, observations: list[ObservedFrame], end_seconds: float,
    ) -> dict[str, object]:
        import torch

        started = time.perf_counter()
        raw, timestamps = raw_observation_features(observations)
        return self.classify_raw_window(raw, timestamps, end_seconds)

    def classify_raw_window(self, raw, timestamps, end_seconds):
        import torch
        started = time.perf_counter()
        features, diagnostics = normalize_time_window(raw, timestamps, end_seconds)
        with torch.inference_mode():
            logits = self.model(
                torch.from_numpy(features[None]).to(self.device)
            )[0].float().cpu().numpy()
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        order = np.argsort(probabilities)[::-1][:3]
        label = self.labels[int(order[0])]
        return {
            "gloss": label,
            "candidate_gloss": label,
            "accepted": label != NO_EMIT,
            "model_score": float(probabilities[order[0]]),
            "top3": [
                {"gloss": self.labels[int(index)],
                 "model_score": float(probabilities[index])}
                for index in order
            ],
            "start_seconds": max(float(timestamps[0]), end_seconds - STAGE1_WINDOW_SECONDS),
            "end_seconds": float(end_seconds),
            "diagnostics": diagnostics,
            "latency_ms": {"total": 1000.0 * (time.perf_counter() - started)},
        }


def final_stage1_window_decode(
    classifier, observations: list[tuple[float, np.ndarray]],
) -> tuple[list[str], list[dict[str, object]]]:
    if len(observations) < 2:
        return [], []
    timestamps = np.asarray([item[0] for item in observations])
    raw = np.stack([item[1] for item in observations])
    transcript = Stage1WindowTranscript()
    results = []
    for end in window_end_times(timestamps, include_final=True):
        if transcript.predictions and end - transcript.predictions[-1].end_seconds > STAGE1_WINDOW_STRIDE_SECONDS + 1e-6:
            transcript.update(NO_EMIT, transcript.predictions[-1].end_seconds + .001)
        try:
            result = classifier.classify_raw_window(raw, timestamps, end)
        except ValueError as error:
            results.append(dict(gloss=NO_EMIT, end_seconds=end, accepted=False,
                                rejection_reasons=[str(error)]))
            transcript.update(NO_EMIT, end)
            continue
        results.append(result)
        transcript.update(str(result["gloss"]), end)
    return list(transcript.words), results


def load_frame_timestamps(path: Path | None) -> list[float] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    values = payload.get("timestamps") if isinstance(payload, dict) else payload
    timestamps = np.asarray(values, dtype=np.float64)
    if (
        timestamps.ndim != 1 or not len(timestamps)
        or not np.isfinite(timestamps).all()
        or (timestamps < 0).any()
        or (len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0))
    ):
        raise ValueError("frame timestamps must be a finite, strictly increasing JSON list")
    return timestamps.tolist()


class ReelCascadeClassifier:
    """Fast landmark proposal with a visual-hand verifier only when required."""

    def __init__(
        self, args: argparse.Namespace, verifier_class=IsolatedClassifier,
    ):
        import coremltools as ct

        self.args = args
        self.full = None
        if getattr(args, "landmark_only_commit", False):
            import torch

            checkpoint = torch.load(
                args.unified_checkpoint, map_location="cpu", weights_only=False
            )
            self.labels = [
                label for label, _ in sorted(
                    checkpoint["label_to_index"].items(), key=lambda row: row[1]
                )
            ]
        else:
            fast_args = copy.copy(args)
            fast_args.mode = "fast"
            self.full = verifier_class(fast_args)
            self.labels = self.full.labels
        self.orientation = ct.models.MLModel(
            str(args.orientation_coreml), compute_units=ct.ComputeUnit.ALL
        )
        self.orientation_output_name = (
            self.orientation.get_spec().description.output[0].name
        )
        self.orientation.predict({
            "landmarks": np.zeros((1, 32, 61, 5), np.float32),
        })

    def provenance(self) -> dict[str, object]:
        base = (
            self.full.provenance() if self.full is not None else {
                "unified_checkpoint": {
                    "path": str(self.args.unified_checkpoint),
                    "sha256": sha256(self.args.unified_checkpoint),
                },
            }
        )
        return {
            **base,
            "orientation_coreml": {
                "path": str(self.args.orientation_coreml), "sha256": "mlpackage",
            },
            "reel_no_emit": {
                "enabled": getattr(
                    self.args, "no_emit_probability_threshold", None
                ) is not None,
                "probability_threshold": getattr(
                    self.args, "no_emit_probability_threshold", None
                ),
            },
        }

    def classify(self, observations: list[ObservedFrame]) -> dict[str, object]:
        """Return a landmark-only proposal; visual verification is scheduled later."""
        started = time.perf_counter()
        original = observations
        trimmed, motion = trim_to_motion(observations, self.args.quiet_motion)
        try:
            features, diagnostics = landmarks_from_observations(trimmed, V17Config())
        except ValueError as error:
            elapsed = 1000.0 * (time.perf_counter() - started)
            return {
                "gloss": "UNKNOWN",
                "candidate_gloss": None,
                "model_score": 0.0,
                "margin": 0.0,
                "gate_score": 0.0,
                "gate_margin": 0.0,
                "top3": [],
                "accepted": False,
                "rejection_reasons": [str(error)],
                "mode": "reel-cascade-landmark",
                "frames": len(original),
                "motion_trimmed_frames": len(trimmed),
                "start_seconds": original[0].seconds,
                "end_seconds": original[-1].seconds,
                "diagnostics": motion,
                "latency_ms": {
                    "tensor_extraction": elapsed,
                    "hand_image_encoding": 0.0,
                    "classification": 0.0,
                    "total": elapsed,
                },
            }
        raw_all = np.asarray(
            self.orientation.predict({"landmarks": features[None]})[
                getattr(self, "orientation_output_name", "var_5535")
            ]
        ).reshape(-1)
        if len(raw_all) not in {len(self.labels), len(self.labels) + 1}:
            raise ValueError(
                f"expected {len(self.labels)} or {len(self.labels) + 1} Reel logits, "
                f"got {len(raw_all)}"
            )
        raw = raw_all[: len(self.labels)]
        probability = np.exp(raw - raw.max())
        probability /= probability.sum()
        no_emit_probability = 0.0
        no_emit_threshold = getattr(
            self.args, "no_emit_probability_threshold", None
        )
        if len(raw_all) == len(self.labels) + 1:
            all_probability = np.exp(raw_all - raw_all.max())
            all_probability /= all_probability.sum()
            no_emit_probability = float(all_probability[-1])
        order = np.argsort(probability)[::-1][:3]
        score = float(probability[order[0]])
        top3 = [
            {"gloss": self.labels[int(index)], "model_score": float(probability[index])}
            for index in order
        ]
        margin = top3[0]["model_score"] - top3[1]["model_score"]
        duration = trimmed[-1].seconds - trimmed[0].seconds
        rejection = []
        if score < self.args.minimum_score:
            rejection.append("low_score")
        if margin < self.args.minimum_margin:
            rejection.append("low_margin")
        if duration > self.args.maximum_accept_seconds:
            rejection.append("clip_too_long")
        if (
            no_emit_threshold is not None
            and no_emit_probability >= float(no_emit_threshold)
        ):
            rejection.append("learned_no_emit")
        elapsed = 1000.0 * (time.perf_counter() - started)
        diagnostics.update(motion)
        diagnostics.update({
            "cascade_primary_score": score,
            "cascade_primary_gloss": top3[0]["gloss"],
            "cascade_score_threshold": self.args.cascade_score,
            "cascade_fallback_used": False,
            "mouth_pixels_used": False,
            "lip_markers_used_for_general_classification": False,
            "no_emit_probability": no_emit_probability,
            "no_emit_probability_threshold": no_emit_threshold,
        })
        return {
            "gloss": top3[0]["gloss"] if not rejection else "UNKNOWN",
            "candidate_gloss": top3[0]["gloss"],
            "model_score": score,
            "margin": margin,
            "gate_score": score,
            "gate_margin": margin,
            "top3": top3,
            "accepted": not rejection,
            "rejection_reasons": rejection,
            "mode": "reel-cascade-landmark",
            "frames": len(original),
            "motion_trimmed_frames": len(trimmed),
            "start_seconds": original[0].seconds,
            "end_seconds": original[-1].seconds,
            "diagnostics": diagnostics,
            "latency_ms": {
                "tensor_extraction": elapsed,
                "hand_image_encoding": 0.0,
                "classification": 0.0,
                "total": elapsed,
            },
        }

    def verify(self, observations: list[ObservedFrame]) -> dict[str, object]:
        if self.full is None:
            raise RuntimeError("the full visual verifier is disabled")
        return self.full.classify(observations)


@dataclass
class StableGlossLock:
    required_hits: int = 3
    release_hits: int = 2
    candidate: str | None = None
    hits: int = 0
    misses: int = 0
    suppressed: str | None = None
    release_candidate: str | None = None
    release_count: int = 0

    def reset(self, *, clear_suppression: bool = True) -> None:
        self.candidate = None
        self.hits = 0
        self.misses = 0
        self.release_candidate = None
        self.release_count = 0
        if clear_suppression:
            self.suppressed = None

    def no_hands(self) -> None:
        self.reset(clear_suppression=True)

    def update(self, result: dict[str, object]) -> str | None:
        accepted = bool(result.get("accepted"))
        label = str(result.get("gloss", "UNKNOWN"))
        if not accepted or label == "UNKNOWN":
            self.misses += 1
            if self.misses >= self.release_hits:
                self.candidate = None
                self.hits = 0
            return None
        self.misses = 0

        if label == self.suppressed:
            self.candidate = None
            self.hits = 0
            self.release_candidate = None
            self.release_count = 0
            return None
        if self.suppressed is not None:
            if label != self.release_candidate:
                self.release_candidate = label
                self.release_count = 1
            else:
                self.release_count += 1
            if self.release_count < self.release_hits:
                return None
            self.suppressed = None

        if label != self.candidate:
            self.candidate = label
            self.hits = 1
        else:
            self.hits += 1
        if self.hits < self.required_hits:
            return None
        proposal = label
        self.candidate = None
        self.hits = 0
        return proposal

    def verified(self, label: str) -> None:
        self.suppressed = label
        self.candidate = None
        self.hits = 0


@dataclass
class VerifiedCommitLock:
    """Require repeatable full-model evidence unless agreement is very strong."""

    required_hits: int = 2
    instant_score: float = 0.80
    candidate: str | None = None
    hits: int = 0

    def reset(self) -> None:
        self.candidate = None
        self.hits = 0

    def update(
        self, label: str, evidence: float, *, proposal: str,
        proposal_score: float, minimum_score: float,
    ) -> bool:
        if evidence < minimum_score:
            return False
        if label != self.candidate:
            self.candidate = label
            self.hits = 1
        else:
            self.hits += 1
        # This pair is buffered rather than spoken, so it can be corrected from
        # following context without delaying segmentation into the next sign.
        if label in {"GOOD", "THANKYOU"}:
            self.reset()
            return True
        immediate = label == proposal and proposal_score >= self.instant_score
        if immediate or self.hits >= self.required_hits:
            self.reset()
            return True
        return False


def resolve_good_thankyou_context(fallback: str, following: str | None) -> str:
    """Resolve the near-homophonous one-handed pair from a following gloss."""
    if following == "FRIEND":
        return "THANKYOU"
    if following == "MORNING":
        return "GOOD"
    return fallback


def collapse_adjacent_glosses(glosses: list[str]) -> list[str]:
    output: list[str] = []
    for gloss in glosses:
        if not output or output[-1] != gloss:
            output.append(gloss)
    return output


def select_finished_sequence(
    stage1: list[str], stage2: list[str], active_seconds: float,
    minimum_phrase_seconds: float, *, stage2_stable: bool = False,
) -> tuple[list[str], str]:
    """Use CTC only for multi-sign input corroborated by Stage-1 activity."""
    stage2 = collapse_adjacent_glosses(stage2)
    normal_multisign = (
        active_seconds >= minimum_phrase_seconds
        and len(stage1) >= 2
        and len(stage2) >= 2
        and abs(len(stage2) - len(stage1)) <= 2
    )
    iterator = iter(stage2)
    stage1_is_subsequence = all(gloss in iterator for gloss in stage1)
    stable_recovery = (
        stage2_stable
        and active_seconds >= minimum_phrase_seconds
        and bool(stage1)
        and len(stage2) >= 2
        and stage1_is_subsequence
    )
    if normal_multisign or stable_recovery:
        return stage2, "stage2_ctc_multisign_arbiter"
    return stage1, "stage1_isolated_or_unconfirmed"


def add_targeted_lip_evidence(
    disambiguator: LipMarkerDisambiguator | None,
    observations: list[ObservedFrame],
    result: dict[str, object],
    proposal: str,
    minimum_confidence: float,
) -> dict[str, object]:
    """Use lips only to break a GOOD/THANKYOU disagreement between two models."""
    if proposal not in {"GOOD", "THANKYOU"}:
        return result
    diagnostics = dict(result.get("diagnostics", {}))
    full_candidate = str(result.get("candidate_gloss", "UNKNOWN"))
    eligible = (
        bool(result.get("accepted"))
        and full_candidate in {"GOOD", "THANKYOU"}
        and full_candidate != proposal
    )
    prediction = None if disambiguator is None else disambiguator.predict([
        getattr(item, "lip_points", None) for item in observations
    ])
    selected = full_candidate
    source = "model_consensus" if full_candidate == proposal else "full_verifier"
    if (
        eligible
        and prediction is not None
        and float(prediction["confidence"]) >= minimum_confidence
    ):
        selected = str(prediction["label"])
        source = "media_pipe_lip_markers"
    diagnostics.update({
        "mouth_pixels_used": False,
        "targeted_lip_verifier": "closed_good_thankyou_pair",
        "targeted_lip_eligible": eligible,
        "targeted_lip_landmark_proposal": proposal,
        "targeted_lip_full_candidate": full_candidate,
        "targeted_lip_selected": selected,
        "targeted_lip_source": source,
        "targeted_lip_prediction": prediction,
        "targeted_lip_minimum_confidence": minimum_confidence,
    })
    top3 = list(result.get("top3", []))
    return {
        **result,
        "gloss": selected if result.get("accepted") else "UNKNOWN",
        "candidate_gloss": selected,
        "top3": top3,
        "diagnostics": diagnostics,
    }


def proposal_as_fast_verifier(result: dict[str, object]) -> dict[str, object]:
    """Reuse a stable learned-emission proposal without the slow image verifier."""
    verifier = copy.deepcopy(result)
    verifier["mode"] = "learned-emission-landmark-fast"
    verifier.setdefault("diagnostics", {})["full_visual_verifier_used"] = False
    return verifier


def persistent_auxiliary_detection(
    detection: FrameDetection,
    body: tuple[np.ndarray, np.ndarray] | None,
    face: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[FrameDetection, tuple[np.ndarray, np.ndarray] | None,
           tuple[np.ndarray, np.ndarray] | None]:
    """Keep the last valid face/body visible without changing model observations."""
    if (detection.body_confidence > 0).any():
        body = (detection.body_xy.copy(), detection.body_confidence.copy())
    if (detection.face_confidence > 0).any():
        face = (detection.face_xy.copy(), detection.face_confidence.copy())
    visible = FrameDetection(
        detection.hands,
        detection.body_xy if body is None else body[0],
        detection.body_confidence if body is None else body[1],
        detection.face_xy if face is None else face[0],
        detection.face_confidence if face is None else face[1],
    )
    return visible, body, face


def _straight_finger(
    xy: np.ndarray, wrist: np.ndarray, mcp: int, pip: int, tip: int,
    palm_axis: np.ndarray,
) -> bool:
    """Return whether one non-thumb finger is extended away from the palm."""
    proximal = xy[pip] - xy[mcp]
    distal = xy[tip] - xy[pip]
    scale = float(np.linalg.norm(proximal) * np.linalg.norm(distal))
    aligned = scale > 1e-6 and float(np.dot(proximal, distal) / scale) > 0.45
    farther = np.linalg.norm(xy[tip] - wrist) > 1.10 * np.linalg.norm(
        xy[pip] - wrist
    )
    beyond_knuckle = float(np.dot(xy[tip] - xy[pip], palm_axis)) > 0.005
    return bool(aligned and farther and beyond_knuckle)


def is_finish_hand(hand: HandDetection | None) -> bool:
    """Detect one upright open palm (all five fingers visibly extended)."""
    if hand is None:
        return False
    required = np.array([0, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16,
                         17, 18, 20])
    if np.any(hand.confidence[required] <= 0):
        return False
    xy = hand.xy
    wrist = xy[0]
    palm = xy[9] - wrist
    palm_length = float(np.linalg.norm(palm))
    if palm_length < 0.025:
        return False
    palm_axis = palm / palm_length
    if palm_axis[1] > -0.25 or wrist[1] > 0.80:
        return False
    fingers_open = all(
        _straight_finger(xy, wrist, mcp, pip, tip, palm_axis)
        for mcp, pip, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16),
                              (17, 18, 20))
    )
    thumb_proximal = xy[3] - xy[2]
    thumb_distal = xy[4] - xy[3]
    thumb_scale = float(
        np.linalg.norm(thumb_proximal) * np.linalg.norm(thumb_distal)
    )
    thumb_open = (
        thumb_scale > 1e-6
        and float(np.dot(thumb_proximal, thumb_distal) / thumb_scale) > 0.15
        and np.linalg.norm(xy[4] - wrist) > 1.05 * np.linalg.norm(xy[3] - wrist)
    )
    return bool(fingers_open and thumb_open)


def is_finish_gesture(
    assigned: dict[str, HandDetection | None],
) -> bool:
    """Detect two separated upright open palms: the ten-finger finish sign."""
    left, right = assigned["left"], assigned["right"]
    if not is_finish_hand(left) or not is_finish_hand(right):
        return False
    assert left is not None and right is not None
    return bool(np.linalg.norm(left.xy[0] - right.xy[0]) >= 0.12)


@dataclass
class FinishGesture:
    """Hold/latch policy that prevents a single noisy frame or repeated finish."""

    hold_seconds: float = 0.4
    dropout_grace_seconds: float = 0.15
    started_at: float | None = None
    last_seen_at: float | None = None
    active: bool = False
    latched: bool = False
    progress: float = 0.0

    def update(
        self, assigned: dict[str, HandDetection | None], seconds: float,
    ) -> bool:
        observed = is_finish_gesture(assigned)
        if observed:
            self.last_seen_at = seconds
            if self.started_at is None:
                self.started_at = seconds
        elif (
            self.last_seen_at is None
            or seconds - self.last_seen_at > self.dropout_grace_seconds
        ):
            self.started_at = None
            self.last_seen_at = None
            self.active = False
            self.latched = False
            self.progress = 0.0
            return False

        self.active = True
        if self.latched:
            self.progress = 1.0
            return False
        started_at = seconds if self.started_at is None else self.started_at
        elapsed = max(0.0, seconds - started_at)
        self.progress = min(1.0, elapsed / self.hold_seconds)
        if elapsed < self.hold_seconds:
            return False
        self.latched = True
        self.progress = 1.0
        return True


def draw_reel_detection(
    frame: np.ndarray, item: ObservedFrame | None, *, mirror: bool
) -> np.ndarray:
    """Draw thin white hand/body bones and white joints for the reel view."""
    output = cv2.flip(frame, 1) if mirror else frame.copy()
    if item is None:
        return output
    height, width = output.shape[:2]

    def pixel(point: np.ndarray) -> tuple[int, int]:
        x = int(round(point[0] * width))
        if mirror:
            x = width - 1 - x
        return x, int(round(point[1] * height))

    def skeleton(xy: np.ndarray, confidence: np.ndarray, connections) -> None:
        for start, end in connections:
            if confidence[start] > 0 and confidence[end] > 0:
                cv2.line(
                    output, pixel(xy[start]), pixel(xy[end]),
                    (255, 255, 255), 1, cv2.LINE_AA,
                )
        for point, score in zip(xy, confidence):
            if score > 0:
                cv2.circle(output, pixel(point), 2, (255, 255, 255), -1,
                           cv2.LINE_AA)

    for hand in item.detection.hands:
        skeleton(hand.xy, hand.confidence, HAND_CONNECTIONS)
    skeleton(
        item.detection.body_xy, item.detection.body_confidence, BODY_CONNECTIONS
    )
    for point, confidence in zip(
        item.detection.face_xy, item.detection.face_confidence
    ):
        if confidence > 0:
            cv2.circle(output, pixel(point), 2, (255, 255, 255), -1, cv2.LINE_AA)
    return output


class DeferredWindows:
    """FIFO for exact observations awaiting Finish, without retaining RGB in RAM.

    Pickle is read only from this process's private temporary file, never a user path.
    """

    def __init__(self):
        # ponytail: disk use grows until Finish; compress if long utterances require it.
        self.file = tempfile.TemporaryFile()
        self.offsets: deque[int] = deque()

    def append(self, window):
        self.file.seek(0, 2)
        offset = self.file.tell()
        pickle.dump(window, self.file, protocol=pickle.HIGHEST_PROTOCOL)
        self.offsets.append(offset)

    def popleft(self):
        self.file.seek(self.offsets.popleft())
        return pickle.load(self.file)

    def __len__(self):
        return len(self.offsets)

    def clear(self):
        self.offsets.clear()
        self.file.seek(0)
        self.file.truncate()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class RetainedFrozenWindows:
    """Disk-backed visual features for Finish-time CTC re-decoding."""

    def __init__(self):
        self.file = tempfile.TemporaryFile()
        self.offsets: list[int] = []

    def append(self, frozen: np.ndarray) -> None:
        self.file.seek(0, 2)
        self.offsets.append(self.file.tell())
        pickle.dump(np.asarray(frozen, dtype=np.float32), self.file,
                    protocol=pickle.HIGHEST_PROTOCOL)

    def snapshot(self) -> list[np.ndarray]:
        values = []
        for offset in self.offsets:
            self.file.seek(offset)
            values.append(pickle.load(self.file))
        return values

    def clear(self) -> None:
        self.offsets.clear()
        self.file.seek(0)
        self.file.truncate()

    def __len__(self) -> int:
        return len(self.offsets)

    def close(self) -> None:
        self.file.close()


def _common_prefix_size(left: list[str], right: list[str]) -> int:
    return next((index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
                min(len(left), len(right)))


def stitch_revisable_ctc_logits(
    chunks: list[tuple[int, np.ndarray]], total_steps: int,
) -> np.ndarray:
    """Average overlapping visual CTC emissions on their absolute time line."""
    if not chunks:
        return np.empty((0, 101), np.float32)
    classes = chunks[0][1].shape[-1]
    summed = np.zeros((total_steps, classes), np.float64)
    counts = np.zeros(total_steps, np.int32)
    for start_window, logits in chunks:
        start = start_window * 8
        end = min(total_steps, start + len(logits))
        summed[start:end] += logits[:end - start]
        counts[start:end] += 1
    if not np.all(counts):
        raise ValueError("Finish-time visual decoder left an uncovered CTC step")
    return (summed / counts[:, None]).astype(np.float32)


def final_revisable_decode(arbiter, retained: RetainedFrozenWindows) -> dict[str, object]:
    """Re-run every retained visual window; the model itself supports eight at once."""
    windows = retained.snapshot()
    if not windows:
        return {"hypothesis": [], "chunks": [], "retained_windows": 0}
    overlap = 2
    stride = MAXIMUM_WINDOWS - overlap
    decoded: list[tuple[int, np.ndarray]] = []
    chunks: list[dict[str, object]] = []
    for start in range(0, len(windows), stride):
        window_slice = windows[start:start + MAXIMUM_WINDOWS]
        if not window_slice:
            break
        logits, metadata = arbiter.decode_frozen_logits(window_slice)
        decoded.append((start, np.asarray(logits, dtype=np.float32)))
        chunks.append({
            "start_window": start,
            "end_window": start + len(window_slice),
            "ctc_steps": len(logits),
            "window_count": len(window_slice),
            "specialist_selected": metadata.get("specialist_selected"),
        })
        if start + len(window_slice) == len(windows):
            break
    stitched = stitch_revisable_ctc_logits(decoded, len(windows) * 8)
    tokens, positions = collapse_ctc_path(stitched, len(stitched))
    tokens, positions = supported_ctc_path(tokens, positions)
    return {
        "hypothesis": [arbiter.labels[token - 1] for token in tokens],
        "token_positions": list(positions),
        "chunks": chunks,
        "retained_windows": len(windows),
        "overlap_windows": overlap,
    }


def run(
    args: argparse.Namespace, classifier_factory=ReelCascadeClassifier,
) -> dict[str, object]:
    window_backend = getattr(args, "transcript_backend", "ctc") == "stage1-window"
    preserve_pending = getattr(args, "preserve_pending_frames", False)
    stage2_at_finish = getattr(args, "stage2_at_finish", False)
    stage2_review_only = getattr(args, "stage2_review_only", False)
    provisional_glosses = getattr(args, "provisional_glosses", False)
    sequence_preview = getattr(args, "sequence_preview", False)
    revisable_transcript = getattr(args, "revisable_transcript", False)
    if revisable_transcript:
        sequence_preview = True
        provisional_glosses = True
    if sequence_preview:
        if args.no_stage2_arbiter:
            raise ValueError("sequence preview requires Stage2")
        stage2_at_finish = False
        stage2_review_only = True
    sequence_arbiter = (
        None if window_backend or args.no_stage2_arbiter else LiveStage2CTC(args)
    )
    classifier = (
        Stage1WindowClassifier(args) if window_backend
        else sequence_arbiter if sequence_preview
        else classifier_factory(args)
    )
    lip_disambiguator = (
        None if window_backend or args.no_lip_marker_verifier
        else LipMarkerDisambiguator(args.lip_marker_model)
    )
    if window_backend:
        provisional_glosses = True
        sequence_preview = False
        revisable_transcript = False
        stage2_at_finish = False
        stage2_review_only = False
    naturalizer = make_naturalizer(args)
    speaker = None if args.no_speech else LiveSpeaker()
    source = str(args.video) if args.video else f"camera:{args.camera}"
    recorder = SessionRecorder(args, classifier, source)
    recorder.data.update({
        "format": "slt_live_reel_stage1_v17_session",
        "version": 1,
        "prototype_scope": (
            "revisable timestamped Stage-1 windows with NO_EMIT"
            if window_backend else
            "growing activity-anchored Stage-1 candidates with label-stability locks"
        ),
        "score_semantics": "uncalibrated Stage-1 softmax with temporal agreement",
    })
    recorder.data["config"].update({
        "candidate_minimum_seconds": args.candidate_minimum_seconds,
        "candidate_maximum_seconds": args.candidate_maximum_seconds,
        "probe_interval_seconds": args.probe_interval_seconds,
        "stability_hits": args.stability_hits,
        "commit_score": args.commit_score,
        "commit_hits": args.commit_hits,
        "instant_commit_score": args.instant_commit_score,
        "full_visual_verifier": not window_backend and not getattr(
            args, "landmark_only_commit", False
        ),
        "release_hits": args.release_hits,
        "transition_overlap_seconds": args.transition_overlap_seconds,
        "no_hand_release_seconds": args.no_hand_release_seconds,
        "capture_policy": "latest-frame webcam; sequential saved video",
        "realtime_video": getattr(args, "realtime_video", False),
        "frame_timestamps": (
            None if getattr(args, "frame_timestamps", None) is None
            else str(args.frame_timestamps)
        ),
        "exclude_final_frames": getattr(args, "exclude_final_frames", 0),
        "auxiliary_display_schedule": "every_processed_frame_with_last-valid_hold",
        "auxiliary_detection_schedule": (
            "every_processed_frame" if args.dense_model_auxiliary
            else "training_sparse_every_eighth_frame_with_display_hold"
        ),
        "auxiliary_model_schedule": (
            "every_processed_frame" if args.dense_model_auxiliary
            else "training_sparse_every_eighth_frame"
        ),
        "lip_marker_model": None if lip_disambiguator is None else str(
            lip_disambiguator.path
        ),
        "lip_marker_minimum_confidence": args.lip_marker_minimum_confidence,
        "stage2_sequence_arbiter": sequence_arbiter is not None,
        "preserve_pending_frames": preserve_pending,
        "provisional_glosses": provisional_glosses,
        "stage2_at_finish": stage2_at_finish,
        "sequence_preview": sequence_preview,
        "revisable_transcript": revisable_transcript or window_backend,
        "transcript_backend": getattr(args, "transcript_backend", "ctc"),
        "stage2_review_only": stage2_review_only,
        "stage2_minimum_phrase_seconds": args.stage2_minimum_phrase_seconds,
        "finish_gesture": not args.no_finish_gesture,
        "finish_gesture_hold_seconds": args.finish_gesture_hold_seconds,
    })
    if sequence_arbiter is not None:
        recorder.data["models"]["stage2_sequence_arbiter"] = (
            sequence_arbiter.provenance()
        )
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
    frame_timestamps = load_frame_timestamps(getattr(args, "frame_timestamps", None))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_timestamps is not None and len(frame_timestamps) < reported_frames:
        capture.release()
        recorder.close()
        raise ValueError('frame timestamps do not cover the video')
    available_frames = (
        min(len(frame_timestamps), reported_frames or len(frame_timestamps)) if frame_timestamps is not None
        else reported_frames if getattr(args, "exclude_final_frames", 0) else 0
    )
    stop_frame = (
        None if available_frames <= 0
        else max(0, available_frames - getattr(args, "exclude_final_frames", 0))
    )
    if available_frames and args.exclude_final_frames >= available_frames:
        capture.release()
        recorder.close()
        raise ValueError('final-frame exclusion removes the entire video')
    detector = AppleVisionDetector(args.minimum_point_confidence)
    lip_tracker = LipMarkerTracker(enabled=lip_disambiguator is not None or (
        not args.no_display and not args.no_live_lips
    ))
    finish_gesture = FinishGesture(args.finish_gesture_hold_seconds)
    previous_wrists = {"left": None, "right": None}
    maximum_frames = max(8, math.ceil(args.candidate_maximum_seconds * args.processing_fps))
    observations: deque[ObservedFrame] = deque(maxlen=maximum_frames)
    lock = StableGlossLock(args.stability_hits, args.release_hits)
    commit_lock = VerifiedCommitLock(args.commit_hits, args.instant_commit_score)
    classifier_executor = ThreadPoolExecutor(max_workers=1)
    ctc_executor = ThreadPoolExecutor(max_workers=1)
    language_executor = ThreadPoolExecutor(max_workers=1)
    warm_future = language_executor.submit(naturalizer.warm)
    future: Future | None = None
    future_kind: str | None = None
    future_clip: list[ObservedFrame] | None = None
    future_result: dict[str, object] | None = None
    future_proposal: str | None = None
    future_epoch = epoch = 0
    naturalizer_future: Future | None = None
    naturalizer_context: dict[str, object] | None = None
    pending_speech: deque[dict[str, object]] = deque()
    results: list[dict[str, object]] = []
    glosses: list[str] = []
    pending_pair: dict[str, object] | None = None
    ctc_buffer = ElapsedWindowBuffer(args.stage2_window_seconds)
    ctc_ready = DeferredWindows() if stage2_at_finish else deque()
    ctc_future: Future | None = None
    ctc_future_epoch = 0
    ctc_prior: list[np.ndarray] = []
    ctc_locked: list[str] = []
    ctc_locked_positions: list[int] = []
    ctc_position_offset = 0
    sequence_prefix = StreamingCTCPrefix()
    ctc_context_hypothesis: list[str] = []
    ctc_context_positions: list[int] = []
    ctc_hypothesis: list[str] = []
    last_stage2_hypothesis: list[str] = []
    ctc_results: list[dict[str, object]] = []
    retained_frozen = RetainedFrozenWindows()
    revisable_hypothesis: list[str] = []
    stage1_window_transcript = Stage1WindowTranscript()
    stage1_window_retained: list[tuple[float, np.ndarray]] = []
    stage1_window_results: list[dict[str, object]] = []
    stage1_next_end: float | None = None
    activity_first_seconds: float | None = None
    activity_last_seconds: float | None = None
    finishing_glosses: list[str] = []
    sentence = ""
    finish_requested = False
    finish_started_at: float | None = None
    last_preview: str | None = None
    active = False
    activity_hits = 0
    no_hand_since: float | None = None
    latest: ObservedFrame | None = None
    display_latest: ObservedFrame | None = None
    display_body: tuple[np.ndarray, np.ndarray] | None = None
    display_face: tuple[np.ndarray, np.ndarray] | None = None
    latest_lips: np.ndarray | None = None
    latest_result: dict[str, object] | None = None
    next_process = next_probe = 0.0
    last_probe_end = float("-inf")
    frame_index = processed = 0
    last_source_seconds = 0.0
    wall_started = time.perf_counter()
    capture_elapsed: float | None = None
    camera = None if args.video else LatestCamera(capture, wall_started)
    camera_sequence = dropped_camera_frames = 0
    display_times: deque[float] = deque(maxlen=30)
    ui_actions: deque[tuple[str, float]] = deque()
    display_size = [1280, 720]
    hud = ReelHud(provisional=provisional_glosses)
    last_button_seconds = {"reset": float("-inf"), "finish": float("-inf")}
    warm_logged = False

    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME)

        def on_mouse(event, x, y, _flags, _parameter):
            if event != cv2.EVENT_LBUTTONUP:
                return
            action = clicked_reel_control(x, y, display_size[0], display_size[1])
            seconds = time.perf_counter() - wall_started
            if (
                action is not None
                and seconds - last_button_seconds[action] >= 0.25
            ):
                last_button_seconds[action] = seconds
                ui_actions.append((action, seconds))

        cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    def clear_candidate(
        *, keep_after: float | None = None, consumed_until: float | None = None,
    ) -> None:
        nonlocal active, activity_hits, next_probe, latest_result, last_preview
        retained = [] if keep_after is None else [
            item for item in observations if item.seconds >= keep_after
        ]
        if consumed_until is not None:
            retained = [item for item in observations if item.seconds > consumed_until]
        observations.clear()
        observations.extend(retained)
        active = bool(retained)
        activity_hits = 0
        next_probe = retained[-1].seconds if retained else 0.0
        if provisional_glosses and not retained:
            if latest_result and not latest_result.get("committed_gloss"):
                latest_result = None
            last_preview = None

    def process_stage1_windows(current_seconds: float) -> None:
        nonlocal stage1_next_end, latest_result
        if not window_backend or not stage1_window_retained:
            return
        if stage1_next_end is None:
            stage1_next_end = (
                stage1_window_retained[0][0] + STAGE1_WINDOW_SECONDS
            )
        while stage1_next_end <= current_seconds + 1e-6:
            end = stage1_next_end
            stage1_next_end += STAGE1_WINDOW_STRIDE_SECONDS
            clip = [
                item for item in stage1_window_retained
                if end - STAGE1_WINDOW_SECONDS - 1e-9 <= item[0] <= end + 1e-9
            ]
            if len(clip) < 2:
                stage1_window_transcript.update(NO_EMIT, end)
                continue
            try:
                result = classifier.classify_raw_window(
                    np.stack([v[1] for v in clip]), [v[0] for v in clip], end,
                )
            except ValueError as error:
                recorder.add_event({
                    "type": "stage1_window_skipped", "end_seconds": end,
                    "reason": str(error),
                })
                stage1_window_transcript.update(NO_EMIT, end)
                continue
            previous = list(stage1_window_transcript.words)
            hypothesis = stage1_window_transcript.update(
                str(result["gloss"]), end
            )
            glosses[:] = hypothesis
            latest_result = result
            results.append(result)
            stage1_window_results.append(result)
            recorder.add(result)
            unchanged = _common_prefix_size(previous, hypothesis)
            recorder.add_event({
                "type": "stage1_window_transcript_update",
                "prediction": result["gloss"],
                "previous": previous,
                "hypothesis": hypothesis,
                "unchanged_prefix": unchanged,
                "replaced_tail": previous[unchanged:],
                "new_tail": hypothesis[unchanged:],
                "window_start_seconds": clip[0][0],
                "window_end_seconds": end,
                "elapsed_seconds": time.perf_counter() - wall_started,
                "observed_seconds": current_seconds,
                "latency_ms": result["latency_ms"],
            })

    def submit() -> None:
        nonlocal future, future_kind, future_clip, future_result
        nonlocal future_proposal, future_epoch, next_probe, last_probe_end
        if window_backend or sequence_preview or future is not None or not active or (finish_requested and not preserve_pending):
            return
        clip = list(observations)
        if len(clip) < 4:
            return
        duration = clip[-1].seconds - clip[0].seconds
        if duration < args.candidate_minimum_seconds:
            return
        if finish_requested:
            # Only new evidence may add a stability hit; never loop on a frozen tail.
            if clip[-1].seconds <= last_probe_end:
                return
        elif clip[-1].seconds < next_probe:
            return
        future_epoch = epoch
        future_clip = clip
        future_result = None
        future_proposal = None
        future_kind = "proposal"
        future = classifier_executor.submit(classifier.classify, clip)
        last_probe_end = clip[-1].seconds
        next_probe = clip[-1].seconds + args.probe_interval_seconds

    def submit_ctc() -> None:
        nonlocal ctc_future, ctc_future_epoch
        nonlocal ctc_locked, ctc_context_hypothesis, ctc_context_positions, ctc_position_offset
        if preserve_pending and finish_requested:
            submit()
        if sequence_arbiter is None or ctc_future is not None or not ctc_ready:
            return
        if stage2_at_finish and (not finish_requested or future is not None):
            return
        if len(ctc_prior) >= MAXIMUM_WINDOWS:
            ctc_locked_positions.extend(ctc_position_offset + p for p in ctc_context_positions if p < 8)
            ctc_position_offset += 8
            ctc_locked, ctc_context_hypothesis, ctc_context_positions = (
                roll_ctc_prefix(
                    ctc_locked, ctc_context_hypothesis, ctc_context_positions
                )
            )
            del ctc_prior[0]
        window = ctc_ready.popleft()
        prior = list(ctc_prior)
        ctc_future_epoch = epoch
        ctc_future = ctc_executor.submit(
            sequence_arbiter.classify_window, window, prior
        )

    def collect_ctc(wait: bool = False) -> None:
        nonlocal ctc_future, ctc_hypothesis
        nonlocal ctc_context_hypothesis, ctc_context_positions
        if ctc_future is None or (not wait and not ctc_future.done()):
            return
        frozen, result = ctc_future.result()
        ctc_future = None
        ignored = ctc_future_epoch != epoch
        if frozen is not None and not ignored:
            ctc_prior.append(frozen)
            if revisable_transcript:
                retained_frozen.append(frozen)
        if result.get("accepted") and not ignored:
            ctc_context_hypothesis = list(result["hypothesis"])
            ctc_context_positions = list(result["token_positions"])
            ctc_hypothesis = [*ctc_locked, *ctc_context_hypothesis]
            result["context_hypothesis"] = list(ctc_context_hypothesis)
            result["locked_prefix"] = list(ctc_locked)
            result["hypothesis"] = list(ctc_hypothesis)
            if revisable_transcript:
                previous = list(revisable_hypothesis)
                revisable_hypothesis[:] = ctc_hypothesis
                glosses[:] = revisable_hypothesis
                unchanged = _common_prefix_size(previous, revisable_hypothesis)
                recorder.add_event({
                    "type": "revisable_transcript_update",
                    "previous": previous,
                    "hypothesis": list(revisable_hypothesis),
                    "unchanged_prefix": unchanged,
                    "replaced_tail": previous[unchanged:],
                    "new_tail": revisable_hypothesis[unchanged:],
                    "retained_windows": len(retained_frozen),
                    "observed_seconds": None if latest is None else latest.seconds,
                    "window_end_seconds": result.get("end_seconds"),
                    "elapsed_seconds": time.perf_counter() - wall_started,
                })
            elif sequence_preview:
                state = sequence_prefix.update(
                    list(ctc_hypothesis),
                    [*ctc_locked_positions, *[ctc_position_offset + p for p in ctc_context_positions]],
                    ctc_position_offset + len(ctc_prior) * 8,
                )
                glosses[:] = state["committed"]
                recorder.add_event({"type": "sequence_prefix_update", **state,
                    "evidence_end": ctc_position_offset + len(ctc_prior) * 8,
                    "observed_seconds": None if latest is None else latest.seconds,
                    "window_end_seconds": result.get("end_seconds"),
                    "elapsed_seconds": time.perf_counter() - wall_started})
        ctc_results.append(result)
        recorder.add_event({
            "type": "stage2_sequence_update",
            "after_finish": finish_requested,
            "hypothesis": list(ctc_hypothesis),
            "accepted": bool(result.get("accepted")),
            "ignored_after_reset": ignored,
            "window_count": result.get("window_count"),
            "token_positions": result.get("token_positions"),
            "start_seconds": result.get("start_seconds"),
            "end_seconds": result.get("end_seconds"),
            "latency_ms": result.get("latency_ms"),
        })
        submit_ctc()

    def collect(wait: bool = False) -> None:
        nonlocal future, future_kind, future_clip, future_result
        nonlocal future_proposal, latest_result, pending_pair
        nonlocal last_preview
        if future is None or (not wait and not future.done()):
            return
        if future_kind == "proposal":
            result = future.result()
            ignored = future_epoch != epoch
            proposal = None if ignored else lock.update(result)
            if not ignored:
                latest_result = result
                if provisional_glosses and result.get("accepted"):
                    label = str(result["gloss"])
                    if label != last_preview and label != lock.suppressed:
                        recorder.add_event({
                            "type": "provisional_gloss", "gloss": label,
                            "start_seconds": result["start_seconds"],
                            "end_seconds": result["end_seconds"],
                            "observed_seconds": None if latest is None else latest.seconds,
                            "elapsed_seconds": time.perf_counter() - wall_started,
                            "since_activity_seconds": (
                                None if latest is None or activity_first_seconds is None
                                else latest.seconds - activity_first_seconds
                            ),
                            "latency_ms": result.get("latency_ms"),
                        })
                        last_preview = label
            if proposal is not None and future_clip is not None:
                future_result = result
                future_proposal = proposal
                if getattr(args, "landmark_only_commit", False):
                    verifier = proposal_as_fast_verifier(result)
                else:
                    future_kind = "verification"
                    future = classifier_executor.submit(classifier.verify, future_clip)
                    return
            else:
                verifier = None
        else:
            verifier = future.result()
            if future_result is None:
                raise RuntimeError("verification completed without a proposal")
            result = future_result
            ignored = future_epoch != epoch
            proposal = None if ignored else future_proposal

        emitted = None
        if proposal is not None and future_clip is not None:
            verifier = add_targeted_lip_evidence(
                lip_disambiguator, future_clip, verifier, proposal,
                args.lip_marker_minimum_confidence,
            )
            lip = verifier.get("diagnostics", {}).get("targeted_lip_prediction")
            lip_score = (
                float(lip["confidence"])
                if lip is not None and verifier["diagnostics"].get(
                    "targeted_lip_source"
                ) == "media_pipe_lip_markers"
                else 0.0
            )
            agreement_score = (
                float(result.get("model_score", 0.0))
                if str(verifier.get("candidate_gloss")) == proposal
                else 0.0
            )
            commit_score = max(
                float(verifier.get("model_score", 0.0)), lip_score,
                agreement_score,
            )
            verifier["commit_score"] = commit_score
            verifier["commit_score_threshold"] = args.commit_score
            verifier["proposal_verifier_agreement_score"] = agreement_score
            commit_ready = bool(verifier.get("accepted")) and commit_lock.update(
                str(verifier.get("candidate_gloss")), commit_score,
                proposal=proposal,
                proposal_score=float(result.get("model_score", 0.0)),
                minimum_score=args.commit_score,
            )
            verifier["commit_stability_candidate"] = commit_lock.candidate
            verifier["commit_stability_hits"] = commit_lock.hits
            verifier["commit_stability_required"] = commit_lock.required_hits
            if commit_ready:
                emitted = str(verifier["gloss"])
                lock.verified(emitted)
            else:
                if commit_score < args.commit_score:
                    verifier.setdefault("rejection_reasons", []).append(
                        "low_commit_score"
                    )
                lock.candidate = None
                lock.hits = 0
        if (
            emitted is not None
            and pending_pair is None
            and glosses
            and glosses[-1] == emitted
        ):
            recorder.add_event({
                "type": "adjacent_duplicate_suppressed", "gloss": emitted,
            })
            emitted = None
        result.update({
            "reel_epoch": future_epoch,
            "ignored_after_reset": ignored,
            "stable_candidate": lock.candidate,
            "stable_hits": lock.hits,
            "lock_proposal": proposal,
            "full_verifier": verifier,
            "committed_gloss": emitted,
            "completed_observed_seconds": None if latest is None else latest.seconds,
            "completed_elapsed_seconds": time.perf_counter() - wall_started,
        })
        results.append(result)
        recorder.add(result)
        if not ignored:
            latest_result = result
        if args.verbose_predictions:
            print(json.dumps(result, indent=2))
        elif emitted is not None:
            print(json.dumps({
                "committed_gloss": emitted,
                "proposal": proposal,
                "score": verifier.get("commit_score") if verifier else None,
            }))
        future = None
        future_kind = None
        future_clip = None
        future_result = None
        future_proposal = None
        if emitted is not None:
            lip_source = verifier.get("diagnostics", {}).get("targeted_lip_source")
            if emitted in {"GOOD", "THANKYOU"} and lip_source != "media_pipe_lip_markers":
                pending_pair = {
                    "fallback": emitted, "reference": len(results) - 1,
                    "created_utc": utc_now(),
                }
                result["pending_ambiguous_pair"] = True
                recorder.add_event({
                    "type": "ambiguous_pair_pending",
                    "candidates": ["GOOD", "THANKYOU"],
                    "fallback": emitted,
                })
            else:
                if pending_pair is not None:
                    resolved = resolve_good_thankyou_context(
                        str(pending_pair["fallback"]), emitted
                    )
                    glosses.append(resolved)
                    pending_speech.append({
                        "text": resolved, "kind": "gloss",
                        "reference": pending_pair["reference"],
                    })
                    recorder.add_event({
                        "type": "ambiguous_pair_resolved", "gloss": resolved,
                        "following_gloss": emitted,
                    })
                    pending_pair = None
                glosses.append(emitted)
                pending_speech.append({
                    "text": emitted, "kind": "gloss", "reference": len(results) - 1,
                })
            end = float(result["end_seconds"])
            if args.transition_overlap_seconds > 0:
                clear_candidate(keep_after=end - args.transition_overlap_seconds)
            elif preserve_pending:
                clear_candidate(consumed_until=end)
            else:
                clear_candidate()
            if preserve_pending:
                recorder.add_event({
                    "type": "committed_buffer_retained", "gloss": emitted,
                    "consumed_until": end, "retained_frames": len(observations),
                    "retained_start": observations[0].seconds if observations else None,
                })
            last_preview = None

    def service_language() -> None:
        nonlocal warm_logged, naturalizer_future, naturalizer_context, sentence
        nonlocal finish_requested, pending_pair
        nonlocal ctc_hypothesis, last_stage2_hypothesis
        nonlocal ctc_context_hypothesis, ctc_context_positions
        nonlocal activity_first_seconds, activity_last_seconds
        nonlocal latest_result, finish_started_at, ctc_position_offset
        nonlocal stage1_next_end
        submit_ctc()
        if not warm_logged and warm_future.done():
            warm_logged = True
            recorder.add_event({"type": "naturalizer_warmup", **warm_future.result()})
        if (
            finish_requested and future is None and ctc_future is None
            and not ctc_ready and naturalizer_future is None
        ):
            finish_requested = False
            if pending_pair is not None:
                resolved = str(pending_pair["fallback"])
                glosses.append(resolved)
                recorder.add_event({
                    "type": "ambiguous_pair_resolved", "gloss": resolved,
                    "following_gloss": None,
                })
                pending_pair = None
            stage1_finished = list(glosses)
            active_seconds = (
                0.0 if activity_first_seconds is None or activity_last_seconds is None
                else activity_last_seconds - activity_first_seconds
            )
            last_stage2_hypothesis = list(ctc_hypothesis)
            accepted_ctc = [
                collapse_adjacent_glosses(list(row["hypothesis"]))
                for row in ctc_results if row.get("accepted")
            ]
            stage2_stable = (
                len(accepted_ctc) >= 2
                and accepted_ctc[-1] == accepted_ctc[-2]
            )
            final_visual = None
            if window_backend:
                finished, final_results = final_stage1_window_decode(
                    classifier, stage1_window_retained
                )
                final_visual = {
                    "hypothesis": finished,
                    "windows": len(final_results),
                    "partial_tail_included": bool(final_results) and not np.isclose(
                        final_results[-1]["end_seconds"],
                        stage1_window_results[-1]["end_seconds"]
                        if stage1_window_results else float("-inf"),
                    ),
                }
                selection_mode = "stage1_window_visual_redecode"
            elif revisable_transcript:
                final_visual = final_revisable_decode(sequence_arbiter, retained_frozen)
                finished = list(final_visual["hypothesis"])
                selection_mode = "revisable_visual_redecode"
            elif sequence_preview:
                finished = stage1_finished
                selection_mode = "confirmed_ctc_prefix_with_provisional_review"
            elif stage2_review_only:
                finished = stage1_finished
                selection_mode = "stage1_with_stage2_review_candidate"
            else:
                finished, selection_mode = select_finished_sequence(
                    stage1_finished, last_stage2_hypothesis, active_seconds,
                    args.stage2_minimum_phrase_seconds,
                    stage2_stable=stage2_stable,
                )
            recorder.add_event({
                "type": "finished_sequence_selected",
                "stage1": [] if sequence_preview else stage1_finished,
                "confirmed_ctc_prefix": stage1_finished if sequence_preview and not revisable_transcript else None,
                "stage2": last_stage2_hypothesis,
                "selected": finished,
                "selection_mode": selection_mode,
                "active_seconds": active_seconds,
                "stage2_stable": stage2_stable,
                "final_visual_redecode": final_visual,
                "finish_decode_ms": (
                    None if finish_started_at is None
                    else 1000 * (time.perf_counter() - finish_started_at)
                ),
            })
            finishing_glosses[:] = finished
            glosses.clear()
            lock.reset()
            commit_lock.reset()
            clear_candidate()
            ctc_buffer.reset()
            ctc_ready.clear()
            ctc_prior.clear()
            ctc_locked.clear()
            ctc_locked_positions.clear()
            ctc_position_offset = 0
            sequence_prefix.reset()
            ctc_context_hypothesis = []
            ctc_context_positions = []
            ctc_hypothesis = []
            ctc_results.clear()
            retained_frozen.clear()
            revisable_hypothesis.clear()
            stage1_window_transcript.reset()
            stage1_window_retained.clear()
            stage1_window_results.clear()
            stage1_next_end = None
            activity_first_seconds = None
            activity_last_seconds = None
            if provisional_glosses:
                latest_result = None
            if not finished:
                sentence = "No recognized signs to finish."
                recorder.add_event({"type": "finish_empty"})
            else:
                naturalizer_context = {
                    "utterance_id": f"reel-{len(recorder.data['utterances']) + 1:04d}",
                    "requested_utc": utc_now(),
                    "glosses": finished,
                    "epoch": epoch,
                    "finish_started_at": finish_started_at,
                }
                finish_started_at = None
                naturalizer_future = language_executor.submit(
                    naturalizer.rephrase, finished
                )
        if naturalizer_future is not None and naturalizer_future.done():
            value = naturalizer_future.result()
            visible = naturalizer_context is not None and (
                naturalizer_context.get("epoch") == epoch
            )
            utterance = {
                **(naturalizer_context or {}), **value, "completed_utc": utc_now(),
                "displayed_and_spoken": visible,
                "finish_total_ms": (
                    None if not naturalizer_context or naturalizer_context.get("finish_started_at") is None
                    else 1000 * (time.perf_counter() - naturalizer_context["finish_started_at"])
                ),
            }
            recorder.add_utterance(utterance)
            if visible:
                sentence = str(value["sentence"])
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
            if provisional_glosses and item["kind"] != "finished_sentence":
                continue
            speaker.enqueue(str(item["text"]), str(item["kind"]), item["reference"])
            recorder.add_event({"type": "speech_queued", **item})
        started = speaker.update()
        if started is not None:
            recorder.add_event({"type": "speech_started", **started})

    def reset_display(seconds: float, source_name: str) -> None:
        nonlocal epoch, sentence, finish_requested, pending_pair
        nonlocal activity_first_seconds, activity_last_seconds, ctc_hypothesis
        nonlocal ctc_context_hypothesis, ctc_context_positions
        nonlocal latest_result
        nonlocal last_preview, last_stage2_hypothesis
        nonlocal last_probe_end, finish_started_at, ctc_position_offset
        nonlocal stage1_next_end
        epoch += 1
        cleared = list(glosses)
        glosses.clear()
        pending_pair = None
        finishing_glosses.clear()
        lock.reset()
        commit_lock.reset()
        clear_candidate()
        ctc_buffer.reset()
        ctc_ready.clear()
        ctc_prior.clear()
        ctc_locked.clear()
        ctc_locked_positions.clear()
        ctc_position_offset = 0
        sequence_prefix.reset()
        ctc_context_hypothesis = []
        ctc_context_positions = []
        ctc_hypothesis = []
        ctc_results.clear()
        retained_frozen.clear()
        revisable_hypothesis.clear()
        stage1_window_transcript.reset()
        stage1_window_retained.clear()
        stage1_window_results.clear()
        stage1_next_end = None
        activity_first_seconds = None
        activity_last_seconds = None
        latest_result = None
        last_preview = None
        last_probe_end = float("-inf")
        finish_started_at = None
        last_stage2_hypothesis = []
        sentence = "Display reset."
        finish_requested = False
        pending_speech.clear()
        if speaker is not None:
            speaker.clear()
        recorder.add_event({
            "type": "reset", "source": source_name, "seconds": seconds,
            "cleared_glosses": cleared, "classifier_pending": future is not None,
        })

    def request_finish(seconds: float, source_name: str) -> None:
        nonlocal finish_requested, sentence, finish_started_at
        if finish_requested or (naturalizer_future is not None and not preserve_pending):
            sentence = "Finish already in progress."
            recorder.add_event({
                "type": "finish_ignored_already_pending",
                "source": source_name, "seconds": seconds,
            })
            return
        finish_requested = True
        finish_started_at = time.perf_counter()
        sentence = "Finishing..."
        if sequence_arbiter is not None:
            tail = ctc_buffer.finish()
            if tail is not None:
                ctc_ready.append(tail)
            submit_ctc()
        recorder.add_event({
            "type": "finish_requested", "source": source_name, "seconds": seconds,
            "active_glosses": list(glosses), "classifier_pending": future is not None,
        })

    try:
        while True:
            if camera is None:
                if stop_frame is not None and frame_index >= stop_frame:
                    break
                ok, raw = capture.read()
                if not ok:
                    break
                seconds = (
                    frame_timestamps[frame_index]
                    if frame_timestamps is not None else frame_index / source_fps
                )
                frame_index += 1
                if getattr(args, "realtime_video", False):
                    delay = wall_started + seconds - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
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
            last_source_seconds = seconds
            canonical = orient_frame(raw, args.rotation, args.input_mirrored)
            recorder.write_frame(canonical, seconds)
            collect()
            collect_ctc()
            service_language()
            if not finish_requested and seconds + 1e-6 >= next_process:
                if sequence_preview or window_backend:
                    latest = observe_stage2_frame(canonical, seconds, processed, detector, previous_wrists, args)
                    detection = latest.detection
                    assigned = latest.assigned
                    latest_lips = lip_tracker.detect(limit_image_side(canonical, args.detection_image_side))
                else:
                    detection_frame = limit_image_side(canonical, args.detection_image_side)
                    frame = limit_image_side(canonical, args.maximum_image_side)
                    feature_frame = args.dense_model_auxiliary or (
                        processed % V17Config().face_interval == 0
                    )
                    body_frame = args.dense_model_auxiliary or (
                        processed % V17Config().body_interval == 0
                    )
                    latest_lips = lip_tracker.detect(detection_frame)
                    detection = detector.detect(
                        detection_frame,
                        include_body=body_frame,
                        include_face=feature_frame,
                        include_hands=True,
                    )
                    assigned = assign_hands(detection.hands, previous_wrists)
                    model_detection = FrameDetection(
                        detection.hands,
                        detection.body_xy if body_frame else np.zeros_like(
                            detection.body_xy
                        ),
                        detection.body_confidence if body_frame else np.zeros_like(
                            detection.body_confidence
                        ),
                        detection.face_xy,
                        detection.face_confidence,
                    )
                    latest = ObservedFrame(
                        frame, model_detection, assigned, seconds,
                        wrist_motion(assigned, previous_wrists),
                        *observation_quality(detection),
                        face_for_features=feature_frame,
                    )
                latest.lip_points = latest_lips
                visible_detection, display_body, display_face = (
                    persistent_auxiliary_detection(
                        detection, display_body, display_face
                    )
                )
                display_latest = copy.copy(latest)
                display_latest.detection = visible_detection
                gesture_triggered = (
                    False if args.no_finish_gesture
                    else finish_gesture.update(assigned, seconds)
                )
                gesture_frame = (
                    not args.no_finish_gesture and finish_gesture.active
                )
                processed += 1
                next_process = max(next_process + 1.0 / args.processing_fps, seconds)
                if gesture_frame:
                    # The control gesture is UI, not vocabulary. Never feed its
                    # frames to Stage 1 or Stage 2.
                    no_hand_since = None
                    if gesture_triggered:
                        request_finish(seconds, "ten_finger_gesture")
                else:
                    observations.append(latest)
                    if window_backend:
                        raw_features, _ = raw_observation_features([latest])
                        stage1_window_retained.append((latest.seconds, raw_features[0]))
                        process_stage1_windows(seconds)
                    if sequence_arbiter is not None:
                        ctc_window = ctc_buffer.add(latest)
                        if ctc_window is not None:
                            ctc_ready.append(ctc_window)
                            submit_ctc()
                    if latest.hand_quality > 0:
                        no_hand_since = None
                    elif no_hand_since is None:
                        no_hand_since = seconds
                    elif seconds - no_hand_since >= args.no_hand_release_seconds:
                        lock.no_hands()
                        clear_candidate()
                    moving = (
                        latest.hand_quality > 0
                        and latest.motion >= args.start_motion
                    )
                    if moving:
                        if provisional_glosses and finishing_glosses:
                            finishing_glosses.clear()
                            last_stage2_hypothesis = []
                            sentence = ""
                        if activity_first_seconds is None:
                            activity_first_seconds = seconds
                        activity_last_seconds = seconds
                    if not active:
                        activity_hits = activity_hits + 1 if moving else 0
                        if activity_hits >= args.start_frames:
                            retained = [
                                item for item in observations
                                if item.seconds >= seconds - args.preroll_seconds
                            ]
                            observations.clear()
                            observations.extend(retained)
                            active = True
                    elif (
                        observations[-1].seconds - observations[0].seconds
                        >= args.candidate_maximum_seconds
                        and future is None and not sequence_preview
                    ):
                        recorder.add_event({
                            "type": "candidate_timeout",
                            "seconds": seconds,
                            "suppressed_gloss": lock.suppressed,
                        })
                        lock.candidate = None
                        lock.hits = 0
                        if args.transition_overlap_seconds > 0:
                            clear_candidate(
                                keep_after=seconds - args.transition_overlap_seconds
                            )
                        else:
                            clear_candidate()
                    submit()

            if not args.no_display:
                now = time.perf_counter()
                display_times.append(now)
                fps = (
                    (len(display_times) - 1) / (display_times[-1] - display_times[0])
                    if len(display_times) > 1 else 0.0
                )
                shown = draw_reel_detection(
                    canonical, display_latest, mirror=not args.no_mirror_display,
                )
                shown = draw_lip_markers(
                    shown, latest_lips, mirror=not args.no_mirror_display
                )
                display_size[:] = [shown.shape[1], shown.shape[0]]
                shown = hud.draw(
                    shown, latest, latest_result, lock, future is not None, active,
                    fps,
                    glosses + (["GOOD/THANKYOU?"] if pending_pair else []),
                    finishing_glosses,
                    (
                        f"Hold both open hands to finish  "
                        f"{round(100 * finish_gesture.progress):d}%"
                        if finish_gesture.active and not finish_requested
                        else sentence
                    ),
                    finish_requested or naturalizer_future is not None,
                    None if speaker is None else speaker.current_text,
                    (revisable_hypothesis if revisable_transcript else ctc_hypothesis)
                    or (last_stage2_hypothesis if stage2_review_only else []),
                    {
                        "dropped": dropped_camera_frames,
                        "observations": processed,
                        "sequence_review": stage2_review_only and not revisable_transcript,
                    },
                    revisable_transcript=revisable_transcript or window_backend,
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

        capture_elapsed = time.perf_counter() - wall_started
        while future is not None:
            collect(wait=True)
        if args.finish_at_eof:
            request_finish(last_source_seconds, "video_eof")
        if stage2_at_finish and not finish_requested:
            ctc_ready.clear()
        while True:
            service_language()
            if future is not None:
                collect(wait=True)
            elif ctc_future is not None:
                collect_ctc(wait=True)
            elif naturalizer_future is not None:
                naturalizer_future.result()
                service_language()
            elif not finish_requested and not ctc_ready:
                break
        service_speech()
    finally:
        if camera is None:
            capture.release()
        else:
            camera.close()
        lip_tracker.close()
        classifier_executor.shutdown(wait=True)
        ctc_executor.shutdown(wait=True)
        language_executor.shutdown(wait=True)
        if stage2_at_finish:
            ctc_ready.close()
        retained_frozen.close()
        recorder.data["capture_stats"] = {
            "frames_seen": frame_index,
            "landmark_observations": processed,
            "stale_camera_frames_dropped": dropped_camera_frames,
            "capture_elapsed_seconds": capture_elapsed,
            "revisable_transcript": revisable_transcript,
            "transcript_backend": getattr(args, "transcript_backend", "ctc"),
        }
        recorder.close()
        cv2.destroyAllWindows()

    expected = [value.upper() for value in args.expected_sequence]
    actual = list(finishing_glosses if args.finish_at_eof else glosses)
    summary = {
        "session": str(recorder.root),
        "history": str(recorder.history_path),
        "video": str(recorder.video_path),
        "predictions": len(results),
        "hypothesis": actual,
        "stage2_hypothesis": last_stage2_hypothesis,
        "expected": expected,
        "exact": actual == expected if expected else None,
        "stale_camera_frames_dropped": dropped_camera_frames,
        "utterances": len(recorder.data["utterances"]),
    }
    print(json.dumps(summary, indent=2))
    return summary


def parser() -> argparse.ArgumentParser:
    value = isolated_parser()
    value.description = __doc__
    value.set_defaults(
        mode="cascade",
        output_root=REPO / "artifacts/reports/live_reel_stage1_v17",
        orientation_coreml=DEFAULT_PHRASE_ADAPTED_COREML,
        unified_checkpoint=DEFAULT_UNIFIED_PHRASE_ADAPTED,
        stage1_coreml=DEFAULT_UNIFIED_PHRASE_ADAPTED_COREML,
        cascade_score=0.55,
        processing_fps=20.0,
        detection_image_side=640,
    )
    value.add_argument("--candidate-minimum-seconds", type=float, default=0.50)
    value.add_argument("--transcript-backend", choices=("ctc", "stage1-window"), default="ctc")
    value.add_argument("--stage1-window-checkpoint", type=Path)
    value.add_argument("--frame-timestamps", type=Path)
    value.add_argument("--exclude-final-frames", type=int, default=0)
    value.add_argument("--candidate-maximum-seconds", type=float, default=2.5)
    value.add_argument("--probe-interval-seconds", type=float, default=0.12)
    value.add_argument(
        "--realtime-video", action="store_true",
        help="pace a saved-video replay by its timestamps for live timing comparisons",
    )
    value.add_argument("--stability-hits", type=int, default=2)
    value.add_argument("--release-hits", type=int, default=1)
    value.add_argument("--commit-score", type=float, default=0.45)
    value.add_argument("--commit-hits", type=int, default=1)
    value.add_argument("--instant-commit-score", type=float, default=0.80)
    value.add_argument("--start-frames", type=int, default=1)
    value.add_argument("--preroll-seconds", type=float, default=0.20)
    value.add_argument("--transition-overlap-seconds", type=float, default=0.0)
    value.add_argument("--no-hand-release-seconds", type=float, default=0.20)
    value.add_argument(
        "--no-live-lips", action="store_true",
        help="hide the live lip overlay (the optional verifier can still use it)",
    )
    value.add_argument(
        "--lip-marker-model", type=Path, default=DEFAULT_LIP_MARKER_MODEL,
    )
    value.add_argument(
        "--lip-marker-minimum-confidence", type=float, default=0.999,
    )
    lips = value.add_mutually_exclusive_group()
    lips.add_argument(
        "--lip-marker-verifier", dest="no_lip_marker_verifier",
        action="store_false",
        help="opt in to the experimental consensus-only GOOD/THANKYOU tie-break",
    )
    lips.add_argument("--no-lip-marker-verifier", action="store_true")
    value.set_defaults(no_lip_marker_verifier=True)
    stage2 = value.add_mutually_exclusive_group()
    stage2.add_argument(
        "--stage2-arbiter", dest="no_stage2_arbiter", action="store_false"
    )
    stage2.add_argument("--no-stage2-arbiter", action="store_true")
    value.set_defaults(no_stage2_arbiter=True)
    value.add_argument("--verbose-predictions", action="store_true")
    value.add_argument(
        "--dense-model-auxiliary", action="store_true",
        help="experimentally feed every-frame face/body points into Stage 1",
    )
    value.add_argument(
        "--stage2-minimum-phrase-seconds", type=float, default=1.6,
    )
    value.add_argument(
        "--stage2-window-seconds", type=float, default=32.0 / 30.0,
    )
    value.add_argument("--stage2-encoder", type=Path, default=DEFAULT_ENCODER)
    value.add_argument("--stage2-primary", type=Path, default=DEFAULT_PRIMARY)
    value.add_argument("--stage2-specialist", type=Path, default=DEFAULT_SPECIALIST)
    value.add_argument("--stage2-selector", type=Path, default=DEFAULT_SELECTOR)
    value.add_argument("--stage2-other-preservation", type=Path)
    value.add_argument("--no-stage2-other-preservation", dest="stage2_other_preservation", action="store_const", const=None)
    value.add_argument(
        "--selector-report", type=Path, default=DEFAULT_SELECTOR_REPORT,
    )
    value.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    value.add_argument("--stage2-live-checkpoint", type=Path, default=None, help="experimental matched-input Stage2 adaptation")
    value.add_argument("--sequence-preview", action="store_true",
                       help="experimental continuous CTC preview with confirmed prefixes; uncertain endings remain for review")
    value.add_argument(
        "--revisable-transcript", action="store_true",
        help="experimental continuous CTC transcript; it may revise until Finish re-decodes retained visual features",
    )
    value.add_argument("--expected-sequence", nargs="*", default=())
    value.add_argument("--finish-at-eof", action="store_true")
    value.add_argument(
        "--finish-gesture-hold-seconds", type=float, default=0.4,
        help="seconds both open palms must remain raised before finishing",
    )
    value.add_argument(
        "--no-finish-gesture", action="store_true",
        help="disable the ten-finger finish gesture; button and F still work",
    )
    return value


def validate_args(args: argparse.Namespace) -> None:
    if args.transcript_backend == "stage1-window":
        if args.stage1_window_checkpoint is None:
            raise ValueError("stage1-window requires --stage1-window-checkpoint")
        if args.revisable_transcript or args.sequence_preview or args.stage2_live_checkpoint:
            raise ValueError("stage1-window cannot combine with CTC experimental flags")
    if args.exclude_final_frames < 0 or ((args.frame_timestamps or args.exclude_final_frames) and not args.video):
        raise ValueError("timestamp/exclusion options require a video and nonnegative exclusions")
    positive = (
        args.processing_fps, args.candidate_minimum_seconds,
        args.candidate_maximum_seconds, args.probe_interval_seconds,
        args.stability_hits, args.release_hits, args.start_frames,
        args.commit_hits,
        args.no_hand_release_seconds, args.preroll_seconds,
        args.stage2_minimum_phrase_seconds, args.stage2_window_seconds,
        args.finish_gesture_hold_seconds,
    )
    if min(positive) <= 0:
        raise ValueError("timing, frame, and stability values must be positive")
    if args.candidate_minimum_seconds >= args.candidate_maximum_seconds:
        raise ValueError("candidate minimum must be below candidate maximum")
    if not 0 <= args.transition_overlap_seconds < args.candidate_minimum_seconds:
        raise ValueError("transition overlap must be nonnegative and shorter than a candidate")
    if not 0.5 <= args.lip_marker_minimum_confidence <= 1.0:
        raise ValueError("lip marker minimum confidence must be between 0.5 and 1")
    if not 0.0 <= args.commit_score <= 1.0:
        raise ValueError("commit score must be between zero and one")
    if not args.commit_score <= args.instant_commit_score <= 1.0:
        raise ValueError("instant commit score must be at least the commit score")


def main() -> None:
    args = parser().parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
