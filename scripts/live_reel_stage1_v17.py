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
from active.v17.lip_marker_v17 import (
    LipMarkerDisambiguator,
    LipMarkerTracker,
    draw_lip_markers,
)
from active.v17.schema_v17 import V17Config
from scripts.live_isolated_v17 import (
    IsolatedClassifier,
    LiveSpeaker,
    ObservedFrame,
    SessionRecorder,
    atomic_json,
    clicked_control,
    draw_detection,
    make_naturalizer,
    landmarks_from_observations,
    observation_quality,
    parser as isolated_parser,
    trim_to_motion,
    utc_now,
    wrist_motion,
)
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
    roll_ctc_prefix,
)


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


class ReelCascadeClassifier:
    """Fast landmark proposal with a visual-hand verifier only when required."""

    def __init__(self, args: argparse.Namespace):
        import coremltools as ct

        fast_args = copy.copy(args)
        fast_args.mode = "fast"
        self.full = IsolatedClassifier(fast_args)
        self.args = args
        self.labels = self.full.labels
        self.orientation = ct.models.MLModel(
            str(args.orientation_coreml), compute_units=ct.ComputeUnit.ALL
        )
        self.orientation.predict({
            "landmarks": np.zeros((1, 32, 61, 5), np.float32),
        })

    def provenance(self) -> dict[str, object]:
        return {
            **self.full.provenance(),
            "orientation_coreml": {
                "path": str(self.args.orientation_coreml), "sha256": "mlpackage",
            },
        }

    def classify(self, observations: list[ObservedFrame]) -> dict[str, object]:
        """Use the landmark model first, then the full verifier on weak evidence."""
        started = time.perf_counter()
        original = observations
        trimmed, motion = trim_to_motion(observations, self.args.quiet_motion)
        try:
            features, diagnostics = landmarks_from_observations(trimmed, V17Config())
        except ValueError:
            return self.full.classify(original)
        raw = np.asarray(
            self.orientation.predict({"landmarks": features[None]})["var_5535"]
        ).reshape(len(self.labels))
        probability = np.exp(raw - raw.max())
        probability /= probability.sum()
        order = np.argsort(probability)[::-1][:3]
        score = float(probability[order[0]])
        if score < self.args.cascade_score:
            result = self.full.classify(original)
            result["mode"] = "reel-cascade-full-fallback"
            result["diagnostics"].update({
                "cascade_primary_score": score,
                "cascade_primary_gloss": self.labels[int(order[0])],
                "cascade_score_threshold": self.args.cascade_score,
                "cascade_fallback_used": True,
            })
            return result

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
        elapsed = 1000.0 * (time.perf_counter() - started)
        diagnostics.update(motion)
        diagnostics.update({
            "cascade_primary_score": score,
            "cascade_primary_gloss": top3[0]["gloss"],
            "cascade_score_threshold": self.args.cascade_score,
            "cascade_fallback_used": False,
            "mouth_pixels_used": False,
            "lip_markers_used_for_general_classification": False,
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
    """Resolve a proposed GOOD/THANKYOU inside that closed pair only."""
    if proposal not in {"GOOD", "THANKYOU"}:
        return result
    diagnostics = dict(result.get("diagnostics", {}))
    prediction = None if disambiguator is None else disambiguator.predict([
        getattr(item, "lip_points", None) for item in observations
    ])
    selected = proposal
    source = "landmark_proposal"
    if prediction is not None and float(prediction["confidence"]) >= minimum_confidence:
        selected = str(prediction["label"])
        source = "media_pipe_lip_markers"
    diagnostics.update({
        "mouth_pixels_used": False,
        "targeted_lip_verifier": "closed_good_thankyou_pair",
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


def draw_hud(
    frame: np.ndarray,
    latest: ObservedFrame | None,
    latest_result: dict[str, object] | None,
    lock: StableGlossLock,
    pending: bool,
    active: bool,
    fps: float,
    glosses: list[str],
    finishing_glosses: list[str],
    sentence: str,
    finish_pending: bool,
    speech_text: str | None,
    ctc_hypothesis: list[str] | None = None,
) -> np.ndarray:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (min(width, 690), 196), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    quality = "hand --  face --  motion --"
    if latest is not None:
        quality = (
            f"hand {latest.hand_quality:.2f}  face {latest.face_quality:.2f}  "
            f"motion {latest.motion:.3f}"
        )
    provisional = "--"
    evidence = ""
    if latest_result is not None:
        provisional = str(latest_result.get("candidate_gloss") or "UNKNOWN")
        evidence = (
            f"  evidence {float(latest_result.get('gate_score', 0.0)):.2f}"
            f"  margin {float(latest_result.get('gate_margin', 0.0)):.2f}"
        )
    lines = [
        f"{'VERIFYING' if pending else ('SIGNING' if active else 'READY')}   {fps:.1f} FPS",
        quality,
        f"provisional {provisional}{evidence}",
        f"stable {lock.hits}/{lock.required_hits}   last committed {lock.suppressed or '--'}",
        "sequence " + (
            " ".join(ctc_hypothesis) if ctc_hypothesis else "--"
        ),
        "No neutral pose required; F/FINISH closes the sentence",
    ]
    for index, text in enumerate(lines):
        cv2.putText(
            frame, text, (14, 28 + 31 * index), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (255, 255, 255), 1, cv2.LINE_AA,
        )

    panel_top = max(224, height - 102)
    cv2.rectangle(frame, (0, panel_top), (width, height), (20, 20, 20), -1)
    shown_glosses = finishing_glosses if finish_pending and finishing_glosses else glosses
    shown = "  ".join(shown_glosses) if shown_glosses else "(empty)"
    max_chars = max(18, (width - 36) // 11)
    if len(shown) > max_chars:
        shown = "..." + shown[-(max_chars - 3):]
    cv2.putText(
        frame, f"GLOSS BUFFER: {shown}", (14, panel_top + 29),
        cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 1, cv2.LINE_AA,
    )
    status = f"SPEAKING: {speech_text}" if speech_text else sentence
    if finish_pending:
        status = "Naturalizing finished glosses..."
    if status:
        cv2.putText(
            frame, status[:max_chars], (14, panel_top + 59),
            cv2.FONT_HERSHEY_SIMPLEX, 0.57, (225, 225, 225), 1, cv2.LINE_AA,
        )
    for action, (left, top, right, bottom) in {
        "reset": (14, height - 54, 134, height - 16),
        "finish": (146, height - 54, 286, height - 16),
    }.items():
        color = (55, 55, 155) if action == "reset" else (35, 145, 70)
        cv2.rectangle(frame, (left, top), (right, bottom), color, -1)
        cv2.rectangle(frame, (left, top), (right, bottom), (245, 245, 245), 1)
        cv2.putText(
            frame, action.upper(), (left + 18, top + 27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame


def run(args: argparse.Namespace) -> dict[str, object]:
    classifier = ReelCascadeClassifier(args)
    sequence_arbiter = None if args.no_stage2_arbiter else LiveStage2CTC(args)
    lip_disambiguator = (
        None if args.no_lip_marker_verifier
        else LipMarkerDisambiguator(args.lip_marker_model)
    )
    naturalizer = make_naturalizer(args)
    speaker = None if args.no_speech else LiveSpeaker()
    source = str(args.video) if args.video else f"camera:{args.camera}"
    recorder = SessionRecorder(args, classifier, source)
    recorder.data.update({
        "format": "slt_live_reel_stage1_v17_session",
        "version": 1,
        "prototype_scope": (
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
        "release_hits": args.release_hits,
        "transition_overlap_seconds": args.transition_overlap_seconds,
        "no_hand_release_seconds": args.no_hand_release_seconds,
        "capture_policy": "latest-frame webcam; sequential saved video",
        "live_face_display": "MediaPipe lips on every processed display frame",
        "lip_marker_model": None if lip_disambiguator is None else str(
            lip_disambiguator.path
        ),
        "lip_marker_minimum_confidence": args.lip_marker_minimum_confidence,
        "stage2_sequence_arbiter": sequence_arbiter is not None,
        "stage2_minimum_phrase_seconds": args.stage2_minimum_phrase_seconds,
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
    detector = AppleVisionDetector(args.minimum_point_confidence)
    lip_tracker = LipMarkerTracker(enabled=lip_disambiguator is not None or (
        not args.no_display and not args.no_live_lips
    ))
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
    future_clip: list[ObservedFrame] | None = None
    future_epoch = epoch = 0
    naturalizer_future: Future | None = None
    naturalizer_context: dict[str, object] | None = None
    pending_speech: deque[dict[str, object]] = deque()
    results: list[dict[str, object]] = []
    glosses: list[str] = []
    pending_pair: dict[str, object] | None = None
    ctc_buffer = ElapsedWindowBuffer(args.stage2_window_seconds)
    ctc_ready: deque[list[ObservedFrame]] = deque()
    ctc_future: Future | None = None
    ctc_future_epoch = 0
    ctc_prior: list[np.ndarray] = []
    ctc_locked: list[str] = []
    ctc_context_hypothesis: list[str] = []
    ctc_context_positions: list[int] = []
    ctc_hypothesis: list[str] = []
    last_stage2_hypothesis: list[str] = []
    ctc_results: list[dict[str, object]] = []
    activity_first_seconds: float | None = None
    activity_last_seconds: float | None = None
    finishing_glosses: list[str] = []
    sentence = ""
    finish_requested = False
    active = False
    activity_hits = 0
    no_hand_since: float | None = None
    latest: ObservedFrame | None = None
    latest_lips: np.ndarray | None = None
    latest_result: dict[str, object] | None = None
    next_process = next_probe = 0.0
    frame_index = processed = 0
    wall_started = time.perf_counter()
    camera = None if args.video else LatestCamera(capture, wall_started)
    camera_sequence = dropped_camera_frames = 0
    display_times: deque[float] = deque(maxlen=30)
    ui_actions: deque[tuple[str, float]] = deque()
    display_size = [1280, 720]
    warm_logged = False

    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME)

        def on_mouse(event, x, y, _flags, _parameter):
            if event != cv2.EVENT_LBUTTONUP:
                return
            action = clicked_control(x, y, display_size[0], display_size[1])
            if action is not None:
                ui_actions.append((action, time.perf_counter() - wall_started))

        cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    def clear_candidate(*, keep_after: float | None = None) -> None:
        nonlocal active, activity_hits, next_probe
        retained = [] if keep_after is None else [
            item for item in observations if item.seconds >= keep_after
        ]
        observations.clear()
        observations.extend(retained)
        active = bool(retained)
        activity_hits = 0
        next_probe = retained[-1].seconds if retained else 0.0

    def submit() -> None:
        nonlocal future, future_clip, future_epoch, next_probe
        if future is not None or not active or finish_requested:
            return
        clip = list(observations)
        if len(clip) < 4:
            return
        duration = clip[-1].seconds - clip[0].seconds
        if duration < args.candidate_minimum_seconds or clip[-1].seconds < next_probe:
            return
        future_epoch = epoch
        future_clip = clip
        future = classifier_executor.submit(classifier.classify, clip)
        next_probe = clip[-1].seconds + args.probe_interval_seconds

    def submit_ctc() -> None:
        nonlocal ctc_future, ctc_future_epoch
        nonlocal ctc_locked, ctc_context_hypothesis, ctc_context_positions
        if sequence_arbiter is None or ctc_future is not None or not ctc_ready:
            return
        if len(ctc_prior) >= MAXIMUM_WINDOWS:
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
        if result.get("accepted") and not ignored:
            ctc_context_hypothesis = list(result["hypothesis"])
            ctc_context_positions = list(result["token_positions"])
            ctc_hypothesis = [*ctc_locked, *ctc_context_hypothesis]
            result["context_hypothesis"] = list(ctc_context_hypothesis)
            result["locked_prefix"] = list(ctc_locked)
            result["hypothesis"] = list(ctc_hypothesis)
        ctc_results.append(result)
        recorder.add_event({
            "type": "stage2_sequence_update",
            "hypothesis": list(ctc_hypothesis),
            "accepted": bool(result.get("accepted")),
            "ignored_after_reset": ignored,
            "window_count": result.get("window_count"),
            "latency_ms": result.get("latency_ms"),
        })
        submit_ctc()

    def collect(wait: bool = False) -> None:
        nonlocal future, future_clip, latest_result, pending_pair
        if future is None or (not wait and not future.done()):
            return
        result = future.result()
        ignored = future_epoch != epoch
        proposal = None if ignored else lock.update(result)
        emitted = None
        verifier = None
        if proposal is not None and future_clip is not None:
            verifier = classifier.verify(future_clip)
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
        result.update({
            "reel_epoch": future_epoch,
            "ignored_after_reset": ignored,
            "stable_candidate": lock.candidate,
            "stable_hits": lock.hits,
            "lock_proposal": proposal,
            "full_verifier": verifier,
            "committed_gloss": emitted,
        })
        results.append(result)
        recorder.add(result)
        latest_result = result
        print(json.dumps(result, indent=2))
        future = None
        future_clip = None
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
            else:
                clear_candidate()

    def service_language() -> None:
        nonlocal warm_logged, naturalizer_future, naturalizer_context, sentence
        nonlocal finish_requested, pending_pair
        nonlocal ctc_hypothesis, last_stage2_hypothesis
        nonlocal ctc_context_hypothesis, ctc_context_positions
        nonlocal activity_first_seconds, activity_last_seconds
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
            finished, selection_mode = select_finished_sequence(
                stage1_finished, last_stage2_hypothesis, active_seconds,
                args.stage2_minimum_phrase_seconds,
                stage2_stable=stage2_stable,
            )
            recorder.add_event({
                "type": "finished_sequence_selected",
                "stage1": stage1_finished,
                "stage2": last_stage2_hypothesis,
                "selected": finished,
                "selection_mode": selection_mode,
                "active_seconds": active_seconds,
                "stage2_stable": stage2_stable,
            })
            finishing_glosses[:] = finished
            glosses.clear()
            lock.reset()
            commit_lock.reset()
            clear_candidate()
            ctc_buffer.reset()
            ctc_prior.clear()
            ctc_locked.clear()
            ctc_context_hypothesis = []
            ctc_context_positions = []
            ctc_hypothesis = []
            ctc_results.clear()
            activity_first_seconds = None
            activity_last_seconds = None
            if not finished:
                sentence = "No recognized signs to finish."
                recorder.add_event({"type": "finish_empty"})
            else:
                naturalizer_context = {
                    "utterance_id": f"reel-{len(recorder.data['utterances']) + 1:04d}",
                    "requested_utc": utc_now(),
                    "glosses": finished,
                }
                naturalizer_future = language_executor.submit(
                    naturalizer.rephrase, finished
                )
        if naturalizer_future is not None and naturalizer_future.done():
            value = naturalizer_future.result()
            utterance = {
                **(naturalizer_context or {}), **value, "completed_utc": utc_now()
            }
            recorder.add_utterance(utterance)
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
            speaker.enqueue(str(item["text"]), str(item["kind"]), item["reference"])
            recorder.add_event({"type": "speech_queued", **item})
        started = speaker.update()
        if started is not None:
            recorder.add_event({"type": "speech_started", **started})

    def reset_display(seconds: float, source_name: str) -> None:
        nonlocal epoch, sentence, finish_requested, pending_pair
        nonlocal activity_first_seconds, activity_last_seconds, ctc_hypothesis
        nonlocal ctc_context_hypothesis, ctc_context_positions
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
        ctc_context_hypothesis = []
        ctc_context_positions = []
        ctc_hypothesis = []
        ctc_results.clear()
        activity_first_seconds = None
        activity_last_seconds = None
        sentence = ""
        finish_requested = False
        pending_speech.clear()
        if speaker is not None:
            speaker.clear()
        recorder.add_event({
            "type": "reset", "source": source_name, "seconds": seconds,
            "cleared_glosses": cleared, "classifier_pending": future is not None,
        })

    def request_finish(seconds: float, source_name: str) -> None:
        nonlocal finish_requested
        if finish_requested:
            return
        finish_requested = True
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
            collect_ctc()
            service_language()
            if not finish_requested and seconds + 1e-6 >= next_process:
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
                    frame, detection, assigned, seconds,
                    wrist_motion(assigned, previous_wrists),
                    *observation_quality(detection),
                    face_for_features=feature_frame,
                )
                latest.lip_points = latest_lips
                observations.append(latest)
                if sequence_arbiter is not None:
                    ctc_window = ctc_buffer.add(latest)
                    if ctc_window is not None:
                        ctc_ready.append(ctc_window)
                        submit_ctc()
                processed += 1
                next_process = max(next_process + 1.0 / args.processing_fps, seconds)
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
                    and future is None
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
                shown = draw_detection(
                    canonical, latest, mirror=not args.no_mirror_display
                )
                shown = draw_lip_markers(
                    shown, latest_lips, mirror=not args.no_mirror_display
                )
                display_size[:] = [shown.shape[1], shown.shape[0]]
                shown = draw_hud(
                    shown, latest, latest_result, lock, future is not None, active,
                    fps,
                    glosses + (["GOOD/THANKYOU?"] if pending_pair else []),
                    finishing_glosses, sentence,
                    finish_requested or naturalizer_future is not None,
                    None if speaker is None else speaker.current_text,
                    ctc_hypothesis,
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

        if future is not None:
            collect(wait=True)
        if args.finish_at_eof:
            request_finish(frame_index / source_fps, "video_eof")
        while ctc_future is not None or ctc_ready:
            collect_ctc(wait=True)
        service_language()
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
        ctc_executor.shutdown(wait=True)
        language_executor.shutdown(wait=True)
        recorder.data["capture_stats"] = {
            "frames_seen": frame_index,
            "landmark_observations": processed,
            "stale_camera_frames_dropped": dropped_camera_frames,
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
    )
    value.add_argument("--candidate-minimum-seconds", type=float, default=0.62)
    value.add_argument("--candidate-maximum-seconds", type=float, default=2.5)
    value.add_argument("--probe-interval-seconds", type=float, default=0.14)
    value.add_argument("--stability-hits", type=int, default=1)
    value.add_argument("--release-hits", type=int, default=1)
    value.add_argument("--commit-score", type=float, default=0.45)
    value.add_argument("--commit-hits", type=int, default=2)
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
        "--lip-marker-minimum-confidence", type=float, default=0.85,
    )
    value.add_argument("--no-lip-marker-verifier", action="store_true")
    value.add_argument("--no-stage2-arbiter", action="store_true")
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
    value.add_argument(
        "--selector-report", type=Path, default=DEFAULT_SELECTOR_REPORT,
    )
    value.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    value.add_argument("--expected-sequence", nargs="*", default=())
    value.add_argument("--finish-at-eof", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    positive = (
        args.processing_fps, args.candidate_minimum_seconds,
        args.candidate_maximum_seconds, args.probe_interval_seconds,
        args.stability_hits, args.release_hits, args.start_frames,
        args.commit_hits,
        args.no_hand_release_seconds, args.preroll_seconds,
        args.stage2_minimum_phrase_seconds, args.stage2_window_seconds,
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
    run(args)


if __name__ == "__main__":
    main()
