#!/usr/bin/env python3
"""Live/offline Apple-Vision diagnostic for the locked v17 100-sign vocabulary.

This is deliberately an isolated-sign prototype.  A short neutral/rest pause closes
each clip; it is not a continuous Stage-2 decoder.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from urllib import request as urllib_request

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from active.v17.extract_hand_rgb_v17 import crop_square, hand_box, union_box
from active.v17.extract_v17 import (
    AppleVisionDetector,
    FrameDetection,
    HandDetection,
    assign_hands,
    limit_image_side,
    orient_frame,
)
from active.v17.extract_visual_speech_v17 import aligned_views, motion_interval
from active.v17.geometry_v17 import (
    body_relative_normalize,
    image_normalized_to_isotropic,
    interpolate_short_gaps,
    resample_features,
)
from active.v17.schema_hand_rgb_v17 import HandRGBV17Config
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
from active.v17.schema_visual_speech_v17 import VisualSpeechV17Config
from active.v17.stage3_mobile_naturalizer_v17 import literal_render


DEFAULT_UNIFIED = REPO / "artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"
DEFAULT_LANDMARK = REPO / "artifacts/models/stage1_v17_citizen_semlex_full_clean_balanced/best_model.pth"
DEFAULT_HAND = REPO / "artifacts/models/stage1_v17_hand_mobileclip2_multisource_balanced/best_model.pth"
DEFAULT_MOUTH = REPO / "artifacts/models/stage1_v17_visual_speech_auto_avsr_mouth_frozen/best_model.pth"
DEFAULT_LOWER = REPO / "artifacts/models/stage1_v17_visual_speech_auto_avsr_lower_face_frozen/best_model.pth"
DEFAULT_IMAGE_ENCODER = REPO / "artifacts/coreml/MobileCLIP2S0ImageEncoderV17FP32.mlpackage"
DEFAULT_STAGE1_COREML = REPO / "artifacts/coreml/Stage1UnifiedMultimodalV17FP32.mlpackage"
DEFAULT_ORIENTATION_COREML = REPO / "artifacts/coreml/Stage1OrientationV17.mlpackage"
CASCADE_SCORE_THRESHOLD = 0.70
VISUAL_TIE_BREAK_GLOSSES = frozenset(("GOOD", "THANKYOU"))
LANDMARK_TIE_BREAK_GLOSSES = frozenset(("YOU", "NEED"))
MODE_BUTTONS = {
    "hybrid": (330, 76, 454, 108),
    "cascade": (466, 76, 606, 108),
}
WINDOW_NAME = "SLT v17 isolated live diagnostic"
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_NATURALIZER_MANIFEST = (
    REPO / "active/v17/stage3_mobile_naturalizer_manifest_v17.json"
)
DEFAULT_STAGE3_TINY = (
    REPO / "artifacts/models/stage3_v17_t5_efficient_tiny_locked100_v1"
)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)
BODY_CONNECTIONS = ((0, 1), (0, 2), (1, 3))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OllamaNaturalizer:
    """Bounded local rephrasing with a deterministic meaning-preserving fallback."""

    def __init__(self, args: argparse.Namespace):
        self.model = args.ollama_model
        self.url = args.ollama_url
        self.timeout = args.ollama_timeout
        self.enabled = not args.no_ollama
        self.keep_alive = "30m"
        self.manifest = json.loads(DEFAULT_NATURALIZER_MANIFEST.read_text(encoding="utf-8"))
        self.templates = {
            tuple(row["glosses"]): str(row["english"])
            for row in self.manifest["reviewed_templates"]
        }

    def _fallback(self, glosses: list[str]) -> tuple[str, str]:
        template = self.templates.get(tuple(glosses))
        if template is not None:
            return template, "reviewed_template"
        return literal_render(glosses, self.manifest), "literal_fallback"

    def _call(self, prompt: str) -> dict[str, object]:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_ctx": 512, "num_predict": 120},
        }).encode("utf-8")
        request = urllib_request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib_request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def warm(self) -> dict[str, object]:
        """Load the small model before FINISH without blocking camera startup."""
        started = time.perf_counter()
        if not self.enabled:
            return {"ok": False, "disabled": True, "latency_ms": 0.0}
        try:
            response = self._call(
                'Return only JSON: {"sentence":"Ready.","used_glosses":[]}'
            )
            return {
                "ok": True,
                "latency_ms": 1000 * (time.perf_counter() - started),
                "load_duration_ns": response.get("load_duration"),
            }
        except Exception as exc:
            return {
                "ok": False,
                "latency_ms": 1000 * (time.perf_counter() - started),
                "error": f"{type(exc).__name__}: {exc}",
            }

    def rephrase(self, glosses: list[str]) -> dict[str, object]:
        started = time.perf_counter()
        fallback, fallback_mode = self._fallback(glosses)
        prompt = (
            "Convert this recognized ASL gloss sequence into one short, natural English "
            "sentence. Preserve its meaning and order. Do not add people, objects, actions, "
            "negation, time, or facts not represented by the glosses. Use every input gloss "
            "exactly once in used_glosses and in the same order. Return only JSON with keys "
            'sentence and used_glosses. Recognized glosses: '
            + json.dumps(glosses)
        )
        result: dict[str, object] = {
            "model": self.model,
            "url": self.url,
            "prompt": prompt,
            "input_glosses": list(glosses),
            "literal_sentence": literal_render(glosses, self.manifest),
            "fallback_sentence": fallback,
            "fallback_mode": fallback_mode,
        }
        if not self.enabled:
            result.update({
                "sentence": fallback,
                "rendering_mode": fallback_mode,
                "safe_fallback_used": True,
                "error": "ollama_disabled",
                "latency_ms": 1000 * (time.perf_counter() - started),
            })
            return result
        raw_response = ""
        try:
            response = self._call(prompt)
            raw_response = str(response.get("response", ""))
            parsed = json.loads(raw_response)
            sentence = " ".join(str(parsed.get("sentence", "")).split())
            used_glosses = parsed.get("used_glosses")
            if used_glosses != glosses:
                raise ValueError("Ollama changed, omitted, duplicated, or reordered glosses")
            if not sentence or len(sentence) > 300:
                raise ValueError("Ollama returned an empty or overlong sentence")
            result.update({
                "sentence": sentence,
                "used_glosses": used_glosses,
                "raw_response": raw_response,
                "rendering_mode": "ollama",
                "safe_fallback_used": False,
                "ollama_total_duration_ns": response.get("total_duration"),
                "ollama_load_duration_ns": response.get("load_duration"),
                "ollama_prompt_eval_count": response.get("prompt_eval_count"),
                "ollama_eval_count": response.get("eval_count"),
            })
        except Exception as exc:
            result.update({
                "sentence": fallback,
                "rendering_mode": fallback_mode,
                "safe_fallback_used": True,
                "raw_response": raw_response,
                "error": f"{type(exc).__name__}: {exc}",
            })
        result["latency_ms"] = 1000 * (time.perf_counter() - started)
        return result


class TinyStage3Naturalizer:
    """Local 15.6M T5 renderer with reviewed-template and literal fallbacks."""

    def __init__(self, args: argparse.Namespace):
        self.checkpoint = args.stage3_checkpoint
        self.device_name = args.stage3_device
        self.manifest = json.loads(DEFAULT_NATURALIZER_MANIFEST.read_text(encoding="utf-8"))
        self.templates = {
            tuple(row["glosses"]): str(row["english"])
            for row in self.manifest["reviewed_templates"]
        }
        self.model = None
        self.tokenizer = None
        self.device = None

    def _fallback(self, glosses: list[str]) -> tuple[str, str]:
        template = self.templates.get(tuple(glosses))
        if template is not None:
            return template, "reviewed_template"
        return literal_render(glosses, self.manifest), "literal_fallback"

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        name = self.device_name
        if name == "auto":
            name = "mps" if torch.backends.mps.is_available() else "cpu"
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable for the Stage-3 naturalizer")
        self.device = torch.device(name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, local_files_only=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.checkpoint, local_files_only=True
        ).to(self.device).eval()

    def _generate(self, glosses: list[str]) -> str:
        import torch

        self._ensure_loaded()
        encoded = self.tokenizer(
            " ".join(glosses).lower(), return_tensors="pt", truncation=True,
            max_length=48,
        ).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **encoded, max_new_tokens=48, num_beams=1, do_sample=False
            )
        return self.tokenizer.decode(output[0].cpu(), skip_special_tokens=True).strip()

    def warm(self) -> dict[str, object]:
        started = time.perf_counter()
        try:
            self._generate(["HELLO"])
            return {
                "ok": True,
                "naturalizer": "t5_efficient_tiny",
                "checkpoint": str(self.checkpoint),
                "device": str(self.device),
                "latency_ms": 1000 * (time.perf_counter() - started),
            }
        except Exception as exc:
            return {
                "ok": False,
                "naturalizer": "t5_efficient_tiny",
                "latency_ms": 1000 * (time.perf_counter() - started),
                "error": f"{type(exc).__name__}: {exc}",
            }

    def rephrase(self, glosses: list[str]) -> dict[str, object]:
        started = time.perf_counter()
        fallback, fallback_mode = self._fallback(glosses)
        result: dict[str, object] = {
            "model": str(self.checkpoint),
            "input_glosses": list(glosses),
            "literal_sentence": literal_render(glosses, self.manifest),
            "fallback_sentence": fallback,
            "fallback_mode": fallback_mode,
        }
        # Human-reviewed exact phrases are both safer and faster than generation.
        if fallback_mode == "reviewed_template":
            result.update({
                "sentence": fallback,
                "rendering_mode": fallback_mode,
                "safe_fallback_used": False,
            })
        else:
            try:
                sentence = " ".join(self._generate(glosses).split())
                if not sentence or len(sentence) > 300:
                    raise ValueError("tiny Stage-3 model returned an empty or overlong sentence")
                result.update({
                    "sentence": sentence,
                    "rendering_mode": "t5_efficient_tiny",
                    "safe_fallback_used": False,
                })
            except Exception as exc:
                result.update({
                    "sentence": fallback,
                    "rendering_mode": fallback_mode,
                    "safe_fallback_used": True,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        result["latency_ms"] = 1000 * (time.perf_counter() - started)
        return result


def make_naturalizer(args: argparse.Namespace):
    if args.naturalizer == "tiny":
        return TinyStage3Naturalizer(args)
    if args.naturalizer == "ollama":
        return OllamaNaturalizer(args)
    disabled = argparse.Namespace(**vars(args))
    disabled.no_ollama = True
    return OllamaNaturalizer(disabled)


@dataclass
class ObservedFrame:
    frame: np.ndarray
    detection: FrameDetection
    assigned: dict[str, HandDetection | None]
    seconds: float
    motion: float
    hand_quality: float
    face_quality: float
    face_for_features: bool = True


@dataclass(frozen=True)
class BoundaryConfig:
    processing_fps: float = 30.0
    preroll_seconds: float = 0.35
    start_motion: float = 0.012
    start_frames: int = 2
    quiet_motion: float = 0.006
    neutral_radius: float = 0.08
    quiet_seconds: float = 0.20
    minimum_sign_seconds: float = 0.45
    maximum_sign_seconds: float = 4.0
    cooldown_seconds: float = 0.35
    require_neutral: bool = True


class AutoBoundary:
    """Small motion/rest state machine for the pause-delimited prototype."""

    def __init__(self, config: BoundaryConfig):
        self.config = config
        self.preroll = deque(maxlen=max(2, math.ceil(config.preroll_seconds * config.processing_fps)))
        self.state = "WAITING"
        self.active_frames = 0
        self.quiet_frames = 0
        self.clip: list[ObservedFrame] = []
        self.cooldown_until = 0.0
        self.neutral_wrists: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.sign_neutral: dict[str, np.ndarray | None] = {"left": None, "right": None}

    @property
    def progress(self) -> float:
        if self.state == "WAITING":
            return min(1.0, self.active_frames / self.config.start_frames)
        if self.state == "SIGNING":
            target = max(1, math.ceil(self.config.quiet_seconds * self.config.processing_fps))
            return min(1.0, self.quiet_frames / target)
        return 0.0

    def reset(self) -> None:
        self.preroll.clear()
        self.state = "WAITING"
        self.active_frames = 0
        self.quiet_frames = 0
        self.clip = []
        self.cooldown_until = 0.0
        self.sign_neutral = {"left": None, "right": None}

    def _update_neutral(self, observation: ObservedFrame) -> None:
        if observation.motion > self.config.quiet_motion:
            return
        for side in ("left", "right"):
            hand = observation.assigned[side]
            if hand is None or hand.confidence[0] <= 0:
                continue
            wrist = hand.xy[0]
            previous = self.neutral_wrists[side]
            self.neutral_wrists[side] = wrist.copy() if previous is None else (
                0.8 * previous + 0.2 * wrist
            )

    def _at_neutral(self, observation: ObservedFrame) -> bool:
        if not self.config.require_neutral:
            return True
        current = {
            side: observation.assigned[side].xy[0]
            for side in ("left", "right")
            if observation.assigned[side] is not None
            and observation.assigned[side].confidence[0] > 0
        }
        if not current:
            return True
        distances = [
            float(np.linalg.norm(wrist - self.sign_neutral[side]))
            for side, wrist in current.items() if self.sign_neutral[side] is not None
        ]
        return bool(distances) and max(distances) <= self.config.neutral_radius

    def update(self, observation: ObservedFrame) -> list[ObservedFrame] | None:
        if self.state == "COOLDOWN":
            if observation.seconds >= self.cooldown_until:
                self.reset()
            else:
                return None

        if self.state == "WAITING":
            self.preroll.append(observation)
            self._update_neutral(observation)
            moving = observation.hand_quality > 0 and observation.motion >= self.config.start_motion
            self.active_frames = self.active_frames + 1 if moving else 0
            if self.active_frames >= self.config.start_frames:
                self.state = "SIGNING"
                self.clip = list(self.preroll)
                self.quiet_frames = 0
                self.sign_neutral = {
                    side: None if wrist is None else wrist.copy()
                    for side, wrist in self.neutral_wrists.items()
                }
            return None

        self.clip.append(observation)
        duration = self.clip[-1].seconds - self.clip[0].seconds
        quiet = (
            observation.motion <= self.config.quiet_motion
            and self._at_neutral(observation)
        )
        self.quiet_frames = self.quiet_frames + 1 if quiet else 0
        quiet_target = max(1, math.ceil(self.config.quiet_seconds * self.config.processing_fps))
        finished = (
            duration >= self.config.maximum_sign_seconds
            or (duration >= self.config.minimum_sign_seconds and self.quiet_frames >= quiet_target)
        )
        if not finished:
            return None
        result = self.clip
        self.clip = []
        self.state = "COOLDOWN"
        self.cooldown_until = observation.seconds + self.config.cooldown_seconds
        return result

    def finish(self, fallback: list[ObservedFrame]) -> list[ObservedFrame] | None:
        if len(self.clip) >= 4:
            result, self.clip = self.clip, []
            return result
        hand_frames = sum(item.hand_quality > 0 for item in fallback)
        return fallback if len(fallback) >= 4 and hand_frames >= 2 else None


def observation_quality(detection: FrameDetection) -> tuple[float, float]:
    hand_values = [hand.score for hand in detection.hands if hand.score > 0]
    hand = float(np.mean(hand_values)) if hand_values else 0.0
    valid_face = detection.face_confidence > 0
    face = float(detection.face_confidence[valid_face].mean()) if valid_face.any() else 0.0
    return hand, face


def wrist_motion(
    assigned: dict[str, HandDetection | None], previous: dict[str, np.ndarray | None]
) -> float:
    values = []
    for side in ("left", "right"):
        hand = assigned[side]
        if hand is None or hand.confidence[0] <= 0:
            continue
        wrist = hand.xy[0]
        values.append(
            0.03 if previous[side] is None else float(np.linalg.norm(wrist - previous[side]))
        )
        previous[side] = wrist.copy()
    return max(values, default=0.0)


def landmarks_from_observations(
    observations: list[ObservedFrame], config: V17Config | None = None
) -> tuple[np.ndarray, dict[str, object]]:
    """Build exact v17 features from the already-computed live Vision pass."""
    config = config or V17Config()
    if len(observations) < 4:
        raise ValueError("an isolated clip needs at least four processed frames")
    height, width = observations[0].frame.shape[:2]
    if any(item.frame.shape[:2] != (height, width) for item in observations):
        raise ValueError("processed frame dimensions changed within a sign")
    count = len(observations)
    xy = np.zeros((count, NUM_NODES, 2), np.float32)
    confidence = np.zeros((count, NUM_NODES), np.float32)
    for index, item in enumerate(observations):
        for slot, start, end in (
            ("left", LHAND_START, LHAND_END), ("right", RHAND_START, RHAND_END)
        ):
            hand = item.assigned[slot]
            if hand is not None:
                xy[index, start:end] = hand.xy
                confidence[index, start:end] = hand.confidence
        # Requests were already sparsely scheduled in the global camera stream.
        # Filtering again by clip-local index discarded valid auxiliary detections
        # whenever a sign started off the global interval boundary.
        if (item.detection.body_confidence > 0).any():
            xy[index, BODY_START:BODY_END] = item.detection.body_xy
            confidence[index, BODY_START:BODY_END] = item.detection.body_confidence
        if (
            config.include_face
            and item.face_for_features
            and (item.detection.face_confidence > 0).any()
        ):
            xy[index, FACE_START:FACE_END] = item.detection.face_xy
            confidence[index, FACE_START:FACE_END] = item.detection.face_confidence

    active = np.flatnonzero((confidence[:, :RHAND_END] > 0).sum(axis=1) >= 5)
    if len(active) < config.minimum_detected_hand_frames:
        raise ValueError("not enough detected hand frames")
    trim_start = max(0, int(active[0]) - config.trim_context_frames)
    trim_end = min(count, int(active[-1]) + config.trim_context_frames + 1)
    xy, confidence = xy[trim_start:trim_end], confidence[trim_start:trim_end]
    valid = confidence > 0
    xy = image_normalized_to_isotropic(xy, width, height, valid)
    xy[:, :RHAND_END], confidence[:, :RHAND_END] = interpolate_short_gaps(
        xy[:, :RHAND_END], confidence[:, :RHAND_END], config.hand_gap_frames
    )
    xy[:, FACE_START:], confidence[:, FACE_START:] = interpolate_short_gaps(
        xy[:, FACE_START:], confidence[:, FACE_START:], config.auxiliary_gap_frames
    )
    normalized_xy, depth, normalization = body_relative_normalize(xy, confidence)
    features = np.zeros((len(xy), NUM_NODES, NUM_CHANNELS), np.float32)
    features[..., :2] = normalized_xy
    features[..., 2] = depth
    features[..., 3] = confidence > 0
    features[..., 4] = np.clip(confidence, 0, 1)
    features = resample_features(features, config.target_frames).astype(np.float32)
    diagnostics = {
        **normalization,
        "source_frames": count,
        "trim_start": trim_start,
        "trim_end_exclusive": trim_end,
        "hand_frame_fraction": float((confidence[:, :RHAND_END] > 0).any(axis=1).mean()),
        "face_presence_fraction": float(features[:, FACE_START:FACE_END, 3].mean()),
        "body_presence_fraction": float(features[:, BODY_START:BODY_END, 3].mean()),
    }
    return features, diagnostics


def trim_to_motion(
    observations: list[ObservedFrame], threshold: float, context_frames: int = 2,
) -> tuple[list[ObservedFrame], dict[str, object]]:
    """Remove live neutral padding while retaining the complete moving interval."""
    active = np.flatnonzero(np.asarray([item.motion for item in observations]) > threshold)
    if len(active) < 2:
        return observations, {
            "motion_trim_start": 0,
            "motion_trim_end_exclusive": len(observations),
            "motion_trim_fallback": True,
        }
    start = max(0, int(active[0]) - context_frames)
    end = min(len(observations), int(active[-1]) + context_frames + 1)
    if end - start < 4:
        return observations, {
            "motion_trim_start": 0,
            "motion_trim_end_exclusive": len(observations),
            "motion_trim_fallback": True,
        }
    return observations[start:end], {
        "motion_trim_start": start,
        "motion_trim_end_exclusive": end,
        "motion_trim_fallback": False,
    }


def hand_inputs(
    observations: list[ObservedFrame], trim_start: int, trim_end: int
) -> tuple[list[list[np.ndarray | None]], np.ndarray, np.ndarray]:
    config = HandRGBV17Config()
    active = observations[trim_start:trim_end]
    positions = np.rint(np.linspace(0, len(active) - 1, config.sequence_length)).astype(int)
    crops: list[list[np.ndarray | None]] = []
    boxes = np.zeros((config.sequence_length, 3, 4), np.float32)
    valid = np.zeros((config.sequence_length, 3), np.float32)
    for output_index, position in enumerate(positions):
        item = active[int(position)]
        frame = item.frame
        height, width = frame.shape[:2]
        observed_boxes = []
        frame_crops: list[np.ndarray | None] = []
        for view, side in enumerate(("left", "right")):
            hand = item.assigned[side]
            box = None if hand is None else hand_box(hand, width, height, config)
            if box is None:
                frame_crops.append(None)
                continue
            boxes[output_index, view] = box / (width, height, width, height)
            valid[output_index, view] = 1
            observed_boxes.append(box)
            frame_crops.append(crop_square(frame, box, config.crop_size))
        combined = union_box(observed_boxes, width, height, config.union_box_scale)
        if combined is None:
            frame_crops.append(None)
        else:
            boxes[output_index, 2] = combined / (width, height, width, height)
            valid[output_index, 2] = 1
            frame_crops.append(crop_square(frame, combined, config.crop_size))
        crops.append(frame_crops)
    return crops, valid, boxes


def visual_inputs(
    observations: list[ObservedFrame], device
) -> tuple[object, object, object, object, dict[str, float]]:
    import torch
    from active.v17.train_stage_1_visual_speech_v17 import prepare_pixels

    config = VisualSpeechV17Config()
    views = []
    shapes = np.zeros((len(observations), 2), np.float32)
    detected = np.zeros(len(observations), np.bool_)
    raw_detected = np.asarray([
        bool((item.detection.face_confidence > 0).any()) for item in observations
    ])
    detected_positions = np.flatnonzero(raw_detected)
    for index, item in enumerate(observations):
        detection = item.detection
        # Face requests are intentionally sparse for live speed. Nearest detected
        # landmarks align each real RGB crop; the mouth pixels themselves are never
        # copied or synthesized.
        if not raw_detected[index] and len(detected_positions):
            nearest = int(detected_positions[np.abs(detected_positions - index).argmin()])
            detection = observations[nearest].detection
        crops, _, shape = aligned_views(
            item.frame, detection.face_xy, detection.face_confidence, config
        )
        views.append(crops)
        if shape is not None:
            shapes[index] = shape
            detected[index] = True
    start, end, _ = motion_interval(shapes, detected, config)
    positions = np.rint(np.linspace(start, max(start, end - 1), config.sequence_length)).astype(int)

    output = []
    validity = []
    for view_index in (0, 1):
        pixels = np.zeros((config.sequence_length, config.crop_size, config.crop_size, 3), np.uint8)
        valid = np.zeros(config.sequence_length, np.bool_)
        for frame_index, position in enumerate(positions):
            crop = views[int(position)][view_index]
            if crop is not None:
                pixels[frame_index] = crop
                valid[frame_index] = True
        tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        valid_tensor = torch.from_numpy(valid).unsqueeze(0).to(device)
        tensor, valid_tensor = prepare_pixels(tensor, valid_tensor, False)
        output.append(tensor)
        validity.append(valid_tensor)
    return output[0], validity[0], output[1], validity[1], {
        "reference_face_fraction": float(raw_detected.mean()),
        "mouth_valid_fraction": float(validity[0].float().mean().cpu()),
        "lower_face_valid_fraction": float(validity[1].float().mean().cpu()),
    }


class IsolatedClassifier:
    def __init__(self, args: argparse.Namespace):
        import coremltools as ct
        import torch

        self.args = args
        self.mode = args.mode
        self.torch = torch
        self.image_encoder = ct.models.MLModel(
            str(args.image_encoder), compute_units=ct.ComputeUnit.ALL
        )
        checkpoint = torch.load(args.unified_checkpoint, map_location="cpu", weights_only=False)
        self.labels = [
            label for label, _ in sorted(checkpoint["label_to_index"].items(), key=lambda row: row[1])
        ]
        self.stage1 = None
        self.orientation = None
        self.device = torch.device(
            "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
        )
        self.models: list[object] = []
        if args.mode in ("fast", "hybrid", "cascade"):
            self.stage1 = ct.models.MLModel(
                str(args.stage1_coreml), compute_units=ct.ComputeUnit.ALL
            )
        if args.mode in ("hybrid", "cascade"):
            self.orientation = ct.models.MLModel(
                str(args.orientation_coreml), compute_units=ct.ComputeUnit.ALL
            )
        if args.mode in ("hybrid", "cascade", "lip-aware"):
            from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
            from active.v17.model_visual_speech_v17 import (
                VisualSpeechTeacherV17,
                VisualSpeechTeacherV17Config,
            )

            def load(path: Path, expected: str, constructor):
                value = torch.load(path, map_location="cpu", weights_only=False)
                if value.get("format") != expected:
                    raise ValueError(f"unexpected checkpoint format: {path}")
                model = constructor(value).eval()
                model.load_state_dict(value["model_state_dict"], strict=True)
                return model

            visual = lambda row: VisualSpeechTeacherV17(
                VisualSpeechTeacherV17Config(**row["model_config"])
            )
            self.mouth = load(args.mouth_checkpoint, "slt_stage1_visual_speech_v17", visual)
            self.lower = load(args.lower_checkpoint, "slt_stage1_visual_speech_v17", visual)
            # The frozen Auto-AVSR frontends are identical. Share one copy in memory;
            # each view still receives its own real pixels and forward pass.
            self.lower.frontend = self.mouth.frontend
            self.models = [self.mouth, self.lower]
            self.landmark = load(
                args.landmark_checkpoint, "slt_stage1_v17",
                lambda row: SLTStage1V17(Stage1V17Config(**row["model_config"])),
            )
            self.models.append(self.landmark)
        if args.mode == "lip-aware":
            from active.v17.model_hand_mobileclip2_v17 import (
                HandMobileCLIP2Stage1Config,
                HandMobileCLIP2Stage1V17,
            )
            self.hand = load(
                args.hand_checkpoint, "slt_stage1_hand_mobileclip2_v17",
                lambda row: HandMobileCLIP2Stage1V17(
                    HandMobileCLIP2Stage1Config(**row["model_config"])
                ),
            )
            self.models.append(self.hand)
        for model in self.models:
            model.to(self.device).eval()
        self._warm_models()

    def _warm_models(self) -> None:
        """Pay lazy Core ML/MPS compilation once before the camera starts."""
        black = Image.fromarray(np.zeros((256, 256, 3), np.uint8))
        self.image_encoder.predict({"image": black})
        if self.stage1 is not None:
            self.stage1.predict({
                "landmarks": np.zeros((1, 32, 61, 5), np.float32),
                "hand_embeddings": np.zeros((1, 16, 3, 512), np.float32),
                "hand_valid": np.zeros((1, 16, 3), np.float32),
                "hand_boxes": np.zeros((1, 16, 3, 4), np.float32),
            })
        if self.orientation is not None:
            self.orientation.predict({
                "landmarks": np.zeros((1, 32, 61, 5), np.float32),
            })
        if self.args.mode == "fast":
            return
        torch = self.torch
        with torch.inference_mode():
            pixels = torch.zeros((1, 32, 1, 88, 88), device=self.device)
            valid_face = torch.ones((1, 32), dtype=torch.bool, device=self.device)
            shared_features = self.mouth.frontend(torch.cat((pixels, pixels), dim=0))
            self.mouth.forward_features(shared_features[:1], valid_face)
            self.lower.forward_features(shared_features[1:], valid_face)
            landmarks = torch.zeros((1, 32, 61, 5), device=self.device)
            self.landmark(landmarks)
            if self.args.mode == "lip-aware":
                embeddings = torch.zeros((1, 16, 3, 512), device=self.device)
                valid_hands = torch.zeros((1, 16, 3), dtype=torch.bool, device=self.device)
                boxes = torch.zeros((1, 16, 3, 4), device=self.device)
                self.hand(embeddings, valid_hands, boxes)
        if self.device.type == "mps":
            torch.mps.synchronize()

    def provenance(self) -> dict[str, object]:
        paths = {
            "unified_checkpoint": self.args.unified_checkpoint,
            "image_encoder": self.args.image_encoder,
        }
        if self.args.mode in ("fast", "hybrid", "cascade"):
            paths["stage1_coreml"] = self.args.stage1_coreml
        if self.args.mode in ("hybrid", "cascade"):
            paths["orientation_coreml"] = self.args.orientation_coreml
        if self.args.mode in ("hybrid", "cascade", "lip-aware"):
            paths.update({
                "landmark_checkpoint": self.args.landmark_checkpoint,
                "mouth_checkpoint": self.args.mouth_checkpoint,
                "lower_checkpoint": self.args.lower_checkpoint,
            })
        if self.args.mode == "lip-aware":
            paths.update({
                "hand_checkpoint": self.args.hand_checkpoint,
            })
        return {
            name: {"path": str(path), "sha256": sha256(path) if path.is_file() else "mlpackage"}
            for name, path in paths.items()
        }

    def set_interactive_mode(self, mode: str) -> None:
        if mode not in ("hybrid", "cascade") or self.orientation is None:
            return
        self.mode = mode

    def encode_hands(self, crops: list[list[np.ndarray | None]], valid: np.ndarray) -> np.ndarray:
        embeddings = np.zeros((16, 3, 512), np.float32)
        for frame in range(16):
            for view in range(3):
                crop = crops[frame][view]
                if crop is None or not valid[frame, view]:
                    continue
                image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                prediction = self.image_encoder.predict({"image": image})
                embeddings[frame, view] = np.asarray(prediction["embedding"]).reshape(512)
        return embeddings

    def classify(self, observations: list[ObservedFrame]) -> dict[str, object]:
        started = time.perf_counter()
        mode = self.mode
        original = observations
        observations, motion_diagnostics = trim_to_motion(
            observations, self.args.quiet_motion
        )
        try:
            features, landmark_diagnostics = landmarks_from_observations(
                observations, V17Config(),
            )
        except ValueError as error:
            if str(error) != "not enough detected hand frames":
                raise
            elapsed = 1000 * (time.perf_counter() - started)
            return {
                "gloss": "UNKNOWN",
                "candidate_gloss": None,
                "model_score": 0.0,
                "margin": 0.0,
                "gate_score": 0.0,
                "gate_margin": 0.0,
                "top3": [],
                "accepted": False,
                "rejection_reasons": ["insufficient_hand_evidence"],
                "mode": mode,
                "frames": len(original),
                "motion_trimmed_frames": len(observations),
                "start_seconds": original[0].seconds,
                "end_seconds": original[-1].seconds,
                "diagnostics": motion_diagnostics,
                "latency_ms": {
                    "tensor_extraction": elapsed,
                    "hand_image_encoding": 0.0,
                    "classification": 0.0,
                    "total": elapsed,
                },
            }
        landmark_diagnostics.update(motion_diagnostics)
        landmark_diagnostics["live_auxiliary_interval"] = V17Config().body_interval
        trim_start = int(landmark_diagnostics["trim_start"])
        trim_end = int(landmark_diagnostics["trim_end_exclusive"])
        cascade_primary_seconds = 0.0
        cascade_fallback = False
        orientation_raw = None
        if mode == "cascade":
            primary_started = time.perf_counter()
            orientation_raw = np.asarray(
                self.orientation.predict({"landmarks": features[None]})["var_5535"]
            ).reshape(100)
            cascade_primary_seconds = time.perf_counter() - primary_started
            orientation_probability = np.exp(orientation_raw - orientation_raw.max())
            orientation_probability /= orientation_probability.sum()
            cascade_fallback = float(orientation_probability.max()) < self.args.cascade_score

        needs_hands = mode != "cascade" or cascade_fallback
        embeddings = valid = boxes = None
        if needs_hands:
            crops, valid, boxes = hand_inputs(observations, trim_start, trim_end)
        extracted = time.perf_counter()
        if needs_hands:
            embeddings = self.encode_hands(crops, valid)
        encoded = time.perf_counter()

        fast_raw = None
        if mode == "cascade" and not cascade_fallback:
            fast_raw = orientation_raw
        elif self.stage1 is not None:
            provider = {
                "landmarks": features[None],
                "hand_embeddings": embeddings[None],
                "hand_valid": valid[None],
                "hand_boxes": boxes[None],
            }
            fast_raw = np.asarray(self.stage1.predict(provider)["var_2927"]).reshape(100)

        if mode == "fast":
            raw = fast_raw
            visual_diagnostics = {"mouth_pixels_used": False, "visual_tie_break": False}
        elif mode in ("hybrid", "cascade"):
            raw = (fast_raw - fast_raw.mean()) / max(float(fast_raw.std()), 1e-6)
            fast_order = np.argsort(fast_raw)[::-1]
            pair = {self.labels[int(index)] for index in fast_order[:2]}
            use_visual = pair == VISUAL_TIE_BREAK_GLOSSES
            use_landmark = pair == LANDMARK_TIE_BREAK_GLOSSES and (
                mode == "hybrid" or cascade_fallback
            )
            visual_diagnostics = {
                "mouth_pixels_used": use_visual,
                "visual_tie_break": use_visual,
                "landmark_tie_break": use_landmark,
                "fast_top3": [self.labels[int(index)] for index in fast_order[:3]],
            }
            if mode == "cascade":
                visual_diagnostics.update({
                    "cascade_primary_score": float(orientation_probability.max()),
                    "cascade_score_threshold": self.args.cascade_score,
                    "cascade_fallback_used": cascade_fallback,
                })

            def zscore_numpy(value):
                return (value - value.mean()) / max(float(value.std()), 1e-6)

            if use_visual:
                mouth, mouth_valid, lower, lower_valid, visual_values = visual_inputs(
                    observations, self.device
                )
                mouth_logits, lower_logits = self._visual_logits(
                    mouth, mouth_valid, lower, lower_valid
                )

                visual_raw = 0.4 * zscore_numpy(mouth_logits) + 0.6 * zscore_numpy(lower_logits)
                indices = [self.labels.index(label) for label in VISUAL_TIE_BREAK_GLOSSES]
                raw[indices] = 0.2 * raw[indices] + 0.8 * visual_raw[indices]
                visual_diagnostics.update(visual_values)
                visual_diagnostics["mouth_top1"] = self.labels[int(mouth_logits.argmax())]
                visual_diagnostics["lower_face_top1"] = self.labels[int(lower_logits.argmax())]
            elif use_landmark:
                with self.torch.inference_mode():
                    landmark_logits = self.landmark(
                        self.torch.from_numpy(features).unsqueeze(0).to(self.device)
                    ).float().cpu().numpy().reshape(-1)
                indices = [
                    self.labels.index(label) for label in LANDMARK_TIE_BREAK_GLOSSES
                ]
                landmark_raw = zscore_numpy(landmark_logits)
                raw[indices] = 0.2 * raw[indices] + 0.8 * landmark_raw[indices]
                visual_diagnostics["landmark_top1"] = self.labels[
                    int(landmark_logits.argmax())
                ]
        else:
            torch = self.torch
            mouth, mouth_valid, lower, lower_valid, visual_diagnostics = visual_inputs(
                observations, self.device
            )
            with torch.inference_mode():
                landmark_logits = self.landmark(
                    torch.from_numpy(features).unsqueeze(0).to(self.device)
                )
                hand_logits = self.hand(
                    torch.from_numpy(embeddings).unsqueeze(0).to(self.device),
                    torch.from_numpy(valid > 0).unsqueeze(0).to(self.device),
                    torch.from_numpy(boxes).unsqueeze(0).to(self.device),
                )
                mouth_logits, lower_logits = self._visual_logits(
                    mouth, mouth_valid, lower, lower_valid
                )

                def zscore(value):
                    return (value - value.mean(1, keepdim=True)) / value.std(
                        1, keepdim=True, unbiased=False
                    ).clamp_min(1e-6)

                raw_tensor = (
                    0.30 * zscore(landmark_logits)
                    + 0.15 * zscore(torch.from_numpy(mouth_logits)[None].to(self.device))
                    + 0.35 * zscore(torch.from_numpy(lower_logits)[None].to(self.device))
                    + 0.20 * zscore(hand_logits)
                )
                raw = raw_tensor.float().cpu().numpy().reshape(100)
            visual_diagnostics["mouth_pixels_used"] = True
        inferred = time.perf_counter()
        probability = np.exp(raw - raw.max())
        probability /= probability.sum()
        order = np.argsort(probability)[::-1][:3]
        top3 = [
            {"gloss": self.labels[int(index)], "model_score": float(probability[index])}
            for index in order
        ]
        gate_score = top3[0]["model_score"]
        gate_margin = top3[0]["model_score"] - top3[1]["model_score"]
        if mode in ("hybrid", "cascade"):
            fast_probability = np.exp(fast_raw - fast_raw.max())
            fast_probability /= fast_probability.sum()
            tie_break_glosses = None
            if visual_diagnostics["visual_tie_break"]:
                tie_break_glosses = VISUAL_TIE_BREAK_GLOSSES
            elif visual_diagnostics["landmark_tie_break"]:
                tie_break_glosses = LANDMARK_TIE_BREAK_GLOSSES
            if tie_break_glosses is not None:
                pair_indices = [
                    self.labels.index(label) for label in tie_break_glosses
                ]
                gate_score = float(fast_probability[pair_indices].sum())
                third = next(
                    int(index) for index in np.argsort(fast_probability)[::-1]
                    if int(index) not in pair_indices
                )
                gate_margin = gate_score - float(fast_probability[third])
            else:
                fast_order = np.argsort(fast_probability)[::-1]
                gate_score = float(fast_probability[fast_order[0]])
                gate_margin = float(
                    fast_probability[fast_order[0]] - fast_probability[fast_order[1]]
                )
            visual_diagnostics["fast_gate_score"] = gate_score
            visual_diagnostics["fast_gate_margin"] = gate_margin
        duration = observations[-1].seconds - observations[0].seconds
        landmark_diagnostics["raw_clip_seconds"] = (
            original[-1].seconds - original[0].seconds
        )
        landmark_diagnostics["motion_trimmed_seconds"] = duration
        landmark_diagnostics["observed_processing_fps"] = (
            (len(original) - 1) / max(landmark_diagnostics["raw_clip_seconds"], 1e-6)
        )
        rejection_reasons = []
        if gate_score < self.args.minimum_score:
            rejection_reasons.append("low_score")
        if gate_margin < self.args.minimum_margin:
            rejection_reasons.append("low_margin")
        if duration > self.args.maximum_accept_seconds:
            rejection_reasons.append("clip_too_long")
        accepted = not rejection_reasons
        return {
            "gloss": top3[0]["gloss"] if accepted else "UNKNOWN",
            "candidate_gloss": top3[0]["gloss"],
            "model_score": top3[0]["model_score"],
            "margin": top3[0]["model_score"] - top3[1]["model_score"],
            "gate_score": gate_score,
            "gate_margin": gate_margin,
            "top3": top3,
            "accepted": accepted,
            "rejection_reasons": rejection_reasons,
            "mode": mode,
            "frames": len(original),
            "motion_trimmed_frames": len(observations),
            "start_seconds": original[0].seconds,
            "end_seconds": original[-1].seconds,
            "diagnostics": {**landmark_diagnostics, **visual_diagnostics},
            "latency_ms": {
                "tensor_extraction": 1000 * (
                    extracted - started - cascade_primary_seconds
                ),
                "hand_image_encoding": 1000 * (encoded - extracted),
                "classification": 1000 * (
                    inferred - encoded + cascade_primary_seconds
                ),
                "total": 1000 * (inferred - started),
            },
        }

    def _visual_logits(self, mouth, mouth_valid, lower, lower_valid):
        """Run the shared frozen visual frontend once for the two genuine RGB views."""
        torch = self.torch
        with torch.inference_mode():
            visual_features = self.mouth.frontend(torch.cat((mouth, lower), dim=0))
            mouth_logits = self.mouth.forward_features(visual_features[:1], mouth_valid)
            lower_logits = self.lower.forward_features(visual_features[1:], lower_valid)
        return (
            mouth_logits.float().cpu().numpy().reshape(-1),
            lower_logits.float().cpu().numpy().reshape(-1),
        )


class SessionRecorder:
    def __init__(self, args: argparse.Namespace, classifier: IsolatedClassifier, source: str):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.root = args.output_root / stamp
        self.root.mkdir(parents=True, exist_ok=False)
        self.history_path = self.root / "history.json"
        self.video_path = self.root / "session_lowres.mp4"
        self.writer = None
        self.next_video_seconds = -1.0
        self.args = args
        self.data = {
            "format": "slt_live_isolated_v17_session",
            "version": 2,
            "prototype_scope": (
                "low-motion-delimited isolated Stage-1; not continuous Stage-2"
                if getattr(args, "end_on_low_motion", False)
                else "pause-delimited isolated Stage-1; not continuous Stage-2"
            ),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "mode": args.mode,
            "score_semantics": "uncalibrated softmax over model/fused scores",
            "config": {
                "processing_fps": args.processing_fps,
                "detection_image_side": args.detection_image_side,
                "maximum_image_side": args.maximum_image_side,
                "record_width": args.record_width,
                "minimum_score": args.minimum_score,
                "minimum_margin": args.minimum_margin,
                "maximum_accept_seconds": args.maximum_accept_seconds,
                "end_on_low_motion": getattr(args, "end_on_low_motion", False),
                "naturalizer": args.naturalizer,
                "stage3_checkpoint": str(args.stage3_checkpoint),
                "stage3_device": args.stage3_device,
                "ollama_model": args.ollama_model,
                "ollama_url": args.ollama_url,
                "ollama_timeout": args.ollama_timeout,
                "ollama_enabled": not args.no_ollama,
            },
            "models": classifier.provenance(),
            "video": str(self.video_path),
            "video_source_timestamps_seconds": [],
            "predictions": [],
            "events": [],
            "utterances": [],
            "test_accessed": False,
        }
        atomic_json(self.history_path, self.data)

    def write_frame(self, frame: np.ndarray, seconds: float) -> None:
        if self.next_video_seconds < 0:
            self.next_video_seconds = seconds
        if seconds + 1e-6 < self.next_video_seconds:
            return
        height, width = frame.shape[:2]
        target_width = min(self.args.record_width, width)
        target_height = max(2, int(round(height * target_width / width)))
        target_height += target_height % 2
        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        if self.writer is None:
            self.writer = cv2.VideoWriter(
                str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                self.args.record_fps, (target_width, target_height),
            )
            if not self.writer.isOpened():
                raise RuntimeError("could not create low-resolution MP4 session video")
        self.writer.write(resized)
        self.data["video_source_timestamps_seconds"].append(float(seconds))
        self.next_video_seconds += 1.0 / self.args.record_fps
        if self.next_video_seconds <= seconds:
            self.next_video_seconds = seconds + 1.0 / self.args.record_fps

    def add(self, result: dict[str, object]) -> None:
        self.data["predictions"].append(result)
        atomic_json(self.history_path, self.data)

    def add_event(self, event: dict[str, object]) -> None:
        event = {"recorded_utc": utc_now(), **event}
        self.data["events"].append(event)
        atomic_json(self.history_path, self.data)

    def add_utterance(self, utterance: dict[str, object]) -> None:
        self.data["utterances"].append(utterance)
        atomic_json(self.history_path, self.data)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        self.data["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.history_path, self.data)


def draw_detection(frame: np.ndarray, item: ObservedFrame | None, mirror: bool) -> np.ndarray:
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
                cv2.line(output, pixel(xy[start]), pixel(xy[end]), (0, 0, 0), 2,
                         cv2.LINE_AA)
        for point, score in zip(xy, confidence):
            if score > 0:
                cv2.circle(output, pixel(point), 3, (255, 255, 255), -1, cv2.LINE_AA)

    for hand in item.detection.hands:
        skeleton(hand.xy, hand.confidence, HAND_CONNECTIONS)
    skeleton(
        item.detection.body_xy, item.detection.body_confidence, BODY_CONNECTIONS
    )
    for point, confidence in zip(item.detection.face_xy, item.detection.face_confidence):
        if confidence > 0:
            cv2.circle(output, pixel(point), 3, (255, 255, 255), -1, cv2.LINE_AA)
    return output


def draw_hud(
    frame: np.ndarray, boundary: AutoBoundary, latest: ObservedFrame | None,
    results: list[dict[str, object]], pending: bool, fps: float, mode: str,
    active_glosses: list[str], finishing_glosses: list[str],
    final_sentence: str, finish_pending: bool, speech_text: str | None,
) -> np.ndarray:
    value = frame
    overlay = value.copy()
    cv2.rectangle(overlay, (0, 0), (min(value.shape[1], 620), 218), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, value, 0.28, 0, value)
    quality = "hand --  face --  motion --"
    if latest is not None:
        quality = (
            f"hand {latest.hand_quality:.2f}  face {latest.face_quality:.2f}  "
            f"motion {latest.motion:.3f}"
        )
    lines = [
        f"{boundary.state}{' / CLASSIFYING' if pending else ''}   {fps:.1f} FPS   {mode.upper()}",
        quality,
        f"boundary progress {boundary.progress * 100:3.0f}%",
    ]
    if results:
        last = results[-1]
        candidate = (
            f" (candidate {last['candidate_gloss']})" if not last["accepted"] else ""
        )
        lines.append(
            f"last: {last['gloss']}{candidate}  evidence {last['gate_score']:.3f}  "
            f"margin {last['gate_margin']:.3f}"
        )
        lines.append("top3: " + " | ".join(
            f"{row['gloss']} {row['model_score']:.2f}" for row in last["top3"]
        ))
    lines.append("Q quit   R reset display   F finish utterance   SPACE close current sign")
    for index, line in enumerate(lines):
        cv2.putText(value, line, (14, 28 + 30 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.rectangle(value, (14, 91), (314, 104), (75, 75, 75), -1)
    cv2.rectangle(value, (14, 91), (14 + int(300 * boundary.progress), 104), (30, 210, 100), -1)
    if mode in MODE_BUTTONS:
        for candidate, (left, top, right, bottom) in MODE_BUTTONS.items():
            active = candidate == mode
            cv2.rectangle(
                value, (left, top), (right, bottom),
                (40, 150, 70) if active else (65, 65, 65), -1,
            )
            cv2.rectangle(value, (left, top), (right, bottom), (240, 240, 240), 1)
            label = "DEFAULT" if candidate == "hybrid" else "CASCADE"
            cv2.putText(
                value, label, (left + 9, top + 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA,
            )

    height, width = value.shape[:2]
    panel_top = max(220, height - 132)
    bottom = value.copy()
    cv2.rectangle(bottom, (0, panel_top), (width, height), (15, 15, 15), -1)
    cv2.addWeighted(bottom, 0.78, value, 0.22, 0, value)

    show_finished = bool(finishing_glosses) and (
        finish_pending
        or (bool(final_sentence) and not active_glosses)
        or (bool(final_sentence) and speech_text == final_sentence)
    )
    shown_glosses = finishing_glosses if show_finished else active_glosses
    label = "FINISHING" if finish_pending else (
        "FINISHED GLOSSES" if show_finished else "GLOSS BUFFER"
    )
    gloss_text = "  ".join(shown_glosses) if shown_glosses else "(empty)"
    max_chars = max(18, (width - 36) // 11)
    if len(gloss_text) > max_chars:
        gloss_text = "..." + gloss_text[-(max_chars - 3):]
    cv2.putText(value, f"{label}: {gloss_text}", (14, panel_top + 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 1, cv2.LINE_AA)
    status = final_sentence
    if speech_text:
        status = f"SPEAKING: {speech_text}"
    elif finish_pending:
        status = "Naturalizing finished glosses..."
    if status:
        if len(status) > max_chars:
            status = status[:max_chars - 3] + "..."
        cv2.putText(value, status, (14, panel_top + 59), cv2.FONT_HERSHEY_SIMPLEX,
                    0.57, (225, 225, 225), 1, cv2.LINE_AA)
    for action, (left, top, right, button_bottom) in control_button_rects(width, height).items():
        color = (55, 55, 155) if action == "reset" else (35, 145, 70)
        cv2.rectangle(value, (left, top), (right, button_bottom), color, -1)
        cv2.rectangle(value, (left, top), (right, button_bottom), (245, 245, 245), 1)
        cv2.putText(value, action.upper(), (left + 18, top + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
    return value


def clicked_mode(x: int, y: int) -> str | None:
    for mode, (left, top, right, bottom) in MODE_BUTTONS.items():
        if left <= x <= right and top <= y <= bottom:
            return mode
    return None


def control_button_rects(width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    top = max(224, height - 54)
    return {
        "reset": (14, top, 134, top + 38),
        "finish": (146, top, 286, top + 38),
    }


def clicked_control(x: int, y: int, width: int, height: int) -> str | None:
    for action, (left, top, right, bottom) in control_button_rects(width, height).items():
        if left <= x <= right and top <= y <= bottom:
            return action
    return None


class LiveSpeaker:
    """Queue native speech so gloss and sentence audio never interrupt each other."""

    def __init__(self) -> None:
        from AppKit import NSSpeechSynthesizer

        self.synthesizer = NSSpeechSynthesizer.alloc().initWithVoice_(None)
        self.synthesizer.setRate_(220.0)
        self.queue: deque[dict[str, object]] = deque()
        self.current: dict[str, object] | None = None

    @property
    def current_text(self) -> str | None:
        return None if self.current is None else str(self.current["text"])

    def enqueue(self, text: str, kind: str, reference: object) -> None:
        if text.strip():
            self.queue.append({"text": text.strip(), "kind": kind, "reference": reference})

    def update(self) -> dict[str, object] | None:
        if self.synthesizer.isSpeaking():
            return None
        self.current = None
        if not self.queue:
            return None
        self.current = self.queue.popleft()
        self.synthesizer.startSpeakingString_(str(self.current["text"]).lower())
        return self.current

    def clear(self) -> None:
        self.queue.clear()
        self.current = None
        self.synthesizer.stopSpeaking()


def run(args: argparse.Namespace) -> dict[str, object]:
    classifier = IsolatedClassifier(args)
    speaker = None if args.no_speech else LiveSpeaker()
    naturalizer = make_naturalizer(args)
    source = str(args.video) if args.video else f"camera:{args.camera}"
    recorder = SessionRecorder(args, classifier, source)
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
    boundary = AutoBoundary(BoundaryConfig(
        processing_fps=args.processing_fps,
        start_motion=args.start_motion,
        quiet_motion=args.quiet_motion,
        quiet_seconds=args.quiet_seconds,
        cooldown_seconds=(
            0.08 if getattr(args, "end_on_low_motion", False) else 0.35
        ),
        require_neutral=not getattr(args, "end_on_low_motion", False),
    ))
    detector = AppleVisionDetector(args.minimum_point_confidence)
    previous_wrists: dict[str, np.ndarray | None] = {"left": None, "right": None}
    segment_video = bool(args.video and getattr(args, "segment_video", False))
    observations = (
        [] if args.video and not segment_video
        else deque(maxlen=math.ceil(args.processing_fps * 5.0))
    )
    results: list[dict[str, object]] = []
    executor = ThreadPoolExecutor(max_workers=1)
    language_executor = ThreadPoolExecutor(max_workers=1)
    warm_future = language_executor.submit(naturalizer.warm)
    warm_logged = False
    future: Future | None = None
    future_buffer_epoch: int | None = None
    future_display_epoch: int | None = None
    naturalizer_future: Future | None = None
    naturalizer_context: dict[str, object] | None = None
    next_process = 0.0
    frame_index = processed = 0
    auxiliary_interval = V17Config().body_interval
    wall_started = time.perf_counter()
    display_times = deque(maxlen=30)
    latest = None
    pending_speech: deque[dict[str, object]] = deque()
    ui_actions: deque[tuple[str, float]] = deque()
    active_glosses: list[str] = []
    finishing_glosses: list[str] = []
    last_sentence = ""
    finish_requested = False
    finish_request_seconds: float | None = None
    buffer_epoch = 0
    display_epoch = 0
    utterance_counter = 0
    display_size = [1280, 720]

    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME)

        def on_mouse(event, x, y, _flags, _parameter):
            if event != cv2.EVENT_LBUTTONUP:
                return
            selected = clicked_mode(x, y)
            if selected is not None:
                classifier.set_interactive_mode(selected)
                return
            action = clicked_control(x, y, display_size[0], display_size[1])
            if action is not None:
                ui_actions.append((action, time.perf_counter() - wall_started))

        cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    def submit(clip: list[ObservedFrame]) -> None:
        nonlocal future, future_buffer_epoch, future_display_epoch
        if future is None and len(clip) >= 4:
            future = executor.submit(classifier.classify, list(clip))
            future_buffer_epoch = buffer_epoch
            future_display_epoch = display_epoch

    def collect_result(wait: bool = False) -> None:
        nonlocal future, future_buffer_epoch, future_display_epoch
        if future is None or (not wait and not future.done()):
            return
        result = future.result()
        expected = args.expected_label
        if expected is None and args.video and args.video.parent.name in classifier.labels:
            expected = args.video.parent.name
        if expected:
            result["expected_gloss"] = expected
            result["correct"] = result["gloss"] == expected
        result["buffer_epoch"] = future_buffer_epoch
        result["display_epoch"] = future_display_epoch
        included = bool(result["accepted"] and future_buffer_epoch == buffer_epoch)
        result["included_in_active_buffer"] = included
        results.append(result)
        recorder.add(result)
        print(json.dumps(result, indent=2))
        if included:
            active_glosses.append(str(result["gloss"]))
            pending_speech.append({
                "text": str(result["gloss"]),
                "kind": "gloss",
                "reference": len(results) - 1,
            })
        future = None
        future_buffer_epoch = None
        future_display_epoch = None

    def maybe_log_warmup() -> None:
        nonlocal warm_logged
        if warm_logged or not warm_future.done():
            return
        warm_logged = True
        recorder.add_event({"type": "naturalizer_warmup", **warm_future.result()})

    def maybe_start_finish() -> None:
        nonlocal finish_requested, finish_request_seconds, naturalizer_future
        nonlocal naturalizer_context, buffer_epoch, utterance_counter, last_sentence
        if not finish_requested or future is not None or naturalizer_future is not None:
            return
        glosses = list(active_glosses)
        active_glosses.clear()
        buffer_epoch += 1
        finish_requested = False
        finishing_glosses[:] = glosses
        if not glosses:
            last_sentence = "No recognized signs to finish."
            recorder.add_event({
                "type": "finish_empty",
                "seconds": finish_request_seconds,
                "buffer_epoch": buffer_epoch,
            })
            finish_request_seconds = None
            return
        utterance_counter += 1
        utterance_id = f"live-{utterance_counter:04d}"
        naturalizer_context = {
            "utterance_id": utterance_id,
            "requested_utc": utc_now(),
            "requested_seconds": finish_request_seconds,
            "display_epoch": display_epoch,
            "glosses": glosses,
        }
        recorder.add_event({"type": "finish_submitted", **naturalizer_context})
        naturalizer_future = language_executor.submit(naturalizer.rephrase, glosses)
        finish_request_seconds = None

    def collect_naturalizer(wait: bool = False) -> None:
        nonlocal naturalizer_future, naturalizer_context, last_sentence
        if naturalizer_future is None or (not wait and not naturalizer_future.done()):
            return
        result = naturalizer_future.result()
        context = dict(naturalizer_context or {})
        visible = context.get("display_epoch") == display_epoch
        utterance = {
            **context,
            **result,
            "completed_utc": utc_now(),
            "displayed_and_spoken": visible,
        }
        recorder.add_utterance(utterance)
        print(json.dumps(utterance, indent=2))
        if visible:
            last_sentence = str(result["sentence"])
            pending_speech.append({
                "text": last_sentence,
                "kind": "finished_sentence",
                "reference": context.get("utterance_id"),
            })
        naturalizer_future = None
        naturalizer_context = None

    def reset_display(seconds: float, source_name: str) -> None:
        nonlocal finish_requested, finish_request_seconds, buffer_epoch
        nonlocal display_epoch, last_sentence
        cleared = list(active_glosses)
        active_glosses.clear()
        finishing_glosses.clear()
        last_sentence = ""
        finish_requested = False
        finish_request_seconds = None
        buffer_epoch += 1
        display_epoch += 1
        boundary.reset()
        pending_speech.clear()
        if speaker is not None:
            speaker.clear()
        recorder.add_event({
            "type": "reset",
            "source": source_name,
            "seconds": seconds,
            "cleared_glosses": cleared,
            "classifier_pending": future is not None,
            "naturalizer_pending": naturalizer_future is not None,
            "buffer_epoch": buffer_epoch,
            "display_epoch": display_epoch,
        })

    def request_finish(seconds: float, source_name: str) -> None:
        nonlocal finish_requested, finish_request_seconds
        if finish_requested:
            recorder.add_event({
                "type": "finish_ignored_already_pending",
                "source": source_name,
                "seconds": seconds,
            })
            return
        finish_requested = True
        finish_request_seconds = seconds
        recorder.add_event({
            "type": "finish_requested",
            "source": source_name,
            "seconds": seconds,
            "active_glosses": list(active_glosses),
            "classifier_pending": future is not None,
            "naturalizer_pending": naturalizer_future is not None,
        })
        if future is None and boundary.state == "SIGNING":
            clip = boundary.finish([])
            boundary.reset()
            if clip is not None:
                submit(clip)
        maybe_start_finish()

    def service_speech() -> None:
        if speaker is None:
            pending_speech.clear()
            return
        while pending_speech:
            item = pending_speech.popleft()
            speaker.enqueue(str(item["text"]), str(item["kind"]), item["reference"])
            recorder.add_event({"type": "speech_queued", **item})
        started_item = speaker.update()
        if started_item is not None:
            recorder.add_event({"type": "speech_started", **started_item})

    try:
        while True:
            ok, raw = capture.read()
            if not ok:
                break
            seconds = (
                frame_index / source_fps if args.video
                else time.perf_counter() - wall_started
            )
            frame_index += 1
            canonical = orient_frame(raw, args.rotation, args.input_mirrored)
            recorder.write_frame(canonical, seconds)
            collect_result()
            maybe_log_warmup()
            collect_naturalizer()
            maybe_start_finish()
            if future is None and not finish_requested and seconds + 1e-6 >= next_process:
                detection_frame = limit_image_side(canonical, args.detection_image_side)
                frame = limit_image_side(canonical, args.maximum_image_side)
                detection = detector.detect(
                    detection_frame,
                    include_body=processed % auxiliary_interval == 0,
                    include_face=True,
                    include_hands=True,
                )
                assigned = assign_hands(detection.hands, previous_wrists)
                motion = wrist_motion(assigned, previous_wrists)
                hand_quality, face_quality = observation_quality(detection)
                latest = ObservedFrame(
                    frame, detection, assigned, seconds, motion, hand_quality,
                    face_quality, face_for_features=processed % auxiliary_interval == 0,
                )
                observations.append(latest)
                clip = None if args.video and not segment_video else boundary.update(latest)
                if clip is not None:
                    submit(clip)
                    if segment_video:
                        collect_result(wait=True)
                processed += 1
                next_process = max(next_process + 1.0 / args.processing_fps, seconds)

            if not args.no_display:
                now = time.perf_counter()
                display_times.append(now)
                display_fps = (
                    (len(display_times) - 1) / (display_times[-1] - display_times[0])
                    if len(display_times) > 1 else 0.0
                )
                shown = draw_detection(canonical, latest, mirror=not args.no_mirror_display)
                display_size[:] = [shown.shape[1], shown.shape[0]]
                visible_results = [
                    row for row in results if row.get("display_epoch") == display_epoch
                ]
                shown = draw_hud(
                    shown, boundary, latest, visible_results, future is not None,
                    display_fps, classifier.mode, active_glosses, finishing_glosses,
                    last_sentence, finish_requested or naturalizer_future is not None,
                    None if speaker is None else speaker.current_text,
                )
                cv2.imshow(WINDOW_NAME, shown)
                key = cv2.waitKey(1) & 0xFF
                service_speech()
                while ui_actions:
                    action, action_seconds = ui_actions.popleft()
                    if action == "reset":
                        reset_display(action_seconds, "button")
                    elif action == "finish":
                        request_finish(action_seconds, "button")
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    reset_display(seconds, "keyboard")
                if key == ord("f"):
                    request_finish(seconds, "keyboard")
                if key == ord(" "):
                    submit(boundary.finish(list(boundary.preroll)) or [])
            else:
                service_speech()

        if future is None:
            clip = (
                list(observations)
                if args.video and not segment_video
                else boundary.finish([] if segment_video else list(observations))
            )
            if clip is not None:
                submit(clip)
        if future is not None:
            collect_result(wait=True)
        maybe_start_finish()
        if naturalizer_future is not None:
            collect_naturalizer(wait=True)
        maybe_log_warmup()
        service_speech()
    finally:
        capture.release()
        executor.shutdown(wait=True)
        language_executor.shutdown(wait=True)
        recorder.close()
        cv2.destroyAllWindows()
    summary = {
        "session": str(recorder.root),
        "history": str(recorder.history_path),
        "video": str(recorder.video_path),
        "predictions": len(results),
        "accepted": sum(bool(row["accepted"]) for row in results),
        "utterances": len(recorder.data["utterances"]),
    }
    print(json.dumps(summary, indent=2))
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    source = value.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="saved video to test before webcam use")
    source.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    value.add_argument(
        "--mode", choices=("hybrid", "cascade", "fast", "lip-aware"), default="hybrid"
    )
    value.add_argument("--device", default="auto")
    value.add_argument("--processing-fps", type=float, default=30.0)
    value.add_argument("--detection-image-side", type=int, default=720)
    value.add_argument("--maximum-image-side", type=int, default=1280)
    value.add_argument("--minimum-point-confidence", type=float, default=0.15)
    value.add_argument("--start-motion", type=float, default=0.012)
    value.add_argument("--quiet-motion", type=float, default=0.006)
    value.add_argument("--quiet-seconds", type=float, default=0.20)
    value.add_argument("--minimum-score", type=float, default=0.25)
    value.add_argument("--minimum-margin", type=float, default=0.08)
    value.add_argument("--cascade-score", type=float, default=CASCADE_SCORE_THRESHOLD)
    value.add_argument("--maximum-accept-seconds", type=float, default=2.5)
    value.add_argument("--expected-label")
    value.add_argument("--rotation", type=float, default=0.0)
    value.add_argument("--input-mirrored", action="store_true")
    value.add_argument("--no-mirror-display", action="store_true")
    value.add_argument("--no-display", action="store_true")
    value.add_argument("--no-speech", action="store_true")
    value.add_argument(
        "--naturalizer", choices=("tiny", "ollama", "literal"), default="tiny",
        help="FINISH renderer; tiny is the promoted local 15.6M checkpoint",
    )
    value.add_argument("--stage3-checkpoint", type=Path, default=DEFAULT_STAGE3_TINY)
    value.add_argument(
        "--stage3-device", choices=("cpu", "mps", "auto"), default="cpu",
        help="CPU avoids competing with the live Stage-1 MPS workload",
    )
    value.add_argument("--no-ollama", action="store_true")
    value.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    value.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    value.add_argument("--ollama-timeout", type=float, default=10.0)
    value.add_argument("--record-width", type=int, default=640)
    value.add_argument("--record-fps", type=float, default=15.0)
    value.add_argument(
        "--output-root", type=Path,
        default=REPO / "artifacts/reports/live_isolated_v17",
    )
    value.add_argument("--unified-checkpoint", type=Path, default=DEFAULT_UNIFIED)
    value.add_argument("--stage1-coreml", type=Path, default=DEFAULT_STAGE1_COREML)
    value.add_argument(
        "--orientation-coreml", type=Path, default=DEFAULT_ORIENTATION_COREML
    )
    value.add_argument("--image-encoder", type=Path, default=DEFAULT_IMAGE_ENCODER)
    value.add_argument("--landmark-checkpoint", type=Path, default=DEFAULT_LANDMARK)
    value.add_argument("--hand-checkpoint", type=Path, default=DEFAULT_HAND)
    value.add_argument("--mouth-checkpoint", type=Path, default=DEFAULT_MOUTH)
    value.add_argument("--lower-checkpoint", type=Path, default=DEFAULT_LOWER)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.processing_fps <= 0 or args.record_fps <= 0 or args.ollama_timeout <= 0:
        raise ValueError("FPS and timeout values must be positive")
    if not 0.0 <= args.cascade_score <= 1.0:
        raise ValueError("cascade score must be in [0, 1]")
    if (
        args.detection_image_side < 320
        or args.maximum_image_side < args.detection_image_side
        or args.record_width < 160
    ):
        raise ValueError("image sizes are too small")
    if args.expected_label:
        args.expected_label = args.expected_label.upper()
    run(args)


if __name__ == "__main__":
    main()
