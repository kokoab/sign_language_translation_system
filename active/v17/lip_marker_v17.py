"""Small MediaPipe lip-landmark features for GOOD/THANKYOU disambiguation.

This deliberately is not a 100-class recognizer.  It is a closed, optional
two-class specialist used only after the hand/landmark recognizer proposes GOOD or
THANKYOU.  Training and live inference share the exact same landmark indices and
normalization.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


OUTER_LIP = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
)
INNER_LIP = (
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
)
LIP_INDICES = OUTER_LIP + INNER_LIP
TARGET_FRAMES = 24
LABELS = ("GOOD", "THANKYOU")


def draw_lip_markers(
    frame: np.ndarray, lips: np.ndarray | None, *, mirror: bool
) -> np.ndarray:
    """Draw the same outer and inner lip points consumed by the verifier."""
    if lips is None:
        return frame
    if lips.shape != (len(LIP_INDICES), 2):
        raise ValueError(
            f"expected {(len(LIP_INDICES), 2)} lip points, got {lips.shape}"
        )
    height, width = frame.shape[:2]
    points = lips.copy()
    if mirror:
        points[:, 0] = 1.0 - points[:, 0]
    pixels = [
        (int(round(point[0] * (width - 1))), int(round(point[1] * (height - 1))))
        for point in points
    ]
    boundary = len(OUTER_LIP)
    for contour in (pixels[:boundary], pixels[boundary:]):
        for first, second in zip(contour, contour[1:] + contour[:1]):
            cv2.line(frame, first, second, (255, 255, 255), 1, cv2.LINE_AA)
    for point in pixels:
        cv2.circle(frame, point, 2, (255, 255, 255), -1, cv2.LINE_AA)
    return frame


class LipMarkerTracker:
    """Track outer and inner lip points; RGB is discarded immediately."""

    def __init__(self, enabled: bool = True):
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
        result = self.mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return None
        face = result.multi_face_landmarks[0].landmark
        return np.asarray(
            [(face[index].x, face[index].y) for index in LIP_INDICES],
            dtype=np.float32,
        )

    def close(self) -> None:
        if self.mesh is not None:
            self.mesh.close()


def _interpolate_missing(sequence: list[np.ndarray | None]) -> np.ndarray | None:
    valid = np.asarray([value is not None for value in sequence])
    if valid.sum() < 4:
        return None
    source = np.flatnonzero(valid).astype(np.float32)
    values = np.stack([value for value in sequence if value is not None])
    positions = np.arange(len(sequence), dtype=np.float32)
    output = np.empty((len(sequence), len(LIP_INDICES), 2), np.float32)
    for point in range(len(LIP_INDICES)):
        for coordinate in range(2):
            output[:, point, coordinate] = np.interp(
                positions, source, values[:, point, coordinate]
            )
    return output


def lip_marker_features(sequence: list[np.ndarray | None]) -> np.ndarray | None:
    """Return a fixed feature vector invariant to face position, scale, and roll."""
    points = _interpolate_missing(sequence)
    if points is None:
        return None
    source = np.linspace(0.0, 1.0, len(points), dtype=np.float32)
    target = np.linspace(0.0, 1.0, TARGET_FRAMES, dtype=np.float32)
    resampled = np.empty((TARGET_FRAMES, len(LIP_INDICES), 2), np.float32)
    for point in range(len(LIP_INDICES)):
        for coordinate in range(2):
            resampled[:, point, coordinate] = np.interp(
                target, source, points[:, point, coordinate]
            )

    left = resampled[:, 0]
    right = resampled[:, 10]
    center = (left + right) * 0.5
    axis = right - left
    width = np.linalg.norm(axis, axis=1).clip(1e-4)
    cosine = axis[:, 0] / width
    sine = axis[:, 1] / width
    relative = resampled - center[:, None]
    aligned = np.empty_like(relative)
    aligned[..., 0] = (
        relative[..., 0] * cosine[:, None]
        + relative[..., 1] * sine[:, None]
    ) / width[:, None]
    aligned[..., 1] = (
        -relative[..., 0] * sine[:, None]
        + relative[..., 1] * cosine[:, None]
    ) / width[:, None]

    # Shape plus its temporal change captures visible mouthing without retaining RGB.
    velocity = np.diff(aligned, axis=0, prepend=aligned[:1])
    return np.concatenate((aligned.reshape(-1), velocity.reshape(-1))).astype(
        np.float32
    )


class LipMarkerDisambiguator:
    """Dependency-free runtime for a saved standardized logistic regressor."""

    def __init__(self, path: Path):
        value = np.load(path, allow_pickle=False)
        if str(value["format"].item()) != "slt_lip_marker_binary_v17":
            raise ValueError(f"unexpected lip-marker model format: {path}")
        self.path = Path(path)
        self.mean = value["mean"].astype(np.float32)
        self.scale = value["scale"].astype(np.float32)
        self.coefficient = value["coefficient"].astype(np.float32)
        self.intercept = float(value["intercept"].item())

    def predict(
        self, sequence: list[np.ndarray | None]
    ) -> dict[str, float | str] | None:
        features = lip_marker_features(sequence)
        if features is None:
            return None
        logit = float(
            np.dot((features - self.mean) / self.scale, self.coefficient)
            + self.intercept
        )
        probability = float(1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0))))
        label = LABELS[int(probability >= 0.5)]
        confidence = probability if label == LABELS[1] else 1.0 - probability
        return {
            "label": label,
            "confidence": confidence,
            "thankyou_probability": probability,
        }
