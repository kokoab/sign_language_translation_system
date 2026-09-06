"""Causal camera features for the experimental continuous v17 runtime.

Unlike stored clip extraction, the normalization window ends at the current camera
frame. This domain difference must be measured with real video replay; matching the
61x5 shape alone does not establish model compatibility or live accuracy.
"""

from collections import deque

import numpy as np

from active.v17.geometry_v17 import body_relative_normalize, image_normalized_to_isotropic
from active.v17.schema_v17 import BODY_START, BODY_END, FACE_START, FACE_END


class CausalVisionFeatures:
    def __init__(self, window=32):
        if window < 1:
            raise ValueError("positive normalization window required")
        self.window = window
        self.reset()

    def reset(self):
        self.xy = deque(maxlen=self.window)
        self.confidence = deque(maxlen=self.window)
        self.dimensions = None
        self.frames_seen = 0
        self.diagnostics = {}

    def add(self, detection, assigned, width, height):
        if self.dimensions is not None and self.dimensions != (width, height):
            raise ValueError("camera dimensions changed; reset normalization first")
        xy = np.zeros((61, 2), np.float32)
        confidence = np.zeros(61, np.float32)
        for side, start in (("left", 0), ("right", 21)):
            hand = assigned[side]
            if hand is not None:
                xy[start:start + 21] = hand.xy
                confidence[start:start + 21] = hand.confidence
        xy[BODY_START:BODY_END] = detection.body_xy
        confidence[BODY_START:BODY_END] = detection.body_confidence
        xy[FACE_START:FACE_END] = detection.face_xy
        confidence[FACE_START:FACE_END] = detection.face_confidence
        if not np.isfinite(xy).all() or not np.isfinite(confidence).all():
            raise ValueError("nonfinite camera detection")
        confidence = np.clip(confidence, 0, 1)
        self.dimensions = (width, height)
        self.xy.append(image_normalized_to_isotropic(xy, width, height, confidence > 0))
        self.confidence.append(confidence)
        normalized, depth, self.diagnostics = body_relative_normalize(
            np.stack(self.xy), np.stack(self.confidence))
        output = np.zeros((61, 5), np.float32)
        output[:, :2] = normalized[-1]
        output[:, 2] = depth[-1]
        output[:, 3] = confidence > 0
        output[:, 4] = confidence
        self.frames_seen += 1
        return output
