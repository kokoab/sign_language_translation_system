"""Novelty gestures for the desktop Live screen (personal use; not an ASL lexicon entry).

Middle finger: either hand upright with the middle finger extended and the index,
ring and little fingers folded, held for a moment. It is detected from the same
Apple Vision hand landmarks as the Finish control pose, never from the sign
recognizer, so it cannot change recognition of the 100 signs. While the pose is
held the frames are kept out of the word decoder, like the Finish pose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.live_reel_stage1_v17 import _straight_finger

MIDDLE_FINGER_TEXT = 'Fuck you!'
MIDDLE_FINGER_GLOSS = 'FUCK-YOU'


def is_middle_finger_hand(hand) -> bool:
    """Upright hand, middle finger straight and long, index/ring/little folded."""
    if hand is None:
        return False
    required = np.array([0, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20])
    if np.any(hand.confidence[required] <= 0):
        return False
    xy = hand.xy
    wrist = xy[0]
    palm = xy[9] - wrist
    palm_length = float(np.linalg.norm(palm))
    if palm_length < 0.025:
        return False
    axis = palm / palm_length
    if axis[1] > -0.35:            # hand must point up (image y grows downward)
        return False
    if not _straight_finger(xy, wrist, 9, 10, 12, axis):
        return False
    if np.linalg.norm(xy[12] - wrist) < 1.5 * palm_length:
        return False
    folded = all(
        np.linalg.norm(xy[tip] - wrist) < 1.05 * np.linalg.norm(xy[mcp] - wrist)
        for mcp, tip in ((5, 8), (13, 16), (17, 20))
    )
    # The extended middle fingertip must stand clearly beyond the folded ones along the
    # hand's own axis (so a tilted hand still counts).
    clear = all(float(np.dot(xy[12] - xy[tip], axis)) > 0.5 * palm_length for tip in (8, 16, 20))
    return bool(folded and clear)


@dataclass
class MiddleFingerGesture:
    """Hold/latch: fires once per held pose; brief detection dropouts are tolerated."""

    hold_seconds: float = 0.6
    dropout_grace_seconds: float = 0.15
    started_at: float | None = None
    last_seen_at: float | None = None
    active: bool = False
    latched: bool = False

    def update(self, assigned, seconds: float) -> bool:
        observed = any(is_middle_finger_hand(assigned.get(side)) for side in ('left', 'right'))
        if observed:
            self.last_seen_at = seconds
            if self.started_at is None:
                self.started_at = seconds
        elif self.last_seen_at is None or seconds - self.last_seen_at > self.dropout_grace_seconds:
            self.started_at = self.last_seen_at = None
            self.active = self.latched = False
            return False
        self.active = True
        if self.latched or seconds - self.started_at < self.hold_seconds:
            return False
        self.latched = True
        return True
