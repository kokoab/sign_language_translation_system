"""Layout checks for the reel HUD.  Rendering only, no model state."""

from pathlib import Path
import sys
import types
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.reel_hud_v17 import (
    ReelHud,
    clicked_reel_control,
    reel_control_button_rects,
)


def observation(**overrides):
    values = {"hand_quality": 0.9, "face_quality": 0.8, "motion": 0.05}
    values.update(overrides)
    return types.SimpleNamespace(**values)


def result(committed=None, score=0.9):
    return {
        "top3": [
            {"gloss": "TECHNOLOGY", "model_score": score},
            {"gloss": "USE", "model_score": 0.03},
            {"gloss": "PHONE", "model_score": 0.01},
        ],
        "latency_ms": {"total": 18.4},
        "model_score": score,
        "committed_gloss": committed,
        "full_verifier": {"commit_score": score},
    }


class ReelHudTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((720, 1280, 3), np.uint8)
        self.lock = types.SimpleNamespace(hits=2, required_hits=3, suppressed=None)
        self.hud = ReelHud()

    def draw(self, **overrides):
        arguments = {
            "frame": self.frame.copy(), "latest": observation(),
            "latest_result": result("TECHNOLOGY"), "lock": self.lock,
            "pending": False, "active": True, "fps": 24.8,
            "glosses": ["TECHNOLOGY", "USE", "IMPROVE"], "finishing_glosses": [],
            "sentence": "Let's use technology to improve lives, not war.",
            "finish_pending": False, "speech_text": None,
            "ctc_hypothesis": ["I", "LOVE", "YOU"],
            "stats": {"dropped": 3, "observations": 412},
        }
        arguments.update(overrides)
        return self.hud.draw(**arguments)

    def test_full_state_keeps_the_frame_geometry(self) -> None:
        shown = self.draw()
        self.assertEqual(shown.shape, self.frame.shape)
        self.assertEqual(shown.dtype, self.frame.dtype)
        self.assertGreater(int(shown.sum()), 0)

    def test_empty_state_draws_without_results_or_glosses(self) -> None:
        shown = self.draw(
            latest=None, latest_result=None, glosses=[], sentence="",
            active=False, ctc_hypothesis=None, stats={},
            lock=types.SimpleNamespace(hits=0, required_hits=3, suppressed=None),
        )
        self.assertEqual(shown.shape, self.frame.shape)

    def test_committed_gloss_is_held_then_cleared_with_the_buffer(self) -> None:
        self.draw()
        self.assertEqual(self.hud._committed, ("TECHNOLOGY", 0.9))
        self.draw(latest_result=result(None))
        self.assertEqual(self.hud._committed, ("TECHNOLOGY", 0.9))
        self.draw(glosses=[], latest_result=None)
        self.assertIsNone(self.hud._committed)

    def test_odd_frame_sizes_keep_controls_inside_the_frame(self) -> None:
        for width, height in [(640, 360), (1280, 720), (1920, 1080)]:
            rects = reel_control_button_rects(width, height)
            for action, (left, top, right, bottom) in rects.items():
                self.assertGreaterEqual(left, 0)
                self.assertLessEqual(right, width)
                self.assertLessEqual(bottom, height)
                self.assertEqual(
                    clicked_reel_control(
                        (left + right) // 2, (top + bottom) // 2, width, height
                    ),
                    action,
                )
            shown = self.hud.draw(
                np.zeros((height, width, 3), np.uint8), observation(),
                result("TECHNOLOGY"), self.lock, True, True, 30.0,
                ["A", "B"], [], "", False, None,
            )
            self.assertEqual(shown.shape, (height, width, 3))


if __name__ == "__main__":
    unittest.main()
