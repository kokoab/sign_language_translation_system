"""Layout checks for the reel HUD.  Rendering only, no model state."""

from pathlib import Path
import sys
import types
import unittest

import numpy as np
from PIL import ImageDraw

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.reel_hud_v17 import (
    NAV_PAGES,
    ReelHud,
    canvas,
    clicked_nav,
    clicked_reel_control,
    draw_nav,
    grid_rects,
    hit,
    nav_height,
    nav_rects,
    reel_control_button_rects,
    row_rects,
    scale,
    to_frame,
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


class SharedSurfaceTest(unittest.TestCase):
    """The toolkit the non-camera pages draw with."""

    def test_canvas_round_trips_to_a_bgr_frame(self) -> None:
        shown = to_frame(canvas(640, 360))
        self.assertEqual(shown.shape, (360, 640, 3))
        self.assertEqual(shown.dtype, np.uint8)

    def test_hit_finds_the_box_and_misses_outside_it(self) -> None:
        rects = {"one": (0, 0, 10, 10), "two": (20, 20, 30, 30)}
        self.assertEqual(hit(rects, 5, 5), "one")
        self.assertEqual(hit(rects, 25, 25), "two")
        self.assertIsNone(hit(rects, 15, 15))

    def test_row_rects_stay_inside_the_frame_and_do_not_overlap(self) -> None:
        for width in (640, 1280, 1920):
            rects = row_rects(("HOME", "GLOSSES", "HISTORY"), width, 10, scale(width))
            boxes = sorted(rects.values())
            self.assertGreaterEqual(boxes[0][0], 0)
            self.assertLessEqual(boxes[-1][2], width)
            for earlier, later in zip(boxes, boxes[1:]):
                self.assertLessEqual(earlier[2], later[0])

    def test_grid_rects_only_returns_visible_tiles(self) -> None:
        rects, rows = grid_rects(100, 1280, 80, 700, scale(1280), columns=5)
        self.assertEqual(rows, 20)
        self.assertLess(len(rects), 100)  # the rest are below the fold
        for left, top, right, bottom in rects.values():
            self.assertGreaterEqual(left, 0)
            self.assertLessEqual(right, 1280)
            self.assertLessEqual(bottom, 700)

    def test_grid_offset_scrolls_without_resizing_tiles(self) -> None:
        first, _ = grid_rects(100, 1280, 80, 700, scale(1280), columns=5)
        scrolled, _ = grid_rects(100, 1280, 80, 700, scale(1280), columns=5, offset=2)
        self.assertNotIn(0, scrolled)
        self.assertIn(10, scrolled)  # row 2 now sits at the top
        self.assertEqual(scrolled[10], first[0])

    def test_grid_handles_an_empty_catalog(self) -> None:
        rects, rows = grid_rects(0, 1280, 80, 700, scale(1280))
        self.assertEqual((rects, rows), ({}, 0))


class NavigationTest(unittest.TestCase):
    def test_every_page_is_clickable_at_its_own_label(self) -> None:
        for width in (640, 1280, 1920):
            rects = nav_rects(width)
            self.assertEqual(tuple(rects), NAV_PAGES)
            for page, (left, top, right, bottom) in rects.items():
                self.assertLessEqual(bottom, nav_height(width))
                self.assertEqual(
                    clicked_nav((left + right) // 2, (top + bottom) // 2, width), page
                )

    def test_a_miss_below_the_bar_selects_nothing(self) -> None:
        self.assertIsNone(clicked_nav(20, nav_height(1280) + 40, 1280))

    def test_nav_paints_without_disturbing_the_surface_size(self) -> None:
        image = canvas(1280, 720)
        draw_nav(image, ImageDraw.Draw(image, "RGBA"), 1280, "GLOSSES", note="ready")
        self.assertEqual(to_frame(image).shape, (720, 1280, 3))


class TopInsetTest(unittest.TestCase):
    """The live overlay must be untouched unless a shell asks for the strip."""

    def setUp(self) -> None:
        self.frame = np.zeros((720, 1280, 3), np.uint8)
        self.lock = types.SimpleNamespace(hits=2, required_hits=3, suppressed=None)

    def render(self, **overrides):
        hud = ReelHud()
        arguments = {
            "frame": self.frame.copy(), "latest": observation(),
            "latest_result": result("TECHNOLOGY"), "lock": self.lock,
            "pending": False, "active": True, "fps": 24.8,
            "glosses": ["TECHNOLOGY"], "finishing_glosses": [], "sentence": "Hello.",
            "finish_pending": False, "speech_text": None, "ctc_hypothesis": None,
            "stats": {},
        }
        arguments.update(overrides)
        hud.draw(**arguments)
        hud._committed_at = -1e6  # settle the commit fade so pixels are stable
        return hud.draw(**arguments)

    def test_default_render_is_byte_identical_to_no_inset(self) -> None:
        self.assertTrue(np.array_equal(self.render(), self.render(top_inset=0)))

    def test_an_inset_moves_the_top_row_down(self) -> None:
        self.assertFalse(np.array_equal(self.render(), self.render(top_inset=56)))

    def test_an_inset_leaves_the_controls_where_they_were(self) -> None:
        plain, inset = self.render(), self.render(top_inset=56)
        top = min(r[1] for r in reel_control_button_rects(1280, 720).values())
        self.assertTrue(np.array_equal(plain[top:], inset[top:]))


class MinimalHudTest(unittest.TestCase):
    """Practice judges one sign: no buffer, no sentence, no reset or finish."""

    def setUp(self) -> None:
        self.frame = np.zeros((720, 1280, 3), np.uint8)
        self.lock = types.SimpleNamespace(hits=2, required_hits=3, suppressed=None)

    def render(self, minimal: bool, **overrides):
        hud = ReelHud(provisional=True)
        arguments = {
            "frame": self.frame.copy(), "latest": observation(),
            "latest_result": result("TECHNOLOGY"), "lock": self.lock,
            "pending": False, "active": True, "fps": 24.8,
            "glosses": ["TECHNOLOGY", "USE", "IMPROVE"], "finishing_glosses": [],
            "sentence": "Let us use technology.", "finish_pending": False,
            "speech_text": None, "ctc_hypothesis": ["I", "LOVE", "YOU"],
            "stats": {"dropped": 3, "observations": 412}, "minimal": minimal,
        }
        arguments.update(overrides)
        hud.draw(**arguments)
        hud._committed_at = -1e6
        return hud.draw(**arguments)

    def test_minimal_differs_from_the_full_overlay(self) -> None:
        self.assertFalse(np.array_equal(self.render(False), self.render(True)))

    def test_the_default_is_still_the_full_overlay(self) -> None:
        hud, other = ReelHud(provisional=True), ReelHud(provisional=True)
        common = {
            "latest": observation(), "latest_result": result("TECHNOLOGY"),
            "lock": self.lock, "pending": False, "active": True, "fps": 24.8,
            "glosses": ["A"], "finishing_glosses": [], "sentence": "Hi.",
            "finish_pending": False, "speech_text": None, "ctc_hypothesis": None,
            "stats": {},
        }
        hud.draw(frame=self.frame.copy(), **common)
        other.draw(frame=self.frame.copy(), minimal=False, **common)
        hud._committed_at = other._committed_at = -1e6
        self.assertTrue(np.array_equal(
            hud.draw(frame=self.frame.copy(), **common),
            other.draw(frame=self.frame.copy(), minimal=False, **common),
        ))

    def test_the_control_row_is_not_painted(self) -> None:
        """Nothing is drawn where RESET and FINISH sit."""
        rects = reel_control_button_rects(1280, 720)
        top = min(box[1] for box in rects.values())
        minimal = self.render(True)
        full = self.render(False)
        self.assertGreater(int(full[top:].sum()), int(minimal[top:].sum()))

    def test_the_buffer_does_not_change_the_minimal_overlay(self) -> None:
        """A growing gloss rail is invisible, so it cannot alter the frame."""
        short = self.render(True, glosses=["ONE"])
        long = self.render(True, glosses=["ONE", "TWO", "THREE", "FOUR", "FIVE"])
        self.assertTrue(np.array_equal(short, long))

    def test_the_sentence_does_not_change_the_minimal_overlay(self) -> None:
        plain = self.render(True, sentence="")
        spoken = self.render(True, sentence="A whole naturalised sentence here.")
        self.assertTrue(np.array_equal(plain, spoken))

    def test_the_running_hypothesis_is_suppressed_too(self) -> None:
        without = self.render(True, ctc_hypothesis=None)
        with_it = self.render(True, ctc_hypothesis=["I", "LOVE", "YOU"])
        self.assertTrue(np.array_equal(without, with_it))


if __name__ == "__main__":
    unittest.main()
