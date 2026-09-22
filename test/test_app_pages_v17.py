"""Page rendering and hit-testing.  Headless: no camera, no window, no models."""

from pathlib import Path
import sys
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import app_pages_v17 as pages
from scripts.reel_hud_v17 import NAV_PAGES, clicked_nav, nav_height, nav_rects

SIZES = ((1280, 720), (1024, 640), (1920, 1080))


def catalog_row(label="ANGRY", **overrides):
    row = {
        "canonical_label": label, "category": "descriptions_and_states",
        "citizen_asl_lex_code": "B_01_070", "participant": "P30",
        "seconds": 1.12, "candidates_considered": 14, "notes": [],
        "assets": {"poster": f"{label}.jpg", "loop": f"{label}.mp4", "fps": 30.0},
    }
    row.update(overrides)
    return row


def swatch(width, height, value=128):
    return np.full((max(1, height), max(1, width), 3), value, np.uint8)


class HomeTest(unittest.TestCase):
    def test_the_animated_backdrop_renders_and_keeps_the_actions(self) -> None:
        backdrop = pages.Backdrop(1280, 720)
        first, rects = pages.render_home(1280, 720, backdrop=backdrop, now=0.0)
        later, _ = pages.render_home(1280, 720, backdrop=backdrop, now=9.0)
        self.assertEqual(first.shape, (720, 1280, 3))
        self.assertIn("LIVE", rects)
        self.assertFalse(np.array_equal(first, later))  # it drifts

    def test_the_backdrop_is_drawn_not_photographed(self) -> None:
        """No corpus imagery belongs on the home screen."""
        backdrop = pages.Backdrop(320, 180)
        frame = backdrop.frame(3.0)
        self.assertEqual(frame.shape, (180, 320, 3))
        self.assertLess(int(frame.max()), 210)  # a wash, never a photograph

    def test_the_backdrop_works_at_awkward_sizes(self) -> None:
        for width, height in ((320, 200), (1920, 1080), (100, 60)):
            frame, _ = pages.render_home(
                width, height, backdrop=pages.Backdrop(width, height), now=2.0,
            )
            self.assertEqual(frame.shape, (height, width, 3))

    def test_hover_changes_the_button_it_is_over(self) -> None:
        plain, rects = pages.render_home(1280, 720)
        box = rects["LIVE"]
        hovered, _ = pages.render_home(
            1280, 720, pointer=((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
        )
        self.assertFalse(np.array_equal(plain, hovered))

    def test_the_entrance_animation_settles(self) -> None:
        early, _ = pages.render_home(1280, 720, now=0.05, entered=0.0)
        settled, _ = pages.render_home(1280, 720, now=5.0, entered=0.0)
        self.assertFalse(np.array_equal(early, settled))

    def test_it_renders_at_every_size_with_every_action(self) -> None:
        for width, height in SIZES:
            frame, rects = pages.render_home(width, height)
            self.assertEqual(frame.shape, (height, width, 3))
            self.assertEqual(
                set(rects), {"LIVE", "PRACTICE", "GLOSSES", "HISTORY"}
            )
            for left, top, right, bottom in rects.values():
                self.assertGreaterEqual(left, 0)
                self.assertLessEqual(right, width)
                self.assertLessEqual(bottom, height)
                self.assertGreater(top, nav_height(width))

    def test_the_nav_is_clickable_at_wherever_it_was_drawn(self) -> None:
        pages.render_home(1280, 720)
        for page, (left, top, right, bottom) in nav_rects(1280).items():
            self.assertEqual(
                clicked_nav((left + right) // 2, (top + bottom) // 2, 1280), page
            )

    def test_the_nav_sits_centred_rather_than_against_the_left_edge(self) -> None:
        rects = nav_rects(1280)
        boxes = sorted(rects.values())
        left_gap = boxes[0][0]
        right_gap = 1280 - boxes[-1][2]
        self.assertGreater(left_gap, 100)
        self.assertLessEqual(abs(left_gap - right_gap), 2)

    def test_an_unwarmed_app_still_renders(self) -> None:
        frame, rects = pages.render_home(1280, 720, ready=False, status="loading…")
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertIn("LIVE", rects)


class GalleryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = [f"SIGN{index:03d}" for index in range(100)]
        self.catalog = {label: catalog_row(label) for label in self.labels}

    def thumb(self, label, width, height):
        return swatch(width, height)

    def test_it_paints_only_the_visible_tiles(self) -> None:
        frame, rects, rows, visible = pages.render_gallery(
            1280, 720, labels=self.labels, catalog=self.catalog, thumb=self.thumb,
        )
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertEqual(rows, 20)
        self.assertGreater(visible, 0)
        tiles = [name for name in rects if name.startswith("tile:")]
        self.assertEqual(len(tiles), visible * 5)

    def test_a_scrollable_gallery_offers_clickable_arrows(self) -> None:
        """The wheel cannot be trusted on this backend, so arrows must exist."""
        _, rects, _, _ = pages.render_gallery(
            1280, 720, labels=self.labels, catalog=self.catalog, thumb=self.thumb,
        )
        self.assertIn("scroll:up", rects)
        self.assertIn("scroll:down", rects)

    def test_a_gallery_that_fits_needs_no_arrows(self) -> None:
        _, rects, _, _ = pages.render_gallery(
            1280, 720, labels=self.labels[:5], catalog=self.catalog,
            thumb=self.thumb,
        )
        self.assertFalse([name for name in rects if name.startswith("scroll:")])

    def test_scrolling_changes_which_tiles_are_hit_testable(self) -> None:
        _, top_rects, _, _ = pages.render_gallery(
            1280, 720, labels=self.labels, catalog=self.catalog, thumb=self.thumb,
        )
        _, scrolled, _, _ = pages.render_gallery(
            1280, 720, labels=self.labels, catalog=self.catalog, thumb=self.thumb,
            offset=3,
        )
        self.assertIn("tile:SIGN000", top_rects)
        self.assertNotIn("tile:SIGN000", scrolled)
        self.assertIn("tile:SIGN015", scrolled)

    def test_a_missing_thumbnail_does_not_break_the_page(self) -> None:
        frame, rects, _, _ = pages.render_gallery(
            1280, 720, labels=self.labels, catalog=self.catalog,
            thumb=lambda *args: None,
        )
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertTrue(rects)

    def test_an_empty_catalog_renders_guidance(self) -> None:
        frame, rects, rows, _ = pages.render_gallery(
            1280, 720, labels=[], catalog={}, thumb=self.thumb,
        )
        self.assertEqual((rows, rects), (0, {}))
        self.assertEqual(frame.shape, (720, 1280, 3))


class GlossDetailTest(unittest.TestCase):
    def test_it_offers_practice_and_back(self) -> None:
        for width, height in SIZES:
            frame, rects = pages.render_gloss_detail(
                width, height, label="ANGRY", row=catalog_row(),
                frame=swatch(480, 360),
            )
            self.assertEqual(frame.shape, (height, width, 3))
            self.assertEqual(set(rects), {"practice", "back"})
            for left, top, right, bottom in rects.values():
                self.assertLessEqual(right, width)
                self.assertLessEqual(bottom, height)

    def test_it_renders_without_a_clip(self) -> None:
        frame, rects = pages.render_gloss_detail(
            1280, 720, label="ANGRY", row=catalog_row(), frame=None,
        )
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertIn("practice", rects)

    def test_edge_notes_are_surfaced_not_hidden(self) -> None:
        plain, _ = pages.render_gloss_detail(
            1280, 720, label="THINK", row=catalog_row("THINK"), frame=None,
        )
        noted, _ = pages.render_gloss_detail(
            1280, 720, label="THINK",
            row=catalog_row("THINK", notes=["activity starts on the first frame"]),
            frame=None,
        )
        self.assertFalse(np.array_equal(plain, noted))

    def test_a_row_missing_every_field_still_renders(self) -> None:
        frame, _ = pages.render_gloss_detail(
            1280, 720, label="MYSTERY", row={}, frame=None,
        )
        self.assertEqual(frame.shape, (720, 1280, 3))


class HistoryTest(unittest.TestCase):
    def describe(self, row):
        return row.get("stamp", "when"), "1:02", row.get("said", "said")

    def rows(self, count):
        return [
            {"stamp": f"2026092{index}_101500_000000", "said": f"Line {index}.",
             "sentences": ["x"], "complete": index % 2 == 0}
            for index in range(count)
        ]

    def test_rows_are_hit_testable_by_stamp(self) -> None:
        sessions = self.rows(4)
        frame, rects, total, visible = pages.render_history(
            1280, 720, sessions=sessions, describe=self.describe,
        )
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertEqual(total, 4)
        self.assertEqual(len(rects), min(4, visible))
        self.assertIn(f"session:{sessions[0]['stamp']}", rects)

    def test_scrolling_past_the_end_still_renders(self) -> None:
        frame, rects, _, _ = pages.render_history(
            1280, 720, sessions=self.rows(3), describe=self.describe, offset=99,
        )
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertEqual(rects, {})

    def test_an_empty_history_renders_guidance(self) -> None:
        frame, rects, total, _ = pages.render_history(
            1280, 720, sessions=[], describe=self.describe,
        )
        self.assertEqual((total, rects), (0, {}))
        self.assertEqual(frame.shape, (720, 1280, 3))

    def test_a_very_long_sentence_does_not_overflow_the_card(self) -> None:
        long_row = [{"stamp": "s", "said": "WORD " * 200, "sentences": [], "complete": False}]
        frame, _, _, _ = pages.render_history(
            1280, 720, sessions=long_row, describe=lambda r: ("when", "0:05", r["said"]),
        )
        self.assertEqual(frame.shape, (720, 1280, 3))


class LiveOverlayTest(unittest.TestCase):
    """The live page's own furniture, including the nav that used to be missing."""

    def frame(self):
        return np.full((720, 1280, 3), 40, np.uint8)

    def test_the_nav_strip_is_painted_and_nothing_below_it_moves(self) -> None:
        source = self.frame()
        painted, _ = pages.overlay_live(source.copy(), page="LIVE")
        strip = nav_height(1280)
        self.assertFalse(np.array_equal(painted[:strip], source[:strip]))
        self.assertTrue(np.array_equal(painted[strip:], source[strip:]))

    def test_a_practice_round_adds_target_score_and_sample(self) -> None:
        plain, _ = pages.overlay_live(self.frame(), page="PRACTICE")
        full, _ = pages.overlay_live(
            self.frame(), page="PRACTICE", target="EAT", sample=swatch(480, 360),
            index=2, total=10, score=2, attempts=3,
        )
        self.assertFalse(np.array_equal(plain, full))
        self.assertEqual(full.shape, (720, 1280, 3))

    def test_a_round_offers_a_visible_skip_button(self) -> None:
        _, rects = pages.overlay_live(self.frame(), page="PRACTICE", target="EAT")
        self.assertIn("skip", rects)
        left, top, right, bottom = rects["skip"]
        self.assertGreaterEqual(left, 0)
        self.assertLessEqual(right, 1280)
        self.assertLessEqual(bottom, 720)

    def test_the_live_page_offers_no_skip(self) -> None:
        _, rects = pages.overlay_live(self.frame(), page="LIVE")
        self.assertNotIn("skip", rects)

    def test_the_skip_button_stays_inside_every_size(self) -> None:
        for width, height in SIZES:
            _, rects = pages.overlay_live(
                np.zeros((height, width, 3), np.uint8), page="PRACTICE", target="EAT",
            )
            left, top, right, bottom = rects["skip"]
            self.assertGreaterEqual(left, 0)
            self.assertLessEqual(right, width)
            self.assertLessEqual(bottom, height)

    def test_the_sample_overlay_stays_inside_the_frame(self) -> None:
        for width, height in SIZES:
            painted, _ = pages.overlay_live(
                np.zeros((height, width, 3), np.uint8), page="PRACTICE",
                target="EAT", sample=swatch(480, 360),
            )
            self.assertEqual(painted.shape, (height, width, 3))

    def test_a_celebration_brightens_the_whole_frame(self) -> None:
        calm, _ = pages.overlay_live(self.frame(), page="PRACTICE", target="EAT")
        party, _ = pages.overlay_live(
            self.frame(), page="PRACTICE", target="EAT", verdict="matched",
            verdict_gloss="EAT", celebrate=1.0,
        )
        self.assertGreater(int(party.sum()), int(calm.sum()))

    def test_a_miss_is_shown_differently_from_a_plain_prompt(self) -> None:
        prompt, _ = pages.overlay_live(self.frame(), page="PRACTICE", target="EAT")
        missed, _ = pages.overlay_live(
            self.frame(), page="PRACTICE", target="EAT", verdict="missed",
            verdict_gloss="DRINK",
        )
        self.assertFalse(np.array_equal(prompt, missed))


class PracticePagesTest(unittest.TestCase):
    def test_setup_offers_a_round_per_size(self) -> None:
        frame, rects = pages.render_practice_setup(1280, 720, available=100)
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertEqual(
            sorted(rects), sorted(f"start:{n}" for n in pages.PRACTICE_SIZES)
        )

    def test_setup_caps_each_round_at_what_exists(self) -> None:
        _, rects = pages.render_practice_setup(1280, 720, available=3)
        self.assertTrue(all(int(name.split(":")[1]) <= 3 for name in rects))

    def test_setup_with_no_examples_offers_nothing_to_start(self) -> None:
        frame, rects = pages.render_practice_setup(1280, 720, available=0)
        self.assertEqual(rects, {})
        self.assertEqual(frame.shape, (720, 1280, 3))

    def test_the_result_page_offers_another_round_or_an_exit(self) -> None:
        frame, rects = pages.render_practice_result(
            1280, 720, score=8, total=10, missed=["DRINK"],
        )
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertEqual(set(rects), {"again", "done"})

    def test_a_perfect_round_reads_differently_from_a_poor_one(self) -> None:
        perfect, _ = pages.render_practice_result(
            1280, 720, score=10, total=10, missed=[],
        )
        poor, _ = pages.render_practice_result(
            1280, 720, score=1, total=10, missed=["A", "B"],
        )
        self.assertFalse(np.array_equal(perfect, poor))

    def test_a_long_miss_list_does_not_overflow(self) -> None:
        frame, _ = pages.render_practice_result(
            1280, 720, score=0, total=40, missed=[f"SIGN{i}" for i in range(40)],
        )
        self.assertEqual(frame.shape, (720, 1280, 3))




if __name__ == "__main__":
    unittest.main()
