"""Shell routing, pausing and practice mode.  No window is ever opened."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import app_shell_v17 as shell_module
from scripts.app_shell_v17 import NO_KEY, AppShell, NullShell


def make_shell(tmp: Path, labels=("ANGRY", "DRINK", "EAT")) -> AppShell:
    assets = tmp / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    examples = {
        label: {
            "canonical_label": label, "category": "x", "participant": "P1",
            "seconds": 1.0, "candidates_considered": 3, "notes": [],
            "assets": {"poster": f"{label}.jpg", "loop": f"{label}.mp4", "fps": 30.0},
        }
        for label in labels
    }
    (assets / "catalog.json").write_text(json.dumps({"examples": examples}))
    shell = AppShell(assets=assets, session_root=tmp / "sessions")
    shell.sessions = []
    return shell


class BaseShellTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="app_shell_"))
        self.shell = make_shell(self.tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class RoutingTest(BaseShellTest):
    def test_it_starts_on_home_and_reports_itself_paused(self) -> None:
        self.assertEqual(self.shell.page, "HOME")
        self.assertTrue(self.shell.paused)

    def test_only_pages_needing_the_camera_are_unpaused(self) -> None:
        for page in ("HOME", "GLOSSES", "HISTORY", "PRACTICE"):
            self.shell.page = page
            self.assertTrue(self.shell.paused, page)
        self.shell.page = "LIVE"
        self.assertFalse(self.shell.paused)

    def test_a_running_practice_round_keeps_the_camera_working(self) -> None:
        self.shell.start_practice(3)
        self.assertEqual(self.shell.page, "PRACTICE")
        self.assertTrue(self.shell.practising)
        self.assertFalse(self.shell.paused)
        self.shell.practice_state = "done"
        self.assertTrue(self.shell.paused)

    def test_number_keys_switch_pages_and_are_swallowed(self) -> None:
        for key, page in ((ord("3"), "GLOSSES"), (ord("4"), "PRACTICE"),
                          (ord("5"), "HISTORY"), (ord("2"), "LIVE")):
            self.assertEqual(self.shell._consume_key(key), NO_KEY)
            self.assertEqual(self.shell.page, page)

    def test_live_keys_are_passed_through_to_the_recognition_loop(self) -> None:
        self.shell.page = "LIVE"
        for key in (ord("r"), ord("f")):
            self.assertEqual(self.shell._consume_key(key), key)

    def test_those_same_keys_are_swallowed_on_other_pages(self) -> None:
        self.shell.page = "GLOSSES"
        self.assertEqual(self.shell._consume_key(ord("f")), NO_KEY)

    def test_quit_is_reported_once_and_remembered(self) -> None:
        self.assertEqual(self.shell._consume_key(ord("q")), ord("q"))
        self.assertTrue(self.shell.quit)

    def test_escape_steps_back_rather_than_quitting(self) -> None:
        self.shell.page = "GLOSSES"
        self.shell.selected = "ANGRY"
        self.shell._consume_key(27)
        self.assertIsNone(self.shell.selected)
        self.assertEqual(self.shell.page, "GLOSSES")
        self.shell._consume_key(27)
        self.assertEqual(self.shell.page, "HOME")
        self.assertFalse(self.shell.quit)


class LeavingLiveTest(BaseShellTest):
    def test_a_page_switch_resets_the_candidate(self) -> None:
        seen = []
        self.shell.attach("w", [1280, 720], 0.0, lambda a, s: seen.append(a))
        self.shell.page = "LIVE"
        self.shell._go("GLOSSES")
        self.assertEqual(seen, ["reset"])

    def test_attach_no_longer_forces_the_live_page(self) -> None:
        self.shell.attach("w", [1280, 720], 0.0, lambda a, s: None)
        self.assertEqual(self.shell.page, "HOME")

    def test_leaving_practice_clears_the_target_and_round(self) -> None:
        self.shell.attach("w", [1280, 720], 0.0, lambda a, s: None)
        self.shell.start_practice(3)
        self.shell._go("HOME")
        self.assertIsNone(self.shell.target)
        self.assertEqual(self.shell.practice_state, "setup")

    def test_switching_between_other_pages_does_not_reset(self) -> None:
        seen = []
        self.shell.attach("w", [1280, 720], 0.0, lambda a, s: seen.append(a))
        self.shell._go("GLOSSES")
        seen.clear()
        self.shell._go("HISTORY")
        self.assertEqual(seen, [])


class PracticeTest(BaseShellTest):
    def setUpRound(self, *signs):
        self.shell.start_practice(len(signs))
        self.shell.practice_set = list(signs)
        self.shell.target = signs[0]
        self.shell.sound = False

    def test_a_match_scores_celebrates_and_advances(self) -> None:
        self.setUpRound("ANGRY", "DRINK")
        self.shell._check_practice(["ANGRY"])
        self.assertEqual(self.shell.verdict[0], "matched")
        self.assertEqual(self.shell.practice_score, 1)
        self.assertGreater(self.shell.celebrated_at, 0.0)
        self.assertEqual(self.shell.target, "DRINK")   # moved on

    def test_a_different_commit_reports_it_and_keeps_the_target(self) -> None:
        self.setUpRound("ANGRY", "DRINK")
        self.shell._check_practice(["DRINK"])
        self.assertEqual(self.shell.verdict[0], "missed")
        self.assertEqual(self.shell.verdict[1], "DRINK")
        self.assertEqual(self.shell.target, "ANGRY")
        self.assertEqual(self.shell.practice_score, 0)
        self.assertEqual(self.shell.practice_attempts, 1)

    def test_a_provisional_marker_is_ignored_when_matching(self) -> None:
        self.setUpRound("ANGRY")
        self.shell._check_practice(["ANGRY?"])
        self.assertEqual(self.shell.verdict[0], "matched")

    def test_the_round_ends_after_the_last_sign(self) -> None:
        self.setUpRound("ANGRY", "DRINK")
        self.shell._check_practice(["ANGRY"])
        self.shell._check_practice(["ANGRY", "DRINK"])
        self.assertEqual(self.shell.practice_state, "done")
        self.assertEqual(self.shell.practice_score, 2)
        self.assertIsNone(self.shell.target)

    def test_skipping_counts_as_an_attempt_and_is_recorded(self) -> None:
        self.setUpRound("ANGRY", "DRINK")
        self.shell._consume_key(ord("n"))
        self.assertEqual(self.shell.practice_missed, ["ANGRY"])
        self.assertEqual(self.shell.target, "DRINK")

    def test_a_round_never_exceeds_the_available_signs(self) -> None:
        self.shell.start_practice(99)
        self.assertEqual(len(self.shell.practice_set), len(self.shell.labels))

    def test_only_new_commits_are_judged(self) -> None:
        self.setUpRound("ANGRY")
        self.shell._check_practice(["DRINK"])
        self.shell.verdict = None
        self.shell._check_practice(["DRINK"])  # unchanged list
        self.assertIsNone(self.shell.verdict)

    def test_a_reset_shrinking_the_list_does_not_misjudge(self) -> None:
        self.setUpRound("ANGRY")
        self.shell._check_practice(["DRINK", "EAT"])
        self.shell.verdict = None
        self.shell._check_practice([])       # reset
        self.shell._check_practice(["ANGRY"])
        self.assertEqual(self.shell.verdict[0], "matched")

    def test_with_no_target_nothing_is_judged(self) -> None:
        self.shell._check_practice(["DRINK"])
        self.assertIsNone(self.shell.verdict)

    def test_choosing_practice_from_a_tile_starts_a_round_of_that_sign(self) -> None:
        self.shell.page = "GLOSSES"
        self.shell._activate("tile:DRINK")
        self.assertEqual(self.shell.selected, "DRINK")
        self.shell._activate("practice")
        self.assertEqual(self.shell.page, "PRACTICE")
        self.assertEqual(self.shell.practice_set, ["DRINK"])
        self.assertEqual(self.shell.target, "DRINK")
        self.assertIsNone(self.shell.selected)

    def test_the_setup_page_starts_a_sized_round(self) -> None:
        self.shell._go("PRACTICE")
        self.shell._activate("start:2")
        self.assertEqual(len(self.shell.practice_set), 2)
        self.assertTrue(self.shell.practising)


class PracticeControlsTest(BaseShellTest):
    """Inside a round there is no reset, no buffer and no way to finish."""

    def test_practice_asks_for_the_minimal_overlay(self) -> None:
        self.assertFalse(self.shell.minimal_hud)
        self.shell.start_practice(2)
        self.assertTrue(self.shell.minimal_hud)

    def test_the_finish_gesture_is_withheld_during_a_round(self) -> None:
        self.assertTrue(self.shell.allow_finish)
        self.shell.start_practice(2)
        self.assertFalse(self.shell.allow_finish)
        self.shell._go("LIVE")
        self.assertTrue(self.shell.allow_finish)

    def test_reset_and_finish_keys_do_not_reach_the_loop(self) -> None:
        self.shell.start_practice(2)
        for key in (ord("r"), ord("f")):
            self.assertEqual(self.shell._consume_key(key), NO_KEY)

    def test_those_keys_still_reach_the_loop_on_the_live_page(self) -> None:
        self.shell.page = "LIVE"
        for key in (ord("r"), ord("f")):
            self.assertEqual(self.shell._consume_key(key), key)

    def test_a_judged_sign_clears_the_buffer(self) -> None:
        seen = []
        self.shell.attach("w", [1280, 720], 0.0, lambda a, s: seen.append(a))
        self.shell.start_practice(2)
        self.shell.sound = False
        self.shell._check_practice(["WRONG"])
        self.assertEqual(seen, ["reset"])
        seen.clear()
        self.shell._check_practice(["WRONG", self.shell.target])
        self.assertEqual(seen, ["reset"])

    def test_only_one_commit_is_judged_before_the_buffer_clears(self) -> None:
        self.shell.attach("w", [1280, 720], 0.0, lambda a, s: None)
        self.shell.start_practice(3)
        self.shell.sound = False
        self.shell._check_practice(["WRONG", "ALSO_WRONG", "STILL_WRONG"])
        self.assertEqual(self.shell.practice_attempts, 1)

    def test_the_skip_button_and_key_share_one_path(self) -> None:
        self.shell.attach("w", [1280, 720], 0.0, lambda a, s: None)
        self.shell.start_practice(3)
        first = self.shell.target
        self.shell._activate("skip")
        self.assertNotEqual(self.shell.target, first)
        self.assertEqual(self.shell.practice_missed, [first])
        second = self.shell.target
        self.shell._consume_key(ord("n"))
        self.assertEqual(self.shell.practice_missed, [first, second])
        self.assertEqual(self.shell.practice_attempts, 2)

    def test_skipping_the_last_sign_ends_the_round(self) -> None:
        self.shell.start_practice(1)
        self.shell.skip_practice()
        self.assertEqual(self.shell.practice_state, "done")

    def test_skip_does_nothing_outside_a_round(self) -> None:
        self.shell.page = "LIVE"
        self.shell.skip_practice()
        self.assertEqual(self.shell.practice_missed, [])

    def test_skipping_clears_the_buffer_too(self) -> None:
        seen = []
        self.shell.attach("w", [1280, 720], 0.0, lambda a, s: seen.append(a))
        self.shell.start_practice(3)
        self.shell.skip_practice()
        self.assertEqual(seen, ["reset"])

    def test_the_live_page_keeps_its_controls(self) -> None:
        self.shell.page = "LIVE"
        self.assertFalse(self.shell.minimal_hud)
        self.assertTrue(self.shell.allow_finish)


class PresentTest(BaseShellTest):
    def frame(self):
        return np.zeros((720, 1280, 3), np.uint8)

    def test_the_live_page_keeps_the_frame_below_its_navigation(self) -> None:
        self.shell.page = "LIVE"
        with mock.patch.object(shell_module.cv2, "imshow") as show, \
             mock.patch.object(shell_module.cv2, "waitKey", return_value=NO_KEY):
            self.shell.present(self.frame(), [])
        painted = show.call_args[0][1]
        strip = shell_module.nav_height(1280)
        self.assertTrue(np.array_equal(painted[strip:], self.frame()[strip:]))

    def test_another_page_replaces_the_frame_entirely(self) -> None:
        self.shell.page = "HOME"
        with mock.patch.object(shell_module.cv2, "imshow") as show, \
             mock.patch.object(shell_module.cv2, "waitKey", return_value=NO_KEY):
            self.shell.present(self.frame(), [])
        self.assertFalse(np.array_equal(show.call_args[0][1], self.frame()))

    def test_the_live_page_paints_its_navigation(self) -> None:
        """The strip used to be reserved and left empty: no visible back button."""
        self.shell.page = "LIVE"
        with mock.patch.object(shell_module.cv2, "imshow") as show, \
             mock.patch.object(shell_module.cv2, "waitKey", return_value=NO_KEY):
            self.shell.present(self.frame(), [])
        painted = show.call_args[0][1]
        strip = painted[:shell_module.nav_height(1280)]
        self.assertGreater(int(strip.sum()), 0)

    def test_a_practice_target_annotates_the_live_frame(self) -> None:
        self.shell.page = "LIVE"
        self.shell.target = "ANGRY"
        with mock.patch.object(shell_module.cv2, "imshow") as show, \
             mock.patch.object(shell_module.cv2, "waitKey", return_value=NO_KEY):
            self.shell.present(self.frame(), [])
        self.assertFalse(np.array_equal(show.call_args[0][1], self.frame()))

    def test_closing_the_native_window_quits_the_recognition_loop(self) -> None:
        self.shell.page = "LIVE"
        self.shell.live_active = True
        with mock.patch.object(shell_module.cv2, "imshow"), \
             mock.patch.object(shell_module.cv2, "waitKey", return_value=NO_KEY), \
             mock.patch.object(shell_module.cv2, "getWindowProperty", return_value=0):
            self.assertEqual(self.shell.present(self.frame(), []), ord("q"))
        self.assertTrue(self.shell.quit)

    def test_the_nav_strip_is_reserved_only_for_the_app(self) -> None:
        self.assertEqual(NullShell().top_inset, 0)
        self.assertGreater(self.shell.top_inset, 0)


class NullShellTest(unittest.TestCase):
    """The default path must stay exactly what the script did before."""

    def test_it_never_pauses_and_reserves_no_strip(self) -> None:
        null = NullShell()
        self.assertFalse(null.paused)
        self.assertEqual(null.top_inset, 0)

    def test_present_is_imshow_plus_waitkey(self) -> None:
        null = NullShell()
        frame = np.zeros((4, 4, 3), np.uint8)
        with mock.patch.object(shell_module.cv2, "imshow") as show, \
             mock.patch.object(shell_module.cv2, "waitKey", return_value=ord("r")) as wait:
            self.assertEqual(null.present(frame), ord("r"))
        show.assert_called_once()
        wait.assert_called_once_with(1)

    def test_attach_wires_the_reel_controls(self) -> None:
        null = NullShell()
        seen = []
        with mock.patch.object(shell_module.cv2, "namedWindow"), \
             mock.patch.object(shell_module.cv2, "setMouseCallback") as bind:
            null.attach("w", [1280, 720], 0.0, lambda a, s: seen.append(a))
        handler = bind.call_args[0][1]
        rects = shell_module.clicked_reel_control
        with mock.patch.object(shell_module, "clicked_reel_control", return_value="finish"):
            handler(shell_module.cv2.EVENT_LBUTTONUP, 640, 700, 0, None)
        self.assertEqual(seen, ["finish"])
        self.assertIs(shell_module.clicked_reel_control, rects)


class ScrollingTest(BaseShellTest):
    """Four ways to scroll, because the wheel alone cannot be trusted here."""

    def setUp(self) -> None:
        super().setUp()
        self.shell.page = "GLOSSES"
        self.shell.labels = [f"S{i:03d}" for i in range(100)]
        self.shell.catalog = {
            label: {"assets": {}} for label in self.shell.labels
        }

    def test_the_wheel_scrolls_both_ways(self) -> None:
        self.shell._on_mouse(shell_module.cv2.EVENT_MOUSEWHEEL, 600, 400, -120, None)
        self.assertEqual(self.shell.offset, 1)
        self.shell._on_mouse(shell_module.cv2.EVENT_MOUSEWHEEL, 600, 400, 120, None)
        self.assertEqual(self.shell.offset, 0)

    def test_the_arrow_buttons_scroll(self) -> None:
        self.shell._activate("scroll:down")
        self.assertEqual(self.shell.offset, 1)
        self.shell._activate("scroll:up")
        self.assertEqual(self.shell.offset, 0)

    def test_arrow_keys_and_page_keys_scroll(self) -> None:
        self.shell._consume_key(1)            # down arrow on macOS
        self.assertEqual(self.shell.offset, 1)
        self.shell._consume_key(0)            # up arrow
        self.assertEqual(self.shell.offset, 0)
        self.shell._consume_key(ord("]"))     # page down
        self.assertEqual(self.shell.offset, 3)
        self.shell._consume_key(ord("["))
        self.assertEqual(self.shell.offset, 0)

    def test_scrolling_never_goes_above_the_top(self) -> None:
        for _ in range(5):
            self.shell.scroll(-1)
        self.assertEqual(self.shell.offset, 0)

    def test_the_renderer_clamps_a_runaway_offset(self) -> None:
        self.shell.offset = 9999
        self.shell._render()
        self.assertLess(self.shell.offset, 100)

    def test_pages_without_a_list_ignore_scrolling(self) -> None:
        for page in ("HOME", "LIVE", "PRACTICE"):
            self.shell.page = page
            self.shell.offset = 0
            self.shell.scroll(1)
            self.assertEqual(self.shell.offset, 0, page)

    def test_an_open_tile_ignores_scrolling(self) -> None:
        self.shell.selected = "S000"
        self.shell.scroll(1)
        self.assertEqual(self.shell.offset, 0)

    def test_pointer_moves_are_tracked_for_hover(self) -> None:
        self.shell._on_mouse(shell_module.cv2.EVENT_MOUSEMOVE, 42, 99, 0, None)
        self.assertEqual(self.shell.pointer, (42, 99))

    def test_a_pointer_move_is_not_a_click(self) -> None:
        self.shell._on_mouse(shell_module.cv2.EVENT_MOUSEMOVE, 42, 99, 0, None)
        self.assertEqual(self.shell.page, "GLOSSES")
        self.assertIsNone(self.shell.selected)


class BackdropTest(BaseShellTest):
    def test_the_backdrop_is_built_without_any_imagery(self) -> None:
        self.shell.build_backdrop()
        self.assertIsNotNone(self.shell.backdrop)

    def test_home_still_renders_before_the_backdrop_exists(self) -> None:
        self.shell.page = "HOME"
        self.assertIsNone(self.shell.backdrop)
        self.assertEqual(self.shell._render().shape, (720, 1280, 3))


class CatalogTest(BaseShellTest):
    def test_a_missing_catalog_leaves_an_empty_gallery_not_a_crash(self) -> None:
        empty = AppShell(assets=self.tmp / "nope", session_root=self.tmp / "s")
        self.assertEqual(empty.labels, [])

    def test_labels_come_from_the_catalog_sorted(self) -> None:
        self.assertEqual(self.shell.labels, ["ANGRY", "DRINK", "EAT"])


if __name__ == "__main__":
    unittest.main()
