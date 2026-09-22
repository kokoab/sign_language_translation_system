"""The shell driving the real pipeline over a clip.

Slower than the other app tests because it loads the models, and skipped when the
gallery assets have not been built.  It is the test that would catch the reel
script's per-frame shell hook regressing.
"""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import app_shell_v17 as shell_module
from scripts.app_shell_v17 import AppShell

CLIP = REPO / "artifacts/app_assets/gloss_examples/DRINK.mp4"


@unittest.skipUnless(CLIP.is_file(), "run scripts/build_gloss_examples_v17.py first")
class LiveNavigationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="app_integration_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_browsing_mid_session_pauses_without_losing_recognition(self) -> None:
        shell = AppShell(session_root=self.tmp)
        # Visit the gallery and history partway through, then return to the feed.
        script = (
            [255] * 6 + [ord("3")] + [255] * 4 + [ord("5")] + [255] * 4
            + [ord("2")] + [255] * 200
        )
        pages_seen, shown = [], []

        def waitkey(_delay=1):
            pages_seen.append(shell.page)
            return script.pop(0) if script else 255

        from scripts.live_reel_stage1_v17 import (
            build_components, parser, run, validate_args,
        )

        args = parser().parse_args([
            "--video", str(CLIP), "--no-speech", "--no-ollama",
            "--output-root", str(self.tmp),
        ])
        validate_args(args)

        with mock.patch.object(shell_module.cv2, "namedWindow"), \
             mock.patch.object(shell_module.cv2, "setMouseCallback"), \
             mock.patch.object(shell_module.cv2, "destroyWindow"), \
             mock.patch.object(
                 shell_module.cv2, "imshow",
                 side_effect=lambda window, frame: shown.append(shell.page)), \
             mock.patch.object(shell_module.cv2, "waitKey", side_effect=waitkey):
            shell.open()
            shell.page = "LIVE"
            # Pre-built exactly as the app does it, off the camera's critical path.
            result = run(args, shell=shell, prebuilt=build_components(args))
            shell.close()

        order, last = [], None
        for page in pages_seen:
            if page != last:
                order.append(page)
                last = page

        self.assertEqual(order, ["LIVE", "GLOSSES", "HISTORY", "LIVE"])
        self.assertIn("GLOSSES", shown)
        self.assertIn("HISTORY", shown)
        self.assertEqual(result["hypothesis"], ["DRINK"])
        self.assertEqual(
            len(shell_module.store.refresh(self.tmp)["sessions"]), 1
        )


if __name__ == "__main__":
    unittest.main()
