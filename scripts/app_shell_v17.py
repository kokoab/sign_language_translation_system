#!/usr/bin/env python3
"""The app shell: one window, four pages, and the reel pipeline behind the live one.

The live page runs the streaming segmental Reel (`scripts/live_segmental_v17.py`) by default;
`--classic-reel` restores the previous cascade (`scripts/live_reel_stage1_v17.py`).  Either
loop owns itself and calls `present()` once per frame; the shell decides what that frame
becomes.  While a page other than LIVE is showing, the shell reports itself
paused and the loop skips its vision and model work, so browsing costs nothing.

Run it with the same flags as the reel script:

    venv/bin/python scripts/app_shell_v17.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import app_pages_v17 as pages
from scripts import app_sessions_v17 as store
from scripts.reel_hud_v17 import clicked_nav, clicked_reel_control, nav_height

WINDOW = "Sign Language Translation"
ASSETS = REPO / "artifacts/app_assets/gloss_examples"
NO_KEY = 255
QUIT_KEYS = (ord("q"), ord("Q"))
PAGE_KEYS = {
    ord("1"): "HOME", ord("2"): "LIVE", ord("3"): "GLOSSES",
    ord("4"): "PRACTICE", ord("5"): "HISTORY",
}
# macOS reports arrow keys as 0-3 after the 0xFF mask; the letters are for anyone
# without arrows to hand.
SCROLL_UP_KEYS = (0, ord("w"), ord("k"))
SCROLL_DOWN_KEYS = (1, ord("s"), ord("j"))
PAGE_UP_KEYS = (ord("["), 11)
PAGE_DOWN_KEYS = (ord("]"), 12)
VERDICT_SECONDS = 3.0
CELEBRATE_SECONDS = 1.4
CORRECT_SOUND = "/System/Library/Sounds/Glass.aiff"


def play(path: str) -> None:
    """Fire-and-forget chime.  Silent wherever afplay or the file is missing."""
    player = shutil.which("afplay")
    if not player or not Path(path).is_file():
        return
    try:
        subprocess.Popen(
            [player, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def _window_closed(window: str) -> bool:
    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return False


class NullShell:
    """What the reel script did before a shell existed: one window, no pages."""

    paused = False
    top_inset = 0
    minimal_hud = False
    allow_finish = True
    live_active = True

    def __init__(self) -> None:
        self._window = WINDOW

    def attach(self, window, display_size, started, on_control) -> None:
        self._window = window
        cv2.namedWindow(window)

        def on_mouse(event, x, y, _flags, _parameter):
            if event != cv2.EVENT_LBUTTONUP:
                return
            action = clicked_reel_control(x, y, display_size[0], display_size[1])
            if action is not None:
                on_control(action, time.perf_counter() - started)

        cv2.setMouseCallback(window, on_mouse)

    def present(self, frame, glosses=()) -> int:
        cv2.imshow(self._window, frame)
        key = cv2.waitKey(1) & 0xFF
        if getattr(self, "live_active", False) and _window_closed(self._window):
            return ord("q")
        return key


class AppShell:
    """Owns the window and the page state; the recognition loop owns the frames."""

    def __init__(
        self, *, assets: Path = ASSETS, session_root: Path = store.SESSION_ROOT,
        width: int = 1280, height: int = 720, columns: int = 5,
        sound: bool = True,
    ) -> None:
        self.assets = assets
        self.session_root = session_root
        self.width, self.height = width, height
        self.columns = columns

        self.page = "HOME"
        self.quit = False
        self.live_active = False
        self.ready = False
        self.status = "warming up…"

        self.catalog = pages_catalog(assets)
        self.labels = sorted(self.catalog)
        self.sessions: list[dict] = []
        self.offset = 0
        self.selected: str | None = None
        self.target: str | None = None
        self.verdict: tuple[str, str, float] | None = None

        # Practice is a drill: a fixed set of signs, one at a time, with a score.
        self.practice_state = "setup"        # setup -> running -> done
        self.practice_set: list[str] = []
        self.practice_index = 0
        self.practice_score = 0
        self.practice_attempts = 0
        self.practice_missed: list[str] = []
        self.celebrated_at = 0.0
        self.sound = sound
        self.components = None
        self.backdrop: pages.Backdrop | None = None
        self.entered = time.perf_counter()

        self._rects: pages.Rects = {}
        self._thumbs: dict[tuple[str, int, int], np.ndarray | None] = {}
        self._clip: cv2.VideoCapture | None = None
        self._clip_label: str | None = None
        self._clip_frame: np.ndarray | None = None
        self._seen_glosses = 0
        self.pointer = (-1, -1)
        self._display_size = [width, height]
        self._on_control = None
        self._started = time.perf_counter()

    # --- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        self.refresh_sessions()

    def close(self) -> None:
        self._release_clip()
        cv2.destroyWindow(WINDOW)

    def refresh_sessions(self) -> None:
        self.sessions = store.refresh(self.session_root).get("sessions", [])

    def build_backdrop(self) -> None:
        """The home screen's drawn background; cheap enough to make eagerly."""
        self.backdrop = pages.Backdrop(self.width, self.height)

    @property
    def practising(self) -> bool:
        return self.page == "PRACTICE" and self.practice_state == "running"

    @property
    def paused(self) -> bool:
        """Only the pages that actually need the camera keep it working."""
        return not (self.page == "LIVE" or self.practising)

    @property
    def minimal_hud(self) -> bool:
        """A practice round judges one sign: no buffer, no sentence, no controls."""
        return self.practising

    @property
    def allow_finish(self) -> bool:
        """Withholds the ten-finger gesture; the buttons and keys go with it."""
        return not self.practising

    @property
    def top_inset(self) -> int:
        return nav_height(self._display_size[0])

    # --- the reel pipeline's view of the shell -----------------------------

    def attach(self, window, display_size, started, on_control) -> None:
        """Called by the reel loop once, instead of it creating its own window."""
        self._display_size = display_size
        self._on_control = on_control
        self._started = started
        self.live_active = True
        self._seen_glosses = 0

    def present(self, frame: np.ndarray, glosses=()) -> int:
        """One frame from the recognition loop.  Returns the key it should see."""
        if self.paused:
            self._show(self._render())
        else:
            self._check_practice(list(glosses))
            shown, self._rects = self._live_frame(frame)
            self._show(shown)
        key = cv2.waitKey(1) & 0xFF
        if self.live_active and _window_closed(WINDOW):
            self.quit = True
            return ord("q")
        return self._consume_key(key)

    def _live_frame(self, frame: np.ndarray) -> tuple:
        verdict = self._live_verdict()
        practising = self.practising
        return pages.overlay_live(
            frame,
            page="PRACTICE" if practising else "LIVE",
            target=self.target if practising else None,
            sample=self._sample_frame(self.target) if practising else None,
            verdict=verdict,
            verdict_gloss=self.verdict[1] if self.verdict else "",
            celebrate=self._celebration(),
            index=self.practice_index if practising else 0,
            total=len(self.practice_set) if practising else 0,
            score=self.practice_score,
            attempts=self.practice_attempts,
        )

    def _celebration(self) -> float:
        if not self.celebrated_at:
            return 0.0
        age = time.perf_counter() - self.celebrated_at
        if age > CELEBRATE_SECONDS:
            self.celebrated_at = 0.0
            return 0.0
        return 1.0 - (age / CELEBRATE_SECONDS) ** 2

    # --- the standalone browsing loop (before the camera starts) -----------

    def browse(self) -> str:
        """Keep the pages interactive until the models are built.

        Only used before the recognition loop starts.  Once it does, that loop
        drives every page through `present()` and this is never entered again.
        """
        while not self.ready:
            self._show(self._render())
            key = self._consume_key(cv2.waitKey(16) & 0xFF)
            if self.quit or key in QUIT_KEYS:
                self.quit = True
                return "quit"
        return "ready"

    # --- input -------------------------------------------------------------

    def _on_mouse(self, event, x, y, flags, _parameter) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self.pointer = (x, y)
            return
        if event in (cv2.EVENT_MOUSEWHEEL, getattr(cv2, "EVENT_MOUSEHWHEEL", -1)):
            # Cocoa may never send these; the on-screen arrows are the guarantee.
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except (AttributeError, cv2.error):
                delta = flags
            self.scroll(-1 if delta > 0 else 1)
            return
        if event != cv2.EVENT_LBUTTONUP:
            return
        if self.page == "LIVE" and self._on_control is not None:
            action = clicked_reel_control(
                x, y, self._display_size[0], self._display_size[1]
            )
            if action is not None:
                self._on_control(action, time.perf_counter() - self._started)
                return
        page = clicked_nav(x, y, self._display_size[0])
        if page is not None:
            self._go(page)
            return
        for name, (left, top, right, bottom) in self._rects.items():
            if left <= x <= right and top <= y <= bottom:
                self._activate(name)
                return

    def scroll(self, rows: int) -> None:
        """Move the visible window; the renderer clamps against the real total."""
        if self.page in ("GLOSSES", "HISTORY") and not self.selected:
            self.offset = max(0, self.offset + rows)

    def _activate(self, name: str) -> None:
        if name.startswith("scroll:"):
            self.scroll(-1 if name.endswith("up") else 1)
            return
        if name == "skip":
            self.skip_practice()
            return
        if name in ("LIVE", "GLOSSES", "HISTORY", "PRACTICE"):
            self._go(name)
        elif name.startswith("tile:"):
            self.selected = name.split(":", 1)[1]
        elif name == "back":
            self.selected = None
            self._release_clip()
        elif name.startswith("start:"):
            self.start_practice(int(name.split(":", 1)[1]))
        elif name == "again":
            self.start_practice(len(self.practice_set) or 5)
        elif name == "done":
            self.practice_state = "setup"
            self._go("HOME")
        elif name == "practice" and self.selected:
            # One sign straight from its tile: a round of exactly that gloss.
            chosen = self.selected
            self.selected = None
            self.start_practice(1)
            self.practice_set = [chosen]
            self.target = chosen

    def _consume_key(self, key: int) -> int:
        if key == NO_KEY:
            return NO_KEY
        if key in QUIT_KEYS:
            self.quit = True
            return ord("q")
        if key in PAGE_KEYS:
            self._go(PAGE_KEYS[key])
            return NO_KEY
        if key == 27:  # escape steps back rather than quitting
            if self.selected:
                self.selected = None
                self._release_clip()
            elif self.practising:
                self.practice_state = "setup"
                self.target = None
                self._release_clip()
            else:
                self._go("HOME")
            return NO_KEY
        if self.page in ("GLOSSES", "HISTORY") and not self.selected:
            if key in SCROLL_UP_KEYS:
                self.scroll(-1)
            elif key in SCROLL_DOWN_KEYS:
                self.scroll(1)
            elif key in PAGE_UP_KEYS:
                self.scroll(-3)
            elif key in PAGE_DOWN_KEYS:
                self.scroll(3)
            return NO_KEY
        if key in (ord("n"), ord(" ")) and self.practising:
            self.skip_practice()
            return NO_KEY
        if not self.paused:
            if self.practising and key in (ord("r"), ord("f")):
                return NO_KEY  # no manual reset or finish inside a round
            return key  # RESET / FINISH belong to the recognition loop
        return NO_KEY

    def _go(self, page: str) -> None:
        if page == self.page:
            return
        was_running = not self.paused
        self.page = page
        self.entered = time.perf_counter()
        self.offset = 0
        self.selected = None
        self._release_clip()
        if was_running and self._on_control is not None:
            # A half-grown candidate must not survive a trip to another page.
            self._on_control("reset", time.perf_counter() - self._started)
        if page == "HISTORY":
            self.refresh_sessions()
        if page == "PRACTICE":
            # Coming back to a finished round should offer a fresh one.
            if self.practice_state == "running":
                self._seen_glosses = 0
        else:
            self.target = None
            self.verdict = None
            self.celebrated_at = 0.0
            if self.practice_state == "running":
                self.practice_state = "setup"

    # --- practice mode -----------------------------------------------------

    def start_practice(self, size: int) -> None:
        pool = list(self.labels)
        random.shuffle(pool)
        self.practice_set = pool[:max(1, min(size, len(pool)))]
        self.practice_index = 0
        self.practice_score = 0
        self.practice_attempts = 0
        self.practice_missed = []
        self.practice_state = "running"
        self.page = "PRACTICE"
        self.target = self.practice_set[0]
        self.verdict = None
        self.celebrated_at = 0.0
        self._seen_glosses = 0
        self._release_clip()

    def skip_practice(self) -> None:
        """Pass on the current sign; it counts as an attempt and is reported."""
        if not self.practising or not self.target:
            return
        self.practice_attempts += 1
        self.practice_missed.append(self.target)
        self.verdict = None
        self._advance_practice()
        self._clear_buffer()

    def _advance_practice(self) -> None:
        self.practice_index += 1
        if self.practice_index >= len(self.practice_set):
            self.practice_state = "done"
            self.target = None
            self._release_clip()
            return
        self.target = self.practice_set[self.practice_index]
        self._release_clip()

    def _clear_buffer(self) -> None:
        """Empty the recognition buffer without showing a reset control.

        The queued action is drained by the loop on its next live frame, which is
        also when the gloss list shrinks and `_seen_glosses` follows it down.
        """
        if self._on_control is not None:
            self._on_control("reset", time.perf_counter() - self._started)

    def _check_practice(self, glosses: list[str]) -> None:
        if not self.target:
            self._seen_glosses = len(glosses)
            return
        if len(glosses) > self._seen_glosses:
            for gloss in glosses[self._seen_glosses:]:
                clean = str(gloss).rstrip("?")
                self.practice_attempts += 1
                if clean == self.target:
                    self.practice_score += 1
                    self.verdict = ("matched", clean, time.perf_counter())
                    self.celebrated_at = time.perf_counter()
                    if self.sound:
                        play(CORRECT_SOUND)
                    self._advance_practice()
                else:
                    self.verdict = ("missed", clean, time.perf_counter())
                # Judged either way: the next attempt starts from nothing.
                self._clear_buffer()
                break
            self._seen_glosses = len(glosses)
        elif len(glosses) < self._seen_glosses:
            self._seen_glosses = len(glosses)

    def _live_verdict(self) -> str | None:
        if not self.verdict:
            return None
        if time.perf_counter() - self.verdict[2] > VERDICT_SECONDS:
            self.verdict = None
            return None
        return self.verdict[0]

    # --- rendering ---------------------------------------------------------

    def _render(self) -> np.ndarray:
        width, height = self._display_size
        if self.page == "GLOSSES" and self.selected:
            row = self.catalog.get(self.selected, {})
            frame, rects = pages.render_gloss_detail(
                width, height, label=self.selected, row=row,
                frame=self._sample_frame(self.selected),
            )
            self._rects = rects
            return frame
        if self.page == "GLOSSES":
            frame, rects, rows, visible = pages.render_gallery(
                width, height, labels=self.labels, catalog=self.catalog,
                thumb=self._thumb, offset=self.offset, columns=self.columns,
            )
            self.offset = max(0, min(self.offset, max(0, rows - visible)))
            self._rects = rects
            return frame
        if self.page == "PRACTICE" and self.practice_state == "done":
            frame, rects = pages.render_practice_result(
                width, height, score=self.practice_score,
                total=len(self.practice_set), missed=self.practice_missed,
            )
            self._rects = rects
            return frame
        if self.page == "PRACTICE":
            frame, rects = pages.render_practice_setup(
                width, height, available=len(self.labels),
            )
            self._rects = rects
            return frame
        if self.page == "HISTORY":
            frame, rects, total, visible = pages.render_history(
                width, height, sessions=self.sessions, describe=store.describe,
                offset=self.offset,
            )
            self.offset = max(0, min(self.offset, max(0, total - visible)))
            self._rects = rects
            return frame
        frame, rects = pages.render_home(
            width, height, status=self.status, ready=self.ready,
            session_count=len(self.sessions), gloss_count=len(self.labels),
            backdrop=self.backdrop, now=time.perf_counter(),
            entered=self.entered, pointer=self.pointer,
        )
        self._rects = rects
        return frame

    def _show(self, frame: np.ndarray) -> None:
        self._display_size[:] = [frame.shape[1], frame.shape[0]]
        cv2.imshow(WINDOW, frame)

    # --- media -------------------------------------------------------------

    def _thumb(self, label: str, width: int, height: int):
        key = (label, width, height)
        if key not in self._thumbs:
            row = self.catalog.get(label, {})
            name = (row.get("assets") or {}).get("poster")
            image = cv2.imread(str(self.assets / name)) if name else None
            if image is not None and width > 0 and height > 0:
                image = cv2.resize(
                    image, (width, height), interpolation=cv2.INTER_AREA
                )
            self._thumbs[key] = image
        return self._thumbs[key]

    def _sample_frame(self, label: str | None):
        """The reference clip, advanced one frame per paint and looping forever."""
        if not label:
            return None
        row = self.catalog.get(label, {})
        name = (row.get("assets") or {}).get("loop")
        if not name:
            return self._thumb(label, 0, 0)
        if self._clip_label != label:
            self._release_clip()
            self._clip = cv2.VideoCapture(str(self.assets / name))
            self._clip_label = label
        if self._clip is None or not self._clip.isOpened():
            return self._thumb(label, 0, 0)
        ok, frame = self._clip.read()
        if not ok:
            self._clip.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._clip.read()
        if ok:
            self._clip_frame = frame
        if self._clip_frame is not None:
            return self._clip_frame
        return self._thumb(label, 0, 0)

    def _release_clip(self) -> None:
        if self._clip is not None:
            self._clip.release()
        self._clip = None
        self._clip_label = None
        self._clip_frame = None


def pages_catalog(assets: Path) -> dict:
    import json

    path = assets / "catalog.json"
    try:
        return json.loads(path.read_text()).get("examples", {})
    except (OSError, ValueError):
        return {}


def uses_segmental(args: argparse.Namespace) -> bool:
    """The streaming segmental Reel is the default live mode (user decision 2026-09-28)."""
    return not any(getattr(args, name, None) for name in (
        "classic_reel", "boundary_checkpoint", "renz_buffered", "familiar_ctc_checkpoint"))


def warm(shell: AppShell, args: argparse.Namespace) -> None:
    """Build the models behind the home screen instead of in front of the camera.

    Importing the module is about a second; constructing the models is about
    eleven.  Warming only the import, as this used to, left the whole wait in
    front of the first camera start.  Nothing here touches a window or the
    camera, so it is safe off the main thread.
    """
    try:
        shell.status = "loading models…"
        if uses_segmental(args):
            from scripts.live_segmental_v17 import SegmentalRecognizer
            shell.components = SegmentalRecognizer(args)
        elif getattr(args, "boundary_checkpoint", None):
            from scripts.live_boundary_v17 import BoundaryRecognizer
            shell.components = BoundaryRecognizer(args)
        elif getattr(args, "renz_buffered", False):
            from scripts.live_renz_v17 import BufferedRenz
            shell.components = BufferedRenz(args)
        elif getattr(args, "familiar_ctc_checkpoint", None):
            import torch
            from active.v17.familiar_live_v17 import FamiliarRecognizer
            device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
            if device == "auto": device = "cpu"
            shell.components = FamiliarRecognizer(args.familiar_ctc_checkpoint, device=device)
        else:
            from scripts.live_reel_stage1_v17 import build_components
            shell.components = build_components(args)
        shell.build_backdrop()
        shell.status = ""
        shell.ready = True
    except Exception as error:  # a demo must say what broke, not vanish
        shell.status = f"model load failed: {type(error).__name__}"
        shell.ready = False


def parser() -> argparse.ArgumentParser:
    from scripts.live_reel_stage1_v17 import parser as reel_parser

    value = reel_parser()
    value.description = __doc__
    value.set_defaults(output_root=store.SESSION_ROOT)
    candidates = value.add_mutually_exclusive_group()
    candidates.add_argument("--segmental", action="store_true", help="streaming segmental Reel (the default; kept for explicit launches)")
    candidates.add_argument("--classic-reel", action="store_true", help="the previous Reel cascade live pipeline")
    candidates.add_argument("--boundary-checkpoint", type=Path, help="opt-in ASL temporal boundary candidate with Reel recognition")
    candidates.add_argument("--renz-buffered", action="store_true", help="experimental 4-second buffered Renz boundaries with Reel classification")
    candidates.add_argument("--familiar-ctc-checkpoint", type=Path, help="opt-in reviewed familiar continuous CTC candidate")
    value.add_argument("--no-fingerspelling", action="store_true",
                       help="segmental Reel: recognise the 100 signs only (no letter head, letter boundary or spelled words)")
    value.add_argument("--stage3-torch", action="store_true",
                       help="render sentences with the PyTorch T5 instead of its Core ML export")
    value.add_argument("--gallery-columns", type=int, default=5)
    value.add_argument("--no-practice-sound", action="store_true")
    value.add_argument("--assets", type=Path, default=ASSETS)
    return value


def main() -> None:
    args = parser().parse_args()
    if uses_segmental(args):
        # Core ML pipeline: skip coremltools' TensorFlow (and, unless asked, transformers) probes.
        from active.v17.coreml_runtime_v17 import lightweight_imports
        lightweight_imports(allow_transformers=args.stage3_torch or args.naturalizer != "tiny")
    shell = AppShell(
        assets=args.assets, session_root=args.output_root,
        columns=args.gallery_columns, sound=not args.no_practice_sound,
    )
    shell.open()
    threading.Thread(target=warm, args=(shell, args), daemon=True).start()
    try:
        # Home is interactive from the first frame; the models arrive behind it.
        if shell.browse() == "quit":
            return
        if uses_segmental(args):
            from scripts.live_segmental_v17 import run as segmental_run
            segmental_run(args, shell, shell.components)
            return
        if args.boundary_checkpoint:
            from scripts.live_boundary_v17 import run as boundary_run
            boundary_run(args, shell, shell.components)
            return
        if args.renz_buffered:
            from scripts.live_renz_v17 import run as renz_run
            renz_run(args, shell, shell.components)
            return
        if args.familiar_ctc_checkpoint:
            from scripts.live_continuous_v17 import main as continuous_main
            live_args = ["--familiar-ctc", "--checkpoint", str(args.familiar_ctc_checkpoint),
                         "--camera", str(args.camera), "--device", str(shell.components.device),
                         "--utterance-gap-seconds", "0"]
            if args.video: live_args += ["--video", str(args.video)]
            if args.input_mirrored: live_args += ["--input-mirrored"]
            if args.rotation: live_args += ["--rotation", str(int(args.rotation))]
            continuous_main(live_args, shell=shell, prebuilt=shell.components)
            return
        from scripts.live_reel_stage1_v17 import run, validate_args

        validate_args(args)
        # One session for the life of the app.  Every page is a state inside
        # this loop, so nothing is ever constructed twice and no page switch
        # costs more than a frame.
        run(args, shell=shell, prebuilt=shell.components)
    finally:
        shell.live_active = False
        shell.close()


if __name__ == "__main__":
    main()
