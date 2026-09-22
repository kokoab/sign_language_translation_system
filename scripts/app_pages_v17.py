#!/usr/bin/env python3
"""The app's non-camera pages: home, the 100-gloss gallery, and session history.

Every page is a pure function of its state: it returns the frame to show and the
rect table it drew, so hit-testing can never disagree with what is on screen.  All
painting goes through `scripts/reel_hud_v17.py`; this module owns no colours, fonts
or primitives of its own.
"""

from __future__ import annotations

import math
from pathlib import Path
import random
import sys
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import ImageDraw

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.reel_hud_v17 import (
    ACCENT,
    AMBER,
    CHIP_ALPHA,
    INK,
    MUTED,
    PANEL_ALPHA,
    WHITE,
    Type,
    chip,
    chip_size,
    draw_nav,
    draw_shadowed_text,
    draw_text,
    frame_canvas,
    grid_rects,
    nav_height,
    panel,
    row_rects,
    scale,
    to_frame,
    wrap,
)

Rects = dict[str, tuple[int, int, int, int]]
Thumb = Callable[[str, int, int], Optional[np.ndarray]]


def _background(width: int, height: int) -> np.ndarray:
    """A page canvas as BGR pixels, so image tiles can be pasted before painting."""
    return np.full((height, width, 3), INK[::-1], np.uint8)


def _paste(background: np.ndarray, image: np.ndarray | None, box) -> None:
    if image is None:
        return
    left, top, right, bottom = box
    height, width = bottom - top, right - left
    if height <= 0 or width <= 0:
        return
    resized = image
    if image.shape[0] != height or image.shape[1] != width:
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    background[top:bottom, left:right] = resized


def _button(
    image, draw, box, label: str, scale_value: float, *,
    primary: bool = False, enabled: bool = True, glow: bool = False,
    alpha: int = 255,
) -> None:
    left, top, right, bottom = box
    if primary and enabled:
        fill, color, fill_alpha, outline = ACCENT, INK, 235, None
    elif enabled:
        fill, color, fill_alpha, outline = INK, WHITE, PANEL_ALPHA, WHITE
    else:
        fill, color, fill_alpha, outline = INK, MUTED, CHIP_ALPHA, None
    radius = (bottom - top) // 2
    fade = max(0.0, min(1.0, alpha / 255))
    if glow and enabled:
        # A halo rather than a size change, so nothing reflows under the pointer.
        for step in (3, 2, 1):
            spread = int(step * 4 * scale_value)
            panel(
                draw,
                (left - spread, top - spread, right + spread, bottom + spread),
                radius + spread, fill=ACCENT if primary else WHITE,
                alpha=int(16 * fade),
            )
    panel(
        draw, box, radius, fill=fill, alpha=int(fill_alpha * fade), outline=outline,
    )
    draw_text(
        image, ((left + right) // 2, (top + bottom) // 2), label,
        Type(int(16 * scale_value), "Semibold"), color, alpha=int(255 * fade),
        anchor="mm",
    )


def _footer(
    image, draw, width: int, height: int, scale_value: float, note: str,
) -> None:
    draw_text(
        image, (int(24 * scale_value), height - int(20 * scale_value)), note,
        Type(int(12 * scale_value), "Medium"), MUTED, anchor="lb",
    )


def _empty(image, draw, width: int, top: int, bottom: int, scale_value: float,
           title: str, hint: str) -> None:
    middle = (top + bottom) // 2
    draw_text(
        image, (width // 2, middle), title,
        Type(int(22 * scale_value), "Semibold"), MUTED, anchor="mm",
    )
    draw_text(
        image, (width // 2, middle + int(30 * scale_value)), hint,
        Type(int(14 * scale_value), "Medium"), MUTED, anchor="mm",
    )


def _scrollbar(
    image, draw, width: int, top: int, bottom: int, scale_value: float,
    offset: int, visible_rows: int, total_rows: int,
) -> Rects:
    """Track, thumb and two clickable arrows.

    The arrows exist because OpenCV's macOS backend cannot be relied on to deliver
    wheel events, and an affordance nobody can see is worse than none.
    """
    if total_rows <= visible_rows or visible_rows <= 0:
        return {}
    button = int(22 * scale_value)
    x = width - int(26 * scale_value)
    right = x + int(18 * scale_value)
    up = (x, top, right, top + button)
    down = (x, bottom - button, right, bottom)

    track_top, track_bottom = top + button + int(6 * scale_value), down[1] - int(
        6 * scale_value
    )
    track = max(1, track_bottom - track_top)
    thumb = max(int(28 * scale_value), int(track * visible_rows / total_rows))
    span = max(1, total_rows - visible_rows)
    position = track_top + int((track - thumb) * min(1.0, offset / span))
    bar_x = x + int(7 * scale_value)
    panel(draw, (bar_x, track_top, bar_x + int(4 * scale_value), track_bottom), 2,
          alpha=80)
    panel(
        draw, (bar_x, position, bar_x + int(4 * scale_value), position + thumb), 2,
        fill=MUTED, alpha=200,
    )

    for box, glyph, live in (
        (up, "\u25b2", offset > 0), (down, "\u25bc", offset < span),
    ):
        panel(draw, box, int(5 * scale_value), alpha=PANEL_ALPHA,
              outline=MUTED if live else None)
        draw_text(
            image, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), glyph,
            Type(int(11 * scale_value), "Semibold"),
            WHITE if live else MUTED, anchor="mm",
        )
    return {"scroll:up": up, "scroll:down": down}


# --- home -------------------------------------------------------------------

HOME_ACTIONS = (
    ("LIVE", "START CAMERA"),
    ("GLOSSES", "THE 100 SIGNS"),
    ("HISTORY", "SESSION HISTORY"),
)


# --- the 100 signs ----------------------------------------------------------

def render_gallery(
    width: int, height: int, *, labels: list[str], catalog: dict,
    thumb: Thumb, offset: int = 0, columns: int = 5,
) -> tuple[np.ndarray, Rects, int, int]:
    """Poster grid.  Returns the frame, its rects, total rows and visible rows."""
    scale_value = scale(width)
    background = _background(width, height)
    top = nav_height(width) + int(52 * scale_value)
    bottom = height - int(34 * scale_value)

    boxes, rows = grid_rects(
        len(labels), width, top, bottom, scale_value, columns=columns, offset=offset,
    )
    caption = int(26 * scale_value)
    for index, box in boxes.items():
        left, box_top, right, box_bottom = box
        _paste(background, thumb(labels[index], right - left, box_bottom - box_top - caption),
               (left, box_top, right, box_bottom - caption))

    image = frame_canvas(background)
    draw = ImageDraw.Draw(image, "RGBA")
    rects: Rects = {}
    for index, box in boxes.items():
        left, box_top, right, box_bottom = box
        label = labels[index]
        panel(
            draw, (left, box_bottom - caption, right, box_bottom), 0,
            fill=INK, alpha=225,
        )
        style = Type(int(13 * scale_value), "Semibold")
        rows_of_text = wrap(label, style, right - left - int(12 * scale_value), 1)
        draw_text(
            image, ((left + right) // 2, box_bottom - caption // 2),
            rows_of_text[0] if rows_of_text else label, style, WHITE, anchor="mm",
        )
        rects[f"tile:{label}"] = box

    draw_nav(image, draw, width, "GLOSSES")
    draw_text(
        image, (int(24 * scale_value), nav_height(width) + int(24 * scale_value)),
        f"{len(labels)} signs the model can recognise",
        Type(int(15 * scale_value), "Semibold"), WHITE, anchor="lm",
    )
    if not labels:
        _empty(
            image, draw, width, top, bottom, scale_value, "No examples built yet",
            "Run scripts/build_gloss_examples_v17.py",
        )

    visible_rows = len({index // columns for index in boxes})
    rects.update(
        _scrollbar(
            image, draw, width, top, bottom, scale_value, offset, visible_rows, rows,
        )
    )
    _footer(
        image, draw, width, height, scale_value,
        "Scroll with the arrows, the wheel, ↑↓ or [ ]",
    )
    return to_frame(image), rects, rows, visible_rows


def render_gloss_detail(
    width: int, height: int, *, label: str, row: dict,
    frame: np.ndarray | None = None,
) -> tuple[np.ndarray, Rects]:
    scale_value = scale(width)
    background = _background(width, height)
    top = nav_height(width) + int(28 * scale_value)
    bottom = height - int(96 * scale_value)

    video_width = int(min(width * 0.46, (bottom - top) * 4 / 3))
    video_height = int(video_width * 3 / 4)
    video_left = int(60 * scale_value)
    video_top = top + max(0, (bottom - top - video_height) // 2)
    video_box = (video_left, video_top, video_left + video_width, video_top + video_height)
    _paste(background, frame, video_box)

    image = frame_canvas(background)
    draw = ImageDraw.Draw(image, "RGBA")
    if frame is None:
        panel(draw, video_box, int(12 * scale_value), alpha=200, outline=MUTED)
        draw_text(
            image, ((video_box[0] + video_box[2]) // 2, (video_box[1] + video_box[3]) // 2),
            "no example clip", Type(int(15 * scale_value), "Medium"), MUTED, anchor="mm",
        )

    text_left = video_box[2] + int(44 * scale_value)
    draw_shadowed_text(
        image, (text_left, video_top), label,
        Type(int(46 * scale_value), "Heavy"), WHITE, anchor="lt",
    )
    cursor = video_top + int(66 * scale_value)
    category = str(row.get("category") or "").replace("_", " ")
    if category:
        chip(
            image, draw, text_left, cursor, category.upper(),
            Type(int(13 * scale_value), "Semibold"), scale_value, color=MUTED,
        )
        cursor += int(44 * scale_value)

    # No provenance: the gallery mixes project recordings and corpus footage, and
    # the local clips have no trustworthy signer identity to quote anyway.
    style = Type(int(14 * scale_value), "Medium")
    draw_text(
        image, (text_left, cursor), f"Clip  {float(row.get('seconds') or 0):.1f}s",
        style, MUTED, anchor="lt",
    )
    cursor += int(24 * scale_value)

    for note in row.get("notes") or []:
        draw_text(
            image, (text_left, cursor), f"· {note}",
            Type(int(13 * scale_value), "Medium"), AMBER, anchor="lt",
        )
        cursor += int(20 * scale_value)

    button_height = int(52 * scale_value)
    button_top = height - int(76 * scale_value)
    rects: Rects = {
        "practice": (
            text_left, button_top, text_left + int(230 * scale_value),
            button_top + button_height,
        ),
        "back": (
            int(60 * scale_value), button_top, int(60 * scale_value) + int(140 * scale_value),
            button_top + button_height,
        ),
    }
    _button(image, draw, rects["practice"], "SIGN THIS NOW", scale_value, primary=True)
    _button(image, draw, rects["back"], "BACK", scale_value)

    draw_nav(image, draw, width, "GLOSSES")
    return to_frame(image), rects


# --- history ----------------------------------------------------------------

def render_history(
    width: int, height: int, *, sessions: list[dict], describe, offset: int = 0,
) -> tuple[np.ndarray, Rects, int, int]:
    scale_value = scale(width)
    image = frame_canvas(_background(width, height))
    draw = ImageDraw.Draw(image, "RGBA")
    draw_nav(image, draw, width, "HISTORY")
    draw_text(
        image, (int(24 * scale_value), nav_height(width) + int(24 * scale_value)),
        f"{len(sessions)} recorded session{'' if len(sessions) == 1 else 's'}",
        Type(int(15 * scale_value), "Semibold"), WHITE, anchor="lm",
    )

    top = nav_height(width) + int(52 * scale_value)
    bottom = height - int(34 * scale_value)
    row_height = int(74 * scale_value)
    gap = int(8 * scale_value)
    margin = int(24 * scale_value)
    visible = max(0, (bottom - top + gap) // (row_height + gap))

    rects: Rects = {}
    for position, row in enumerate(sessions[offset:offset + visible]):
        row_top = top + position * (row_height + gap)
        box = (margin, row_top, width - margin - int(18 * scale_value), row_top + row_height)
        rects[f"session:{row.get('stamp')}"] = box
        panel(draw, box, int(10 * scale_value), alpha=PANEL_ALPHA, outline=MUTED)
        when, length, said = describe(row)
        draw_text(
            image, (box[0] + int(18 * scale_value), row_top + int(22 * scale_value)),
            when, Type(int(15 * scale_value), "Semibold"), WHITE, anchor="lm",
        )
        draw_text(
            image, (box[2] - int(18 * scale_value), row_top + int(22 * scale_value)),
            length, Type(int(14 * scale_value), "Medium", True), MUTED, anchor="rm",
        )
        # The state chip shares the text line, so the text has to yield room.
        badge = None if row.get("complete") else "unfinished"
        badge_style = Type(int(11 * scale_value), "Semibold")
        reserved = (
            0 if badge is None
            else chip_size(badge, badge_style, scale_value, dot=False)[0]
            + int(16 * scale_value)
        )
        said_line = row_top + int(50 * scale_value)
        style = Type(int(14 * scale_value), "Medium")
        limit = box[2] - box[0] - int(36 * scale_value) - reserved
        lines = wrap(said, style, max(int(40 * scale_value), limit), 1)
        draw_text(
            image, (box[0] + int(18 * scale_value), said_line),
            lines[0] if lines else said, style,
            WHITE if row.get("sentences") else MUTED, anchor="lm",
        )
        if badge is not None:
            badge_width, badge_height = chip_size(
                badge, badge_style, scale_value, dot=False
            )
            chip(
                image, draw, box[2] - int(18 * scale_value) - badge_width,
                said_line - badge_height // 2, badge, badge_style, scale_value,
                color=AMBER,
            )

    if not sessions:
        _empty(
            image, draw, width, top, bottom, scale_value, "No sessions yet",
            "Recordings appear here after you use the camera",
        )

    rects.update(
        _scrollbar(
            image, draw, width, top, bottom, scale_value, offset, visible,
            len(sessions),
        )
    )
    _footer(
        image, draw, width, height, scale_value,
        "Saved to artifacts/app_sessions · scores are uncalibrated",
    )
    return to_frame(image), rects, len(sessions), visible


# --- the live feed's own furniture ------------------------------------------
# The recognition loop hands over a finished HUD frame; everything the *app*
# adds on top of it lives here, so the live page is never a bare frame.

PRACTICE_SIZES = (5, 10, 20)


def overlay_live(
    frame: np.ndarray, *, page: str = "LIVE", target: str | None = None,
    sample: np.ndarray | None = None, verdict: str | None = None,
    verdict_gloss: str = "", celebrate: float = 0.0, index: int = 0,
    total: int = 0, score: int = 0, attempts: int = 0,
) -> tuple[np.ndarray, Rects]:
    """Paint the nav bar and, in a round, its furniture.

    Returns the rect table as every page does, so the pointer and the paint can
    never disagree about where the skip button is.
    """
    height, width = frame.shape[:2]
    scale_value = scale(width)
    background = frame
    pip = None
    if sample is not None:
        background = frame.copy()
        pip_width = int(width * 0.24)
        pip_height = int(pip_width * sample.shape[0] / sample.shape[1])
        left = width - pip_width - int(24 * scale_value)
        top = nav_height(width) + int(66 * scale_value)
        pip = (left, top, left + pip_width, top + pip_height)
        _paste(background, sample, pip)

    image = frame_canvas(background)
    draw = ImageDraw.Draw(image, "RGBA")

    if pip is not None:
        panel(draw, pip, 0, fill=INK, alpha=0, outline=WHITE)
        label_top = pip[3]
        panel(
            draw, (pip[0], label_top, pip[2], label_top + int(24 * scale_value)), 0,
            fill=INK, alpha=225,
        )
        draw_text(
            image, ((pip[0] + pip[2]) // 2, label_top + int(12 * scale_value)),
            "HOW IT LOOKS", Type(int(11 * scale_value), "Semibold"), MUTED,
            anchor="mm",
        )

    if target:
        box = (
            width // 2 - int(210 * scale_value), nav_height(width) + int(10 * scale_value),
            width // 2 + int(210 * scale_value), nav_height(width) + int(62 * scale_value),
        )
        if verdict == "missed":
            text, color = f"SAW  {verdict_gloss}  ·  TRY AGAIN", AMBER
        else:
            text, color = f"SIGN THIS:  {target}", WHITE
        panel(draw, box, int(26 * scale_value), alpha=225, outline=color)
        draw_text(
            image, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), text,
            Type(int(17 * scale_value), "Semibold"), color, anchor="mm",
        )

    if total:
        chip(
            image, draw, int(24 * scale_value),
            nav_height(width) + int(18 * scale_value),
            f"ROUND {index + 1} OF {total}   ·   SCORE {score}/{attempts}",
            Type(int(13 * scale_value), "Semibold"), scale_value, color=WHITE,
        )

    rects: Rects = {}
    if target:
        # Sits where the reel's controls would be, which a round does not use.
        button_width, button_height = int(190 * scale_value), int(46 * scale_value)
        left = (width - button_width) // 2
        top = height - int(26 * scale_value) - button_height
        rects["skip"] = (left, top, left + button_width, top + button_height)
        _button(image, draw, rects["skip"], "SKIP THIS SIGN", scale_value)

    if celebrate > 0:
        _celebrate(image, draw, width, height, scale_value, verdict_gloss, celebrate)

    draw_nav(image, draw, width, page)
    return to_frame(image), rects


def _celebrate(
    image, draw, width: int, height: int, scale_value: float, gloss: str,
    strength: float,
) -> None:
    """A full-frame flash that reads from across a room."""
    strength = max(0.0, min(1.0, strength))
    panel(draw, (0, 0, width, height), 0, fill=ACCENT, alpha=int(150 * strength))
    draw_shadowed_text(
        image, (width // 2, height // 2 - int(30 * scale_value)), "YEHEY!",
        Type(int(76 * scale_value), "Heavy"), WHITE, alpha=int(255 * strength),
        anchor="mm", offset=4,
    )
    draw_shadowed_text(
        image, (width // 2, height // 2 + int(42 * scale_value)), f"✓  {gloss}",
        Type(int(34 * scale_value), "Semibold"), INK, alpha=int(255 * strength),
        anchor="mm", offset=2,
    )


def render_practice_setup(
    width: int, height: int, *, available: int, sizes=PRACTICE_SIZES,
) -> tuple[np.ndarray, Rects]:
    scale_value = scale(width)
    image = frame_canvas(_background(width, height))
    draw = ImageDraw.Draw(image, "RGBA")
    draw_nav(image, draw, width, "PRACTICE")

    top = nav_height(width) + int(70 * scale_value)
    draw_shadowed_text(
        image, (width // 2, top), "Practice",
        Type(int(40 * scale_value), "Heavy"), WHITE, anchor="mt",
    )
    draw_text(
        image, (width // 2, top + int(54 * scale_value)),
        "The app picks signs at random.  Sign each one; the reference stays on screen.",
        Type(int(15 * scale_value), "Medium"), MUTED, anchor="mt",
    )

    rects: Rects = {}
    if not available:
        _empty(
            image, draw, width, top, height - int(60 * scale_value), scale_value,
            "No examples built yet", "Run scripts/build_gloss_examples_v17.py",
        )
        return to_frame(image), rects

    options = tuple(f"{min(size, available)} SIGNS" for size in sizes)
    row = row_rects(
        options, width, top + int(130 * scale_value), scale_value,
        height=int(56 * scale_value),
    )
    for (label, box), size in zip(row.items(), sizes):
        rects[f"start:{min(size, available)}"] = box
        _button(image, draw, box, label, scale_value, primary=size == sizes[0])

    _footer(
        image, draw, width, height, scale_value,
        f"{available} signs available",
    )
    return to_frame(image), rects


def render_practice_result(
    width: int, height: int, *, score: int, total: int, missed: list,
) -> tuple[np.ndarray, Rects]:
    scale_value = scale(width)
    image = frame_canvas(_background(width, height))
    draw = ImageDraw.Draw(image, "RGBA")
    draw_nav(image, draw, width, "PRACTICE")

    top = nav_height(width) + int(76 * scale_value)
    draw_shadowed_text(
        image, (width // 2, top), f"{score} of {total}",
        Type(int(64 * scale_value), "Heavy"),
        ACCENT if score == total else WHITE, anchor="mt",
    )
    remark = (
        "Perfect round." if total and score == total
        else "Nice work." if total and score >= total * 0.6
        else "Keep going."
    )
    draw_text(
        image, (width // 2, top + int(84 * scale_value)), remark,
        Type(int(18 * scale_value), "Medium"), MUTED, anchor="mt",
    )

    if missed:
        draw_text(
            image, (width // 2, top + int(122 * scale_value)),
            "Not recognised: " + ", ".join(str(item) for item in missed[:8]),
            Type(int(14 * scale_value), "Medium"), AMBER, anchor="mt",
        )

    row = row_rects(
        ("PRACTICE AGAIN", "DONE"), width, top + int(168 * scale_value), scale_value,
        height=int(52 * scale_value),
    )
    rects: Rects = {}
    for index, (label, box) in enumerate(row.items()):
        rects["again" if index == 0 else "done"] = box
        _button(image, draw, box, label, scale_value, primary=index == 0)

    _footer(
        image, draw, width, height, scale_value,
        "A miss may be the model, not you — scores are uncalibrated",
    )
    return to_frame(image), rects


# --- animated home ----------------------------------------------------------
# The mosaic is rendered once into a canvas larger than the window, then each
# frame takes a drifting crop of it.  That keeps the motion continuous without
# resizing a hundred images per frame.

class Backdrop:
    """A slow aurora: soft colour fields drifting over a dark gradient.

    Composited at a fraction of the window size and scaled up, which is what makes
    the gradients smooth and the cost negligible.  No photography — the home screen
    should not need the corpus to look like something.
    """

    PALETTE = (
        # (blue, green, red), drift rates, phase, radius, strength
        ((120, 210, 70), 0.055, 0.041, 0.0, 0.42, 0.55),
        ((190, 120, 60), 0.037, 0.062, 2.1, 0.38, 0.40),
        ((90, 90, 180), 0.047, 0.033, 4.2, 0.34, 0.30),
    )

    def __init__(self, width: int, height: int, *, divisor: int = 12) -> None:
        self.width, self.height = width, height
        self.small_width = max(12, width // divisor)
        self.small_height = max(12, height // divisor)
        xs = np.linspace(0.0, 1.0, self.small_width, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, self.small_height, dtype=np.float32)
        self.grid_x, self.grid_y = np.meshgrid(xs, ys)
        # A dark vertical wash so the top of the page is heavier than the bottom.
        top = np.array(INK[::-1], np.float32)
        bottom = top + np.array((16.0, 12.0, 10.0), np.float32)
        ramp = self.grid_y[..., None]
        self.base = top * (1.0 - ramp) + bottom * ramp
        self.vignette = 1.0 - 0.55 * (
            ((self.grid_x - 0.5) ** 2 + (self.grid_y - 0.5) ** 2) / 0.5
        ).clip(0.0, 1.0)

    def frame(self, now: float) -> np.ndarray:
        field = self.base.copy()
        for color, rate_x, rate_y, phase, radius, strength in self.PALETTE:
            centre_x = 0.5 + 0.40 * math.sin(now * rate_x * 2 * math.pi + phase)
            centre_y = 0.5 + 0.32 * math.sin(now * rate_y * 2 * math.pi + phase * 1.7)
            distance = (self.grid_x - centre_x) ** 2 + (self.grid_y - centre_y) ** 2
            glow = np.exp(-distance / (2 * radius * radius)) * strength
            field += glow[..., None] * np.array(color, np.float32)
        field *= self.vignette[..., None]
        small = np.clip(field, 0, 255).astype(np.uint8)
        return cv2.resize(
            small, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )


def _veil(draw, width: int, height: int, scale_value: float) -> None:
    """A light settling wash; the backdrop is already dark and smooth."""
    panel(draw, (0, 0, width, height), 0, fill=INK, alpha=58)


def render_home(
    width: int, height: int, *, status: str = "", ready: bool = True,
    session_count: int = 0, gloss_count: int = 0, backdrop: Backdrop | None = None,
    now: float = 0.0, entered: float | None = None, pointer: tuple = (-1, -1),
) -> tuple[np.ndarray, Rects]:
    scale_value = scale(width)
    background = (
        backdrop.frame(now) if backdrop is not None else _background(width, height)
    )
    image = frame_canvas(background)
    draw = ImageDraw.Draw(image, "RGBA")
    if backdrop is not None:
        _veil(draw, width, height, scale_value)

    # Everything eases in together on first paint, then settles.  Without an
    # entry time there is nothing to animate from, so draw it already settled.
    age = 1e6 if entered is None else max(0.0, now - entered)
    rise = 1.0 - (1.0 - min(1.0, age / 0.7)) ** 3
    lift = int((1 - rise) * 26 * scale_value)
    alpha = int(255 * rise)

    draw_nav(image, draw, width, "HOME", note=status or None)

    top = nav_height(width) + int(54 * scale_value) + lift
    draw_shadowed_text(
        image, (width // 2, top), "Sign Language Translation",
        Type(int(42 * scale_value), "Heavy"), WHITE, alpha=alpha, anchor="mt",
        offset=3,
    )
    draw_shadowed_text(
        image, (width // 2, top + int(58 * scale_value)),
        f"{gloss_count or 100} signs  ·  Apple Vision  ·  entirely on this device",
        Type(int(15 * scale_value), "Medium"), WHITE, alpha=int(alpha * 0.72),
        anchor="mt",
    )

    button_height = int(56 * scale_value)
    gap = int(14 * scale_value)
    button_width = int(min(width - 2 * int(40 * scale_value), 420 * scale_value))
    left = (width - button_width) // 2
    first = top + int(124 * scale_value)
    actions = (
        ("LIVE", "START CAMERA"),
        ("PRACTICE", "PRACTICE"),
        ("GLOSSES", f"THE {gloss_count or 100} SIGNS"),
        ("HISTORY", "SESSION HISTORY"),
    )
    rects: Rects = {}
    for index, (action, label) in enumerate(actions):
        # Each button arrives a beat after the one above it.
        step = 1.0 - (1.0 - min(1.0, max(0.0, age - 0.08 * index) / 0.6)) ** 3
        box_top = first + index * (button_height + gap) + int((1 - step) * 22 * scale_value)
        box = (left, box_top, left + button_width, box_top + button_height)
        rects[action] = box
        enabled = ready or action == "GLOSSES"
        text = label
        if action == "LIVE" and not ready:
            text = status or "WARMING UP…"
        elif action == "HISTORY" and session_count:
            text = f"{label}  ·  {session_count}"
        hover = (
            box[0] <= pointer[0] <= box[2] and box[1] <= pointer[1] <= box[3]
        )
        _button(
            image, draw, box, text, scale_value, primary=index == 0,
            enabled=enabled, glow=hover, alpha=int(255 * step),
        )

    return to_frame(image), rects
