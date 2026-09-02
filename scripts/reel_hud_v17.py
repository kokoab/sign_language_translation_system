#!/usr/bin/env python3
"""Reel-style HUD rendering for the live Stage-1 experiment.

Layout only.  Nothing here reads or changes model state: it receives the same
values the old OpenCV overlay received and paints them with TrueType text.

    top          committed gloss chips
    upper        naturalized sentence
    center       last committed gloss + confidence
    bottom left  benchmarks
    bottom right top-3 candidates
    bottom mid   RESET / FINISH controls

Rasterising a string costs ~0.5 ms, so every text run is cached as a small RGBA
tile and pasted; shapes are drawn directly because they are effectively free.
"""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import time
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

INK = (14, 15, 19)
WHITE = (245, 246, 248)
MUTED = (138, 143, 152)
ACCENT = (61, 220, 132)
AMBER = (245, 165, 36)

PANEL_ALPHA = 168
CHIP_ALPHA = 150

SANS_CANDIDATES = (
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
MONO_CANDIDATES = (
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)


class Type(NamedTuple):
    """A resolved text style: size in pixels, variable-font weight, family."""

    size: int
    weight: str = "Regular"
    mono: bool = False


@lru_cache(maxsize=1)
def _font_files() -> tuple[str | None, str | None]:
    def first(paths: tuple[str, ...]) -> str | None:
        return next((path for path in paths if Path(path).exists()), None)

    return first(SANS_CANDIDATES), first(MONO_CANDIDATES)


@lru_cache(maxsize=64)
def _font(style: Type):
    sans, monospace = _font_files()
    path = monospace if style.mono else sans
    if path is None:
        return ImageFont.load_default(size=style.size)
    font = ImageFont.truetype(path, style.size)
    try:  # variable system fonts carry the weight axis; static ones do not
        font.set_variation_by_name(style.weight)
    except (OSError, AttributeError, ValueError):
        pass
    return font


@lru_cache(maxsize=512)
def _text_width(style: Type, text: str) -> int:
    return int(math.ceil(_font(style).getlength(text)))


@lru_cache(maxsize=512)
def _tile(
    style: Type, text: str, color: tuple[int, int, int], alpha: int
) -> Image.Image:
    """One text run rendered on transparency, sized to the full line box."""
    font = _font(style)
    ascent, descent = font.getmetrics()
    tile = Image.new(
        "RGBA", (max(1, _text_width(style, text) + 2), max(1, ascent + descent)),
        (0, 0, 0, 0),
    )
    ImageDraw.Draw(tile).text((0, 0), text, font=font, fill=(*color, alpha))
    return tile


def _blit(
    image: Image.Image, position: tuple[int, int], tile: Image.Image,
    anchor: str = "lt",
) -> None:
    x, y = position
    width, height = tile.size
    if anchor[0] == "m":
        x -= width // 2
    elif anchor[0] == "r":
        x -= width
    if anchor[1] == "m":
        y -= height // 2
    elif anchor[1] == "b":
        y -= height
    image.paste(tile, (x, y), tile)


def _text(
    image: Image.Image, position: tuple[int, int], text: str, style: Type,
    color: tuple[int, int, int], *, alpha: int = 255, anchor: str = "lt",
) -> None:
    _blit(image, position, _tile(style, text, color, alpha), anchor)


def _shadowed(
    image: Image.Image, position: tuple[int, int], text: str, style: Type,
    color: tuple[int, int, int], *, alpha: int = 255, anchor: str = "lt",
    offset: int = 2,
) -> None:
    x, y = position
    _blit(image, (x + offset, y + offset), _tile(style, text, (0, 0, 0), alpha // 2),
          anchor)
    _blit(image, (x, y), _tile(style, text, color, alpha), anchor)


def _scale(width: int) -> float:
    return max(0.45, min(1.6, width / 1280.0))


def reel_control_button_rects(
    width: int, height: int
) -> dict[str, tuple[int, int, int, int]]:
    """Bottom-centre pills.  Drawing and hit-testing share this one source."""
    scale = _scale(width)
    button_height = int(46 * scale)
    reset_width = int(140 * scale)
    finish_width = int(160 * scale)
    gap = int(12 * scale)
    top = max(0, height - int(26 * scale) - button_height)
    left = max(0, (width - (reset_width + gap + finish_width)) // 2)
    return {
        "reset": (left, top, left + reset_width, top + button_height),
        "finish": (
            left + reset_width + gap, top,
            left + reset_width + gap + finish_width, top + button_height,
        ),
    }


def clicked_reel_control(x: int, y: int, width: int, height: int) -> str | None:
    for action, (left, top, right, bottom) in reel_control_button_rects(
        width, height
    ).items():
        if left <= x <= right and top <= y <= bottom:
            return action
    return None


def _panel(
    draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int,
    *, fill=INK, alpha: int = PANEL_ALPHA, outline: tuple[int, int, int] | None = None,
) -> None:
    draw.rounded_rectangle(box, radius, fill=(*fill, alpha))
    if outline is not None:
        draw.rounded_rectangle(box, radius, outline=(*outline, 90), width=1)


def _dot(
    draw: ImageDraw.ImageDraw, centre: tuple[int, int], radius: int,
    color: tuple[int, int, int], alpha: int = 255,
) -> None:
    x, y = centre
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius), fill=(*color, alpha)
    )


def _chip_size(text: str, style: Type, scale: float, *, dot: bool) -> tuple[int, int]:
    width = _text_width(style, text) + 2 * int(12 * scale)
    if dot:
        width += int(14 * scale)
    return width, int(30 * scale)


def _chip(
    image: Image.Image, draw: ImageDraw.ImageDraw, x: int, y: int, text: str,
    style: Type, scale: float, *, dot: tuple[int, int, int] | None = None,
    color=WHITE, fill=INK, alpha: int = CHIP_ALPHA,
) -> int:
    width, height = _chip_size(text, style, scale, dot=dot is not None)
    _panel(draw, (x, y, x + width, y + height), height // 2, fill=fill, alpha=alpha)
    pad = int(12 * scale)
    text_x = x + pad
    if dot is not None:
        radius = max(2, int(3.5 * scale))
        _dot(draw, (x + pad + radius, y + height // 2), radius, dot)
        text_x += 2 * radius + int(8 * scale)
    _text(image, (text_x, y + height // 2), text, style, color, anchor="lm")
    return width


@lru_cache(maxsize=32)
def _wrap(text: str, style: Type, limit: int, lines: int) -> tuple[str, ...]:
    rows: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and _text_width(style, candidate) > limit:
            rows.append(current)
            current = word
            if len(rows) == lines:
                break
        else:
            current = candidate
    if current and len(rows) < lines:
        rows.append(current)
    if rows and len(rows) == lines:
        used = sum(len(row.split()) for row in rows)
        if used < len(text.split()):
            while rows[-1] and _text_width(style, rows[-1] + "...") > limit:
                rows[-1] = rows[-1][:-1]
            rows[-1] += "..."
    return tuple(rows)


class ReelHud:
    """Stateless with respect to the pipeline; remembers only what it draws."""

    def __init__(self) -> None:
        self._committed: tuple[str, float] | None = None
        self._committed_at = 0.0

    def _remember_commit(self, latest_result: dict[str, object] | None) -> None:
        if not latest_result:
            return
        committed = latest_result.get("committed_gloss")
        if not committed:
            return
        verifier = latest_result.get("full_verifier")
        fallback = float(latest_result.get("model_score", 0.0) or 0.0)
        score = float(
            verifier.get("commit_score", fallback)
            if isinstance(verifier, dict) else fallback
        )
        if self._committed != (str(committed), score):
            self._committed = (str(committed), score)
            self._committed_at = time.perf_counter()

    def draw(
        self,
        frame: np.ndarray,
        latest,
        latest_result: dict[str, object] | None,
        lock,
        pending: bool,
        active: bool,
        fps: float,
        glosses: list[str],
        finishing_glosses: list[str],
        sentence: str,
        finish_pending: bool,
        speech_text: str | None,
        ctc_hypothesis: list[str] | None = None,
        stats: dict[str, object] | None = None,
    ) -> np.ndarray:
        if not glosses and not finishing_glosses:
            self._committed = None
        self._remember_commit(latest_result)
        height, width = frame.shape[:2]
        scale = _scale(width)
        margin = int(24 * scale)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image, "RGBA")

        shown_glosses = (
            finishing_glosses if finish_pending and finishing_glosses else glosses
        )
        state, state_color = (
            ("VERIFYING", AMBER) if pending
            else ("SIGNING", ACCENT) if active else ("READY", MUTED)
        )
        style = Type(int(15 * scale), "Semibold")
        rail_left = margin + _chip_size(
            f"{state}   {fps:.1f} FPS", style, scale, dot=True
        )[0] + int(16 * scale)
        _chip(
            image, draw, margin, margin, f"{state}   {fps:.1f} FPS", style, scale,
            dot=state_color, alpha=PANEL_ALPHA,
        )
        self._gloss_rail(
            image, draw, width - margin, margin, scale, shown_glosses,
            max(rail_left, int(width * 0.34)),
        )
        self._sentence(
            image, draw, margin, int(80 * scale), width, scale,
            sentence, speech_text, finish_pending,
        )
        controls_top = min(
            rect[1] for rect in reel_control_button_rects(width, height).values()
        )
        panel_bottom = controls_top - int(14 * scale)
        self._center(image, draw, width, panel_bottom, scale, lock, active)
        self._benchmarks(
            image, draw, margin, panel_bottom, scale,
            latest, latest_result, fps, ctc_hypothesis, stats or {},
        )
        self._candidates(
            image, draw, width - margin, panel_bottom, scale, latest_result
        )
        self._controls(image, draw, width, height, scale, finish_pending)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def _gloss_rail(
        self, image: Image.Image, draw: ImageDraw.ImageDraw, right: int, top: int,
        scale: float, glosses: list[str], left_limit: int,
    ) -> None:
        if not glosses:
            return
        style = Type(int(15 * scale), "Semibold")
        gap = int(8 * scale)
        x = right
        for index, gloss in enumerate(reversed(glosses)):
            newest = index == 0
            ambiguous = gloss.endswith("?")
            width = _chip_size(gloss, style, scale, dot=True)[0]
            if x - width < left_limit:
                _chip(
                    image, draw, max(left_limit, x - int(40 * scale)), top, "...",
                    style, scale, color=MUTED,
                )
                return
            _chip(
                image, draw, x - width, top, gloss, style, scale,
                dot=AMBER if ambiguous else (ACCENT if newest else MUTED),
                color=AMBER if ambiguous else (WHITE if newest else MUTED),
                alpha=PANEL_ALPHA if newest else CHIP_ALPHA,
            )
            x -= width + gap

    def _sentence(
        self, image: Image.Image, draw: ImageDraw.ImageDraw, margin: int, top: int,
        width: int, scale: float, sentence: str, speech_text: str | None,
        finish_pending: bool,
    ) -> None:
        waiting = finish_pending and not speech_text
        text = "Naturalizing..." if waiting else (speech_text or sentence)
        if not text:
            return
        style = Type(int(34 * scale), "Semibold")
        radius = max(9, int(13 * scale))
        left = margin + 2 * radius + int(14 * scale)
        rows = _wrap(text, style, int(width * 0.80) - left, 3)
        step = int(42 * scale)
        _dot(
            draw, (margin + radius, top + int(17 * scale)), radius,
            ACCENT if speech_text else WHITE, 235 if speech_text else 90,
        )
        for index, row in enumerate(rows):
            _shadowed(
                image, (left, top + index * step), row, style,
                MUTED if waiting else WHITE,
            )

    def _center(
        self, image: Image.Image, draw: ImageDraw.ImageDraw, width: int, bottom: int,
        scale: float, lock, active: bool,
    ) -> None:
        """Sits just above the controls, not over the signer's face."""
        if self._committed is None:
            return
        gloss, score = self._committed
        age = time.perf_counter() - self._committed_at
        alpha = 255 if age < 1.6 else max(96, int(255 - (age - 1.6) * 110))
        alpha -= alpha % 16  # keep the fade on a few cacheable steps
        centre_x = width // 2
        top = bottom - int(128 * scale)
        style = Type(int(56 * scale), "Heavy")
        room = int(width * 0.62)
        if _text_width(style, gloss) > room:  # long glosses stay inside the frame
            style = Type(
                max(14, style.size * room // _text_width(style, gloss)), "Heavy"
            )
        _shadowed(
            image, (centre_x, top), gloss, style, WHITE,
            alpha=alpha, anchor="mt", offset=3,
        )

        bar_width = int(240 * scale)
        bar_top = top + int(76 * scale)
        bar_height = max(4, int(6 * scale))
        left = centre_x - bar_width // 2
        draw.rounded_rectangle(
            (left, bar_top, left + bar_width, bar_top + bar_height),
            bar_height // 2, fill=(*INK, min(alpha, PANEL_ALPHA)),
        )
        filled = int(bar_width * max(0.0, min(1.0, score)))
        if filled > bar_height:
            draw.rounded_rectangle(
                (left, bar_top, left + filled, bar_top + bar_height),
                bar_height // 2, fill=(*ACCENT, alpha),
            )
        _text(
            image, (centre_x, bar_top + int(16 * scale)), f"{score:.2f}",
            Type(int(15 * scale), "Medium", True), MUTED, alpha=alpha, anchor="mt",
        )

        required = max(1, int(getattr(lock, "required_hits", 1) or 1))
        hits = int(getattr(lock, "hits", 0))
        if not active and hits == 0:
            return
        radius = max(2, int(3 * scale))
        gap = int(10 * scale)
        span = required * 2 * radius + (required - 1) * gap
        dot_x = centre_x - span // 2 + radius
        dot_y = bar_top + int(46 * scale)
        for index in range(required):
            _dot(
                draw, (dot_x, dot_y), radius, ACCENT if index < hits else MUTED,
                alpha if index < hits else min(alpha, 120),
            )
            dot_x += 2 * radius + gap

    def _benchmarks(
        self, image: Image.Image, draw: ImageDraw.ImageDraw, left: int, bottom: int,
        scale: float, latest, latest_result: dict[str, object] | None, fps: float,
        ctc_hypothesis: list[str] | None, stats: dict[str, object],
    ) -> None:
        value_style = Type(int(13 * scale), "Medium", True)
        label_style = Type(int(11 * scale), "Semibold", True)
        latency = 0.0
        if latest_result:
            timings = latest_result.get("latency_ms")
            if isinstance(timings, dict):
                latency = float(timings.get("total", 0.0) or 0.0)
        rows = [
            ("FPS", f"{fps:.1f}"),
            ("STAGE 1", f"{latency:.1f} ms"),
            (
                "HAND/FACE",
                "--" if latest is None
                else f"{latest.hand_quality:.2f} / {latest.face_quality:.2f}",
            ),
            ("MOTION", "--" if latest is None else f"{latest.motion:.3f}"),
        ]
        if "dropped" in stats:
            rows.append(("DROPPED", str(int(stats["dropped"]))))
        if "observations" in stats:
            rows.append(("FRAMES", str(int(stats["observations"]))))
        if ctc_hypothesis:
            rows.append(("SEQUENCE", " ".join(ctc_hypothesis)[-26:]))

        pad = int(14 * scale)
        step = int(19 * scale)
        value_x = int(88 * scale)
        box_width = value_x + pad + max(
            _text_width(value_style, value) for _, value in rows
        ) + pad
        top = bottom - (len(rows) * step + 2 * pad)
        _panel(draw, (left, top, left + box_width, bottom), int(14 * scale),
               outline=WHITE)
        for index, (label, value) in enumerate(rows):
            y = top + pad + index * step
            _text(image, (left + pad, y), label, label_style, MUTED)
            _text(image, (left + pad + value_x, y), value, value_style, WHITE)

    def _candidates(
        self, image: Image.Image, draw: ImageDraw.ImageDraw, right: int, bottom: int,
        scale: float, latest_result: dict[str, object] | None,
    ) -> None:
        top3 = list((latest_result or {}).get("top3") or [])
        if not top3:
            return
        gloss_style = Type(int(14 * scale), "Medium")
        score_style = Type(int(13 * scale), "Medium", True)
        step = int(34 * scale)
        pad = int(12 * scale)
        chip_height = int(28 * scale)
        chip_width = int(200 * scale)
        y = bottom - len(top3) * step
        for index, entry in enumerate(top3):
            score = float(entry.get("model_score", 0.0) or 0.0)
            gloss = str(entry.get("gloss", "--"))[:16]
            left = right - chip_width
            color = WHITE if index == 0 else MUTED
            _panel(
                draw, (left, y, right, y + chip_height), chip_height // 2,
                alpha=PANEL_ALPHA if index == 0 else CHIP_ALPHA,
            )
            radius = max(2, int(3.5 * scale))
            shade = tuple(
                int(value * (0.45 + 0.55 * min(1.0, max(0.0, score))))
                for value in ACCENT
            )
            _dot(draw, (left + pad + radius, y + chip_height // 2), radius, shade)
            _text(
                image, (left + pad + 2 * radius + int(8 * scale),
                        y + chip_height // 2),
                gloss, gloss_style, color, anchor="lm",
            )
            _text(
                image, (right - pad, y + chip_height // 2), f"{score:.2f}",
                score_style, color, anchor="rm",
            )
            y += step

    def _controls(
        self, image: Image.Image, draw: ImageDraw.ImageDraw, width: int, height: int,
        scale: float, finish_pending: bool,
    ) -> None:
        style = Type(int(15 * scale), "Semibold")
        for action, (left, top, right, bottom) in reel_control_button_rects(
            width, height
        ).items():
            working = action == "finish" and finish_pending
            if action == "reset":
                fill, color, alpha, outline = INK, WHITE, PANEL_ALPHA, WHITE
            elif working:
                fill, color, alpha, outline = AMBER, INK, 235, None
            else:
                fill, color, alpha, outline = ACCENT, INK, 235, None
            _panel(
                draw, (left, top, right, bottom), (bottom - top) // 2,
                fill=fill, alpha=alpha, outline=outline,
            )
            _text(
                image, ((left + right) // 2, (top + bottom) // 2),
                "WORKING" if working else action.upper(), style, color, anchor="mm",
            )
