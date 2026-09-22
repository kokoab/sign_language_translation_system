#!/usr/bin/env python3
"""Input encoding for the evidence-conditioned Stage-3 renderer.

The deployed renderer receives only lowercased gloss text, so it cannot tell a sign the
recognizer was sure about from a 0.28 flicker between two confident signs. It therefore
renders every recognizer error faithfully: "I GO LESS SCHOOL TOMORROW MORNING" becomes
"I am going to the less school tomorrow morning."

This module defines the replacement contract. Each gloss is preceded by a confidence
bucket token, so the renderer sees evidence alongside the word. Tokenizer cost was
measured against the alternatives: per-gloss word tags cost 11 tokens for a five-gloss
buffer versus 6 untagged, while inline punctuation markers fragment the gloss itself
and a tilde suffix tokenizes to <unk>.

Bucket edges follow the live distribution in artifacts/app_sessions: 2,006 accepted
predictions with median 0.615 and p10 0.315.
"""

from __future__ import annotations

from typing import Iterable, Sequence

HIGH_EDGE = 0.60
MID_EDGE = 0.40

HIGH = "hi"
MID = "mid"
LOW = "lo"


def bucket(confidence: float | None) -> str:
    """Map a recognizer score to its evidence token.

    A missing score is treated as high: callers without per-gloss evidence, such as the
    warm-up path, must not have their glosses silently discarded.
    """
    if confidence is None:
        return HIGH
    if confidence >= HIGH_EDGE:
        return HIGH
    if confidence >= MID_EDGE:
        return MID
    return LOW


def encode(glosses: Sequence[str], confidences: Sequence[float] | None = None) -> str:
    """Build the model input: one bucket token before each lowercased gloss."""
    if confidences is not None and len(confidences) != len(glosses):
        raise ValueError("confidences must align one-to-one with glosses")
    parts: list[str] = []
    for index, gloss in enumerate(glosses):
        score = None if confidences is None else confidences[index]
        parts.append(bucket(score))
        parts.append(str(gloss).lower())
    return " ".join(parts)


def decode_glosses(encoded: str) -> list[str]:
    """Recover the gloss sequence from an encoded input, for tests and logging."""
    tokens = encoded.split()
    return [tokens[i] for i in range(1, len(tokens), 2)]
