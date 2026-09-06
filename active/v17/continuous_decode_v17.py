"""Incremental CTC alternatives and revisable text; never gates the input stream."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


NEG_INF = float("-inf")


def logadd(*values: float) -> float:
    maximum = max(values)
    if maximum == NEG_INF:
        return maximum
    return maximum + math.log(sum(math.exp(v - maximum) for v in values))


class CTCPrefixDecoder:
    """CTC prefix beam with separate blank/nonblank mass and repeated-label support."""

    def __init__(self, beam_width: int = 8, token_topk: int = 12):
        if beam_width < 1 or token_topk < 1:
            raise ValueError("beam width and token limit must be positive")
        self.beam_width, self.token_topk = beam_width, token_topk
        self.reset()

    def reset(self):
        self.beams = {(): (0.0, NEG_INF)}
        self.steps = 0

    def step(self, log_probs: np.ndarray):
        scores = np.asarray(log_probs, dtype=np.float64)
        if scores.ndim != 1 or len(scores) < 2 or not np.isfinite(scores).all():
            raise ValueError("finite one-dimensional CTC log probabilities required")
        count = min(self.token_topk, len(scores))
        tokens = set(np.argpartition(scores, -count)[-count:].tolist())
        tokens.add(0)
        # A previous label must remain in consideration for its repeated-path mass.
        tokens.update(prefix[-1] for prefix in self.beams if prefix)
        next_beams = {}

        def add(prefix, blank=NEG_INF, nonblank=NEG_INF):
            pb, pnb = next_beams.get(prefix, (NEG_INF, NEG_INF))
            next_beams[prefix] = (logadd(pb, blank), logadd(pnb, nonblank))

        for prefix, (pb, pnb) in self.beams.items():
            total = logadd(pb, pnb)
            add(prefix, blank=total + float(scores[0]))
            for token in tokens - {0}:
                probability = float(scores[token])
                if prefix and token == prefix[-1]:
                    add(prefix, nonblank=pnb + probability)
                    add(prefix + (token,), nonblank=pb + probability)
                else:
                    add(prefix + (token,), nonblank=total + probability)
        ranked = sorted(next_beams.items(), key=lambda row: logadd(*row[1]), reverse=True)
        self.beams = dict(ranked[:self.beam_width])
        self.steps += 1
        return self.alternatives()

    def alternatives(self):
        return sorted(
            [(prefix, logadd(*mass)) for prefix, mass in self.beams.items()],
            key=lambda row: row[1], reverse=True,
        )


@dataclass(frozen=True)
class PartialHypothesis:
    tokens: tuple[int, ...]
    stable_tokens: tuple[int, ...]
    revised_from: int
    alternatives: tuple[tuple[tuple[int, ...], float], ...]


class RevisableTranscript:
    """A display policy, independent of recognition, that exposes every new hypothesis.

Stability is provisional and can retract. Speech consumers must avoid speaking a
provisional suffix; already spoken words cannot be repaired silently.
"""

    def __init__(self, stability_seconds: float = 0.6, evidence_margin: float = 2.0):
        if stability_seconds < 0 or evidence_margin < 0:
            raise ValueError("nonnegative transcript policy values required")
        self.stability_seconds = stability_seconds
        self.evidence_margin = evidence_margin
        self.reset()

    def reset(self):
        self.previous = ()
        self.since = []
        self.last_seconds = NEG_INF

    def update(self, alternatives, seconds: float, *, final: bool = False):
        if not alternatives or not math.isfinite(seconds) or seconds < self.last_seconds:
            raise ValueError("ordered timestamp and nonempty alternatives required")
        self.last_seconds = seconds
        tokens = tuple(alternatives[0][0])
        common = 0
        for first, second in zip(self.previous, tokens):
            if first != second:
                break
            common += 1
        self.since = self.since[:common] + [seconds] * (len(tokens) - common)
        credible = [tuple(p) for p, score in alternatives if score >= alternatives[0][1] - self.evidence_margin]
        consensus = len(tokens)
        for candidate in credible:
            n = 0
            for first, second in zip(tokens, candidate):
                if first != second:
                    break
                n += 1
            consensus = min(consensus, n)
        stable = 0
        for started in self.since[:consensus]:
            if seconds - started < self.stability_seconds:
                break
            stable += 1
        self.previous = tokens
        return PartialHypothesis(tokens, tokens if final else tokens[:stable], common, tuple(alternatives))
