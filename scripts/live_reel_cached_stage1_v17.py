#!/usr/bin/env python3
"""Reel Stage 1 with an exact-input LRU cache for repeated hand crops.

This separate experiment keeps every landmark, RGB view, and temporal position used
by ``live_reel_stage1_v17.py``. Only byte-identical MobileCLIP inputs are reused.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.live_isolated_v17 import IsolatedClassifier
from scripts.live_reel_stage1_v17 import (
    ReelCascadeClassifier,
    parser as reel_parser,
    run,
    validate_args,
)


class CachedHandClassifier(IsolatedClassifier):
    """Reuse MobileCLIP embeddings only for byte-identical RGB crops."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.embedding_cache_size = args.hand_embedding_cache_size
        self.embedding_cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self.embedding_cache_hits = 0
        self.embedding_cache_misses = 0

    @staticmethod
    def crop_key(crop: np.ndarray) -> bytes:
        digest = hashlib.blake2b(digest_size=20)
        digest.update(str(crop.shape).encode("ascii"))
        digest.update(crop.dtype.str.encode("ascii"))
        digest.update(crop.tobytes())
        return digest.digest()

    def encode_hands(
        self, crops: list[list[np.ndarray | None]], valid: np.ndarray,
    ) -> np.ndarray:
        embeddings = np.zeros((16, 3, 512), np.float32)
        pending: dict[bytes, tuple[np.ndarray, list[tuple[int, int]]]] = {}
        for frame_index in range(16):
            for view_index in range(3):
                crop = crops[frame_index][view_index]
                if crop is None or not valid[frame_index, view_index]:
                    continue
                key = self.crop_key(crop)
                cached = self.embedding_cache.get(key)
                if cached is not None:
                    self.embedding_cache.move_to_end(key)
                    embeddings[frame_index, view_index] = cached
                    self.embedding_cache_hits += 1
                elif key in pending:
                    pending[key][1].append((frame_index, view_index))
                    self.embedding_cache_hits += 1
                else:
                    pending[key] = (crop, [(frame_index, view_index)])
                    self.embedding_cache_misses += 1

        for key, (crop, positions) in pending.items():
            image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            value = np.asarray(
                self.image_encoder.predict({"image": image})["embedding"]
            ).reshape(512).astype(np.float32, copy=False)
            self.embedding_cache[key] = value.copy()
            self.embedding_cache.move_to_end(key)
            while len(self.embedding_cache) > self.embedding_cache_size:
                self.embedding_cache.popitem(last=False)
            for frame_index, view_index in positions:
                embeddings[frame_index, view_index] = value
        return embeddings

    def classify(self, observations):
        hits, misses = self.embedding_cache_hits, self.embedding_cache_misses
        result = super().classify(observations)
        result.setdefault("diagnostics", {})["hand_embedding_cache"] = {
            "hits": self.embedding_cache_hits - hits,
            "misses": self.embedding_cache_misses - misses,
            "entries": len(self.embedding_cache),
            "capacity": self.embedding_cache_size,
        }
        return result

    def provenance(self) -> dict[str, object]:
        return {
            **super().provenance(),
            "hand_embedding_cache": {
                "policy": "LRU over byte-identical RGB crops",
                "capacity": self.embedding_cache_size,
                "changes_model_inputs": False,
            },
        }


def classifier(args: argparse.Namespace) -> ReelCascadeClassifier:
    return ReelCascadeClassifier(args, verifier_class=CachedHandClassifier)


def parser() -> argparse.ArgumentParser:
    value = reel_parser()
    value.description = __doc__
    value.set_defaults(
        output_root=REPO / "artifacts/reports/live_reel_cached_stage1_v17"
    )
    value.add_argument("--hand-embedding-cache-size", type=int, default=512)
    return value


def main() -> None:
    args = parser().parse_args()
    validate_args(args)
    if args.hand_embedding_cache_size <= 0:
        raise ValueError("hand embedding cache size must be positive")
    run(args, classifier_factory=classifier)


if __name__ == "__main__":
    main()
