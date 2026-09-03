from __future__ import annotations

from collections import OrderedDict
import unittest

import numpy as np

from scripts.live_isolated_v17 import IsolatedClassifier
from scripts.live_reel_cached_stage1_v17 import CachedHandClassifier, parser


class CountingEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, provider):
        self.calls += 1
        pixels = np.asarray(provider["image"], dtype=np.float32)
        return {"embedding": np.full(512, pixels.mean(), np.float32)}


def classifier(kind, encoder):
    value = kind.__new__(kind)
    value.image_encoder = encoder
    if kind is CachedHandClassifier:
        value.embedding_cache_size = 8
        value.embedding_cache = OrderedDict()
        value.embedding_cache_hits = 0
        value.embedding_cache_misses = 0
    return value


class CachedReelTest(unittest.TestCase):
    def test_cache_preserves_embeddings_and_skips_identical_inputs(self) -> None:
        first = np.full((24, 24, 3), 17, np.uint8)
        second = np.full((24, 24, 3), 91, np.uint8)
        crops = [[None, None, None] for _ in range(16)]
        valid = np.zeros((16, 3), np.float32)
        for frame in range(16):
            crops[frame][0] = first if frame < 8 else second
            valid[frame, 0] = 1

        baseline_encoder = CountingEncoder()
        baseline = classifier(IsolatedClassifier, baseline_encoder).encode_hands(
            crops, valid
        )
        cached_encoder = CountingEncoder()
        cached_classifier = classifier(CachedHandClassifier, cached_encoder)
        cached = cached_classifier.encode_hands(crops, valid)

        np.testing.assert_array_equal(cached, baseline)
        self.assertEqual(baseline_encoder.calls, 16)
        self.assertEqual(cached_encoder.calls, 2)
        cached_classifier.encode_hands(crops, valid)
        self.assertEqual(cached_encoder.calls, 2)

    def test_parser_keeps_the_experiment_separate(self) -> None:
        args = parser().parse_args([])
        self.assertIn("live_reel_cached_stage1_v17", str(args.output_root))
        self.assertEqual(args.hand_embedding_cache_size, 512)


if __name__ == "__main__":
    unittest.main()
