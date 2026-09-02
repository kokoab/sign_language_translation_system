from __future__ import annotations

import unittest

import numpy as np

from active.v17.lip_marker_v17 import (
    LIP_INDICES,
    draw_lip_markers,
    lip_marker_features,
)


class LipMarkerFeaturesTest(unittest.TestCase):
    def test_draws_outer_and_inner_markers(self) -> None:
        frame = np.full((100, 100, 3), 127, dtype=np.uint8)
        angles = np.linspace(0, 2 * np.pi, len(LIP_INDICES), endpoint=False)
        points = np.stack((
            0.5 + 0.2 * np.cos(angles),
            0.5 + 0.1 * np.sin(angles),
        ), axis=1).astype(np.float32)
        rendered = draw_lip_markers(frame, points, mirror=False)
        self.assertGreater(np.count_nonzero(np.all(rendered == 255, axis=2)), 0)

    def test_shape_and_position_invariance(self) -> None:
        rng = np.random.default_rng(17)
        sequence = [rng.normal(size=(len(LIP_INDICES), 2)).astype(np.float32)] * 8
        moved = [value * 2.5 + (0.2, -0.4) for value in sequence]
        first = lip_marker_features(sequence)
        second = lip_marker_features(moved)
        self.assertEqual(first.shape, (24 * 40 * 2 * 2,))
        np.testing.assert_allclose(first, second, atol=2e-5)

    def test_missing_frames_are_interpolated(self) -> None:
        points = np.zeros((len(LIP_INDICES), 2), np.float32)
        points[:, 0] = np.linspace(0.2, 0.8, len(LIP_INDICES))
        points[:, 1] = np.linspace(0.4, 0.6, len(LIP_INDICES))
        self.assertIsNotNone(
            lip_marker_features([points, None, points, None, points, points])
        )
        self.assertIsNone(lip_marker_features([None, points, None]))


if __name__ == "__main__":
    unittest.main()
