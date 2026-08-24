import unittest

import numpy as np

from active.v16.extract_v16 import (
    LHAND_END,
    LHAND_START,
    NUM_NODES,
    aspect_correct_image_coordinates,
    normalize_sequence,
)


def _physical_sequence():
    """Return a small skeleton in centered, isotropic image coordinates."""
    frames = 12
    xy = np.zeros((frames, NUM_NODES, 2), dtype=np.float32)
    mask = np.zeros((frames, NUM_NODES), dtype=np.float32)
    offsets = np.stack([
        np.linspace(-0.035, 0.045, LHAND_END - LHAND_START),
        np.linspace(0.0, -0.11, LHAND_END - LHAND_START),
    ], axis=-1).astype(np.float32)
    # Ensure wrist-to-middle-MCP has a nonzero vector on both axes.
    offsets[0] = (0.0, 0.0)
    offsets[9] = (0.055, -0.075)
    for frame in range(frames):
        wrist_motion = np.array([
            -0.03 + 0.006 * frame,
            0.02 * np.sin(frame / 3.0),
        ], dtype=np.float32)
        xy[frame, LHAND_START:LHAND_END] = offsets + wrist_motion
    mask[:, LHAND_START:LHAND_END] = 1.0
    return xy, mask


def _to_vision_normalized(centered_xy, width, height):
    longest = float(max(width, height))
    result = centered_xy.copy()
    result[..., 0] = (width / 2.0 + centered_xy[..., 0] * longest) / width
    result[..., 1] = (height / 2.0 + centered_xy[..., 1] * longest) / height
    return result


def _correct_and_normalize(centered_xy, mask, width, height):
    vision_xy = _to_vision_normalized(centered_xy, width, height)
    corrected = aspect_correct_image_coordinates(vision_xy, width, height)
    longest = float(max(width, height))
    default_center = (0.5 * width / longest, 0.5 * height / longest)
    return normalize_sequence(corrected, mask, default_center=default_center)


class AspectCorrectCoordinatesTest(unittest.TestCase):
    def test_portrait_and_landscape_produce_same_normalized_skeleton(self):
        physical_xy, mask = _physical_sequence()
        landscape = _correct_and_normalize(physical_xy, mask, 1600, 900)
        portrait = _correct_and_normalize(physical_xy, mask, 900, 1600)
        np.testing.assert_allclose(landscape, portrait, atol=2e-6, rtol=0)

    def test_legacy_geometry_is_orientation_dependent(self):
        physical_xy, mask = _physical_sequence()
        landscape = normalize_sequence(
            _to_vision_normalized(physical_xy, 1600, 900), mask)
        portrait = normalize_sequence(
            _to_vision_normalized(physical_xy, 900, 1600), mask)
        self.assertGreater(float(np.max(np.abs(landscape - portrait))), 0.1)

    def test_square_coordinates_are_unchanged(self):
        rng = np.random.default_rng(42)
        xy = rng.random((4, NUM_NODES, 2), dtype=np.float32)
        corrected = aspect_correct_image_coordinates(xy, 1024, 1024)
        np.testing.assert_array_equal(corrected, xy)

    def test_invalid_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            aspect_correct_image_coordinates(np.zeros((1, 1, 2)), 0, 100)


if __name__ == "__main__":
    unittest.main()
