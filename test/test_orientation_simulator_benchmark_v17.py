import unittest
import struct
import tempfile
from pathlib import Path

import numpy as np

from scripts.run_orientation_simulator_benchmark_v17 import (
    DEVICE_NAME,
    DEVICE_TYPE,
    angle_slug,
    residual_roll,
    rotation_filter,
    version_key,
    write_stage2_crop_bundle,
)


class OrientationSimulatorBenchmarkV17Test(unittest.TestCase):
    def test_exact_quadrants_use_lossless_filters(self):
        self.assertIsNone(rotation_filter(0.0))
        self.assertEqual(rotation_filter(90.0), "transpose=clock")
        self.assertEqual(rotation_filter(180.0), "hflip,vflip")
        self.assertEqual(rotation_filter(270.0), "transpose=cclock")

    def test_arbitrary_rotation_expands_to_even_canvas(self):
        value = rotation_filter(37.0)
        self.assertIn("rotw", value)
        self.assertIn("roth", value)
        self.assertIn("ceil", value)
        self.assertIn("c=black", value)

    def test_residual_roll_wraps_to_signed_circle(self):
        self.assertEqual(residual_roll(90.0, 270.0), 0.0)
        self.assertEqual(residual_roll(73.0, 270.0), -17.0)
        self.assertEqual(residual_roll(123.0, 270.0), 33.0)

    def test_angle_slug_is_stable(self):
        self.assertEqual(angle_slug(37.0), "37")
        self.assertEqual(angle_slug(37.25), "37p25")

    def test_harness_is_pinned_to_iphone_13(self):
        self.assertEqual(DEVICE_NAME, "SLT Orientation Benchmark iPhone 13")
        self.assertEqual(
            DEVICE_TYPE, "com.apple.CoreSimulator.SimDeviceType.iPhone-13"
        )

    def test_runtime_versions_sort_numerically(self):
        self.assertLess(version_key("26.2"), version_key("26.3.1"))

    def test_stage2_crop_bundle_is_little_endian_and_fail_closed(self):
        arrays = {
            "hand_jpeg_offsets": np.full((1, 16, 3, 2), (-1, 0), np.int64),
            "hand_valid": np.zeros((1, 16, 3), np.bool_),
            "hand_boxes_normalized": np.zeros((1, 16, 3, 4), np.float16),
            "hand_jpeg_blob": np.empty(0, np.uint8),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crops.bin"
            write_stage2_crop_bundle(path, arrays)
            data = path.read_bytes()
        self.assertEqual(data[:8], b"SLTHRGB1")
        self.assertEqual(struct.unpack("<I", data[8:12]), (1,))
        expected = 8 + 4 + (16 * 3 * 4) + (16 * 3 * 4 * 4) + (16 * 3 * 4)
        self.assertEqual(len(data), expected)


if __name__ == "__main__":
    unittest.main()
