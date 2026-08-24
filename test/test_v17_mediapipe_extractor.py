import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from active.v17.geometry_v17 import interpolate_scalar_short_gaps
from active.v17.schema_mediapipe_v17 import (
    MediaPipeV17Config,
    schema_fingerprint as mediapipe_fingerprint,
)
from active.v17.schema_v17 import V17Config, schema_fingerprint as apple_fingerprint
from active.v17.extract_mediapipe_v17 import (
    DEFAULT_MODEL_PATH,
    MediaPipeHybridDetector,
    extract_frames_mediapipe_v17,
    load_mediapipe_v17_result,
    save_mediapipe_v17_result,
)


class MediaPipeV17UnitTest(unittest.TestCase):
    def test_scalar_depth_interpolation_is_bounded_and_never_extrapolates(self):
        values = np.zeros((5, 1), dtype=np.float32)
        confidence = np.zeros((5, 1), dtype=np.float32)
        values[1, 0], values[3, 0] = 2.0, 6.0
        confidence[1, 0], confidence[3, 0] = 0.8, 0.6
        result, result_confidence = interpolate_scalar_short_gaps(
            values, confidence, max_gap=1
        )
        self.assertEqual(result[:, 0].tolist(), [0.0, 2.0, 4.0, 6.0, 0.0])
        self.assertAlmostEqual(float(result_confidence[2, 0]), 0.3)
        self.assertEqual(float(result_confidence[0, 0]), 0.0)
        self.assertEqual(float(result_confidence[4, 0]), 0.0)

    def test_mediapipe_and_apple_archives_cannot_share_a_fingerprint(self):
        self.assertNotEqual(
            mediapipe_fingerprint(MediaPipeV17Config()),
            apple_fingerprint(V17Config()),
        )


@unittest.skipUnless(DEFAULT_MODEL_PATH.exists(), "MediaPipe task model unavailable")
class MediaPipeV17RealTest(unittest.TestCase):
    VIDEO = Path(
        "data/local/ios100_audit/asl_citizen/train/HELLO/"
        "11225598264242453-HELLO.mp4"
    )

    def test_real_orientation_equivalence_world_depth_and_schema_round_trip(self):
        if not self.VIDEO.exists():
            self.skipTest("Citizen audit video unavailable")
        capture = cv2.VideoCapture(str(self.VIDEO))
        frames = []
        while len(frames) < 16:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        config = MediaPipeV17Config(
            maximum_source_frames=32, include_apple_auxiliary=False
        )
        detector = MediaPipeHybridDetector(DEFAULT_MODEL_PATH, config)
        try:
            baseline = extract_frames_mediapipe_v17(frames, config, detector)
            rotated = extract_frames_mediapipe_v17(
                [cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) for frame in frames],
                config,
                detector,
                rotation_clockwise=270,
            )
            mirrored = extract_frames_mediapipe_v17(
                [cv2.flip(frame, 1) for frame in frames],
                config,
                detector,
                input_mirrored=True,
            )
        finally:
            detector.close()
        self.assertIsNotNone(baseline)
        np.testing.assert_array_equal(baseline.features, rotated.features)
        np.testing.assert_array_equal(baseline.features, mirrored.features)
        present_depth = baseline.features[..., 2][baseline.features[..., 3] > 0]
        self.assertGreater(float(np.abs(present_depth).max()), 0.0)
        missing = baseline.features[..., 3] == 0
        self.assertEqual(float(np.abs(baseline.features[..., :3][missing]).max(initial=0)), 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.v17.npz"
            save_mediapipe_v17_result(path, baseline, config)
            loaded = load_mediapipe_v17_result(path, config)
            np.testing.assert_array_equal(loaded.features, baseline.features)
            with self.assertRaises(ValueError):
                load_mediapipe_v17_result(
                    path, MediaPipeV17Config(include_apple_auxiliary=True)
                )


if __name__ == "__main__":
    unittest.main()
