import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from active.v17.extract_v17 import extract_frames_v17
from active.v17.mediapipe_full_v17 import (
    DEFAULT_MODEL_DIR,
    FACE_MESH_INDICES,
    MediaPipeFullDetector,
    MediaPipeFullV17Config,
    load_result_features,
    save_result,
    schema_fingerprint,
    stamp_result,
)
from active.v17.schema_mediapipe_v17 import MediaPipeV17Config, schema_fingerprint as hybrid_fingerprint
from active.v17.schema_v17 import V17Config, schema_fingerprint as apple_fingerprint


class MediaPipeFullUnitTest(unittest.TestCase):
    def test_fingerprint_is_distinct_from_apple_hybrid_and_each_variant(self):
        full = schema_fingerprint(MediaPipeFullV17Config())
        self.assertNotEqual(full, apple_fingerprint(V17Config()))
        self.assertNotEqual(full, hybrid_fingerprint(MediaPipeV17Config()))
        self.assertNotEqual(full, schema_fingerprint(MediaPipeFullV17Config(backend="cpu")))
        self.assertNotEqual(full, schema_fingerprint(MediaPipeFullV17Config(pose_model="pose_landmarker_full")))

    def test_face_map_is_one_to_one(self):
        self.assertEqual(len(FACE_MESH_INDICES), 15)
        self.assertEqual(len(set(FACE_MESH_INDICES)), 15)

    def test_unreviewed_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            MediaPipeFullV17Config(pose_model="pose_landmarker_heavy").validate()
        with self.assertRaises(ValueError):
            MediaPipeFullV17Config(backend="metal").validate()


@unittest.skipUnless((DEFAULT_MODEL_DIR / "face_landmarker.task").exists(), "MediaPipe task models unavailable")
class MediaPipeFullRealTest(unittest.TestCase):
    VIDEO = Path("data/local/citizen100_v17/raw/train/HELLO")

    def frames(self):
        videos = sorted(self.VIDEO.glob("*.mp4")) if self.VIDEO.exists() else []
        if not videos:
            self.skipTest("Citizen raw video unavailable")
        capture = cv2.VideoCapture(str(videos[0]))
        frames = []
        while len(frames) < 96:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        return frames

    def test_orientation_reset_missing_values_and_schema_round_trip(self):
        frames = self.frames()
        config = MediaPipeFullV17Config()
        detector = MediaPipeFullDetector(config)
        try:
            baseline = extract_frames_v17(frames, config, detector=detector)
            rotated = extract_frames_v17([cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE) for f in frames], config,
                                         detector=detector, rotation_clockwise=270)
            mirrored = extract_frames_v17([cv2.flip(f, 1) for f in frames], config,
                                          detector=detector, input_mirrored=True)
            repeated = extract_frames_v17(frames, config, detector=detector)
        finally:
            detector.close()
        self.assertIsNotNone(baseline)
        np.testing.assert_array_equal(baseline.features, rotated.features)
        np.testing.assert_array_equal(baseline.features, mirrored.features)
        # A reused detector resets tracking between sequences.
        np.testing.assert_array_equal(baseline.features, repeated.features)
        features = baseline.features.astype(np.float32)
        missing = features[..., 3] == 0
        self.assertEqual(float(np.abs(features[..., :3][missing]).max(initial=0)), 0.0)
        self.assertEqual(float(np.abs(features[..., 4][missing]).max(initial=0)), 0.0)
        self.assertGreater(float(features[:, :42, 3].mean()), 0.0)
        stamp_result(baseline, config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.v17.npz"
            save_result(path, baseline, config)
            np.testing.assert_array_equal(load_result_features(path, config), baseline.features)
            with self.assertRaises(FileExistsError):
                save_result(path, baseline, config)
            with self.assertRaises(ValueError):
                load_result_features(path, MediaPipeFullV17Config(backend="cpu"))


if __name__ == "__main__":
    unittest.main()
