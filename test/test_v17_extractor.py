import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from active.v17.extract_v17 import (
    AppleVisionDetector,
    FrameDetection,
    HandDetection,
    VISION_AVAILABLE,
    _upright_orientation_score,
    _upright_axis_score,
    _select_coarse_orientation_from_scores,
    assign_hands,
    choose_coarse_orientation_v17,
    extract_frames_v17,
    iter_videos,
    load_v17_result,
    limit_image_side,
    orient_frame,
    read_video_frames,
    rotate_frame_clockwise,
    save_v17_result,
)
from active.v17.geometry_v17 import (
    body_relative_normalize,
    image_normalized_to_isotropic,
    interpolate_short_gaps,
    resample_features,
)
from active.v17.schema_v17 import (
    BODY_START,
    NUM_NODES,
    V17Config,
)


class V17GeometryTest(unittest.TestCase):
    def test_batch_inventory_accepts_webm(self):
        with tempfile.TemporaryDirectory() as temporary:
            class_root = Path(temporary) / "HELLO"
            class_root.mkdir()
            (class_root / "clip.webm").touch()
            (class_root / "ignore.txt").touch()
            self.assertEqual(
                list(iter_videos(temporary)), [("HELLO", class_root / "clip.webm")]
            )

    def test_isotropic_coordinates_match_portrait_and_landscape(self):
        physical = np.array(
            [[[0.04, -0.08], [0.12, 0.03], [-0.02, 0.05]]],
            dtype=np.float32,
        )
        outputs = []
        for width, height in ((1600, 900), (900, 1600)):
            longest = float(max(width, height))
            normalized = physical.copy()
            normalized[..., 0] = (width / 2 + physical[..., 0] * longest) / width
            normalized[..., 1] = (height / 2 + physical[..., 1] * longest) / height
            outputs.append(image_normalized_to_isotropic(normalized, width, height))
        np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-7)

    def test_gap_interpolation_fills_every_joint_without_partial_zero_bug(self):
        xy = np.zeros((3, 2, 2), dtype=np.float32)
        confidence = np.zeros((3, 2), dtype=np.float32)
        xy[0] = ((0.0, 0.0), (1.0, 2.0))
        xy[2] = ((2.0, 4.0), (3.0, 6.0))
        confidence[[0, 2]] = 1.0
        filled, filled_conf = interpolate_short_gaps(xy, confidence, max_gap=1)
        np.testing.assert_allclose(filled[1], ((1.0, 2.0), (2.0, 4.0)))
        np.testing.assert_allclose(filled_conf[1], (0.5, 0.5))

    def test_normalization_keeps_missing_nodes_exactly_zero(self):
        xy = np.zeros((4, NUM_NODES, 2), dtype=np.float32)
        confidence = np.zeros((4, NUM_NODES), dtype=np.float32)
        xy[:, BODY_START] = (-0.1, 0.0)
        xy[:, BODY_START + 1] = (0.1, 0.0)
        confidence[:, BODY_START:BODY_START + 2] = 1.0
        xy[:, 21] = (0.2, -0.2)
        confidence[:, 21] = 1.0
        normalized, depth, _ = body_relative_normalize(xy, confidence)
        missing = confidence == 0
        self.assertEqual(float(np.abs(normalized[missing]).max(initial=0)), 0.0)
        self.assertEqual(float(np.abs(depth[missing]).max(initial=0)), 0.0)

    def test_resampling_preserves_binary_presence(self):
        features = np.zeros((5, NUM_NODES, 5), dtype=np.float32)
        features[1:4, 0, 0] = 2.0
        features[1:4, 0, 3] = 1.0
        features[1:4, 0, 4] = 0.8
        result = resample_features(features, 32)
        self.assertEqual(set(np.unique(result[..., 3])), {0.0, 1.0})
        self.assertEqual(float(np.abs(result[..., :3][result[..., 3] == 0]).max(initial=0)), 0.0)


class V17HandAssignmentTest(unittest.TestCase):
    @staticmethod
    def detection(chirality, wrist_x, score=0.9):
        xy = np.zeros((21, 2), dtype=np.float32)
        confidence = np.ones(21, dtype=np.float32) * score
        xy[0] = (wrist_x, 0.5)
        return HandDetection(xy, confidence, chirality, score)

    def test_known_chirality_is_not_reversed(self):
        left = self.detection("left", 0.8)
        right = self.detection("right", 0.2)
        assigned = assign_hands(
            [right, left], {"left": None, "right": None}
        )
        self.assertIs(assigned["left"], left)
        self.assertIs(assigned["right"], right)

    def test_unknown_hand_uses_temporal_continuity(self):
        unknown = self.detection("unknown", 0.22)
        assigned = assign_hands(
            [unknown],
            {"left": np.array([0.85, 0.5]), "right": np.array([0.20, 0.5])},
        )
        self.assertIs(assigned["right"], unknown)


class V17FrameTransformTest(unittest.TestCase):
    def test_coarse_selector_uses_face_for_near_axis_ties(self):
        scores = {
            "0": {"body": 0.7, "axis": 0.95, "face": 0.2},
            "90": {"body": 0.4, "axis": 0.1, "face": 0.0},
            "180": {"body": 0.7, "axis": 1.0, "face": -0.2},
            "270": {"body": 0.4, "axis": 0.1, "face": 0.0},
        }
        self.assertEqual(_select_coarse_orientation_from_scores(scores), 0.0)
        scores["270"] = {"body": 0.7, "axis": 0.99, "face": 0.9}
        scores["180"]["axis"] = 0.7
        self.assertEqual(_select_coarse_orientation_from_scores(scores), 270.0)

    def test_upright_score_prefers_mouth_below_eyes(self):
        body_xy = np.zeros((4, 2), dtype=np.float32)
        body_confidence = np.ones(4, dtype=np.float32) * 0.5
        face_xy = np.zeros((15, 2), dtype=np.float32)
        face_confidence = np.ones(15, dtype=np.float32)
        face_xy[:2, 1] = 0.3
        face_xy[7:11, 1] = 0.6
        upright = FrameDetection([], body_xy, body_confidence, face_xy, face_confidence)
        face_xy_inverted = face_xy.copy()
        face_xy_inverted[:2, 1] = 0.7
        face_xy_inverted[7:11, 1] = 0.4
        inverted = FrameDetection(
            [], body_xy, body_confidence, face_xy_inverted, face_confidence
        )
        self.assertGreater(
            _upright_orientation_score(upright),
            _upright_orientation_score(inverted),
        )
        body_xy[0] = (0.2, 0.4)
        body_xy[1] = (0.8, 0.4)
        horizontal = FrameDetection(
            [], body_xy.copy(), body_confidence, face_xy, face_confidence
        )
        body_xy[1] = (0.2, 0.9)
        vertical = FrameDetection([], body_xy, body_confidence, face_xy, face_confidence)
        self.assertGreater(_upright_axis_score(horizontal), _upright_axis_score(vertical))

    def test_rotation_and_unmirror_are_exactly_reversible(self):
        rng = np.random.default_rng(4)
        frame = rng.integers(0, 256, (31, 47, 3), dtype=np.uint8)
        rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        restored = orient_frame(rotated, 270, False)
        np.testing.assert_array_equal(restored, frame)
        mirrored = cv2.flip(frame, 1)
        np.testing.assert_array_equal(orient_frame(mirrored, 0, True), frame)

    def test_image_limit_preserves_aspect_and_never_upscales(self):
        portrait = np.zeros((2592, 1944, 3), dtype=np.uint8)
        resized = limit_image_side(portrait, 1280)
        self.assertEqual(resized.shape[:2], (1280, 960))
        small = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertIs(limit_image_side(small, 1280), small)

    def test_arbitrary_rotation_expands_canvas_without_stretching_or_cropping(self):
        frame = np.zeros((41, 73, 3), dtype=np.uint8)
        frame[0:4, 0:4] = (10, 20, 250)
        frame[0:4, -4:] = (20, 250, 10)
        frame[-4:, 0:4] = (250, 10, 20)
        frame[-4:, -4:] = (250, 250, 20)
        rotated = rotate_frame_clockwise(frame, 37.0)
        self.assertGreater(rotated.shape[0], frame.shape[0])
        self.assertGreater(rotated.shape[1], frame.shape[1])
        # Every unique corner survives; an in-place same-size rotation would crop them.
        for channel in range(3):
            self.assertGreater(int(rotated[..., channel].max()), 200)
        original_diagonal = float(np.hypot(*frame.shape[:2]))
        rotated_diagonal = float(np.hypot(*rotated.shape[:2]))
        self.assertGreaterEqual(rotated_diagonal, original_diagonal)

    def test_arbitrary_rotation_rejects_nonfinite_angles(self):
        frame = np.zeros((12, 8, 3), dtype=np.uint8)
        for angle in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                rotate_frame_clockwise(frame, angle)

    def test_underreported_video_length_stays_memory_bounded(self):
        class FakeCapture:
            def __init__(self, _):
                self.index = 0

            def isOpened(self):
                return True

            def get(self, property_id):
                if property_id == cv2.CAP_PROP_FRAME_COUNT:
                    return 10
                if property_id == cv2.CAP_PROP_FPS:
                    return 30
                return 0

            def set(self, *_):
                return True

            def read(self):
                if self.index >= 120:
                    return False, None
                frame = np.full((12, 8, 3), self.index, dtype=np.uint8)
                self.index += 1
                return True, frame

            def release(self):
                return None

        with patch("active.v17.extract_v17.cv2.VideoCapture", FakeCapture):
            frames, metadata = read_video_frames(
                "synthetic.mp4", maximum_frames=32, maximum_image_side=1280
            )
        self.assertEqual(len(frames), 32)
        self.assertEqual(metadata["decoded_frame_count"], 120)
        sampled_values = [int(frame[0, 0, 0]) for frame in frames]
        self.assertEqual(sampled_values, sorted(sampled_values))


@unittest.skipUnless(VISION_AVAILABLE, "Apple Vision PyObjC bridge unavailable")
class V17RealVisionTest(unittest.TestCase):
    VIDEO = Path(
        "data/local/ios100_audit/asl_citizen/train/HELLO/"
        "11225598264242453-HELLO.mp4"
    )

    def test_real_rotated_and_mirrored_pixels_produce_same_features(self):
        if not self.VIDEO.exists():
            self.skipTest("external audit video not downloaded")
        capture = cv2.VideoCapture(str(self.VIDEO))
        frames = []
        while len(frames) < 24:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        config = V17Config(maximum_source_frames=32)
        baseline = extract_frames_v17(frames, config)
        rotated_frames = [cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) for frame in frames]
        rotated = extract_frames_v17(
            rotated_frames, config, rotation_clockwise=270
        )
        mirrored_frames = [cv2.flip(frame, 1) for frame in frames]
        mirrored = extract_frames_v17(
            mirrored_frames, config, input_mirrored=True
        )
        self.assertIsNotNone(baseline)
        self.assertIsNotNone(rotated)
        self.assertIsNotNone(mirrored)
        np.testing.assert_array_equal(baseline.features, rotated.features)
        np.testing.assert_array_equal(baseline.features, mirrored.features)
        self.assertGreater(baseline.diagnostics["face_presence_fraction"], 0.0)

    def test_real_coarse_orientation_recovers_right_angles(self):
        if not self.VIDEO.exists():
            self.skipTest("external audit video not downloaded")
        capture = cv2.VideoCapture(str(self.VIDEO))
        frames = []
        while len(frames) < 24:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        detector = AppleVisionDetector(0.15)
        for source_angle, expected_correction in (
            (0, 0),
            (17, 0),
            (37, 0),
            (73, 270),
            (90, 270),
            (123, 270),
            (180, 180),
            (270, 90),
        ):
            source = [rotate_frame_clockwise(frame, source_angle) for frame in frames]
            correction, _ = choose_coarse_orientation_v17(source, detector)
            self.assertEqual(correction, expected_correction)

    def test_result_round_trip_enforces_schema(self):
        if not self.VIDEO.exists():
            self.skipTest("external audit video not downloaded")
        capture = cv2.VideoCapture(str(self.VIDEO))
        frames = []
        while len(frames) < 16:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        config = V17Config(maximum_source_frames=32)
        result = extract_frames_v17(frames, config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.v17.npz"
            save_v17_result(path, result, config)
            loaded = load_v17_result(path, config)
            np.testing.assert_array_equal(loaded.features, result.features)
            with self.assertRaises(ValueError):
                load_v17_result(path, V17Config(face_interval=4))


if __name__ == "__main__":
    unittest.main()
