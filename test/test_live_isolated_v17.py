import unittest
from types import SimpleNamespace

import numpy as np

from active.v17.extract_v17 import FrameDetection, HandDetection
from scripts.live_isolated_v17 import (
    AutoBoundary,
    BoundaryConfig,
    OllamaNaturalizer,
    ObservedFrame,
    SessionRecorder,
    TinyStage3Naturalizer,
    clicked_control,
    clicked_mode,
    draw_detection,
    landmarks_from_observations,
    trim_to_motion,
)


def observed(seconds: float, motion: float, hand_quality: float = 0.8) -> ObservedFrame:
    detection = FrameDetection(
        hands=[],
        body_xy=np.zeros((4, 2), np.float32),
        body_confidence=np.zeros(4, np.float32),
        face_xy=np.zeros((15, 2), np.float32),
        face_confidence=np.zeros(15, np.float32),
    )
    return ObservedFrame(
        frame=np.zeros((360, 640, 3), np.uint8), detection=detection,
        assigned={"left": None, "right": None}, seconds=seconds,
        motion=motion, hand_quality=hand_quality, face_quality=0.7,
    )


def observed_at(seconds: float, motion: float, x: float) -> ObservedFrame:
    hand = HandDetection(
        xy=np.tile(np.asarray((x, 0.5), np.float32), (21, 1)),
        confidence=np.ones(21, np.float32), chirality="right", score=0.9,
    )
    value = observed(seconds, motion)
    value.assigned = {"left": None, "right": hand}
    return value


class SessionRecorderTests(unittest.TestCase):
    def test_finishing_video_releases_the_writer_only_once(self):
        released = []
        recorder = SessionRecorder.__new__(SessionRecorder)
        recorder.writer = SimpleNamespace(release=lambda: released.append(True))
        recorder.video_finished = False

        recorder.finish_video()
        recorder.finish_video()

        self.assertEqual(released, [True])
        self.assertTrue(recorder.video_finished)


class AutoBoundaryTests(unittest.TestCase):
    def test_mode_buttons_select_default_and_cascade(self):
        self.assertEqual(clicked_mode(350, 90), "hybrid")
        self.assertEqual(clicked_mode(500, 90), "cascade")
        self.assertIsNone(clicked_mode(100, 90))

    def test_bottom_controls_follow_frame_height(self):
        self.assertEqual(clicked_control(50, 680, 1280, 720), "reset")
        self.assertEqual(clicked_control(200, 680, 1280, 720), "finish")
        self.assertIsNone(clicked_control(400, 680, 1280, 720))

    def test_motion_then_rest_emits_one_clip_with_preroll(self):
        config = BoundaryConfig(
            processing_fps=10, preroll_seconds=0.3, start_frames=2,
            quiet_seconds=0.3, minimum_sign_seconds=0.2,
        )
        boundary = AutoBoundary(config)
        emitted = []
        motions = [0, 0, 0.02, 0.03, 0.02, 0.01, 0, 0, 0]
        for index, motion in enumerate(motions):
            clip = boundary.update(observed(index / 10, motion))
            if clip is not None:
                emitted.append(clip)
        self.assertEqual(len(emitted), 1)
        self.assertLessEqual(emitted[0][0].seconds, 0.2)
        self.assertEqual(boundary.state, "COOLDOWN")

    def test_eof_falls_back_when_short_clip_never_crosses_motion_gate(self):
        boundary = AutoBoundary(BoundaryConfig(processing_fps=10))
        frames = [observed(index / 10, 0.0) for index in range(5)]
        for frame in frames:
            boundary.update(frame)
        self.assertEqual(boundary.finish(frames), frames)

    def test_static_hold_away_from_neutral_does_not_end_sign(self):
        config = BoundaryConfig(
            processing_fps=10, start_frames=2, quiet_seconds=0.3,
            minimum_sign_seconds=0.2,
        )
        boundary = AutoBoundary(config)
        frames = [
            observed_at(0.0, 0.0, 0.2),
            observed_at(0.1, 0.02, 0.35),
            observed_at(0.2, 0.02, 0.5),
            observed_at(0.3, 0.0, 0.5),
            observed_at(0.4, 0.0, 0.5),
            observed_at(0.5, 0.0, 0.5),
            observed_at(0.6, 0.0, 0.5),
        ]
        self.assertTrue(all(boundary.update(frame) is None for frame in frames))
        self.assertEqual(boundary.state, "SIGNING")

    def test_low_motion_mode_ends_at_a_non_neutral_hold(self):
        config = BoundaryConfig(
            processing_fps=10, start_frames=2, quiet_seconds=0.3,
            minimum_sign_seconds=0.2, require_neutral=False,
        )
        boundary = AutoBoundary(config)
        frames = [
            observed_at(0.0, 0.0, 0.2),
            observed_at(0.1, 0.02, 0.35),
            observed_at(0.2, 0.02, 0.5),
            observed_at(0.3, 0.0, 0.5),
            observed_at(0.4, 0.0, 0.5),
            observed_at(0.5, 0.0, 0.5),
        ]
        emitted = [boundary.update(frame) for frame in frames]
        self.assertIsNotNone(emitted[-1])
        self.assertEqual(boundary.state, "COOLDOWN")

    def test_motion_trim_removes_live_neutral_padding(self):
        frames = [
            observed(index / 30, motion)
            for index, motion in enumerate((0, 0, 0, .02, .03, .02, 0, 0, 0))
        ]
        trimmed, diagnostics = trim_to_motion(frames, .006, context_frames=1)
        self.assertEqual(trimmed, frames[2:7])
        self.assertFalse(diagnostics["motion_trim_fallback"])

    def test_sparse_global_body_frame_is_not_discarded_by_clip_offset(self):
        frames = [observed_at(index / 30, .02, .3 + index * .01) for index in range(5)]
        frames[1].detection.body_xy[:] = np.asarray(
            ((.4, .4), (.6, .4), (.4, .7), (.6, .7)), np.float32
        )
        frames[1].detection.body_confidence[:] = 1
        _, diagnostics = landmarks_from_observations(frames)
        self.assertGreater(diagnostics["body_presence_fraction"], 0)
        self.assertEqual(diagnostics["normalization_scale_source"], "shoulder_width")

    def test_overlay_draws_white_joints_and_black_bones(self):
        frame = np.full((100, 100, 3), 128, np.uint8)
        hand = HandDetection(
            xy=np.zeros((21, 2), np.float32),
            confidence=np.zeros(21, np.float32), chirality="right", score=.9,
        )
        hand.xy[0], hand.xy[1] = (.2, .5), (.4, .5)
        hand.confidence[:2] = 1
        item = observed(0, 0)
        item.detection.hands = [hand]
        output = draw_detection(frame, item, mirror=False)
        self.assertTrue(np.array_equal(output[50, 20], (255, 255, 255)))
        self.assertTrue(np.array_equal(output[50, 30], (0, 0, 0)))

    def test_live_face_can_draw_every_frame_but_feed_stage1_sparsely(self):
        frames = [observed_at(index / 30, .02, .3 + index * .01) for index in range(5)]
        for frame in frames:
            frame.detection.face_xy[:] = .5
            frame.detection.face_confidence[:] = 1
            frame.face_for_features = False
        _, hidden = landmarks_from_observations(frames)
        frames[1].face_for_features = True
        _, sparse = landmarks_from_observations(frames)
        self.assertEqual(hidden["face_presence_fraction"], 0)
        self.assertGreater(sparse["face_presence_fraction"], 0)


class OllamaNaturalizerTests(unittest.TestCase):
    def naturalizer(self) -> OllamaNaturalizer:
        return OllamaNaturalizer(SimpleNamespace(
            ollama_model="llama3.2:1b",
            ollama_url="http://127.0.0.1:11434/api/generate",
            ollama_timeout=1.0,
            no_ollama=False,
        ))

    def test_accepts_exact_gloss_audit_from_ollama(self):
        naturalizer = self.naturalizer()
        naturalizer._call = lambda _prompt: {
            "response": '{"sentence":"I need water.",'
                        '"used_glosses":["I","NEED","WATER"]}',
            "total_duration": 10,
        }
        result = naturalizer.rephrase(["I", "NEED", "WATER"])
        self.assertEqual(result["sentence"], "I need water.")
        self.assertFalse(result["safe_fallback_used"])

    def test_rejects_changed_gloss_audit_and_uses_reviewed_template(self):
        naturalizer = self.naturalizer()
        naturalizer._call = lambda _prompt: {
            "response": '{"sentence":"I want water.",'
                        '"used_glosses":["I","WANT","WATER"]}',
        }
        result = naturalizer.rephrase(["I", "NEED", "WATER"])
        self.assertEqual(result["sentence"], "I need water.")
        self.assertTrue(result["safe_fallback_used"])
        self.assertEqual(result["rendering_mode"], "reviewed_template")


class TinyStage3NaturalizerTests(unittest.TestCase):
    def test_reviewed_template_does_not_load_model(self):
        naturalizer = TinyStage3Naturalizer(SimpleNamespace(
            stage3_checkpoint="unused",
            stage3_device="cpu",
        ))
        naturalizer._ensure_loaded = lambda: self.fail("template should not load model")
        result = naturalizer.rephrase(["I", "NEED", "WATER"])
        self.assertEqual(result["sentence"], "I need water.")
        self.assertEqual(result["rendering_mode"], "reviewed_template")
        self.assertFalse(result["safe_fallback_used"])


if __name__ == "__main__":
    unittest.main()
