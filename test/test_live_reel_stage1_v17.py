from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.live_reel_stage1_v17 import (
    FinishGesture,
    ReelCascadeClassifier,
    StableGlossLock,
    VerifiedCommitLock,
    add_targeted_lip_evidence,
    clicked_reel_control,
    draw_reel_detection,
    is_finish_gesture,
    parser,
    persistent_auxiliary_detection,
    resolve_good_thankyou_context,
    select_finished_sequence,
)
from active.v17.extract_v17 import HandDetection


class FixedLipModel:
    def __init__(self, label: str, confidence: float):
        self.value = {
            "label": label,
            "confidence": confidence,
            "thankyou_probability": 0.9 if label == "THANKYOU" else 0.1,
        }

    def predict(self, _sequence):
        return self.value


def prediction(label: str, accepted: bool = True) -> dict[str, object]:
    return {"gloss": label if accepted else "UNKNOWN", "accepted": accepted}


def open_hand(x: float) -> HandDetection:
    points = np.zeros((21, 2), np.float32)
    points[0] = (x, 0.72)
    points[1:5] = (
        (x + 0.02, 0.67), (x + 0.04, 0.62),
        (x + 0.07, 0.59), (x + 0.10, 0.56),
    )
    for offset, (mcp, pip, dip, tip) in zip(
        (-0.06, -0.02, 0.02, 0.06),
        ((5, 6, 7, 8), (9, 10, 11, 12),
         (13, 14, 15, 16), (17, 18, 19, 20)),
    ):
        points[mcp] = (x + offset, 0.62)
        points[pip] = (x + offset, 0.54)
        points[dip] = (x + offset, 0.47)
        points[tip] = (x + offset, 0.39)
    return HandDetection(
        xy=points, confidence=np.ones(21, np.float32),
        chirality="unknown", score=1.0,
    )


class StableGlossLockTest(unittest.TestCase):
    def test_ten_finger_finish_requires_two_open_upright_hands(self) -> None:
        hands = {"left": open_hand(0.32), "right": open_hand(0.68)}
        self.assertTrue(is_finish_gesture(hands))
        closed = open_hand(0.68)
        closed.xy[[8, 12, 16, 20], 1] = 0.60
        self.assertFalse(is_finish_gesture({"left": hands["left"], "right": closed}))
        self.assertFalse(is_finish_gesture({"left": hands["left"], "right": None}))

    def test_finish_gesture_holds_latches_and_rearms_after_release(self) -> None:
        hands = {"left": open_hand(0.32), "right": open_hand(0.68)}
        gesture = FinishGesture(hold_seconds=0.65)
        self.assertFalse(gesture.update(hands, 0.0))
        self.assertFalse(gesture.update(hands, 0.3))
        self.assertTrue(gesture.update(hands, 0.7))
        self.assertFalse(gesture.update(hands, 0.9))
        self.assertFalse(gesture.update({"left": None, "right": None}, 1.1))
        self.assertFalse(gesture.update(hands, 1.2))
        self.assertTrue(gesture.update(hands, 1.9))

    def test_requires_repeated_agreement(self) -> None:
        lock = StableGlossLock(required_hits=3, release_hits=2)
        self.assertIsNone(lock.update(prediction("HELLO")))
        self.assertIsNone(lock.update(prediction("HELLO")))
        self.assertEqual(lock.update(prediction("HELLO")), "HELLO")

    def test_suppresses_duplicate_until_a_different_label_is_stable(self) -> None:
        lock = StableGlossLock(required_hits=2, release_hits=2)
        lock.update(prediction("HELLO"))
        self.assertEqual(lock.update(prediction("HELLO")), "HELLO")
        lock.verified("HELLO")
        self.assertIsNone(lock.update(prediction("HELLO")))
        self.assertIsNone(lock.update(prediction("HOW")))
        self.assertIsNone(lock.update(prediction("HOW")))
        self.assertEqual(lock.update(prediction("HOW")), "HOW")

    def test_rejected_predictions_break_a_partial_lock(self) -> None:
        lock = StableGlossLock(required_hits=2, release_hits=2)
        lock.update(prediction("YOU"))
        lock.update(prediction("UNKNOWN", accepted=False))
        lock.update(prediction("UNKNOWN", accepted=False))
        self.assertIsNone(lock.candidate)
        self.assertIsNone(lock.update(prediction("YOU")))
        self.assertEqual(lock.update(prediction("YOU")), "YOU")

    def test_no_hands_rearms_an_identical_next_sign(self) -> None:
        lock = StableGlossLock(required_hits=2, release_hits=2)
        lock.update(prediction("YES"))
        self.assertEqual(lock.update(prediction("YES")), "YES")
        lock.verified("YES")
        lock.no_hands()
        lock.update(prediction("YES"))
        self.assertEqual(lock.update(prediction("YES")), "YES")

    def test_parser_points_to_a_separate_phrase_adapted_model(self) -> None:
        args = parser().parse_args([])
        self.assertEqual(args.mode, "cascade")
        self.assertIn("Stage1PhraseAdaptReelV17", str(args.orientation_coreml))
        self.assertIn("UnifiedPhraseActivityAdaptReelV17", str(args.stage1_coreml))
        self.assertIn("unified_phrase_activity_adapt_reel", str(args.unified_checkpoint))
        self.assertIn("live_reel_stage1_v17", str(args.output_root))
        self.assertIn("phrase_crops", str(args.lip_marker_model))
        self.assertEqual(args.lip_marker_minimum_confidence, 0.999)
        self.assertEqual(args.processing_fps, 20.0)
        self.assertEqual(args.detection_image_side, 640)
        self.assertEqual(args.candidate_minimum_seconds, 0.50)
        self.assertEqual(args.probe_interval_seconds, 0.12)
        self.assertEqual(args.start_frames, 1)
        self.assertEqual(args.stability_hits, 2)
        self.assertEqual(args.release_hits, 1)
        self.assertEqual(args.transition_overlap_seconds, 0.0)
        self.assertEqual(args.commit_score, 0.45)
        self.assertEqual(args.commit_hits, 1)
        self.assertEqual(args.instant_commit_score, 0.80)
        self.assertTrue(args.no_stage2_arbiter)
        self.assertTrue(args.no_lip_marker_verifier)
        self.assertFalse(
            parser().parse_args(["--lip-marker-verifier"]).no_lip_marker_verifier
        )
        self.assertFalse(args.dense_model_auxiliary)
        self.assertFalse(args.no_finish_gesture)
        self.assertEqual(args.finish_gesture_hold_seconds, 0.4)

    def test_large_reel_controls_match_their_drawn_area(self) -> None:
        from scripts.reel_hud_v17 import reel_control_button_rects

        for action, (left, top, right, bottom) in reel_control_button_rects(
            1280, 720
        ).items():
            self.assertEqual(
                clicked_reel_control(
                    (left + right) // 2, (top + bottom) // 2, 1280, 720
                ),
                action,
            )
        self.assertIsNone(clicked_reel_control(20, 20, 1280, 720))

    def test_auxiliary_display_cache_does_not_modify_model_detection(self) -> None:
        from active.v17.extract_v17 import FrameDetection

        first = FrameDetection(
            [], np.ones((4, 2), np.float32), np.ones(4, np.float32),
            np.ones((15, 2), np.float32), np.ones(15, np.float32),
        )
        visible, body, face = persistent_auxiliary_detection(first, None, None)
        self.assertEqual(float(visible.body_confidence.sum()), 4.0)
        missing = FrameDetection(
            [], np.zeros((4, 2), np.float32), np.zeros(4, np.float32),
            np.zeros((15, 2), np.float32), np.zeros(15, np.float32),
        )
        visible, _, _ = persistent_auxiliary_detection(missing, body, face)
        self.assertEqual(float(visible.body_confidence.sum()), 4.0)
        self.assertEqual(float(visible.face_confidence.sum()), 15.0)
        self.assertEqual(float(missing.body_confidence.sum()), 0.0)

    def test_reel_overlay_uses_thin_white_bones(self) -> None:
        from active.v17.extract_v17 import FrameDetection, HandDetection

        frame = np.full((100, 100, 3), 64, np.uint8)
        hand = HandDetection(
            xy=np.zeros((21, 2), np.float32),
            confidence=np.zeros(21, np.float32), chirality="right", score=1.0,
        )
        hand.xy[0], hand.xy[1] = (0.2, 0.5), (0.4, 0.5)
        hand.confidence[:2] = 1
        detection = FrameDetection(
            [hand], np.zeros((4, 2), np.float32), np.zeros(4, np.float32),
            np.zeros((15, 2), np.float32), np.zeros(15, np.float32),
        )
        item = SimpleNamespace(detection=detection)
        output = draw_reel_detection(frame, item, mirror=False)
        self.assertTrue(np.all(output[50, 30] > 200))
        self.assertTrue(np.array_equal(output[48, 30], (64, 64, 64)))

    def test_weak_proposal_does_not_run_visual_fallback(self) -> None:
        class Orientation:
            def predict(self, _provider):
                logits = np.zeros((1, 100), np.float32)
                logits[0, 7] = 4.0
                return {"var_5535": logits}

        class Full:
            def classify(self, _observations):
                raise AssertionError("visual fallback ran inside the proposal")

        classifier = ReelCascadeClassifier.__new__(ReelCascadeClassifier)
        classifier.orientation = Orientation()
        classifier.full = Full()
        classifier.labels = [f"GLOSS_{index}" for index in range(100)]
        classifier.args = SimpleNamespace(
            quiet_motion=0.006,
            minimum_score=0.25,
            minimum_margin=0.08,
            maximum_accept_seconds=2.5,
            cascade_score=0.55,
        )
        observations = [SimpleNamespace(seconds=0.0), SimpleNamespace(seconds=1.0)]
        with patch(
            "scripts.live_reel_stage1_v17.trim_to_motion",
            return_value=(observations, {}),
        ), patch(
            "scripts.live_reel_stage1_v17.landmarks_from_observations",
            return_value=(np.zeros((32, 61, 5), np.float32), {}),
        ):
            result = classifier.classify(observations)
        self.assertEqual(result["candidate_gloss"], "GLOSS_7")
        self.assertEqual(result["mode"], "reel-cascade-landmark")
        self.assertEqual(result["latency_ms"]["hand_image_encoding"], 0.0)

    def test_learned_no_emit_rejects_a_partial_proposal(self) -> None:
        class Orientation:
            def predict(self, _provider):
                logits = np.zeros((1, 101), np.float32)
                logits[0, 7] = 4.0
                logits[0, 100] = 8.0
                return {"var_5535": logits}

        classifier = ReelCascadeClassifier.__new__(ReelCascadeClassifier)
        classifier.orientation = Orientation()
        classifier.labels = [f"GLOSS_{index}" for index in range(100)]
        classifier.args = SimpleNamespace(
            quiet_motion=0.006,
            minimum_score=0.25,
            minimum_margin=0.08,
            maximum_accept_seconds=2.5,
            cascade_score=0.55,
            no_emit_probability_threshold=0.77,
        )
        observations = [SimpleNamespace(seconds=0.0), SimpleNamespace(seconds=1.0)]
        with patch(
            "scripts.live_reel_stage1_v17.trim_to_motion",
            return_value=(observations, {}),
        ), patch(
            "scripts.live_reel_stage1_v17.landmarks_from_observations",
            return_value=(np.zeros((32, 61, 5), np.float32), {}),
        ):
            result = classifier.classify(observations)
        self.assertFalse(result["accepted"])
        self.assertIn("learned_no_emit", result["rejection_reasons"])
        self.assertGreater(result["diagnostics"]["no_emit_probability"], 0.77)

    def test_lips_resolve_only_a_closed_pair_model_disagreement(self) -> None:
        base = {
            "accepted": True,
            "gloss": "GOOD",
            "candidate_gloss": "GOOD",
            "diagnostics": {},
            "top3": [{"gloss": "GOOD", "model_score": 0.6}],
        }
        selected = add_targeted_lip_evidence(
            FixedLipModel("THANKYOU", 0.9), [], base, "THANKYOU", 0.6
        )
        self.assertEqual(selected["gloss"], "THANKYOU")
        self.assertEqual(
            selected["diagnostics"]["targeted_lip_source"],
            "media_pipe_lip_markers",
        )
        unrelated = {**base, "gloss": "GOODBYE", "candidate_gloss": "GOODBYE"}
        untouched = add_targeted_lip_evidence(
            FixedLipModel("GOOD", 0.9), [], unrelated, "HELLO", 0.6
        )
        self.assertEqual(untouched["gloss"], "GOODBYE")

    def test_agreeing_models_cannot_be_overridden_by_lips(self) -> None:
        base = {
            "accepted": True,
            "gloss": "GOOD",
            "candidate_gloss": "GOOD",
            "diagnostics": {},
            "top3": [{"gloss": "GOOD", "model_score": 0.6}],
        }
        selected = add_targeted_lip_evidence(
            FixedLipModel("THANKYOU", 1.0), [], base, "GOOD", 0.6
        )
        self.assertEqual(selected["gloss"], "GOOD")
        self.assertEqual(
            selected["diagnostics"]["targeted_lip_source"],
            "model_consensus",
        )

    def test_weak_lip_result_keeps_the_full_verifier(self) -> None:
        base = {
            "accepted": True,
            "gloss": "THANKYOU",
            "candidate_gloss": "THANKYOU",
            "diagnostics": {},
            "top3": [{"gloss": "THANKYOU", "model_score": 0.6}],
        }
        selected = add_targeted_lip_evidence(
            FixedLipModel("GOOD", 0.51), [], base, "GOOD", 0.6
        )
        self.assertEqual(selected["gloss"], "THANKYOU")
        self.assertEqual(
            selected["diagnostics"]["targeted_lip_source"], "full_verifier"
        )

    def test_ambiguous_pair_uses_only_grounded_following_context(self) -> None:
        self.assertEqual(resolve_good_thankyou_context("GOOD", "FRIEND"), "THANKYOU")
        self.assertEqual(resolve_good_thankyou_context("THANKYOU", "MORNING"), "GOOD")
        self.assertEqual(resolve_good_thankyou_context("GOOD", "SCHOOL"), "GOOD")

    def test_commit_lock_rejects_one_transition_but_accepts_repeat(self) -> None:
        lock = VerifiedCommitLock(required_hits=2, instant_score=0.8)
        self.assertFalse(lock.update(
            "EASY", 0.73, proposal="EASY", proposal_score=0.73,
            minimum_score=0.45,
        ))
        self.assertFalse(lock.update(
            "HOW", 0.52, proposal="HOW", proposal_score=0.55,
            minimum_score=0.45,
        ))
        self.assertTrue(lock.update(
            "HOW", 0.51, proposal="HELP", proposal_score=0.60,
            minimum_score=0.45,
        ))

    def test_commit_lock_accepts_strong_agreement_immediately(self) -> None:
        lock = VerifiedCommitLock(required_hits=2, instant_score=0.8)
        self.assertTrue(lock.update(
            "YOU", 0.84, proposal="YOU", proposal_score=0.84,
            minimum_score=0.45,
        ))

    def test_stage2_only_arbitrates_corroborated_multisign_input(self) -> None:
        self.assertEqual(
            select_finished_sequence(
                ["HELLO", "GOODBYE", "HOW"],
                ["HELLO", "HOW", "YOU"], 4.0, 1.6,
            ),
            (["HELLO", "HOW", "YOU"], "stage2_ctc_multisign_arbiter"),
        )
        self.assertEqual(
            select_finished_sequence(
                ["GOOD", "MORNING"], ["GOOD", "GOOD", "MORNING"],
                3.0, 1.6,
            )[0],
            ["GOOD", "MORNING"],
        )
        self.assertEqual(
            select_finished_sequence(["YOU"], ["YOU", "NEED"], 2.0, 1.6)[0],
            ["YOU"],
        )
        self.assertEqual(
            select_finished_sequence(
                ["THANKYOU"], ["THANKYOU", "THANKYOU", "FRIEND"],
                3.0, 1.6, stage2_stable=True,
            )[0],
            ["THANKYOU", "FRIEND"],
        )


if __name__ == "__main__":
    unittest.main()
