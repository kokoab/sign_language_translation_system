from __future__ import annotations

import unittest

from scripts.live_reel_stage1_v17 import (
    StableGlossLock,
    VerifiedCommitLock,
    add_targeted_lip_evidence,
    parser,
    resolve_good_thankyou_context,
    select_finished_sequence,
)


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


class StableGlossLockTest(unittest.TestCase):
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
        self.assertEqual(args.start_frames, 1)
        self.assertEqual(args.release_hits, 1)
        self.assertEqual(args.transition_overlap_seconds, 0.0)
        self.assertEqual(args.commit_score, 0.45)
        self.assertEqual(args.commit_hits, 2)
        self.assertEqual(args.instant_commit_score, 0.80)
        self.assertFalse(args.no_stage2_arbiter)

    def test_lips_resolve_only_the_closed_good_thankyou_pair(self) -> None:
        base = {
            "accepted": True,
            "gloss": "GOODBYE",
            "candidate_gloss": "GOODBYE",
            "diagnostics": {},
            "top3": [],
        }
        selected = add_targeted_lip_evidence(
            FixedLipModel("GOOD", 0.9), [], base, "THANKYOU", 0.6
        )
        self.assertEqual(selected["gloss"], "GOOD")
        self.assertEqual(
            selected["diagnostics"]["targeted_lip_source"],
            "media_pipe_lip_markers",
        )
        untouched = add_targeted_lip_evidence(
            FixedLipModel("GOOD", 0.9), [], base, "HELLO", 0.6
        )
        self.assertEqual(untouched["gloss"], "GOODBYE")

    def test_weak_lip_result_keeps_the_landmark_proposal(self) -> None:
        base = {
            "accepted": True,
            "gloss": "GOODBYE",
            "candidate_gloss": "GOODBYE",
            "diagnostics": {},
            "top3": [],
        }
        selected = add_targeted_lip_evidence(
            FixedLipModel("THANKYOU", 0.51), [], base, "GOOD", 0.6
        )
        self.assertEqual(selected["gloss"], "GOOD")
        self.assertEqual(
            selected["diagnostics"]["targeted_lip_source"],
            "landmark_proposal",
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
