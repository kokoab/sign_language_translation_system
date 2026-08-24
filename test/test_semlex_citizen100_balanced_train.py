import unittest

from scripts.select_semlex_citizen100_balanced_train import select_balanced


class SemLexBalancedTrainTests(unittest.TestCase):
    def test_caps_signers_and_excludes_mismatch_priority(self):
        rows = [
            {
                "canonical_label": label,
                "semlex_signer_id": signer,
                "semlex_video_id": f"{label}-{signer}",
                "quality_score": score,
                "observed_hand_frame_fraction": score,
                "face_presence_fraction": score,
            }
            for label, signer, score in [
                ("HELLO", "1", 0.9),
                ("HELLO", "2", 0.8),
                ("HELLO", "3", 0.7),
                ("TAKE", "1", 1.0),
            ]
        ]
        selected, excluded = select_balanced(
            rows, {"HELLO": "model_consistent", "TAKE": "mismatch_review_priority"}, 2
        )
        self.assertEqual([row["semlex_signer_id"] for row in selected], ["1", "2"])
        self.assertEqual(excluded, ["TAKE"])

    def test_zero_cap_keeps_all_quality_passing_distinct_signers(self):
        rows = [
            {
                "canonical_label": "HELLO",
                "semlex_signer_id": signer,
                "semlex_video_id": f"HELLO-{signer}",
                "quality_score": observed,
                "observed_hand_frame_fraction": observed,
                "hand_presence_fraction": 0.5,
                "face_presence_fraction": 0.8,
            }
            for signer, observed in [("1", 0.9), ("2", 0.8), ("3", 0.6)]
        ]
        selected, excluded = select_balanced(
            rows,
            {"HELLO": "model_consistent"},
            0,
            minimum_observed_hand=0.7,
            minimum_hand_presence=0.3,
            minimum_face_presence=0.5,
        )
        self.assertEqual(
            [row["semlex_signer_id"] for row in selected], ["1", "2"]
        )
        self.assertEqual(excluded, [])


if __name__ == "__main__":
    unittest.main()
