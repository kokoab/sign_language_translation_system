import unittest

from scripts.select_local_citizen100_review_shortlist import (
    passes_clip_review_gate,
    passes_review_gate,
)


class LocalCitizen100ReviewShortlistTest(unittest.TestCase):
    def setUp(self):
        self.provenance = {"lexical_tier": "canonical_and_pinned_raw_text_equal"}
        self.prediction = {"top5_hit": "True"}
        self.triage = {"triage": "model_consistent_manual_variant_review_required"}
        self.diagnostics = {"observed_hand_frame_fraction": 0.8, "face_presence_fraction": 0.5}

    def test_exact_boundary_passes(self):
        self.assertTrue(passes_review_gate(self.provenance, self.prediction, self.triage, self.diagnostics))

    def test_clip_gate_does_not_require_class_consistency(self):
        self.triage["triage"] = "ambiguous_manual_variant_review_required"
        self.assertTrue(passes_clip_review_gate(self.provenance, self.prediction, self.diagnostics))
        self.assertFalse(
            passes_review_gate(self.provenance, self.prediction, self.triage, self.diagnostics)
        )

    def test_variant_name_fails_closed(self):
        self.provenance["lexical_tier"] = "canonical_only_variant_review_required"
        self.assertFalse(passes_review_gate(self.provenance, self.prediction, self.triage, self.diagnostics))

    def test_low_hand_coverage_fails(self):
        self.diagnostics["observed_hand_frame_fraction"] = 0.79
        self.assertFalse(passes_review_gate(self.provenance, self.prediction, self.triage, self.diagnostics))


if __name__ == "__main__":
    unittest.main()
