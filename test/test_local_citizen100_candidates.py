import unittest

import numpy as np

from scripts.audit_local_citizen100_candidates import (
    Candidate,
    classify_source,
    is_exact_pinned_raw,
    select_diverse,
)
from pathlib import Path


def candidate(name, quality, source="local_hex_unknown_session"):
    return Candidate(
        canonical_label="HELP",
        citizen_raw_gloss="HELP",
        citizen_asl_lex_code="D_01_042",
        lexical_tier="canonical_and_pinned_raw_text_equal",
        source_kind=source,
        raw_path=name,
        crop_path=name + ".jpg",
        width=640,
        height=480,
        frames=45,
        fps=30.0,
        duration_seconds=1.5,
        brightness=128.0,
        clipped_fraction=0.0,
        sharpness=100.0,
        quality_score=quality,
    )


class LocalCitizen100CandidateTest(unittest.TestCase):
    def test_exact_pinned_raw_gate_does_not_normalize_variants(self):
        self.assertTrue(
            is_exact_pinned_raw(
                {"canonical_label": "DRINK", "citizen_raw_gloss": "DRINK"}
            )
        )
        self.assertFalse(
            is_exact_pinned_raw(
                {"canonical_label": "DRINK", "citizen_raw_gloss": "DRINK2"}
            )
        )

    def test_source_classification_is_fail_closed(self):
        self.assertEqual(
            classify_source(Path("deadbeef.mp4"), "HELP"),
            "local_hex_unknown_session",
        )
        self.assertEqual(
            classify_source(Path("HELP_7.mp4"), "HELP"),
            "local_numbered_single_session",
        )
        self.assertEqual(classify_source(Path("msasl_x.mp4"), "HELP"), "msasl")
        self.assertEqual(classify_source(Path("mystery.mp4"), "HELP"), "unknown")

    def test_diversity_beats_near_duplicate_quality(self):
        first = candidate("first", 1.0)
        duplicate = candidate("duplicate", 0.99)
        diverse = candidate("diverse", 0.80)
        descriptors = {
            "first": np.array([1.0, 0.0], dtype=np.float32),
            "duplicate": np.array([0.999, 0.045], dtype=np.float32),
            "diverse": np.array([0.0, 1.0], dtype=np.float32),
        }
        selected = select_diverse([first, duplicate, diverse], descriptors, 2, 0.08)
        self.assertEqual([item.raw_path for item in selected], ["first", "diverse"])

    def test_numbered_single_session_is_capped_at_one(self):
        items = [
            candidate("a", 1.0, "local_numbered_single_session"),
            candidate("b", 0.9, "local_numbered_single_session"),
            candidate("c", 0.8),
        ]
        descriptors = {
            "a": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "b": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "c": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        selected = select_diverse(items, descriptors, 3, 0.08)
        self.assertEqual(sum(x.source_kind == "local_numbered_single_session" for x in selected), 1)


if __name__ == "__main__":
    unittest.main()
