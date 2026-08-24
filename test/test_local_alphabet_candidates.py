import unittest
from pathlib import Path

from scripts.audit_local_alphabet_candidates import alphabet_source, cap_named_sessions
from scripts.audit_local_citizen100_candidates import Candidate


def candidate(path: str, score: float) -> Candidate:
    return Candidate("A", "A", "FINGERSPELL_A", "exact", "source", path, "", 640, 480, 30, 30, 1, 100, 0, 50, score)


class LocalAlphabetCandidatesTests(unittest.TestCase):
    def test_source_rules_exclude_duplicate_imports(self):
        self.assertEqual(
            alphabet_source(Path("abcd1234__from_MARIAH_A__copy.mp4"), "A")[0],
            "duplicate_mariah_copy",
        )
        self.assertEqual(alphabet_source(Path("abcd1234.mp4"), "A")[0], "local_hex_unknown_session")

    def test_each_named_session_contributes_at_most_one(self):
        rows = [
            (candidate("a.mp4", 0.8), object(), "dwight"),
            (candidate("b.mp4", 0.9), object(), "dwight"),
            (candidate("c.mp4", 0.7), object(), None),
        ]
        selected = cap_named_sessions(rows)
        self.assertEqual({row.raw_path for row, _ in selected}, {"b.mp4", "c.mp4"})


if __name__ == "__main__":
    unittest.main()
