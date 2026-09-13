"""Keeps PROJECT_GROUND_TRUTH.md from growing back into a 833 KB log.

It did once: 12,098 lines re-sent on every agent turn. Current state belongs in
PROJECT_GROUND_TRUTH.md, history belongs in docs/ground_truth/<topic>/.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GT = ROOT / "PROJECT_GROUND_TRUTH.md"
ARCHIVE = ROOT / "docs" / "ground_truth"
GT_MAX_BYTES = 25_000  # ~6k tokens. Raise only with a deliberate reason.


class GroundTruthLayout(unittest.TestCase):
    def test_ground_truth_stays_small(self):
        size = GT.stat().st_size
        self.assertLess(
            size, GT_MAX_BYTES,
            f"PROJECT_GROUND_TRUTH.md is {size} B (cap {GT_MAX_BYTES}). It is current "
            f"state, not a log — move dated entries to docs/ground_truth/<topic>/log.md.",
        )

    def test_ground_truth_holds_no_dated_log_entries(self):
        dated = [l for l in GT.read_text().splitlines() if re.match(r"^##\s+20\d\d-", l)]
        self.assertEqual(dated, [], f"dated entries belong in the archive: {dated[:3]}")

    def test_every_archive_topic_is_in_the_map(self):
        mapped = (ARCHIVE / "MAP.md").read_text()
        for topic in sorted(p.name for p in ARCHIVE.iterdir() if p.is_dir()):
            self.assertIn(f"`{topic}/`", mapped, f"{topic} missing from MAP.md")

    def test_large_file_index_stays_a_lookup_table(self):
        """The index must not become the thing it protects against."""
        idx = ROOT / "artifacts" / "LARGE_FILES.md"
        if not idx.exists():
            self.skipTest("run scripts/index_large_artifacts_v17.py to generate it")
        size = idx.stat().st_size
        self.assertLess(size, 40_000, f"LARGE_FILES.md is {size} B — raise MIN_BYTES.")

    def test_archive_entries_are_dated(self):
        for f in sorted(ARCHIVE.glob("*/*.md")):
            for line in f.read_text().splitlines():
                if re.match(r"^##[^#]", line) and "undated" not in line:
                    self.assertRegex(
                        line, r"^##\s+20\d\d-\d\d-\d\d",
                        f"{f.relative_to(ROOT)}: undated entry heading {line!r}",
                    )


if __name__ == "__main__":
    unittest.main()
