import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

from pathlib import Path

from scripts.download_msasl_citizen100_gap_candidates import (
    acquire_class,
    build_candidates,
    materialize_retained,
    video_id,
)


class MSASLCitizen100GapCandidateTest(unittest.TestCase):
    def test_video_id_is_youtube_only(self):
        self.assertEqual(video_id("https://www.youtube.com/watch?v=abc123&t=2s"), "abc123")
        self.assertEqual(video_id("https://example.com/watch?v=abc123"), "")

    def test_selection_is_exact_train_metadata_and_signer_unique(self):
        manifest = {"classes": [{
            "canonical_label": "HELP",
            "citizen_raw_gloss": "HELP",
            "citizen_asl_lex_code": "D_01_042",
        }]}
        base = {
            "text": "help", "label": 1, "url": "https://www.youtube.com/watch?v=abc",
            "start_time": 0.0, "end_time": 2.0, "width": 1280, "height": 720,
            "fps": 30, "box": [0, 0, 1, 1],
        }
        rows = [{**base, "signer_id": 1}, {**base, "signer_id": 1}, {**base, "signer_id": 2}]
        output = build_candidates(rows, manifest, set(), 8)
        self.assertEqual([row["msasl_signer_id"] for row in output], ["1", "2"])
        self.assertTrue(all(row["training_eligible"] is False for row in output))

    def test_covered_or_variant_only_class_is_excluded(self):
        manifest = {"classes": [{
            "canonical_label": "WHAT",
            "citizen_raw_gloss": "WHAT1",
            "citizen_asl_lex_code": "X",
        }]}
        row = {
            "text": "what", "label": 1, "signer_id": 1,
            "url": "https://www.youtube.com/watch?v=abc", "start_time": 0.0,
            "end_time": 2.0, "width": 1280, "height": 720, "fps": 30,
            "box": [0, 0, 1, 1],
        }
        self.assertEqual(build_candidates([row], manifest, set(), 8), [])

    def test_class_acquisition_stops_at_target(self):
        rows = [{"id": index} for index in range(8)]
        side_effect = [
            {"status": "failed"},
            {"status": "downloaded"},
            {"status": "downloaded"},
            {"status": "downloaded"},
        ]
        with patch(
            "scripts.download_msasl_citizen100_gap_candidates.acquire_one",
            side_effect=side_effect,
        ) as mocked:
            output = acquire_class(rows, Path("out"), Path("python"), 3)
        self.assertEqual(len(output), 4)
        self.assertEqual(mocked.call_count, 4)

    def test_materialize_retained_links_only_declared_rows(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"video")
            output = root / "retained"
            materialize_retained(
                [{"destination": str(source), "canonical_label": "HELP"}], output
            )
            link = output / "HELP" / "source.mp4"
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), source.resolve())


if __name__ == "__main__":
    unittest.main()
