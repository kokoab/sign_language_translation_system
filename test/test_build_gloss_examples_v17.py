"""Selection rules for the app's gloss examples.  Scoring only, no media written."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import build_gloss_examples_v17 as builder


def sidecar(
    path: Path, video: Path, *, hand=0.9, frames=0.9, face=0.9, finite=True,
    start=4, end=40, total=48, fps=30.0,
) -> None:
    metadata = {
        "video_path": str(video.relative_to(REPO)) if REPO in video.parents
        else str(video),
        "fps": fps, "hand_trim_start_frame": start,
        "hand_trim_end_frame_exclusive": end, "source_frames_before_hand_trim": total,
    }
    diagnostics = {
        "finite": finite, "hand_presence_fraction": hand,
        "observed_hand_frame_fraction": frames, "face_presence_fraction": face,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, features=np.zeros((2, 2), np.float16),
        metadata_json=np.array(json.dumps(metadata)),
        diagnostics_json=np.array(json.dumps(diagnostics)),
    )


class DurationFitTest(unittest.TestCase):
    def test_the_ideal_band_scores_full_marks(self) -> None:
        low, high = builder.IDEAL_SECONDS
        for seconds in (low, (low + high) / 2, high):
            self.assertEqual(builder.duration_fit(seconds), 1.0)

    def test_it_decays_outside_the_band_and_floors_at_zero(self) -> None:
        low, high = builder.IDEAL_SECONDS
        self.assertLess(builder.duration_fit(low - 0.5), 1.0)
        self.assertLess(builder.duration_fit(high + 0.5), 1.0)
        self.assertEqual(builder.duration_fit(high + 99), 0.0)
        self.assertEqual(builder.duration_fit(0.0), 0.0)


class ScoringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(
            __import__("tempfile").mkdtemp(prefix="gloss_examples_")
        )
        self.video = self.tmp / "clip-ANGRY.mp4"
        self.video.write_bytes(b"not decoded during scoring")

    def tearDown(self) -> None:
        __import__("shutil").rmtree(self.tmp, ignore_errors=True)

    def read(self, **overrides):
        path = self.tmp / "clip-ANGRY.v17.npz"
        sidecar(path, self.video, **overrides)
        return builder.read_candidate(path, "ANGRY", {})

    def test_a_clean_clip_scores_and_reports_no_notes(self) -> None:
        candidate = self.read()
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.notes, [])
        self.assertGreater(candidate.score, 0.8)
        self.assertAlmostEqual(candidate.seconds, 36 / 30.0, places=3)

    def test_a_non_finite_extraction_is_refused(self) -> None:
        self.assertIsNone(self.read(finite=False))

    def test_activity_touching_either_edge_is_flagged_and_penalised(self) -> None:
        clean = self.read()
        start = self.read(start=0)
        end = self.read(end=48, total=48)
        both = self.read(start=0, end=48, total=48)
        self.assertEqual(start.components["edges"], 0.5)
        self.assertEqual(both.components["edges"], 0.0)
        self.assertLess(start.score, clean.score)
        self.assertLess(both.score, start.score)
        self.assertTrue(start.notes and end.notes)

    def test_better_tracking_outranks_worse(self) -> None:
        self.assertGreater(self.read(hand=0.95).score, self.read(hand=0.30).score)

    def test_provenance_supplies_participant_and_hash(self) -> None:
        path = self.tmp / "clip-ANGRY.v17.npz"
        sidecar(path, self.video)
        rows = {self.video.name: {"participant": "P30", "sha256": "abc123"}}
        candidate = builder.read_candidate(path, "ANGRY", rows)
        self.assertEqual((candidate.participant, candidate.sha256), ("P30", "abc123"))

    def test_a_missing_video_is_refused(self) -> None:
        self.video.unlink()
        self.assertIsNone(self.read())


class SplitGuardTest(unittest.TestCase):
    """Validation is reserved and the test gate was consumed once; neither is read."""

    def test_the_generator_only_ever_points_at_train(self) -> None:
        self.assertTrue(str(builder.RAW_TRAIN).endswith("/raw/train"))
        self.assertTrue(str(builder.LANDMARKS_TRAIN).endswith("/landmarks/train"))
        source = Path(builder.__file__).read_text()
        for forbidden in ("raw/val", "raw/test", "landmarks/val", "landmarks/test"):
            self.assertNotIn(forbidden, source)


class CatalogTest(unittest.TestCase):
    """The shipped catalogue is the gallery's contract; check what it promises."""

    @classmethod
    def setUpClass(cls) -> None:
        path = builder.DEFAULT_OUTPUT / "catalog.json"
        if not path.is_file():
            raise unittest.SkipTest("run scripts/build_gloss_examples_v17.py first")
        cls.catalog = json.loads(path.read_text())

    def test_it_covers_the_locked_hundred_from_train_only(self) -> None:
        self.assertEqual(self.catalog["classes_covered"], 100)
        self.assertEqual(self.catalog["classes_missing"], [])
        self.assertEqual(self.catalog["source_split"], "train")
        self.assertFalse(self.catalog["val_accessed"])
        self.assertFalse(self.catalog["test_accessed"])

    def test_every_example_is_traceable_and_present(self) -> None:
        for label, row in self.catalog["examples"].items():
            self.assertIn(row["source"], ("local", "citizen"), label)
            if row["source"] == "citizen":
                # Corpus clips keep their hash and must come from train only.
                self.assertTrue(row["source_sha256"], label)
                self.assertIn("/raw/train/", row["source_video"], label)
            else:
                self.assertTrue(
                    row["source_video"].startswith("data/raw_videos/"), label
                )
            for key in ("poster", "loop"):
                self.assertTrue(
                    (builder.DEFAULT_OUTPUT / row["assets"][key]).is_file(), label
                )

    def test_local_recordings_are_preferred_over_the_corpus(self) -> None:
        rows = self.catalog["examples"].values()
        local = sum(1 for row in rows if row["source"] == "local")
        self.assertEqual(local, self.catalog["classes_from_local"])
        self.assertGreater(local, self.catalog["classes_from_citizen"])

    def test_the_local_pool_is_the_exact_variant_one(self) -> None:
        self.assertIn("cap14_exact", self.catalog["local_pool"])

    def test_no_rejected_clip_was_selected(self) -> None:
        skipped = set(self.catalog["rejected_videos_skipped"])
        chosen = {
            Path(row["source_video"]).name
            for row in self.catalog["examples"].values()
        }
        self.assertEqual(chosen & skipped, set())


if __name__ == "__main__":
    unittest.main()
