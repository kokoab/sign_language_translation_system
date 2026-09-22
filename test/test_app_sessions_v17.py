"""History indexing for the app.  Reads recorded sessions; writes only an index."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import app_sessions_v17 as store


def write_session(root: Path, stamp: str, **history) -> Path:
    folder = root / stamp
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "slt_live_isolated_v17_session",
        "started_utc": "2026-09-22T10:15:00+00:00",
        "finished_utc": "2026-09-22T10:15:42+00:00",
        "source": "camera:0", "mode": "cascade", "test_accessed": False,
        "predictions": [], "events": [], "utterances": [],
    }
    payload.update(history)
    (folder / "history.json").write_text(json.dumps(payload))
    return folder


def utterance(glosses, sentence, *, shown=True) -> dict:
    return {
        "glosses": glosses, "sentence": sentence, "displayed_and_spoken": shown,
    }


class SummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="app_sessions_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_a_finished_session_reports_its_sentence_and_length(self) -> None:
        folder = write_session(
            self.root, "20260922_101500_000000",
            utterances=[utterance(["I", "WANT", "EAT"], "I want to eat.")],
        )
        row = store.summarize(folder)
        self.assertEqual(row["glosses"], ["I", "WANT", "EAT"])
        self.assertEqual(row["sentences"], ["I want to eat."])
        self.assertEqual(row["duration_seconds"], 42.0)
        self.assertTrue(row["complete"])

    def test_an_unfinished_session_falls_back_to_committed_glosses(self) -> None:
        folder = write_session(
            self.root, "20260922_102000_000000", finished_utc=None,
            video_source_timestamps_seconds=[0.0, 9.5],
            predictions=[
                {"committed_gloss": "HELLO"}, {"committed_gloss": None},
                {"committed_gloss": "YOU"},
            ],
        )
        row = store.summarize(folder)
        self.assertEqual(row["glosses"], ["HELLO", "YOU"])
        self.assertEqual(row["duration_seconds"], 9.5)
        self.assertFalse(row["complete"])

    def test_unspoken_utterances_are_not_counted_as_said(self) -> None:
        folder = write_session(
            self.root, "20260922_103000_000000",
            utterances=[utterance(["A"], "Stale.", shown=False)],
            predictions=[{"committed_gloss": "A"}],
        )
        row = store.summarize(folder)
        self.assertEqual(row["sentences"], [])
        self.assertEqual(row["glosses"], ["A"])

    def test_a_corrupt_history_is_skipped_not_raised(self) -> None:
        folder = self.root / "20260922_104000_000000"
        folder.mkdir(parents=True)
        (folder / "history.json").write_text("{ not json")
        self.assertIsNone(store.summarize(folder))

    def test_video_is_reported_only_when_the_file_exists(self) -> None:
        folder = write_session(self.root, "20260922_105000_000000")
        self.assertIsNone(store.summarize(folder)["video"])
        (folder / "session_lowres.mp4").write_bytes(b"clip")
        self.assertIsNotNone(store.summarize(folder)["video"])


class IndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="app_index_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_sessions_are_listed_newest_first(self) -> None:
        for stamp in ("20260920_090000_000000", "20260922_090000_000000",
                      "20260921_090000_000000"):
            write_session(self.root, stamp)
        stamps = [row["stamp"] for row in store.build_index(self.root)["sessions"]]
        self.assertEqual(stamps, sorted(stamps, reverse=True))

    def test_refresh_writes_an_index_that_reloads(self) -> None:
        write_session(self.root, "20260922_090000_000000")
        written = store.refresh(self.root)
        self.assertEqual(written["session_count"], 1)
        self.assertEqual(store.load_index(self.root)["session_count"], 1)

    def test_unchanged_sessions_are_reused_rather_than_reparsed(self) -> None:
        folder = write_session(
            self.root, "20260922_090000_000000",
            utterances=[utterance(["I"], "Me.")],
        )
        first = store.refresh(self.root)
        # A cached row is returned verbatim; prove it by poisoning the cache.
        poisoned = json.loads(json.dumps(first))
        poisoned["sessions"][0]["sentences"] = ["FROM CACHE"]
        reused = store.build_index(self.root, poisoned)
        self.assertEqual(reused["sessions"][0]["sentences"], ["FROM CACHE"])
        # Touching the history invalidates it.
        (folder / "history.json").write_text(
            json.dumps({"utterances": [utterance(["YOU"], "You.")]})
        )
        fresh = store.build_index(self.root, poisoned)
        self.assertEqual(fresh["sessions"][0]["sentences"], ["You."])

    def test_a_missing_root_indexes_to_nothing(self) -> None:
        index = store.build_index(self.root / "not_here")
        self.assertEqual((index["session_count"], index["sessions"]), (0, []))

    def test_a_stray_file_in_the_root_is_ignored(self) -> None:
        write_session(self.root, "20260922_090000_000000")
        (self.root / "README.md").write_text("not a session")
        self.assertEqual(store.build_index(self.root)["session_count"], 1)


class DescribeTest(unittest.TestCase):
    def test_it_prefers_the_last_sentence(self) -> None:
        _, length, said = store.describe({
            "started_utc": "2026-09-22T10:15:00+00:00", "duration_seconds": 102.0,
            "sentences": ["First.", "Second."], "glosses": ["A"],
        })
        self.assertEqual((length, said), ("1:42", "Second."))

    def test_it_falls_back_to_glosses_then_to_a_plain_note(self) -> None:
        _, _, glossed = store.describe(
            {"duration_seconds": 5, "sentences": [], "glosses": ["I", "EAT"]}
        )
        self.assertEqual(glossed, "I EAT")
        _, _, empty = store.describe(
            {"stamp": "x", "duration_seconds": 0, "sentences": [], "glosses": []}
        )
        self.assertEqual(empty, "No recognized signs")


if __name__ == "__main__":
    unittest.main()
