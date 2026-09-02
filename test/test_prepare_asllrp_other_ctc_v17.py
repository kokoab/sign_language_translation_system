import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_asllrp_other_ctc_v17 import (
    OTHER,
    OTHER_INDEX,
    build_rows,
    chunk_events,
    collapse_other,
)
from scripts.finalize_asllrp_other_ctc_manifest_v17 import run as finalize_manifest


class PrepareAsllrpOtherCTCV17Test(unittest.TestCase):
    def test_adjacent_other_annotations_collapse(self):
        rows = collapse_other([
            {"label": OTHER, "index": OTHER_INDEX, "variant": "A", "start": 0, "end": 4},
            {"label": OTHER, "index": OTHER_INDEX, "variant": "B", "start": 5, "end": 9},
            {"label": "HELP", "index": 8, "variant": "HELP", "start": 10, "end": 14},
        ])
        self.assertEqual([row["label"] for row in rows], [OTHER, "HELP"])
        self.assertEqual(rows[0]["variants"], ["A", "B"])

    def test_chunker_never_splits_an_annotation(self):
        events = [
            {"label": "HELP", "index": 8, "variant": "HELP", "start": 0, "end": 20},
            {"label": OTHER, "index": OTHER_INDEX, "variant": "X", "start": 220, "end": 240},
            {"label": "YOU", "index": 1, "variant": "YOU", "start": 250, "end": 270},
        ]
        chunks = chunk_events(events, 256)
        self.assertEqual([[row["label"] for row in chunk] for chunk in chunks], [
            ["HELP", OTHER], ["YOU"],
        ])

    def test_exact_target_and_unknown_context_are_preserved(self):
        def row(start, variant, occurrence=None, sign_type="Lexical Signs"):
            return {
                "Entry/variant gloss label": variant,
                "Occurrence label": occurrence or variant,
                "Start frame of the sign video": str(start),
                "End frame of the sign video": str(start + 4),
                "Start frame of the containing utterance": "100",
                "End frame of the containing utterance": "199",
                "Utterance video filename": "u.mp4",
                "Source collection": "Cory_1",
                "Sign type": sign_type,
                "Hidden": "F",
            }

        targets = [{
            "canonical_label": "HELP", "class_index": 7,
            "signbank_annotation_id": "HELP",
        }]
        rows = build_rows([
            row(110, "WAVE", sign_type="Gestures"),
            row(120, "HELP", "HELP+"),
            row(130, "OTHER-LEXEME"),
        ], targets, 256, 5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target_sequence"], [OTHER, "HELP", OTHER])
        self.assertEqual(rows[0]["target_indices"], [OTHER_INDEX, 8, OTHER_INDEX])
        self.assertEqual(rows[0]["split_role"], "train")
        self.assertEqual(rows[0]["crop_start_frame_local"], 5)
        self.assertEqual(rows[0]["crop_end_frame_local"], 39)

    def test_finalizer_converts_ctc_ids_to_archive_class_ids_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "span.mp4"
            video.write_bytes(b"video")
            digest = hashlib.sha256(b"video").hexdigest()
            acquisition = root / "acquisition.json"
            acquisition.write_text(json.dumps({
                "failures": [],
                "verified_spans": 1,
                "expected_spans": 1,
                "spans": [{
                    "target_sequence": ["I", OTHER],
                    "target_indices": [1, OTHER_INDEX],
                    "path": video.as_posix(),
                    "sha256": digest,
                    "split_role": "train",
                    "utterance_video_filename": "utterance.mp4",
                    "span_index_in_utterance": 0,
                    "signer_id": "CORY",
                    "supported_target_token_count": 1,
                    "other_token_count": 1,
                    "frames": 32,
                    "duration_seconds": 1.0,
                    "parent_sha256": "parent",
                }],
            }))
            output = root / "manifest.json"
            manifest = finalize_manifest(argparse.Namespace(
                acquisition_manifest=acquisition, output=output,
            ))
            self.assertEqual(manifest["rows"][0]["target_indices"], [0, 100])
            self.assertEqual(manifest["other_class_index"], 100)
            self.assertEqual(manifest["other_ctc_index"], 101)


if __name__ == "__main__":
    unittest.main()
