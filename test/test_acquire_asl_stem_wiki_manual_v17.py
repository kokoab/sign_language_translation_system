from pathlib import Path
import struct
import tempfile
import unittest

from scripts.acquire_asl_stem_wiki_manual_v17 import build_candidates, parse_central_directory


class AcquireAslStemWikiManualV17Test(unittest.TestCase):
    def test_build_candidates_rejects_unbalanced_gloss_annotation(self):
        rows = [{"filename": "bad.mp4", "GLOSSED SENTENCE": "CL:C(segment) ) WATER"}]
        metadata = {"bad.mp4": {"participant": "P1", "video length": "1"}}
        candidates, rejected = build_candidates(rows, metadata, {"WATER": "WATER"}, {"P1"})
        self.assertEqual(candidates, [])
        self.assertEqual(rejected[0]["reason"], "unbalanced gloss annotation")

    def test_build_candidates_keeps_provisional_pool_in_candidate_verification(self):
        manual_rows = [
            {
                "user": "reviewed-user",
                "filename": "good.mp4",
                "GLOSSED SENTENCE": "WATER fs-TERM CL:1(topic) GIVE",
            },
            {
                "user": "reviewed-user",
                "filename": "empty.mp4",
                "GLOSSED SENTENCE": "",
            },
            {
                "user": "other-user",
                "filename": "other.mp4",
                "GLOSSED SENTENCE": "WATER GIVE",
            },
        ]
        metadata = {
            "good.mp4": {"participant": "P12", "video length": "2.0"},
            "empty.mp4": {"participant": "P12", "video length": "2.0"},
            "other.mp4": {"participant": "P4", "video length": "2.0"},
        }
        raw_to_canonical = {"WATER": "WATER", "GIVE": "GIVE"}

        candidates, rejected = build_candidates(
            manual_rows, metadata, raw_to_canonical, {"P12"}
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["participant"], "P12")
        self.assertEqual(candidates[0]["role"], "candidate_verification")
        self.assertFalse(candidates[0]["training_eligible"])
        self.assertFalse(candidates[0]["signer_quality_verified"])
        self.assertFalse(candidates[0]["variant_verified"])
        self.assertEqual(candidates[0]["gloss_tokens"], ["WATER", "fs-TERM", "CL:1(topic)", "GIVE"])
        self.assertEqual(candidates[0]["ctc_sequence"], ["WATER", "OTHER", "OTHER", "GIVE"])
        self.assertEqual(
            {row["reason"] for row in rejected},
            {"empty gloss sequence", "participant outside provisional download pool"},
        )

    def test_build_candidates_rejects_rows_absent_from_source_metadata(self):
        with self.assertRaisesRegex(ValueError, "missing.mp4"):
            build_candidates(
                [{"user": "reviewed-user", "filename": "missing.mp4", "GLOSSED SENTENCE": "WATER"}],
                {},
                {"WATER": "WATER"},
                {"P12"},
            )

    def test_parse_central_directory_preserves_member_crc32(self):
        name = b"videos/one.mp4"
        record = struct.pack(
            "<4s6H3L5H2L",
            b"PK\x01\x02",
            20,
            20,
            0,
            8,
            0,
            0,
            0x78D5C3C6,
            11,
            12,
            len(name),
            0,
            0,
            0,
            0,
            0,
            100,
        ) + name
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "central.bin"
            path.write_bytes(record)
            parsed = parse_central_directory(path)
        self.assertEqual(parsed["videos/one.mp4"]["crc32"], 0x78D5C3C6)


if __name__ == "__main__":
    unittest.main()
