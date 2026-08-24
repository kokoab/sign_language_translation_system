import io
import json
from pathlib import Path
import unittest
import zipfile

from scripts.download_citizen100_v17 import decode_member
from scripts.build_citizen100_v17 import build_manifest


class Citizen100V17DownloaderTest(unittest.TestCase):
    def test_raw_deflate_decode_checks_size_and_crc(self):
        buffer = io.BytesIO()
        expected = b"offline v17 citizen data" * 100
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("video.mp4", expected)
        with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
            info = archive.getinfo("video.mp4")
            offset = info.header_offset
            raw = buffer.getvalue()
            filename_length = int.from_bytes(raw[offset + 26:offset + 28], "little")
            extra_length = int.from_bytes(raw[offset + 28:offset + 30], "little")
            start = offset + 30 + filename_length + extra_length
            compressed = raw[start:start + info.compress_size]
        self.assertEqual(decode_member(info, compressed), expected)

    def test_decode_rejects_corrupted_payload(self):
        info = zipfile.ZipInfo("video.mp4")
        info.compress_type = zipfile.ZIP_STORED
        info.file_size = 3
        info.compress_size = 3
        info.CRC = 0
        with self.assertRaises(RuntimeError):
            decode_member(info, b"bad")

    def test_frozen_manifest_is_reproducible_and_variant_unique(self):
        seed_path = Path("active/v17/citizen100_seed.json")
        manifest_path = Path("active/v17/citizen100_manifest.json")
        cache_dir = Path("data/local/dataset_metadata")
        if not all(
            (cache_dir / f"asl_citizen_{split}.csv").exists()
            for split in ("train", "val", "test")
        ):
            self.skipTest("cached official ASL Citizen split metadata unavailable")
        seed = json.loads(seed_path.read_text())
        frozen = json.loads(manifest_path.read_text())
        rebuilt = build_manifest(seed, cache_dir)
        self.assertEqual(len(frozen["classes"]), 100)
        self.assertEqual(rebuilt["classes"], frozen["classes"])
        self.assertEqual(rebuilt["source_csv_sha256"], frozen["source_csv_sha256"])
        self.assertEqual(
            len(
                {
                    (item["citizen_raw_gloss"], item["citizen_asl_lex_code"])
                    for item in frozen["classes"]
                }
            ),
            100,
        )


if __name__ == "__main__":
    unittest.main()
