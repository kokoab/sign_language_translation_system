import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.download_popsign_v17 import (
    participant_from_filename,
    safe_video_members,
    select_balanced_members,
    source_records,
)


class PopSignV17DownloaderTest(unittest.TestCase):
    def test_participant_parser(self):
        name = "gtsignstudy4a.8031-thankyou-2023_01_24_20_27_17.907-0.mp4"
        self.assertEqual(participant_from_filename(name, "thankyou"), "gtsignstudy4a.8031")

    def test_source_records_and_balanced_selection(self):
        source_map = {
            "test": {
                "prev_name": {"0": "0-thankyou.mp4", "1": "1-thankyou.mp4", "2": "2-thankyou.mp4"},
                "orig_name": {
                    "0": "personA-thankyou-2023_01_01_00_00_00-0.mp4",
                    "1": "personA-thankyou-2023_01_01_00_00_01-0.mp4",
                    "2": "personB-thankyou-2023_01_01_00_00_02-0.mp4",
                },
            }
        }
        records = source_records(source_map, "test", "thankyou")
        members = [tarfile.TarInfo(f"nested/{name}") for name in records]
        selected = select_balanced_members(members, records, 1)
        self.assertEqual([item[1]["participant"] for item in selected], ["personA", "personB"])

    def test_tar_filter_accepts_only_regular_unique_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.tar"
            with tarfile.open(path, "w") as archive:
                payload = b"video"
                info = tarfile.TarInfo("safe/0-thankyou.mp4")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
                text = tarfile.TarInfo("safe/readme.txt")
                text.size = 1
                archive.addfile(text, io.BytesIO(b"x"))
                link = tarfile.TarInfo("unsafe/link.mp4")
                link.type = tarfile.SYMTYPE
                link.linkname = "../../outside"
                archive.addfile(link)
            with tarfile.open(path) as archive:
                videos = safe_video_members(archive)
            self.assertEqual([Path(item.name).name for item in videos], ["0-thankyou.mp4"])


if __name__ == "__main__":
    unittest.main()
