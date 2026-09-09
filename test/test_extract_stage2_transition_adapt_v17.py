import tempfile
import unittest
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np

from scripts.extract_stage2_transition_adapt_v17 import (
    interval_sample_indices,
    read_interval_frames,
    extract_interval_row,
    output_path,
)


class TransitionAdaptExtractionTests(unittest.TestCase):
    def test_direct_cli_imports(self):
        result = subprocess.run(
            ["venv/bin/python", "scripts/extract_stage2_transition_adapt_v17.py", "--help"],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_inclusive_interval_samples_at_30fps(self):
        indices, timestamps = interval_sample_indices(30, 59, 29.97, with_timestamps=True)
        self.assertEqual(indices[0], 30)
        self.assertEqual(indices[-1], 59)
        self.assertEqual(len(timestamps), 31)
        self.assertTrue(np.allclose(np.diff(timestamps), 1 / 30.0))

    def test_non_30fps_uses_exact_30fps_timestamp_grid(self):
        indices, timestamps = interval_sample_indices(0, 59, 60.0, with_timestamps=True)
        self.assertEqual(indices.tolist(), list(range(0, 60, 2)))
        self.assertEqual(timestamps.tolist(), [value / 30.0 for value in range(30)])

    def test_interval_read_preserves_inclusive_bounds_and_geometry(self):
        import cv2

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.avi"
            writer = cv2.VideoWriter(path.as_posix(), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (16, 8))
            for value in range(6):
                writer.write(np.full((8, 16, 3), value * 30, dtype=np.uint8))
            writer.release()
            frames, metadata = read_interval_frames(path, 2, 4)
        self.assertEqual(len(frames), 3)
        self.assertEqual(metadata["source_frame_indices"], [2, 3, 4])
        self.assertEqual(frames[0].shape[:2], (8, 16))
        self.assertEqual(metadata["geometry_transform"], "none")

    def test_extract_interval_row_passes_required_source_group(self):
        row = {"interval": [0, 4], "source_group": "stem:P0", "video_path": "missing.mp4"}
        with patch("scripts.extract_stage2_transition_adapt_v17.base.extract_row", return_value=({}, {"video_metadata": {}})) as call:
            extract_interval_row(row, "manifest")
        self.assertEqual(call.call_args.args[0]["source_group"], "stem:P0")

    def test_output_path_uses_shared_role_source_contract(self):
        row = {"role": "train", "source": "asl_stem_wiki_verified_interval", "source_item_id": "stem:P0:000"}
        self.assertEqual(output_path(Path("root"), row).as_posix(), "root/train/asl_stem_wiki_verified_interval/stem_P0_000.stage2_rgb_v17.npz")


if __name__ == "__main__":
    unittest.main()
