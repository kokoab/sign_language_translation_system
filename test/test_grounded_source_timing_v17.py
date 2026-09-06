import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scripts.generate_grounded_text_to_sign_v17 import load_isolated


class GroundedTimingTest(unittest.TestCase):
    def test_recognition_accepts_real_duration_without_changing_animation(self):
        from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
        from scripts.generate_grounded_text_to_sign_v17 import recognize
        model = SLTStage1V17(Stage1V17Config(num_classes=2, dim=32, depth=1, heads=4)).eval()
        features = np.ones((60, 61, 5), np.float32)
        label, score = recognize(model, {0: "A", 1: "B"}, features)
        self.assertIn(label, ("A", "B"))
        self.assertTrue(0 <= score <= 1)
        self.assertEqual(features.shape[0], 60)

    def test_normalized_archives_restore_seconds_from_source_fps(self):
        features = np.ones((32, 61, 5), np.float32)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "source.npz"
            for processed, fps, expected in ((30, 30., 15), (30, 15., 30)):
                np.savez(path, features=features, metadata_json=json.dumps(dict(
                    source_frames_processed=processed, fps=fps, decoded_frame_count=60,
                    sampled_frame_count=60)))
                self.assertEqual(len(load_isolated(path)), expected)
