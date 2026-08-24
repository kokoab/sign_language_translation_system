import tempfile
import unittest
from pathlib import Path

import numpy as np

from active.v17.movinet_data_v17 import (
    FRAMES,
    VIEWS,
    load_aligned_records,
    mirror_sign_views,
)


class MoViNetDataV17Test(unittest.TestCase):
    def test_mirror_is_an_involution_and_swaps_anatomical_hands(self):
        pixels = np.zeros((FRAMES, VIEWS, 8, 8, 3), dtype=np.uint8)
        pixels[:, 0, :, :4] = 50
        pixels[:, 1, :, 4:] = 200
        valid = np.ones((FRAMES, VIEWS), dtype=np.bool_)
        valid[0, 0] = False
        pixels[0, 0] = 0
        boxes = np.tile(
            np.asarray([[[0.1, 0.2, 0.3, 0.4], [0.6, 0.2, 0.8, 0.4], [0.1, 0.1, 0.8, 0.8]]]),
            (FRAMES, 1, 1),
        ).astype(np.float32)
        boxes[~valid] = 0

        mirrored = mirror_sign_views(pixels, valid, boxes)
        restored = mirror_sign_views(*mirrored)
        np.testing.assert_array_equal(restored[0], pixels)
        np.testing.assert_array_equal(restored[1], valid)
        np.testing.assert_allclose(restored[2], boxes, atol=1e-7)
        self.assertFalse(bool(mirrored[1][0, 1]))
        self.assertTrue((mirrored[0][0, 1] == 0).all())

    def test_alignment_uses_item_ids_and_rejects_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            crop = root / "crops" / "train" / "HELLO" / "clip.hand_rgb_v17.npz"
            crop.parent.mkdir(parents=True)
            crop.touch()
            cache = root / "landmark_train.npz"
            np.savez_compressed(
                cache,
                features=np.zeros((1, 256), dtype=np.float32),
                logits=np.zeros((1, 100), dtype=np.float32),
                targets=np.asarray([7], dtype=np.int64),
                item_ids=np.asarray(["HELLO/clip"]),
                mode=np.asarray("landmark"),
                split=np.asarray("train"),
            )
            records = load_aligned_records(root / "crops", cache, "train")
            self.assertEqual(records[0].target, 7)
            self.assertEqual(records[0].crop_path, crop)
            with self.assertRaises(ValueError):
                load_aligned_records(root / "crops", cache, "test")


if __name__ == "__main__":
    unittest.main()
