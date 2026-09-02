import unittest

import numpy as np
import torch

from scripts.encode_stage2_hand_mobileclip2_v17 import preprocess_crop_batch


class EncodeStage2HandMobileCLIP2V17Tests(unittest.TestCase):
    def test_vectorized_fixed_crop_preprocessing(self):
        crops = np.zeros((2, 3, 256, 256, 3), dtype=np.uint8)
        crops[0, 1] = 255
        crops[1, 2, ..., 0] = 128
        locations = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
        value = preprocess_crop_batch(
            crops, locations, torch.device("cpu"), torch.float32
        )
        self.assertEqual(tuple(value.shape), (2, 3, 256, 256))
        self.assertEqual(float(value[0].min()), 1.0)
        self.assertEqual(float(value[0].max()), 1.0)
        self.assertAlmostEqual(float(value[1, 0, 0, 0]), 128 / 255, places=7)
        self.assertEqual(float(value[1, 1:].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
