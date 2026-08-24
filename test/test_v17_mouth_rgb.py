from __future__ import annotations

import unittest

import numpy as np

from active.v17.extract_mouth_rgb_v17 import (
    crop_square,
    decode_packed_crops,
    mouth_square_box,
    pack_crops,
    selected_frame_indices,
)
from active.v17.schema_mouth_rgb_v17 import CROP_SIZE, MouthRGBV17Config
from active.v17.model_mouth_rgb_v17 import MouthRGBStage1, MouthRGBStage1Config
from active.v17.train_stage_1_mouth_rgb_v17 import augment_mouth


class MouthRGBV17Test(unittest.TestCase):
    def test_selected_indices_stay_inside_frozen_interval(self):
        indices = selected_frame_indices(7, 48)
        self.assertEqual(len(indices), 16)
        self.assertEqual(int(indices[0]), 7)
        self.assertEqual(int(indices[-1]), 47)
        self.assertTrue(np.all(indices[1:] >= indices[:-1]))

    def test_face_points_produce_centered_square_crop(self):
        xy = np.zeros((15, 2), dtype=np.float32)
        confidence = np.zeros(15, dtype=np.float32)
        xy[7:11] = ((0.45, 0.55), (0.55, 0.55), (0.50, 0.53), (0.50, 0.57))
        xy[11:14] = ((0.35, 0.60), (0.50, 0.68), (0.65, 0.60))
        confidence[7:14] = 1.0
        box = mouth_square_box(xy, confidence, 640, 480, MouthRGBV17Config())
        self.assertIsNotNone(box)
        self.assertAlmostEqual(float((box[0] + box[2]) / 2), 320.0, places=4)
        self.assertAlmostEqual(float((box[1] + box[3]) / 2), 264.0, places=3)
        crop = crop_square(np.full((480, 640, 3), 127, np.uint8), box)
        self.assertEqual(crop.shape, (CROP_SIZE, CROP_SIZE, 3))

    def test_packed_missing_frame_decodes_to_exact_zero(self):
        crops = [np.full((CROP_SIZE, CROP_SIZE, 3), 127, np.uint8), None]
        blob, offsets = pack_crops(crops, 90)
        decoded = decode_packed_crops(blob, offsets)
        self.assertEqual(decoded.shape, (2, CROP_SIZE, CROP_SIZE, 3))
        self.assertGreater(int(decoded[0].sum()), 0)
        self.assertEqual(int(decoded[1].sum()), 0)

    def test_small_visual_speech_model_forward_and_masked_augmentation(self):
        model = MouthRGBStage1(MouthRGBStage1Config(num_classes=7)).eval()
        pixels = np.zeros((2, 16, 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        valid = np.ones((2, 16), dtype=np.bool_)
        valid[0, -2:] = False
        import torch
        pixel_tensor = torch.from_numpy(pixels)
        valid_tensor = torch.from_numpy(valid)
        augmented, augmented_valid = augment_mouth(pixel_tensor, valid_tensor)
        self.assertEqual(float(augmented[~augmented_valid].abs().max()), 0.0)
        with torch.no_grad():
            logits = model(augmented, augmented_valid)
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertLess(model.parameter_count, 1_000_000)


if __name__ == "__main__":
    unittest.main()
