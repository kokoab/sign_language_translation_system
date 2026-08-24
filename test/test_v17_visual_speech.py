from __future__ import annotations

import unittest

import numpy as np
import torch

from active.v17.extract_visual_speech_v17 import (
    aligned_views,
    decode_packed_crops,
    motion_interval,
    pack_crops,
)
from active.v17.schema_visual_speech_v17 import (
    CROP_SIZE,
    SEQUENCE_LENGTH,
    VIEW_NAMES,
    VisualSpeechV17Config,
)
from active.v17.model_visual_speech_v17 import (
    MultiViewVisualSpeechHeadV17,
    MultiViewVisualSpeechV17Config,
    VisualSpeechTeacherV17,
    VisualSpeechTeacherV17Config,
)
from active.v17.extract_visual_speech_features_semlex_v17 import (
    SemLexVisualSpeechDataset,
)


class VisualSpeechV17Test(unittest.TestCase):
    def test_semlex_visual_dataset_rejects_unknown_view_before_io(self):
        with self.assertRaisesRegex(ValueError, "unknown visual-speech view"):
            SemLexVisualSpeechDataset(None, None, None, "eyes")

    def test_multiview_head_forward_is_finite_and_missing_aware(self):
        model = MultiViewVisualSpeechHeadV17(
            MultiViewVisualSpeechV17Config(
                num_classes=7, dim=64, view_dim=32, depth=1, heads=4
            )
        ).eval()
        features = torch.randn(2, 4, 2, 512)
        valid = torch.tensor(
            [
                [[True, True], [True, False], [False, True], [True, True]],
                [[False, False], [False, False], [False, False], [False, False]],
            ]
        )
        features *= valid.unsqueeze(-1)
        with torch.inference_mode():
            logits, embeddings = model(features, valid, return_embeddings=True)
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertEqual(tuple(embeddings.shape), (2, 64))
        self.assertTrue(torch.isfinite(logits).all())

    def test_visual_speech_teacher_forward_is_finite_and_missing_aware(self):
        model = VisualSpeechTeacherV17(
            VisualSpeechTeacherV17Config(
                num_classes=7, dim=64, depth=1, heads=4
            )
        ).eval()
        pixels = torch.randn(1, 4, 1, 88, 88)
        valid = torch.tensor([[True, True, False, True]])
        pixels[:, 2] = 0
        with torch.inference_mode():
            logits = model(pixels, valid)
        self.assertEqual(tuple(logits.shape), (1, 7))
        self.assertTrue(torch.isfinite(logits).all())

    def test_motion_interval_uses_full_utterance_when_signal_is_absent(self):
        config = VisualSpeechV17Config()
        shapes = np.zeros((64, 2), np.float32)
        valid = np.ones(64, np.bool_)
        start, end, diagnostics = motion_interval(shapes, valid, config)
        self.assertEqual((start, end), (0, 64))
        self.assertEqual(diagnostics["mode"], "full_utterance_fallback")

    def test_motion_interval_keeps_context_and_minimum_duration(self):
        config = VisualSpeechV17Config()
        shapes = np.zeros((80, 2), np.float32)
        shapes[36:44, 0] = np.asarray((0, 1, 0, 1, 0, 1, 0, 1), np.float32)
        valid = np.ones(80, np.bool_)
        start, end, diagnostics = motion_interval(shapes, valid, config)
        self.assertEqual(diagnostics["mode"], "mouth_motion")
        self.assertLessEqual(start, 36)
        self.assertGreaterEqual(end, 44)
        self.assertGreaterEqual(end - start, int(np.ceil(80 * config.minimum_interval_fraction)))

    def test_eye_aligned_three_view_crop_and_pack_round_trip(self):
        config = VisualSpeechV17Config()
        frame = np.zeros((240, 320, 3), np.uint8)
        frame[..., 1] = np.arange(320, dtype=np.uint8)[None, :]
        points = np.zeros((15, 2), np.float32)
        points[0] = (0.38, 0.36)
        points[1] = (0.62, 0.40)
        points[7:11] = ((0.43, 0.57), (0.57, 0.59), (0.50, 0.56), (0.50, 0.62))
        points[11:14] = ((0.34, 0.67), (0.50, 0.78), (0.66, 0.69))
        points[2:7] = ((0.35, 0.30), (0.45, 0.29), (0.55, 0.31), (0.65, 0.32), (0.50, 0.48))
        points[14] = (0.50, 0.35)
        confidence = np.ones(15, np.float32)
        crops, boxes, shape = aligned_views(frame, points, confidence, config)
        self.assertEqual(len(crops), len(VIEW_NAMES))
        self.assertEqual(tuple(boxes.shape), (len(VIEW_NAMES), 4))
        self.assertIsNotNone(shape)
        self.assertTrue(all(crop is not None for crop in crops))
        blob, offsets = pack_crops([crops] * SEQUENCE_LENGTH, config.jpeg_quality)
        decoded = decode_packed_crops(blob, offsets)
        self.assertEqual(
            tuple(decoded.shape),
            (SEQUENCE_LENGTH, len(VIEW_NAMES), CROP_SIZE, CROP_SIZE, 3),
        )
        self.assertGreater(int(decoded.sum()), 0)


if __name__ == "__main__":
    unittest.main()
