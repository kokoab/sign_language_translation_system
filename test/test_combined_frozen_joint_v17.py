"""Focused contracts for the paired frozen/adapted joint runner."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from active.v17.joint_ctc_v17 import Chunks
from scripts.train_combined_frozen_joint_v17 import (
    cached_chunks, collapse_path, configure_arm, make_records, paired_plan, weighted_ctc_loss,
)


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def encode(self, x):
        encoded = x[..., 0, 0].unsqueeze(-1) * self.weight
        return encoded, None


class CombinedFrozenJointTest(unittest.TestCase):
    def test_shared_signer_ids_do_not_overwrite_examples(self):
        rows = [dict(feature_path=path, source_item_id="same_signer", source="o5s5",
                     role="train", representation="timestamp_normalized_positive_core",
                     supervision="single_verified_sign") for path in ("a.npz", "b.npz")]
        with patch("scripts.train_combined_frozen_joint_v17.load_features",
                   return_value=(np.zeros((1, 32, 61, 5), np.float16), [1])):
            records = make_records({"records": rows}, {"HELLO": 0})["train"]
        self.assertEqual([r["id"] for r in records], ["a.npz", "b.npz"])
        self.assertTrue(all(r["features"].dtype == np.float32 for r in records))
        with self.assertRaisesRegex(ValueError, "unique"):
            paired_plan(["a", "a"], ["phrase"], 7, 1)

    def test_cached_chunks_preserve_windows_and_record_lengths(self):
        features = np.stack([
            np.full((32, 61, 5), 1, np.float32),
            np.full((32, 61, 5), 2, np.float32),
        ])
        value = cached_chunks(features)
        self.assertIsInstance(value, Chunks)
        self.assertEqual(value.length, 64)
        self.assertTrue(all(keep.all() for keep in value.keep))
        np.testing.assert_array_equal(value.features, features)
        plan = paired_plan(["s2", "s1", "s3"], ["p2", "p1"], 7, 2)
        self.assertEqual(sorted(plan["singles"]), ["s1", "s2", "s3"])
        self.assertEqual(len(plan["phrases"]), len(plan["singles"]))
        self.assertEqual(plan, paired_plan(["s2", "s1", "s3"], ["p2", "p1"], 7, 2))

    def test_freeze_vs_adapted_gradients_and_repeated_ctc(self):
        logits = torch.zeros(1, 3, 102, requires_grad=True)
        loss = weighted_ctc_loss(logits, [(1, 1)], [3], [1.0])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(collapse_path([1, 101, 101, 1, 1, 0]), [1, 101, 1])
        for arm, expected in (("frozen", False), ("adapted", True)):
            base = TinyBase()
            configure_arm(base, arm)
            output = base.encode(torch.ones(1, 32, 61, 5))[0].sum()
            if expected:
                output.backward()
                self.assertTrue(torch.isfinite(base.weight.grad))
            else:
                self.assertFalse(base.weight.requires_grad)
                self.assertFalse(base.training)

    def test_weighted_ctc_is_target_normalized_before_weighting(self):
        logits = torch.zeros(2, 3, 102)
        logits[0, :, 1] = 3
        logits[1, :, 2] = 3
        actual = weighted_ctc_loss(logits, [(1,), (2, 2)], [3, 3], [1., .5])
        raw = torch.nn.functional.ctc_loss(logits.log_softmax(-1).transpose(0, 1),
            torch.tensor([1, 2, 2]), torch.tensor([3, 3]), torch.tensor([1, 2]), blank=0, reduction="none")
        expected = (raw[0] + raw[1] / 2 * .5) / 1.5
        self.assertTrue(torch.allclose(actual.cpu(), expected))


if __name__ == "__main__":
    unittest.main()
