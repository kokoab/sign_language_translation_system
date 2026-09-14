"""Critical alignment, ownership, and gradient contracts for the bounded experiment."""
import unittest
import numpy as np
import torch

from active.v17.joint_ctc_v17 import chunks, sequence_targets, ctc_loss, JointCTC, core_crop, Chunks, decode
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config


class JointCTCTest(unittest.TestCase):
    def test_single_observation_core_preserves_pose_without_inventing_motion(self):
        raw = np.zeros((3, 61, 5), np.float32)
        raw[..., 4] = 1
        raw[..., 0] = np.linspace(-.3, .3, 61)
        raw[..., 1] = np.linspace(.1, .9, 61)
        value, count = core_crop(raw, np.array([0., .1, .2]),
                                 {'start_seconds': .09, 'end_seconds': .11})
        self.assertEqual(count, 1)
        self.assertEqual(value.shape, (32, 61, 5))
        np.testing.assert_array_equal(value, np.repeat(value[:1], 32, axis=0))
        with self.assertRaises(ValueError):
            core_crop(raw, np.array([0., .1, .2]),
                      {'start_seconds': .02, 'end_seconds': .03})

    def test_timestamp_ownership_and_future_independence(self):
        raw = np.zeros((41, 61, 5), np.float32)
        raw[..., 4] = 1
        raw[..., 0] = np.linspace(-.3, .3, 61)
        raw[..., 1] = np.linspace(.1, .9, 61)
        times = np.arange(41) / 30
        value = chunks(raw, times)
        token_times = np.concatenate([t[k] for t, k in zip(value.times, value.keep)])
        self.assertTrue(np.all(np.diff(token_times) > 0))
        self.assertAlmostEqual(token_times[-1], times[-1])
        self.assertTrue(all(np.all(t[k] <= end + 1e-9)
                            for t, k, end in zip(value.times, value.keep, value.ends)))
        raw[times > value.ends[0], :, 0] += .2
        changed = chunks(raw, times)
        np.testing.assert_array_equal(value.features[0], changed.features[0])

    def test_irregular_observation_chunks_respect_physical_duration_bound(self):
        raw = np.zeros((30,61,5),np.float32)
        raw[...,4] = 1
        raw[...,0] = np.linspace(-.3,.3,61)
        raw[...,1] = np.linspace(.1,.9,61)
        times = np.arange(30)*.1
        value = chunks(raw,times)
        self.assertTrue(all(t[-1]-t[0] <= .53+1e-9 for t in value.times))

    def test_complete_targets_preserve_repeats_and_other(self):
        row = {'all_signs_annotated': True, 'intervals': [
            {'label': 'A'}, {'label': 'A'}, {'label': '__OTHER__'}]}
        self.assertEqual(sequence_targets(row, {'A': 0}), (1, 1, 101))
        row['all_signs_annotated'] = False
        with self.assertRaises(ValueError):
            sequence_targets(row, {'A': 0})
        with self.assertRaises(ValueError):
            ctc_loss(torch.randn(1, 2, 102), [(1, 1)], [2])
        self.assertEqual(decode([1, 1, 101, 1, 0, 1]), [1, 1, 1])

    def test_full_sequence_agrees_with_available_prefix(self):
        torch.manual_seed(2)
        model = JointCTC(SLTStage1V17(Stage1V17Config(dim=32, depth=1, heads=4))).eval()
        x = np.random.default_rng(2).normal(size=(2,32,61,5)).astype(np.float32)
        x[...,3:] = 1
        times = [np.linspace(0,.5,32),np.linspace(.5,1.,32)]
        keep = [np.ones(32,bool),np.arange(32)>0]
        full = Chunks(x,times,keep,[.5,1.])
        prefix = Chunks(x[:1],times[:1],keep[:1],[.5])
        with torch.inference_mode():
            a,_ = model.sequences([full]);b,_ = model.sequences([prefix])
        torch.testing.assert_close(a[:,:32],b,atol=2e-5,rtol=2e-5)

    def test_ctc_gradient_reaches_encoder_and_freeze_stops_it(self):
        torch.manual_seed(1)
        base = SLTStage1V17(Stage1V17Config(dim=32, depth=1, heads=4))
        model = JointCTC(base)
        x = torch.randn(2, 32, 61, 5)
        x[..., 3:] = 1
        logits = model.tokens(x)
        self.assertEqual(tuple(logits.shape), (2, 32, 102))
        loss = ctc_loss(logits, [(1, 1), (2,)], [32, 32])
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in base.parameters()))
        model.zero_grad(set_to_none=True)
        base.requires_grad_(False)
        ctc_loss(model.tokens(x), [(1,), (2,)], [32, 32]).backward()
        self.assertTrue(all(p.grad is None for p in base.parameters()))


if __name__ == '__main__':
    unittest.main()
