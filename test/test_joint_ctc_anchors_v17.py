"""Anchors must respect timestamp ownership, annotation ambiguity and CTC IDs."""
import unittest
import numpy as np
import torch
from active.v17.joint_ctc_anchors_v17 import event_anchors, anchor_loss

class AnchorTest(unittest.TestCase):
    def test_overlaps_repeats_and_no_extrapolation(self):
        times=np.arange(9)/10
        events=[{'start_seconds':a,'end_seconds':b} for a,b in
                [(0,.2),(.2,.4),(.45,.55),(.49,.51),(1,2)]]
        anchors=event_anchors(times,events,[1,1,101,2,3])
        self.assertEqual(anchors,[(1,1),(3,1)])
        with self.assertRaises(ValueError):event_anchors(times,events,[1])
        with self.assertRaises(ValueError):event_anchors(times,events,[0,1,1,1,1])

    def test_anchor_gradient_ignores_unverified_frames(self):
        logits=torch.zeros(2,6,102,requires_grad=True)
        loss=anchor_loss(logits,[[(2,1),(4,101)],[(3,100)]],(2,1))
        loss.backward()
        self.assertLess(logits.grad[0,2,1],0)
        self.assertGreater(logits.grad[0,2,0],0)
        self.assertEqual(float(logits.grad[0,0].abs().sum()),0)
        self.assertEqual(float(logits.grad[1,5].abs().sum()),0)

    def test_epoch_weight_is_invariant_to_batch_partition(self):
        torch.manual_seed(5)
        logits=torch.randn(3,5,102)
        anchors=[[(1,1)],[(1,101),(3,101)],[(2,100)]]
        population=(2,2)
        whole=anchor_loss(logits,anchors,population)
        parts=sum(anchor_loss(logits[i:i+1],anchors[i:i+1],population) for i in range(3))
        torch.testing.assert_close(whole,parts)

if __name__=='__main__':unittest.main()
