"""Dense sequence CE must exclude ambiguous/unguarded and incomplete annotation regions."""
import unittest
import numpy as np
import torch
from active.v17.joint_ctc_frame_supervision_v17 import sequence_frame_targets, sequence_frame_loss

class FrameSupervisionTest(unittest.TestCase):
    def test_verified_intervals_other_and_guarded_interior_gaps(self):
        row={'all_signs_annotated':True,'intervals':[
            {'start_seconds':.1,'end_seconds':.3,'label':'A'},
            {'start_seconds':.7,'end_seconds':.9,'label':'__OTHER__'}]}
        times=np.array([0,.1,.2,.3,.35,.45,.55,.65,.75,.95])
        targets=sequence_frame_targets(times,row,{'A':0})
        self.assertEqual(targets.tolist(),[-100,1,1,1,-100,0,0,-100,101,-100])
        row['all_signs_annotated']=False
        with self.assertRaises(ValueError):sequence_frame_targets(times,row,{'A':0})

    def test_overlap_and_padding_have_no_loss_or_gradient(self):
        row={'all_signs_annotated':True,'intervals':[
            {'start_seconds':0,'end_seconds':.3,'label':'A'},
            {'start_seconds':.2,'end_seconds':.4,'label':'A'}]}
        t=sequence_frame_targets(np.array([0,.1,.2,.3,.4]),row,{'A':0})
        self.assertEqual(t.tolist(),[1,1,-100,-100,1])
        logits=torch.zeros(1,7,102,requires_grad=True)
        loss=sequence_frame_loss(logits,[t],3);loss.backward()
        self.assertLess(float(logits.grad[0,0,1]),0)
        self.assertEqual(float(logits.grad[0,2:4].abs().sum()),0)
        self.assertEqual(float(logits.grad[0,5:].abs().sum()),0)
        with self.assertRaises(ValueError):sequence_frame_loss(logits,[t],0)
        whole=sequence_frame_loss(torch.cat([logits,logits]),[t,t],6)
        pieces=2*sequence_frame_loss(logits,[t],6)
        torch.testing.assert_close(whole,pieces)

if __name__=='__main__':unittest.main()
