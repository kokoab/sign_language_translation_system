"""CTC and positive frame CE must not retain the measured blank local minimum."""
import unittest
import torch
from active.v17.joint_ctc_v17 import ctc_loss
from active.v17.joint_ctc_loss_balance_v17 import frame_normalized_ctc_loss

class LossBalanceTest(unittest.TestCase):
    def test_input_lengths_and_padding(self):
        torch.manual_seed(4)
        logits=torch.randn(2,7,102,requires_grad=True)
        targets=[(1,1),(100,)];lengths=[7,4]
        expected=(ctc_loss(logits[:1],targets[:1],[7])*2/7+
                  ctc_loss(logits[1:],targets[1:],[4])/4)/2
        actual=frame_normalized_ctc_loss(logits,targets,lengths)
        torch.testing.assert_close(actual,expected)
        actual.backward();self.assertEqual(float(logits.grad[1,4:].abs().sum()),0)
        with self.assertRaises(ValueError):frame_normalized_ctc_loss(logits,targets,[7])

    def test_frame_normalization_escapes_blank_basin_gradient(self):
        theta=torch.tensor(-2.44,requires_grad=True)
        blank=theta*0
        other=theta*0-100
        logits=torch.stack([blank,theta]+[other]*100).repeat(1,32,1)
        ce=torch.nn.functional.cross_entropy(logits[:,16],torch.tensor([1]))
        old=ctc_loss(logits,[(1,)],[32])+ce
        new=frame_normalized_ctc_loss(logits,[(1,)],[32])+ce
        self.assertGreater(float(torch.autograd.grad(old,theta,retain_graph=True)[0]),0)
        self.assertLess(float(torch.autograd.grad(new,theta)[0]),0)

if __name__=='__main__':unittest.main()
