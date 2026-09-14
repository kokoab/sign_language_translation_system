"""Positive isolated/core supervision must teach the actual CTC output."""
import unittest
import torch
from active.v17.joint_ctc_v17 import JointCTC
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.joint_ctc_supervision_v17 import positive_ctc_loss


class PositiveCTCTest(unittest.TestCase):
    def test_positive_loss_reverses_blank_collapse_and_reaches_head(self):
        torch.manual_seed(3)
        model=JointCTC(SLTStage1V17(Stage1V17Config(dim=32,depth=1,heads=4))).eval()
        x=torch.randn(2,32,61,5);x[...,3:]=1
        targets=torch.tensor([0,99])
        pooled=torch.nn.functional.cross_entropy(model.base(x),targets)
        grads=torch.autograd.grad(pooled,tuple(model.head.parameters()),allow_unused=True)
        self.assertTrue(all(g is None for g in grads))
        with torch.no_grad():
            model.head.blank.weight.zero_();model.head.blank.bias.fill_(12)
        loss=positive_ctc_loss(model.tokens(x),targets)
        loss.backward()
        self.assertGreater(float(model.head.blank.bias.grad),0)
        self.assertLess(float(model.head.gloss_delta.bias.grad[0]),0)
        self.assertLess(float(model.head.gloss_delta.bias.grad[99]),0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.base.parameters()))

    def test_other_and_no_emit_are_not_isolated_positive_targets(self):
        with self.assertRaises(ValueError):
            positive_ctc_loss(torch.randn(1,32,102),torch.tensor([100]))


if __name__=='__main__':unittest.main()
