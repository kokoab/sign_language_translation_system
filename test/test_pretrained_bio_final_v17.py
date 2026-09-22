import unittest
import numpy as np
class BioTargetsTest(unittest.TestCase):
 def test_targets_preserve_unknown_and_overlap(self):
  from scripts.train_pretrained_bio_final_v17 import targets
  t=np.arange(12)/20
  np.testing.assert_array_equal(targets(t,[[.1,.25]]),[-1,-1,2,3,3,3,-1,-1,-1,-1,-1,-1])
  y=targets(t,[[.1,.25],[.2,.4]])
  self.assertTrue(np.all(y[4:6]==-1))
  self.assertNotIn(1,y)
 def test_selection_rejects_lost_signs_and_extra_words(self):
  from scripts.train_pretrained_bio_final_v17 import eligible
  b=dict(wer=.5,correct=100,insertions=5,baseline_correct_events=100,retained_baseline_correct_events=100)
  self.assertTrue(eligible(dict(b,wer=.4),b))
  self.assertFalse(eligible(dict(b,wer=.4,insertions=6),b))
  self.assertFalse(eligible(dict(b,wer=.4,retained_baseline_correct_events=97),b))
