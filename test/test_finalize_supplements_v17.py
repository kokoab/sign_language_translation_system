import unittest
import numpy as np
from scripts.finalize_supplements_v17 import feature_check

class FeatureContractTest(unittest.TestCase):
    def test_rejects_invalid_features(self):
        x=np.zeros((32,61,5),np.float32)
        x[:2,0,3:]=1
        feature_check(x)
        feature_check(np.zeros_like(x),require_hands=False)
        for change in ('missing','confidence','presence','nonfinite','no_hands'):
            bad=x.copy()
            if change=='missing': bad[3,0,0]=1
            if change=='confidence': bad[0,0,4]=2
            if change=='presence': bad[0,0,3]=.5
            if change=='nonfinite': bad[0,0,0]=np.nan
            if change=='no_hands': bad[:]=0
            with self.subTest(change=change),self.assertRaises(ValueError):
                feature_check(bad)

if __name__=='__main__': unittest.main()
