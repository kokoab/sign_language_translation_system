import unittest
from types import SimpleNamespace
import numpy as np

from scripts.compare_avatar_source_v17 import match_world_hands


class SourceHandMatchTest(unittest.TestCase):
    def test_wrong_chirality_cannot_swap_observed_wrist_track(self):
        hand = SimpleNamespace(xy=np.array([[.3, .5]]), chirality="left")
        reference = {"left": None, "right": SimpleNamespace(xy=np.array([[.31, .5]]))}
        matched = match_world_hands([hand], reference)
        self.assertIs(matched["right"], hand)
        self.assertIsNone(matched["left"])
