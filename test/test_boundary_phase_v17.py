import unittest

import torch

from active.v17.boundary_phase_v17 import (
    BoundaryPhaseHeadV17,
    BoundaryPhaseTranscript,
    KNOWN,
    TRANSITION,
    UNKNOWN,
    WINDOW_SECONDS,
)


class BoundaryPhaseV17Test(unittest.TestCase):
    def test_contract_and_state_machine(self):
        head = BoundaryPhaseHeadV17(256, 100)
        gloss = torch.randn(2, 100)
        encoded = torch.randn(2, 32, 256)
        self.assertEqual(tuple(head(encoded, gloss).shape), (2, 3))
        self.assertEqual(WINDOW_SECONDS, (0.27, 0.53))

        transcript = BoundaryPhaseTranscript(agreements=2)
        transcript.update(KNOWN, "HELLO", 0.27)
        transcript.update(KNOWN, "HELLO", 0.34)
        transcript.update(KNOWN, "HELLO", 0.41)
        self.assertEqual(transcript.words, ["HELLO"])
        transcript.update(UNKNOWN, None, 0.48)
        transcript.update(KNOWN, "HELLO", 0.55)
        transcript.update(KNOWN, "HELLO", 0.62)
        self.assertEqual(transcript.words, ["HELLO"])
        transcript.update(TRANSITION, None, 0.69)
        transcript.update(KNOWN, "HELLO", 0.76)
        transcript.update(KNOWN, "HELLO", 0.83)
        self.assertEqual(transcript.words, ["HELLO", "HELLO"])

    def test_unknown_is_never_visible(self):
        transcript = BoundaryPhaseTranscript(agreements=1)
        transcript.update(UNKNOWN, "OUTSIDE", 0.27)
        self.assertEqual(transcript.words, [])
        self.assertIsNone(transcript.provisional)


if __name__ == "__main__":
    unittest.main()
