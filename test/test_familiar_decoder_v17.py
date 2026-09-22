import unittest
import numpy as np

from scripts.evaluate_familiar_decoder_v17 import prefix_beam, build_bigram, uniform_prior


class FamiliarDecoderTest(unittest.TestCase):
    def test_prefix_beam_keeps_repeated_token_when_blank_separates_it(self):
        probabilities = np.full((3, 102), 1e-9, dtype=np.float64)
        probabilities[0, 4] = .999; probabilities[1, 0] = .999; probabilities[2, 4] = .999
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        self.assertEqual(prefix_beam(np.log(probabilities), width=8), [4, 4])

    def test_prior_has_no_forced_first_word_or_eos(self):
        prior, info = build_bigram([
            {"single": False, "targets": (2, 3)}, {"single": False, "targets": (2, 3)},
        ])
        self.assertEqual(info["distinct_phrase_transcripts"], 1)
        self.assertAlmostEqual(float(np.exp(prior[2, 1:]).sum()), 1.0)
        self.assertEqual(float(prior[2, 0]), float("-inf"))
        self.assertAlmostEqual(float(np.exp(uniform_prior()[2, 1:]).sum()), 1.0)


if __name__ == "__main__": unittest.main()
