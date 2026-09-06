import itertools
import unittest

import numpy as np

from active.v17.continuous_decode_v17 import CTCPrefixDecoder, RevisableTranscript
from active.v17.train_streaming_tcn_ctc_v17 import collapse_ctc


class ContinuousDecodeTest(unittest.TestCase):
    def test_prefix_probability_matches_all_paths_including_repeated_words(self):
        probs = np.asarray([[.1, .8, .1], [.7, .2, .1], [.1, .8, .1]])
        expected = {}
        for path in itertools.product(range(3), repeat=3):
            prefix = collapse_ctc(np.asarray(path))
            expected[prefix] = expected.get(prefix, 0.0) + np.prod([probs[i, t] for i, t in enumerate(path)])
        decoder = CTCPrefixDecoder(beam_width=100, token_topk=3)
        for row in probs:
            decoder.step(np.log(row))
        observed = {p: np.exp(score) for p, score in decoder.alternatives()}
        for prefix, probability in expected.items():
            self.assertAlmostEqual(observed[prefix], probability, places=12)
        self.assertEqual(decoder.alternatives()[0][0], (1, 1))

    def test_words_continue_before_previous_word_stabilizes_and_can_revise(self):
        transcript = RevisableTranscript(stability_seconds=.6)
        first = transcript.update([((1,), 0.0)], 0.1)
        second = transcript.update([((1, 2), 0.0)], .2)
        self.assertEqual(first.stable_tokens, ())
        self.assertEqual(second.tokens, (1, 2))
        revised = transcript.update([((3, 2), 0.0)], .3)
        self.assertEqual(revised.revised_from, 0)
        self.assertEqual(revised.stable_tokens, ())
        stable = transcript.update([((3, 2, 4), 0.0)], 1.0)
        self.assertEqual(stable.stable_tokens, (3, 2))


if __name__ == "__main__":
    unittest.main()
