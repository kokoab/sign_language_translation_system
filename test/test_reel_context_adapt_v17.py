import unittest
from scripts.train_reel_context_adapt_v17 import crop_bounds, accepted_candidate


class ContextAdaptChecks(unittest.TestCase):
    def test_complete_crop_and_retention(self):
        event = {'start_seconds': 1., 'end_seconds': 2.}
        neighbours = [event, {'start_seconds': 2.05, 'end_seconds': 3.}]
        self.assertEqual(crop_bounds(event, neighbours, .1), (.9, 2.05))
        self.assertIsNone(crop_bounds(event, neighbours + [{'start_seconds': 1.5, 'end_seconds': 2.5}], .1))
        base = dict(wer=.5, correct=82, insertions=6)
        candidate = dict(wer=.49, correct=84, insertions=6,
                         retained_baseline_correct_events=82)
        self.assertTrue(accepted_candidate(candidate, base))
        candidate['retained_baseline_correct_events'] = 79
        self.assertFalse(accepted_candidate(candidate, base))
        candidate['retained_baseline_correct_events'] = 82
        candidate['insertions'] = 7
        self.assertFalse(accepted_candidate(candidate, base))


if __name__ == '__main__':
    unittest.main()
