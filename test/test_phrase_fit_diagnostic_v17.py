import unittest
from scripts.diagnose_phrase_fit_v17 import breakdown

class FitDiagnosticTests(unittest.TestCase):
    def test_repeated_tokens_and_other_are_counted_without_inflating_matches(self):
        rows=[dict(source='local_phrases',expected=[1,1,101],predicted=[1,2])]
        phrases,signs=breakdown(rows,{1:'A',2:'B',101:'OTHER'})
        self.assertEqual(phrases[0]['exact'],0)
        self.assertEqual(signs['A']['references'],2)
        self.assertEqual(signs['A']['matched'],1)
        self.assertEqual(signs['A']['missed'],1)
        self.assertEqual(signs['B']['unmatched_predictions'],1)
        self.assertNotIn('OTHER',signs)
