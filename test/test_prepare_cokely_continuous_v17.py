import unittest
from pathlib import Path
import tempfile

from scripts.prepare_cokely_continuous_v17 import exact_runs, read_events


class CokelyPreparationTests(unittest.TestCase):
    def test_event_at_zero_is_linked_to_parent_utterance(self):
        xml = '''<ANNOTATION_DOCUMENT><HEADER TIME_UNITS="milliseconds"/>
        <TIME_ORDER><TIME_SLOT TIME_SLOT_ID="t0" TIME_VALUE="0"/>
        <TIME_SLOT TIME_SLOT_ID="t1" TIME_VALUE="100"/>
        <TIME_SLOT TIME_SLOT_ID="t2" TIME_VALUE="200"/></TIME_ORDER>
        <TIER TIER_ID="ASL-TT"><ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="u1"
        TIME_SLOT_REF1="t0" TIME_SLOT_REF2="t2"><ANNOTATION_VALUE>x</ANNOTATION_VALUE>
        </ALIGNABLE_ANNOTATION></ANNOTATION></TIER>
        <TIER TIER_ID="ASL-individual-cp"><ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a1"
        TIME_SLOT_REF1="t0" TIME_SLOT_REF2="t1"><ANNOTATION_VALUE>GOOD</ANNOTATION_VALUE>
        </ALIGNABLE_ANNOTATION></ANNOTATION></TIER></ANNOTATION_DOCUMENT>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.eaf'
            path.write_text(xml)
            self.assertEqual(read_events(path)['main'][0]['utterance_id'], 'u1')

    def test_unknown_and_missing_timing_break_runs(self):
        events = [
            {'raw_gloss': 'GOOD', 'start_ms': 0, 'end_ms': 100},
            {'raw_gloss': 'BAD', 'start_ms': 100, 'end_ms': 200},
            {'raw_gloss': 'OTHER', 'start_ms': 200, 'end_ms': 300},
            {'raw_gloss': 'GOOD', 'start_ms': None, 'end_ms': 400},
            {'raw_gloss': 'GOOD', 'start_ms': 400, 'end_ms': 500},
            {'raw_gloss': 'BAD', 'start_ms': 500, 'end_ms': 600},
        ]
        self.assertEqual(
            [[e['raw_gloss'] for e in run] for run in exact_runs(events, {'GOOD', 'BAD'})],
            [['GOOD', 'BAD'], ['GOOD', 'BAD']],
        )

    def test_keeps_maximal_run_and_repeated_tokens(self):
        events = [
            {'raw_gloss': 'GOOD', 'start_ms': i * 100, 'end_ms': (i + 1) * 100,
             'utterance_id': 'u1'}
            for i in range(13)
        ]
        self.assertEqual([len(run) for run in exact_runs(events, {'GOOD'})], [13])

    def test_utterance_boundaries_and_other_hand_activity_break_runs(self):
        events = [
            {'raw_gloss': 'GOOD', 'start_ms': 0, 'end_ms': 100, 'utterance_id': 'u1'},
            {'raw_gloss': 'BAD', 'start_ms': 100, 'end_ms': 200, 'utterance_id': 'u1'},
            {'raw_gloss': 'GOOD', 'start_ms': 200, 'end_ms': 300, 'utterance_id': 'u2'},
            {'raw_gloss': 'BAD', 'start_ms': 300, 'end_ms': 400, 'utterance_id': 'u2'},
        ]
        supplementary = [
            # A matching simultaneous tag is duplicate evidence, not another token.
            {'raw_gloss': 'GOOD', 'start_ms': 0, 'end_ms': 100},
            # A different hand tag conflicts with the second utterance.
            {'raw_gloss': 'OTHER', 'start_ms': 250, 'end_ms': 350},
        ]
        runs = exact_runs(events, {'GOOD', 'BAD'}, supplementary=supplementary)
        self.assertEqual([[e['raw_gloss'] for e in run] for run in runs], [['GOOD', 'BAD']])


if __name__ == '__main__':
    unittest.main()
