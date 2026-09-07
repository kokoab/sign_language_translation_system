import tempfile
from pathlib import Path
import unittest

from scripts.audit_molo_continuous_v17 import candidate_spans, read_eaf


def event(gloss, start, end, hand='RightHand_IDg', cve='1', notes=None):
    return dict(raw_gloss=gloss, start_ms=start, end_ms=end, hand=hand,
                cve_ref=cve, annotation_id=f'{hand}:{start}', notes=notes or [])


class MoLoAuditTests(unittest.TestCase):
    def test_deduplicates_two_hands_but_preserves_repeated_signs(self):
        events = [event('GO', 0, 100), event('GO', 10, 110, 'LeftHand_IDg'),
                  event('GO', 150, 250)]
        spans = candidate_spans(events, {'GO': 'GO'}, 300)
        self.assertEqual([e['raw_gloss'] for e in spans[0]], ['GO', 'GO'])
        self.assertEqual(spans[0][0]['end_ms'], 110)

    def test_other_hand_oov_or_conflicting_variant_breaks_the_run(self):
        base = [event('GO', 0, 100), event('SCHOOL', 200, 300)]
        for blocker in [event('OTHER', 110, 180, 'LeftHand_IDg'),
                        event('OTHER', 10, 80, 'LeftHand_IDg'),
                        event('GO', 10, 80, 'LeftHand_IDg', cve='different')]:
            self.assertEqual(candidate_spans(base + [blocker], {'GO': 'GO', 'SCHOOL': 'SCHOOL'}, 300), [])

    def test_no_aliases_uncertain_notes_or_long_unannotated_gap(self):
        for first in [event('go', 0, 100), event('GO2', 0, 100),
                      event('GO', 0, 100, notes=['[?]'])]:
            self.assertEqual(candidate_spans([first, event('SCHOOL', 150, 250)],
                                             {'GO': 'GO', 'SCHOOL': 'SCHOOL'}, 300), [])
        self.assertEqual(candidate_spans([event('GO', 0, 100), event('GO', 500, 600)],
                                         {'GO': 'GO'}, 300), [])

    def test_parser_resolves_notes_and_rejects_missing_timing(self):
        xml = '''<ANNOTATION_DOCUMENT><HEADER TIME_UNITS="milliseconds">
          <MEDIA_DESCRIPTOR MEDIA_URL="file:///private/source%20one.mp4"/></HEADER>
          <TIME_ORDER><TIME_SLOT TIME_SLOT_ID="t1" TIME_VALUE="100"/>
            <TIME_SLOT TIME_SLOT_ID="t2" TIME_VALUE="200"/></TIME_ORDER>
          <TIER TIER_ID="RightHand_IDg" LINGUISTIC_TYPE_REF="ID Gloss" PARTICIPANT="A">
            <ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a" TIME_SLOT_REF1="t1" TIME_SLOT_REF2="t2" CVE_REF="1">
            <ANNOTATION_VALUE>GO</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION></TIER>
          <TIER TIER_ID="RightHand_Append" PARENT_REF="RightHand_IDg">
            <ANNOTATION><REF_ANNOTATION ANNOTATION_ID="b" ANNOTATION_REF="a">
            <ANNOTATION_VALUE>[?]</ANNOTATION_VALUE></REF_ANNOTATION></ANNOTATION></TIER>
        </ANNOTATION_DOCUMENT>'''
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'MoLo001_N_A_version.eaf'
            p.write_text(xml)
            result = read_eaf(p)
            self.assertEqual(result['events'][0]['notes'], ['[?]'])
            self.assertEqual(result['media_names'], ['source one.mp4'])
            self.assertEqual(result['participant_fields'], ['A'])
            p.write_text(xml.replace('TIME_VALUE="200"', ''))
            with self.assertRaises(ValueError):
                read_eaf(p)


if __name__ == '__main__':
    unittest.main()
