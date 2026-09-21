import unittest

from scripts.audit_annotation_identities_v17 import classify_identity


class AnnotationIdentityTests(unittest.TestCase):
    def test_only_established_distinct_identities_are_oov(self):
        index = {'KNOWN': [{'Code': 'K', 'LemmaID': 'family'}],
                 'VARIANT': [{'Code': 'V', 'LemmaID': 'family'}],
                 'OUT': [{'Code': 'O', 'LemmaID': 'other'}],
                 'AMBIG': [{'Code': 'K', 'LemmaID': 'family'}, {'Code': 'O', 'LemmaID': 'other'}]}
        def status(variant, occurrence=None, kind='Lexical Signs'):
            return classify_identity(variant, occurrence or variant, kind, index, {'K'}, {'family'})[0]
        self.assertEqual(status('KNOWN', 'KNOWN++'), 'known')
        self.assertEqual(status('OUT'), 'oov')
        for variant, occurrence, kind in [('VARIANT', None, 'Lexical Signs'),
                                           ('AMBIG', None, 'Lexical Signs'),
                                           ('MISSING', None, 'Lexical Signs'),
                                           ('OUT', 'OUT2', 'Lexical Signs'),
                                           ('OUT', None, 'Fingerspelled Signs')]:
            with self.subTest(variant=variant, occurrence=occurrence):
                self.assertEqual(status(variant, occurrence, kind), 'unresolved')


if __name__ == '__main__':
    unittest.main()
