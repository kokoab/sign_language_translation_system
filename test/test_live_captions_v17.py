"""Desktop live captions: parity with the experiment engine and session behavior."""
import json
from pathlib import Path
import random
import re
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.live_captions_v17 import DEFAULT_MODEL_DIR, CaptionModels, LiveCaptions

EXPERIMENT = REPO / 'experiments/streaming_translation/caption_tagger'
CASES = REPO / 'experiments/streaming_translation/cases.json'
HAVE_MODELS = (DEFAULT_MODEL_DIR / 'manifest.json').is_file()
_MODELS = None


def models():
    global _MODELS
    if _MODELS is None:
        _MODELS = CaptionModels()
    return _MODELS


def replay(session, glosses, scores, arrivals):
    """Feed a timed stream; returns the sequence of displayed captions and spoken text."""
    shown, spoken = [], []
    for j, (g, s, t) in enumerate(zip(glosses, scores, arrivals)):
        u = session.push(g, s, t)
        shown.append(u.text); spoken.append(u.speak)
        nxt = arrivals[j + 1] if j + 1 < len(arrivals) else None
        if nxt is None or nxt > t + session.end_seconds:
            u = session.tick(t + session.end_seconds)
            if u is not None and u.text != shown[-1]:
                shown.append(u.text)
            if u is not None:
                spoken.append(u.speak)
    return shown, [x for x in spoken if x]


@unittest.skipUnless(HAVE_MODELS, 'export the caption models first (export_runtime.py)')
class SessionTest(unittest.TestCase):
    def test_user_example_streams_word_by_word(self):
        c = LiveCaptions(models())
        shown, _ = replay(c, 'I GO HOSPITAL TOMORROW MORNING'.split(), [.9] * 5, [0., 1., 2., 3., 4.])
        self.assertEqual(shown[:3], ['I', 'I go', 'I go to the hospital'])
        self.assertTrue(shown[-1].startswith('I go to the hospital tomorrow morning'))

    def test_meaning_fixes_at_clause_end(self):
        for glosses, final in (('I WANT WATER NO', 'I do not want water'),
                               ('YOU GO SCHOOL WHERE', 'Where do you go to school?')):
            c = LiveCaptions(models())
            shown, _ = replay(c, glosses.split(), [.9] * 4, [0., 1., 2., 3.])
            self.assertEqual(shown[-1], final)

    def test_frozen_text_never_changes_and_speech_is_not_repeated(self):
        random.seed(3)
        glosses = sorted(models().known)
        for _ in range(40):
            seq = random.choices(glosses, k=random.randint(4, 14))
            c = LiveCaptions(models())
            frozen, spoken = [], []
            for i, g in enumerate(seq):
                u = c.push(g, .9, float(i))
                self.assertEqual(c.state['frozen'][:len(frozen)], frozen)
                frozen = list(c.state['frozen'])
                spoken.append(u.speak)
            u = c.finish()
            words = ' '.join(x for x in spoken + [u.speak] if x).lower().split()
            self.assertLessEqual(len(words), len(c.text.split()) + 2)

    def test_fingerspelling_and_unknown_glosses_render_as_names(self):
        c = LiveCaptions(models())
        for i, g in enumerate(['MY', 'NAME', 'fs-GE', 'fs-LO']):
            c.push(g, .9, float(i))
        self.assertIn('Gelo', c.finish().text)
        c.reset()
        c.push('ZEBRA', .9, 0.)
        self.assertEqual(c.finish().text, 'Zebra.')

    def test_spoken_names_follow_their_own_sentence(self):
        c = LiveCaptions(models())
        spoken = []
        for i, g in enumerate(['MY', 'NAME', 'fs-JO']):
            spoken.append(c.push(g, .9, float(i)).speak)
        spoken.append(c.tick(10.).speak)
        for i, g in enumerate(['MY', 'FRIEND', 'NAME', 'fs-QQ']):
            spoken.append(c.push(g, .9, 20. + i).speak)
        spoken.append(c.finish().speak)
        text = ' '.join(x for x in spoken if x)
        self.assertIn('Jo', text)
        self.assertIn('Qq', text)
        self.assertEqual(text.count('Jo'), 1)

    def test_end_of_turn_then_resume_continues_the_caption(self):
        c = LiveCaptions(models())
        c.push('HELLO', .9, 0.)
        self.assertIsNotNone(c.tick(4.0))
        self.assertIsNone(c.tick(5.0))          # only once per pause
        u = c.push('FRIEND', .9, 6.0)
        self.assertTrue(u.text.lower().startswith('hello'))
        self.assertIn('friend', u.text.lower())

    def test_long_silence_starts_a_new_sentence_and_speech_skips_nothing(self):
        c = LiveCaptions(models(), end_seconds=1.5, freeze_seconds=3.5)
        spoken = []
        for i, g in enumerate(['HELLO', 'HOW', 'YOU']):
            spoken.append(c.push(g, .9, float(i)).speak)
        u = c.tick(3.5)
        spoken.append(u.speak)
        settled = c.text
        self.assertEqual(settled, 'Hello, how are you?')
        for i, g in enumerate(['I', 'FEEL', 'HUNGRY']):
            spoken.append(c.push(g, .9, 15. + i).speak)
            self.assertTrue(c.text.startswith(settled))       # the earlier sentence never changes
        spoken.append(c.tick(19.).speak)
        said = ' '.join(x for x in spoken if x)
        self.assertEqual(said, 'Hello, how are you? I feel hungry')

    def test_short_hesitation_can_still_continue_the_sentence(self):
        c = LiveCaptions(models(), end_seconds=1.5, freeze_seconds=3.5)
        c.push('HELLO', .9, 0.)
        c.tick(1.6)
        u = c.push('FRIEND', .9, 2.5)
        self.assertTrue(u.text.lower().startswith('hello'))
        self.assertNotIn('.', u.text.split()[0])

    def test_reset_clears(self):
        c = LiveCaptions(models())
        c.push('HELLO', .9, 0.)
        c.reset()
        self.assertEqual(c.text, '')
        self.assertEqual(c.finish().text, '')


@unittest.skipUnless(HAVE_MODELS and (EXPERIMENT / 'data.json').is_file() and CASES.is_file(),
                     'experiment engine or its data is not present')
class ParityTest(unittest.TestCase):
    """The app runtime must show exactly what the evaluated experiment engine shows."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(EXPERIMENT))
        from live_caption import LiveCaptioner
        from scripts.live_captions_v17 import CONTEXT
        cls.ref = LiveCaptioner(end_s=3.5, draft_mode='no_post', live_path=['tagger_k1.pt', 'tagger_k1b.pt'],
                                context=CONTEXT)

    @staticmethod
    def visible(texts):
        """What a viewer sees: the app drops orphan leading punctuation (deliberate
        display difference); consecutive identical captions are one view."""
        out = []
        for t in texts:
            t = re.sub(r'^[\s.,!?;:]+', '', t)
            t = t[:1].upper() + t[1:]
            if not out or out[-1] != t:
                out.append(t)
        return out

    def compare(self, glosses, scores, arrivals):
        r = self.ref.run(glosses, scores, arrivals)
        expected = self.visible(e['text'] for e in r['events'])
        shown, _ = replay(LiveCaptions(models()), glosses, scores, arrivals)
        self.assertEqual(self.visible(shown), expected, glosses)

    def test_recorded_and_diagnostic_streams(self):
        cases = [c for c in json.loads(CASES.read_text()) if c.get('arrivals') and c.get('event_complete')]
        self.assertGreaterEqual(len(cases), 40)
        for c in cases:
            self.compare(c['glosses'], c['scores'], c['arrivals'])

    def test_random_streams(self):
        random.seed(11)
        glosses = sorted(models().known)
        for _ in range(60):
            n = random.randint(2, 18)
            seq = random.choices(glosses, k=n)
            gaps = [random.choice([.4, 1., 1.3, 2.5, 4.]) for _ in range(n)]
            arrivals = [sum(gaps[:i]) for i in range(n)]
            self.compare(seq, [random.choice([.3, .5, .9]) for _ in seq], arrivals)


if __name__ == '__main__':
    unittest.main()
