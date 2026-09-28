import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from scripts.repair_stage3_composition_v17 import build_rows, key, USER_PROBES


class CompositionDataTests(unittest.TestCase):
    def setUp(self):
        self.source = [
            dict(glosses=['I', 'NEED', 'WATER'], english='I need water.', split='train',
                 confidences=[.9]*3, noise_indices=[], structure='svo'),
            dict(glosses=['YOU', 'HAPPY'], english='You are happy.', split='train',
                 confidences=[.9]*2, noise_indices=[], structure='state'),
            dict(glosses=['MY', 'NAME', 'FS0'], english='My name is FS0.', split='test',
                 confidences=[.9]*3, noise_indices=[], structure='name'),
        ]

    def test_reserved_sequences_and_user_phrases_never_enter_training(self):
        replay, added, reserved = build_rows(self.source)
        self.assertFalse({key(r) for r in replay + added} & reserved)
        self.assertTrue(USER_PROBES <= reserved)
        self.assertTrue(all(r['split'] == 'train' for r in replay + added))

    def test_preserves_time_pronoun_and_address_roles_in_targets(self):
        _, added, _ = build_rows(self.source)
        rows = {key(r): r['english'] for r in added}
        self.assertEqual(rows['I GO DOCTOR TOMORROW MORNING'], 'I am going to the doctor tomorrow morning.')
        self.assertEqual(rows['YESTERDAY MY MOTHER GO HOSPITAL'], 'My mother went to the hospital yesterday.')
        self.assertEqual(rows['GOOD MORNING HOW YOU FRIEND'], 'Good morning. How are you, friend?')
        self.assertEqual(rows['I HUNGRY MY NAME FS0'], 'I am hungry. My name is FS0.')

    def test_training_generation_is_reproducible_and_confidences_align(self):
        a = build_rows(self.source)
        self.assertEqual(a, build_rows(self.source))
        self.assertTrue(all(len(r['glosses']) == len(r['confidences']) for r in a[1]))

    def test_existing_cross_split_overlap_fails_closed(self):
        self.source.append(dict(self.source[0], split='validation'))
        with self.assertRaises(ValueError):
            build_rows(self.source)


class NeuralOnlyContractTests(unittest.TestCase):
    def test_model_receives_how_you_together_across_long_pause(self):
        from scripts.live_segmental_v17 import render_utterance
        seen = []
        def rephrase(glosses, scores):
            seen.append(glosses)
            return dict(sentence='How are you?', rendering_mode='t5_efficient_tiny')
        renderer = SimpleNamespace(full_utterance=True, rephrase=rephrase)
        words = [dict(gloss='HOW', score=.9, start_seconds=0, end_seconds=1),
                 dict(gloss='YOU', score=.9, start_seconds=4, end_seconds=5)]
        result = render_utterance(renderer, words)
        self.assertEqual(seen, [['HOW', 'YOU']])
        self.assertEqual(result['clauses'], [['HOW', 'YOU']])
        self.assertEqual(render_utterance(renderer, [])['sentence'], '')

    def test_torch_contract_uses_model_even_for_old_reviewed_phrase(self):
        from scripts.live_isolated_v17 import TinyStage3Naturalizer
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'stage3_input_contract.json').write_text(json.dumps(
                dict(encoding='evidence', reviewed_templates_enabled=False)))
            renderer = TinyStage3Naturalizer(SimpleNamespace(
                stage3_checkpoint=path, stage3_device='cpu', stage3_encoding='evidence'))
            renderer._generate = lambda *args: 'Neural output.'
            result = renderer.rephrase(['I', 'NEED', 'WATER'])
            self.assertEqual(result['sentence'], 'Neural output.')
            self.assertEqual(result['rendering_mode'], 't5_efficient_tiny')

    def test_coreml_contract_uses_model_even_for_old_reviewed_phrase(self):
        from active.v17.stage3_coreml_v17 import CoreMLStage3Naturalizer
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'stage3_tokens.json').write_text(json.dumps(dict(
                word_ids={}, pieces=[], special_ids=[], eos_id=1, decoder_start_id=0,
                max_length=64, reviewed_templates_enabled=False)))
            with patch('active.v17.coreml_runtime_v17.load', return_value=object()):
                renderer = CoreMLStage3Naturalizer(path)
            renderer._generate = lambda *args: 'Neural output.'
            result = renderer.rephrase(['I', 'NEED', 'WATER'])
            self.assertEqual(result['sentence'], 'Neural output.')
            self.assertEqual(result['rendering_mode'], 't5_efficient_tiny')

    def test_coreml_overlong_input_falls_back_without_silent_truncation(self):
        from active.v17.stage3_coreml_v17 import CoreMLStage3Naturalizer
        renderer = CoreMLStage3Naturalizer.__new__(CoreMLStage3Naturalizer)
        renderer.word_ids = {'hi': [1], 'hello': [2]}
        renderer.full_utterance = True
        renderer.length = 4
        with self.assertRaisesRegex(ValueError, 'refuse to drop words'):
            renderer._generate(['HELLO', 'HELLO'], [.9, .9])


if __name__ == '__main__':
    unittest.main()
