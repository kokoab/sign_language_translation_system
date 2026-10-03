"""Focused checks for the streaming segmental runtime (no video, no checkpoints)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from active.v17.av_boundary_v17 import AVBoundary, AVBoundaryStream, clip_bio
from active.v17.segmental_runtime_v17 import B, I, O, UNK, both_spans, segmental_decode


def synthetic_raw(n, seed=0):
    rng = np.random.default_rng(seed)
    raw = np.zeros((n, 61, 5), np.float32)
    raw[:, :, :2] = rng.normal(0, .2, (n, 61, 2))
    raw[:, 57:59, 0] = [-.3, .3]           # shoulders present so the body frame is defined
    raw[:, :, 3] = 1
    raw[:, :, 4] = .9
    return raw, np.arange(n) / 20.


class StreamingBoundaryParity(unittest.TestCase):
    def test_stream_plus_flush_equals_clip(self):
        torch.manual_seed(0)
        model = AVBoundary(hidden=32, layers=1, heads=4, lookahead=6).eval()
        raw, times = synthetic_raw(40)
        reference = clip_bio(model, raw, times)
        stream = AVBoundaryStream(model)
        rows = []
        for i in range(len(times)):
            out = stream.update(raw[i], times[i])
            if out is not None and out[0] is not None:
                rows.append(out[1])
        rows += stream.flush()
        self.assertEqual(len(rows), len(reference))
        np.testing.assert_allclose(np.asarray(rows), reference, atol=1e-4)


class Decoder(unittest.TestCase):
    def bio(self, pattern):
        table = {'O': [.01, .97, .01, .01], 'B': [.01, .01, .97, .01], 'I': [.01, .01, .01, .97]}
        return np.log(np.asarray([table[c] for c in pattern], np.float64))

    def scores(self, spans, gloss_for):
        memo = {}
        for s, e in spans:
            logits = np.full(4, -5.)
            logits[gloss_for(s, e)] = 5.
            memo[(s, e)] = dict(v_raw=logits)
        return memo

    def test_two_signs_separated_by_rest(self):
        bio = self.bio('OOOIIIIIOOOIIIIIOOO')
        spans = both_spans(bio)
        memo = self.scores(spans, lambda s, e: 1 if e < 9 else 2)
        cfg = dict(alpha=1., w_r=2, w_a=.25, w_b=.5, c=-1.5, log_theta=np.log(.5), duplicate_gap=20)
        words = [g['gloss'] for g in segmental_decode(bio, memo, spans, ['A', 'B', 'C', 'D'], cfg) if g['gloss']]
        self.assertEqual(words, ['B', 'C'])

    def test_repeated_movement_collapses(self):
        bio = self.bio('OOIIIIOIIIIOOO')
        spans = both_spans(bio)
        memo = self.scores(spans, lambda s, e: 3)
        cfg = dict(alpha=1., w_r=2, w_a=.25, w_b=.5, c=-1.5, log_theta=np.log(.5), duplicate_gap=20)
        words = [g['gloss'] for g in segmental_decode(bio, memo, spans, ['A', 'B', 'C', 'D'], cfg) if g['gloss']]
        self.assertEqual(words, ['D'])

    def test_low_confidence_is_not_emitted(self):
        bio = self.bio('OOOIIIIIOOO')
        spans = both_spans(bio)
        memo = {sp: dict(v_raw=np.zeros(4)) for sp in spans}   # uniform: 0.25 < theta 0.5
        cfg = dict(alpha=1., w_r=2, w_a=.25, w_b=.5, c=-1.5, log_theta=np.log(.5), duplicate_gap=20)
        self.assertEqual([g for g in segmental_decode(bio, memo, spans, ['A', 'B', 'C', 'D'], cfg) if g['gloss']], [])


if __name__ == '__main__':
    unittest.main()


class Spelling(unittest.TestCase):
    def word(self, gloss, t):
        return dict(gloss=gloss, start_frame=int(t * 20), end_frame=int(t * 20) + 3, start_seconds=t,
                    end_seconds=t + .15, commit_seconds=t + .4, score=.9, early=False)

    def test_run_becomes_spelled_word_and_single_letter_is_dropped(self):
        from active.v17.segmental_runtime_v17 import SpellingBuffer
        b = SpellingBuffer()
        out = []
        for i, g in enumerate(['HELLO', 'FS_J', 'FS_O', 'FS_H', 'FS_N', 'YOU', 'FS_X', 'GO']):
            out += b.push(self.word(g, i * .5))
        out += b.flush()
        self.assertEqual([w['gloss'] for w in out], ['HELLO', 'fs-JOHN', 'YOU', 'GO'])
        self.assertEqual([w['gloss'] for w in b.dropped], ['FS_X'])

    def test_pause_ends_a_spelled_word(self):
        from active.v17.segmental_runtime_v17 import SpellingBuffer
        b = SpellingBuffer(gap_seconds=1.)
        for i, g in enumerate(['FS_A', 'FS_B']):
            self.assertEqual(b.push(self.word(g, i * .4)), [])
        self.assertEqual([w['gloss'] for w in b.tick(3.)], ['fs-AB'])

    def test_later_word_removes_its_provisional_letters(self):
        from active.v17.segmental_runtime_v17 import SpellingBuffer
        b = SpellingBuffer()
        for gloss, seconds in [('FS_C', 1), ('FS_B', 1.4)]:
            b.push(self.word(gloss, seconds))
        hello = self.word('HELLO', 1)
        hello.update(end_seconds=1.7, end_frame=34, commit_seconds=2.)
        self.assertEqual([w['gloss'] for w in b.push(hello)], ['HELLO'])
        self.assertEqual([w['gloss'] for w in b.dropped], ['FS_C', 'FS_B'])

    def test_word_does_not_erase_adjacent_valid_spelling(self):
        from active.v17.segmental_runtime_v17 import SpellingBuffer
        b = SpellingBuffer()
        for gloss, seconds in [('FS_G', 0), ('FS_Q', .4)]:
            b.push(self.word(gloss, seconds))
        self.assertEqual([w['gloss'] for w in b.push(self.word('HELLO', .56))], ['fs-GQ', 'HELLO'])

    def test_late_partial_word_cannot_erase_established_name(self):
        from active.v17.segmental_runtime_v17 import SpellingBuffer
        b = SpellingBuffer()
        b.push(self.word('FS_G', 0)); b.push(self.word('FS_E', .4))
        take = self.word('TAKE', .4)
        take.update(end_seconds=1.5, commit_seconds=1.8)
        self.assertEqual([w['gloss'] for w in b.push(take)], ['fs-GE', 'TAKE'])

    def test_idle_lock_uses_sign_end_and_active_hands_wait_for_next_word(self):
        from active.v17.segmental_runtime_v17 import SpellingBuffer
        b = SpellingBuffer()
        b.push(self.word('FS_A', 0)); b.push(self.word('FS_N', .4))
        self.assertEqual(b.tick(2.6, active_hands=True), [])
        self.assertEqual([w['gloss'] for w in b.push(self.word('HELLO', 3))], ['fs-AN', 'HELLO'])
        b.push(self.word('FS_A', 4)); b.push(self.word('FS_N', 4.4))
        self.assertEqual(b.tick(6.5, active_hands=False), [])
        self.assertEqual([w['gloss'] for w in b.tick(6.6, active_hands=False)], ['fs-AN'])


class FistGeometryRefinement(unittest.TestCase):
    def test_reshares_only_the_fist_group(self):
        from active.v17.fist_geometry_v17 import FistGeometry
        fg = FistGeometry()
        lm = np.zeros((32, 61, 5), np.float32)
        rng = np.random.default_rng(0)
        lm[:, :21, :2] = rng.normal(size=(21, 2)) * .1
        lm[:, 9, :2] = [0, -.3]
        lm[:, :21, 3] = 1
        logits = np.full(27, -5., np.float32); logits[18] = 6; logits[2] = 3   # S top, C second
        out = fg.refine(logits, lm)
        p0 = np.exp(logits - logits.max()); p0 /= p0.sum()
        p1 = np.exp(out - out.max()); p1 /= p1.sum()
        group = [0, 4, 12, 13, 18, 19]
        others = [i for i in range(27) if i not in group]
        np.testing.assert_allclose(p1[others], p0[others], atol=1e-6)        # other letters untouched
        self.assertAlmostEqual(float(p1[group].sum()), float(p0[group].sum()), places=6)
        logits[18], logits[2] = 3, 6                                         # top not a fist letter
        np.testing.assert_array_equal(fg.refine(logits, lm), logits)
        lm[:, :, 3] = 0                                                      # no hand: unchanged
        logits[18] = 9
        np.testing.assert_array_equal(fg.refine(logits, lm), logits)


class LetterGeometry(unittest.TestCase):
    def test_direction_crossing_and_abstention(self):
        from active.v17.segmental_runtime_v17 import refine_letter_geometry
        raw = np.zeros((16, 61, 5), np.float32)
        raw[:, :21, 4] = .9
        raw[:, 5, :2] = [.1, .2]
        raw[:, 9, :2] = [.2, .2]
        raw[:, 8, :2] = [.3, .2]  # horizontal index: G
        raw[:, 12, :2] = [.2, .1]
        logits = np.full(27, -10., np.float32); logits[16] = 10
        fixed = refine_letter_geometry(logits, raw)            # sideways index: Q -> G
        self.assertEqual(int(fixed.argmax()), 6)
        np.testing.assert_array_equal(np.sort(fixed), np.sort(logits))
        np.testing.assert_array_equal(refine_letter_geometry(logits, raw, q_to_g=False), logits)
        raw[:, 8, :2] = [.1, .5]  # downward index: Q
        logits[16], logits[6] = -10, 10
        fixed = refine_letter_geometry(logits, raw)
        self.assertEqual(int(fixed.argmax()), 16)
        np.testing.assert_array_equal(np.sort(fixed), np.sort(logits))
        raw[:, 8, :2] = [.25, .05]  # crossing relative to MCP ordering: R
        raw[:, 12, :2] = [.15, .05]
        logits[:] = -10; logits[20] = 10
        self.assertEqual(int(refine_letter_geometry(logits, raw).argmax()), 17)
        raw[:, 8, :2] = [.05, .05]  # uncrossed: U
        logits[20], logits[17] = -10, 10
        self.assertEqual(int(refine_letter_geometry(logits, raw).argmax()), 20)
        raw[:, :, 4] = 0
        np.testing.assert_array_equal(refine_letter_geometry(logits, raw), logits)
        logits[:] = -10; logits[13] = 10
        np.testing.assert_array_equal(refine_letter_geometry(logits, raw), logits)


class HandBatching(unittest.TestCase):
    def test_right_hand_and_union_keep_their_slots_and_rest_skips_encoder(self):
        from types import SimpleNamespace
        from active.v17.segmental_runtime_v17 import SpanRecognizer
        from active.v17.extract_v17 import HandDetection
        from active.v17.schema_hand_rgb_v17 import HandRGBV17Config
        calls = []
        def predict(feeds):
            calls.append(feeds)
            return [{'embedding': np.full(512, i + 1, np.float32)} for i in range(len(feeds))]
        recognizer = SpanRecognizer.__new__(SpanRecognizer)
        recognizer.encoder = SimpleNamespace(predict=predict)
        recognizer.config = HandRGBV17Config()
        hand = HandDetection(np.full((21, 2), .5, np.float32), np.ones(21, np.float32), 'right', 1.)
        observation = SimpleNamespace(frame=np.zeros((160, 240, 3), np.uint8), assigned={'left': None, 'right': hand})
        emb, valid, boxes = recognizer.frame_hand(observation)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 2)
        np.testing.assert_array_equal(valid, [0, 1, 1])
        np.testing.assert_array_equal(emb[:, 0], [0, 1, 2])
        np.testing.assert_array_equal(boxes[0], np.zeros(4))
        observation.assigned['right'] = None
        emb, valid, boxes = recognizer.frame_hand(observation)
        self.assertEqual(len(calls), 1)
        self.assertFalse(emb.any() or valid.any() or boxes.any())


class SlotRendering(unittest.TestCase):
    class Echo:
        def __init__(self, sentence):
            self.sentence = sentence

        def rephrase(self, glosses, scores):
            return dict(sentence=self.sentence.format(*glosses), input_glosses=list(glosses))

    def test_slots_are_restored_exactly(self):
        from scripts.live_segmental_v17 import render_sentence
        out = render_sentence(self.Echo('My name is FS0.'), ['MY', 'NAME', 'fs-KOKOAB'], [.9] * 3)
        self.assertEqual(out['sentence'], 'My name is Kokoab.')
        self.assertEqual(out['spelled_slot_mode'], 'renderer')

    def test_dropped_slot_falls_back_to_literal(self):
        from scripts.live_segmental_v17 import render_sentence
        out = render_sentence(self.Echo('My name is a year.'), ['MY', 'NAME', 'fs-KOKOAB'], [.9] * 3)
        self.assertEqual(out['sentence'], 'My name Kokoab.')
        self.assertEqual(out['spelled_slot_mode'], 'literal_fallback')


class SpellingPieces(unittest.TestCase):
    def test_split_holds_merge_and_slow_spelling_stays_one_word(self):
        from active.v17.segmental_runtime_v17 import SpellingBuffer
        stream = [('FS_C', 0, .5, 1.8), ('FS_C', .37, .97, 2.2), ('FS_A', 2.5, 3.0, 4.37),
                  ('FS_T', 4.2, 4.8, 6.0), ('FS_T', 4.87, 5.37, 6.7)]
        b, out, t = SpellingBuffer(), [], 0.
        for g, s, e, c in stream:
            while t < c:
                out += b.tick(t)
                t += .05
            out += b.push(dict(gloss=g, start_seconds=s, end_seconds=e, commit_seconds=c, start_frame=int(s * 20),
                               end_frame=int(e * 20), score=.9, early=False))
        out += b.flush()
        self.assertEqual([w['gloss'] for w in out], ['fs-CAT'])
