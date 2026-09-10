"""Deterministic stream checks; only camera/model boundaries are substituted."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from active.v17.extract_v17 import FrameDetection
from scripts import live_reel_stage1_v17 as reel
from scripts import reel_hud_v17 as hud


class DelayedFuture:
    def __init__(self, fn, args):
        self.fn, self.args, self.ticks = fn, args, 3
        self.called = False

    def done(self):
        self.ticks -= 1
        return self.ticks <= 0

    def result(self):
        if not self.called:
            self.value = self.fn(*self.args)
            self.called = True
        return self.value


class Executor:
    def __init__(self, **_):
        pass

    def submit(self, fn, *args):
        future = DelayedFuture(fn, args)
        future.ticks = getattr(fn, 'delay_ticks', 3)
        return future

    def shutdown(self, **_):
        pass


def replay(root, continuous, finish=True, actions=None, language_delay=3,
           stability_hits=2, verification_delay=3, label_boundary=1, sequence_preview=False,
           revisable_transcript=False, sequence_hypotheses=None, frame_count=70):
    args = reel.parser().parse_args([
        '--video', 'fixture.mp4', '--no-display', '--no-speech',
        '--no-finish-gesture', '--naturalizer', 'literal',
    ] + (['--finish-at-eof'] if finish else []))
    args.output_root = Path(root)
    args.preserve_pending_frames = continuous
    args.provisional_glosses = continuous
    args.stage2_at_finish = continuous
    args.stage2_review_only = continuous
    args.no_stage2_arbiter = not continuous
    args.stability_hits = stability_hits
    args.sequence_preview = sequence_preview
    args.revisable_transcript = revisable_transcript
    if actions:
        args.no_display = False
    camera = SimpleNamespace(index=0, count=frame_count)

    def read():
        camera.index += 1
        return (camera.index <= camera.count, np.zeros((64, 64, 3), np.uint8))

    capture = SimpleNamespace(
        isOpened=lambda: True, set=lambda *a: None, get=lambda *a: 20.0,
        read=read, release=lambda: None,
    )
    proposals, verified, sequence_calls = [], [], []

    def classify(frames):
        proposals.append([f.seconds for f in frames])
        label = 'HELLO' if frames[0].seconds < label_boundary else 'YOU'
        return dict(gloss=label, candidate_gloss=label, accepted=True,
                    model_score=0.9, start_seconds=frames[0].seconds,
                    end_seconds=frames[-1].seconds, diagnostics={})

    def verify(frames):
        verified.append((frames[-1].seconds, camera.index))
        label = 'HELLO' if frames[0].seconds < label_boundary else 'YOU'
        return dict(gloss=label, candidate_gloss=label, accepted=True,
                    model_score=0.9, diagnostics={})
    verify.delay_ticks = verification_delay

    def classify_window(frames, prior):
        sequence_calls.append((camera.index, [f.seconds for f in frames]))
        index = len(sequence_calls) - 1
        hypothesis = (
            sequence_hypotheses[min(index, len(sequence_hypotheses) - 1)]
            if sequence_hypotheses else ['WRONG', 'WORDS']
        )
        return np.zeros((32, 612)), dict(
            accepted=True, hypothesis=hypothesis,
            token_positions=list(range(len(hypothesis))) if sequence_preview else [0, 9],
            window_count=len(prior) + 1, latency_ms={'total': 1},
        )

    def decode_frozen_logits(windows):
        logits = np.full((len(windows) * 8, 101), -20., np.float32)
        logits[:, 0] = 5.
        logits[1, 1] = 20.
        logits[3, 2] = 20.
        return logits, {'window_count': len(windows), 'specialist_selected': False}

    classifier = SimpleNamespace(classify=classify, verify=verify, provenance=lambda: {})
    sequence = SimpleNamespace(
        classify_window=classify_window, decode_frozen_logits=decode_frozen_logits,
        labels=['VISUAL', 'FINAL'], provenance=lambda: {},
    )
    detection = FrameDetection([], np.zeros((19, 2)), np.zeros(19),
                               np.zeros((4, 2)), np.zeros(4))
    def rephrase(words):
        return {'sentence': ' '.join(words)}
    rephrase.delay_ticks = language_delay
    naturalizer = SimpleNamespace(warm=lambda: {}, rephrase=rephrase)
    replacements = {
        'ThreadPoolExecutor': Executor,
        'LiveStage2CTC': lambda args: sequence,
        'AppleVisionDetector': lambda *a: SimpleNamespace(detect=lambda *a, **kw: detection),
        'LipMarkerTracker': lambda **kw: SimpleNamespace(detect=lambda *a: None, close=lambda: None),
        'make_naturalizer': lambda args: naturalizer,
        'observation_quality': lambda detection: (1.0, 1.0),
        'wrist_motion': lambda *a: 0.1,
    }
    with ExitStack() as stack, redirect_stdout(io.StringIO()):
        for name, value in replacements.items():
            stack.enter_context(patch.object(reel, name, value))
        stack.enter_context(patch.object(reel.cv2, 'VideoCapture', lambda *a: capture))
        stack.enter_context(patch.object(reel.SessionRecorder, 'write_frame', lambda *a: None))
        if actions:
            for name in ('namedWindow', 'setMouseCallback', 'imshow'):
                stack.enter_context(patch.object(reel.cv2, name, lambda *a: None))
            stack.enter_context(patch.object(reel.cv2, 'waitKey', lambda *a: ord(actions.get(camera.index, ' '))))
            stack.enter_context(patch.object(reel.ReelHud, 'draw', lambda self, frame, *a, **kw: frame))
            stack.enter_context(patch.object(reel, 'draw_reel_detection', lambda frame, *a, **kw: frame))
            stack.enter_context(patch.object(reel, 'draw_lip_markers', lambda frame, *a, **kw: frame))
        summary = reel.run(args, classifier_factory=lambda args: classifier)
    history = json.loads(Path(summary['history']).read_text())
    return proposals, verified, sequence_calls, history, summary


class ContinuousReelTests(unittest.TestCase):
    def test_revisable_transcript_replaces_a_wrong_tail_and_finish_uses_visual_decode(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, _, history, summary = replay(
                root, True, revisable_transcript=True,
                sequence_hypotheses=[['HELLO', 'YOU'], ['KNOW', 'YOU']],
            )
        updates = [event for event in history['events']
                   if event['type'] == 'revisable_transcript_update']
        self.assertGreaterEqual(len(updates), 2)
        self.assertEqual(updates[1]['replaced_tail'], ['HELLO', 'YOU'])
        self.assertEqual(updates[1]['new_tail'], ['KNOW', 'YOU'])
        self.assertFalse(any(event['type'] == 'sequence_prefix_update'
                             for event in history['events']))
        selected = next(event for event in history['events']
                        if event['type'] == 'finished_sequence_selected')
        self.assertEqual(selected['selection_mode'], 'revisable_visual_redecode')
        self.assertEqual(selected['selected'], ['VISUAL', 'FINAL'])
        self.assertEqual(summary['hypothesis'], ['VISUAL', 'FINAL'])
        self.assertTrue(history['capture_stats']['revisable_transcript'])

    def test_revisable_hud_labels_the_transcript_as_changeable(self):
        view = hud.ReelHud(provisional=True)
        words = []
        original = hud._shadowed
        def record(*args, **kwargs):
            words.append(args[2])
            return original(*args, **kwargs)
        with patch.object(hud, '_shadowed', record):
            view.draw(
                np.zeros((720, 1280, 3), np.uint8), None, None,
                reel.StableGlossLock(), False, True, 20, ['HELLO', 'HOW'], [],
                '', False, None, ['HELLO', 'HOW'], {}, revisable_transcript=True,
            )
        self.assertIn('Live transcript · may change', words)

    def test_revisable_stitch_preserves_repeated_sign_across_a_blank_at_a_chunk_seam(self):
        left = np.full((64, 4), -20., np.float32)
        right = np.full((64, 4), -20., np.float32)
        left[:, 0] = right[:, 0] = 5.
        left[45, 2] = 20.
        right[22, 2] = 20.
        stitched = reel.stitch_revisable_ctc_logits([(0, left), (6, right)], 112)
        tokens, _ = reel.collapse_ctc_path(stitched, len(stitched))
        self.assertEqual(tokens, (2, 2))

    def test_revisable_final_decode_covers_long_retained_utterances(self):
        class Decoder:
            def __init__(self):
                self.calls = []
                self.labels = ['SIGN']
            def decode_frozen_logits(self, windows):
                self.calls.append(len(windows))
                logits = np.full((len(windows) * 8, 101), -20., np.float32)
                logits[:, 0] = 5.
                return logits, {'specialist_selected': False}
        decoder = Decoder()
        retained = reel.RetainedFrozenWindows()
        try:
            for _ in range(19):
                retained.append(np.zeros((32, 612), np.float32))
            result = reel.final_revisable_decode(decoder, retained)
        finally:
            retained.close()
        self.assertEqual(decoder.calls, [8, 8, 7])
        self.assertEqual(result['retained_windows'], 19)
        self.assertEqual(result['hypothesis'], [])

    def test_revisable_reset_discards_stale_decoder_result_and_retained_windows(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, _, history, _ = replay(
                root, True, finish=False, actions={23: 'r', 62: 'f'},
                sequence_preview=True, revisable_transcript=True,
            )
        self.assertTrue(any(event['type'] == 'reset' for event in history['events']))
        updates = [event for event in history['events']
                   if event['type'] == 'revisable_transcript_update']
        self.assertTrue(updates)
        self.assertLessEqual(updates[-1]['retained_windows'], 2)

    def test_sequence_prefix_evidence_advances_after_context_rollover(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, _, history, _ = replay(root, True, sequence_preview=True, frame_count=230)
        updates = [e for e in history['events'] if e['type'] == 'sequence_prefix_update']
        ends = [e['evidence_end'] for e in updates]
        self.assertGreater(max(ends), 64)
        self.assertEqual(ends, sorted(set(ends)))
        self.assertEqual(updates[-1]['committed'][:2], ['WRONG', 'WORDS'])

    def test_reset_during_sequence_work_excludes_old_utterance(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, calls, history, _ = replay(
                root, True, finish=False, actions={25: 'f', 28: 'r', 62: 'f'},
                stability_hits=1,
            )
        selected = [e for e in history['events'] if e['type'] == 'finished_sequence_selected']
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['stage1'], ['YOU'])
        self.assertTrue(any(e.get('ignored_after_reset') for e in history['events']
                            if e['type'] == 'stage2_sequence_update'))
        self.assertTrue(all(t >= 1.4 for _, window in calls[1:] for t in window))

    def test_finish_during_verification_drains_newer_candidate_before_sequence(self):
        with tempfile.TemporaryDirectory() as root:
            proposals, _, calls, history, _ = replay(
                root, True, finish=False, actions={30: 'f'},
                stability_hits=1, verification_delay=18,
                label_boundary=0.5,
            )
        selected = next(e for e in history['events'] if e['type'] == 'finished_sequence_selected')
        self.assertIn('YOU', selected['stage1'])
        self.assertTrue(any(p[0] > 0.5 and p[-1] == 1.45 for p in proposals))
        self.assertGreater(calls[0][0], 30)

    def test_second_finish_is_accepted_while_previous_sentence_is_naturalizing(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, _, history, _ = replay(
                root, True, finish=False, actions={25: 'f', 60: 'f'},
                language_delay=60, stability_hits=1,
            )
        requests = [e for e in history['events'] if e['type'] == 'finish_requested']
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(history['utterances']), 2)
        self.assertTrue(all(u['finish_total_ms'] is not None for u in history['utterances']))

    def test_separate_command_keeps_thresholds_and_can_pace_saved_video(self):
        from scripts.live_reel_continuous_v17 import parser
        args = parser().parse_args(['--realtime-video'])
        baseline = reel.parser().parse_args([])
        self.assertTrue(args.realtime_video)
        self.assertFalse(baseline.realtime_video)
        for name in ('stability_hits', 'commit_hits', 'minimum_score', 'minimum_margin'):
            self.assertEqual(getattr(args, name), getattr(baseline, name))
        self.assertTrue(args.stage2_review_only)
        self.assertFalse(args.no_stage2_arbiter)

    def test_next_candidate_keeps_frames_captured_during_verification(self):
        with tempfile.TemporaryDirectory() as root:
            proposals, verified, _, _, _ = replay(root, True)
        end, completion_frame = verified[0]
        following = next(p for p in proposals if p[0] > end)
        self.assertLess(following[0], (completion_frame - 1) / 20)
        self.assertAlmostEqual(following[0], end + 1 / 20)

    def test_sequence_runs_only_at_finish_and_cannot_replace_transcript(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, calls, history, summary = replay(root, True)
        self.assertTrue(calls)
        self.assertTrue(all(index > 70 for index, _ in calls))
        times = [t for _, window in calls for t in window]
        self.assertEqual(times, list(np.arange(70) / 20))
        selected = next(e for e in history['events'] if e['type'] == 'finished_sequence_selected')
        self.assertEqual(selected['selected'], selected['stage1'])
        self.assertEqual(selected['selection_mode'], 'stage1_with_stage2_review_candidate')
        self.assertEqual(summary['stage2_hypothesis'], ['WRONG', 'WORDS'])

    def test_quit_without_finish_does_not_run_or_hang_on_deferred_windows(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, calls, _, _ = replay(root, True, finish=False)
        self.assertEqual(calls, [])

    def test_sequence_preview_runs_before_finish_without_isolated_proposals(self):
        with tempfile.TemporaryDirectory() as root:
            proposals, _, calls, _, _ = replay(root, True, sequence_preview=True)
            history = json.loads(next(Path(root).glob('*/history.json')).read_text())
        self.assertFalse(proposals)
        self.assertTrue(calls)
        updates = [e for e in history['events'] if e['type'] == 'stage2_sequence_update']
        self.assertTrue(any(not e['after_finish'] for e in updates))
        self.assertTrue(any(e['type'] == 'sequence_prefix_update' for e in history['events']))

    def test_recording_retains_source_timestamps_for_saved_frames(self):
        recorder = reel.SessionRecorder.__new__(reel.SessionRecorder)
        recorder.next_video_seconds = -1
        recorder.args = SimpleNamespace(record_width=8, record_fps=15)
        written = []
        recorder.writer = SimpleNamespace(write=lambda frame: written.append(frame))
        recorder.data = {'video_source_timestamps_seconds': []}
        frame = np.zeros((8, 8, 3), np.uint8)
        for seconds in [0., .03, .2]:
            recorder.write_frame(frame, seconds)
        self.assertEqual(len(written), 2)
        self.assertEqual(recorder.data['video_source_timestamps_seconds'], [0., .2])

    def test_sequence_reset_discards_pending_decoder_results(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, _, history, _ = replay(root, True, finish=False,
                actions={23: 'r', 62: 'f'}, sequence_preview=True)
        self.assertTrue(any(e['type']=='reset' for e in history['events']))
        prefixes = [e for e in history['events'] if e['type']=='sequence_prefix_update']
        self.assertTrue(prefixes)
        self.assertFalse(prefixes[-1]['conflict'])

    def test_disk_queue_preserves_arrays_and_resets(self):
        self.assertTrue(hasattr(reel, 'DeferredWindows'))
        with reel.DeferredWindows() as queue:
            frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
            queue.append([SimpleNamespace(seconds=0.5, frame=frame)])
            queue.append([SimpleNamespace(seconds=1.5, frame=frame + 1)])
            self.assertEqual(len(queue), 2)
            np.testing.assert_array_equal(queue.popleft()[0].frame, frame)
            queue.append([SimpleNamespace(seconds=2.5, frame=frame + 2)])
            self.assertEqual(queue.popleft()[0].seconds, 1.5)
            self.assertEqual(queue.popleft()[0].seconds, 2.5)
            self.assertFalse(queue)
            queue.clear()
            self.assertEqual(queue.file.tell(), 0)
            queue.append([1])
            self.assertEqual(queue.popleft(), [1])
        self.assertTrue(queue.file.closed)

    def test_preview_is_visible_before_commit_and_never_becomes_a_commit(self):
        view = hud.ReelHud(provisional=True)
        result = dict(gloss='HELLO', candidate_gloss='HELLO', accepted=True,
                      model_score=0.7)
        words = []
        original = hud._shadowed
        def record(*args, **kwargs):
            words.append(args[2])
            return original(*args, **kwargs)
        with patch.object(hud, '_shadowed', record):
            view.draw(np.zeros((720, 1280, 3), np.uint8), None, result,
                      reel.StableGlossLock(), True, True, 20, [], [], '', False, None)
        self.assertIn('HELLO?', words)
        self.assertIsNone(view._committed)
        self.assertNotIn('committed_gloss', result)


if __name__ == '__main__':
    unittest.main()
