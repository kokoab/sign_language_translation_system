"""Native-clock boundary contracts, including the failure modes of old window gates."""
import importlib.util
import unittest

import numpy as np
import torch


class TemporalBoundaryTest(unittest.TestCase):
    def module(self):
        name = 'active.v17.temporal_boundary_v17'
        self.assertIsNotNone(importlib.util.find_spec(name), 'native temporal boundary implementation missing')
        return __import__(name, fromlist=['*'])

    def test_causal_features_preserve_stationary_wrist_finger_motion(self):
        m = self.module()
        raw = np.zeros((12, 61, 5), np.float32)
        raw[:, :, 3:] = 1
        raw[:, 57, :2] = [-.2, .1]
        raw[:, 58, :2] = [.2, .1]
        raw[:, 9, :2] = [.05, .05]
        raw[:, 4, 0] = np.arange(12) / 100
        times = np.arange(12) / 20
        a = m.boundary_features(raw, times, hand_geometry=True)
        np.testing.assert_array_equal(a[:6], m.boundary_features(raw[:6], times[:6], hand_geometry=True))
        self.assertGreater(float(np.abs(a[4] - a[3]).sum()), 0)
        raw[5, 4, 3:] = 0
        self.assertTrue(np.isfinite(m.boundary_features(raw, times)).all())
        with self.assertRaises(ValueError):
            m.boundary_features(raw, times[::-1])

    def test_touching_signs_and_incomplete_regions_are_not_background(self):
        m = self.module()
        times = np.arange(21) / 20
        y = m.boundary_targets(times, [(0.1, .5), (.5, .9)], [], False)
        self.assertEqual(y[10, 0], 1)
        self.assertEqual(y[10, 1], 1)
        self.assertTrue((y[0] == -1).all())
        y = m.boundary_targets(times, [(.1, .9)], [(.4, .6)], True)
        self.assertTrue((y[8:13] == -1).all())
        self.assertTrue((y[0] == 0).all())

    def test_network_has_exact_bounded_future_context(self):
        m = self.module()
        torch.manual_seed(5)
        model = m.TemporalBoundary(12, lookahead=4).eval()
        x = torch.randn(1, 70, 12)
        with torch.no_grad():
            expected = model(x)
            changed = x.clone(); changed[:, 40:] += 100
            actual = model(changed)
        torch.testing.assert_close(expected[:, :36], actual[:, :36])
        self.assertEqual(expected.shape, (1, 66, 2))
        with torch.no_grad():
            # Fresh chunk with the complete receptive left context is identical.
            chunk = model(x[:, 10:])
        torch.testing.assert_close(expected[:, 40:50], chunk[:, 30:40])

    def test_reviewed_intervals_can_bypass_wrist_trimming_explicitly(self):
        from types import SimpleNamespace
        from scripts.live_isolated_v17 import trim_to_motion
        import inspect
        self.assertIn('enabled', inspect.signature(trim_to_motion).parameters)
        obs = [SimpleNamespace(motion=float(i in (4, 5))) for i in range(12)]
        kept, meta = trim_to_motion(obs, .1, enabled=False)
        self.assertEqual(kept, obs)
        self.assertTrue(meta['motion_trim_disabled'])
        self.assertLess(len(trim_to_motion(obs, .1)[0]), len(obs))

    def test_adjacent_repeats_hold_reset_and_no_fabricated_eof(self):
        m = self.module()
        decoder = m.BoundaryDecoder()
        events = []
        for i in range(31):
            t = i / 20
            p = [float(i in (2, 14)), float(i in (14, 28))]
            events.extend(decoder.update(t, p))
        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(events[0]['end_seconds'], events[1]['start_seconds'])
        decoder.reset()
        self.assertEqual(decoder.update(0, [1, 0]), [])
        self.assertEqual(decoder.update(.5, [0, 0]), [])
        self.assertEqual(decoder.finish(), [])
        with self.assertRaises(ValueError):
            decoder.update(.5, [0, 0])

    def test_stream_probabilities_match_complete_prefix_without_future_leakage(self):
        m = self.module()
        self.assertTrue(hasattr(m, 'BoundaryStream'), 'shared streaming inference missing')
        torch.manual_seed(7)
        raw = np.zeros((80, 61, 5), np.float32)
        raw[..., 3:] = 1
        raw[:, 57, 0], raw[:, 58, 0] = -.2, .2
        raw[:, 4, 0] = np.sin(np.arange(80) / 10) * .1
        clock = np.arange(80) / 20
        model = m.TemporalBoundary().eval()
        stream = m.BoundaryStream(model)
        observed = []
        for r, t in zip(raw, clock):
            result = stream.update(r, t)
            if result is not None:
                observed.append(result['probabilities'])
        with torch.no_grad():
            expected = model(torch.from_numpy(m.boundary_features(raw, clock))[None]).sigmoid()[0].numpy()
        np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-6)
        self.assertLessEqual(len(stream.raw), 26)
        stream.reset()
        self.assertIsNone(stream.update(raw[0], 0))

    def test_occurrence_dedup_cannot_hide_cross_role_parent_leakage(self):
        from scripts.train_temporal_boundary_v17 import event_key
        event = dict(label='A', annotation_start_frame_global=10, annotation_end_frame_global=20)
        train = dict(source='asllrp_contiguous', role='train', source_item_id='asllrp:123.mp4:span00', signer_id='X', identity='parent:event:0', label='A')
        val = {**train, 'role': 'validation'}
        sources = {(r['source'], r['role'], r['source_item_id']): {'intervals': [event]} for r in (train, val)}
        self.assertEqual(event_key(train, sources), event_key(val, sources), 'identity key must expose rather than hide role conflicts')
        # Same frame numbers in another recording are a different occurrence.
        other = {**train, 'source_item_id': 'asllrp:456.mp4:span00'}
        sources[(other['source'], other['role'], other['source_item_id'])] = {'intervals': [event]}
        self.assertNotEqual(event_key(train, sources), event_key(other, sources))

    def test_app_candidate_is_explicit_and_mutually_exclusive(self):
        from scripts.app_shell_v17 import parser
        self.assertIn('--boundary-checkpoint', parser().format_help())
        args = parser().parse_args(['--boundary-checkpoint', 'boundary.pth'])
        self.assertEqual(str(args.boundary_checkpoint), 'boundary.pth')
        from unittest.mock import patch
        with patch('sys.stderr'), self.assertRaises(SystemExit):
            parser().parse_args(['--boundary-checkpoint', 'boundary.pth', '--renz-buffered'])

    def test_full_transcript_counts_do_not_reward_deleting_correct_signs(self):
        from scripts.evaluate_temporal_boundary_v17 import edit_counts, summarize
        reference = ['WORK', 'WORK', 'GO']
        baseline = dict(id='a', reference=reference, hypothesis=['WORK', 'WORK', 'GO', 'NO'])
        baseline['metrics'] = edit_counts(reference, baseline['hypothesis'])
        gated = dict(id='a', reference=reference, hypothesis=['WORK'])
        gated['metrics'] = edit_counts(reference, gated['hypothesis'])
        result = summarize([gated], [baseline])
        self.assertEqual(baseline['metrics']['insertions'], 1)
        self.assertEqual(result['insertions'], 0)
        self.assertEqual(result['deletions'], 2)
        self.assertEqual(result['baseline_correct_events'], 3)
        self.assertEqual(result['retained_baseline_correct_events'], 1)

    def test_runtime_eof_drains_events_without_collapsing_repeated_labels(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch
        from concurrent.futures import Future
        from scripts.app_shell_v17 import parser
        from scripts.live_boundary_v17 import run
        args = parser().parse_args(['--boundary-checkpoint', 'fake.pth', '--video', 'fake.mp4', '--no-display', '--no-speech'])
        frame = np.zeros((16, 24, 3), np.uint8)
        cap = MagicMock(); cap.isOpened.return_value = True; cap.get.return_value = 20.
        cap.read.side_effect = [(True, frame)] * 8 + [(False, None)]
        model = MagicMock(); model.model.lookahead = 4
        def observe(o):
            model.observations = [o]
            return dict(seconds=o.seconds, available_seconds=o.seconds, probabilities=[1, 1],
                        events=[dict(start_seconds=max(0, o.seconds-.2), end_seconds=o.seconds)] if o.seconds in (.2, .3) else [])
        model.observe.side_effect = observe
        model.classify.side_effect = lambda obs, event: {**event, 'committed_gloss': 'WORK'}
        executor = MagicMock()
        def submit(fn, *values):
            future = Future(); future.set_result(fn(*values)); return future
        executor.submit.side_effect = submit
        with patch('scripts.live_boundary_v17.cv2.VideoCapture', return_value=cap), patch('scripts.live_boundary_v17.ThreadPoolExecutor', return_value=executor), patch('scripts.live_isolated_v17.SessionRecorder') as recorder, patch('active.v17.extract_v17.AppleVisionDetector'), patch('scripts.live_stage2_ctc_v17.observe_stage2_frame', side_effect=lambda f, t, *rest: SimpleNamespace(seconds=t)):
            recorder.return_value.data = {'boundary_trace': []}
            result = run(args, model=model)
        self.assertEqual(result['hypothesis'], ['WORK', 'WORK'])
        self.assertEqual(recorder.return_value.add.call_count, 2)
        recorder.return_value.close.assert_called_once()

    def test_repeated_start_without_end_does_not_create_a_completed_sign(self):
        m = self.module()
        decoder = m.BoundaryDecoder()
        self.assertEqual(decoder.update(0, [1, 0]), [])
        self.assertEqual(decoder.update(.2, [0, 0]), [])
        self.assertEqual(decoder.update(.4, [1, 0]), [])

    def test_live_contract_rejects_incompatible_sampling_and_unreachable_vote_count(self):
        import scripts.live_boundary_v17 as live
        from scripts.app_shell_v17 import parser
        self.assertTrue(hasattr(live, 'validate_boundary_args'))
        args = parser().parse_args([])
        live.validate_boundary_args(args)
        args.commit_hits = 2
        with self.assertRaises(ValueError): live.validate_boundary_args(args)
        args.commit_hits = 1; args.dense_model_auxiliary = True
        with self.assertRaises(ValueError): live.validate_boundary_args(args)

    def test_edge_bands_are_order_independent_and_missing_edges_fail_closed(self):
        m = self.module()
        times = np.arange(21) / 20
        events = [(.1, .3), (.33, .6)]
        np.testing.assert_array_equal(m.boundary_targets(times, events, [], False),
                                      m.boundary_targets(times, events[::-1], [], False))
        from scripts import train_temporal_boundary_v17 as trainer
        self.assertTrue(hasattr(trainer, 'validate_event_coverage'))
        trainer.validate_event_coverage(times, events)
        with self.assertRaises(ValueError): trainer.validate_event_coverage(times, [(0., 2.)])


if __name__ == '__main__':
    unittest.main()
