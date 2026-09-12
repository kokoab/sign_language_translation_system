import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from scripts import evaluate_stage1_window_v17 as evaluation


class WindowEvaluationTests(unittest.TestCase):
    def test_every_gate_is_required_and_boundary_is_inclusive(self):
        metrics = dict(connected_wer=90., connected_insertions=4, connected_deletions=5,
                       familiar_wer=10., citizen_accuracy=.89, semlex_accuracy=.79,
                       transition_false_emissions=2, median_delay_seconds=1.,
                       matched_pool=True, complete_streaming=True, latency_includes_runtime=True)
        baseline = dict(connected_wer=100., connected_insertions=4, connected_deletions=5,
                        familiar_wer=10., citizen_accuracy=.90, semlex_accuracy=.80,
                        transition_false_emissions=2)
        self.assertEqual(evaluation.failed_gates(metrics, baseline), [])
        for key, value in [('connected_wer', 90.1), ('connected_insertions', 5),
                           ('connected_deletions', 6), ('familiar_wer', 10.1),
                           ('citizen_accuracy', .889), ('semlex_accuracy', .789),
                           ('transition_false_emissions', 3), ('median_delay_seconds', 1.01),
                           ('complete_streaming', False), ('latency_includes_runtime', False)]:
            self.assertTrue(evaluation.failed_gates(dict(metrics, **{key: value}), baseline), key)
        self.assertTrue(evaluation.failed_gates({}, baseline))

    def test_latency_keeps_misses_and_distinguishes_repeated_occurrences(self):
        events = [dict(label='A', end_seconds=.3), dict(label='A', end_seconds=1.2),
                  dict(label='B', end_seconds=2.)]
        updates = [dict(seconds=.6, hypothesis=['A']), dict(seconds=1.5, hypothesis=['A', 'A'])]
        result = evaluation.word_latency(events, updates)
        self.assertEqual(result['missed_signs'], 1)
        self.assertEqual(result['total_signs'], 3)
        self.assertEqual(result['first_correct_seconds'], [.6, 1.5, None])

    def test_selection_requires_all_epochs_and_runtime_evidence(self):
        from scripts import select_stage1_window_v17 as selection
        with TemporaryDirectory() as directory:
            root = Path(directory)
            def write(name, value):
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            write('development_freeze.json', dict(status='baselines_frozen', ctc_baseline='ctc.json'))
            write('evaluation_manifest.json', dict(rows=[dict(source_item_id='one', verified_reference_intervals=True)]))
            metric = dict(wer_percent=100., insertions=2, deletions=2)
            write('ctc.json', dict(metrics=dict(asllrp_other_ctc=metric)))
            write('older_reel_familiar/summary.json', metric)
            write('transitions/baseline.json', dict(ctc_false_emissions=2))
            manifest_hash = selection.sha256(root/'evaluation_manifest.json')
            report = dict(rows=[dict(item_id='one')], metrics=dict(
                asllrp_other_ctc=dict(metric, wer_percent=80.), local_phrases=metric),
                isolated_baseline=dict(citizen=.9, semlex=.8), isolated=dict(citizen=.9, semlex=.8),
                checkpoint='candidate', checkpoint_sha256='hash', manifest_sha256=manifest_hash,
                transition_false_emissions=1, complete_streaming=True)
            with patch.object(selection, 'ROOT', root):
                for epoch in range(1, 12):
                    write(f'evaluations/epoch_{epoch:02}.json', dict(report, epoch=epoch))
                with self.assertRaisesRegex(ValueError, 'all 12'):
                    selection.select(root, root/'evaluations')
                write('evaluations/epoch_12.json', dict(report, epoch=12))
                self.assertFalse(selection.select(root, root/'evaluations')['eligible'])
                write('runtime.json', {'hash': dict(median_delay_seconds=.5, complete_annotated_pool=True)})
                self.assertFalse(selection.select(root, root/'evaluations', root/'runtime.json')['eligible'])
                write('runtime.json', {'hash': dict(median_delay_seconds=.5, complete_annotated_pool=True,
                                                   manifest_sha256=manifest_hash, annotated_item_ids=['different'])})
                self.assertFalse(selection.select(root, root/'evaluations', root/'runtime.json')['eligible'])
                write('runtime.json', {'hash': dict(median_delay_seconds=.5, complete_annotated_pool=True,
                                                   manifest_sha256=manifest_hash, annotated_item_ids=['one'])})
                selected = selection.select(root, root/'evaluations', root/'runtime.json')
                self.assertEqual(selected['selected']['epoch'], 1)


if __name__ == '__main__':
    unittest.main()
