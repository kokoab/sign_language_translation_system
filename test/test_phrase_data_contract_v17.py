import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from active.v17.train_unified_streaming_ctc_v17 import phrase_sequences, EvidenceSequence, RawSequence, encode
from active.v17.train_unified_streaming_aligned_grounded_v17 import ncslgr_alignments, collate_aligned


class PhraseDataContractTests(unittest.TestCase):
    def test_feasibility_uses_actual_encoder_steps(self):
        from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
        model = SLTStage1V17(Stage1V17Config(num_classes=2, dim=32, depth=1, heads=4)).eval()
        sample = RawSequence(np.ones((1, 32, 61, 5), np.float32), (1, 1), 'probe', 'repeat')
        with torch.inference_mode():
            with self.assertRaises(ValueError):
                encode(model, [sample], torch.device('cpu'), 1, 'window')
            encoded = encode(model, [sample], torch.device('cpu'), 1, 'frame')
        self.assertGreaterEqual(len(encoded[0].evidence), 3)

    def test_loader_rejects_invalid_archives_without_changing_valid_targets(self):
        classes = json.loads(Path('active/v17/citizen100_manifest.json').read_text())['classes']
        label = next(r['canonical_label'] for r in classes if r['class_index'] == 0)
        metadata = dict(role='train', source='probe', source_item_id='probe',
                        schema_fingerprint='b872fa3dcc16aab5',
                        sampled_source_frames=64, target_sequence=[label, '__OTHER__', label])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train/probe/a.npz'
            path.parent.mkdir(parents=True)
            def write(meta=None, **changes):
                arrays = dict(landmarks=np.ones((2, 32, 61, 5), np.float16),
                              window_source_ranges=np.array([[0, 32], [32, 64]]),
                              target_indices=np.array([0, 100, 0]),
                              metadata_json=json.dumps(meta or metadata))
                arrays.update(changes)
                np.savez(path, **arrays)
            write()
            self.assertEqual(phrase_sequences(Path(tmp), 'train', 4, 8)[0].targets, (1, 101, 1))
            with self.assertRaises(ValueError):
                phrase_sequences(Path(tmp), 'train', 4, 8, labels={label: 99})
            write({**metadata, 'schema_fingerprint': 'multimodal-container',
                   'schema': {'landmark_schema_fingerprint': 'b872fa3dcc16aab5'}})
            self.assertEqual(phrase_sequences(Path(tmp), 'train', 4, 8)[0].source_frames, 64)
            cases = [
                ({**metadata, 'schema_fingerprint': 'wrong'}, {}),
                ({**metadata, 'sampled_source_frames': 66, 'dropped_tail_frames': 2}, {}),
                ({**metadata, 'sampled_source_frames': 66}, {}),
                ({**metadata, 'target_sequence': [label, label, label]}, {}),
                (metadata, dict(window_source_ranges=np.array([[0, 32], [8, 40]]))),
                (metadata, dict(window_source_ranges=np.array([[0, 32], [33, 65]]))),
                (metadata, dict(target_indices=np.array([0., 100., .5]))),
                (metadata, dict(landmarks=np.full((2, 32, 61, 5), np.nan))),
                ({**metadata, 'target_sequence': [label] * 9}, dict(target_indices=np.zeros(9, dtype=int))),
            ]
            for meta, changes in cases:
                with self.subTest(meta=meta, fields=list(changes)):
                    write(meta, **changes)
                    with self.assertRaises(ValueError):
                        phrase_sequences(Path(tmp), 'train', 4, 8)
            # Frame-level encoders may yield multiple steps per window; feasibility
            # is then checked on encoded evidence, not the window count.
            self.assertEqual(len(phrase_sequences(Path(tmp), 'train', 4, 8,
                                                  evidence_level='frame')[0].targets), 9)

    def test_alignment_uses_cached_frame_count_and_rejects_cut_sign(self):
        row = dict(source_item_id='clip', has_strict_target=True, source_frame_count=98,
                   events=[dict(canonical_label='A', source_start_frame=88,
                                source_end_frame_exclusive=96)])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            path.write_text(json.dumps({'rows': [row]}))
            aligned = ncslgr_alignments(path, {'A': 0}, 4, 8, {'clip': 96})['clip']
            self.assertEqual(len(aligned), 23)
            self.assertEqual(aligned[-1], 1)
            with self.assertRaises(ValueError):
                ncslgr_alignments(path, {'A': 0}, 4, 8, {'clip': 92})

    def test_padding_has_no_auxiliary_loss_or_gradient(self):
        short = EvidenceSequence(np.ones((2, 3), np.float32), (1,), 'ncslgr_strict', 'short')
        long = EvidenceSequence(np.ones((5, 3), np.float32), (1,), 'probe', 'long')
        alignment = {'short': np.array([1, 0])}
        batch = collate_aligned([short, long], alignment)
        logits = torch.randn(2, 5, 3, requires_grad=True)
        loss = F.cross_entropy(logits.reshape(-1, 3), batch['aligned'].reshape(-1), ignore_index=-100)
        expected = F.cross_entropy(logits[0, :2], torch.tensor([1, 0]))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(logits.grad[0, 2:].abs().sum().item(), 0)
        self.assertEqual(logits.grad[1].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            collate_aligned([short, long], {'short': np.array([1, 0, 0])})


if __name__ == '__main__':
    unittest.main()
