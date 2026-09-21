import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from active.v17.approved_phrase_data_v17 import verify_manifest, require_training_manifest, digest

class ApprovedManifestTests(unittest.TestCase):
    def test_membership_hash_and_training_gate(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'phrases'; archive=data/'train/source/one.npz'
            archive.parent.mkdir(parents=True); archive.write_bytes(b'cached evidence')
            evidence=root/'evidence.json'; evidence.write_text('{}')
            manifest=root/'manifest.json'
            payload=dict(format='approved_phrase_data_v17', training_ready=False,
                         blockers=['recipe needs approved-source metrics'],
                         roots={'phrase_root':str(data),'other_root':str(root/'other')},
                         evidence_sha256={str(evidence):digest(evidence)},
                         admitted=[dict(path=str(archive),sha256=digest(archive))])
            (root/'other').mkdir()
            def save(): manifest.write_text(json.dumps(payload))
            save(); self.assertEqual(verify_manifest(manifest)['archives'],1)
            args=SimpleNamespace(dataset_manifest=manifest,phrase_root=data,other_root=root/'other',supplement_root=None)
            with self.assertRaisesRegex(ValueError,'not training-ready'):require_training_manifest(args)
            args.phrase_root=root/'old'
            with self.assertRaisesRegex(ValueError,'phrase_root'):require_training_manifest(args)
            args.phrase_root=data
            payload['training_entrypoint']='scripts/train_clean_phrase_baseline_v17.py'; save()
            with self.assertRaisesRegex(ValueError,'dedicated training entrypoint'):require_training_manifest(args)
            payload.pop('training_entrypoint');save()
            archive.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash'):verify_manifest(manifest)
            archive.write_bytes(b'cached evidence'); extra=archive.with_name('extra.npz');extra.write_bytes(b'x')
            with self.assertRaisesRegex(ValueError,'membership'):verify_manifest(manifest)
            extra.unlink(); evidence.write_text('changed')
            with self.assertRaisesRegex(ValueError,'hash'):verify_manifest(manifest)

    def test_entrypoints_gate_before_loading_models(self):
        from unittest.mock import patch
        from active.v17 import train_unified_streaming_ctc_v17 as plain
        from active.v17 import train_unified_streaming_aligned_grounded_v17 as aligned
        for module in (plain,aligned):
            with patch.object(module,'require_training_manifest',side_effect=ValueError('manifest gate')) as gate:
                with self.assertRaisesRegex(ValueError,'manifest gate'): module.run(module.parser().parse_args([]))
                gate.assert_called_once()

    def test_historical_launchers_gate_before_pretraining(self):
        from unittest.mock import patch
        from scripts import train_flores_other_v17 as flores
        from scripts import train_youtube_motion_pilot_v17 as motion
        for module in (flores,motion):
            for function in (module.run,module.preflight):
                with patch.object(module,'require_training_manifest',side_effect=ValueError('manifest gate')) as gate:
                    with self.assertRaisesRegex(ValueError,'manifest gate'): function()
                    gate.assert_called_once()
