import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.train_stage_1_v17 import build_parser, initialize_exact_stage1_finetune


class LegacyFineTuneConfigTest(unittest.TestCase):
    def test_defaults_are_compatible_but_changed_behavior_is_rejected(self):
        source = SLTStage1V17(Stage1V17Config(num_classes=2, dim=32, depth=1))
        restored = SLTStage1V17(source.config)
        config = source.config.to_dict()
        for key in ['canonicalize_camera_roll', 'static_hand_token', 'use_attention_score_mixing']:
            del config[key]
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / 'manifest.json'
            manifest.write_text('{}')
            path = Path(folder) / 'checkpoint.pth'
            payload = dict(format='slt_stage1_v17', model_config=config,
                model_state_dict=source.state_dict(), label_to_index={'a':0,'b':1},
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                schema_fingerprint='schema',
                training_data_provenance={'citizen_test_accessed':False,'semlex_test_accessed':False})
            torch.save(payload,path)
            initialize_exact_stage1_finetune(restored,path,manifest,'schema',{'a':0,'b':1})
            for key,value in source.state_dict().items():
                torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
            payload['model_config']['canonicalize_camera_roll'] = True
            torch.save(payload,path)
            with self.assertRaisesRegex(ValueError,'model config mismatch'):
                initialize_exact_stage1_finetune(restored,path,manifest,'schema',{'a':0,'b':1})


class DiagnosticCheckpointFlagTest(unittest.TestCase):
    def test_diagnostic_checkpoints_are_opt_in(self):
        parser = build_parser()
        self.assertFalse(parser.parse_args([]).save_diagnostic_checkpoints)
        self.assertTrue(parser.parse_args(['--save-diagnostic-checkpoints']).save_diagnostic_checkpoints)


if __name__ == '__main__':
    unittest.main()
