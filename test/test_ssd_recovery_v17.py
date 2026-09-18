"""CPU-only regression checks for SSD checkpoint guards and epoch-7 recovery."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


class SSDRecoveryChecks(unittest.TestCase):
    def module(self):
        p=ROOT/'artifacts/reports/stage1_direct_translation_20260917/run_experiment.py'
        s=importlib.util.spec_from_file_location('ssd_baseline_test',p)
        m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

    def test_storage_refuses_missing_volume_and_low_space(self):
        m=self.module()
        self.assertTrue(hasattr(m,'check_checkpoint_space'))
        with tempfile.TemporaryDirectory() as t,patch.dict(os.environ,{'SLT_CHECKPOINT_ROOT':t,'SLT_CHECKPOINT_VOLUME':t}):
            m.MODELS=Path(t)/'models'
            with patch.object(m.os.path,'ismount',return_value=False):
                with self.assertRaises(RuntimeError):m.check_checkpoint_space()
            with patch.object(m.os.path,'ismount',return_value=True):
                with patch.object(m.shutil,'disk_usage',return_value=SimpleNamespace(free=1)):
                    with self.assertRaises(RuntimeError):m.check_checkpoint_space()
                with patch.object(m.shutil,'disk_usage',return_value=SimpleNamespace(free=20*2**30)):
                    m.check_checkpoint_space()

    def test_startup_reserve_is_separate_from_steady_checkpoint_space(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as t,patch.dict(os.environ,{'SLT_CHECKPOINT_ROOT':t,'SLT_CHECKPOINT_VOLUME':''}):
            m.MODELS=Path(t)/'models'
            with patch.object(m.shutil,'disk_usage',return_value=SimpleNamespace(free=6*2**30)):
                m.check_checkpoint_space()
                with self.assertRaises(RuntimeError):m.check_checkpoint_space(startup=True)

    def test_restore_uses_verified_epoch_seven_and_optimizer(self):
        m=self.module();torch=m.torch
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);m.HERE=p;m.MODELS=p
            source=p/'source_manifest.json';source.write_text('{"input_hashes": {}}')
            (p/'manifest.json').write_text(source.read_text())
            model=torch.nn.Linear(2,2);optimizer=torch.optim.SGD(model.parameters(),lr=.25)
            checkpoint=p/'epoch7.pth'
            torch.save(dict(format='slt_stage1_direct_translation_v17',epoch=7,seed=m.SEED,
                manifest_sha256=m.sha256(source),state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),history=[{'epoch':i} for i in range(1,8)]),checkpoint)
            record=p/'recovery.json';record.write_text(json.dumps(dict(checkpoint=str(checkpoint),
                checkpoint_sha256=m.sha256(checkpoint),epoch=7,source_manifest=str(source),manifest_sha256=m.sha256(source))))
            fresh=torch.nn.Linear(2,2);opt=torch.optim.SGD(fresh.parameters(),lr=.01)
            with patch.dict(os.environ,{'SLT_RECOVERY_RECORD':str(record)}),patch.object(m,'release_unused_mps',return_value={}):
                history=m.restore(fresh,opt)
                self.assertEqual(history[-1]['epoch'],7)
                self.assertEqual(opt.param_groups[0]['lr'],.25)
                torch.testing.assert_close(fresh.weight,model.weight)
                record.write_text(record.read_text().replace(m.sha256(checkpoint),'bad-hash'))
                with self.assertRaises(AssertionError):m.restore(fresh,opt)


if __name__=='__main__':unittest.main()
