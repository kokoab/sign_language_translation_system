"""Bounded CPU checks for event-based queueing and non-truncating targets."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

PATH=Path(__file__).resolve().parents[1]/'artifacts/reports/english_comparison_20260917/run_comparison.py'


class ComparisonChecks(unittest.TestCase):
    def module(self):
        self.assertTrue(PATH.exists(), 'approved comparison runner must exist')
        spec=importlib.util.spec_from_file_location('english_comparison',PATH)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_event_wait_observes_exit_without_polling(self):
        m=self.module()
        p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(.25)'])
        m.wait_for_exit(p.pid)
        self.assertEqual(p.wait(timeout=1),0)
        m.wait_for_exit(p.pid)  # Already gone must not block.

    def test_target_bucket_covers_every_token_and_rejects_overflow(self):
        m=self.module()
        self.assertEqual([m.target_bucket(n) for n in [1,16,17,33,65,128]], [16,16,32,48,96,128])
        with self.assertRaises(ValueError):m.target_bucket(129)
        with self.assertRaises(ValueError):m.target_bucket(0)

    def test_bart_labels_preserve_bos_eos_and_mask_only_added_padding(self):
        import torch
        from types import SimpleNamespace
        from transformers import AutoTokenizer
        m=self.module()
        tokenizer=AutoTokenizer.from_pretrained(str(m.ASSETS/'bart-base'),local_files_only=True)
        b=SimpleNamespace(DEVICE='cpu',release_unused_mps=lambda:{})
        m.configure_bart(b,torch)
        rows=[{'reference':'Hello.'},{'reference':'I want to go to the store tomorrow.'}]
        labels=b.labels(tokenizer,rows)
        for i,row in enumerate(rows):
            ids=tokenizer(row['reference'])['input_ids']
            self.assertEqual(labels[i,:len(ids)].tolist(),ids)
            self.assertTrue((labels[i,len(ids):]==-100).all())
            self.assertEqual(ids[0],tokenizer.bos_token_id)
            self.assertEqual(ids[-1],tokenizer.eos_token_id)
        with self.assertRaises(ValueError):b.labels(tokenizer,[{'reference':'hello '*150}])


if __name__=='__main__':unittest.main()
