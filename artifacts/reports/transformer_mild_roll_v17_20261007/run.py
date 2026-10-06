"""Explicit no-full-roll Transformer baseline using the existing family trainer."""
import functools
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path('/Volumes/secret/SLT/SLT')
sys.path.insert(0, str(ROOT))
import torch
from scripts import benchmark_stage1_families_v17 as trainer

OUT = Path(__file__).resolve().parent

def status(**values):
    (OUT / 'status.json').write_text(json.dumps(values, indent=2) + '\n')

def main():
    torch.set_num_threads(1)
    trainer.augment_v17 = functools.partial(trainer.augment_v17,
        full_roll_probability=0.0, maximum_roll_degrees=180.0, mild_roll_degrees=12.0)
    assert trainer.augment_v17.keywords['full_roll_probability'] == 0.0
    paths = ['active/v17/citizen100_manifest.json',
             'data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json',
             'scripts/benchmark_stage1_families_v17.py', 'active/v17/train_stage_1_v17.py']
    recipe = dict(family='transformer', seed=1701, epochs=160, patience=30,
                  batch_size=64, full_roll_probability=0.0, mild_roll_degrees=12.0,
                  optimizer='AdamW lr3e-4 wd.03; warmup8 cosine; EMA.999; label smoothing.1',
                  test_accessed=False, hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                  caveat='Current explicit mild-roll recipe; historical flat Squeezeformer metadata does not pin exact augmentation implementation.')
    (OUT / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
    try:
        status(state='running', full_roll_probability=0.0)
        result = trainer.train_family('transformer', output_root=OUT,
            device=torch.device('mps'), epochs=160, patience=30, batch_size=64, seed=1701)
        status(state='complete', validation=result['validation'], best_epoch=result['best_epoch'])
    except BaseException:
        status(state='failed', error=traceback.format_exc())
        raise
    finally:
        subprocess.run(['osascript', '-e', 'display notification "No-full-roll Transformer run finished. Check status.json." with title "ATLAS"'], check=False)

if __name__ == '__main__':
    main()
