"""Top-5 on the exact saved primary-head validation membership; no test access."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from active.v17.export_unified_multimodal_coreml_v17 import load_model

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / '20261020_primary_metrics'
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

preserved = {p.name: sha(p) for p in OLD.iterdir() if p.is_file()}
source = ROOT / 'artifacts/reports/current_lineage_components_v17_20261013/results.json'
saved = json.loads(source.read_text())
pin = saved['models']['current_primary']
checkpoint = Path(pin['path'])
assert sha(checkpoint) == pin['sha256']
model, _ = load_model(checkpoint)
torch.set_num_threads(2)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = model.to(device).eval()
cache = ROOT / 'artifacts/reports/citizen_variant_local_filter_v17_20261008/run_v2/raw_cache_citizen_variant'
receipt = {'scope': 'Selected primary head, identical 3957 saved isolated validation examples. No training or test access.',
           'source': str(source.relative_to(ROOT)), 'source_sha256': sha(source), 'model': pin,
           'domains': {}, 'original_chart_files_sha256': preserved}
for name, domain in saved['domains'].items():
    path = cache / f'{name}_val.npz'
    with np.load(path, allow_pickle=False) as z:
        ids = z['item_ids'].tolist()
        assert len(ids) == len(set(ids))
        index = {v: i for i, v in enumerate(ids)}
        take = np.array([index[i] for i in domain['ids']])
        arrays = [z[k][take] for k in ['landmarks', 'hand_embeddings', 'hand_valid', 'hand_boxes']]
        targets = z['targets'][take]
    assert targets.tolist() == domain['targets']
    top5 = []
    with torch.inference_mode():
        for start in range(0, len(targets), 128):
            inputs = [torch.as_tensor(a[start:start+128], device=device).float() for a in arrays]
            inputs[2] = inputs[2] > .5
            logits = model(*inputs)
            assert torch.isfinite(logits).all()
            top5.extend(logits.topk(5, dim=1).indices.cpu().tolist())
    assert [p[0] for p in top5] == domain['predictions']['current_primary']
    correct = sum(int(t in p) for t, p in zip(targets, top5))
    receipt['domains'][name] = {'n': len(targets), 'correct_top5': correct,
                               'cache_sha256': sha(path), 'top5_indices': top5}
    print(name, correct, '/', len(targets), flush=True)
receipt['n'] = sum(d['n'] for d in receipt['domains'].values())
receipt['correct_top5'] = sum(d['correct_top5'] for d in receipt['domains'].values())
receipt['top5_percent'] = 100 * receipt['correct_top5'] / receipt['n']
assert receipt['n'] == 3957
assert all(sha(OLD / name) == digest for name, digest in preserved.items())
(OUT / 'primary_top5.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('Primary Top-5:', receipt['top5_percent'], flush=True)
