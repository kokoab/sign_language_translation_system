"""Selected-checkpoint validation cross-entropy; no training or test access."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

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
receipt = {'scope': 'Epoch-4 primary head, mean unsmoothed classification cross-entropy on identical 3957 saved isolated validation examples. No training or test access.',
           'selected_epoch': 4, 'definition': 'Mean -log softmax(logits)[target], natural logarithm; no label smoothing, KD, rejection or anchor terms.',
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
    predicted = []
    losses = []
    with torch.inference_mode():
        for start in range(0, len(targets), 128):
            inputs = [torch.as_tensor(a[start:start+128], device=device).float() for a in arrays]
            inputs[2] = inputs[2] > .5
            logits = model(*inputs)
            assert torch.isfinite(logits).all()
            cpu = logits.float().cpu()
            target = torch.tensor(targets[start:start+128], dtype=torch.long)
            predicted.extend(cpu.argmax(1).tolist())
            losses.extend(F.cross_entropy(cpu, target, reduction='none').double().tolist())
    assert predicted == domain['predictions']['current_primary']
    total_loss = sum(losses)
    receipt['domains'][name] = {'n': len(targets), 'loss_sum': total_loss,
                               'mean_cross_entropy': total_loss / len(targets),
                               'cache_sha256': sha(path), 'per_example_cross_entropy': losses}
    print(name, total_loss / len(targets), flush=True)
receipt['n'] = sum(d['n'] for d in receipt['domains'].values())
receipt['validation_cross_entropy'] = sum(d['loss_sum'] for d in receipt['domains'].values()) / receipt['n']
assert receipt['n'] == 3957
assert all(sha(OLD / name) == digest for name, digest in preserved.items())
(OUT / 'primary_validation_loss.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('Primary validation CE:', receipt['validation_cross_entropy'], flush=True)
