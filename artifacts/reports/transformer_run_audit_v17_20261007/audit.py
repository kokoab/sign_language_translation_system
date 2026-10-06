"""Read-only revalidation of historical family runs. Never loads test features."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path('/Volumes/secret/SLT/SLT')
sys.path.insert(0, str(ROOT))
from active.v17.coreml_runtime_v17 import lightweight_imports
lightweight_imports()
import numpy as np
import torch
from scripts import benchmark_stage1_families_v17 as bench

OUT = Path(__file__).resolve().parent
FAMILIES = ['transformer', 'partwise_transformer', 'conv_transformer',
            'anatomical_token_transformer', 'compact_transformer', 'squeezeformer']
torch.set_num_threads(1)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

train, val, classes, data = bench.make_loaders(1701, 64)
assert classes == 100
datasets = [*train.dataset.datasets, val.dataset]
assert datasets[0].split == 'train' and datasets[2].split == 'val'
assert datasets[1].source_name == 'semlex'
files = [p for ds in datasets for p in ds.files]
assert all('test' not in p.parts for p in files)
assert len(set(p.resolve() for p in files)) == len(files)
membership = []
for ds in datasets:
    role = getattr(ds, 'split', 'train_only')
    for p, target in zip(ds.files, ds.targets.tolist()):
        membership.append({'path': str(p.relative_to(ROOT)), 'source': ds.source_name,
                           'split': role, 'target': target, 'sha256': sha(p)})
(OUT / 'current_input_manifest.json').write_text(json.dumps(membership, indent=2) + '\n')
provenance = {}
with (ROOT / 'data/local/citizen100_v17/provenance.csv').open() as handle:
    for row in csv.DictReader(handle):
        if row['split'] in ('train', 'val'):
            key = (row['split'], row['canonical_label'], row['video'])
            assert key not in provenance
            provenance[key] = row
signers, raw_hashes = {}, {}
for ds in [datasets[0], datasets[2]]:
    selected = [provenance[(ds.split, p.parent.name, p.name.removesuffix('.v17.npz') + '.mp4')]
                for p in ds.files]
    signers[ds.split] = sorted({r['participant'] for r in selected})
    raw_hashes[ds.split] = {r['sha256'] for r in selected}
assert not set(signers['train']) & set(signers['val'])
assert not raw_hashes['train'] & raw_hashes['val']
supplement = json.loads(datasets[1].manifest_path.read_text())
assert supplement['split'] == 'train_only'
assert all(r['semlex_split'] == 'train' for r in supplement['videos'])
assert not {r['sha256'] for r in supplement['videos']} & raw_hashes['val']
admission_path = ROOT / 'artifacts/reports/supplement_finalization_v17_20260922/semlex.json'
admission = json.loads(admission_path.read_text())
admitted = {r['feature_path']: r for r in admission['records'] if r['role'] == 'train'}
current = {r['path']: r for r in membership if r['source'] == 'semlex'}
assert set(admitted) == set(current)
assert all(admitted[p]['feature_sha256'] == r['sha256'] for p, r in current.items())
aggregate = json.loads((bench.OUTPUT / 'result.json').read_text())
agg = {r['family']: r for r in aggregate['results']}
results, predictions = [], {}
for family in FAMILIES:
    directory = bench.OUTPUT / family
    recorded = json.loads((directory / 'result.json').read_text())
    history = recorded['history']
    assert recorded['seed'] == 1701 and not recorded['test_accessed']
    assert recorded['data'] == data
    assert [r['epoch'] for r in history] == list(range(1, len(history) + 1))
    first_best = max(history, key=lambda r: r['top1'])
    assert recorded['best_epoch'] == first_best['epoch']
    assert recorded['validation']['top1'] == first_best['top1']
    assert recorded['epochs_completed'] == len(history)
    assert len(history) == 160 or len(history) - first_best['epoch'] == 30
    for row in history:
        e = row['epoch']  # scheduler.step() has already advanced to index e
        factor = (e + 1) / 8 if e < 8 else .02 + .98 * .5 * (1 + math.cos(math.pi * (e - 8) / 152))
        assert math.isclose(row['lr'], 3e-4 * factor, rel_tol=1e-9, abs_tol=1e-12)
    checkpoint = torch.load(directory / 'best_model.pth', map_location='cpu', weights_only=False)
    assert checkpoint['family'] == family and checkpoint['num_classes'] == 100
    model = bench.build_model(family, classes).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    assert sum(p.numel() for p in model.parameters()) == recorded['parameters']
    metrics = bench.evaluate(model, val, torch.device('cpu'))
    for key in ['top1', 'top5', 'macro_f1']:
        assert abs(metrics[key] - recorded['validation'][key]) < 1e-9
        assert metrics[key] == agg[family]['validation'][key]
    with torch.no_grad():
        logits = torch.cat([model(x) for x, _ in val])
        assert torch.isfinite(logits).all()
        x = val.dataset[0][0][None]
        assert torch.isfinite(model(torch.zeros_like(x))).all()
    predictions[family] = logits.argmax(1).tolist()
    # Verify the temporal layers did not remain identical to one another after training.
    layer_difference = None
    if hasattr(model, 'encoder'):
        layers = model.encoder.layers
        layer_difference = float((layers[0].linear1.weight - layers[1].linear1.weight).abs().max())
        assert layer_difference > 1e-6
    result = {'family': family, 'reproduced': metrics, 'recorded': recorded['validation'],
              'best_epoch': recorded['best_epoch'], 'epochs_completed': len(history),
              'checkpoint_sha256': sha(directory / 'best_model.pth'),
              'checkpoint_keys': list(checkpoint), 'parameters': recorded['parameters'],
              'temporal_layers_max_difference': layer_difference}
    results.append(result)
    print(json.dumps({k: result[k] for k in ['family', 'reproduced', 'best_epoch', 'epochs_completed']}), flush=True)
targets = val.dataset.targets.numpy()
t = np.array(predictions['transformer']) == targets
s = np.array(predictions['squeezeformer']) == targets
report = {'protocol': aggregate['protocol'], 'current_data': data, 'citizen_signers': signers,
          'semlex_training_eligible_flag': supplement['training_eligible'],
          'semlex_row_eligibility_flags': sorted({r['training_eligible'] for r in supplement['videos']}),
          'semlex_current_admission': {'matched': True, 'all_feature_hashes_match': True,
              'manifest': str(admission_path.relative_to(ROOT)), 'historical_false_flag_superseded': True},
          'split_checks_passed': True, 'protected_test_loaded': False,
          'results': results, 'predictions': predictions,
          'paired_transformer_squeezeformer': {'transformer_only_correct': int((t & ~s).sum()),
             'squeezeformer_only_correct': int((s & ~t).sum()), 'both_wrong': int((~s & ~t).sum())},
          'current_input_manifest_sha256': sha(OUT / 'current_input_manifest.json'),
          'current_source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
              ROOT / 'scripts/benchmark_stage1_families_v17.py', ROOT / 'active/v17/model_v17.py',
              ROOT / 'active/v17/train_stage_1_v17.py']}}
(OUT / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
print('AUDIT PASSED', flush=True)
