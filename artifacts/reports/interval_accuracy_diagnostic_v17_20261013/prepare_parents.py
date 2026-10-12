"""Re-wrap the two family-benchmark parents as slt_stage1_v17 checkpoints; verify unchanged logits.

No training.  Weights are copied tensor-for-tensor.  The flat Transformer is tagged with its own
architecture (active/v17/model_stage1_family_v17.py); the matched part-wise/global Squeezeformer
keeps the plain SLTStage1V17 config that the family benchmark itself constructed.  Citizen
validation only; the protected test split is never loaded.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
R = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(R))
from active.v17.model_stage1_family_v17 import FlatTransformerStage1Config, build_stage1_landmark_model
from active.v17.model_v17 import Stage1V17Config, make_stage1_checkpoint
from active.v17.train_stage_1_v17 import Citizen100V17Dataset, extractor_schema_fingerprint
from active.v17.train_unfrozen_phrase_adapt_v17 import tensors
from scripts.benchmark_stage1_families_v17 import build_model

O = Path(__file__).resolve().parent
BENCH = R / 'artifacts/reports/capstone1_v17_revision_checklist_v1/stage1_family_benchmark'
PARENT_9683 = R / 'artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/best_model.pth'
MANIFEST = R / 'active/v17/citizen100_manifest.json'
PINNED = {'transformer': '5d1b4e566856ad93e46d804d509e970c3e1b868bb4d85a08abef72392a7d1e03',
          'squeezeformer': '95cdef72e7102c4fbf16808de4984cb3636d5c3c39d2cec3880de48d24d658bd'}
CONFIGS = {'transformer': FlatTransformerStage1Config(num_classes=100).to_dict(),
           'squeezeformer': Stage1V17Config(num_classes=100, temporal_encoder='partwise_global', part_depth=1).to_dict()}
torch.set_num_threads(2)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


ref = torch.load(PARENT_9683, map_location='cpu', weights_only=False)
schema = extractor_schema_fingerprint('apple')
data = Citizen100V17Dataset(R / 'data/local/citizen100_v17/landmarks', 'val', MANIFEST,
                            R / 'data/local/citizen100_v17/rejections.csv', expected_schema=schema)
assert all('test' not in p.parts for p in data.files) and len(data.files) == 378
assert data.label_to_index == ref['label_to_index'] and sha(MANIFEST) == ref['manifest_sha256'] and schema == ref['schema_fingerprint']
assert Stage1V17Config(**ref['model_config']).to_dict() == CONFIGS['squeezeformer'], 'family Squeezeformer differs from 96.83 architecture'
citizen = torch.stack([data[i][0] for i in range(len(data))]); targets = data.targets
raw = {d: tensors(dict(np.load(R / f'artifacts/generated/unfrozen_phrase_adapt_v17/{d}_val.npz', allow_pickle=False)))
       for d in ('citizen', 'semlex', 'local')}
report = {'scope': 'container re-wrap of existing family-benchmark weights; no training; validation only',
          'citizen_test_accessed': False, 'semlex_test_accessed': False, 'parents': {}}
for family, expected in PINNED.items():
    source = BENCH / family / 'best_model.pth'; assert sha(source) == expected, family
    result = json.loads((BENCH / family / 'result.json').read_text()); assert result['test_accessed'] is False
    family_ck = torch.load(source, map_location='cpu', weights_only=False)
    original = build_model(family, 100); original.load_state_dict(family_ck['state_dict'], strict=True); original.eval()
    model = build_stage1_landmark_model(CONFIGS[family]); model.load_state_dict(family_ck['state_dict'], strict=True); model.eval()
    assert list(model.state_dict()) == list(family_ck['state_dict'])
    with torch.inference_mode():
        a = torch.cat([original(citizen[i:i + 64]) for i in range(0, 378, 64)])
        b, emb = zip(*[model(citizen[i:i + 64], return_embeddings=True) for i in range(0, 378, 64)])
        b = torch.cat(b); emb = torch.cat(emb)
        domains = {}
        for d, v in raw.items():
            pred = torch.cat([model(v[0][i:i + 256]) for i in range(0, len(v[4]), 256)]).argmax(1)
            domains[d] = {'n': len(v[4]), 'correct': int((pred == v[4]).sum())}
    identical = bool(torch.equal(a, b)); correct = int((b.argmax(1) == targets).sum())
    assert identical and abs(100 * correct / 378 - result['validation']['top1']) < 1e-9, (family, identical, correct)
    metrics = {'top1': 100 * correct / 378, 'top5': float(100 * (b.topk(5, 1).indices == targets[:, None]).any(1).float().mean()),
               'samples': 378, 'loss': float(torch.nn.functional.cross_entropy(b, targets))}
    ck = make_stage1_checkpoint(model, {k: v.clone() for k, v in family_ck['state_dict'].items()}, epoch=result['best_epoch'],
                                validation_metrics=metrics, label_to_index=data.label_to_index,
                                manifest_sha256=sha(MANIFEST), schema_fingerprint=schema)
    ck['architecture'] = CONFIGS[family].get('architecture', 'slt_stage1_v17_partwise_global')
    ck['training_data_provenance'] = {
        'citizen_test_accessed': False, 'semlex_test_accessed': False,
        'source': 'stage1 family benchmark (seed 1701, Citizen+SemLex 2,863 balanced, full roll 0.35 default)',
        'family_checkpoint': str(source.relative_to(R)), 'family_checkpoint_sha256': expected,
        'family_result_sha256': sha(BENCH / family / 'result.json'), 're_wrap_only': True}
    out = O / 'parents' / family; out.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out / 'best_model.pth')
    reload = build_stage1_landmark_model(torch.load(out / 'best_model.pth', map_location='cpu', weights_only=False)['model_config'])
    reload.load_state_dict(torch.load(out / 'best_model.pth', map_location='cpu', weights_only=False)['model_state_dict'], strict=True)
    with torch.inference_mode():
        assert torch.equal(torch.cat([reload.eval()(citizen[i:i + 64]) for i in range(0, 378, 64)]), a)
    report['parents'][family] = {'source_sha256': expected, 'wrapped': str((out / 'best_model.pth').relative_to(R)),
        'wrapped_sha256': sha(out / 'best_model.pth'), 'model_config': CONFIGS[family], 'citizen_correct': correct,
        'recorded_top1': result['validation']['top1'], 'logits_bitwise_identical_to_benchmark_class': identical,
        'embedding_dim': int(emb.shape[1]), 'raw_cache_domains': domains}
    print(family, report['parents'][family]['citizen_correct'], domains, flush=True)
(O / 'parents' / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print('COMPLETE')
