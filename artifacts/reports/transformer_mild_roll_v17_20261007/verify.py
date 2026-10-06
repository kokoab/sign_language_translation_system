"""Fresh CPU validation and identical landmark rotation diagnostics; no training."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path('/Volumes/secret/SLT/SLT')
sys.path.insert(0, str(ROOT))
from scripts.benchmark_stage1_families_v17 import build_model
from active.v17.train_stage_1_v17 import Citizen100V17Dataset, extractor_schema_fingerprint, rotate_camera_roll_v17
from active.v17.evaluate_orientation_robustness_v17 import DEFAULT_ANGLES, classification_metrics

OUT = Path(__file__).resolve().parent
torch.set_num_threads(1)
recipe = json.loads((OUT / 'recipe.json').read_text())
assert recipe['full_roll_probability'] == 0
for path, expected in recipe['hashes'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, path
result = json.loads((OUT/'transformer/result.json').read_text())
history = result['history']
assert [r['epoch'] for r in history] == list(range(1,len(history)+1))
best = max(history, key=lambda r:r['top1'])
assert best['epoch'] == result['best_epoch']
assert result['epochs_completed'] == result['best_epoch'] + 30
data = Citizen100V17Dataset(ROOT/'data/local/citizen100_v17/landmarks', 'val',
    ROOT/'active/v17/citizen100_manifest.json', ROOT/'data/local/citizen100_v17/rejections.csv',
    expected_schema=extractor_schema_fingerprint('apple'))
assert all('val' in p.parts and 'test' not in p.parts for p in data.files)
batches = list(DataLoader(data,batch_size=64,shuffle=False))
targets = torch.cat([y for x,y in batches]).numpy()
paths = {'no_full_roll':OUT/'transformer/best_model.pth',
         'previous_transformer':ROOT/'artifacts/reports/capstone1_v17_revision_checklist_v1/stage1_family_benchmark/transformer/best_model.pth'}
records = {}
for name,path in paths.items():
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    model = build_model('transformer',100).eval()
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    rows=[]; saved=[]
    with torch.inference_mode():
        for angle in DEFAULT_ANGLES:
            logits = torch.cat([model(rotate_camera_roll_v17(x,angle*np.pi/180)) for x,y in batches]).numpy()
            assert np.isfinite(logits).all()
            rows.append({'angle_degrees':angle,**classification_metrics(logits,targets)})
            saved.append(logits)
    expected = result['validation']['top1'] if name=='no_full_roll' else 95.5026455026455
    assert abs(rows[0]['top1']-expected)<1e-8
    np.savez_compressed(OUT/(name+'_rotation_logits.npz'), logits=np.stack(saved),targets=targets,
                        angles=DEFAULT_ANGLES,item_ids=np.asarray([str(p.relative_to(ROOT)) for p in data.files]))
    records[name]={'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'per_angle':rows,'mean_rotated_top1':float(np.mean([r['top1'] for r in rows[1:]])),
        'worst_rotated_top1':min(r['top1'] for r in rows[1:])}
(OUT/'verification.json').write_text(json.dumps({'protected_test_accessed':False,'models':records},indent=2)+'\n')
lines=['# No-full-roll Transformer result','','Fresh CPU validation matches the saved selection. Training stopped at epoch 77; first best was epoch 47. Explicit full-roll probability 0, mild roll ±12 degrees.','',
'| Checkpoint | Upright Top-1 | Mean rotated Top-1 | Worst rotated Top-1 |','|---|---:|---:|---:|']
for name,r in records.items():
    lines.append(f"| {name} | {r['per_angle'][0]['top1']:.2f}% | {r['mean_rotated_top1']:.2f}% | {r['worst_rotated_top1']:.2f}% |")
lines+=['','Eight angles: 0, 17, 37, 73, 90, 123, 180, 270 degrees. Rotated mean excludes zero. Software landmark diagnostics, not raw-camera or phone accuracy.',
'','Single-seed development comparison. Historical checkpoint augmentation implementation is not immutably recorded, so do not claim an isolated causal augmentation effect. No protected test access, model promotion or manuscript edits.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({n:{'upright':r['per_angle'][0]['top1'],'mean_rotated':r['mean_rotated_top1'],'worst_rotated':r['worst_rotated_top1']} for n,r in records.items()},indent=2))
