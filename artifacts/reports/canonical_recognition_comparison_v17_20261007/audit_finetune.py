"""Verify completed selections against their actual weights and rotation logits."""
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'finetune'
assert json.loads((OUT / 'status.json').read_text())['state'] == 'complete'
source = Path('/Volumes/secret/SLT/SLT/artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/best_model.pth')
original = torch.load(source, map_location='cpu', weights_only=False)
rows = []
reference_ids = None
for name in ('original', 'mild_control', 'full_roll'):
    path = source if name == 'original' else OUT / name / 'best_model.pth'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    metrics = json.loads((OUT / ('orientation_' + name) / 'metrics.json').read_text())
    with np.load(OUT / ('orientation_' + name) / 'logits.npz') as saved:
        logits, targets, angles, ids = (saved[k] for k in ('logits', 'targets', 'angles_degrees', 'item_ids'))
    assert np.isfinite(logits).all()
    if reference_ids is None:
        reference_ids = ids.copy()
    assert np.array_equal(ids, reference_ids)
    recalculated = [float(100 * (v.argmax(1) == targets).mean()) for v in logits]
    assert np.allclose(recalculated, [v['top1'] for v in metrics['per_angle']])
    zero = list(angles).index(0)
    assert abs(recalculated[zero] - checkpoint['validation_metrics']['top1']) < 1e-5
    state, reference = checkpoint['model_state_dict'], original['model_state_dict']
    same = set(state) == set(reference) and all(torch.equal(state[k], reference[k]) for k in reference)
    history = [] if name == 'original' else json.loads((OUT / name / 'history.json').read_text())
    rows.append(dict(name=name, selected_epoch=checkpoint['epoch'], weights_equal_original=same,
                     upright_top1=recalculated[zero], mean_rotated_top1=metrics['mean_nonzero_top1'],
                     worst_rotated_top1=metrics['worst_nonzero_top1'],
                     trained_epochs=len(history), best_trained_epoch_top1=max((r['top1'] for r in history), default=None),
                     final_trained_epoch_top1=history[-1]['top1'] if history else None,
                     per_angle=metrics['per_angle'], checkpoint_sha256=metrics['checkpoint_sha256']))
(OUT / 'audit.json').write_text(json.dumps({'protected_test_accessed': False, 'models': rows}, indent=2) + '\n')
lines = ['# Paired fine-tuning verification', '',
         'Same starting checkpoint, data, seed and 20-epoch budget. Validation development experiment; no protected test access.', '',
         '| Selection | Epoch | Upright Top-1 | Mean rotated Top-1 | Worst rotated Top-1 | Original weights? | Best / final trained epoch Top-1 |',
         '|---|---:|---:|---:|---:|---|---|']
fmt = lambda v: 'n/a' if v is None else f'{v:.2f}%'
for r in rows:
    lines.append(f"| {r['name']} | {r['selected_epoch']} | {r['upright_top1']:.2f}% | {r['mean_rotated_top1']:.2f}% | {r['worst_rotated_top1']:.2f}% | {r['weights_equal_original']} | {fmt(r['best_trained_epoch_top1'])} / {fmt(r['final_trained_epoch_top1'])} |")
lines += ['', 'Selection maximizes upright validation accuracy and retains epoch 0 when there is no strict improvement. A retained original is not evidence that orientation training improved the model.',
          '', 'Rotation diagnostics transform existing landmarks. They do not measure raw-camera extraction or independent phone generalization.',
          '', 'Full per-angle scores, actual selected weight identity and best trained-epoch accuracy are recorded in audit.json. No manuscript or deployment change.']
(OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(rows, indent=2))
