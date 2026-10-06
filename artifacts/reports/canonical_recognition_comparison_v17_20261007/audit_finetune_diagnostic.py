"""Verify the diagnostic rerun: reproducibility, unchanged selection, trained-weight rotation scores."""
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
FIRST, OUT = ROOT / 'finetune', ROOT / 'finetune_diagnostic'
assert json.loads((OUT / 'status.json').read_text())['state'] == 'complete'
SOURCE_SHA = '5c40b13336b4692d5f7e1e70a9ba430aa2b35ef4e12946952e15fe1f9e54924b'
source = torch.load('/Volumes/secret/SLT/SLT/artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/'
                    'stage1_v17_partwise_v2/best_model.pth', map_location='cpu', weights_only=False)['model_state_dict']


def load_orientation(folder):
    metrics = json.loads((folder / 'metrics.json').read_text())
    with np.load(folder / 'logits.npz') as saved:
        logits, targets, angles, ids = (saved[k] for k in ('logits', 'targets', 'angles_degrees', 'item_ids'))
    assert np.isfinite(logits).all()
    recalculated = [float(100 * (v.argmax(1) == targets).mean()) for v in logits]
    assert np.allclose(recalculated, [v['top1'] for v in metrics['per_angle']])
    return metrics, recalculated, list(angles), ids


reference_ids = load_orientation(FIRST / 'orientation_original')[3]
arms, rows = {}, []
for arm in ('mild_control', 'full_roll'):
    first = json.loads((FIRST / arm / 'history.json').read_text())
    rerun = json.loads((OUT / arm / 'history.json').read_text())
    selected = torch.load(OUT / arm / 'best_model.pth', map_location='cpu', weights_only=False)
    keys = ('top1', 'top5', 'macro_f1', 'loss', 'train_loss')
    arms[arm] = dict(
        trained_epochs=len(rerun),
        rerun_selected_epoch=selected['epoch'],
        rerun_selected_equals_original=set(selected['model_state_dict']) == set(source) and all(
            torch.equal(selected['model_state_dict'][k], v) for k, v in source.items()),
        history_identical_to_first_run=len(first) == len(rerun) and all(
            a[k] == b[k] for a, b in zip(first, rerun) for k in keys),
        max_abs_top1_difference_vs_first_run=max(abs(a['top1'] - b['top1']) for a, b in zip(first, rerun)),
        top1_by_epoch=[r['top1'] for r in rerun])
    for kind in ('final', 'best_trained'):
        name = f'{arm}_{kind}'
        checkpoint = torch.load(OUT / arm / f'{kind}_model.pth', map_location='cpu', weights_only=False)
        expected = rerun[-1] if kind == 'final' else max(rerun, key=lambda r: r['top1'])  # first-best on ties
        assert checkpoint['epoch'] == int(expected['epoch']), (name, checkpoint['epoch'], expected['epoch'])
        assert checkpoint['diagnostic_selection']
        metrics, recalculated, angles, ids = load_orientation(OUT / ('orientation_' + name))
        assert np.array_equal(ids, reference_ids)
        zero = angles.index(0)
        assert abs(recalculated[zero] - checkpoint['validation_metrics']['top1']) < 1e-5
        rows.append(dict(name=name, epoch=checkpoint['epoch'], diagnostic_selection=checkpoint['diagnostic_selection'],
                         upright_top1=recalculated[zero], mean_rotated_top1=metrics['mean_nonzero_top1'],
                         worst_rotated_top1=metrics['worst_nonzero_top1'], per_angle=metrics['per_angle'],
                         checkpoint_sha256=metrics['checkpoint_sha256']))
for name, folder in (('original_96.83', FIRST / 'orientation_original'),
                     ('reference_a7490409_from_scratch', OUT / 'orientation_reference_a7490409')):
    metrics, recalculated, angles, ids = load_orientation(folder)
    assert np.array_equal(ids, reference_ids)
    rows.insert(len(rows) if 'reference' in name else 0,
                dict(name=name, epoch=None, upright_top1=recalculated[angles.index(0)],
                     mean_rotated_top1=metrics['mean_nonzero_top1'], worst_rotated_top1=metrics['worst_nonzero_top1'],
                     per_angle=metrics['per_angle'], checkpoint_sha256=metrics['checkpoint_sha256']))
assert rows[0]['checkpoint_sha256'] == SOURCE_SHA
(OUT / 'audit.json').write_text(json.dumps({'protected_test_accessed': False, 'arms': arms, 'models': rows}, indent=2) + '\n')

angle_labels = [f"{a['angle_degrees']:g}°" for a in rows[0]['per_angle']]
lines = ['# Diagnostic rerun: trained-epoch weights at eight landmark roll angles', '',
         'Identical paired recipe rerun with diagnostic checkpoints. best_model.pth selection unchanged; '
         'diagnostic weights are not promoted. Validation development experiment; no protected test access.', '',
         '| Arm | Trained epochs | Selected epoch | Selected = original | History identical to first run | Max |ΔTop-1| vs first run |',
         '|---|---:|---:|---|---|---:|']
for arm, a in arms.items():
    lines.append(f"| {arm} | {a['trained_epochs']} | {a['rerun_selected_epoch']} | {a['rerun_selected_equals_original']} | "
                 f"{a['history_identical_to_first_run']} | {a['max_abs_top1_difference_vs_first_run']:.2f} |")
lines += ['', '| Weights | Epoch | ' + ' | '.join(angle_labels) + ' | Mean rotated | Worst rotated |',
          '|---|---:|' + '---:|' * (len(angle_labels) + 2)]
for r in rows:
    cells = ' | '.join(f"{a['top1']:.2f}" for a in r['per_angle'])
    lines.append(f"| {r['name']} | {'' if r['epoch'] is None else r['epoch']} | {cells} | "
                 f"{r['mean_rotated_top1']:.2f} | {r['worst_rotated_top1']:.2f} |")
lines += ['', 'best_trained is chosen on upright validation among trained epochs, so its upright score is optimistically selected. '
          'final is the last trained epoch. The a7490409 reference is a separate 138-epoch from-scratch run, not a matched budget.',
          '', 'Rotation diagnostics transform existing landmarks. They do not measure raw-camera extraction or independent phone generalization.']
(OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
