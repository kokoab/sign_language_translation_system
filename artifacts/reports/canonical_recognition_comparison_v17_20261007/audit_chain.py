"""Verify CHAIN_PLAN.md results from saved artifacts and write chain_9683/REPORT.md."""
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/Volumes/secret/SLT/SLT')
HERE = Path(__file__).resolve().parent
import sys
CHAIN, RAW = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'chain_9683'), HERE / 'raw_orientation'
REFERENCE_CHAIN = CHAIN if (CHAIN / 'fusion_reference_august').is_dir() else HERE / 'chain_9683'
for folder in (CHAIN, RAW):
    assert json.loads((folder / 'status.json').read_text())['state'] == 'complete', folder
AUGUST_RAW = ROOT / 'artifacts/reports/stage1_v17_raw_orientation_robustness/augmentation_plus_vision_auto_axis_faceband/metrics.json'
AUGUST_FUSION = ROOT / 'artifacts/models/stage1_v17_unified_multimodal_student_v1/result.json'
AUGUST_LANDMARK_SHA = '12a74a18d71712abf525350e8120e26694d24a62234b4c268639220a37da47a1'
SOURCE_SHA = '5c40b13336b4692d5f7e1e70a9ba430aa2b35ef4e12946952e15fe1f9e54924b'
audit, lines = {'protected_test_accessed': False}, [f'# 96.83% chain ({CHAIN.name}): verified results', '']


def raw_summary(path):
    data = json.loads(path.read_text())
    angles = [float(a) for a in data['angles_degrees_clockwise']]
    per = {}
    for angle in angles:
        rows = [r for r in data['rows'] if r['angle_degrees_clockwise'] == angle]
        recomputed = sum(bool(r['correct']) for r in rows)
        assert recomputed == data['metrics'][str(angle)]['correct'], (path, angle)
        per[angle] = dict(correct=recomputed, attempted=len(rows),
                          extracted=sum(bool(r['extracted']) for r in rows),
                          agreement=sum(bool(r['upright_prediction_agreement']) for r in rows))
    keys = sorted(r['key'] for r in data['rows'] if r['angle_degrees_clockwise'] == 0.0)
    assert data['test_accessed'] is False
    return dict(checkpoint_sha256=data['checkpoint_sha256'], per_angle=per, keys=keys,
                mean_all=float(np.mean([v['correct'] for v in per.values()])))


# Step 2 — raw-pixel roll with Vision auto-orientation.
raw = {'96.83 (5c40b133)': raw_summary(RAW / 'partwise_9683/metrics.json'),
       'a7490409 rerun': raw_summary(RAW / 'reference_a7490409/metrics.json'),
       'a7490409 August record': raw_summary(AUGUST_RAW)}
if (CHAIN / 'raw_orientation_selected/metrics.json').is_file():
    raw['new local branch'] = raw_summary(CHAIN / 'raw_orientation_selected/metrics.json')
reference_keys = raw['a7490409 August record']['keys']
for name, value in raw.items():
    assert value['keys'] == reference_keys, f'{name} evaluated different clips'
assert raw['96.83 (5c40b133)']['checkpoint_sha256'] == SOURCE_SHA
reproduced = ({a: v['correct'] for a, v in raw['a7490409 rerun']['per_angle'].items()}
              == {a: v['correct'] for a, v in raw['a7490409 August record']['per_angle'].items()})
audit['raw_orientation'] = {k: {**{kk: vv for kk, vv in v.items() if kk != 'keys'},
                                'per_angle': {str(a): x for a, x in v['per_angle'].items()}} for k, v in raw.items()}
audit['raw_orientation_reference_reproduced'] = reproduced
angles = list(raw['a7490409 August record']['per_angle'])
lines += ['## Step 2 — rotated video with automatic orientation correction', '',
          'Same 100 Citizen validation clips (one per class) for every row; pixels rotated on an expanded canvas, '
          'Vision re-run with auto-orientation. Correct out of 100.', '',
          '| Model | ' + ' | '.join(f'{a:g}°' for a in angles) + ' | 8-angle mean |',
          '|---|' + '---:|' * (len(angles) + 1)]
for name, value in raw.items():
    lines.append(f'| {name} | ' + ' | '.join(str(value['per_angle'][a]['correct']) for a in angles)
                 + f" | {value['mean_all']:.2f}% |")
lines += ['', f'a7490409 rerun reproduces the August per-angle record exactly: **{reproduced}**.', '']

# Step 3a — local replay.
recipe = json.loads((CHAIN / 'recipe.json').read_text())
result = json.loads((CHAIN / 'local_replay/result.json').read_text())
history = json.loads((CHAIN / 'local_replay/history.json').read_text())
init = result['training_data_provenance']['initialization']
assert init['sha256'] == SOURCE_SHA and init['strict_state_dict'] is True
assert round(init['initial_validation_metrics']['top1'] * 378 / 100) == 366
assert result['test_evaluated'] is False
selection = json.loads((CHAIN / 'selection.json').read_text())
for row in selection['candidates']:
    checkpoint = torch.load(row['path'], map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == row['epoch']
    assert round(checkpoint['validation_metrics']['top1'] * 378 / 100) == row['citizen_correct']
    expected = dict(citizen=row['citizen_correct'] >= recipe['citizen_floor'],
                    semlex=row['semlex_correct'] >= recipe['semlex_floor'],
                    local=row['local_top1'] > row['parent_local_top1'],
                    orientation=row['worst_angle_correct'] >= 2)
    assert expected == row['gates'] and row['eligible'] == all(expected.values())
    orientation = json.loads((CHAIN / f"orientation_{row['name']}/metrics.json").read_text())
    row['per_angle_correct'] = [a['top1_correct'] for a in orientation['per_angle']]
eligible = [r for r in selection['candidates'] if r['eligible']]
assert selection['selected'] == (eligible[0] if eligible else selection['candidates'][0])['name']
audit['local_replay'] = dict(epochs_completed=len(history), parent_initial_local_top1=init['initial_local_validation_metrics']['top1'],
                             best_citizen_epoch_top1=max(r['top1'] for r in history), selection=selection)
august_landmark = dict(epoch=21, citizen_correct=361, semlex_correct=860, local_top1=96.33977900552486,
                       worst_angle_correct=356, per_angle_correct=[361, 362, 361, 357, 357, 356, 356, 358])
lines += ['## Step 3a — local replay from 96.83', '',
          f"Epochs completed: {len(history)} (patience {recipe.get('patience', '20')}, full-roll probability "
          f"{recipe.get('full_roll_probability', '0.35')}). Parent initial masked local Top-1: "
          f"{init['initial_local_validation_metrics']['top1']:.2f}%. Gates: Citizen ≥{recipe['citizen_floor']}/378, "
          f"SemLex ≥{recipe['semlex_floor']}/978, local > parent, worst landmark-roll angle ≥2/378.", '',
          '| Candidate | Epoch | Citizen | SemLex | Local masked | Worst rotated angle | Mean rotated | Gates passed |',
          '|---|---:|---:|---:|---:|---:|---:|---|']
for r in selection['candidates']:
    passed = ', '.join(k for k, v in r['gates'].items() if v) or 'none'
    lines.append(f"| {r['name']} | {r['epoch']} | {r['citizen_correct']}/378 ({100*r['citizen_correct']/378:.2f}%) | "
                 f"{r['semlex_correct']}/978 ({100*r['semlex_correct']/978:.2f}%) | {r['local_top1']:.2f}% | "
                 f"{r['worst_angle_correct']}/378 | {r['mean_rotated_top1']:.2f}% | {passed} |")
lines.append(f"| August branch (a7490409 parent) | 21 | 361/378 (95.50%) | 860/978 (87.93%) | 96.34% | 356/378 | — | reference |")
lines += ['', f"Selected for fusion: **{selection['selected']}** (gate-eligible: {selection['gate_eligible']}).", '']

# Step 3b — fusion.
def fusion_summary(folder, landmark_sha):
    data = json.loads((folder / 'result.json').read_text())
    for cache in folder.parent.glob(folder.name + '_cache/*.npz'):
        with np.load(cache, allow_pickle=False) as payload:
            meta = json.loads(str(payload['metadata_json']))
        assert meta['landmark_checkpoint_sha256'] == landmark_sha, cache
        assert not meta['citizen_test_accessed'] and not meta['semlex_test_accessed'] and not meta['local_test_accessed']
    v = data['validation_metrics']
    return dict(seed=data['selected_seed'], epoch=data['selected_epoch'],
                citizen=v['citizen']['top1_correct'], semlex=v['semlex']['top1_correct'], local=v['local']['top1_correct'],
                seeds=[dict(seed=c['seed'], epoch=c['selected_epoch'], citizen=c['domain_metrics']['citizen']['top1_correct'],
                            semlex=c['domain_metrics']['semlex']['top1_correct'], local=c['domain_metrics']['local']['top1_correct'])
                       for c in data['candidate_results']])
august = json.loads(AUGUST_FUSION.read_text())
fusion = {}
fusion['August record'] = dict(seed=august['selected_seed'], epoch=august['selected_epoch'],
                               citizen=august['validation_metrics']['citizen']['top1_correct'],
                               semlex=august['validation_metrics']['semlex']['top1_correct'],
                               local=august['validation_metrics']['local']['top1_correct'])
fusion['August rerun (fresh cache)'] = fusion_summary(REFERENCE_CHAIN / 'fusion_reference_august', AUGUST_LANDMARK_SHA)
fusion['New (96.83 chain)'] = fusion_summary(CHAIN / 'fusion', selection['selected_sha256'])
fusion_reproduced = all(fusion['August rerun (fresh cache)'][k] == fusion['August record'][k]
                        for k in ('seed', 'epoch', 'citizen', 'semlex', 'local'))
audit['fusion'] = fusion
audit['fusion_reference_reproduced'] = fusion_reproduced
lines += ['## Step 3b — multimodal fusion and distillation', '',
          'Hand branch, four-stream teacher, data, seeds and selection rule unchanged; caches rebuilt and their '
          'landmark-encoder hashes verified.', '',
          '| Run | Seed | Epoch | Citizen | SemLex | Local |', '|---|---:|---:|---:|---:|---:|']
for name, f in fusion.items():
    lines.append(f"| {name} | {f['seed']} | {f['epoch']} | {f['citizen']}/378 ({100*f['citizen']/378:.2f}%) | "
                 f"{f['semlex']}/978 ({100*f['semlex']/978:.2f}%) | {f['local']}/2896 ({100*f['local']/2896:.2f}%) |")
lines += ['', f'August fusion rerun reproduces the recorded selection exactly: **{fusion_reproduced}**.', '',
          'Per-seed (new chain): ' + '; '.join(f"seed {s['seed']} epoch {s['epoch']}: {s['citizen']}/{s['semlex']}/{s['local']}"
                                                for s in fusion['New (96.83 chain)']['seeds']), '',
          '## Not run', '', 'Phrase-activity adaptation and the interval recognizer: canonical phrase verifier reports '
          'training_ready=false.', '',
          'Validation development evidence only. SemLex validation reuses SemLex train signers; local validation is '
          'familiar-signer. Landmark/pixel rotation is not phone evidence. No test access, promotion or manuscript change.']
(CHAIN / 'audit.json').write_text(json.dumps(audit, indent=2, default=str) + '\n')
(CHAIN / 'REPORT.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
