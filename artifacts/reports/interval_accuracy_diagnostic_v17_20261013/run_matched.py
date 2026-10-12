"""Driver for PLAN.md.  --part primary: preflight pins, R1, arms T and S, then their ablations.
--part ablations_h: interval seeds/ablations for the historical 96.83 chain (independent of primary).
Every stage writes <part>_status.json; a failed ablation is recorded and the driver continues."""
import argparse, hashlib, json, os, subprocess, sys, traceback
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path('/Volumes/secret/SLT/SLT'); HERE = Path(__file__).resolve().parent
PY = str(ROOT / 'venv/bin/python')
H = ROOT / 'artifacts/reports/canonical_recognition_comparison_v17_20261007'
RECIPE = ROOT / 'active/v17/phrase_segment_recipe_manifest_20261007.json'
VIEW = ROOT / 'data/local/phrase_segment_recipe_v17_20261007'
RAW = ROOT / 'artifacts/generated/unfrozen_phrase_adapt_v17'
ARMS = {'transformer': HERE / 'parents/transformer/best_model.pth',
        'squeezeformer': HERE / 'parents/squeezeformer/best_model.pth'}
H_PHRASE = H / 'downstream_recipe/chain_9683/phrase_adapt'
SPAN_CFG = '{"lookahead":8,"span_mode":"both","alpha":0.7,"w_r":2,"w_a":0.25,"w_b":0.5,"c":-1,"log_theta":-1.6094}'
ENV = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
CLI = argparse.ArgumentParser(); CLI.add_argument('--part', choices=('primary', 'ablations_h'), required=True)
CLI = CLI.parse_args(); STATUS = HERE / f'{CLI.part}_status.json'
LOCAL = ['--local-root', 'data/local/local_deep_clean_v17/landmarks/train',
         '--local-manifest', 'data/local/local_deep_clean_v17/train_final_manifest.json',
         '--local-tiers', 'owner_approved_v16_deep_clean', '--approve-local-supplement',
         '--local-validation-root', 'data/local/local_deep_clean_v17/landmarks/val',
         '--local-validation-manifest', 'data/local/local_deep_clean_v17/val_final_manifest.json',
         '--mask-local-mouth-nodes']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


log = {'part': CLI.part, 'stages': {}}
def status(**kw):
    STATUS.write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), **log, **kw}, indent=2) + '\n')


def call(name, cmd, timeout, out):
    status(state='running', stage=name)
    with (out / (name + '.log')).open('w') as f:
        subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    log['stages'][name] = 'complete'


def floors(domains):
    # H's 2026-09-25 rule: no domain more than 1 point below the parent; a domain already below 90 keeps the parent value.
    return {d: (v - 1.0 if v - 1.0 >= 90.0 else v) for d, v in ((d, domains[d]['top1']) for d in ('citizen', 'semlex', 'local'))}


def span(name, base, out, floor, *extra, seed='27927'):
    call(name, [PY, 'scripts/train_span_recognizer_v17.py', '--output-dir', str(out / name), '--base', str(base),
                '--epochs', '8', '--seed', seed, '--sets', 'local_train', '--exclude-prefix-spans',
                '--recipe-manifest', str(RECIPE), '--floors', json.dumps(floor), '--cfg', SPAN_CFG, *extra], 4 * 3600, out)


def ablations(base, out, floor):
    for name, extra, seed in (('interval_seed1', (), '1'), ('interval_seed2', (), '2'),
                              ('interval_freeze_encoders', ('--freeze-encoders',), '27927'),
                              ('interval_no_spans', ('--span-share', '0'), '27927')):
        try:
            span(name, base, out, floor, *extra, seed=seed)
        except Exception:
            log['stages'][name] = 'failed: ' + traceback.format_exc()[-800:]


def preflight():
    verified = subprocess.run([PY, '-m', 'active.v17.approved_phrase_data_v17'], cwd=ROOT, capture_output=True, text=True, check=True)
    canonical = json.loads(verified.stdout[verified.stdout.index('{'):])
    recipe = json.loads(RECIPE.read_text())
    assert canonical['training_ready'] is False and recipe['canonical_sha256'] == canonical['sha256'] and recipe['recipe_training_ready']
    expected = {(e['split'], kind, Path(e[kind]).name): e[kind + '_sha256'] for e in recipe['entries'] for kind in ('rgb', 'hand')}
    found = {}
    for kind, folder in (('rgb', 'multimodal'), ('hand', 'hand')):
        for split in ('train', 'validation'):
            for link in (VIEW / folder / split / 'local_phrases').iterdir(): found[(split, kind, link.name)] = link
    assert set(found) == set(expected), 'recipe view membership differs from manifest'
    for key, link in found.items(): assert sha(link) == expected[key], key
    sys.path.insert(0, str(ROOT))
    from scripts import segmental_lab_v17 as lab
    train = {e['video_sha256'] for e in recipe['entries'] if e['split'] == 'train'}
    rows = [r for r in lab.rows_for('local_train') if r['video_sha256'] in train]; tune = lab.rows_for('tune')
    assert len(rows) == 179 and len(tune) == 89
    span_files = [lab.CACHE / 'train_labels' / (lab.key(r['source_item_id']) + '.json') for r in rows]
    span_files += [lab.memo_path(r) for r in rows + tune] + [lab.CACHE / 'dgs' / (lab.key(r['source_item_id']) + '.npz') for r in tune]
    pins = {'canonical': canonical, 'recipe_manifest_sha256': sha(RECIPE), 'recipe_view_files_verified': len(found),
            'span_inputs': {str(p.relative_to(ROOT)): sha(p) for p in span_files if p.exists()},
            'span_inputs_missing': [str(p.relative_to(ROOT)) for p in span_files if not p.exists()],
            'raw_isolated_cache': {p.name: sha(p) for p in sorted(RAW.glob('*.npz'))},
            'parents': {k: sha(v) for k, v in ARMS.items()},
            'historical_phrase_parent': sha(H_PHRASE / 'best_model.pth'),
            'code_sha256': {p: sha(ROOT / p) for p in ['active/v17/train_stage_1_v17.py', 'active/v17/model_v17.py',
                'active/v17/model_stage1_family_v17.py', 'scripts/benchmark_stage1_families_v17.py',
                'active/v17/train_unified_multimodal_student_v17.py', 'active/v17/train_unified_phrase_adapt_v17.py',
                'active/v17/export_unified_multimodal_coreml_v17.py', 'scripts/train_span_recognizer_v17.py',
                'scripts/evaluate_semlex_citizen100_train_audit.py', 'active/v17/evaluate_orientation_robustness_v17.py',
                'active/v17/train_unfrozen_phrase_adapt_v17.py', 'scripts/segmental_lab_v17.py']},
            'test_accessed': False}
    (HERE / 'pins.json').write_text(json.dumps(pins, indent=1) + '\n')


def evaluate_landmark(name, path, out):
    call('semlex_' + name, [PY, 'scripts/evaluate_semlex_citizen100_train_audit.py', str(path),
         '--feature-root', 'data/local/semlex_citizen100_val_audit/landmarks_v17',
         '--provenance', 'data/local/semlex_citizen100_val_audit/download_provenance.json',
         '--expected-split', 'val', '--output-dir', str(out / ('semlex_' + name))], 1800, out)
    call('orientation_' + name, [PY, '-m', 'active.v17.evaluate_orientation_robustness_v17', str(path),
         '--device', 'mps', '--output-dir', str(out / ('orientation_' + name))], 1800, out)
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    semlex = json.loads((out / ('semlex_' + name) / 'summary.json').read_text())
    orient = json.loads((out / ('orientation_' + name) / 'metrics.json').read_text())
    m = ck['validation_metrics']
    return dict(name=name, path=str(path), sha256=sha(path), epoch=ck['epoch'],
                citizen_correct=int(round(m['top1'] * m['samples'] / 100.0)),
                semlex_correct=int(round(semlex['clip_top1'] * semlex['clips'])),
                local_top1=ck.get('local_validation_metrics', {}).get('top1'),
                worst_angle_correct=min(a['top1_correct'] for a in orient['per_angle'] if a['angle_degrees'] != 0),
                mean_rotated_top1=orient['mean_nonzero_top1'])


def arm(name, parent):
    out = HERE / 'chains' / name; out.mkdir(parents=True, exist_ok=False)
    p = evaluate_landmark('parent', parent, out)
    citizen_floor = p['citizen_correct'] - 5
    train = [PY, '-m', 'active.v17.train_stage_1_v17', '--fine-tune-from', str(parent),
             '--temporal-encoder', 'partwise_global', '--part-depth', '1', '--device', 'mps',
             '--epochs', '80', '--patience', '80', '--batch-size', '64', '--workers', '0', '--seed', '1701',
             '--lr', '0.00005', '--warmup-epochs', '4', '--weight-decay', '0.03', '--label-smoothing', '0.1',
             '--ema-decay', '0.999', '--sampling', 'class_source_balanced',
             '--source-probabilities', 'citizen=.34,semlex=.33,local=.33',
             '--full-roll-probability', '0.35', '--maximum-roll-degrees', '180', '--mild-roll-degrees', '12',
             '--supplement-root', 'data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17',
             '--supplement-manifest', 'data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json',
             '--approve-supplement', *LOCAL, '--citizen-top1-floor-correct', str(citizen_floor),
             '--output', str(out / 'local_replay')]
    (out / 'recipe.json').write_text(json.dumps({'plan': 'PLAN.md', 'parent': p, 'citizen_floor': citizen_floor,
        'citizen_floor_parent_minus_1': p['citizen_correct'] - 1, 'semlex_floor': p['semlex_correct'],
        'worst_angle_floor': p['worst_angle_correct'], 'local_replay_command': train, 'test_accessed': False}, indent=2) + '\n')
    call('local_replay_train', train, 6 * 3600, out)
    result = json.loads((out / 'local_replay/result.json').read_text())
    parent_local = result['training_data_provenance']['initialization']['initial_local_validation_metrics']['top1']
    rows = []
    for n, f in (('promotion_gate', 'best_promotion_gate_model.pth'), ('citizen_best', 'best_model.pth')):
        if not (out / 'local_replay' / f).is_file(): continue
        r = evaluate_landmark(n, out / 'local_replay' / f, out); r['parent_local_top1'] = parent_local
        r['gates'] = dict(citizen=r['citizen_correct'] >= citizen_floor, semlex=r['semlex_correct'] >= p['semlex_correct'],
                          local=r['local_top1'] > parent_local, orientation=r['worst_angle_correct'] >= p['worst_angle_correct'])
        r['eligible'] = all(r['gates'].values()); r['meets_parent_minus_1'] = r['citizen_correct'] >= p['citizen_correct'] - 1
        rows.append(r)
    eligible = [r for r in rows if r['eligible']]; selected = eligible[0] if eligible else rows[0]
    (out / 'selection.json').write_text(json.dumps(dict(candidates=rows, selected=selected['name'], selected_path=selected['path'],
        selected_sha256=selected['sha256'], gate_eligible=bool(eligible)), indent=2) + '\n')
    call('fusion', [PY, '-m', 'active.v17.train_unified_multimodal_student_v17', '--rebuild-cache',
                    '--cache-dir', str(out / 'fusion_cache'), '--output', str(out / 'fusion'),
                    '--citizen-floor-correct', str(selected['citizen_correct']),
                    '--landmark-checkpoint', selected['path']], 7200, out)
    call('phrase_adapt', [PY, '-m', 'active.v17.train_unified_phrase_adapt_v17', '--base', str(out / 'fusion/best_model.pth'),
                          '--isolated-cache', str(out / 'fusion_cache'), '--phrase-rgb', str(VIEW / 'multimodal'),
                          '--phrase-hand', str(VIEW / 'hand'), '--selection-key', 'reel_v2', '--pair-loss-weight', '0',
                          '--pair-sampling-multiplier', '1', '--output-dir', str(out / 'phrase_adapt')], 3600, out)
    floor = floors(json.loads((out / 'phrase_adapt/result.json').read_text())['selected']['domains'])
    span('span_recognizer', out / 'phrase_adapt/best_model.pth', out, floor)
    return out, floor


def run():
    notice = 'ATLAS interval experiment (%s) finished or stopped. Check %s.' % (CLI.part, STATUS.name)
    try:
        if CLI.part == 'primary':
            preflight(); log['stages']['preflight'] = 'complete'
            out = HERE / 'reproduction'; out.mkdir(exist_ok=False)
            hist = json.loads((H / 'downstream_recipe/chain_9683/span_recognizer/history.json').read_text())
            floor = floors(json.loads((H_PHRASE / 'result.json').read_text())['selected']['domains'])
            assert floor == hist['provenance']['floors'], (floor, hist['provenance']['floors'])
            span('interval_r1', H_PHRASE / 'best_model.pth', out, floor)
            done = [arm(name, parent) for name, parent in ARMS.items()]
            status(state='primary complete; ablations running')
            for out, floor in done: ablations(out / 'phrase_adapt/best_model.pth', out, floor)
        else:
            out = HERE / 'ablations_h'; out.mkdir(exist_ok=False)
            floor = floors(json.loads((H_PHRASE / 'result.json').read_text())['selected']['domains'])
            ablations(H_PHRASE / 'best_model.pth', out, floor)
        status(state='complete', promotion='none; development evidence only')
    except Exception:
        status(state='failed', error=traceback.format_exc()); raise
    finally:
        subprocess.run(['osascript', '-e', f'display notification "{notice}" with title "ATLAS experiment"'], check=False)


if __name__ == '__main__':
    run()
