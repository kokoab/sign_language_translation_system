"""Paired development regression check on existing cached inputs; never reads test data.

Run from repository root with venv/bin/python, arguments: SETS VARIANT[,VARIANT...].
Historical Python inputs are a separate check from the PTS-matched Swift experiment.
"""
import copy
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from active.v17.segmental_runtime_v17 import build_runtime, SegmentalRuntime
from scripts.evaluate_temporal_boundary_v17 import edit_counts

HERE = Path(__file__).resolve().parent
OLD = ROOT / 'artifacts/reports/go_split_root_cause_v17_20261009'
CACHE = Path('/Users/frnzlo/slt_go_split_cache')
torch.set_num_threads(1)


def main(sets, variants):
    names = variants.split(',')
    assert set(names) <= {'baseline', 'noearly_big', 'have_cost2', 'have_cost25', 'cost2'}
    cfg = json.loads((ROOT / 'artifacts/reports/citizen_variant_local_filter_v17_20261008/user_take_replay/config_v2_installed.json').read_text())
    v = json.loads((OLD / 'variants_have_rescue.json').read_text())['have_rescue']
    for k in ('recognizer', 'letter_head'):
        cfg[k] = v[k]
    for k in ('decoder', 'stream'):
        cfg[k].update(v[k])
    config = HERE / 'python_config.json'
    config.write_text(json.dumps(cfg, indent=2))
    built = build_runtime(config, device='mps')
    base = built.words if hasattr(built, 'words') else built
    # The experiment copy differs only by an opt-in per-HAVE segment cost.
    spec = importlib.util.spec_from_file_location('omission_runtime', HERE / 'runtime_experiment.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = ROOT
    runtimes = {}
    for name in names:
        rt = mod.SegmentalRuntime(base.boundary.model, base.recognizer, copy.deepcopy(cfg), mode='both', letter_gate=.95)
        if name != 'baseline':
            rt.early_unsafe.add('BIG')
        if name == 'cost2':
            rt.cfg['c'] = -2.
        if name.startswith('have_cost'):
            rt.cfg['diagnostic_have_cost'] = -2.5 if name.endswith('25') else -2.
        runtimes[name] = rt
    results = {}
    for dataset in sets.split(','):
        assert dataset in {'citizen_val', 'pair_val', 'tune', 'takes', 'takes2', 'takes3', 'takes4', 'live_trace'}
        per = {n: [] for n in names}
        for i, path in enumerate(sorted((CACHE / dataset).glob('*.pkl'))):
            item = pickle.load(path.open('rb'))
            assert '/test/' not in item['meta']['video']
            cap = cv2.VideoCapture(item['meta']['video'])
            ok, frame = cap.read(); cap.release()
            assert ok
            view = np.broadcast_to(np.zeros((1, 1, 1), np.uint8), frame.shape)
            obs, hands = item['obs'], item['hands']
            for o in obs:
                o.frame = view
            if dataset in ('citizen_val', 'pair_val'):
                obs, hands = list(obs), list(hands)
                last = obs[-1]
                for j in range(20):
                    o = copy.copy(last); o.seconds = last.seconds + .05 * (j + 1)
                    obs.append(o); hands.append(hands[-1])
            cachefile = CACHE / (dataset + '_spans_have_rescue') / path.name
            cache = pickle.load(cachefile.open('rb')) if cachefile.exists() else {}
            for name, rt in runtimes.items():
                rt.reset(); rt.raw_cache = cache
                words = []
                for o, h in zip(obs, hands):
                    words += rt.observe(o, hands=h)
                words += rt.finish()
                visible = [w for w in words if not w['gloss'].startswith('FS_')]
                per[name].append(dict(key=path.stem, target=item['meta']['target'], words=visible))
            if i % 50 == 0:
                print(dataset, i, flush=True)
        for name, rows in per.items():
            summary = summarize(dataset, rows)
            results[dataset + ':' + name] = dict(summary=summary, rows=rows)
            print(dataset, name, json.dumps(summary), flush=True)
        (HERE / ('regression_' + sets.replace(',', '_') + '.json')).write_text(json.dumps(results))


def summarize(dataset, rows):
    hyp = [[w['gloss'] for w in r['words']] for r in rows]
    if dataset in ('citizen_val', 'pair_val'):
        targets = [r['target'][0] for r in rows]
        return dict(n=len(rows), correct=sum(t in h for t, h in zip(targets, hyp)),
                    exact=sum(h == [t] for t, h in zip(targets, hyp)), silent=sum(not h for h in hyp),
                    extra=sum(max(0, len(h)-1) for h in hyp),
                    false_have=sum('HAVE' in h and t != 'HAVE' for t, h in zip(targets, hyp)))
    if dataset == 'tune':
        counts = [edit_counts(r['target'], h) for r, h in zip(rows, hyp)]
        totals = {k: sum(c[k] for c in counts) for k in ('substitutions', 'deletions', 'insertions', 'correct', 'references')}
        totals['wer'] = sum(totals[k] for k in ('substitutions', 'deletions', 'insertions')) / totals['references']
        return totals
    return {r['key']: ' '.join(h) for r, h in zip(rows, hyp)}


if __name__ == '__main__':
    main(*sys.argv[1:])
