"""Paired development regression check on existing cached inputs; never reads test data.

Run from repository root with venv/bin/python, arguments: SETS VARIANT[,VARIANT...].
Historical Python inputs are a separate check from the PTS-matched Swift experiment.
"""
import copy
import os
import hashlib
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
    cfg = json.loads((ROOT/'artifacts/reports/have_root_fix_v17_20261010/config_fixed.json').read_text())
    built = build_runtime(ROOT/'artifacts/reports/have_root_fix_v17_20261010/config_fixed.json', device='mps')
    base = built.words if hasattr(built, 'words') else built
    spec = importlib.util.spec_from_file_location('runtime_experiment', HERE/'runtime_experiment.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    spec2=importlib.util.spec_from_file_location('runtime_ends', HERE/'runtime_ends.py')
    ends=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(ends)
    spec3=importlib.util.spec_from_file_location('runtime_hybrid', HERE/'runtime_hybrid.py')
    hybrid=importlib.util.module_from_spec(spec3);spec3.loader.exec_module(hybrid)
    from active.v17.export_unified_multimodal_coreml_v17 import load_model
    old,meta=load_model(ROOT/'artifacts/models/span_recognizer_v17_local_a/best_model.pth')
    assert sorted(meta['label_to_index'],key=meta['label_to_index'].get)==list(base.recognizer.labels)
    old.to('mps').eval()
    def old_vote(x):
        batch=[torch.from_numpy(np.asarray(a)[None]).to('mps') for a in x]
        batch[2]=batch[2]>.5
        with torch.inference_mode():return old(*batch)[0].cpu().numpy()
    base.recognizer.experiment_old_vote=old_vote
    runtimes = {}
    for name in names:
        local = copy.deepcopy(cfg)
        stream=local['stream'];stream['experiment_arm']='ordered' if name in ('ordered_ends','hybrid') else name
        if name.startswith('rollback'):
            stream['have_contact_closure']=False
            if name not in ('rollback_geometry','rollback_both'):
                stream.pop('crop_pair',None);stream.pop('handshape_pair',None)
            if name not in ('rollback_roll','rollback_both'):stream.pop('tired_rescue_droop',None)
        runtimes[name] = (hybrid if name=='hybrid' else ends if name=='ordered_ends' else mod).SegmentalRuntime(base.boundary.model, base.recognizer, local, mode='both', letter_gate=.95)
    boundary_forward = base.boundary.model.forward
    boundary_cache = {}
    def cached_boundary(*args):
        key = hashlib.sha256(b''.join(a.cpu().numpy().tobytes() for a in args)).digest()
        if key not in boundary_cache:
            boundary_cache[key] = boundary_forward(*args).detach().clone()
        return boundary_cache[key].clone()
    base.boundary.model.forward = cached_boundary
    original_inputs = base.recognizer.inputs
    input_cache = {}
    def cached_inputs(observations, hands):
        key = tuple((id(o), id(h)) for o,h in zip(observations,hands))
        if key not in input_cache:
            input_cache[key] = original_inputs(observations,hands)
        return input_cache[key]
    base.recognizer.inputs = cached_inputs
    original_logits = base.recognizer.logits
    raw_model_cache = {}
    def cached_model_logits(batch):
        keys = [hashlib.sha256(b''.join(np.asarray(a).tobytes() for a in x)).digest() for x in batch]
        missing = [(k, x) for k, x in zip(keys, batch) if k not in raw_model_cache]
        if missing:
            output = original_logits([x for _, x in missing])
            raw_model_cache.update((k, np.array(l, copy=True)) for (k, _), l in zip(missing, output))
        return np.asarray([raw_model_cache[k].copy() for k in keys])
    base.recognizer.logits = cached_model_logits
    results = {}
    for dataset in sets.split(','):
        assert dataset in {'citizen_val', 'pair_val', 'tune', 'takes', 'takes2', 'takes3', 'takes4', 'live_trace'}
        per = {n: [] for n in names}
        paths = sorted((CACHE / dataset).glob('*.pkl'))
        if os.environ.get('CASES'): paths = [p for p in paths if p.stem in os.environ['CASES'].split(',')]
        for i, path in enumerate(paths):
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
            raw_model_cache.clear(); boundary_cache.clear(); input_cache.clear()
            for name, rt in runtimes.items():
                rt.reset(); rt.raw_cache = {}
                words = []
                for o, h in zip(obs, hands):
                    words += rt.observe(o, hands=h)
                words += rt.finish()
                visible = [w for w in words if not w['gloss'].startswith('FS_')]
                per[name].append(dict(key=path.stem, target=item['meta']['target'], words=visible))
            (HERE/('progress_hybrid_'+dataset+'.json')).write_text(json.dumps(per))
            if i % 25 == 0:
                print(dataset, i, flush=True)
        for name, rows in per.items():
            summary = summarize(dataset, rows)
            results[dataset + ':' + name] = dict(summary=summary, rows=rows)
            print(dataset, name, json.dumps(summary), flush=True)
        (HERE / ('development_hybrid_' + sets.replace(',', '_') + '.json')).write_text(json.dumps(results))


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
