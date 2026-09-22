"""Fixed frozen-feature comparison; cached paired evidence, no live promotion."""
import contextlib
import hashlib
import json
import runpy
import sys
import time
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
OLD = OUT.parent / 'reel_decision_probe_v17_20260922'
SMOKE = OUT.parent / 'shubert_probe_v17_20260922'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def split(rows):
    train = [r for r in rows if r['role']=='train']
    videos = sorted({r['video'] for r in train}, key=lambda v: hashlib.sha256(v.encode()).hexdigest())
    calibration = set(videos[::5])
    val = [r for r in rows if r['role']=='validation']
    assert not set(videos) & {r['video'] for r in val}
    return [r for r in train if r['video'] not in calibration], [r for r in train if r['video'] in calibration], val


def pool(features, clock, start, end):
    selected = features[(clock >= start) & (clock <= end)]
    if len(selected)<4:
        raise ValueError('fewer than four frames in paired interval')
    return np.concatenate([part.mean(axis=0) for part in np.array_split(selected, 4)]).tolist()


def verify_contract():
    from active.v17.approved_phrase_data_v17 import verify_manifest
    verify_manifest()
    recipe = json.loads((OUT/'recipe.json').read_text())
    assert recipe['training_ready'] and recipe['no_promotion']
    for path, expected in recipe['inputs'].items():
        assert digest(ROOT/path)==expected, path
    for path, expected in recipe['weights'].items():
        assert digest(ROOT/path)==expected, path
    return recipe


def extract(recipe, rows):
    smoke = runpy.run_path(str(SMOKE/'smoke.py'))
    manifest = json.loads((ROOT/'data/local/combined_dataset_v17_20260922/manifest.json').read_text())
    sources = {r['video_path']:r for r in manifest['records']}
    videos = sorted({r['video'] for r in rows})
    for index, video in enumerate(videos):
        row = sources[video]
        assert row['role'] in ['train', 'validation'] and row['source']=='asllrp_contiguous'
        assert digest(ROOT/video)==row['video_sha256']
        cache = OUT/'cache'/hashlib.sha256(video.encode()).hexdigest()[:16]
        cache.mkdir(parents=True, exist_ok=True)
        done = cache/'complete.json'
        if done.exists():
            saved = json.loads(done.read_text())
            assert saved['recipe_sha256']==digest(OUT/'recipe.json')
            for name, expected in saved['files'].items():
                assert digest(cache/name)==expected
            continue
        start = time.perf_counter()
        with (cache/'extract.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            smoke['main'](video=video, output=cache, signer_crop=True)
        report = json.loads((cache/'smoke.json').read_text())
        assert report['video_sha256']==row['video_sha256'] and report['finite']
        saved = dict(recipe_sha256=digest(OUT/'recipe.json'), elapsed_seconds=time.perf_counter()-start,
                     files={name:digest(cache/name) for name in ['smoke.json','features.npy','streams.npz']})
        done.write_text(json.dumps(saved, indent=2)+'\n')
        print(json.dumps(dict(completed=index+1, total=len(videos), video=video,
                              seconds=saved['elapsed_seconds'], presence=report['presence'])), flush=True)


def paired_features(rows):
    output = {'shubert':[], 'dino_body':[]}
    for video in sorted({r['video'] for r in rows}):
        cache = OUT/'cache'/hashlib.sha256(video.encode()).hexdigest()[:16]
        info = json.loads((cache/'smoke.json').read_text())
        features = np.load(cache/'features.npy')[0]
        streams = np.load(cache/'streams.npz')
        dino = np.concatenate([streams[key] for key in ['face','left_hand','right_hand','body_posture']], axis=1)
        # Match the original Reel 20 Hz sampling schedule on the decoded source clock.
        source_clock = np.arange(info['frames'])/info['fps']
        keep, deadline = [], 0.
        for i, t in enumerate(source_clock):
            if t+1e-6>=deadline:
                keep.append(i)
                deadline=max(deadline+1/20,t)
        clock = source_clock[keep]
        for row in [r for r in rows if r['video']==video]:
            for arm, values in [('shubert',features),('dino_body',dino)]:
                temporal=pool(values[keep], clock, row['start'], row['end'])
                assert np.isfinite(temporal).all()
                output[arm].append(dict(row, temporal=temporal))
    keys = {(r['video'],r['id'],r['kind']):i for i,r in enumerate(rows)}
    for arm in output:
        output[arm].sort(key=lambda r:keys[r['video'],r['id'],r['kind']])
    return output


def metrics(rows, scores, threshold):
    result = {}
    for kind in ['core','context_100ms','context_250ms','gap']:
        indices = [i for i,r in enumerate(rows) if r['kind']==kind]
        passed = [i for i in indices if scores[i]>=threshold]
        result[kind] = dict(n=len(indices), gate_pass=len(passed), rejected=len(indices)-len(passed),
            conditional_commits=sum(bool(rows[i]['conditional_commit']) for i in passed),
            correct_conditional_commits=sum(bool(rows[i]['conditional_commit'] and rows[i]['verifier']['candidate_gloss']==rows[i]['target']) for i in passed))
    low = [i for i,r in enumerate(rows) if r['kind']=='core' and not r['motion_start_possible_within_window']]
    result['low_motion_core_proxy'] = dict(n=len(low),gate_pass=sum(bool(scores[i]>=threshold) for i in low))
    return result


def evaluate(rows):
    old = runpy.run_path(str(OLD/'run.py'))
    arms = {'landmarks':rows, **paired_features(rows)}
    results, score_rows = {}, []
    for name, data in arms.items():
        fit, cal, val = split(data)
        scores = old['fit_scores'](fit, cal+val)
        threshold = float(min(scores[i] for i,r in enumerate(cal) if r['kind']!='gap'))
        results[name] = dict(threshold=threshold,fit=metrics(fit,old['fit_scores'](fit,fit),threshold),
                            calibration=metrics(cal,scores[:len(cal)],threshold),validation=metrics(val,scores[len(cal):],threshold))
        for r, score in zip(val,scores[len(cal):]):
            score_rows.append(dict(arm=name,id=r['id'],video=r['video'],kind=r['kind'],target=r['target'],
                score=float(score),pass_gate=bool(score>=threshold),conditional_commit=r['conditional_commit'],
                candidate=r['verifier']['candidate_gloss']))
    fit, cal, val = split(rows)
    for name in ['unchanged','confidence']:
        c=np.array([1. if name=='unchanged' else min(r['proposal']['model_score'],r['verifier']['model_score']) for r in cal])
        v=np.array([1. if name=='unchanged' else min(r['proposal']['model_score'],r['verifier']['model_score']) for r in val])
        threshold=0. if name=='unchanged' else float(min(c[i] for i,r in enumerate(cal) if r['kind']!='gap'))
        results[name]=dict(threshold=threshold,calibration=metrics(cal,c,threshold),validation=metrics(val,v,threshold))
    payload=dict(recipe_sha256=digest(OUT/'recipe.json'),arms=results,validation_rows=score_rows,
                 counts={name:{kind:sum(r['kind']==kind for r in group) for kind in ['core','context_100ms','context_250ms','gap']} for name,group in [('fit',fit),('calibration',cal),('validation',val)]},
                 limitations=['Tiny reused-development gap set; oracle intervals; not full-stream WER',
                              'Noncausal full-clip SHuBERT features; no live latency claim',
                              'Fixed ridge readout; no neural weights updated; no live promotion',
                              'Low-motion flag is an existing wrist-start proxy, not measured live recall'])
    (OUT/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
    verify_completed()
    print(json.dumps(dict(counts=payload['counts'],arms=results)),flush=True)


def verify_completed():
    if not (OUT/'results.json').exists():
        return
    current=json.loads((OUT/'results.json').read_text())
    previous=json.loads((OLD/'results.json').read_text())
    assert current['counts']==previous['counts']
    for name, old_name in [('landmarks','temporal_ridge'),('unchanged','unchanged'),('confidence','confidence')]:
        a=current['arms'][name];b=previous['arms'][old_name]
        assert abs(a['threshold']-b['threshold'])<1e-10
        for kind, expected in b['validation'].items():
            assert all(a['validation'][kind][key]==value for key,value in expected.items())
    for arm in current['arms'].values():
        for kind in ['core','context_100ms','context_250ms']:
            assert arm['calibration'][kind]['rejected']==0
            assert arm['validation'][kind]['conditional_commits']<=current['arms']['unchanged']['validation'][kind]['conditional_commits']


if __name__=='__main__':
    recipe=verify_contract()
    rows=json.loads((OLD/'evidence.json').read_text())['results']
    extract(recipe,rows)
    evaluate(rows)
