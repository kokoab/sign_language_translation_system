"""Letter arbitration off vs on over the cached 93-clip correctness set (no training, no tuning).

Uses the observation caches written by live_correctness_v17_20260929/final_replay.py, so both
variants see identical Apple Vision observations and hand embeddings.
"""
import gzip, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from active.v17.segmental_runtime_v17 import build_runtime, SpellingBuffer
C = ROOT / 'artifacts/reports/live_correctness_v17_20260929'
torch.set_num_threads(4)
rt = build_runtime(device='cpu')
records = json.loads((C / 'final_replay.json').read_text())['records']
original_logits = rt.recognizer.logits
out = []
for rec in records:
    path = C / 'cache' / (hashlib.sha256(rec['id'].encode()).hexdigest()[:16] + '.pkl.gz')
    with gzip.open(path, 'rb') as f:
        observations, hands = pickle.load(f)
    hand = {id(o): h for o, h in zip(observations, hands)}
    rt.recognizer.frame_hand = lambda obs: hand[id(obs)]
    cache = {}
    def logits(batch):
        keys = [hashlib.sha256(b''.join(np.asarray(v).tobytes() for v in values)).digest() for values in batch]
        miss = [i for i, k in enumerate(keys) if k not in cache]
        if miss:
            for i, row in zip(miss, original_logits([batch[i] for i in miss])): cache[keys[i]] = row
        return np.asarray([cache[k] for k in keys])
    rt.recognizer.logits = logits
    letters = str(rec['expected']).startswith('fs-') and isinstance(rec['expected'], str)
    runs = {}
    for mode in (False, True):
        rt.config['stream']['letter_arbitration'] = mode
        rt.reset()
        speller = SpellingBuffer(minimum=1 if letters else 2)
        result = []
        for obs in observations:
            for w in rt.observe(obs):
                result += speller.push(w)
            result += speller.tick(obs.seconds, active_hands=any(h is not None for h in obs.assigned.values()))
        for w in rt.finish():
            result += speller.push(w)
        result += speller.flush()
        runs['on' if mode else 'off'] = dict(glosses=[w['gloss'] for w in result], revised=speller.revised)
    expected = rec['expected'] if isinstance(rec['expected'], list) else [rec['expected']]
    out.append(dict(id=rec['id'], expected=expected, **{k: v for k, v in runs.items()}))
    print(rec['id'], expected, runs['off']['glosses'], runs['on']['glosses'], flush=True)
scored = [r for r in out if not str(r['id']).startswith('angelo')]
ex = lambda r, k: r[k]['glosses'] == r['expected']
summary = dict(clips=len(out), scored=len(scored), exact_off=sum(ex(r, 'off') for r in scored),
               exact_on=sum(ex(r, 'on') for r in scored),
               fixes=[r['id'] for r in scored if ex(r, 'on') and not ex(r, 'off')],
               regressions=[r['id'] for r in scored if ex(r, 'off') and not ex(r, 'on')],
               changed=[dict(id=r['id'], off=r['off']['glosses'], on=r['on']['glosses']) for r in out
                        if r['off']['glosses'] != r['on']['glosses']])
(Path(__file__).parent / 'arbitration_check.json').write_text(json.dumps(dict(summary=summary, records=out), indent=1))
print(json.dumps(summary, indent=1))
