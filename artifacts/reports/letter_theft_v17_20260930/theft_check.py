"""How often does the letter track take words away? Letters on (production v3) vs off (words only).

1. Isolated sign clips: every ASL Citizen *validation* clip of the 100 signs (378; never the official
   test). Each clip has one known sign. Counted per clip: sign found (off / on), exact, spelled output.
2. The user's recorded desktop sessions (cached observations, no transcript): outputs aligned in time,
   words the words-only run found that the letters-on run lost or replaced with a spelled word.
Both runs see identical Apple Vision observations and hand-crop embeddings. No training or tuning.
"""
import gzip, hashlib, json, pickle, sys
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from active.v17.segmental_runtime_v17 import build_runtime, SpellingBuffer
from scripts.evaluate_temporal_boundary_v17 import arguments
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
from active.v17.extract_v17 import AppleVisionDetector

on = build_runtime()
off = build_runtime(fingerspelling=False)
encode_hands = on.recognizer.frame_hand          # the real hand-crop encoder (runs are patched below)
args = arguments('data', ROOT / 'artifacts/generated/tmp_sessions')


def observe(path):
    key = HERE / 'cache' / (hashlib.sha256(str(path).encode()).hexdigest()[:16] + '.pkl.gz')
    if key.exists():
        with gzip.open(key, 'rb') as f:
            return pickle.load(f)
    det = AppleVisionDetector(args.minimum_point_confidence)
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    wr, i, deadline, obs, hands, blank = {'left': None, 'right': None}, 0, 0., [], [], {}
    while True:
        ok, frame = cap.read()
        if not ok: break
        t = i / fps; i += 1
        if t + 1e-6 < deadline: continue
        deadline = max(deadline + .05, t)
        o = observe_stage2_frame(frame, t, len(obs), det, wr, args)
        hands.append(encode_hands(o))
        blank.setdefault(o.frame.shape, np.zeros_like(o.frame)); o.frame = blank[o.frame.shape]
        obs.append(o)
    cap.release()
    with gzip.open(key, 'wb') as f:
        pickle.dump((obs, hands), f)
    return obs, hands


def run(rt, obs, hands):
    table = {id(o): h for o, h in zip(obs, hands)}
    rt.recognizer.frame_hand = lambda o: table[id(o)]
    rt.reset()
    speller, out, last = SpellingBuffer(), [], None
    for o in obs:
        if last is not None and o.seconds - last > .26:
            for w in rt.finish(): out += speller.push(w)
            rt.reset()
        last = o.seconds
        for w in rt.observe(o): out += speller.push(w)
        out += speller.tick(o.seconds, active_hands=any(h is not None for h in o.assigned.values()))
    for w in rt.finish(): out += speller.push(w)
    out += speller.flush()
    return [dict(gloss=w['gloss'], start=round(w['start_seconds'], 2), end=round(w['end_seconds'], 2)) for w in out]


clips = sorted(p for p in (ROOT / 'data/local/citizen100_v17/raw/val').glob('*/*.mp4') if not p.name.startswith('._'))
if '--limit' in sys.argv:
    clips = clips[:int(sys.argv[sys.argv.index('--limit') + 1])]
shard = sys.argv[sys.argv.index('--shard') + 1] if '--shard' in sys.argv else '0/1'
k_shard, n_shard = map(int, shard.split('/'))
clips = clips[k_shard::n_shard]
SUFFIX = '' if n_shard == 1 else f'_{k_shard}of{n_shard}'
SESSIONS = k_shard == 0
rows = []
for k, path in enumerate(clips):
    gloss = path.parent.name
    obs, hands = observe(path)
    r_off, r_on = run(off, obs, hands), run(on, obs, hands)
    g_off, g_on = [w['gloss'] for w in r_off], [w['gloss'] for w in r_on]
    rows.append(dict(clip=str(path.relative_to(ROOT)), sign=gloss, off=g_off, on=g_on,
                     found_off=gloss in g_off, found_on=gloss in g_on,
                     exact_off=g_off == [gloss], exact_on=g_on == [gloss],
                     spelled_on=[g for g in g_on if g.startswith('fs-')]))
    if k % 25 == 0:
        print(k, len(clips), gloss, g_off, g_on, flush=True)
lost = [r for r in rows if r['found_off'] and not r['found_on']]
gained = [r for r in rows if r['found_on'] and not r['found_off']]
per_sign = {}
for r in rows:
    s = per_sign.setdefault(r['sign'], Counter())
    s['clips'] += 1; s['found_off'] += r['found_off']; s['found_on'] += r['found_on']; s['spelled_on'] += bool(r['spelled_on'])
summary = dict(clips=len(rows), found_off=sum(r['found_off'] for r in rows), found_on=sum(r['found_on'] for r in rows),
               exact_off=sum(r['exact_off'] for r in rows), exact_on=sum(r['exact_on'] for r in rows),
               clips_with_spelled_output=sum(bool(r['spelled_on']) for r in rows),
               words_lost_to_letters=[dict(clip=r['clip'], sign=r['sign'], off=r['off'], on=r['on']) for r in lost],
               words_gained_with_letters=[dict(clip=r['clip'], sign=r['sign'], off=r['off'], on=r['on']) for r in gained],
               signs_affected={s: dict(c) for s, c in per_sign.items() if c['found_off'] != c['found_on'] or c['spelled_on']})
# user sessions (cached observations from the letter evaluations)
cache = ROOT / 'artifacts/reports/letter_arbitration_v17_20260929/session_cache'
sessions = {}
for f in sorted(cache.glob('*.pkl.gz')) if SESSIONS else []:
    with gzip.open(f, 'rb') as fh:
        obs, hands = pickle.load(fh)
    r_off, r_on = run(off, obs, hands), run(on, obs, hands)
    words_on = [w for w in r_on if not w['gloss'].startswith('fs-')]
    missing = []
    for w in r_off:
        if not any(v['gloss'] == w['gloss'] and min(v['end'], w['end']) - max(v['start'], w['start']) > -.3 for v in words_on):
            spelled = [v['gloss'] for v in r_on if v['gloss'].startswith('fs-') and min(v['end'], w['end']) - max(v['start'], w['start']) > 0]
            missing.append(dict(word=w['gloss'], at=w['start'], replaced_by=spelled))
    sessions[f.name.split('.')[0]] = dict(words_off=len(r_off), words_on=len(words_on),
                                          spelled_on=[v['gloss'] for v in r_on if v['gloss'].startswith('fs-')],
                                          words_lost=missing)
    print(f.name, 'off words', len(r_off), 'on words', len(words_on), 'lost', [(m['word'], m['replaced_by']) for m in missing], flush=True)
summary['sessions'] = sessions
(HERE / f'theft_check{SUFFIX}.json').write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
print(json.dumps({k: v for k, v in summary.items() if k in ('clips', 'found_off', 'found_on', 'exact_off', 'exact_on', 'clips_with_spelled_output')}))
print('lost:', [(d['sign'], d['on']) for d in summary['words_lost_to_letters']])
print('gained:', [(d['sign'], d['off']) for d in summary['words_gained_with_letters']])
