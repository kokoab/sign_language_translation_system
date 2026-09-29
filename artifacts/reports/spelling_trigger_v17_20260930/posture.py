"""Does posture separate the user's fingerspelling from their word signs? (feasibility of an automatic trigger)

Segments: the live app's own timeline in each recorded session (history.json segmental_words):
spelled runs (fs-...) vs word signs. Features from the cached Apple Vision observations of the same
session (same clock): dominant wrist height above the shoulder line and lateral distance from the
same-side shoulder (both in shoulder widths), wrist speed (shoulder widths / s), fraction of frames
with the hand present. No models, no tuning.
"""
import gzip, json, pickle
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
import sys; sys.path.insert(0, str(ROOT))
cache = ROOT / 'artifacts/reports/letter_arbitration_v17_20260929/session_cache'
out = []
for f in sorted(cache.glob('*.pkl.gz')):
    name = f.name.split('.')[0]
    history = json.loads((ROOT / 'artifacts/app_sessions' / name / 'history.json').read_text())
    with gzip.open(f, 'rb') as fh:
        obs, _ = pickle.load(fh)
    t = np.array([o.seconds for o in obs])
    # shoulders: last detection carried forward (body runs every 8th frame)
    sh, last = [], None
    for o in obs:
        c = o.detection.body_confidence
        if c[0] > 0 and c[1] > 0:
            last = o.detection.body_xy[:2].copy()
        sh.append(last)
    for w in history['segmental_words']:
        idx = np.flatnonzero((t >= w['start_seconds']) & (t <= w['end_seconds']))
        if len(idx) < 3:
            continue
        rows = []
        for i in idx:
            o, s = obs[i], sh[i]
            hands = [h for h in (o.assigned['left'], o.assigned['right']) if h is not None and h.confidence[0] > 0]
            if s is None or not hands:
                rows.append(None); continue
            H, W = o.frame.shape[:2]
            px = lambda p: np.array([p[0] * W, p[1] * H])
            ls, rs = px(s[0]), px(s[1]); width = np.linalg.norm(rs - ls)
            if width < 1: rows.append(None); continue
            wrist = max(hands, key=lambda h: -h.xy[0][1])   # the higher hand is the active one
            wr = px(wrist.xy[0])
            near = ls if np.linalg.norm(wr - ls) < np.linalg.norm(wr - rs) else rs
            rows.append(((((ls + rs) / 2)[1] - wr[1]) / width, abs(wr[0] - near[0]) / width, wr / width, o.seconds))
        ok = [r for r in rows if r is not None]
        if len(ok) < 3:
            continue
        pos = np.array([r[2] for r in ok]); ts = np.array([r[3] for r in ok])
        speed = np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.maximum(np.diff(ts), 1e-3)) if len(ok) > 2 else 0
        out.append(dict(session=name, gloss=w['gloss'], spelled=w['gloss'].startswith('fs-'),
                        height=float(np.median([r[0] for r in ok])), lateral=float(np.median([r[1] for r in ok])),
                        speed=float(speed), present=len(ok) / len(rows), seconds=w['end_seconds'] - w['start_seconds']))
spelled = [r for r in out if r['spelled']]; words = [r for r in out if not r['spelled']]
print('segments: spelled', len(spelled), 'words', len(words))
def auc(a, b):   # probability a random spelled segment scores higher than a random word segment
    a, b = np.array(a), np.array(b)
    return float(((a[:, None] > b[None]).mean() + .5 * (a[:, None] == b[None]).mean()))
for k in ('height', 'lateral', 'speed', 'present'):
    a = [r[k] for r in spelled]; b = [r[k] for r in words]
    print(f'{k:8s} spelled median {np.median(a):6.2f} [{np.percentile(a,10):.2f}, {np.percentile(a,90):.2f}]   '
          f'words median {np.median(b):6.2f} [{np.percentile(b,10):.2f}, {np.percentile(b,90):.2f}]   AUC {auc(a, b):.2f}')
by_word = {}
for r in words: by_word.setdefault(r['gloss'], []).append(r['height'])
print('word signs held highest (median height):', sorted(((round(float(np.median(v)), 2), g, len(v)) for g, v in by_word.items()), reverse=True)[:8])
json.dump(out, open(Path(__file__).parent / 'posture.json', 'w'), indent=1)
