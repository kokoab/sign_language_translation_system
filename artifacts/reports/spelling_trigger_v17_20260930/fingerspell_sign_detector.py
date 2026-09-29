"""Draft detector for the ASL sign FINGERSPELL and its false-trigger rate on existing signing.

FINGERSPELL (ASL-LEX fingerspell B_01_048; HandSpeak): dominant '5' hand, palm down, slides sideways
once while the fingers wiggle. Detector over a sliding 1.0 s window (20 Hz observations):
  open5   >= 70% of frames: index..pinky tips farther from the wrist than their PIP joints (x1.15)
  wiggle  finger-tip motion relative to the palm (palm-normalised, per second) above WIGGLE
  slide   net horizontal wrist travel >= SLIDE palm lengths, at least twice the vertical travel
  single  the other hand is not also open-5 (rules out WAIT and the two-palm Finish gesture)
Negatives only (no FINGERSPELL videos are available locally): every cached ASL Citizen validation
clip of the 100 signs and the user's recorded desktop sessions (including their fingerspelling).
"""
import gzip, json, pickle, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
WIGGLE, SLIDE = 1.5, 1.2


def hand_frame(h):
    if h is None or (h.confidence[[0, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20]] <= 0).any():
        return None
    xy = h.xy.astype(np.float64)
    palm = np.linalg.norm(xy[9] - xy[0])
    if palm < 1e-3:
        return None
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    open5 = all(np.linalg.norm(xy[t] - xy[0]) > 1.15 * np.linalg.norm(xy[p] - xy[0]) for t, p in zip(tips, pips))
    rel = (xy[tips] - xy[0]) / palm
    return dict(open5=open5, rel=rel, wrist=xy[0], palm=palm)


def triggers(obs):
    """Seconds at which the detector fires (after each firing it waits for the window to clear)."""
    frames = []
    for o in obs:
        H, W = o.frame.shape[:2]
        scale = np.array([W, H]) / max(W, H)
        per = {}
        for side in ('left', 'right'):
            f = hand_frame(o.assigned[side])
            if f is not None:
                f['wrist'] = f['wrist'] * scale; f['palm'] *= float(np.mean(scale))
            per[side] = f
        frames.append((o.seconds, per))
    fired, last = [], -1e9
    for i in range(len(frames)):
        t0 = frames[i][0]
        window = [f for f in frames[i:] if f[0] - t0 <= 1.0]
        if len(window) < 8 or t0 < last + 1.0:
            continue
        for side, other in (('left', 'right'), ('right', 'left')):
            hs = [f[1][side] for f in window if f[1][side] is not None]
            if len(hs) < 0.7 * len(window) or np.mean([h['open5'] for h in hs]) < .7:
                continue
            os_ = [f[1][other] for f in window if f[1][other] is not None]
            if os_ and np.mean([h['open5'] for h in os_]) >= .5:
                continue            # both hands open: WAIT / Finish, not FINGERSPELL
            palm = np.median([h['palm'] for h in hs])
            dt = window[-1][0] - window[0][0]
            wig = np.mean([np.abs(b['rel'] - a['rel']).mean() for a, b in zip(hs, hs[1:])]) * len(hs) / max(dt, 1e-3)
            d = (hs[-1]['wrist'] - hs[0]['wrist']) / palm
            if wig >= WIGGLE and abs(d[0]) >= SLIDE and abs(d[0]) >= 2 * abs(d[1]):
                fired.append(round(t0, 2)); last = t0
                break
    return fired


result = {}
theft_cache = ROOT / 'artifacts/reports/letter_theft_v17_20260930/cache'
clip_fires, n = [], 0
for f in sorted(theft_cache.glob('*.pkl.gz')):
    with gzip.open(f, 'rb') as fh:
        obs, _ = pickle.load(fh)
    n += 1
    fire = triggers(obs)
    if fire:
        clip_fires.append(f.name)
result['citizen_clips'] = dict(clips=n, clips_with_false_trigger=len(clip_fires))
sess_cache = ROOT / 'artifacts/reports/letter_arbitration_v17_20260929/session_cache'
total_s, sess = 0.0, {}
for f in sorted(sess_cache.glob('*.pkl.gz')):
    with gzip.open(f, 'rb') as fh:
        obs, _ = pickle.load(fh)
    dur = obs[-1].seconds - obs[0].seconds
    total_s += dur
    sess[f.name.split('.')[0]] = dict(seconds=round(dur, 1), false_triggers=triggers(obs))
result['user_sessions'] = dict(minutes=round(total_s / 60, 1), sessions=sess,
                               false_triggers=sum(len(v['false_triggers']) for v in sess.values()))
result['thresholds'] = dict(wiggle=WIGGLE, slide=SLIDE)
(Path(__file__).parent / 'fingerspell_sign_detector.json').write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1))
