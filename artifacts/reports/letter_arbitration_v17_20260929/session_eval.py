"""Replay recorded desktop sessions (cached Apple Vision observations) and score spelled GELO/ANGELO.

  session_eval.py cache <session dirs...>                      build observation/hand caches
  session_eval.py score --variant NAME [--arbitration] [--letter-head PATH] <session dirs...>

Scoring: every spelled run is matched to the closest target (GELO / ANGELO); a run is an attempt
when its edit distance to the target is <= 2. Reported: attempts, exact attempts, per-position
letter accuracy (aligned). Session videos are 640x360 at ~12-15 fps (the live app runs 1280x720,
20 Hz), so this approximates live behaviour. No training or tuning here.
"""
import argparse, gzip, hashlib, json, pickle, sys
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).parent
CACHE = HERE / 'session_cache'
from scripts.harvest_user_letters_v17 import align


def cache(sessions):
    from active.v17.segmental_runtime_v17 import build_runtime
    from scripts.evaluate_temporal_boundary_v17 import arguments
    from scripts.live_stage2_ctc_v17 import observe_stage2_frame
    from active.v17.extract_v17 import AppleVisionDetector
    rt = build_runtime()
    args = arguments('data', ROOT / 'artifacts/generated/tmp_sessions')
    CACHE.mkdir(exist_ok=True)
    for session in sessions:
        path = CACHE / f'{session.name}.pkl.gz'
        if path.exists():
            continue
        history = json.loads((session / 'history.json').read_text())
        stamps = history.get('video_source_timestamps_seconds') or []
        det = AppleVisionDetector(args.minimum_point_confidence)
        cap = cv2.VideoCapture(str(session / 'session_lowres.mp4'))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        wr, i, deadline, obs_list, hands, blank = {'left': None, 'right': None}, 0, 0., [], [], {}
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = stamps[i] if i < len(stamps) else i / fps
            i += 1
            if t + 1e-6 < deadline:
                continue
            deadline = max(deadline + 1 / 20, t)
            obs = observe_stage2_frame(frame, t, len(obs_list), det, wr, args)
            hands.append(rt.recognizer.frame_hand(obs))
            blank.setdefault(obs.frame.shape, np.zeros_like(obs.frame))
            obs.frame = blank[obs.frame.shape]
            obs_list.append(obs)
        cap.release()
        with gzip.open(path, 'wb') as f:
            pickle.dump((obs_list, hands), f)
        print('cached', session.name, len(obs_list), flush=True)


def score(sessions, variant, arbitration, letter_head, targets=('GELO', 'ANGELO'), q_to_g=False, fist=False):
    from active.v17.segmental_runtime_v17 import build_runtime, SpellingBuffer
    # PyTorch path for every variant, so a letter head can be swapped in (the Core ML graph embeds one).
    rt = build_runtime(backend='torch', device='mps')
    if letter_head:
        rt.recognizer.attach_letters(Path(letter_head))
        rt.recognizer.letter_threshold = float(rt.config.get('letter_threshold', rt.recognizer.letter_threshold))
    rt.config['stream']['letter_arbitration'] = arbitration
    rt.config['stream']['geometry_q_to_g'] = q_to_g
    rt.config['stream']['fist_geometry'] = fist
    original = rt.recognizer.logits
    report = dict(variant=variant, arbitration=arbitration, letter_head=letter_head, sessions={})
    for session in sessions:
        with gzip.open(CACHE / f'{session.name}.pkl.gz', 'rb') as f:
            obs_list, hands = pickle.load(f)
        hand = {id(o): h for o, h in zip(obs_list, hands)}
        rt.recognizer.frame_hand = lambda obs: hand[id(obs)]
        rt.reset()
        speller = SpellingBuffer()
        out, last = [], None
        for obs in obs_list:
            if last is not None and obs.seconds - last > .26:
                for w in rt.finish():
                    out += speller.push(w)
                rt.reset()
            last = obs.seconds
            for w in rt.observe(obs):
                out += speller.push(w)
            out += speller.tick(obs.seconds, active_hands=any(h is not None for h in obs.assigned.values()))
        for w in rt.finish():
            out += speller.push(w)
        out += speller.flush()
        runs = [w['gloss'][3:] for w in out if w['gloss'].startswith('fs-')]
        attempts, positions = [], {}
        for seq in runs:
            dist, target = min((align(seq, t)[0], t) for t in targets)
            if dist > 2:
                continue
            _, pairs = align(seq, target)
            attempts.append(dict(run=seq, target=target, exact=seq == target))
            # position accuracy over the target letters (missing = wrong)
            aligned = {}
            for i, letter in pairs:
                aligned.setdefault(letter, []).append(seq[i] == letter)
            for letter in target:
                ok = aligned.get(letter, [])
                positions.setdefault(letter, []).append(bool(ok and ok.pop(0)))
        report['sessions'][session.name] = dict(
            output=[w['gloss'] for w in out], attempts=attempts,
            exact=sum(a['exact'] for a in attempts), n=len(attempts),
            letter_accuracy={k: [sum(v), len(v)] for k, v in positions.items()},
            revised=speller.revised)
        print(variant, session.name, f"{sum(a['exact'] for a in attempts)}/{len(attempts)} exact",
              {k: f'{sum(v)}/{len(v)}' for k, v in positions.items()}, [a['run'] for a in attempts], flush=True)
    (HERE / f'session_eval_{variant}.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('cache', 'score'))
    ap.add_argument('sessions', nargs='+', type=Path)
    ap.add_argument('--variant', default='baseline')
    ap.add_argument('--arbitration', action='store_true')
    ap.add_argument('--letter-head', default=None)
    ap.add_argument('--q-to-g', action='store_true')
    ap.add_argument('--fist', action='store_true')
    a = ap.parse_args()
    cache(a.sessions) if a.mode == 'cache' else score(a.sessions, a.variant, a.arbitration, a.letter_head, q_to_g=a.q_to_g, fist=a.fist)
