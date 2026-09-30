"""Harvest the user's own fingerspelled letters from recorded desktop app sessions.

The user reported which name each spelling attempt was (GELO; ANGELO). Each session video is
replayed through the live runtime; committed letters are grouped into runs (a run ends at a word,
or after a 2 s gap, like the SpellingBuffer) and every run close to a target name is aligned to it
(edit-distance alignment). Letters on a match or substitution position take the target's letter
as their label; inserted letters (no target position) are not used. Each labelled span is saved
with the exact recognizer inputs the live runtime built for it.

Caveat: session videos are saved at 640x360 and ~12-15 fps, while the live app processes
1280x720 at 20 Hz, so these spans approximate (not reproduce) the live inputs.
No model is trained or tuned here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data/local/fingerspelling_letters_v17/user_sessions'


def align(seq, target):
    """Edit-distance alignment; returns (distance, [(index in seq, target letter)])."""
    n, m = len(seq), len(target)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0] = range(n + 1)
    d[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (seq[i - 1] != target[j - 1]))
    pairs, i, j = [], n, m
    while i and j:
        if d[i, j] == d[i - 1, j - 1] + (seq[i - 1] != target[j - 1]):
            pairs.append((i - 1, target[j - 1])); i -= 1; j -= 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return int(d[n, m]), pairs[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sessions', nargs='+', type=Path)
    ap.add_argument('--targets', default='GELO,ANGELO')
    ap.add_argument('--max-distance', type=int, default=2)
    a = ap.parse_args()
    from active.v17.segmental_runtime_v17 import build_runtime
    from scripts.evaluate_temporal_boundary_v17 import arguments
    from scripts.live_stage2_ctc_v17 import observe_stage2_frame
    from active.v17.extract_v17 import AppleVisionDetector
    targets = a.targets.split(',')
    rt = build_runtime()
    args = arguments('data', ROOT / 'artifacts/generated/tmp_sessions')
    summary = []
    for session in a.sessions:
        rt.reset()
        det = AppleVisionDetector(args.minimum_point_confidence)
        history = json.loads((session / 'history.json').read_text())
        stamps = history.get('video_source_timestamps_seconds') or []
        committed = []          # merged output in order: words and letters (with inputs)

        def take(word_out, letter_out):
            for w in rt._merge(word_out, letter_out):
                if w['gloss'].startswith('FS_'):
                    sub = rt.words if w in word_out else rt.letter_rt
                    ids = sub._span_obs(w['start_frame'], w['end_frame'])
                    x = rt.recognizer.inputs([sub.obs[i] for i in ids], [sub.hands[i] for i in ids]) if len(ids) >= 4 else None
                    committed.append(dict(w, inputs=x))
                else:
                    committed.append(dict(w, inputs=None))

        cap = cv2.VideoCapture(str(session / 'session_lowres.mp4'))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        wr, i, deadline, processed, last_t = {'left': None, 'right': None}, 0, 0., 0, None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = stamps[i] if i < len(stamps) else i / fps
            i += 1
            if t + 1e-6 < deadline:
                continue
            if last_t is not None and t - last_t > .26:
                take(rt.words.finish(), rt.letter_rt.finish())
                rt.reset()
            last_t = t
            deadline = max(deadline + 1 / 20, t)
            obs = observe_stage2_frame(frame, t, processed, det, wr, args)
            processed += 1
            word_out = rt.words.observe(obs)
            letter_out = rt.letter_rt.observe(obs, hands=rt.words.hands[-1])
            take(word_out, letter_out)
        take(rt.words.finish(), rt.letter_rt.finish())
        cap.release()
        # Runs like the SpellingBuffer: same letter within .5 s is one letter; a word or a 2 s gap ends a run.
        runs, run = [], []
        for w in committed:
            if not w['gloss'].startswith('FS_'):
                if run: runs.append(run)
                run = []
                continue
            if run and w['start_seconds'] - run[-1]['end_seconds'] > 2.0:
                runs.append(run); run = []
            if run and run[-1]['gloss'] == w['gloss'] and w['start_seconds'] - run[-1]['end_seconds'] <= .5:
                continue
            run.append(w)
        if run: runs.append(run)
        saved = 0
        for k, run in enumerate(runs):
            seq = ''.join(w['gloss'][3:] for w in run)
            dist, target = min((align(seq, t)[0], t) for t in targets)
            if len(seq) < 2 or dist > a.max_distance:
                summary.append(dict(session=session.name, run=seq, target=None))
                continue
            _, pairs = align(seq, target)
            labels = []
            for index, letter in pairs:
                w = run[index]
                labels.append(letter)
                if w['inputs'] is None:
                    continue
                folder = OUT / session.name / letter
                folder.mkdir(parents=True, exist_ok=True)
                x = w['inputs']
                meta = dict(letter=letter, predicted=w['gloss'][3:], name=f'{session.name}_{k:03d}_{index}',
                            role='user_session', session=session.name, target=target, run=seq,
                            start_seconds=w['start_seconds'], end_seconds=w['end_seconds'])
                np.savez_compressed(folder / f'{k:03d}_{index}.npz', landmarks=x[0], hand_embeddings=x[1],
                                    hand_valid=x[2], hand_boxes=x[3], meta=json.dumps(meta))
                saved += 1
            summary.append(dict(session=session.name, run=seq, target=target, distance=dist,
                                aligned=''.join(labels)))
        print(session.name, 'runs', len(runs), 'saved', saved, flush=True)
    (OUT / 'harvest_summary.json').write_text(json.dumps(summary, indent=1))
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
