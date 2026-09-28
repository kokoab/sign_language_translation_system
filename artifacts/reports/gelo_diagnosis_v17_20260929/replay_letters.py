"""Replay one desktop session and log every committed letter with its top-3 letter probabilities.

Diagnosis only (no tuning, no training). The saved session video is 640x360 at ~12.6 fps,
so the replay approximates, and does not reproduce, the live 1280x720 / 20 Hz run.
"""
import json, sys
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from active.v17.segmental_runtime_v17 import SpellingBuffer, build_runtime, log_softmax
from scripts.evaluate_temporal_boundary_v17 import arguments
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
from active.v17.extract_v17 import AppleVisionDetector

session = Path(sys.argv[1])
rt = build_runtime()
history = json.loads((session / 'history.json').read_text())
stamps = history['video_source_timestamps_seconds']
args = arguments('data', ROOT / 'artifacts/generated/tmp_sessions')
det = AppleVisionDetector(args.minimum_point_confidence)
speller = SpellingBuffer()
labels = rt.recognizer.letter_classes

def top3(sub, w):
    ids = sub._span_obs(w['start_frame'], w['end_frame'])
    x = rt.recognizer.inputs([sub.obs[i] for i in ids], [sub.hands[i] for i in ids])
    if x is None:
        return None
    p = np.exp(log_softmax(rt.recognizer.logits([x])[0][100:]))
    order = np.argsort(p)[::-1][:3]
    return [(labels[i][3:] if labels[i] != 'NONE' else 'NONE', round(float(p[i]), 3)) for i in order]

letters, out = [], []
cap = cv2.VideoCapture(str(session / 'session_lowres.mp4'))
wr, i, deadline, processed, last_t = {'left': None, 'right': None}, 0, 0., 0, None
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = stamps[i] if i < len(stamps) else i / 15
    i += 1
    if t + 1e-6 < deadline:
        continue
    if last_t is not None and t - last_t > .26:
        for w in rt.finish():
            out += speller.push(w)
        rt.reset()
    last_t = t
    deadline = max(deadline + 1 / 20, t)
    obs = observe_stage2_frame(frame, t, processed, det, wr, args)
    processed += 1
    word_out = rt.words.observe(obs)
    letter_out = rt.letter_rt.observe(obs, hands=rt.words.hands[-1])
    for src, group in (('word_decoder', word_out), ('letter_decoder', letter_out)):
        for w in group:
            if w['gloss'].startswith('FS_'):
                sub = rt.words if src == 'word_decoder' else rt.letter_rt
                letters.append(dict(letter=w['gloss'][3:], source=src, start=round(w['start_seconds'], 2),
                                    end=round(w['end_seconds'], 2), frames=w['end_frame'] - w['start_frame'] + 1,
                                    score=round(w['score'], 3), top3=top3(sub, w)))
    for w in rt._merge(word_out, letter_out):
        out += speller.push(w)
    out += speller.tick(t, active_hands=any(h is not None for h in obs.assigned.values()))
for w in rt.finish():
    out += speller.push(w)
out += speller.flush()
result = dict(session=session.name, live=[w['gloss'] for w in history['segmental_words']],
              replay=[w['gloss'] for w in out], letters=letters)
(Path(__file__).parent / f'{session.name}_letters.json').write_text(json.dumps(result, indent=1))
print(json.dumps(dict(live=result['live'], replay=result['replay'])))
