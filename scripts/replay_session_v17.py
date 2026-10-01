"""Replay a recorded app session video through the segmental runtime (evaluation only).

Uses the session's own source timestamps and the same Apple Vision path as the app; prints committed
words after the spelling buffer so a config can be compared on the user's real signing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('session', type=Path)
    ap.add_argument('--config', default=None)
    ap.add_argument('--start', type=float, default=0.)
    ap.add_argument('--end', type=float, default=1e9)
    a = ap.parse_args()
    from scripts.evaluate_temporal_boundary_v17 import arguments
    from scripts.live_stage2_ctc_v17 import observe_stage2_frame
    from active.v17.extract_v17 import AppleVisionDetector
    from active.v17.segmental_runtime_v17 import SpellingBuffer, build_runtime
    rt = build_runtime(config=a.config)
    merge = rt.config['stream'].get('moving_merge', True)
    speller = SpellingBuffer(rescore=getattr(rt, 'score_letters', None) if merge else None)
    history = json.loads((a.session / 'history.json').read_text())
    stamps = history.get('video_source_timestamps_seconds') or []
    args = arguments('data', ROOT / 'artifacts/generated/tmp_sessions')
    det = AppleVisionDetector(args.minimum_point_confidence)
    cap = cv2.VideoCapture(str(a.session / 'session_lowres.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    wr, i, deadline, processed, out = {'left': None, 'right': None}, 0, 0., 0, []
    last_t = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = stamps[i] if i < len(stamps) else i / fps
        i += 1
        if t < a.start or t > a.end or t + 1e-6 < deadline:
            continue
        if last_t is not None and t - last_t > .26:
            for w in rt.finish():
                out += speller.push(w)
            rt.reset()
        last_t = t
        deadline = max(deadline + 1 / 20, t)
        for w in rt.observe(observe_stage2_frame(frame, t, processed, det, wr, args)):
            out += speller.push(w)
        out += speller.tick(t)
        processed += 1
    for w in rt.finish():
        out += speller.push(w)
    out += speller.flush()
    print(json.dumps(dict(words=[(w['gloss'], round(w['start_seconds'], 2)) for w in out], merged=speller.merged,
                          dropped=[w['gloss'] for w in speller.dropped])))


if __name__ == '__main__':
    main()
