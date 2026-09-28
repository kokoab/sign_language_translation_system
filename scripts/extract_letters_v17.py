"""Static fingerspelling letters (A-Z) as live span-recognizer inputs, from the local alphabet clips.

Source: data/raw_videos/ASL VIDEOS/{A..Z}. Excluded like the 2026-08-10 audit: MARIAH duplicate
copies and scraped sources (wlasl, signasl, yt), and clips outside 0.4-6 s. Each clip runs through
the live Apple Vision observation path at 20 Hz; the whole clip is one span (landmarks trimmed to
hand activity, per-frame hand crops encoded) exactly as the runtime builds recognizer inputs.

Split by recording session (signer identities are not established): validation = the named
DWIGHT session + the numbered single session; train = the rest. Letters stay separate classes
(FS_A..FS_Z), never merged with lexical signs (letter I is not the sign I/ME).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'data/raw_videos/ASL VIDEOS'
OUT = ROOT / 'data/local/fingerspelling_letters_v17'
LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]


def session_of(name, letter):
    if 'MARIAH' in name or re.match(r'^(wlasl|signasl|yt)', name, re.I):
        return None
    if '__from_' in name:
        return 'named:' + name.split('__from_')[1].split('_')[0]
    if re.match(rf'^{letter}_\d+\.mp4$', name):
        return 'numbered'
    if re.match(r'^[0-9a-f]{8}\.mp4$', name):
        return 'hex_unknown'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-letter-train', type=int, default=0, help='cap train clips per letter (0 = all)')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    a = ap.parse_args()
    from scripts.evaluate_temporal_boundary_v17 import arguments
    from scripts.live_stage2_ctc_v17 import observe_stage2_frame
    from active.v17.extract_v17 import AppleVisionDetector
    from active.v17.segmental_runtime_v17 import build_runtime
    from active.v17.stage1_window_v17 import raw_observation_features
    rt = build_runtime()
    rec = rt.recognizer
    args = arguments('data', OUT / 'sessions')
    det = AppleVisionDetector(args.minimum_point_confidence)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for letter in LETTERS:
        names = sorted(p.name for p in (SOURCE / letter).glob('*.mp4'))
        train = [n for n in names if session_of(n, letter) == 'hex_unknown']
        if a.per_letter_train:
            train = train[:a.per_letter_train]
        val = [n for n in names if (session_of(n, letter) or '').startswith(('named:', 'numbered'))]
        rows += [(letter, n, 'train') for n in train] + [(letter, n, 'validation') for n in val]
    rows = [r for i, r in enumerate(rows) if i % a.shards == a.shard]
    tick = time.perf_counter()
    done = 0
    for letter, name, role in rows:
        target = OUT / 'spans' / role / letter / (name[:-4] + '.npz')
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(SOURCE / letter / name))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        status = dict(letter=letter, name=name, role=role, session=session_of(name, letter))
        if not fps or not (0.4 <= n / fps <= 6.):
            np.savez_compressed(target, skipped=np.asarray('duration'), meta=np.asarray(json.dumps(status)))
            cap.release()
            continue
        obs, wr, i, deadline = [], {'left': None, 'right': None}, 0, 0.
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            i += 1
            if t + 1e-6 < deadline:
                continue
            deadline = max(deadline + 1 / 20, t)
            obs.append(observe_stage2_frame(frame, t, len(obs), det, wr, args))
        cap.release()
        hands = [rec.frame_hand(o) for o in obs]
        x = rec.inputs(obs, hands) if len(obs) >= 4 else None
        if x is None:
            np.savez_compressed(target, skipped=np.asarray('no_usable_hands'), meta=np.asarray(json.dumps(status)))
            continue
        raw, times = raw_observation_features(obs)
        np.savez_compressed(target, landmarks=x[0].astype(np.float16), hand_embeddings=x[1].astype(np.float16),
                            hand_valid=x[2], hand_boxes=x[3].astype(np.float16), raw=raw.astype(np.float16),
                            times=times, meta=np.asarray(json.dumps(status)))
        done += 1
        if done % 100 == 0:
            print('letters shard %d: %d done (%.0fs)' % (a.shard, done, time.perf_counter() - tick), flush=True)
    print('SHARD DONE', a.shard, done, flush=True)


if __name__ == '__main__':
    main()
