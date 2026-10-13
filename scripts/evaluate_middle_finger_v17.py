"""False-trigger and recall check for the middle-finger novelty gesture.

Negatives: real signing — the 100 gloss example clips and every recorded desktop
session — through the app's Apple Vision observation path at 20 Hz. Counts frames
in the pose and gesture triggers (0.6 s hold).
Positives: real upright open-palm frames found in those videos with the index,
ring and little fingers folded synthetically toward the palm (geometry only;
not a recording of the gesture).

    venv/bin/python scripts/evaluate_middle_finger_v17.py
"""
from __future__ import annotations

import copy
import glob
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'artifacts/reports/middle_finger_gesture_v17_20261014'


def videos():
    yield from (('gloss_example', p) for p in sorted(glob.glob(str(ROOT / 'artifacts/app_assets/gloss_examples/*.mp4'))))
    yield from (('desktop_session', p) for p in sorted(glob.glob(str(ROOT / 'artifacts/app_sessions/*/session_lowres.mp4'))))


def folded(hand):
    """Fold index, ring and little fingers toward the palm of a real open hand."""
    h = copy.deepcopy(hand)
    xy = h.xy.copy()
    wrist = xy[0]
    for mcp, joints in ((5, (6, 7, 8)), (13, (14, 15, 16)), (17, (18, 19, 20))):
        for k, j in enumerate(joints):
            xy[j] = xy[mcp] + (0.15 + 0.15 * k) * (wrist - xy[mcp]) * 0.8
    h.xy = xy
    return h


def main():
    from scripts.app_shell_v17 import parser
    from scripts.live_stage2_ctc_v17 import observe_stage2_frame
    from scripts.live_reel_stage1_v17 import is_finish_hand
    from scripts.gesture_extras_v17 import MiddleFingerGesture, is_middle_finger_hand
    from active.v17.extract_v17 import AppleVisionDetector
    args = parser().parse_args([])
    detector = AppleVisionDetector(args.minimum_point_confidence)
    rows, positives, t0 = [], {'open_palm_frames': 0, 'detected_after_folding': 0}, time.time()
    for kind, path in videos():
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.
        wrists = {'left': None, 'right': None}
        gesture, frames, pose, triggers, deadline, i, processed = MiddleFingerGesture(), 0, 0, 0, 0., 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            seconds = i / fps; i += 1
            if seconds + 1e-6 < deadline:
                continue
            deadline = max(deadline + 1 / 20, seconds)
            obs = observe_stage2_frame(frame, seconds, processed, detector, wrists, args)
            processed += 1; frames += 1
            pose += any(is_middle_finger_hand(obs.assigned.get(s)) for s in ('left', 'right'))
            triggers += gesture.update(obs.assigned, seconds)
            for s in ('left', 'right'):
                hand = obs.assigned.get(s)
                if is_finish_hand(hand):
                    positives['open_palm_frames'] += 1
                    positives['detected_after_folding'] += is_middle_finger_hand(folded(hand))
        cap.release()
        rows.append({'kind': kind, 'video': str(Path(path).relative_to(ROOT)), 'frames': frames,
                     'pose_frames': pose, 'triggers': triggers})
        print(rows[-1], flush=True)
    summary = {k: {'videos': sum(r['kind'] == k for r in rows), 'frames': sum(r['frames'] for r in rows if r['kind'] == k),
                   'pose_frames': sum(r['pose_frames'] for r in rows if r['kind'] == k),
                   'triggers': sum(r['triggers'] for r in rows if r['kind'] == k)}
               for k in ('gloss_example', 'desktop_session')}
    result = {'summary': summary, 'synthetic_positives': positives, 'seconds': round(time.time() - t0, 1),
              'videos': rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'result.json').write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != 'videos'}, indent=1))


if __name__ == '__main__':
    main()
