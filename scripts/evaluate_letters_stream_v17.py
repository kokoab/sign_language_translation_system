"""Held-out static letters through the full streaming runtime (Apple Vision, boundary, decoder).

Validation sessions only (DWIGHT + numbered). A clip is correct when its committed words contain
exactly the right FS_ letter (single letters are what a spelled word is built from; the app's
minimum-run rule is applied later). Also reports how often a letter clip yields a lexical word.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--per-letter', type=int, default=12)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--session', default=None, help="'numbered' or 'named:DWIGHT' (default: both)")
    a = ap.parse_args()
    import glob
    from scripts.evaluate_temporal_boundary_v17 import arguments
    from scripts.live_stage2_ctc_v17 import observe_stage2_frame
    from active.v17.extract_v17 import AppleVisionDetector
    from active.v17.segmental_runtime_v17 import build_runtime
    rt = build_runtime(config=a.config)
    args = arguments('data', ROOT / 'artifacts/generated/tmp_sessions')
    det = AppleVisionDetector(args.minimum_point_confidence)
    rows, per = [], Counter()
    for path in sorted(glob.glob(str(ROOT / 'data/local/fingerspelling_letters_v17/spans/validation/*/*.npz'))):
        d = np.load(path)
        if 'landmarks' not in d:
            continue
        meta = json.loads(str(d['meta']))
        if a.session and meta['session'] != a.session:
            continue
        if per[meta['letter']] >= a.per_letter:
            continue
        per[meta['letter']] += 1
        rt.reset()
        cap = cv2.VideoCapture(str(ROOT / 'data/raw_videos/ASL VIDEOS' / meta['letter'] / meta['name']))
        fps = cap.get(cv2.CAP_PROP_FPS)
        wr, i, deadline, words = {'left': None, 'right': None}, 0, 0., []
        processed = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            i += 1
            if t + 1e-6 < deadline:
                continue
            deadline = max(deadline + 1 / 20, t)
            words += rt.observe(observe_stage2_frame(frame, t, processed, det, wr, args))
            processed += 1
        cap.release()
        words += rt.finish()
        # Same letter post-processing as the app (split holds, moving letters), keeping single letters.
        from active.v17.segmental_runtime_v17 import SpellingBuffer
        merge = rt.config['stream'].get('moving_merge', True)
        speller = SpellingBuffer(minimum=1, rescore=getattr(rt, 'score_letters', None) if merge else None)
        final = []
        for w in words:
            final += speller.push(w)
        final += speller.flush()
        glosses = [w['gloss'] for w in final]
        letters = ['FS_' + c for g in glosses if g.startswith('fs-') for c in g[3:]]
        rows.append(dict(letter=meta['letter'], session=meta['session'], glosses=glosses,
                         correct=letters == ['FS_' + meta['letter']], any_correct=('FS_' + meta['letter']) in letters,
                         lexical=[g for g in glosses if not g.startswith('FS_')]))
    n = len(rows)
    by_letter = {L: sum(r['correct'] for r in rows if r['letter'] == L) / max(1, sum(r['letter'] == L for r in rows))
                 for L in sorted(per)}
    confusions = Counter((r['letter'], (r['glosses'] or ['none'])[0]) for r in rows if not r['any_correct'])
    out = dict(clips=n, exact_single_correct=sum(r['correct'] for r in rows) / n,
               contains_correct=sum(r['any_correct'] for r in rows) / n,
               no_output=sum(not r['glosses'] for r in rows) / n,
               lexical_word_emitted=sum(bool(r['lexical']) for r in rows) / n,
               by_letter=by_letter, top_confusions=confusions.most_common(15), rows=rows,
               scope='validation sessions (DWIGHT + numbered), isolated held letters, streaming runtime')
    a.output.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
