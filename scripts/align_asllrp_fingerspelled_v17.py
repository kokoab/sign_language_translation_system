"""Locate each ASLLRP fingerspelled-word clip inside its utterance video by frame matching.

The utterance word times in utterances_manifest.json are wrong by ~2x for some collections (frame-rate
conversion). The sign clips are cut from the same recordings, so sliding each clip over its utterance
(grayscale 64x48 mean squared error) gives the word's true start/end on the utterance timeline. Writes
matched times into live_features/manifest.json (originals kept as annotated_start/annotated_end) and
match quality, so evaluation windows are correct.
"""
import json
from pathlib import Path
import cv2
import numpy as np

R = Path(__file__).resolve().parents[1] / 'data/local/asllrp_fingerspelled_v17'


def frames(p, size=(64, 48)):
    c = cv2.VideoCapture(str(p)); out = []; fps = c.get(cv2.CAP_PROP_FPS)
    while True:
        ok, f = c.read()
        if not ok:
            break
        out.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), size).astype(np.float32))
    return np.array(out), fps


def main():
    rows = json.loads((R / 'manifest.json').read_text())['rows']
    by_utt = {}
    for r in rows:
        by_utt.setdefault(r['utterance_video'], []).append(r)
    path = R / 'live_features/manifest.json'
    man = json.loads(path.read_text())
    stats = []
    for clip in man['clips']:
        u, fps = frames(R / 'utterances' / clip['clipFilename'])
        cands = by_utt.get(clip['clipFilename'], [])
        used = set()
        for w in clip['words']:
            w.setdefault('annotated_start', w['start']); w.setdefault('annotated_end', w['end'])
            best = None
            for r in cands:
                if r['gloss'] != w['gloss'] or r['sign_video'] in used:
                    continue
                s, _ = frames(R / 'clips' / r['sign_video'])
                if len(s) == 0 or len(s) > len(u):
                    continue
                err = np.array([((u[o:o + len(s)] - s) ** 2).mean() for o in range(len(u) - len(s) + 1)])
                o = int(err.argmin())
                # prefer the occurrence closest to the annotated time when a gloss repeats
                gap = abs(o / fps - w['annotated_start'] / 2), abs(o / fps - w['annotated_start'])
                key = (round(float(err[o]), 1), min(gap))
                if best is None or key < best[0]:
                    best = (key, r['sign_video'], o, len(s), float(err[o]), float(np.median(err)))
            if best is None:
                w['match'] = None
                continue
            _, video, o, n, e, med = best
            used.add(video)
            w['start'], w['end'] = o / fps, (o + n) / fps
            w['match'] = dict(sign_video=video, mse=round(e, 2), median_mse=round(med, 2))
            stats.append((e, med, w['start'] / max(1e-6, w['annotated_start']) if w['annotated_start'] > 0 else 1))
    path.write_text(json.dumps(man, indent=1))
    e = np.array(stats)
    good = e[:, 0] < .1 * e[:, 1]
    ratio = e[:, 2]
    print(f'words matched {len(e)}; clean match (mse < 10% of median) {good.sum()}; '
          f'start/annotated ratio: ~1 {np.sum(abs(ratio - 1) < .15)}, ~0.5 {np.sum(abs(ratio - .5) < .1)}, other {np.sum((abs(ratio - 1) >= .15) & (abs(ratio - .5) >= .1))}')


if __name__ == '__main__':
    main()
