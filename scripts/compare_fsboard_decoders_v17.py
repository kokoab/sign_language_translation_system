"""Compare two FSboard extractions of the same clips (ffmpeg pre-scaled vs OpenCV full-resolution decode).

Inputs: frame times, per-slot hand presence, landmark positions (in palm lengths) and hand-crop embedding
cosine. Output: the live letter decoder's spelled string versus the phrase letters (CER) for each path.
Reference scale: the same landmarks' frame-to-frame movement within the baseline extraction.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import re

import numpy as np


def cer(hyp, ref):
    d = np.arange(len(ref) + 1)
    for i, h in enumerate(hyp, 1):
        prev, d[0] = d[0], i
        for j, r in enumerate(ref, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (h != r))
    return d[-1] / max(1, len(ref))


def present(raw):
    return (raw[:, :42, 4] > 0).reshape(len(raw), 2, 21).sum(2) >= 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('candidate', type=Path)
    ap.add_argument('baseline', type=Path)
    ap.add_argument('--json', type=Path)
    a = ap.parse_args()
    agree, both, lm, jitter, cos, rows = [], 0, [], [], [], []
    for f in sorted(glob.glob(str(a.baseline / '*.npz'))):
        g = a.candidate / Path(f).name
        if not g.exists():
            continue
        b, c = np.load(f), np.load(g)
        assert np.allclose(b['times'], c['times']), f
        rb, rc = b['raw'].astype(np.float32), c['raw'].astype(np.float32)
        pb, pc = present(rb), present(rc)
        agree.append((pb == pc).mean())
        for s, o in ((0, 0), (1, 21)):
            m = pb[:, s] & pc[:, s]
            both += m.sum()
            if m.sum() < 2:
                continue
            xb, xc = rb[m, o:o + 21, :2], rc[m, o:o + 21, :2]
            palm = np.linalg.norm(xb[:, 9] - xb[:, 0], axis=1)[:, None]
            lm.append((np.linalg.norm(xb - xc, axis=2) / np.maximum(palm, 1e-4)).ravel())
            jitter.append((np.linalg.norm(np.diff(xb, axis=0), axis=2) / np.maximum(palm[1:], 1e-4)).ravel())
        vb, vc = b['hand_valid'], c['hand_valid']
        m = vb & vc
        eb, ec = b['hand_embeddings'][m].astype(np.float32), c['hand_embeddings'][m].astype(np.float32)
        cos.append((eb * ec).sum(1) / np.maximum(np.linalg.norm(eb, axis=1) * np.linalg.norm(ec, axis=1), 1e-6))
        ref = re.sub(r'[^A-Z]', '', str(b['phrase']).upper())
        rows.append(dict(clip=Path(f).name, ref=ref, base=str(b['letter_output']), cand=str(c['letter_output']),
                         cer_base=cer(str(b['letter_output']), ref), cer_cand=cer(str(c['letter_output']), ref)))
    lm, jitter, cos = np.concatenate(lm), np.concatenate(jitter), np.concatenate(cos)
    pct = lambda x: {p: round(float(np.percentile(x, p)), 4) for p in (50, 90, 99)}
    d = np.array([r['cer_cand'] - r['cer_base'] for r in rows])
    rng = np.random.default_rng(0)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    report = dict(
        clips=len(rows), hand_presence_agreement=round(float(np.mean(agree)), 4), paired_hand_frames=int(both),
        landmark_shift_palms=pct(lm), frame_to_frame_motion_palms=pct(jitter),
        embedding_cosine={p: round(float(np.percentile(cos, p)), 4) for p in (1, 10, 50)},
        letter_cer_baseline=round(float(np.mean([r['cer_base'] for r in rows])), 4),
        letter_cer_candidate=round(float(np.mean([r['cer_cand'] for r in rows])), 4),
        cer_diff_mean=round(float(d.mean()), 4), cer_diff_95ci=[round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)],
        identical_letter_output=sum(r['base'] == r['cand'] for r in rows),
        letters_output_baseline=sum(len(r['base']) for r in rows), letters_output_candidate=sum(len(r['cand']) for r in rows),
        letters_in_phrases=sum(len(r['ref']) for r in rows))
    print(json.dumps(report, indent=1))
    if a.json:
        a.json.write_text(json.dumps(dict(report, rows=rows), indent=1))


if __name__ == '__main__':
    main()
