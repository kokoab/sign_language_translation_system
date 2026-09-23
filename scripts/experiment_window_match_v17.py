"""Does the recognizer want the isolated temporal envelope rather than a real sign?

Isolated training clips run 2.07s, and 1.17s after the automatic hand-trim. Real signs in
continuous video run 0.33s (ASLLRP) to 0.27s (hand-annotated local). The recognizer is
therefore trained on roughly 4x more signing per example than it is asked to identify at
inference, and the trimmed isolated clip still contains preparatory movement and release
that continuous signing does not have.

If that mismatch is real and costly, recognition accuracy on hand-annotated continuous
intervals should IMPROVE as the interval is padded outward toward the isolated envelope,
peaking near a total width of about 1.17s. If accuracy peaks at little or no padding, the
recognizer is not asking for the isolated envelope and re-windowing isolated training data
would buy nothing.

Uses the reviewer's own intervals as ground truth for both timing and identity. The frozen
Reel cascade is read only; nothing is trained or promoted.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.evaluate_temporal_boundary_v17 import arguments, observations
from scripts.live_reel_stage1_v17 import build_components
from active.v17.extract_v17 import AppleVisionDetector

REVIEW = ROOT / 'artifacts/reports/local_phrase_alignment_v17_20260923/boundary_review_corrected.json'
COMBINED = ROOT / 'data/local/combined_dataset_v17_20260922/manifest.json'
REPORT = ROOT / 'artifacts/reports/window_match_v17_20260923'
PADS = (0., .05, .1, .15, .2, .3, .4, .55)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    args_cli = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)

    review = json.loads(REVIEW.read_text())
    reviewed = [r for r in review['records'] if r['reviewed']]
    combined = json.loads(COMBINED.read_text())
    by_sha = {r['video_sha256']: r for r in combined['records'] if 'source_item_id' in r}
    rows = [(r, by_sha[r['video_sha256']]) for r in reviewed if r['video_sha256'] in by_sha]
    if args_cli.limit:
        rows = rows[:args_cli.limit]
    marks = sum(1 for r, _ in rows for i in r['intervals'] if i)
    widths = [b - a for r, _ in rows for i in r['intervals'] if i for a, b in [i]]
    print('reviewed clips %d | annotated sign intervals %d | median width %.3fs'
          % (len(rows), marks, np.median(widths)), flush=True)

    args = arguments(rows[0][1]['video_path'], REPORT / 'sessions')
    args.no_motion_trim = True
    reel = build_components(args)['classifier']
    reel.args.no_motion_trim = reel.full.args.no_motion_trim = True
    detector = AppleVisionDetector(args.minimum_point_confidence)

    tally = {p: dict(n=0, prop=0, ver=0, either=0, width=[], prop_score=[], ver_score=[]) for p in PADS}
    per_gloss = {}
    tick = time.perf_counter()
    for idx, (rec, row) in enumerate(rows):
        try:
            obs = observations(row, args, detector)
        except Exception:
            continue
        if not obs:
            continue
        for gloss, interval in zip(rec['target_sequence'], rec['intervals']):
            if not interval:
                continue
            start, end = interval
            for pad in PADS:
                selected = [o for o in obs if start - pad <= o.seconds <= end + pad]
                if len(selected) < 4:
                    continue
                proposal = reel.classify(selected)
                verifier = reel.verify(selected)
                t = tally[pad]
                t['n'] += 1
                p_hit = proposal.get('candidate_gloss') == gloss
                v_hit = verifier.get('candidate_gloss') == gloss
                t['prop'] += p_hit
                t['ver'] += v_hit
                t['either'] += (p_hit or v_hit)
                t['width'].append(selected[-1].seconds - selected[0].seconds)
                pick = lambda r: next((float(x['model_score']) for x in r['top3'] if x['gloss'] == gloss), 0.)
                t['prop_score'].append(pick(proposal))
                t['ver_score'].append(pick(verifier))
                per_gloss.setdefault(gloss, {}).setdefault(pad, [0, 0])
                per_gloss[gloss][pad][0] += v_hit
                per_gloss[gloss][pad][1] += 1
        if (idx + 1) % 20 == 0:
            print('  %d/%d clips  %.0fs' % (idx + 1, len(rows), time.perf_counter() - tick), flush=True)

    print('\n%-7s %7s %10s %10s %10s %11s %11s' % (
        'pad', 'n', 'window', 'proposal', 'verifier', 'either', 'ver score'))
    results = []
    for pad in PADS:
        t = tally[pad]
        if not t['n']:
            continue
        row = dict(pad=pad, n=t['n'], median_window=float(np.median(t['width'])),
                   proposal_top1=t['prop'] / t['n'], verifier_top1=t['ver'] / t['n'],
                   either_top1=t['either'] / t['n'],
                   verifier_score=float(np.mean(t['ver_score'])),
                   proposal_score=float(np.mean(t['prop_score'])))
        results.append(row)
        print('%-7.2f %7d %9.2fs %9.1f%% %9.1f%% %10.1f%% %11.3f' % (
            pad, t['n'], row['median_window'], 100 * row['proposal_top1'],
            100 * row['verifier_top1'], 100 * row['either_top1'], row['verifier_score']))

    best = max(results, key=lambda r: r['verifier_top1'])
    print('\npeak verifier accuracy %.1f%% at pad %.2fs (median window %.2fs)'
          % (100 * best['verifier_top1'], best['pad'], best['median_window']))
    print('isolated training envelope after hand-trim: 1.17s median')
    zero = next(r for r in results if r['pad'] == 0.)
    print('gain from padding: %+.1f points over the annotated interval alone'
          % (100 * (best['verifier_top1'] - zero['verifier_top1'])))
    (REPORT / 'window_match.json').write_text(json.dumps(dict(
        scope='recognition accuracy on hand-annotated continuous intervals as a function of '
              'symmetric padding; tests whether the recognizer wants the isolated envelope',
        clips=len(rows), intervals=marks, median_annotated_width=float(np.median(widths)),
        isolated_trimmed_median_seconds=1.17, reel=reel.provenance(),
        results=results,
        per_gloss={g: {str(p): v for p, v in d.items()} for g, d in per_gloss.items()}), indent=1) + '\n')
    print('wrote', REPORT / 'window_match.json')


if __name__ == '__main__':
    main()
