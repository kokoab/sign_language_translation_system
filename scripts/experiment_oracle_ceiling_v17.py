"""With perfect boundaries, how good is the recogniser? That sets the ceiling.

On 401 hand-annotated intervals from local TRAIN clips the verifier reached 78.2% top-1.
If that transfers, boundaries are the bottleneck and annotation is the lever. But the
deployed classifier was phrase-adapted on 4,168 phrase segments that plausibly include
those very clips, and this project has already measured heavy memorisation elsewhere
(232/232 train against 30/199 validation). So the figure may be inflated.

This measures the same quantity on material the recogniser has not been adapted to: the 12
held-out ASLLRP development videos, which carry curated per-sign intervals and labels. The
gap between the two numbers is the memorisation premium.

Read-only; trains nothing and promotes nothing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.evaluate_temporal_boundary_v17 import arguments, observations, evaluation_rows
from scripts.live_reel_stage1_v17 import build_components
from active.v17.extract_v17 import AppleVisionDetector

REPORT = ROOT / 'artifacts/reports/window_match_v17_20260923'
PADS = (0., .05, .1, .15, .2)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    rows = evaluation_rows(report=REPORT)
    args = arguments(rows[0]['video_path'], REPORT / 'oracle_sessions')
    args.no_motion_trim = True
    reel = build_components(args)['classifier']
    reel.args.no_motion_trim = reel.full.args.no_motion_trim = True
    detector = AppleVisionDetector(args.minimum_point_confidence)

    tally = {p: dict(n=0, prop=0, ver=0, either=0) for p in PADS}
    labels = set(reel.labels)
    missing, tick = set(), time.perf_counter()
    for row in rows:
        obs = observations(row, args, detector)
        if not obs:
            continue
        for event in row['events']:
            gloss = str(event['label'])
            if gloss not in labels:
                missing.add(gloss)
                continue
            for pad in PADS:
                sel = [o for o in obs if event['start'] - pad <= o.seconds <= event['end'] + pad]
                if len(sel) < 4:
                    continue
                proposal, verifier = reel.classify(sel), reel.verify(sel)
                t = tally[pad]
                t['n'] += 1
                p_hit = proposal.get('candidate_gloss') == gloss
                v_hit = verifier.get('candidate_gloss') == gloss
                t['prop'] += p_hit; t['ver'] += v_hit; t['either'] += (p_hit or v_hit)
        print('  %-28s %.0fs' % (row['source_item_id'][:28], time.perf_counter() - tick), flush=True)

    print('\nORACLE-INTERVAL recognition on HELD-OUT ASLLRP (unseen by phrase adaptation)')
    print('%-7s %6s %11s %11s %10s' % ('pad', 'n', 'proposal', 'verifier', 'either'))
    out = []
    for pad in PADS:
        t = tally[pad]
        if not t['n']:
            continue
        out.append(dict(pad=pad, n=t['n'], proposal=t['prop'] / t['n'],
                        verifier=t['ver'] / t['n'], either=t['either'] / t['n']))
        print('%-7.2f %6d %10.1f%% %10.1f%% %9.1f%%' % (
            pad, t['n'], 100 * out[-1]['proposal'], 100 * out[-1]['verifier'], 100 * out[-1]['either']))
    if missing:
        print('\nglosses outside the 100-label vocabulary, skipped: %d (%s)'
              % (len(missing), ', '.join(sorted(missing)[:8])))
    best = max(out, key=lambda r: r['verifier'])
    print('\nheld-out oracle verifier peak : %.1f%%' % (100 * best['verifier']))
    print('local TRAIN hand-annot peak   : 78.2%  (401 intervals)')
    print('memorisation premium          : %.1f points' % (78.2 - 100 * best['verifier']))
    (REPORT / 'oracle_ceiling.json').write_text(json.dumps(dict(
        scope='recognition with curated ground-truth intervals on the 12 held-out ASLLRP '
              'videos; upper bound on what perfect boundaries could deliver',
        results=out, local_train_reference=0.782,
        unmapped_glosses=sorted(missing), reel=reel.provenance()), indent=1) + '\n')
    print('wrote', REPORT / 'oracle_ceiling.json')


if __name__ == '__main__':
    main()
