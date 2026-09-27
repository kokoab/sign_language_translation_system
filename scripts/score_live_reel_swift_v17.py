"""Score the Swift live Reel port (mobile app LiveReel/*.swift) on an evaluation set.

`list` writes the video list the macOS harness reads; `score` computes WER / precision / recall /
latency from the harness output exactly as scripts/replay_segmental_v17.py does for Python.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('list', 'score'))
    ap.add_argument('--set', default='tune')
    ap.add_argument('--input', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--python-replay', type=Path, default=None, help='Python replay JSON to compare per video')
    a = ap.parse_args()
    from scripts import segmental_lab_v17 as lab
    rows = lab.rows_for(a.set)
    if a.mode == 'list':
        a.output.write_text(json.dumps([dict(id=r['source_item_id'], path=str(ROOT / r['video_path']),
                                             reference=r['target_sequence'], subset=r.get('subset')) for r in rows]))
        return
    from scripts.evaluate_temporal_boundary_v17 import edit_counts
    subset = {r['source_item_id']: r.get('subset') for r in rows}
    data = json.loads(a.input.read_text())
    records = []
    for r in data['records']:
        m = edit_counts(r['reference'], r['hypothesis'])
        records.append(dict(r, subset=subset.get(r['id']), metrics={k: m[k] for k in ('substitutions', 'deletions', 'insertions', 'correct', 'references')}))

    def summary(recs):
        t = {k: sum(r['metrics'][k] for r in recs) for k in ('substitutions', 'deletions', 'insertions', 'correct', 'references')}
        h = sum(len(r['hypothesis']) for r in recs)
        lat = np.array([w['commit_seconds'] - w['end_seconds'] + w['compute_seconds']
                        for r in recs for w in r['words'] if w['compute_seconds'] >= 0] or [0.])
        return dict(videos=len(recs), wer=(t['substitutions'] + t['deletions'] + t['insertions']) / max(t['references'], 1),
                    precision=t['correct'] / max(h, 1), recall=t['correct'] / max(t['references'], 1), hypotheses=h, **t,
                    latency_median=float(np.median(lat)), latency_p90=float(np.percentile(lat, 90)),
                    latency_max=float(lat.max()), latency_under_05=float((lat < .5).mean()))
    out = dict(set=a.set, overall=summary(records),
               subsets={s: summary([r for r in records if r['subset'] == s]) for s in sorted({str(r['subset']) for r in records})},
               compute_ms=data.get('compute_ms'))
    if a.python_replay:
        py = {r['id']: r['hypothesis'] for r in json.loads(a.python_replay.read_text())['records']}
        same = [r['id'] for r in records if py.get(r['id']) == r['hypothesis']]
        out['python_comparison'] = dict(identical_hypotheses=len(same), videos=len(records),
                                        differing=[dict(id=r['id'], reference=r['reference'], swift=r['hypothesis'], python=py.get(r['id']))
                                                   for r in records if r['id'] not in same])
    out['records'] = records
    a.output.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('overall', 'subsets', 'compute_ms')}, indent=1))
    if 'python_comparison' in out:
        pc = out['python_comparison']
        print('identical to Python:', pc['identical_hypotheses'], '/', pc['videos'])
        for d in pc['differing']:
            print(' ', d['id'], d['reference'], 'swift', d['swift'], 'python', d['python'])


if __name__ == '__main__':
    main()
