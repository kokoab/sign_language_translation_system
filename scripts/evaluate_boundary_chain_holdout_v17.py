"""Final step of the boundary chain: the winner on the untouched held-out set.

Selection happened entirely on the 89-clip local tuning pool. This script reports the
72-video / 186-sign held-out set (12 ASLLRP development videos + the 60 held-out local
phrase clips), which no round of the chain touched.

Verified before running, not assumed:
  tuning pool INTERSECT held-out                = 0 clips
  boundary TRAIN split INTERSECT held-out       = 0 clips
  the 9 ASLLRP eval videos present in the recipe are all in its VALIDATION split, which
  neither training nor epoch selection reads (train -> train, selection -> calibration).

Arms, five fixed seeds each, chosen a priori because calibration loss does not predict WER
(r=+0.08 edges, +0.19 bio_gap, both n.s.) so no honest per-seed selection exists:
  edges      the pinned start/end formulation, retrained in this harness
  bio_gap    the chain winner, 4-class BIO with short-gap negatives
Plus the deployed production checkpoint evaluated exactly as it ships.

Nothing is promoted.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from active.v17.temporal_boundary_v17 import TemporalBoundary
from scripts.train_temporal_boundary_v17 import load_recipe
from scripts.evaluate_temporal_boundary_v17 import arguments, observations, evaluation_rows, summarize
from scripts.evaluate_boundary_expanded_v17 import local_rows
from scripts.live_reel_stage1_v17 import build_components
from active.v17.extract_v17 import AppleVisionDetector
from scripts.sweep_boundary_head_v17 import build_chunks, TARGETS, train_arm, evaluate as head_evaluate

REPORT = ROOT / 'artifacts/reports/boundary_tcn_sweep_v17_20260922'
POOL = ROOT / 'artifacts/reports/pretrained_bio_final_v17_20260922/evaluation_poses.json'
INCUMBENT = ROOT / 'artifacts/models/asl_temporal_boundary_v17_20260922/seed_17621_geometry_1.pth'


def check_separation(recipe, rows):
    pool = {r['video_sha256'] for r in json.loads(POOL.read_text())['records']}
    held = {r['video_sha256'] for r in rows}
    train = {r['video_sha256'] for r in recipe['records'] if r['split'] == 'train'}
    selection = {r['video_sha256'] for r in recipe['records'] if r['split'] == 'calibration'}
    leaks = dict(tuning_pool=len(pool & held), train_split=len(train & held),
                 calibration_split=len(selection & held))
    if any(leaks.values()):
        raise ValueError('held-out set is contaminated: %s' % leaks)
    return leaks


def subsets(records, rows):
    where = {r['source_item_id']: r['subset'] for r in rows}
    out = {}
    for name in ('asllrp12', 'local60'):
        chosen = [r for r in records if where.get(r['id']) == name]
        if chosen:
            out[name] = summarize(chosen)
    out['combined'] = summarize(records)
    out['combined']['scope'] = ('pooled across two sources with different annotation '
                                'coverage; the subsets are the honest read')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='17621,17622,17623,17624,17625')
    parser.add_argument('--arms', default='edges,bio_gap')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dilations', default='1,2,4,8,16')
    parser.add_argument('--out', default='holdout')
    args_cli = parser.parse_args()
    dilations = tuple(int(d) for d in args_cli.dilations.split(','))
    REPORT.mkdir(parents=True, exist_ok=True)

    recipe = load_recipe()
    asllrp = evaluation_rows(report=REPORT)
    for row in asllrp:
        row['subset'] = 'asllrp12'
    local = local_rows()
    for row in local:
        row['subset'] = 'local60'
    rows = asllrp + local
    leaks = check_separation(recipe, rows)
    print('separation verified, zero leakage: %s' % leaks, flush=True)
    print('held-out: %d videos, %d reference signs (asllrp12 %d + local60 %d)'
          % (len(rows), sum(len(r['target_sequence']) for r in rows),
             sum(len(r['target_sequence']) for r in asllrp),
             sum(len(r['target_sequence']) for r in local)), flush=True)

    args = arguments(rows[0]['video_path'], REPORT / 'holdout_sessions')
    args.no_motion_trim = True
    reel = build_components(args)['classifier']
    reel.args.no_motion_trim = reel.full.args.no_motion_trim = True
    detector = AppleVisionDetector(args.minimum_point_confidence)

    tick, prepared = time.perf_counter(), []
    for i, row in enumerate(rows):
        try:
            obs = observations(row, args, detector)
        except Exception as exc:
            print('skip', row['source_item_id'], str(exc)[:70], flush=True)
            continue
        if obs:
            prepared.append((row, obs))
        if (i + 1) % 25 == 0:
            print('  observations %d/%d  %.0fs' % (i + 1, len(rows), time.perf_counter() - tick), flush=True)
    print('observations cached for %d clips in %.0fs\n' % (len(prepared), time.perf_counter() - tick), flush=True)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    seeds = [int(s) for s in args_cli.seeds.split(',')]
    results = []

    state = torch.load(INCUMBENT, map_location='cpu', weights_only=False)
    incumbent = TemporalBoundary(**state['model_config'])
    incumbent.load_state_dict(state['model_state_dict'], strict=True)
    summary, records = head_evaluate(incumbent.eval(), 'edges', state['hand_geometry'],
                                     prepared, reel, args)
    results.append(dict(arm='deployed_incumbent', seeds=[state.get('seed')],
                        checkpoint=str(INCUMBENT.relative_to(ROOT)),
                        hand_geometry=state['hand_geometry'], runs=[dict(summary=summary)],
                        subsets=subsets(records, rows)))
    print('%-20s %9s %7s   asllrp12 %7s   local60 %7s' % ('arm', 'meanWER', 'SD', 'WER', 'WER'))
    r = results[-1]
    print('%-20s %8.2f%% %7s   %15.2f%% %14.2f%%' % (
        'deployed_incumbent', 100 * summary['wer'], '-',
        100 * r['subsets']['asllrp12']['wer'], 100 * r['subsets']['local60']['wer']), flush=True)

    for arm in args_cli.arms.split(','):
        chunks = build_chunks(recipe, True, 2 * sum(dilations), TARGETS[arm]['fn'])
        runs, subs = [], []
        for seed in seeds:
            model, cal = train_arm(arm, chunks, recipe, seed, device, args_cli.hidden, dilations)
            summary, records = head_evaluate(model, arm, True, prepared, reel, args)
            runs.append(dict(seed=seed, calibration_loss=cal, summary=summary))
            subs.append(subsets(records, rows))
        w = np.array([x['summary']['wer'] for x in runs])
        results.append(dict(arm=arm, seeds=seeds, runs=runs, mean_wer=float(w.mean()),
                            sd=float(w.std(ddof=1)),
                            subsets={k: dict(wer=float(np.mean([s[k]['wer'] for s in subs])),
                                             sd=float(np.std([s[k]['wer'] for s in subs], ddof=1)))
                                     for k in subs[0]}))
        r = results[-1]
        print('%-20s %8.2f%% %6.2f   %15.2f%% %14.2f%%' % (
            arm, 100 * r['mean_wer'], 100 * r['sd'],
            100 * r['subsets']['asllrp12']['wer'], 100 * r['subsets']['local60']['wer']), flush=True)
        (REPORT / (args_cli.out + '.json')).write_text(json.dumps(dict(
            scope='72-video / 186-sign held-out set, untouched by every selection round',
            separation=leaks, seeds=seeds, trunk=dict(hidden=args_cli.hidden, dilations=list(dilations)),
            reel=reel.provenance(), results=results), indent=1) + '\n')
    print('\nwrote', REPORT / (args_cli.out + '.json'))


if __name__ == '__main__':
    main()
