"""Does hand-corrected in-domain boundary supervision move end-to-end WER?

The boundary model trains on 1121 records: asllrp_other_ctc 1077, asllrp_contiguous 38,
o5s5 6, local_phrases ZERO. It is then scored mostly on local video, and the chain's
held-out gain sat almost entirely in the local subset (61.73% -> 54.94%).

55 local clips have now been reviewed by hand. On those clips the machine aligner had an
interval for 79 of 165 signs; the reviewer kept 18 of those unchanged, moved 61, and added
86 the aligner never proposed. The reviewer's starts land +0.166s later than the machine's
(median), independently reproducing the audit's finding that the aligner begins 0.117s too
early against curated ASLLRP and so swallows the preparatory movement.

Four arms separate the two things manual annotation supplies:

  baseline        recipe only, no local data at all
  machine         + the 79 aligner intervals               (coverage: machine)
  manual_matched  + the same 79 signs, reviewer's edges    (quality alone)
  manual          + all 165 reviewer intervals             (quality + coverage)

manual_matched vs machine isolates whether better EDGES matter. manual vs manual_matched
isolates whether more COVERAGE matters. Held-out is the untouched 72-video / 186-sign set;
the reviewed clips are training-role and verified disjoint from it.

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
from active.v17.temporal_boundary_v17 import boundary_features
from scripts.train_temporal_boundary_v17 import load_recipe, read_raw, safe_path
from scripts.evaluate_temporal_boundary_v17 import arguments, observations, evaluation_rows
from scripts.evaluate_boundary_expanded_v17 import local_rows
from scripts.live_reel_stage1_v17 import build_components
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.stage1_window_v17 import raw_observation_features
from scripts.sweep_boundary_head_v17 import (
    bio_targets, train_arm, evaluate as head_evaluate, TARGETS)

REVIEW = ROOT / 'artifacts/reports/local_phrase_alignment_v17_20260923/boundary_review_corrected.json'
COMBINED = ROOT / 'data/local/combined_dataset_v17_20260922/manifest.json'
REPORT = ROOT / 'artifacts/reports/local_phrase_alignment_v17_20260923'
CACHE = ROOT / 'artifacts/cache/local_phrase_raw_observations_v17'


def local_observations(rows, args, detector):
    """Apple Vision raw features for local training clips, cached by video sha."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out, tick = {}, time.perf_counter()
    for i, row in enumerate(rows):
        path = CACHE / (row['video_sha256'] + '.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                out[row['video_sha256']] = (z['raw_features'].astype(np.float32),
                                            z['timestamps_seconds'].astype(np.float64))
            continue
        obs = observations(row, args, detector)
        if not obs:
            continue
        raw, times = raw_observation_features(obs)
        raw = np.asarray(raw, np.float32); times = np.asarray(times, np.float64)
        np.savez(path, raw_features=raw, timestamps_seconds=times)
        out[row['video_sha256']] = (raw, times)
        if (i + 1) % 20 == 0:
            print('  local observations %d/%d  %.0fs' % (i + 1, len(rows), time.perf_counter() - tick), flush=True)
    print('local observations ready for %d clips (%.0fs)' % (len(out), time.perf_counter() - tick), flush=True)
    return out


def chunk(x, y, recipe, left_context, store):
    """The pinned chunking, applied to one clip's features and targets."""
    times_len = len(y)
    for start in range(0, times_len, recipe['chunk_frames']):
        stop = min(start + recipe['chunk_frames'], times_len)
        left = max(0, start - left_context)
        right = min(times_len, stop + recipe['lookahead'])
        targets = y[left:right - recipe['lookahead']].copy()
        targets[:start - left] = -1
        if len(targets) and (targets >= 0).any():
            store.append((x[left:right], targets))


def base_chunks(recipe, geometry, left_context):
    out = {'train': [], 'calibration': [], 'validation': []}
    for record in recipe['records']:
        raw, times, metadata = read_raw(safe_path(record['raw_path']))
        if metadata['observer_contract'] != recipe['observer_contract']:
            raise ValueError('prepared observer contract changed')
        x = boundary_features(raw, times, geometry)
        y = bio_targets(times, record['intervals'], True)
        breaks = [0, *(np.flatnonzero(np.diff(times) > .26) + 1).tolist(), len(times)]
        for first, last in zip(breaks, breaks[1:]):
            for start in range(first, last, recipe['chunk_frames']):
                stop = min(start + recipe['chunk_frames'], last)
                left = max(first, start - left_context)
                right = min(last, stop + recipe['lookahead'])
                targets = y[left:right - recipe['lookahead']].copy()
                targets[:start - left] = -1
                if len(targets) and (targets >= 0).any():
                    out[record['split']].append((x[left:right], targets))
    return out


def local_chunks(selected, obs_cache, recipe, geometry, left_context):
    """Extra TRAIN chunks from local clips; intervals must sit on the observed clock."""
    rows, dropped = [], 0
    for sha, intervals in selected.items():
        if sha not in obs_cache or not intervals:
            dropped += 1
            continue
        raw, times = obs_cache[sha]
        keep = [iv for iv in intervals
                if iv and times[0] - 1e-8 <= iv[0] < iv[1] <= times[-1] + 1e-8]
        if not keep:
            dropped += 1
            continue
        x = boundary_features(raw, times, geometry)
        y = bio_targets(times, keep, True)
        chunk(x, y, recipe, left_context, rows)
    return rows, dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default=','.join(str(17621 + i) for i in range(8)))
    parser.add_argument('--arms', default='baseline,machine,manual_matched,manual')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dilations', default='1,2,4,8,16')
    parser.add_argument('--out', default='local_supervision')
    args_cli = parser.parse_args()
    dilations = tuple(int(d) for d in args_cli.dilations.split(','))
    left_context = 2 * sum(dilations)
    REPORT.mkdir(parents=True, exist_ok=True)

    review = json.loads(REVIEW.read_text())
    reviewed = [r for r in review['records'] if r['reviewed']]
    combined = json.loads(COMBINED.read_text())
    by_sha = {r['video_sha256']: r for r in combined['records'] if 'source_item_id' in r}
    rows_local = [by_sha[r['video_sha256']] for r in reviewed if r['video_sha256'] in by_sha]

    # the three interval sets under test
    machine, matched, manual = {}, {}, {}
    for r in reviewed:
        sha = r['video_sha256']
        machine[sha] = [iv for iv in r['machine_intervals'] if iv]
        matched[sha] = [u for m, u in zip(r['machine_intervals'], r['intervals']) if m and u]
        manual[sha] = [iv for iv in r['intervals'] if iv]
    # BoundaryDecoder discards anything under 0.15s at inference, so intervals below that
    # train the model to predict spans it will then throw away. This arm keeps only the
    # reviewer's intervals that clear that floor.
    long_only = {sha: [iv for iv in v if iv[1] - iv[0] >= .15] for sha, v in manual.items()}
    sets = {'machine': machine, 'manual_matched': matched, 'manual': manual,
            'manual_long': long_only}
    print('reviewed clips %d | machine %d | matched %d | manual %d | manual>=0.15s %d'
          % (len(reviewed), sum(map(len, machine.values())), sum(map(len, matched.values())),
             sum(map(len, manual.values())), sum(map(len, long_only.values()))), flush=True)

    asllrp = evaluation_rows(report=REPORT)
    for row in asllrp:
        row['subset'] = 'asllrp12'
    local_held = local_rows()
    for row in local_held:
        row['subset'] = 'local60'
    rows_eval = asllrp + local_held
    leak = {r['video_sha256'] for r in rows_local} & {r['video_sha256'] for r in rows_eval}
    if leak:
        raise ValueError('reviewed clips leak into the held-out set: %d' % len(leak))
    print('separation verified: 0 of %d reviewed clips appear in the held-out set' % len(rows_local), flush=True)

    args = arguments(rows_eval[0]['video_path'], REPORT / 'exp_sessions')
    args.no_motion_trim = True
    reel = build_components(args)['classifier']
    reel.args.no_motion_trim = reel.full.args.no_motion_trim = True
    detector = AppleVisionDetector(args.minimum_point_confidence)

    obs_cache = local_observations(rows_local, args, detector)

    tick, prepared = time.perf_counter(), []
    for i, row in enumerate(rows_eval):
        try:
            obs = observations(row, args, detector)
        except Exception:
            continue
        if obs:
            prepared.append((row, obs))
        if (i + 1) % 25 == 0:
            print('  held-out observations %d/%d  %.0fs' % (i + 1, len(rows_eval), time.perf_counter() - tick), flush=True)
    print('held-out ready: %d clips (%.0fs)\n' % (len(prepared), time.perf_counter() - tick), flush=True)

    recipe = load_recipe()
    base = base_chunks(recipe, True, left_context)
    print('base chunks: %s' % {k: len(v) for k, v in base.items()}, flush=True)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    seeds = [int(s) for s in args_cli.seeds.split(',')]
    results = []
    print('\n%-16s %7s %9s %7s %8s %11s %10s' % (
        'arm', 'added', 'meanWER', 'SD', 'best', 'asllrp12', 'local60'))
    for arm in args_cli.arms.split(','):
        chunks = {k: list(v) for k, v in base.items()}
        added = 0
        if arm != 'baseline':
            extra, dropped = local_chunks(sets[arm], obs_cache, recipe, True, left_context)
            chunks['train'].extend(extra)
            added = len(extra)
            if dropped:
                print('  %s: %d clips contributed no usable chunk' % (arm, dropped), flush=True)
        runs, subs = [], []
        for seed in seeds:
            model, cal = train_arm('bio_gap', chunks, recipe, seed, device, args_cli.hidden, dilations)
            summary, records = head_evaluate(model, 'bio_gap', True, prepared, reel, args)
            where = {r['source_item_id']: r['subset'] for r in rows_eval}
            per = {}
            for name in ('asllrp12', 'local60'):
                chosen = [c for c in records if where.get(c['id']) == name]
                ref = sum(c['metrics']['references'] for c in chosen)
                err = sum(c['metrics']['substitutions'] + c['metrics']['deletions'] +
                          c['metrics']['insertions'] for c in chosen)
                per[name] = err / max(ref, 1)
            runs.append(dict(seed=seed, calibration_loss=cal, summary=summary, subsets=per))
            subs.append(per)
            print('   %-15s seed %d  WER %.2f%%' % (arm, seed, 100 * summary['wer']), flush=True)
        w = np.array([r['summary']['wer'] for r in runs])
        results.append(dict(arm=arm, added_chunks=added, runs=runs, mean_wer=float(w.mean()),
                            sd=float(w.std(ddof=1)), best=float(w.min()),
                            subsets={k: float(np.mean([s[k] for s in subs])) for k in subs[0]}))
        r = results[-1]
        print('%-16s %7d %8.2f%% %6.2f %7.2f%% %10.2f%% %9.2f%%' % (
            arm, added, 100 * r['mean_wer'], 100 * r['sd'], 100 * r['best'],
            100 * r['subsets']['asllrp12'], 100 * r['subsets']['local60']), flush=True)
        (REPORT / (args_cli.out + '.json')).write_text(json.dumps(dict(
            scope='marginal value of hand-corrected in-domain boundary supervision; '
                  'held-out 72-video / 186-sign set, untouched',
            reviewed_clips=len(reviewed), seeds=seeds,
            interval_counts={k: sum(map(len, v.values())) for k, v in sets.items()},
            trunk=dict(hidden=args_cli.hidden, dilations=list(dilations)),
            reel=reel.provenance(), results=results), indent=1) + '\n')

    ranked = sorted(results, key=lambda r: r['mean_wer'])
    print('\nbest: %s at %.2f%% mean WER' % (ranked[0]['arm'], 100 * ranked[0]['mean_wer']))


if __name__ == '__main__':
    main()
