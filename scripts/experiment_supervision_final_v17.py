"""Why does wide boundary supervision beat tight, and what fixes the reviewer's intervals?

Established at n=142 on the untouched held-out set: adding the aligner's wide intervals
(median 0.667s) is worth -3.90 WER, p=0.031, while the reviewer's tight intervals
(median 0.167s) are worth -0.47, p=0.833, and are WORSE than the aligner's despite being
twice as many. Dropping everything under the decoder's 0.15s minimum recovered only 1.14.

Mechanism under test: BIO training labels frames INSIDE a sign. A 0.167s span is ~3 frames
at 20Hz, so a tight corpus teaches the model that signs are 3 frames long; the model then
proposes spans shorter than BoundaryDecoder's 0.15s minimum, which discards them, and the
recognizer never sees the sign at all. If that is right, every arm's PROPOSED interval
width should track its training width, and tight arms should propose fewer intervals.
Each arm therefore reports what it proposes, not just its WER.

Arms:
  baseline          no local data
  machine_all       every aligner interval over all 197 aligned clips, not just reviewed
  manual_extended   reviewer intervals widened about their own centre to >= TARGET
  manual_to_teacher reviewer centres, but the extent of the teacher segment they fall in
  combined          machine_all plus manual_extended, deduplicated per clip

The reviewer's centres are kept in every manual arm because their +0.166s start correction
was independently confirmed by the ASLLRP audit; only the WIDTH convention is in question.
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
from active.v17.stage1_window_v17 import raw_observation_features
from scripts.train_temporal_boundary_v17 import load_recipe
from scripts.evaluate_temporal_boundary_v17 import arguments, observations, evaluation_rows, edit_counts, summarize
from scripts.evaluate_boundary_expanded_v17 import local_rows
from scripts.live_boundary_v17 import classify_interval
from scripts.live_reel_stage1_v17 import build_components
from active.v17.extract_v17 import AppleVisionDetector
from scripts.sweep_boundary_head_v17 import bio_targets, train_arm, BioStream
from scripts.align_local_phrases_v17 import decode_segments
from scripts.experiment_local_supervision_v17 import base_chunks, local_observations, chunk

ALIGN = ROOT / 'artifacts/reports/local_phrase_alignment_v17_20260923/alignment.json'
REVIEW = ROOT / 'artifacts/reports/local_phrase_alignment_v17_20260923/boundary_review_corrected.json'
COMBINED = ROOT / 'data/local/combined_dataset_v17_20260922/manifest.json'
TEACHER = ROOT / 'artifacts/cache/local_phrase_teacher_segments_v17'
REPORT = ROOT / 'artifacts/reports/local_phrase_alignment_v17_20260923'
TARGET = .60                      # the aligner's median width, which is what wins


def widen(interval, limit, target=TARGET, lo=0., hi=None):
    """Grow an interval about its centre toward `target`, without crossing its neighbours.

    Widening each interval independently overlapped 69% of adjacent pairs and pushed total
    annotated span to 186% of clip duration, because these clips run ~0.9s with three signs.
    That labels the whole clip as sign. Expansion is therefore bounded by the space actually
    available between the previous interval's end and the next one's start.
    """
    hi = limit if hi is None else hi
    a, b = interval
    centre = (a + b) / 2
    room = max(0., hi - lo)
    width = min(max(b - a, target), room)
    a, b = centre - width / 2, centre + width / 2
    if a < lo:
        a, b = lo, lo + width
    if b > hi:
        a, b = hi - width, hi
    return [float(max(lo, a)), float(min(hi, b))]


def widen_clip(intervals, limit, target=TARGET, margin=.02):
    """Widen every interval in one clip, sharing the gaps between them fairly."""
    order = sorted(intervals)
    out = []
    for i, iv in enumerate(order):
        prev_end = order[i - 1][1] if i else 0.
        next_start = order[i + 1][0] if i + 1 < len(order) else limit
        lo = max(0., (prev_end + iv[0]) / 2 + margin / 2) if i else 0.
        hi = min(limit, (iv[1] + next_start) / 2 - margin / 2) if i + 1 < len(order) else limit
        if hi - lo < .15:
            lo, hi = max(0., iv[0] - .02), min(limit, iv[1] + .02)
        out.append(widen(iv, limit, target, lo, hi))
    return out


def teacher_span(interval, segments, limit):
    """The teacher segment containing this interval's centre, else a widened interval."""
    centre = (interval[0] + interval[1]) / 2
    hit = [s for s in segments if s[0] - 1e-9 <= centre <= s[1] + 1e-9]
    if not hit:
        hit = sorted(segments, key=lambda s: abs((s[0] + s[1]) / 2 - centre))[:1]
    if not hit or hit[0][1] - hit[0][0] < .15:
        return widen(interval, limit)
    return [float(hit[0][0]), float(hit[0][1])]


def evaluate(model, geometry, prepared, reel, args):
    """WER plus what the model actually proposed, which is the diagnostic that matters."""
    records, widths, total = [], [], 0
    for row, obs in prepared:
        raw, times = raw_observation_features(obs)
        stream = BioStream(model, geometry)
        events = []
        for i in range(len(times)):
            result = stream.update(raw[i], float(times[i]))
            if result is not None:
                events.extend(result['events'])
        widths.extend(e['end_seconds'] - e['start_seconds'] for e in events)
        total += len(events)
        predictions = [classify_interval(reel, obs, e, args, .1) for e in events]
        hyp = [p['committed_gloss'] for p in predictions if p['committed_gloss']]
        records.append(dict(id=row['source_item_id'], subset=row['subset'],
                            reference=row['target_sequence'], hypothesis=hyp,
                            metrics=edit_counts(row['target_sequence'], hyp)))
    s = summarize(records)
    s['proposed'] = total
    s['proposed_median_width'] = float(np.median(widths)) if widths else 0.
    for name in ('asllrp12', 'local60'):
        chosen = [c for c in records if c['subset'] == name]
        ref = sum(c['metrics']['references'] for c in chosen)
        err = sum(c['metrics']['substitutions'] + c['metrics']['deletions'] +
                  c['metrics']['insertions'] for c in chosen)
        s[name] = err / max(ref, 1)
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default=','.join(str(17621 + i) for i in range(8)))
    parser.add_argument('--arms', default='baseline,machine_all,manual_extended,manual_to_teacher,combined')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dilations', default='1,2,4,8,16')
    parser.add_argument('--out', default='supervision_final')
    args_cli = parser.parse_args()
    dilations = tuple(int(d) for d in args_cli.dilations.split(','))
    left_context = 2 * sum(dilations)

    align = json.loads(ALIGN.read_text())
    review = json.loads(REVIEW.read_text())
    combined = json.loads(COMBINED.read_text())
    by_sha = {r['video_sha256']: r for r in combined['records'] if 'source_item_id' in r}
    reviewed = [r for r in review['records'] if r['reviewed']]

    machine_all, manual_ext, manual_teach = {}, {}, {}
    limits, segs = {}, {}
    for r in align['records']:
        machine_all[r['video_sha256']] = [list(map(float, iv)) for iv in r['intervals']]
    for sha in set(machine_all) | {r['video_sha256'] for r in reviewed}:
        path = TEACHER / (sha + '.json')
        if path.exists():
            data = json.loads(path.read_text())
            limits[sha] = data['frames'] / data['fps'] if data.get('fps') else 4.
            segs[sha] = [[s['start'], s['end']] for s in decode_segments(data['labels'], data['fps'])] \
                if data.get('labels') else []
    for r in reviewed:
        sha = r['video_sha256']
        ivs = [iv for iv in r['intervals'] if iv]
        if not ivs:
            continue
        limit = limits.get(sha, max(b for _, b in ivs) + .2)
        manual_ext[sha] = widen_clip(ivs, limit)
        manual_teach[sha] = [teacher_span(iv, segs.get(sha, []), limit) for iv in ivs]

    def merge(a, b):
        out = {k: list(v) for k, v in a.items()}
        for k, v in b.items():
            out.setdefault(k, [])
            for iv in v:
                if not any(min(iv[1], o[1]) - max(iv[0], o[0]) > .05 for o in out[k]):
                    out[k].append(iv)
        return out

    sets = {'machine_all': machine_all, 'manual_extended': manual_ext,
            'manual_to_teacher': manual_teach, 'combined': merge(machine_all, manual_ext)}
    print('training-interval inventory:')
    for name, v in sets.items():
        w = [b - a for ivs in v.values() for a, b in ivs]
        print('  %-18s %3d clips  %4d intervals  median width %.3fs'
              % (name, len(v), len(w), np.median(w)), flush=True)

    asllrp = evaluation_rows(report=REPORT)
    for row in asllrp:
        row['subset'] = 'asllrp12'
    held = local_rows()
    for row in held:
        row['subset'] = 'local60'
    rows_eval = asllrp + held
    rows_local = [by_sha[s] for s in sets['combined'] if s in by_sha]
    leak = {r['video_sha256'] for r in rows_local} & {r['video_sha256'] for r in rows_eval}
    if leak:
        raise ValueError('local training clips leak into held-out: %d' % len(leak))
    print('\nseparation verified: 0 of %d local clips in the held-out set' % len(rows_local), flush=True)

    args = arguments(rows_eval[0]['video_path'], REPORT / 'final_sessions')
    args.no_motion_trim = True
    reel = build_components(args)['classifier']
    reel.args.no_motion_trim = reel.full.args.no_motion_trim = True
    detector = AppleVisionDetector(args.minimum_point_confidence)
    obs_cache = local_observations(rows_local, args, detector)

    tick, prepared = time.perf_counter(), []
    for row in rows_eval:
        try:
            obs = observations(row, args, detector)
        except Exception:
            continue
        if obs:
            prepared.append((row, obs))
    print('held-out ready: %d clips (%.0fs)\n' % (len(prepared), time.perf_counter() - tick), flush=True)

    recipe = load_recipe()
    base = base_chunks(recipe, True, left_context)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    seeds = [int(s) for s in args_cli.seeds.split(',')]
    results = []
    print('%-18s %6s %9s %6s %10s %10s %9s %9s' % (
        'arm', 'chunks', 'meanWER', 'SD', 'proposed', 'propwidth', 'asllrp12', 'local60'))
    for arm in args_cli.arms.split(','):
        chunks = {k: list(v) for k, v in base.items()}
        added = 0
        if arm != 'baseline':
            for sha, intervals in sets[arm].items():
                if sha not in obs_cache:
                    continue
                raw, times = obs_cache[sha]
                keep = [iv for iv in intervals if times[0] - 1e-8 <= iv[0] < iv[1] <= times[-1] + 1e-8]
                if not keep:
                    continue
                x = boundary_features(raw, times, True)
                y = bio_targets(times, keep, True)
                before = len(chunks['train'])
                chunk(x, y, recipe, left_context, chunks['train'])
                added += len(chunks['train']) - before
        runs = []
        for seed in seeds:
            model, cal = train_arm('bio_gap', chunks, recipe, seed, device, args_cli.hidden, dilations)
            s = evaluate(model, True, prepared, reel, args)
            runs.append(dict(seed=seed, calibration_loss=cal, summary=s))
            print('   %-17s seed %d  WER %.2f%%  proposed %d @ %.2fs'
                  % (arm, seed, 100 * s['wer'], s['proposed'], s['proposed_median_width']), flush=True)
        w = np.array([r['summary']['wer'] for r in runs])
        results.append(dict(arm=arm, added_chunks=added, runs=runs, mean_wer=float(w.mean()),
                            sd=float(w.std(ddof=1)),
                            proposed=float(np.mean([r['summary']['proposed'] for r in runs])),
                            proposed_width=float(np.mean([r['summary']['proposed_median_width'] for r in runs])),
                            asllrp12=float(np.mean([r['summary']['asllrp12'] for r in runs])),
                            local60=float(np.mean([r['summary']['local60'] for r in runs]))))
        r = results[-1]
        print('%-18s %6d %8.2f%% %6.2f %10.0f %9.2fs %8.2f%% %8.2f%%' % (
            arm, added, 100 * r['mean_wer'], 100 * r['sd'], r['proposed'],
            r['proposed_width'], 100 * r['asllrp12'], 100 * r['local60']), flush=True)
        (REPORT / (args_cli.out + '.json')).write_text(json.dumps(dict(
            scope='final supervision comparison on the untouched 72-video / 186-sign held-out set; '
                  'each arm reports what it proposes so the mechanism is visible, not inferred',
            target_width=TARGET, seeds=seeds,
            inventory={k: dict(clips=len(v), intervals=sum(map(len, v.values())),
                               median_width=float(np.median([b - a for ivs in v.values() for a, b in ivs])))
                       for k, v in sets.items()},
            reel=reel.provenance(), results=results), indent=1) + '\n')

    ranked = sorted(results, key=lambda r: r['mean_wer'])
    print('\nbest: %s at %.2f%%' % (ranked[0]['arm'], 100 * ranked[0]['mean_wer']))


if __name__ == '__main__':
    main()
