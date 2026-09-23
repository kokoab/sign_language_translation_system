"""Is the live path's fixed 0.1s context pad the right one?

Every call site pads a proposed boundary interval by a hard-coded 0.1s on each side before
handing it to Reel (`classify_interval(..., .1)`). The padding sweep over 401 hand-annotated
continuous intervals put verifier top-1 at 78.2% with a 0.05s pad against 72.6% at 0.10s,
and falling monotonically beyond that. That measured isolated recognition on hand-marked
intervals; this measures end-to-end WER on the untouched held-out set, where intervals come
from the boundary model and are wider than hand annotation, so the optimum may differ.

The deployed boundary checkpoint is evaluated as it ships. Because the model and the cached
observations are fixed, each context value is deterministic - no seed averaging is needed
for that arm, and any difference between contexts is exact rather than sampled. bio_gap
seeds are included to check the answer is not specific to one boundary model.

Read-only; nothing is trained on the held-out set and nothing is promoted.
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
from active.v17.temporal_boundary_v17 import TemporalBoundary, BoundaryStream
from active.v17.stage1_window_v17 import raw_observation_features
from scripts.train_temporal_boundary_v17 import load_recipe
from scripts.evaluate_temporal_boundary_v17 import arguments, observations, evaluation_rows, edit_counts, summarize
from scripts.evaluate_boundary_expanded_v17 import local_rows
from scripts.live_boundary_v17 import classify_interval
from scripts.live_reel_stage1_v17 import build_components
from active.v17.extract_v17 import AppleVisionDetector
from scripts.sweep_boundary_head_v17 import build_chunks, TARGETS, train_arm, BioStream

REPORT = ROOT / 'artifacts/reports/window_match_v17_20260923'
INCUMBENT = ROOT / 'artifacts/models/asl_temporal_boundary_v17_20260922/seed_17621_geometry_1.pth'
CONTEXTS = (.025, .05, .075, .10, .15, .20)


def run(model, kind, geometry, prepared, reel, args, context):
    records = []
    for row, obs in prepared:
        raw, times = raw_observation_features(obs)
        stream = (BioStream if kind == 'bio' else BoundaryStream)(model, geometry)
        events = []
        for i in range(len(times)):
            result = stream.update(raw[i], float(times[i]))
            if result is not None:
                events.extend(result['events'])
        predictions = [classify_interval(reel, obs, e, args, context) for e in events]
        hyp = [p['committed_gloss'] for p in predictions if p['committed_gloss']]
        records.append(dict(id=row['source_item_id'], subset=row['subset'],
                            reference=row['target_sequence'], hypothesis=hyp,
                            metrics=edit_counts(row['target_sequence'], hyp)))
    s = summarize(records)
    for name in ('asllrp12', 'local60'):
        chosen = [c for c in records if c['subset'] == name]
        ref = sum(c['metrics']['references'] for c in chosen)
        err = sum(c['metrics']['substitutions'] + c['metrics']['deletions'] +
                  c['metrics']['insertions'] for c in chosen)
        s[name] = err / max(ref, 1)
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='17621,17622,17623,17624')
    parser.add_argument('--dilations', default='1,2,4,8,16')
    parser.add_argument('--hidden', type=int, default=64)
    args_cli = parser.parse_args()
    dilations = tuple(int(d) for d in args_cli.dilations.split(','))
    REPORT.mkdir(parents=True, exist_ok=True)

    asllrp = evaluation_rows(report=REPORT)
    for row in asllrp:
        row['subset'] = 'asllrp12'
    local = local_rows()
    for row in local:
        row['subset'] = 'local60'
    rows = asllrp + local
    print('held-out: %d videos, %d reference signs'
          % (len(rows), sum(len(r['target_sequence']) for r in rows)), flush=True)

    args = arguments(rows[0]['video_path'], REPORT / 'ctx_sessions')
    args.no_motion_trim = True
    reel = build_components(args)['classifier']
    reel.args.no_motion_trim = reel.full.args.no_motion_trim = True
    detector = AppleVisionDetector(args.minimum_point_confidence)

    tick, prepared = time.perf_counter(), []
    for row in rows:
        try:
            obs = observations(row, args, detector)
        except Exception:
            continue
        if obs:
            prepared.append((row, obs))
    print('observations cached for %d clips (%.0fs)\n' % (len(prepared), time.perf_counter() - tick), flush=True)

    models = []
    state = torch.load(INCUMBENT, map_location='cpu', weights_only=False)
    incumbent = TemporalBoundary(**state['model_config'])
    incumbent.load_state_dict(state['model_state_dict'], strict=True)
    models.append(('deployed_incumbent', 'edges', incumbent.eval(), state['hand_geometry']))

    recipe = load_recipe()
    chunks = build_chunks(recipe, True, 2 * sum(dilations), TARGETS['bio_gap']['fn'])
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    for seed in [int(s) for s in args_cli.seeds.split(',')]:
        model, _ = train_arm('bio_gap', chunks, recipe, seed, device, args_cli.hidden, dilations)
        models.append(('bio_gap_seed_%d' % seed, 'bio', model, True))
    print('models under test: %s\n' % [m[0] for m in models], flush=True)

    results = []
    print('%-22s %8s %9s %10s %10s' % ('model', 'context', 'WER', 'asllrp12', 'local60'))
    for name, kind, model, geometry in models:
        for context in CONTEXTS:
            s = run(model, kind, geometry, prepared, reel, args, context)
            results.append(dict(model=name, context=context, wer=s['wer'],
                                asllrp12=s['asllrp12'], local60=s['local60'],
                                correct=s['correct'], insertions=s['insertions']))
            print('%-22s %8.3f %8.2f%% %9.2f%% %9.2f%%' % (
                name, context, 100 * s['wer'], 100 * s['asllrp12'], 100 * s['local60']), flush=True)
            (REPORT / 'context.json').write_text(json.dumps(dict(
                scope='end-to-end WER on the untouched 72-video / 186-sign held-out set as a '
                      'function of the classify_interval context pad',
                deployed_context=0.1, contexts=list(CONTEXTS),
                reel=reel.provenance(), results=results), indent=1) + '\n')

    print('\nmean WER by context across all models:')
    for context in CONTEXTS:
        vals = [r['wer'] for r in results if r['context'] == context]
        print('  %.3fs : %.2f%%  (n=%d models)' % (context, 100 * np.mean(vals), len(vals)))
    best = min(CONTEXTS, key=lambda c: np.mean([r['wer'] for r in results if r['context'] == c]))
    cur = np.mean([r['wer'] for r in results if r['context'] == .10])
    print('\nbest context %.3fs at %.2f%% vs deployed 0.100s at %.2f%% -> %+.2f points'
          % (best, 100 * np.mean([r['wer'] for r in results if r['context'] == best]),
             100 * cur, 100 * (np.mean([r['wer'] for r in results if r['context'] == best]) - cur)))


if __name__ == '__main__':
    main()
