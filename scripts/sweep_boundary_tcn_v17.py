"""Architecture sweep for the live TemporalBoundary TCN, selected on end-to-end WER.

Round 1 of a chain. Each variant is trained on the pinned boundary supervision and then
scored the way it would actually run: BoundaryStream over Apple Vision observations,
intervals through frozen Reel with unchanged commit rules, whole-video WER.

Selection uses the 89-clip local tuning pool. The 72-video / 186-sign set (12 ASLLRP +
60 held-out local) is NOT touched here so it stays clean for the final report.

Nothing is promoted. The pinned trainer, recipe and production model file are unmodified;
this module subclasses the production TemporalBoundary and rebuilds chunks with a
configurable left context, verified against the pinned builder at left_context=30.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from active.v17.temporal_boundary_v17 import (
    TemporalBoundary, BoundaryStream, boundary_features, boundary_targets,
    masked_boundary_loss, validate_clock)
from active.v17.stage1_window_v17 import raw_observation_features
from active.v17.approved_phrase_data_v17 import digest
from scripts.train_temporal_boundary_v17 import load_recipe, data as pinned_data, read_raw, safe_path
from scripts.evaluate_temporal_boundary_v17 import arguments, observations, edit_counts, summarize
from scripts.live_boundary_v17 import classify_interval
from scripts.live_reel_stage1_v17 import build_components
from active.v17.extract_v17 import AppleVisionDetector

REPORT = ROOT / 'artifacts/reports/boundary_tcn_sweep_v17_20260922'
MODELS = ROOT / 'artifacts/models/boundary_tcn_sweep_v17_20260922'
POOL = ROOT / 'artifacts/reports/pretrained_bio_final_v17_20260922/evaluation_poses.json'
COMBINED = ROOT / 'data/local/combined_dataset_v17_20260922/manifest.json'


class SweptBoundary(TemporalBoundary):
    """Production TemporalBoundary with configurable width and dilation schedule."""

    def __init__(self, input_dim=450, hidden=64, lookahead=4, dilations=(1, 2, 4, 8)):
        nn.Module.__init__(self)
        if input_dim <= 0 or hidden <= 0 or not 0 <= lookahead <= 10:
            raise ValueError('invalid model dimensions/lookahead')
        self.config = dict(input_dim=input_dim, hidden=hidden, lookahead=lookahead,
                           dilations=list(dilations))
        self.lookahead = lookahead
        self.project = nn.Conv1d(input_dim, hidden, 1)
        self.layers = nn.ModuleList(nn.Conv1d(hidden, hidden, 3, dilation=d) for d in dilations)
        self.output = nn.Conv1d(hidden, 2, 1)
        self.left_context = 2 * sum(dilations)     # exact causal receptive field


def build_chunks(recipe, geometry, left_context):
    """Mirror of scripts/train_temporal_boundary_v17.data() with configurable left context."""
    output = {'train': [], 'calibration': [], 'validation': []}
    for record in recipe['records']:
        raw, times, metadata = read_raw(safe_path(record['raw_path']))
        if metadata['observer_contract'] != recipe['observer_contract']:
            raise ValueError('prepared observer contract changed')
        x = boundary_features(raw, times, geometry)
        y = boundary_targets(times, record['intervals'], [], False)
        breaks = [0, *(np.flatnonzero(np.diff(times) > .26) + 1).tolist(), len(times)]
        for first, last in zip(breaks, breaks[1:]):
            for start in range(first, last, recipe['chunk_frames']):
                stop = min(start + recipe['chunk_frames'], last)
                left = max(first, start - left_context)
                right = min(last, stop + recipe['lookahead'])
                targets = y[left:right - recipe['lookahead']].copy()
                targets[:start - left] = -1
                if len(targets) and (targets >= 0).any():
                    output[record['split']].append((x[left:right], targets))
    return output


def parity_check(recipe, geometry):
    """My builder must reproduce the pinned one exactly at the pinned left context of 30."""
    mine = build_chunks(recipe, geometry, 30)
    pinned = pinned_data(recipe, geometry)
    for split in mine:
        if len(mine[split]) != len(pinned[split]):
            raise ValueError('chunk count mismatch in %s: %d vs %d'
                             % (split, len(mine[split]), len(pinned[split])))
        for (xa, ya), (xb, yb) in zip(mine[split], pinned[split]):
            if xa.shape != xb.shape or ya.shape != yb.shape:
                raise ValueError('chunk shape mismatch in ' + split)
            if not np.array_equal(xa, xb) or not np.array_equal(ya, yb):
                raise ValueError('chunk content mismatch in ' + split)
    return {k: len(v) for k, v in mine.items()}


def batch(rows, lookahead, device):
    length, dim = max(len(x) for x, _ in rows), rows[0][0].shape[1]
    xs = np.zeros((len(rows), length, dim), np.float32)
    ys = np.full((len(rows), length - lookahead, 2), -1, np.float32)
    for i, (x, y) in enumerate(rows):
        xs[i, :len(x)], ys[i, :len(y)] = x, y
    return torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device)


def positive_weights(rows, device):
    y = np.concatenate([y for _, y in rows])
    positive = (y == 1).sum(0)
    if (positive == 0).any():
        raise ValueError('missing start/end training positives')
    return torch.as_tensor(np.clip((y == 0).sum(0) / positive, 1, 10), dtype=torch.float32, device=device)


def train_variant(spec, rows, recipe, seed, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SweptBoundary(input_dim=rows['train'][0][0].shape[1], hidden=spec['hidden'],
                          lookahead=recipe['lookahead'], dilations=spec['dilations']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=recipe['lr'])
    weight = positive_weights(rows['train'], device)
    best, best_state = float('inf'), None
    for epoch in range(1, recipe['epochs'] + 1):
        model.train()
        order = rng.permutation(len(rows['train']))
        for s in range(0, len(order), recipe['batch_size']):
            chosen = [rows['train'][i] for i in order[s:s + recipe['batch_size']]]
            x, y = batch(chosen, recipe['lookahead'], device)
            loss = masked_boundary_loss(model(x), y, weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        total = count = 0.
        with torch.inference_mode():
            for s in range(0, len(rows['calibration']), 16):
                x, y = batch(rows['calibration'][s:s + 16], recipe['lookahead'], device)
                n = int((y >= 0).sum())
                total += float(masked_boundary_loss(model(x), y, weight)) * n
                count += n
        value = total / count
        if value < best:
            best = value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.cpu().eval(), best


def evaluate(model, geometry, prepared, reel, args):
    records = []
    for row, obs in prepared:
        raw, times = raw_observation_features(obs)
        stream = BoundaryStream(model, geometry)
        events = []
        for i in range(len(times)):
            # update() returns a trace dict, or None while the lookahead fills.
            result = stream.update(raw[i], float(times[i]))
            if result is not None:
                events.extend(result['events'])
        predictions = [classify_interval(reel, obs, e, args, .1) for e in events]
        hyp = [p['committed_gloss'] for p in predictions if p['committed_gloss']]
        records.append(dict(id=row['source_item_id'], reference=row['target_sequence'],
                            hypothesis=hyp, metrics=edit_counts(row['target_sequence'], hyp),
                            intervals=len(events)))
    s = summarize(records)
    s['proposed_intervals'] = sum(r['intervals'] for r in records)
    return s, records


VARIANTS = [
    dict(name='baseline_h64_d4', hidden=64, dilations=(1, 2, 4, 8)),
    dict(name='h96_d4', hidden=96, dilations=(1, 2, 4, 8)),
    dict(name='h128_d4', hidden=128, dilations=(1, 2, 4, 8)),
    dict(name='h192_d4', hidden=192, dilations=(1, 2, 4, 8)),
    dict(name='h64_d5', hidden=64, dilations=(1, 2, 4, 8, 16)),
    dict(name='h128_d5', hidden=128, dilations=(1, 2, 4, 8, 16)),
    dict(name='h64_d3', hidden=64, dilations=(1, 2, 4)),
    dict(name='h128_d6', hidden=128, dilations=(1, 2, 4, 8, 16, 32)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='17621,17622', help='averaged; seed spread is the noise floor')
    parser.add_argument('--geometry', action='store_true')
    parser.add_argument('--only', default=None, help='comma-separated variant names')
    parser.add_argument('--limit-clips', type=int, default=None, help='smoke test only')
    parser.add_argument('--epochs', type=int, default=None, help='override recipe epochs')
    args_cli = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    recipe = load_recipe()
    started = time.perf_counter()
    counts = parity_check(recipe, args_cli.geometry)
    print('parity OK against pinned builder at left_context=30: %s (%.0fs)'
          % (counts, time.perf_counter() - started), flush=True)

    pool = json.loads(POOL.read_text())
    shas = {r['video_sha256'] for r in pool['records']}
    combined = json.loads(COMBINED.read_text())
    rows_eval = sorted([r for r in combined['records'] if r['video_sha256'] in shas],
                       key=lambda r: r['source_item_id'])
    if args_cli.limit_clips:
        rows_eval = rows_eval[:args_cli.limit_clips]
    if args_cli.epochs:
        recipe = {**recipe, 'epochs': args_cli.epochs}
    print('tuning pool: %d clips, %d reference signs'
          % (len(rows_eval), sum(len(r['target_sequence']) for r in rows_eval)), flush=True)

    args = arguments(rows_eval[0]['video_path'], REPORT / 'sessions')
    args.no_motion_trim = True
    reel = build_components(args)['classifier']
    reel.args.no_motion_trim = reel.full.args.no_motion_trim = True
    detector = AppleVisionDetector(args.minimum_point_confidence)

    tick = time.perf_counter()
    prepared = []
    for i, row in enumerate(rows_eval):
        try:
            obs = observations(row, args, detector)
        except Exception as exc:
            print('skip', row['source_item_id'], str(exc)[:70], flush=True)
            continue
        if obs:
            prepared.append((row, obs))
        if (i + 1) % 25 == 0:
            print('  observations %d/%d  %.0fs' % (i + 1, len(rows_eval), time.perf_counter() - tick), flush=True)
    print('observations cached for %d clips in %.0fs\n' % (len(prepared), time.perf_counter() - tick), flush=True)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    selected = [v for v in VARIANTS if not args_cli.only or v['name'] in args_cli.only.split(',')]
    seeds = [int(s) for s in args_cli.seeds.split(',')]
    results = []
    print('%-18s %8s %5s %8s %9s %7s %8s %7s %7s' % (
        'variant', 'params', 'rf', 'cal_loss', 'meanWER', 'spread', 'correct', 'ins', 'ivals'))
    for spec in selected:
        chunks = build_chunks(recipe, args_cli.geometry, 2 * sum(spec['dilations']))
        runs = []
        for seed in seeds:
            model, cal = train_variant(spec, chunks, recipe, seed, device)
            summary, _ = evaluate(model, args_cli.geometry, prepared, reel, args)
            torch.save(dict(format='boundary_tcn_sweep_v17', spec=spec, seed=seed,
                            hand_geometry=args_cli.geometry, model_config=model.config,
                            calibration_loss=cal, model_state_dict=model.state_dict()),
                       MODELS / ('%s_seed_%d.pth' % (spec['name'], seed)))
            runs.append(dict(seed=seed, calibration_loss=cal, summary=summary))
        wers = [r['summary']['wer'] for r in runs]
        params = sum(p.numel() for p in model.parameters())
        results.append(dict(spec=spec, parameters=params, receptive_field=model.left_context,
                            mean_wer=float(np.mean(wers)), seed_spread=float(max(wers) - min(wers)),
                            mean_calibration_loss=float(np.mean([r['calibration_loss'] for r in runs])),
                            runs=runs))
        r = results[-1]
        print('%-18s %8d %5d %8.4f %8.2f%% %6.2f%% %8.1f %7.1f %7.1f' % (
            spec['name'], params, model.left_context, r['mean_calibration_loss'],
            100 * r['mean_wer'], 100 * r['seed_spread'],
            np.mean([x['summary']['correct'] for x in runs]),
            np.mean([x['summary']['insertions'] for x in runs]),
            np.mean([x['summary']['proposed_intervals'] for x in runs])), flush=True)
        results.sort(key=lambda r: r['mean_wer'])
        (REPORT / 'round1.json').write_text(json.dumps(dict(
            round=1, objective='architecture sweep, supervised start/end targets, selected on '
            'mean end-to-end WER over the 89-clip local tuning pool; 186-sign held-out set untouched',
            seeds=seeds, hand_geometry=args_cli.geometry, recipe_epochs=recipe['epochs'],
            reel=reel.provenance(), results=results), indent=1) + '\n')

    print('\nbest: %s at %.2f%% mean WER (seed spread %.2f)'
          % (results[0]['spec']['name'], 100 * results[0]['mean_wer'], 100 * results[0]['seed_spread']))
    print('wrote', REPORT / 'round1.json')


if __name__ == '__main__':
    main()
