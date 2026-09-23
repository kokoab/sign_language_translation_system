"""Round 4 of the boundary chain: distil the frozen DGS BIO teacher into the live TCN.

Rounds 1-3 established two things. Architecture is not the lever: eight trunks from 66K to
530K parameters spanned less than the seed noise (F(7,8)=1.54, p=0.28). Output formulation
is: at 24 seeds per arm the 4-class BIO head beat the pinned start/end edge head by 2.67
WER points, 95% CI [1.35, 4.00], p<0.001, and proposed 241 intervals against 226 references
where the edge head proposed 291.

What no student arm could fix is coverage. The pinned recipe supervises only frames inside
accepted intervals and says so: "no supervised outside class". 61% of every clip is scored
at inference but never supervised in training, and only gaps under 0.4s can be labelled
honestly from the annotation alone.

The teacher removes that limit. It labels every frame, it is noncausal and full-video, it
reached 40.32% WER frozen where the edge student reached 54.30%, and its poses are already
cached for all 1121 records. load_pose resamples to the same 20Hz clock the student runs
on, so teacher frame i and student frame i are the same instant - verified here, not
assumed. Distillation therefore supplies dense targets over the 61% that has none.

Arms (identical trunk and identical live decoding; 4-class BIO throughout):
  bio_gap     hard labels only - the round-3 winner, re-run here as a paired control
  distill     KL to the teacher on EVERY frame; no hard labels at all
  distill_ce  half KL on every frame, half cross-entropy on the honestly labelled frames

Nothing is promoted. The pinned trainer, recipe, checkpoint and the teacher weights are
read-only; the teacher is never fine-tuned.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
UPSTREAM = ROOT / 'artifacts/reports/pose_boundary_transfer_v17_20260922'
sys.path[:0] = [str(ROOT), str(UPSTREAM), str(UPSTREAM / 'dependencies')]
from active.v17.temporal_boundary_v17 import boundary_features
from scripts.train_temporal_boundary_v17 import load_recipe, read_raw, safe_path
from scripts.evaluate_temporal_boundary_v17 import arguments, observations
from scripts.live_reel_stage1_v17 import build_components
from active.v17.extract_v17 import AppleVisionDetector
from scripts.sweep_boundary_head_v17 import (
    HeadBoundary, BioStream, bio_targets, bio_batch, bio_weights, masked_bio_loss,
    evaluate as head_evaluate, TARGETS, UNK, O, B, I)

REPORT = ROOT / 'artifacts/reports/boundary_tcn_sweep_v17_20260922'
CACHE = ROOT / 'artifacts/cache/boundary_teacher_bio_v17'
POOL = ROOT / 'artifacts/reports/pretrained_bio_final_v17_20260922/evaluation_poses.json'
COMBINED = ROOT / 'data/local/combined_dataset_v17_20260922/manifest.json'
PREPARED = UPSTREAM / 'prepared_manifest.json'
FPS = 20


def cache_teacher():
    """Run the frozen teacher once over every prepared pose; save 20Hz-aligned log-probs."""
    from safetensors.torch import load_file
    from check import model_from_upstream
    from probe import upstream_helpers
    from active.v17.pretrained_boundary_v17 import load_pose
    CACHE.mkdir(parents=True, exist_ok=True)
    helpers = upstream_helpers()
    weights = ROOT / 'artifacts/models/pose_boundary_dgs_2026'
    model = model_from_upstream(json.loads((weights / 'config.json').read_text())).float().eval()
    model.load_state_dict(load_file(str(weights / 'model.safetensors')), strict=True)
    model.to('mps')
    rows = json.loads(PREPARED.read_text())['records']
    tick, made = time.perf_counter(), 0
    for i, row in enumerate(rows):
        out = CACHE / (row['raw_sha256'] + '.npy')
        if out.exists():
            continue
        pose = load_pose(row)
        processed = helpers['preprocess_pose'](pose)
        data = processed.body.data.filled(0)[:, 0, :, :3].astype('float32')
        times = np.arange(len(data), dtype='float32') / FPS
        features = np.concatenate([data, helpers['compute_velocity'](data, times)], axis=-1)
        with torch.inference_mode():
            logp = model(torch.from_numpy(features)[None].to('mps'),
                         timestamps=torch.from_numpy(times)[None].to('mps'))['sign'][0].cpu().numpy()
        np.save(out, logp.astype(np.float32))
        made += 1
        if (i + 1) % 250 == 0:
            print('  teacher %d/%d  %.0fs' % (i + 1, len(rows), time.perf_counter() - tick), flush=True)
    print('teacher cache ready: %d records (%d new) in %.0fs'
          % (len(rows), made, time.perf_counter() - tick), flush=True)
    return {r['raw_sha256']: r for r in rows}


def teacher_agreement(recipe):
    """Gate: does the teacher agree with the annotation where the annotation exists?"""
    hit = total = 0
    inside_as_sign = inside_total = 0
    for record in recipe['records']:
        path = CACHE / (record['raw_sha256'] + '.npy')
        if not path.exists():
            continue
        raw, times, _ = read_raw(safe_path(record['raw_path']))
        logp = np.load(path)
        if len(logp) != len(times):
            continue
        y = bio_targets(times, record['intervals'], True)
        pred = logp.argmax(-1)
        known = y >= 0
        hit += int((pred[known] == y[known]).sum()); total += int(known.sum())
        inside = np.isin(y, (B, I))
        inside_as_sign += int(np.isin(pred[inside], (B, I)).sum()); inside_total += int(inside.sum())
    return dict(frame_agreement=hit / max(total, 1), supervised_frames=total,
                sign_frames_recalled=inside_as_sign / max(inside_total, 1))


def build_chunks(recipe, geometry, left_context):
    """Pinned chunking, carrying hard BIO targets and the teacher's per-frame log-probs."""
    output = {'train': [], 'calibration': [], 'validation': []}
    missing = 0
    for record in recipe['records']:
        path = CACHE / (record['raw_sha256'] + '.npy')
        raw, times, metadata = read_raw(safe_path(record['raw_path']))
        if metadata['observer_contract'] != recipe['observer_contract']:
            raise ValueError('prepared observer contract changed')
        if not path.exists():
            missing += 1
            continue
        logp = np.load(path)
        if len(logp) != len(times):
            # Teacher and student must be the same instants; never stretch one onto the other.
            missing += 1
            continue
        x = boundary_features(raw, times, geometry)
        y = bio_targets(times, record['intervals'], True)
        breaks = [0, *(np.flatnonzero(np.diff(times) > .26) + 1).tolist(), len(times)]
        for first, last in zip(breaks, breaks[1:]):
            for start in range(first, last, recipe['chunk_frames']):
                stop = min(start + recipe['chunk_frames'], last)
                left = max(first, start - left_context)
                right = min(last, stop + recipe['lookahead'])
                cut = slice(left, right - recipe['lookahead'])
                targets, soft = y[cut].copy(), logp[cut].copy()
                keep = np.zeros(len(targets), bool)
                keep[start - left:] = True          # the pinned chunk's own frames
                targets[~keep] = -1
                if keep.any():
                    output[record['split']].append((x[left:right], targets, soft, keep))
    if missing:
        print('  %d records without an aligned teacher trace were dropped' % missing, flush=True)
    return output


def distil_batch(rows, lookahead, device):
    length, dim = max(len(x) for x, _, _, _ in rows), rows[0][0].shape[1]
    n = length - lookahead
    xs = np.zeros((len(rows), length, dim), np.float32)
    ys = np.full((len(rows), n), -1, np.int64)
    ts = np.zeros((len(rows), n, 4), np.float32)
    ks = np.zeros((len(rows), n), bool)
    for i, (x, y, soft, keep) in enumerate(rows):
        xs[i, :len(x)], ys[i, :len(y)], ts[i, :len(soft)], ks[i, :len(keep)] = x, y, soft, keep
    return (torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device),
            torch.from_numpy(ts).to(device), torch.from_numpy(ks).to(device))


def distil_loss(logits, hard, soft, keep, weight, alpha, mode='all', floor=0.):
    """alpha weights the teacher term; mode decides WHICH frames the teacher may speak for.

    'all'  the teacher competes with the annotation on every covered frame.
    'fill' the teacher speaks only where the annotation is silent. Round 4 showed the
           noncausal teacher makes a causal student hedge (196 intervals for 226 signs),
           so here it supplies the 61% that has no labels and never overrides the 39%
           that does. floor>0 additionally ignores frames the teacher is unsure about.
    """
    parts = []
    valid = hard >= 0
    region = keep if mode == 'all' else (keep & ~valid)
    if floor > 0:
        region = region & (soft.exp().amax(-1) >= floor)
    if alpha > 0 and region.any():
        parts.append(alpha * F.kl_div(F.log_softmax(logits[region], dim=-1),
                                      soft[region].exp(), reduction='batchmean'))
    if alpha < 1 or mode == 'fill':
        if not valid.any():
            raise ValueError('batch has no supervised frames')
        scale = 1. if mode == 'fill' else (1 - alpha)
        parts.append(scale * F.cross_entropy(logits[valid], hard[valid], weight=weight))
    if not parts:
        raise ValueError('loss has no terms')
    return sum(parts)


# arm -> (alpha, mode, confidence floor)
ARMS = {'bio_gap': (0., 'all', 0.), 'distill': (1., 'all', 0.), 'distill_ce': (.5, 'all', 0.),
        'distill_fill': (1., 'fill', 0.), 'distill_fill_conf': (1., 'fill', .9),
        'distill_fill_half': (.5, 'fill', 0.)}


def train_arm(setting, rows, recipe, seed, device, hidden, dilations):
    alpha, mode, floor = setting
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = HeadBoundary(input_dim=rows['train'][0][0].shape[1], hidden=hidden,
                         lookahead=recipe['lookahead'], dilations=dilations, channels=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=recipe['lr'])
    weight = bio_weights([(x, y) for x, y, _, _ in rows['train']], device)
    best, best_state = float('inf'), None
    for epoch in range(1, recipe['epochs'] + 1):
        model.train()
        order = rng.permutation(len(rows['train']))
        for s in range(0, len(order), recipe['batch_size']):
            chosen = [rows['train'][i] for i in order[s:s + recipe['batch_size']]]
            x, y, soft, keep = distil_batch(chosen, recipe['lookahead'], device)
            loss = distil_loss(model(x), y, soft, keep, weight, alpha, mode, floor)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        # Selection always uses the annotation, never the teacher: a distilled student that
        # merely imitates a wrong teacher must not be allowed to win on that basis.
        total = count = 0.
        with torch.inference_mode():
            for s in range(0, len(rows['calibration']), 16):
                chunk = rows['calibration'][s:s + 16]
                x, y = bio_batch([(a, b) for a, b, _, _ in chunk], recipe['lookahead'], device)
                n = int((y >= 0).sum())
                total += float(masked_bio_loss(model(x), y, weight)) * n
                count += n
        if total / count < best:
            best = total / count
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.cpu().eval(), best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default=','.join(str(17621 + i) for i in range(12)))
    parser.add_argument('--geometry', action='store_true')
    parser.add_argument('--arms', default='bio_gap,distill,distill_ce')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dilations', default='1,2,4,8,16')
    parser.add_argument('--limit-clips', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--out', default='round4_distil')
    args_cli = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    dilations = tuple(int(d) for d in args_cli.dilations.split(','))

    cache_teacher()
    recipe = load_recipe()
    if args_cli.epochs:
        recipe = {**recipe, 'epochs': args_cli.epochs}
    gate = teacher_agreement(recipe)
    print('teacher vs annotation: frame agreement %.1f%% over %d supervised frames; '
          'sign frames recalled %.1f%%'
          % (100 * gate['frame_agreement'], gate['supervised_frames'],
             100 * gate['sign_frames_recalled']), flush=True)

    pool = json.loads(POOL.read_text())
    shas = {r['video_sha256'] for r in pool['records']}
    combined = json.loads(COMBINED.read_text())
    rows_eval = sorted([r for r in combined['records'] if r['video_sha256'] in shas],
                       key=lambda r: r['source_item_id'])
    if args_cli.limit_clips:
        rows_eval = rows_eval[:args_cli.limit_clips]
    print('tuning pool: %d clips, %d reference signs'
          % (len(rows_eval), sum(len(r['target_sequence']) for r in rows_eval)), flush=True)

    args = arguments(rows_eval[0]['video_path'], REPORT / 'sessions')
    args.no_motion_trim = True
    reel = build_components(args)['classifier']
    reel.args.no_motion_trim = reel.full.args.no_motion_trim = True
    detector = AppleVisionDetector(args.minimum_point_confidence)

    tick, prepared = time.perf_counter(), []
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

    chunks = build_chunks(recipe, args_cli.geometry, 2 * sum(dilations))
    print('chunks: %s\n' % {k: len(v) for k, v in chunks.items()}, flush=True)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    seeds = [int(s) for s in args_cli.seeds.split(',')]
    results = []
    print('%-12s %9s %7s %8s %8s %7s %7s' % ('arm', 'meanWER', 'SD', 'min', 'correct', 'ins', 'ivals'))
    for arm in args_cli.arms.split(','):
        runs = []
        for seed in seeds:
            model, cal = train_arm(ARMS[arm], chunks, recipe, seed, device, args_cli.hidden, dilations)
            summary, per_clip = head_evaluate(model, 'bio_gap', args_cli.geometry, prepared, reel, args)
            runs.append(dict(seed=seed, calibration_loss=cal, summary=summary))
            print('   %-11s seed %d  WER %.2f%%  cal %.4f' % (arm, seed, 100 * summary['wer'], cal), flush=True)
        w = np.array([r['summary']['wer'] for r in runs])
        results.append(dict(arm=arm, setting=list(ARMS[arm]), runs=runs, mean_wer=float(w.mean()),
                            sd=float(w.std(ddof=1)), best=float(w.min())))
        r = results[-1]
        print('%-12s %8.2f%% %6.2f %7.2f%% %8.1f %7.1f %7.1f' % (
            arm, 100 * r['mean_wer'], 100 * r['sd'], 100 * r['best'],
            np.mean([x['summary']['correct'] for x in runs]),
            np.mean([x['summary']['insertions'] for x in runs]),
            np.mean([x['summary']['proposed_intervals'] for x in runs])), flush=True)
        (REPORT / (args_cli.out + '.json')).write_text(json.dumps(dict(
            round=4, objective='distillation from the frozen DGS BIO teacher into the live TCN; '
            'selected on mean end-to-end WER over the 89-clip local tuning pool',
            teacher=dict(weights='artifacts/models/pose_boundary_dgs_2026', frozen=True, **gate),
            trunk=dict(hidden=args_cli.hidden, dilations=list(dilations)), seeds=seeds,
            reel=reel.provenance(), results=results), indent=1) + '\n')

    ranked = sorted(results, key=lambda r: r['mean_wer'])
    print('\nbest: %s at %.2f%% mean WER (SD %.2f)'
          % (ranked[0]['arm'], 100 * ranked[0]['mean_wer'], 100 * ranked[0]['sd']))


if __name__ == '__main__':
    main()
