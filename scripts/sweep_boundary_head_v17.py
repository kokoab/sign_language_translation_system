"""Round 2 of the boundary chain: output formulation and supervision coverage.

Round 1 showed architecture is not the lever: eight variants from 66K to 530K parameters
and receptive fields from 14 to 126 frames spanned 68.8-75.2% WER while the seed noise was
2.2 points (F(7,8)=1.54, p=0.28). This round changes what the network is asked to predict
and which frames are supervised at all, holding the trunk fixed at the round-1 winner.

The pinned recipe states its own limitation: "negatives only inside accepted intervals;
all other values ignored ... no supervised outside class". 61% of every clip is therefore
unsupervised at training time but scored at inference, and the decoder proposes ~280
intervals for 226 reference signs.

Arms (identical trunk, identical data, identical live decoding path):
  edges            pinned control: 2 sigmoid edges, +-50ms band, supervision inside signs only
  edges_wide       as above with a +-150ms band; isolates POSITIVE SPARSITY alone
  edges_gap_o      pinned band, plus negative supervision across inter-sign gaps <= 0.4s;
                   isolates OUTSIDE SUPERVISION alone
  bio_gap          4-class UNK/O/B/I at matched supervision coverage (46.9% of frames
                   against edges_gap_o's 48.5%); isolates OUTPUT FORMULATION

Only gaps <= 0.4s are labelled non-sign. Median sign duration is 0.33s, so a shorter gap
cannot conceal an unannotated sign plus its transitions. Longer gaps, lead-in and tail-out
stay unknown, because 96% of these records come from asllrp_other_ctc, whose glossing is
not exhaustive. Nothing is promoted; the pinned trainer, recipe and model file are unread.
"""
from __future__ import annotations
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from active.v17.temporal_boundary_v17 import (
    BoundaryStream, BoundaryDecoder, boundary_features, boundary_targets,
    masked_boundary_loss, validate_clock, START, END)
from active.v17.stage1_window_v17 import raw_observation_features
from scripts.train_temporal_boundary_v17 import load_recipe, read_raw, safe_path
from scripts.evaluate_temporal_boundary_v17 import arguments, observations, edit_counts, summarize
from scripts.live_boundary_v17 import classify_interval
from scripts.live_reel_stage1_v17 import build_components
from active.v17.extract_v17 import AppleVisionDetector
from scripts.sweep_boundary_tcn_v17 import SweptBoundary, batch as edge_batch, positive_weights

REPORT = ROOT / 'artifacts/reports/boundary_tcn_sweep_v17_20260922'
MODELS = ROOT / 'artifacts/models/boundary_tcn_sweep_v17_20260922'
POOL = ROOT / 'artifacts/reports/pretrained_bio_final_v17_20260922/evaluation_poses.json'
COMBINED = ROOT / 'data/local/combined_dataset_v17_20260922/manifest.json'

UNK, O, B, I = 0, 1, 2, 3          # upstream sign_language_segmentation BIO indices
GAP_MAX = .4                        # honest non-sign window; see module docstring
MIN_SECONDS, MAX_SECONDS = .15, 4.  # identical bounds to BoundaryDecoder


def quiet_gaps(times, intervals, gap_max=GAP_MAX):
    """Frames lying in an inter-sign gap short enough to be a transition, not a lost sign."""
    quiet = np.zeros(len(times), bool)
    for (_, end), (start, _) in zip(sorted(intervals), sorted(intervals)[1:]):
        if 0 < start - end <= gap_max:
            quiet |= (times > end + 1e-8) & (times < start - 1e-8)
    return quiet


def edge_targets(times, intervals, band=.05, gap_o=False):
    """boundary_targets with a configurable band, optionally negative on short gaps."""
    times = validate_clock(times)
    y = np.full((len(times), 2), -1, np.float32)
    for start, end in intervals:
        if not np.isfinite([start, end]).all() or end <= start:
            raise ValueError('invalid sign interval')
        inside = (times >= start - 1e-8) & (times <= end + 1e-8)
        y[inside[:, None] & (y < 0)] = 0
        for edge, channel in ((start, START), (end, END)):
            if times[0] - .025 <= edge <= times[-1] + .025:
                y[np.abs(times - edge) <= band + 1e-6, channel] = 1
    if gap_o:
        y[quiet_gaps(times, intervals)] = 0    # no start and no end during a transition
    return y


def bio_targets(times, intervals, gap_o=True):
    """4-class targets over exactly the frames edge_targets(gap_o=True) supervises."""
    times = validate_clock(times)
    y = np.full(len(times), -1, np.int64)
    for start, end in sorted(intervals):
        inside = (times >= start - 1e-8) & (times <= end + 1e-8)
        if not inside.any():
            continue
        y[inside] = I
        y[np.flatnonzero(inside)[0]] = B
    if gap_o:
        y[quiet_gaps(times, intervals)] = O
    return y


TARGETS = {
    'edges':       dict(channels=2, fn=lambda t, iv: edge_targets(t, iv, .05, False)),
    'edges_wide':  dict(channels=2, fn=lambda t, iv: edge_targets(t, iv, .15, False)),
    'edges_gap_o': dict(channels=2, fn=lambda t, iv: edge_targets(t, iv, .05, True)),
    'bio_gap':     dict(channels=4, fn=lambda t, iv: bio_targets(t, iv, True)),
}


def build_chunks(recipe, geometry, left_context, target_fn):
    """Mirror of the pinned data() with configurable left context and target builder."""
    output = {'train': [], 'calibration': [], 'validation': []}
    for record in recipe['records']:
        raw, times, metadata = read_raw(safe_path(record['raw_path']))
        if metadata['observer_contract'] != recipe['observer_contract']:
            raise ValueError('prepared observer contract changed')
        x = boundary_features(raw, times, geometry)
        y = target_fn(times, record['intervals'])
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
    """The control arm must reproduce the pinned supervision value for value."""
    for record in recipe['records'][:150]:
        raw, times, _ = read_raw(safe_path(record['raw_path']))
        mine = edge_targets(times, record['intervals'], .05, False)
        pinned = boundary_targets(times, record['intervals'], [], False)
        if not np.array_equal(mine, pinned):
            raise ValueError('control arm targets diverge from the pinned builder')
    return True


class HeadBoundary(SweptBoundary):
    """Round-1 trunk with a configurable number of output channels."""

    def __init__(self, input_dim=450, hidden=64, lookahead=4, dilations=(1, 2, 4, 8, 16),
                 channels=2):
        super().__init__(input_dim=input_dim, hidden=hidden, lookahead=lookahead,
                         dilations=dilations)
        if channels not in (2, 4):
            raise ValueError('channels must be 2 or 4')
        self.output = nn.Conv1d(hidden, channels, 1)
        self.config['channels'] = channels


class BioSegmentDecoder:
    """Streaming form of upstream likeliest_probs_to_segments + filter_segments.

    Causal and emits at the frame that closes a segment, exactly like BoundaryDecoder.
    Bounds are the decoder's own: shorter than 0.15s is noise, longer than 4s is not a sign.
    """

    def __init__(self, minimum=MIN_SECONDS, maximum=MAX_SECONDS):
        self.minimum, self.maximum = minimum, maximum
        self.reset()

    def reset(self):
        self.start = None
        self.last_time = -np.inf
        self.previous_time = -np.inf

    def _close(self, end):
        event = []
        if self.start is not None and self.minimum - 1e-8 <= end - self.start <= self.maximum:
            event = [dict(start_seconds=self.start, end_seconds=float(end))]
        self.start = None
        return event

    def update(self, time, label):
        if not np.isfinite(time) or time <= self.last_time:
            raise ValueError('increasing timestamps required')
        if time - self.last_time > .26:
            self.reset()
        events = []
        if label in (B, I):
            if self.start is None:
                self.start = float(time)
            elif time - self.start > self.maximum:
                self.start = None          # expiry discards; never fabricates a sign
        else:
            events.extend(self._close(self.previous_time if self.start is not None else time))
        self.previous_time = float(time)
        self.last_time = float(time)
        return events


class BioStream:
    """BoundaryStream with the BIO readout; buffering and reset rules are identical."""

    def __init__(self, model, hand_geometry=True):
        self.model = model.eval()
        self.hand_geometry = hand_geometry
        self.decoder = BioSegmentDecoder()
        self.reset()

    def reset(self):
        self.raw, self.raw_times = deque(), deque()
        self.features = deque(maxlen=self.model.left_context + 1)
        self.times = deque(maxlen=self.model.lookahead + 1)
        self.decoder.reset()
        self.last_time = -np.inf

    @torch.inference_mode()
    def update(self, raw_frame, seconds):
        if not np.isfinite(seconds) or seconds <= self.last_time:
            raise ValueError('stream timestamps must increase')
        if seconds - self.last_time > .26:
            self.reset()
        self.last_time = float(seconds)
        self.raw.append(np.asarray(raw_frame, np.float32))
        self.raw_times.append(float(seconds))
        while len(self.raw_times) > 2 and self.raw_times[1] < seconds - 1.2:
            self.raw.popleft(); self.raw_times.popleft()
        feature = boundary_features(np.asarray(self.raw), self.raw_times, self.hand_geometry)[-1]
        self.features.append(feature)
        self.times.append(float(seconds))
        if len(self.features) <= self.model.lookahead:
            return None
        x = torch.as_tensor(np.asarray(self.features)[None])
        label = int(self.model(x)[0, -1].argmax())
        source_time = self.times[0]
        return dict(seconds=source_time, available_seconds=float(seconds),
                    label=label, events=self.decoder.update(source_time, label))


def bio_batch(rows, lookahead, device):
    length, dim = max(len(x) for x, _ in rows), rows[0][0].shape[1]
    xs = np.zeros((len(rows), length, dim), np.float32)
    ys = np.full((len(rows), length - lookahead), -1, np.int64)
    for i, (x, y) in enumerate(rows):
        xs[i, :len(x)], ys[i, :len(y)] = x, y
    return torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device)


def bio_weights(rows, device):
    y = np.concatenate([y for _, y in rows])
    counts = np.array([(y == c).sum() for c in (UNK, O, B, I)], np.float64)
    counts[UNK] = 0
    share = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.)
    return torch.as_tensor(np.clip(share / max(share.max(), 1e-9) * 10, 0, 10),
                           dtype=torch.float32, device=device)


def masked_bio_loss(logits, targets, weight):
    valid = targets >= 0
    if not valid.any():
        raise ValueError('batch has no supervised frames')
    return F.cross_entropy(logits[valid], targets[valid], weight=weight)


def train_arm(arm, rows, recipe, seed, device, hidden, dilations):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    channels = TARGETS[arm]['channels']
    model = HeadBoundary(input_dim=rows['train'][0][0].shape[1], hidden=hidden,
                         lookahead=recipe['lookahead'], dilations=dilations,
                         channels=channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=recipe['lr'])
    make_batch = bio_batch if channels == 4 else edge_batch
    weight = (bio_weights if channels == 4 else positive_weights)(rows['train'], device)
    criterion = masked_bio_loss if channels == 4 else masked_boundary_loss
    best, best_state = float('inf'), None
    for epoch in range(1, recipe['epochs'] + 1):
        model.train()
        order = rng.permutation(len(rows['train']))
        for s in range(0, len(order), recipe['batch_size']):
            chosen = [rows['train'][i] for i in order[s:s + recipe['batch_size']]]
            x, y = make_batch(chosen, recipe['lookahead'], device)
            loss = criterion(model(x), y, weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        total = count = 0.
        with torch.inference_mode():
            for s in range(0, len(rows['calibration']), 16):
                x, y = make_batch(rows['calibration'][s:s + 16], recipe['lookahead'], device)
                n = int((y >= 0).sum())
                total += float(criterion(model(x), y, weight)) * n
                count += n
        if total / count < best:
            best = total / count
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.cpu().eval(), best


def evaluate(model, arm, geometry, prepared, reel, args):
    records = []
    for row, obs in prepared:
        raw, times = raw_observation_features(obs)
        stream = (BioStream if TARGETS[arm]['channels'] == 4 else BoundaryStream)(model, geometry)
        events = []
        for i in range(len(times)):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='17621,17622')
    parser.add_argument('--geometry', action='store_true')
    parser.add_argument('--arms', default='edges,edges_wide,edges_gap_o,bio_gap')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dilations', default='1,2,4,8,16', help='round-1 winner h64_d5')
    parser.add_argument('--limit-clips', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--out', default='round2', help='report basename')
    parser.add_argument('--no-save', action='store_true', help='skip checkpoints')
    args_cli = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    dilations = tuple(int(d) for d in args_cli.dilations.split(','))

    recipe = load_recipe()
    tick = time.perf_counter()
    parity_check(recipe, args_cli.geometry)
    print('control-arm targets match the pinned builder value for value (%.0fs)'
          % (time.perf_counter() - tick), flush=True)

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
    seeds = [int(s) for s in args_cli.seeds.split(',')]
    results = []
    print('%-14s %9s %9s %7s %8s %7s %7s' % (
        'arm', 'sup_frames', 'meanWER', 'spread', 'correct', 'ins', 'ivals'))
    for arm in args_cli.arms.split(','):
        chunks = build_chunks(recipe, args_cli.geometry, 2 * sum(dilations), TARGETS[arm]['fn'])
        supervised = int(sum((y >= 0).sum() for _, y in chunks['train']))
        runs = []
        for seed in seeds:
            model, cal = train_arm(arm, chunks, recipe, seed, device, args_cli.hidden, dilations)
            summary, per_clip = evaluate(model, arm, args_cli.geometry, prepared, reel, args)
            if not args_cli.no_save:
                torch.save(dict(format='boundary_head_sweep_v17', arm=arm, seed=seed,
                                hand_geometry=args_cli.geometry, model_config=model.config,
                                calibration_loss=cal, model_state_dict=model.state_dict()),
                           MODELS / ('head_%s_seed_%d.pth' % (arm, seed)))
            runs.append(dict(seed=seed, calibration_loss=cal, summary=summary,
                             clips=[dict(id=c['id'], intervals=c['intervals'],
                                         reference=len(c['reference']), **c['metrics'])
                                    for c in per_clip]))
            print('   %-12s seed %d  WER %.2f%%  cal %.4f' % (arm, seed, 100 * summary['wer'], cal), flush=True)
        wers = [r['summary']['wer'] for r in runs]
        results.append(dict(arm=arm, supervised_frames=supervised, runs=runs,
                            mean_wer=float(np.mean(wers)), seed_spread=float(max(wers) - min(wers))))
        r = results[-1]
        print('%-14s %9d %8.2f%% %6.2f%% %8.1f %7.1f %7.1f' % (
            arm, supervised, 100 * r['mean_wer'], 100 * r['seed_spread'],
            np.mean([x['summary']['correct'] for x in runs]),
            np.mean([x['summary']['insertions'] for x in runs]),
            np.mean([x['summary']['proposed_intervals'] for x in runs])), flush=True)
        (REPORT / (args_cli.out + '.json')).write_text(json.dumps(dict(
            round=2, objective='output formulation and supervision coverage at a fixed trunk; '
            'selected on mean end-to-end WER over the 89-clip local tuning pool',
            trunk=dict(hidden=args_cli.hidden, dilations=list(dilations)), seeds=seeds,
            gap_max_seconds=GAP_MAX, hand_geometry=args_cli.geometry,
            reel=reel.provenance(), results=results), indent=1) + '\n')

    ranked = sorted(results, key=lambda r: r['mean_wer'])
    print('\nbest: %s at %.2f%% mean WER (spread %.2f)'
          % (ranked[0]['arm'], 100 * ranked[0]['mean_wer'], 100 * ranked[0]['seed_spread']))
    print('wrote', REPORT / (args_cli.out + '.json'))


if __name__ == '__main__':
    main()
