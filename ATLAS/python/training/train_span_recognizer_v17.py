"""Fine-tune the unified Reel verifier on decoder-matched continuous-signing spans.

Training spans are the ones the segmental decoder actually proposes (frozen DGS BIO
candidates) plus jittered gold/forced-aligned spans, labelled by IoU >= 0.5 with ASLLRP
curated timing or with forced alignment on local phrase transcripts. Inputs are the exact
live verifier inputs captured by scripts/segmental_lab_v17.py, so train and test share one
feature path. Isolated Citizen/SemLex/local clips are replayed with KD to the base model and
guarded by the same floors as the 2026-09-25 recipe.

Selection: tuning-set decode WER (local tuning pool, never the held-out test), subject to
isolated floors. Nothing is promoted by this script.
"""
from __future__ import annotations

import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.8'  # an imported trainer would otherwise cap MPS at 12%
os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.4'
import argparse
import copy
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from active.v17.export_unified_multimodal_coreml_v17 import load_model
from active.v17.train_unfrozen_phrase_adapt_v17 import isolated_raw, tensors, logits_for, evaluate, SOURCES
from scripts import segmental_lab_v17 as lab

BASE = ROOT / 'artifacts/models/stage1_v17_unified_phrase_activity_adapt_reel_v2/best_model.pth'
RAW_CACHE = ROOT / 'artifacts/generated/unfrozen_phrase_adapt_v17'
# Isolated retention: every domain >= 90 (user, 2026-09-25), SemLex (already <90) keeps its
# incumbent 89.16, and no domain falls more than 1.0 point below the shipped reel_v2.
FLOORS = dict(citizen=95.03, semlex=89.16, local=96.03)


class LandmarkOnlySpan(torch.nn.Module):
    """Android landmark-only recognizer: the v17 landmark model behind the 4-input span signature.

    Hand embeddings, validity and boxes are accepted and ignored, so the trainer, replay KD, tuning
    decode and isolated floors run unchanged.
    """

    def __init__(self, landmark):
        super().__init__()
        self.net = landmark

    def forward(self, landmarks, hand_embeddings=None, hand_valid=None, hand_boxes=None):
        return self.net(landmarks)


def load_landmark_only(path):
    from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('format') != 'slt_stage1_v17':
        raise ValueError(f'not a v17 landmark checkpoint: {path}')
    net = SLTStage1V17(Stage1V17Config(**checkpoint['model_config']))
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return LandmarkOnlySpan(net.eval()), checkpoint


EXCLUDE_PREFIX = False  # local_a predates the prefix spans added for (rejected) recognizer B
ALLOWED_TRAIN_VIDEOS = None  # optional recipe-manifest restriction (video_sha256 set)


def span_dataset(names, labels, drop_other=True):
    index = {g: i for i, g in enumerate(labels)}
    keep = {k: [] for k in ('landmarks', 'hand_embeddings', 'hand_valid', 'hand_boxes', 'targets')}
    sources = []
    videos = set()
    for name in names:
        for row in lab.rows_for(name):
            if ALLOWED_TRAIN_VIDEOS is not None and row['video_sha256'] not in ALLOWED_TRAIN_VIDEOS:
                continue
            path = lab.CACHE / 'train_labels' / (lab.key(row['source_item_id']) + '.json')
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            memo = lab.load_memo(row)
            for item in payload['spans']:
                if EXCLUDE_PREFIX and item.get('prefix'):
                    continue
                gloss = item['gloss']
                if gloss not in index:
                    if drop_other:
                        continue
                r = memo.get(tuple(item['span']))
                if r is None or r.get('inputs') is None:
                    continue
                for k in ('landmarks', 'hand_embeddings', 'hand_valid', 'hand_boxes'):
                    keep[k].append(r['inputs'][k])
                keep['targets'].append(index[gloss])
                sources.append(name)
            videos.add(row['video_sha256'])
    values = {k: np.stack(v) if k != 'targets' else np.asarray(v, np.int64) for k, v in keep.items()}
    return values, sources, videos


def fragment(lm, he, hv, hb, low=.5, high=.85, generator=None):
    """Partial views of whole isolated clips (prefix, suffix or middle; `low`-`high` of the trimmed
    clip), resampled back to the model's 32 landmark / 16 hand frames. A decoder span that covers
    only part of a sign (e.g. the raised end of BIG) is labelled with the parent sign instead of
    being read as whichever sign it resembles (GO)."""
    n, t_lm, t_hand = lm.shape[0], lm.shape[1], he.shape[1]
    device = lm.device
    r = torch.empty(n, device=device).uniform_(low, high)
    where = torch.randint(0, 3, (n,), device=device)                     # 0 prefix, 1 suffix, 2 middle
    span = r * (t_lm - 1)
    start = torch.where(where == 0, torch.zeros_like(span), torch.where(
        where == 1, (t_lm - 1) - span, torch.rand(n, device=device) * ((t_lm - 1) - span)))
    pos = start[:, None] + span[:, None] * torch.linspace(0, 1, t_lm, device=device)[None]
    lo = pos.floor().long().clamp(0, t_lm - 1); hi = (lo + 1).clamp(max=t_lm - 1)
    w = (pos - lo.float())[..., None, None]
    take = lambda x, i: torch.gather(x, 1, i[:, :, None, None].expand(-1, -1, *x.shape[2:]))
    out = take(lm, lo).clone()
    out[..., :3] = take(lm, lo)[..., :3] * (1 - w) + take(lm, hi)[..., :3] * w
    out[..., 3:] = take(lm, pos.round().long().clamp(0, t_lm - 1))[..., 3:]
    hpos = (start[:, None] + span[:, None] * torch.linspace(0, 1, t_hand, device=device)[None]) * (t_hand - 1) / (t_lm - 1)
    hi_ = hpos.round().long().clamp(0, t_hand - 1)
    hand = lambda x: torch.gather(x, 1, hi_.view(n, t_hand, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:]))
    return out, hand(he), hand(hv), hand(hb)


def in_context(lm, he, hv, hb, edge=.15, generator=None):
    """Sentence context for isolated clips: the core of each clip (15-85%) between the late part of another
    clip (70-85%, the previous sign ending) and the early part of a third (15-30%, the next sign starting),
    resampled to the model's 32/16 frames; the label stays the middle sign. Isolated clips always enter
    from and leave to rest, so a sign leaving upward into the next one (HAVE before BIG) otherwise reads
    as a sign whose own movement goes up (ANGRY)."""
    n, t_lm, t_hand = lm.shape[0], lm.shape[1], he.shape[1]
    device = lm.device
    prev, nxt = torch.roll(torch.arange(n, device=device), 1), torch.roll(torch.arange(n, device=device), 2)
    k = torch.empty(n, device=device).uniform_(.10, .20)                    # share of each neighbour
    u = torch.linspace(0, 1, t_lm, device=device)[None].expand(n, -1)
    a, b = k[:, None], 1 - k[:, None]
    # source clip and position (0..1 of that clip) for every output frame
    src = torch.where(u < a, prev[:, None], torch.where(u > b, nxt[:, None], torch.arange(n, device=device)[:, None]))
    pos = torch.where(u < a, .70 + .15 * u / a, torch.where(u > b, .15 + .15 * (u - b) / a, edge + (1 - 2 * edge) * (u - a) / (b - a)))
    li = (pos * (t_lm - 1)).round().long().clamp(0, t_lm - 1)
    out = lm[src, li]
    hu = torch.linspace(0, 1, t_hand, device=device)[None].expand(n, -1)
    hsrc = torch.where(hu < a, prev[:, None], torch.where(hu > b, nxt[:, None], torch.arange(n, device=device)[:, None]))
    hpos = torch.where(hu < a, .70 + .15 * hu / a, torch.where(hu > b, .15 + .15 * (hu - b) / a, edge + (1 - 2 * edge) * (hu - a) / (b - a)))
    hi = (hpos * (t_hand - 1)).round().long().clamp(0, t_hand - 1)
    return out, he[hsrc, hi], hv[hsrc, hi], hb[hsrc, hi]


def tuning_wer(model, data, labels, cfg, device):
    """Rescore the tuning memos with the current model and decode."""
    model.eval()
    rescored = []
    for row, sign, memo in data:
        keys = [k for k, r in memo.items() if r is not None and r.get('inputs') is not None]
        vals = {k: [] for k in ('landmarks', 'hand_embeddings', 'hand_valid', 'hand_boxes')}
        for k in keys:
            for f in vals:
                vals[f].append(memo[k]['inputs'][f])
        new = {k: (None if r is None else dict(r)) for k, r in memo.items()}
        if keys:
            outs = []
            with torch.inference_mode():
                for i in range(0, len(keys), 128):
                    sl = slice(i, i + 128)
                    outs.append(model(torch.from_numpy(np.stack(vals['landmarks'][sl]).astype(np.float32)).to(device),
                                      torch.from_numpy(np.stack(vals['hand_embeddings'][sl]).astype(np.float32)).to(device),
                                      torch.from_numpy(np.stack(vals['hand_valid'][sl])).to(device),
                                      torch.from_numpy(np.stack(vals['hand_boxes'][sl]).astype(np.float32)).to(device)).float().cpu().numpy())
            out = np.concatenate(outs)
            for k, l in zip(keys, out):
                new[k]['v_raw'] = l
        for k, r in new.items():
            if r is not None and r.get('inputs') is None:
                r['v_raw'] = None
        rescored.append((row, sign, new))
    summary, _ = lab.evaluate_decoder(rescored, labels, cfg['lookahead'], cfg)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--base', type=Path, default=BASE)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--samples', type=int, default=12000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--encoder-lr', type=float, default=1e-5)
    ap.add_argument('--head-lr', type=float, default=2e-5)
    ap.add_argument('--span-share', type=float, default=.5)
    ap.add_argument('--freeze-encoders', action='store_true')
    ap.add_argument('--sets', default='local_train,asllrp_train')
    ap.add_argument('--span-split', default='', help='relative shares of each span set, e.g. 0.6,0.4')
    ap.add_argument('--seed', type=int, default=27927)
    ap.add_argument('--extractor', choices=('apple', 'mediapipe_full'), default='apple')
    ap.add_argument('--mediapipe-root', type=Path, default=None)
    ap.add_argument('--raw-cache', type=Path, default=RAW_CACHE)
    ap.add_argument('--landmark-only', action='store_true',
                    help='--base is a v17 landmark checkpoint; train without hand images (Android low-end mode)')
    ap.add_argument('--exclude-prefix-spans', action='store_true', help='train on the local_a span set (no prefix spans)')
    ap.add_argument('--recipe-manifest', type=Path, default=None,
                    help='restrict training videos to this phrase-segment recipe manifest\'s train split')
    ap.add_argument('--floors', default='', help='JSON isolated floors; default the Apple 2026-09-25 floors')
    ap.add_argument('--fragment-prob', type=float, default=0.,
                    help='share of isolated replay samples replaced by a 50-85%% partial view with the parent label')
    ap.add_argument('--fragment-low', type=float, default=.5, help='shortest partial view kept with the parent label')
    ap.add_argument('--reject-prob', type=float, default=0., help='share of replay samples replaced by a tiny piece')
    ap.add_argument('--reject-range', default='0.10,0.25', help='piece length range for --reject-prob')
    ap.add_argument('--reject-weight', type=float, default=.5, help='weight of the flat (no sign) target on tiny pieces')
    ap.add_argument('--boost-classes', default='', help='comma-separated glosses sampled --boost-factor times more often')
    ap.add_argument('--boost-factor', type=float, default=1.)
    ap.add_argument('--boost-context', type=float, default=0., help='in_context probability for boosted-class replay samples')
    ap.add_argument('--context-prob', type=float, default=0.,
                    help='share of isolated replay samples placed between neighbouring clips (in_context)')
    ap.add_argument('--cfg', default='{"lookahead":8,"alpha":1.0,"w_r":0.5,"w_a":0.5,"w_b":0.5,"c":-1,"log_theta":-1.0498}')
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    global FLOORS, EXCLUDE_PREFIX, ALLOWED_TRAIN_VIDEOS
    EXCLUDE_PREFIX = args.exclude_prefix_spans
    if args.recipe_manifest is not None:
        recipe = json.loads(args.recipe_manifest.read_text())
        if recipe.get('format') != 'slt_v17_phrase_segment_recipe_manifest' or not recipe.get('recipe_training_ready'):
            raise ValueError('not a training-ready phrase-segment recipe manifest')
        ALLOWED_TRAIN_VIDEOS = {e['video_sha256'] for e in recipe['entries'] if e['split'] == 'train'}
    if args.floors:
        FLOORS = json.loads(args.floors)
    if args.extractor == 'mediapipe_full':
        # Android family: MediaPipe span inputs (same DGS-teacher span keys and Apple train labels),
        # MediaPipe isolated replay, and a raw cache that is never the Apple one.
        if args.mediapipe_root is None or args.raw_cache == RAW_CACHE or not args.floors:
            raise ValueError('mediapipe_full needs --mediapipe-root, --floors and a new --raw-cache')
        spans_root = args.mediapipe_root / 'continuous' / 'spans'
        lab.memo_path = lambda row: spans_root / (lab.key(row['source_item_id']) + '.pkl')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    cfg = json.loads(args.cfg)

    model, checkpoint = load_landmark_only(args.base) if args.landmark_only else load_model(args.base)
    labels = [l for l, _ in sorted(checkpoint['label_to_index'].items(), key=lambda r: r[1])]
    spans, span_sources, train_videos = span_dataset(args.sets.split(','), labels)
    held = {r['video_sha256'] for r in lab.rows_for('test')} | {r['video_sha256'] for r in lab.rows_for('tune')}
    if train_videos & held:
        raise ValueError('training spans leak into test/tune: %d' % len(train_videos & held))
    print(json.dumps(dict(span_examples=len(spans['targets']), videos=len(train_videos),
                          by_source=Counter(span_sources), classes=len(set(spans['targets'].tolist())))), flush=True)

    raw = isolated_raw(args.raw_cache, args.extractor, args.mediapipe_root)
    span_names = args.sets.split(',')
    span_parts = []
    src = np.asarray(span_sources)
    for name in span_names:
        keep = src == name
        span_parts.append(tensors({k: v[keep] for k, v in spans.items()}))
    parts = [tensors(raw[f'{s}_train']) for s in SOURCES] + span_parts
    combined = tuple(torch.cat([p[i] for p in parts]) for i in range(5))
    source_ids = torch.cat([torch.full((len(p[4]),), i, dtype=torch.long) for i, p in enumerate(parts)])
    teacher = copy.deepcopy(model).to(device)
    teacher_logits = logits_for(teacher, combined, device)
    teacher.cpu(); del teacher

    iso_share = (1 - args.span_share) / len(SOURCES)
    split = [float(x) for x in args.span_split.split(',')] if args.span_split else [1.] * len(span_names)
    shares = [iso_share] * len(SOURCES) + [args.span_share * w / sum(split) for w in split]
    pair_counts = Counter(zip(source_ids.tolist(), combined[4].tolist()))
    class_counts = Counter(s for s, _ in pair_counts)
    weights = torch.tensor([shares[s] / class_counts[s] / pair_counts[(s, int(t))]
                            for s, t in zip(source_ids.tolist(), combined[4].tolist())], dtype=torch.double)
    boost = torch.tensor([labels.index(c) for c in args.boost_classes.split(',') if c], dtype=torch.long)
    if len(boost) and args.boost_factor != 1:
        # Synthetic emphasis for thin classes (e.g. no local-domain clips): sampled more often, and
        # --boost-context of those draws are placed in synthetic sentence context (in_context).
        weights = torch.where(torch.isin(combined[4], boost), weights * args.boost_factor, weights)
    sampler = WeightedRandomSampler(weights, args.samples, replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(TensorDataset(*combined, teacher_logits, source_ids), batch_size=args.batch,
                        sampler=sampler, num_workers=0)

    if args.landmark_only:  # classifier = head (head LR); everything before it = encoder (encoder LR)
        head_ids = {id(p) for p in model.net.classifier.parameters()}
        encoders = [p for p in model.net.parameters() if id(p) not in head_ids]
        groups = [{'params': list(model.net.classifier.parameters()), 'lr': args.head_lr}]
    else:
        encoders = list(model.landmark_model.parameters()) + list(model.hand_model.parameters())
        groups = [{'params': list(model.fusion_head.parameters()), 'lr': args.head_lr}]
    if args.freeze_encoders:
        for p in encoders:
            p.requires_grad = False
    else:
        groups.append({'params': encoders, 'lr': args.encoder_lr})
    trainable = [p for g in groups for p in g['params']]
    anchors = [p.detach().clone().to(device) for p in trainable]
    model.to(device)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    validation = {s: tensors(raw[f'{s}_val']) for s in SOURCES}

    tune = [(r, lab.load_dgs(r)[0], lab.load_memo(r)) for r in lab.rows_for('tune')]

    def snapshot(epoch, loss):
        domains = {s: evaluate(model, v, device) for s, v in validation.items()}
        ok = all(domains[s]['top1'] >= FLOORS[s] - 1e-9 for s in SOURCES)
        return dict(epoch=epoch, loss=loss, eligible=ok, domains={s: domains[s]['top1'] for s in SOURCES},
                    tune=tuning_wer(model, tune, labels, cfg, device))

    history = [snapshot(0, float('nan'))]
    print(json.dumps(history[-1]), flush=True)
    best, started = None, time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_encoders and not args.landmark_only:
            model.landmark_model.eval(); model.hand_model.eval()
        elif args.freeze_encoders:  # frozen landmark encoder: keep its normalisation statistics fixed
            model.net.eval(); model.net.classifier.train()
        total = seen = 0.
        for lm, he, hv, hb, target, t_logits, source in loader:
            lm, he, hv, hb, target, t_logits, source = (x.to(device) for x in (lm, he, hv, hb, target, t_logits, source))
            replay = source < len(SOURCES)
            partial = torch.zeros_like(replay)
            if args.fragment_prob > 0:
                partial = replay & (torch.rand(len(replay), device=device) < args.fragment_prob)
                if partial.any():
                    f_lm, f_he, f_hv, f_hb = fragment(lm[partial], he[partial], hv[partial], hb[partial], low=args.fragment_low)
                    lm, he, hv, hb = lm.clone(), he.clone(), hv.clone(), hb.clone()
                    lm[partial], he[partial], hv[partial], hb[partial] = f_lm, f_he.to(he.dtype), f_hv, f_hb
            if args.context_prob > 0 or (len(boost) and args.boost_context > 0):
                pool = replay & ~partial
                prob = torch.full((len(replay),), args.context_prob, device=device)
                if len(boost):
                    prob = torch.where(torch.isin(target, boost.to(device)), torch.full_like(prob, args.boost_context), prob)
                ctx = pool & (torch.rand(len(replay), device=device) < prob)
                if ctx.any() and int(pool.sum()) >= 3:
                    # neighbours come from the other replay clips of the batch
                    idx = torch.nonzero(pool).flatten()
                    c_lm, c_he, c_hv, c_hb = in_context(lm[idx], he[idx], hv[idx], hb[idx])
                    keep = ctx[idx]
                    lm, he, hv, hb = lm.clone(), he.clone(), hv.clone(), hb.clone()
                    lm[idx[keep]], he[idx[keep]] = c_lm[keep], c_he[keep].to(he.dtype)
                    hv[idx[keep]], hb[idx[keep]] = c_hv[keep], c_hb[keep]
                    partial = partial | ctx
            reject = torch.zeros_like(replay)
            if args.reject_prob > 0:
                # Tiny pieces of a sign are not a sign: flat target, so the decoder leaves them uncommitted.
                reject = replay & ~partial & (torch.rand(len(replay), device=device) < args.reject_prob)
                if reject.any():
                    lo_, hi_ = (float(v) for v in args.reject_range.split(','))
                    r_lm, r_he, r_hv, r_hb = fragment(lm[reject], he[reject], hv[reject], hb[reject], low=lo_, high=hi_)
                    lm, he, hv, hb = lm.clone(), he.clone(), hv.clone(), hb.clone()
                    lm[reject], he[reject], hv[reject], hb[reject] = r_lm, r_he.to(he.dtype), r_hv, r_hb
            output = model(lm, he.float(), hv, hb)
            keep = ~reject
            hard = F.cross_entropy(output[keep], target[keep]) if keep.any() else output.sum() * 0.
            if reject.any():
                hard = hard + args.reject_weight * (-F.log_softmax(output[reject], 1).mean(1)).mean()
            partial = partial | reject
            replay = replay & ~partial          # the teacher saw the whole clip, not the partial view
            distill = (F.kl_div(F.log_softmax(output[replay] / 2., 1), F.softmax(t_logits[replay] / 2., 1),
                                reduction='batchmean') * 4. if replay.any() else output.sum() * 0.)
            anchor = .01 * sum((p - a).square().mean() for p, a in zip(trainable, anchors))
            loss = hard + distill + anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(target); seen += len(target)
        scheduler.step()
        row = snapshot(epoch, total / seen)
        row['minutes'] = (time.perf_counter() - started) / 60
        history.append(row)
        print(json.dumps(row), flush=True)
        if row['eligible'] and (best is None or row['tune']['wer'] < best['tune']['wer']):
            best = dict(row, state=copy.deepcopy(model.state_dict()))

    args.output_dir.mkdir(parents=True)
    provenance = dict(fragment_low=args.fragment_low, reject_prob=args.reject_prob, reject_range=args.reject_range, reject_weight=args.reject_weight, fragment_prob=args.fragment_prob, context_prob=args.context_prob, boost_classes=args.boost_classes, boost_factor=args.boost_factor, boost_context=args.boost_context, base=str(args.base), recipe_manifest=None if args.recipe_manifest is None else str(args.recipe_manifest), extractor=args.extractor, landmark_only=args.landmark_only, exclude_prefix_spans=args.exclude_prefix_spans, sets=args.sets, epochs=args.epochs, seed=args.seed, cfg=cfg,
                      span_examples=len(spans['targets']), train_video_sha256=sorted(train_videos),
                      selection='tuning-pool decode WER subject to isolated floors; held-out test never used',
                      floors=FLOORS, freeze_encoders=args.freeze_encoders, test_accessed=False)
    (args.output_dir / 'history.json').write_text(json.dumps(dict(provenance=provenance, history=history), indent=1) + '\n')
    if best is None:
        print('NO ELIGIBLE EPOCH', flush=True)
        return
    model.load_state_dict(best.pop('state'))
    selected = copy.deepcopy(checkpoint)
    if args.landmark_only:
        selected['model_state_dict'] = model.net.state_dict()
        selected['epoch'] = int(best['epoch'])
        selected['span_adaptation'] = dict(provenance, selected=best, landmark_only=True)
        selected['test_evaluated'] = False
        torch.save(selected, args.output_dir / 'best_model.pth')
        print(json.dumps(dict(selected_epoch=best['epoch'], tune=best['tune'], domains=best['domains'])), flush=True)
        return
    selected['landmark_model_state_dict'] = model.landmark_model.state_dict()
    selected['hand_model_state_dict'] = model.hand_model.state_dict()
    selected['head_state_dict'] = model.fusion_head.state_dict()
    selected['epoch'] = int(best['epoch'])
    selected['span_adaptation'] = dict(provenance, selected=best)
    selected['test_evaluated'] = False
    torch.save(selected, args.output_dir / 'best_model.pth')
    print(json.dumps(dict(selected_epoch=best['epoch'], tune=best['tune'], domains=best['domains'])), flush=True)


if __name__ == '__main__':
    main()
