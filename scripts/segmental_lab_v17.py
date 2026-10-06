"""Offline lab for transition-aware segmental decoding over frozen DGS BIO + frozen Reel.

Caches, per evaluation video:
  * DGS sign/phrase BIO log-probabilities for every window ending at each 20 Hz frame,
    at every position, so any lookahead 0..10 frames is read from one forward pass;
  * Reel proposal (landmark) and verifier (fast visual) full score vectors for any
    requested span, memoised on disk so decoders are pure numpy afterwards.

Nothing here trains, promotes or changes a live default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.pretrained_boundary_v17 import UPSTREAM, PretrainedBoundary, load_pose, FPS, FRAMES, TARGET
from active.v17.approved_phrase_data_v17 import digest

CACHE = ROOT / 'artifacts/cache/segmental_decoder_v17'
REPORT = ROOT / 'artifacts/reports/segmental_decoder_v17_20260927'
CALIBRATION = ROOT / 'artifacts/reports/boundary_local_calibration_v17_20260922'
EXPANDED = ROOT / 'artifacts/reports/boundary_expanded_eval_v17_20260922'
PREPARED = ROOT / 'artifacts/reports/pose_boundary_transfer_v17_20260922/prepared_manifest.json'
CURATED = None  # resolved lazily from evaluate_temporal_boundary_v17
CONTEXT = .1


def key(item):
    return hashlib.sha256(item.encode()).hexdigest()[:20]


# ----------------------------------------------------------------------------- rows

def rows_for(name):
    """Evaluation sets. 'test' is the established 72-video/186-sign held-out comparison."""
    if name == 'test':
        from scripts.evaluate_temporal_boundary_v17 import evaluation_rows
        from scripts.evaluate_boundary_expanded_v17 import local_rows
        asllrp = evaluation_rows(report=REPORT)
        for r in asllrp:
            r['subset'] = 'asllrp12'
            r['pose_dir'] = EXPANDED / 'evaluation_poses'
        local = local_rows()
        for r in local:
            r['subset'] = 'local60'
            r['pose_dir'] = EXPANDED / 'evaluation_poses'
        return asllrp + local
    if name == 'tune':
        manifest = json.loads((CALIBRATION / 'manifest.json').read_text())
        rows = []
        for phase in ('calibration', 'confirmation'):
            for r in manifest['phases'][phase]:
                r = dict(r, subset='local_tune', pose_dir=CALIBRATION / (phase + '_poses'))
                rows.append(r)
        return rows
    if name == 'asllrp_val':
        return asllrp_validation_rows()
    if name == 'asllrp_train':
        return asllrp_validation_rows(role='train')
    if name == 'local_train':
        return local_training_rows()
    raise ValueError(name)


def local_training_rows():
    """Every approved local phrase clip that is neither held-out test nor decoder tuning."""
    combined = json.loads((ROOT / 'data/local/combined_dataset_v17_20260922/manifest.json').read_text())
    held = {r['video_sha256'] for r in rows_for('test')} | {r['video_sha256'] for r in rows_for('tune')}
    rows, seen = [], set()
    for r in combined['records']:
        if r['source'] != 'local_phrases' or r['video_sha256'] in held or r['video_sha256'] in seen:
            continue
        seen.add(r['video_sha256'])
        rows.append(dict(r, subset='local_train', pose_dir=CACHE / 'train_poses', all_events=[], events=[]))
    return rows


def asllrp_validation_rows(role='validation'):
    """Complete ASLLRP utterances with timed known/OTHER events."""
    from scripts.evaluate_temporal_boundary_v17 import CURATED as curated_path
    curated = json.loads(curated_path.read_text())
    events = {}
    for e in curated['events']:
        events.setdefault(e['item'], []).append(e)
    prepared = json.loads(PREPARED.read_text())['records']
    held = {r['video_sha256'] for r in rows_for('test')}
    rows = []
    for p in prepared:
        if p['video_sha256'] in held:
            continue
        if p['role'] != role or p['source'] not in ('asllrp_other_ctc', 'asllrp_contiguous'):
            continue
        ev = sorted(events.get(p['item'], []), key=lambda e: e['start'])
        if not ev:
            continue
        known = [e for e in ev if e['kind'] == 'known']
        rows.append(dict(source_item_id=p['item'], video_path=p['video_path'], video_sha256=p['video_sha256'],
                         target_sequence=[e['label'] for e in known], all_events=ev, events=known,
                         subset='asllrp_' + role[:5], pose_path=p['pose_path'], pose_sha256=p['pose_sha256'],
                         pose_fps=p['pose_fps'], signer=p.get('signer')))
    return rows


# ----------------------------------------------------------------------------- DGS

_PREPARED = None


def pose_record(row):
    global _PREPARED
    if row.get('pose_path'):
        return dict(pose_path=row['pose_path'], pose_fps=row['pose_fps'])
    if _PREPARED is None:
        _PREPARED = {r['item']: r for r in json.loads(PREPARED.read_text())['records']}
    hit = _PREPARED.get(row['source_item_id'])
    if hit is not None and hit['video_sha256'] == row['video_sha256']:
        return dict(pose_path=hit['pose_path'], pose_fps=hit['pose_fps'])
    path = row['pose_dir'] / (key(row['source_item_id']) + '.pose')
    if not path.exists():
        sys.path.insert(0, str(UPSTREAM))
        from prepare import extract
        record = extract(dict(item=row['source_item_id'], role='validation', video_path=row['video_path'],
                              video_sha256=row['video_sha256'], intervals=[]), row['pose_dir'])
        return record
    from pose_format import Pose
    with path.open('rb') as f:
        pose = Pose.read(f)
    return dict(pose_path=str(path.relative_to(ROOT)), pose_fps=float(pose.body.fps))


def window_at_end(pose, end):
    """Window of FRAMES observations ending at `end` (inclusive); invalid positions zeroed."""
    from pose_format import Pose
    from pose_format.numpy.pose_body import NumPyPoseBody
    from pose_anonymization.data.normalization import normalize_mean_std
    n = len(pose.body.data)
    idx = np.arange(end - FRAMES + 1, end + 1)
    valid = (idx >= 0) & (idx < n)
    idx = np.clip(idx, 0, n - 1)
    raw = pose.body.data[idx].filled(0).copy()
    confidence = pose.body.confidence[idx].copy()
    confidence[~valid] = 0
    window = normalize_mean_std(Pose(pose.header, NumPyPoseBody(FPS, raw, confidence)))
    xyz = window.body.data.filled(0)[:, 0].astype(np.float32)
    velocity = np.zeros_like(xyz)
    velocity[1:] = np.diff(xyz, axis=0) * FPS
    velocity[~valid] = 0
    velocity[np.flatnonzero(valid)[0]] = 0
    features = np.concatenate((xyz, velocity), axis=-1)
    features[~valid] = 0
    return features


class DGS:
    def __init__(self, device='mps'):
        self.model = PretrainedBoundary().to(device).eval()
        self.device = device
        self.times = torch.tensor((np.arange(FRAMES) - TARGET)[None] / FPS, dtype=torch.float32, device=device)

    @torch.inference_mode()
    def all_windows(self, pose):
        """[n, FRAMES, 4] sign and phrase BIO log-probs for windows ending at each frame."""
        n = len(pose.body.data)
        sign, phrase = [], []
        bb = self.model.backbone
        for start in range(0, n, 32):
            x = np.stack([window_at_end(pose, e) for e in range(start, min(start + 32, n))])
            z = self.model.project(torch.from_numpy(x).to(self.device))
            h = z
            for layer in bb.encoder_attn:
                h = layer(h, self.times.expand(len(h), -1))
            sign.append(bb.sign_bio_head(h).log_softmax(-1).cpu())
            phrase.append(bb.sentence_bio_head(h).log_softmax(-1).cpu())
        return torch.cat(sign).numpy().astype(np.float16), torch.cat(phrase).numpy().astype(np.float16)


def bio_at_lookahead(windows, lookahead):
    """Per-frame BIO log-probs using `lookahead` future frames (shrinks only at end of video)."""
    n = len(windows)
    out = np.empty((n, windows.shape[-1]), np.float32)
    for j in range(n):
        end = min(j + lookahead, n - 1)
        out[j] = windows[end, FRAMES - 1 - (end - j)]
    return out


# ----------------------------------------------------------------------------- Reel spans

class SpanScorer:
    """Memoised full proposal/verifier score vectors for [start, end] second spans."""

    def __init__(self):
        from scripts.evaluate_temporal_boundary_v17 import arguments
        from scripts.live_reel_stage1_v17 import build_components
        from active.v17.extract_v17 import AppleVisionDetector
        self.args = arguments('data', REPORT / 'sessions')
        self.args.no_motion_trim = True
        self.reel = build_components(self.args)['classifier']
        self.reel.args.no_motion_trim = self.reel.full.args.no_motion_trim = True
        self.labels = list(self.reel.labels)
        self.detector = AppleVisionDetector(self.args.minimum_point_confidence)
        self._encode_cache = {}
        full = self.reel.full
        original = full.encode_hands

        # Hand-crop embeddings are a pure function of the crop pixels; memoise by content.
        def encode_hands(crops, valid):
            embeddings = np.zeros((16, 3, 512), np.float32)
            for f in range(16):
                for v in range(3):
                    crop = crops[f][v]
                    if crop is None or not valid[f, v]:
                        continue
                    h = hashlib.blake2b(crop.tobytes(), digest_size=16).digest() + str(crop.shape).encode()
                    if h not in self._encode_cache:
                        single = [[None] * 3 for _ in range(16)]
                        single[0][0] = crop
                        mask = np.zeros_like(valid)
                        mask[0, 0] = 1
                        self._encode_cache[h] = original(single, mask)[0, 0].copy()
                    embeddings[f, v] = self._encode_cache[h]
            return embeddings
        full.encode_hands = encode_hands
        # Capture full raw logit vectors; the runtime only returns its top3.
        self.last = {}

        def tap(model, name, output):
            predict = model.predict

            def wrapped(inputs, *a, **k):
                out = predict(inputs, *a, **k)
                self.last[name] = np.asarray(out[output]).reshape(-1).astype(np.float32)
                if name == 'verifier':
                    self.last['inputs'] = dict(
                        landmarks=np.asarray(inputs['landmarks'][0], np.float16),
                        hand_embeddings=np.asarray(inputs['hand_embeddings'][0], np.float16),
                        hand_valid=np.asarray(inputs['hand_valid'][0], np.bool_),
                        hand_boxes=np.asarray(inputs['hand_boxes'][0], np.float16))
                return out
            model.predict = wrapped
        tap(self.reel.orientation, 'proposal', self.reel.orientation_output_name)
        tap(full.stage1, 'verifier', 'var_2927')

    def observations(self, row):
        from scripts.evaluate_temporal_boundary_v17 import observations
        from active.v17.stage1_window_v17 import raw_observation_features
        self._encode_cache = {}
        obs = observations(row, self.args, self.detector)
        target = CACHE / 'av_raw' / (key(row['source_item_id']) + '.npz')
        if obs and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            raw, times = raw_observation_features(obs)
            np.savez_compressed(target, raw=raw.astype(np.float16), times=times)
        return obs

    def score(self, obs, start, end):
        """Raw logits for proposal (100 + optional no-emit) and verifier (100)."""
        selected = [o for o in obs if start - CONTEXT <= o.seconds <= end + CONTEXT]
        if len(selected) < 4:
            return None
        self.last = {}
        p = self.reel.classify(selected)
        p_raw = self.last.get('proposal')
        self.last = {}
        v = self.reel.verify(selected)
        v_raw = self.last.get('verifier')
        inputs = self.last.get('inputs')
        return dict(p_top=p.get('top3'), p_score=p['model_score'], p_margin=p.get('margin', 0.),
                    p_gloss=p.get('candidate_gloss'), p_accepted=bool(p.get('accepted')),
                    p_noemit=p.get('diagnostics', {}).get('no_emit_probability', 0.),
                    p_raw=p_raw, v_raw=v_raw, inputs=inputs,
                    v_score=v['model_score'], v_margin=v.get('margin', 0.), v_gloss=v.get('candidate_gloss'),
                    v_accepted=bool(v.get('accepted')), p_reasons=p.get('rejection_reasons', []),
                    v_reasons=v.get('rejection_reasons', []))


def memo_path(row):
    return CACHE / 'spans' / (key(row['source_item_id']) + '.pkl')


def load_memo(row):
    path = memo_path(row)
    if path.exists():
        return pickle.loads(path.read_bytes())
    return {}


def save_memo(row, memo):
    path = memo_path(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_bytes(pickle.dumps(memo))
    tmp.replace(path)


# ----------------------------------------------------------------------------- prepare

def prepare(name, device='mps'):
    rows = rows_for(name)
    dgs = DGS(device)
    out = CACHE / 'dgs'
    out.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    for i, row in enumerate(rows):
        target = out / (key(row['source_item_id']) + '.npz')
        if target.exists():
            continue
        record = pose_record(row)
        pose = load_pose(dict(pose_path=record['pose_path'], pose_fps=record['pose_fps']))
        sign, phrase = dgs.all_windows(pose)
        np.savez_compressed(target, sign=sign, phrase=phrase, frames=len(sign))
        print('dgs %d/%d %s frames=%d (%.0fs)' % (i + 1, len(rows), row['source_item_id'], len(sign),
                                                  time.perf_counter() - tick), flush=True)


def load_dgs(row):
    data = np.load(CACHE / 'dgs' / (key(row['source_item_id']) + '.npz'))
    return data['sign'].astype(np.float32), data['phrase'].astype(np.float32)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare'])
    ap.add_argument('--set', default='test')
    a = ap.parse_args()
    if a.command == 'prepare':
        for s in a.set.split(','):
            prepare(s)


# ----------------------------------------------------------------------------- spans + scoring

def ensure_spans(scorer, row, spans, memo=None):
    """Score any (start_frame, end_frame) spans at 20 Hz not already memoised."""
    memo = load_memo(row) if memo is None else memo
    missing = [s for s in spans if s not in memo or (memo[s] is not None and 'inputs' not in memo[s])]
    if missing:
        obs = scorer.observations(row)
        for s in missing:
            memo[s] = scorer.score(obs, s[0] / FPS, s[1] / FPS)
        save_memo(row, memo)
    return memo


def upstream_segments(bio, min_frames=3):
    sys.path.insert(0, str(UPSTREAM))
    from probe import upstream_helpers
    h = upstream_helpers()
    return [(s['start'], s['end']) for s in h['filter_segments'](h['likeliest_probs_to_segments'](torch.from_numpy(bio)), min_frames)]


def reel_commit(r, commit_score, instant):
    """classify_interval's decision with commit_hits=1, from memoised scores."""
    if r is None or not (r['p_accepted'] and r['v_accepted']):
        return None
    agreement = r['p_score'] if r['p_gloss'] == r['v_gloss'] else 0.
    if max(r['v_score'], agreement) < commit_score:
        return None
    return r['v_gloss']


# ----------------------------------------------------------------------------- segmental decoder

from active.v17.segmental_runtime_v17 import (UNK, O, B, I, log_softmax, candidate_spans, fused,
                                               segmental_decode, dense_spans, both_spans)










# ----------------------------------------------------------------------------- forced alignment

def forced_align(memo, spans, labels, target, skip_penalty=6., key='v_raw'):
    """Ordered, non-overlapping spans for the known transcript maximising sum log q(gloss).

    Glosses may be skipped at a fixed penalty when no candidate span supports them.
    """
    index = {g: i for i, g in enumerate(labels)}
    scored = []
    for s, e in spans:
        r = memo.get((s, e))
        if r is None or r.get(key) is None:
            continue
        scored.append((s, e, log_softmax(r[key])))
    if not scored or not target:
        return []
    n = max(e for _, e, _ in scored) + 2
    K = len(target)
    import functools

    @functools.lru_cache(maxsize=None)
    def best(k, t):
        if k == K:
            return 0., ()
        value, path = best(k + 1, t)
        value -= skip_penalty
        if target[k] in index:
            gi = index[target[k]]
            for s, e, lq in scored:
                if s < t:
                    continue
                v, p = best(k + 1, e + 1)
                v += float(lq[gi])
                if v > value:
                    value, path = v, ((k, s, e),) + p
        return value, path
    _, path = best(0, 0)
    return [dict(gloss=target[k], index=k, start=s, end=e) for k, s, e in path]


# ----------------------------------------------------------------------------- PyTorch rescoring

class TorchVerifier:
    """Any unified v17 checkpoint applied to the captured live verifier inputs."""

    def __init__(self, path, device='mps'):
        from active.v17.export_unified_multimodal_coreml_v17 import load_model
        self.model, checkpoint = load_model(Path(path))
        self.model.to(device).eval()
        self.device = device
        self.labels = [l for l, _ in sorted(checkpoint['label_to_index'].items(), key=lambda r: r[1])]
        self.tag = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]

    @torch.inference_mode()
    def logits(self, inputs_list, batch=256):
        out = []
        for i in range(0, len(inputs_list), batch):
            chunk = inputs_list[i:i + batch]
            lm = torch.from_numpy(np.stack([c['landmarks'] for c in chunk]).astype(np.float32)).to(self.device)
            he = torch.from_numpy(np.stack([c['hand_embeddings'] for c in chunk]).astype(np.float32)).to(self.device)
            hv = torch.from_numpy(np.stack([c['hand_valid'] for c in chunk])).to(self.device)
            hb = torch.from_numpy(np.stack([c['hand_boxes'] for c in chunk]).astype(np.float32)).to(self.device)
            out.append(self.model(lm, he, hv, hb).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 100), np.float32)


def rescore_memo(memo, verifier):
    """Return a copy of memo whose v_raw comes from `verifier` (proposal kept)."""
    keys = [k for k, r in memo.items() if r is not None and r.get('inputs') is not None]
    logits = verifier.logits([memo[k]['inputs'] for k in keys])
    out = {}
    for k, r in memo.items():
        out[k] = None if r is None else dict(r)
    for k, l in zip(keys, logits):
        out[k]['v_raw'] = l
    for k, r in out.items():
        if r is not None and r.get('inputs') is None:
            out[k]['v_raw'] = None
    return out


def evaluate_decoder(data, labels, lookahead, cfg, span_kw=None, ignore_other=False):
    """data: list of (row, sign_windows, memo). Returns summary counts + per-video hyps."""
    from scripts.evaluate_temporal_boundary_v17 import edit_counts
    S = D = Ins = C = R = H = 0
    per = []
    for row, sign, memo in data:
        bio = bio_at_lookahead(sign, lookahead)
        kw = dict(span_kw or {})
        mode = kw.pop('mode', cfg.get('span_mode', 'cand'))
        spans = candidate_spans(bio, **kw)
        if mode == 'both':
            spans = sorted(set(spans) | set(dense_spans(bio)))
        elif mode == 'dense':
            spans = dense_spans(bio)
        segs = segmental_decode(bio, memo, spans, labels, cfg)
        if ignore_other and row.get('all_events'):
            others = [(e['start'], e['end']) for e in row['all_events'] if e['kind'] != 'known']
            segs = [g for g in segs if not any(a <= (g['start'] + g['end']) / 2 / FPS <= b for a, b in others)]
        hyp = [g['gloss'] for g in segs if g['gloss']]
        m = edit_counts(row['target_sequence'], hyp)
        S += m['substitutions']; D += m['deletions']; Ins += m['insertions']; C += m['correct']
        R += m['references']; H += len(hyp)
        per.append(dict(id=row['source_item_id'], reference=row['target_sequence'], hypothesis=hyp, segments=segs))
    return dict(wer=(S + D + Ins) / max(R, 1), correct=C, S=S, D=D, I=Ins, refs=R, hyps=H,
                precision=C / max(H, 1), recall=C / max(R, 1)), per


# ----------------------------------------------------------------------------- training spans

def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union else 0.


def jitter(span, n, deltas=((-1, 0), (1, 0), (0, -1), (0, 1), (-2, 1), (1, 2), (-1, -2), (2, -1))):
    out = set()
    for ds, de in deltas:
        s, e = span[0] + ds, span[1] + de
        if 0 <= s and e < n and e - s + 1 >= 3:
            out.add((s, e))
    return out


def build_training(name, labels, scorer):
    """Score decoder-matched spans on training videos and label them from timing/forced alignment."""
    rows = rows_for(name)
    out_dir = CACHE / 'train_labels'
    out_dir.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    for i, row in enumerate(rows):
        target_path = out_dir / (key(row['source_item_id']) + '.json')
        if target_path.exists():
            continue
        sign, _ = load_dgs(row)
        n = len(sign)
        spans = set()
        for L in (8, 10):
            spans |= set(candidate_spans(bio_at_lookahead(sign, L), b_threshold=.1))
        timed = []
        for e in row.get('all_events', []):
            s, t = int(round(e['start'] * FPS)), int(round(e['end'] * FPS))
            t = min(t, n - 1)
            if t - s + 1 >= 2:
                timed.append(dict(gloss=e['label'] if e['kind'] == 'known' else 'OTHER', start=s, end=max(t, s + 2)))
        for g in timed:
            if g['gloss'] != 'OTHER':
                spans.add((g['start'], g['end']))
                spans |= jitter((g['start'], g['end']), n)
        memo = load_memo(row)
        obs = None

        def fill(want):
            nonlocal obs
            missing = [s for s in want if s not in memo or (memo[s] is not None and 'inputs' not in memo[s])]
            if missing:
                if obs is None:
                    obs = scorer.observations(row)
                for s in missing:
                    memo[s] = scorer.score(obs, s[0] / FPS, s[1] / FPS)
        fill(sorted(spans))
        if not timed:
            aligned = forced_align(memo, sorted(spans), labels, row['target_sequence'])
            timed = [dict(gloss=a['gloss'], start=a['start'], end=a['end']) for a in aligned]
            extra = set()
            for g in timed:
                extra |= jitter((g['start'], g['end']), n)
            fill(sorted(extra))
            spans |= extra
        save_memo(row, memo)
        labelled = []
        for s in sorted(spans):
            r = memo.get(s)
            if r is None or r.get('inputs') is None:
                continue
            best = max(timed, key=lambda g: iou(s, (g['start'], g['end'])), default=None)
            if best is None:
                continue
            overlap = iou(s, (best['start'], best['end']))
            if overlap >= .5:
                labelled.append(dict(span=list(s), gloss=best['gloss'], iou=overlap))
        target_path.write_text(json.dumps(dict(id=row['source_item_id'], video_sha256=row['video_sha256'],
                                               source=name, timing=timed, spans=labelled,
                                               forced=not row.get('all_events'))))
        if i % 25 == 0:
            print('train %s %d/%d spans=%d labelled=%d (%.0fs)' % (name, i, len(rows), len(spans), len(labelled),
                                                                  time.perf_counter() - tick), flush=True)




# ----------------------------------------------------------------------------- streaming simulation

def stream_decode(bio, memo, span_fn, labels, cfg, lookahead, lag=1, context_frames=2, stable=1,
                  soft_extra=0, hard=.5, early_k=0, early_q=.9, early_min=4):
    """Fixed-lag online version of segmental_decode.

    At input frame T the BIO is known through T-lookahead and observations through T.
    The DP is re-run over frames after the last committed segment; any emitted segment of the
    best path ending at or before (T - lookahead - lag) is committed and never revised.
    Returns committed segments with `commit_frame` (the input frame at which it is shown).
    """
    n = len(bio)
    committed, origin = [], 0
    last = None
    seen = {}
    early = {}          # open-segment start -> (gloss, consecutive steps)
    shown_early = []    # (start, gloss, commit_frame)
    for T in range(n + lookahead + lag + context_frames + 1):
        known = min(T - lookahead, n - 1)
        final = T >= n - 1 + lookahead
        if known < origin:
            continue
        sub = bio[origin:known + 1]
        # Candidates come from the whole known prefix (identical to prefix_spans), and only
        # those starting after the last committed segment are decodable.
        spans = [(s - origin, e - origin) for s, e in span_fn(bio[:known + 1], 0)
                 if s >= origin and (e + context_frames <= T or final)]
        local = {(s, e): memo.get((s + origin, e + origin)) for s, e in spans}
        segs = segmental_decode(sub, local, spans, labels, dict(cfg, merge_duplicates=False))
        horizon = known - lag if not final else known
        new_origin = origin
        current = {}
        for g in segs:
            sig = (g['start'] + origin, g['end'] + origin, g['gloss'])
            current[sig] = seen.get(sig, 0) + 1
        seen = current
        # Early identity: the still-open segment's label may be shown once it is stable and
        # confident, before its end boundary is known (adjacent signs only resolve once the next
        # sign is visible). Its later finalisation is then not shown again.
        if early_k and not final:
            opened = [g for g in segs if g['end'] + origin > horizon and g['gloss']]
            nxt = {}
            for g in opened:
                st = g['start'] + origin
                prev = early.get(st)
                count = prev[1] + 1 if prev and prev[0] == g['gloss'] else 1
                nxt[st] = (g['gloss'], count)
                if (count >= early_k and np.exp(g['log_q']) >= early_q and g['end'] - g['start'] + 1 >= early_min
                        and not any(abs(st - a) <= 2 for a, _, _ in shown_early)):
                    if last is not None and last['gloss'] == g['gloss'] and st - last['end'] <= cfg.get('duplicate_gap', 2):
                        shown_early.append((st, g['gloss'], T))
                        continue
                    last = dict(gloss=g['gloss'], start=st, end=g['end'] + origin, commit_end=None,
                                commit_frame=T, log_q=g['log_q'], eof=False, early=True)
                    committed.append(last)
                    shown_early.append((st, g['gloss'], T))
            early = nxt
        for g in segs:
            end = g['end'] + origin
            if end > horizon:
                break
            if not final and seen[(g['start'] + origin, end, g['gloss'])] < stable:
                break
            if not final and soft_extra and end + 1 <= known:
                p_next = np.exp(bio[end + 1])
                # A boundary with neither rest nor sign-start evidence right after it is only
                # the recognizer's guess inside a continuing movement: give it more frames.
                if p_next[O] + p_next[UNK] + p_next[B] < hard and end > horizon - soft_extra:
                    break
            hit = [e for e in shown_early if abs(e[0] - (g['start'] + origin)) <= 2]
            if hit:
                for c in committed:
                    if c.get('early') and abs(c['start'] - hit[0][0]) <= 2 and c['commit_end'] is None:
                        c['end'] = c['commit_end'] = end
                new_origin = end + 1
                continue
            if g['gloss']:
                gloss = g['gloss']
                start = g['start'] + origin
                if (last is not None and last['gloss'] == gloss
                        and start - last['end'] <= cfg.get('duplicate_gap', 2)):
                    last['end'] = end
                else:
                    last = dict(gloss=gloss, start=start, end=end, commit_end=end,
                                commit_frame=min(T, n - 1 + lookahead + lag), log_q=g['log_q'], eof=bool(final))
                    committed.append(last)
            new_origin = end + 1
        origin = new_origin
        if final:
            break
    return committed


# ----------------------------------------------------------------------------- end-to-end streaming evaluation



def student_bio(model, row, n_grid, device='cpu'):
    """Student log-probs mapped onto the DGS 20 Hz grid by nearest observation time."""
    from active.v17.av_boundary_v17 import clip_bio
    data = np.load(CACHE / 'av_raw' / (key(row['source_item_id']) + '.npz'))
    raw, times = data['raw'].astype(np.float32), data['times'].astype(np.float64)
    bio = clip_bio(model, raw, times, device)
    grid = np.arange(n_grid) / FPS
    idx = np.clip(np.searchsorted(times, grid - 1e-6), 0, len(times) - 1)
    prev = np.clip(idx - 1, 0, len(times) - 1)
    idx = np.where(np.abs(times[prev] - grid) < np.abs(times[idx] - grid), prev, idx)
    return bio[idx]


def prefix_spans(bio, lookahead, lag=1, context_frames=2):
    """Union of spans any streaming step can request (for scoring before decoding)."""
    n = len(bio)
    out = set()
    for known in range(n):
        out |= set(both_spans(bio[:known + 1]))
    return sorted(out)


def evaluate_stream(rows, labels, cfg, lookahead, lag, bio_fn, scorer=None, verifier=None, offline=False, stable=1,
                    soft_extra=0, early_k=0, early_q=.9):
    """bio_fn(row, sign_windows) -> per-frame log-probs already at `lookahead`."""
    from scripts.evaluate_temporal_boundary_v17 import edit_counts
    tot = dict(S=0, D=0, I=0, C=0, R=0, H=0)
    latencies, per = [], []
    for row in rows:
        sign, _ = load_dgs(row)
        bio = bio_fn(row, sign)
        spans = prefix_spans(bio, lookahead) if not offline else both_spans(bio)
        memo = load_memo(row)
        need = [s for s in spans if s not in memo or (memo[s] is not None and 'inputs' not in memo[s])]
        if need:
            if scorer is None:
                raise RuntimeError('unscored spans and no scorer: %d' % len(need))
            memo = ensure_spans(scorer, row, spans, memo)
        if verifier is not None:
            memo = rescore_memo(memo, verifier)
        if offline:
            segs = [g for g in segmental_decode(bio, memo, both_spans(bio), labels, cfg) if g['gloss']]
        else:
            segs = stream_decode(bio, memo, lambda sub, origin: both_spans(sub), labels, cfg, lookahead, lag, stable=stable,
                                 soft_extra=soft_extra, early_k=early_k, early_q=early_q)
            latencies += [(g['commit_frame'] - g['commit_end']) / FPS for g in segs
                          if not g.get('eof') and g.get('commit_end') is not None]
            eof_count = sum(bool(g.get('eof')) for g in segs)
            tot['eof'] = tot.get('eof', 0) + eof_count
        hyp = [g['gloss'] for g in segs]
        m = edit_counts(row['target_sequence'], hyp)
        for k, v in (('S', 'substitutions'), ('D', 'deletions'), ('I', 'insertions'), ('C', 'correct'), ('R', 'references')):
            tot[k] += m[v]
        tot['H'] += len(hyp)
        per.append(dict(id=row['source_item_id'], subset=row.get('subset'), frames=len(bio),
                        latencies=[(g['commit_frame'] - g['commit_end']) / FPS for g in segs
                                   if not g.get('eof') and g.get('commit_end') is not None] if not offline else [],
                        reference=row['target_sequence'],
                        hypothesis=hyp, segments=segs, metrics={k: m[k] for k in ('substitutions', 'deletions', 'insertions', 'correct', 'references')}))
    s = dict(wer=(tot['S'] + tot['D'] + tot['I']) / max(tot['R'], 1), precision=tot['C'] / max(tot['H'], 1),
             recall=tot['C'] / max(tot['R'], 1), **tot)
    if latencies:
        lat = np.asarray(latencies)
        s.update(latency_median=float(np.median(lat)), latency_p90=float(np.percentile(lat, 90)),
                 latency_max=float(lat.max()), latency_under_05=float((lat < .5).mean()), live_commits=int(len(lat)))
    return s, per


def add_prefix_spans(name, scorer, fractions=(.4, .6, .8)):
    """Label early partial views of each timed sign with that sign (streaming shows words early)."""
    rows = rows_for(name)
    tick = time.perf_counter()
    for i, row in enumerate(rows):
        path = CACHE / 'train_labels' / (key(row['source_item_id']) + '.json')
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if payload.get('prefix_added'):
            continue
        want = []
        for g in payload['timing']:
            if g['gloss'] == 'OTHER':
                continue
            length = g['end'] - g['start'] + 1
            for f in fractions:
                e = g['start'] + int(np.ceil(f * length)) - 1
                if e - g['start'] + 1 >= 3 and e < g['end']:
                    want.append(((g['start'], e), g['gloss']))
        memo = load_memo(row)
        missing = [s for s, _ in want if s not in memo or (memo[s] is not None and 'inputs' not in memo[s])]
        if missing:
            obs = scorer.observations(row)
            for s in missing:
                memo[s] = scorer.score(obs, s[0] / FPS, s[1] / FPS)
            save_memo(row, memo)
        have = {tuple(x['span']) for x in payload['spans']}
        for s, gloss in want:
            if s not in have and memo.get(s) is not None and memo[s].get('inputs') is not None:
                payload['spans'].append(dict(span=list(s), gloss=gloss, iou=None, prefix=True))
        payload['prefix_added'] = True
        path.write_text(json.dumps(payload))
        if i % 50 == 0:
            print('prefix %s %d/%d (%.0fs)' % (name, i, len(rows), time.perf_counter() - tick), flush=True)
