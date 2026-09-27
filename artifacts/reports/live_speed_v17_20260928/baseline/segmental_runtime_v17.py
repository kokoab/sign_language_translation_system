"""Streaming transition-aware segmental recognition for the live Reel (Apple Vision only).

Per frame: Apple Vision observation -> per-frame hand crops/embeddings (cached once) -> Apple Vision
boundary student (bounded lookahead) -> candidate sign spans -> span recognizer -> semi-Markov
sign/rest DP -> fixed-lag commit with early identity for the still-open segment.

Decoder functions are shared with scripts/segmental_lab_v17.py, which produced the offline evidence
in artifacts/reports/segmental_decoder_v17_20260927/. Nothing here changes a live default.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FPS = 20
UNK, O, B, I = 0, 1, 2, 3
# Current live configuration (v1 frozen held-out; v2 early-display guard; v3 letters + Core ML).
CONFIG = ROOT / 'artifacts/reports/segmental_decoder_v17_20260927/stream_config_v3_final.json'


def log_softmax(x):
    x = np.asarray(x, np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def candidate_spans(bio, b_threshold=.2, min_len=3, max_len=30, max_inner_o=3):
    """Start/end candidates from BIO runs and B-probability peaks; spans between them."""
    p = np.exp(bio)
    arg = p.argmax(1)
    inside = (arg == B) | (arg == I)
    n = len(p)
    starts, ends = set(), set()
    t = 0
    while t < n:
        if inside[t]:
            u = t
            while u + 1 < n and inside[u + 1]:
                u += 1
            starts.add(t); ends.add(u)
            t = u + 1
        else:
            t += 1
    pb = p[:, B]
    for t in range(n):
        left = pb[t - 1] if t else 0.
        right = pb[t + 1] if t + 1 < n else 0.
        if pb[t] >= b_threshold and pb[t] >= left and pb[t] >= right:
            starts.add(t)
            if t:
                ends.add(t - 1)
    o_run = np.cumsum(arg == O)
    spans = []
    for s in sorted(starts):
        for e in sorted(ends):
            length = e - s + 1
            if length < min_len or length > max_len:
                continue
            if o_run[e] - (o_run[s - 1] if s else 0) > max_inner_o:
                continue
            spans.append((s, e))
    return spans


def fused(record, alpha):
    if record is None or record.get('v_raw') is None:
        return None
    if alpha >= 1:
        return log_softmax(record['v_raw'])
    if record.get('p_raw') is None:
        return None
    return alpha * log_softmax(record['v_raw']) + (1 - alpha) * log_softmax(record['p_raw'][:100])


def segmental_decode(bio, memo, spans, labels, cfg):
    """Semi-Markov DP: each frame is rest/transition or inside one labelled segment.

    A sign segment scores  w_r*max(log q(g), log theta) + w_a*sum log P(B|I) + w_b*log P(B at start) - c.
    Rest frames score w_a*log P(O).  Segments whose best label is below theta are covered but
    not emitted (unknown/OOV signing), which keeps them from being forced into a vocabulary word.
    """
    p = np.exp(bio)
    act = np.log(np.clip(p[:, B] + p[:, I], 1e-6, 1))
    rest = np.log(np.clip(p[:, O] + p[:, UNK], 1e-6, 1))
    logb = np.log(np.clip(p[:, B], 1e-6, 1))
    cact = np.concatenate([[0], np.cumsum(act)])
    n = len(p)
    by_end = {}
    for s, e in spans:
        q = fused(memo.get((s, e)), cfg['alpha'])
        if q is None:
            continue
        g = int(q.argmax())
        lq = float(q[g])
        emit = lq >= cfg['log_theta']
        # Letters may use their own per-segment cost: the word bonus for extra segments splits
        # a held or moving letter into pieces that each look like another static letter.
        cost = cfg.get('c_letter', cfg['c']) if emit and labels[g].startswith('FS_') else cfg['c']
        seg = (cfg['w_r'] * max(lq, cfg['log_theta']) + cfg['w_a'] * (cact[e + 1] - cact[s])
               + cfg['w_b'] * logb[s] - cost)
        by_end.setdefault(e, []).append((s, seg, labels[g] if emit else None, lq))
    best = np.full(n + 1, -np.inf)
    back = [None] * (n + 1)
    best[0] = 0.
    for t in range(1, n + 1):
        best[t] = best[t - 1] + cfg['w_a'] * rest[t - 1]
        back[t] = ('rest', t - 1)
        for s, seg, gloss, lq in by_end.get(t - 1, []):
            v = best[s] + seg
            if v > best[t]:
                best[t] = v
                back[t] = ('sign', s, gloss, lq)
    out, t = [], n
    while t > 0:
        b = back[t]
        if b[0] == 'rest':
            t -= 1
        else:
            out.append(dict(start=b[1], end=t - 1, gloss=b[2], log_q=b[3]))
            t = b[1]
    out.reverse()
    if cfg.get('merge_duplicates', True):
        merged = []
        for seg in out:
            if (merged and seg['gloss'] and merged[-1]['gloss'] == seg['gloss']
                    and seg['start'] - merged[-1]['end'] <= cfg.get('duplicate_gap', 2)):
                merged[-1]['end'] = seg['end']
                continue
            merged.append(seg)
        out = merged
    return out


def dense_spans(bio, step=2, min_len=3, max_len=20, active=.3, max_inner_o=3):
    """Every grid span inside BIO-active regions; lets the recognizer decide split points."""
    p = np.exp(bio)
    act = p[:, B] + p[:, I]
    n = len(p)
    on = act >= active
    o_run = np.cumsum(~on)
    points = [t for t in range(n) if on[t]]
    if not points:
        return []
    grid = sorted(set(points[::step]) | {t for t in points if t == 0 or not on[t - 1]} |
                  {t for t in points if t == n - 1 or not on[t + 1]})
    spans = []
    for s in grid:
        for e in grid:
            length = e - s + 1
            if length < min_len or length > max_len:
                continue
            if o_run[e] - (o_run[s - 1] if s else 0) > max_inner_o:
                continue
            spans.append((s, e))
    return spans


def both_spans(bio):
    return sorted(set(candidate_spans(bio)) | set(dense_spans(bio)))


# ----------------------------------------------------------------------------- live runtime

class SpanRecognizer:
    """Unified v17 checkpoint over live verifier inputs, with per-frame hand crop embeddings."""

    def __init__(self, checkpoint, image_encoder, device='mps', coreml=None, compute_units='ALL'):
        import coremltools as ct
        import torch
        from active.v17.export_unified_multimodal_coreml_v17 import load_model
        from active.v17.schema_hand_rgb_v17 import HandRGBV17Config
        self.torch = torch
        self.model, payload = load_model(Path(checkpoint))
        self.torch_device = device if device != 'mps' or torch.backends.mps.is_available() else 'cpu'
        self.device = self.torch_device
        self.model.to(self.torch_device).eval()
        self.labels = [l for l, _ in sorted(payload['label_to_index'].items(), key=lambda r: r[1])]
        # Optional Core ML graph of the same checkpoint (the phone path); PyTorch stays as reference.
        self.coreml = None
        if coreml is not None:
            self.coreml = ct.models.MLModel(str(coreml), compute_units=getattr(ct.ComputeUnit, compute_units))
            outputs = [o.name for o in self.coreml.get_spec().description.output]
            self.coreml_output = outputs[0]
            # A two-output graph carries the letter head: [word logits, letter logits].
            self.coreml_letter_output = outputs[1] if len(outputs) > 1 else None
            shape = self.coreml.get_spec().description.input[0].type.multiArrayType
            enumerated = [tuple(x.shape)[0] for x in shape.enumeratedShapes.shapes]
            # One fixed shape: switching enumerated shapes re-plans the model (~280 ms per call).
            self.coreml_batches = (8,) if 8 in enumerated else (1,)
            self.device = 'coreml'
        self.encoder = ct.models.MLModel(str(image_encoder), compute_units=ct.ComputeUnit.ALL)
        self.config = HandRGBV17Config()
        self.checkpoint = str(checkpoint)
        from PIL import Image
        self.encoder.predict({'image': Image.fromarray(np.zeros((256, 256, 3), np.uint8))})
        with torch.inference_mode():
            self.model(torch.zeros((1, 32, 61, 5), device=self.torch_device), torch.zeros((1, 16, 3, 512), device=self.torch_device),
                       torch.zeros((1, 16, 3), dtype=torch.bool, device=self.torch_device), torch.zeros((1, 16, 3, 4), device=self.torch_device))

    def _encode(self, crop):
        import cv2
        from PIL import Image
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return np.asarray(self.encoder.predict({'image': image})['embedding'], np.float32).reshape(512)

    def frame_hand(self, observation):
        """Embeddings/valid/boxes for one frame; identical to hand_inputs for that frame."""
        from scripts.live_isolated_v17 import hand_box, crop_square, union_box
        frame = observation.frame
        height, width = frame.shape[:2]
        emb = np.zeros((3, 512), np.float32)
        valid = np.zeros(3, np.float32)
        boxes = np.zeros((3, 4), np.float32)
        seen = []
        for view, side in enumerate(('left', 'right')):
            hand = observation.assigned[side]
            box = None if hand is None else hand_box(hand, width, height, self.config)
            if box is None:
                continue
            boxes[view] = box / (width, height, width, height)
            valid[view] = 1
            seen.append(box)
            emb[view] = self._encode(crop_square(frame, box, self.config.crop_size))
        combined = union_box(seen, width, height, self.config.union_box_scale)
        if combined is not None:
            boxes[2] = combined / (width, height, width, height)
            valid[2] = 1
            emb[2] = self._encode(crop_square(frame, combined, self.config.crop_size))
        return emb, valid, boxes

    def inputs(self, observations, hands):
        """Verifier inputs for one span (no motion trim), or None when landmarks are unusable."""
        from scripts.live_isolated_v17 import landmarks_from_observations
        from active.v17.schema_v17 import V17Config
        try:
            features, diagnostics = landmarks_from_observations(observations, V17Config())
        except ValueError:
            return None
        start, end = int(diagnostics['trim_start']), int(diagnostics['trim_end_exclusive'])
        positions = np.rint(np.linspace(0, end - start - 1, self.config.sequence_length)).astype(int) + start
        return (features.astype(np.float32), np.stack([hands[i][0] for i in positions]),
                np.stack([hands[i][1] for i in positions]) > .5, np.stack([hands[i][2] for i in positions]))

    def attach_letters(self, path):
        """Supplemental static-letter head (FS_A..FS_Z + NONE) on the same span features."""
        import torch
        from active.v17.letter_head_v17 import CLASSES, FORMAT, LetterHead
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('format') != FORMAT:
            raise ValueError('not a letter head checkpoint')
        self.letter_head = LetterHead(**payload['config'])
        self.letter_head.load_state_dict(payload['state_dict'])
        self.letter_head.to(self.torch_device).eval()
        self.letter_classes = CLASSES
        self.letter_threshold = float(payload['threshold'])
        self.letter_path = str(path)

    def logits(self, batch):
        """Word logits [N,100], or [N,100+27] word+letter logits when a letter head is attached."""
        if getattr(self, 'letter_head', None) is not None and self.coreml is None:
            from active.v17.letter_head_v17 import unified_features
            torch = self.torch
            out = []
            with torch.inference_mode():
                for i in range(0, len(batch), 64):
                    chunk = batch[i:i + 64]
                    x = [torch.from_numpy(np.stack([c[k] for c in chunk])).to(self.torch_device) for k in range(4)]
                    lf, hf, words = unified_features(self.model, *x)
                    out.append(torch.cat([words, self.letter_head(lf, hf, words)], -1).float().cpu().numpy())
            return np.concatenate(out) if out else np.zeros((0, len(self.labels) + 27), np.float32)
        if self.coreml is not None:
            if not batch:
                return np.zeros((0, len(self.labels)), np.float32)
            rows = []
            sizes = getattr(self, 'coreml_batches', (1,))
            i = 0
            while i < len(batch):
                left = len(batch) - i
                size = min([b for b in sizes if b >= left] or [max(sizes)])
                chunk = batch[i:i + size]
                pad = size - len(chunk)
                chunk = list(chunk) + [chunk[-1]] * pad      # enumerated shapes need exact sizes
                feed = {name: np.stack([c[k] for c in chunk]).astype(np.float32)
                        for k, name in enumerate(('landmarks', 'hand_embeddings', 'hand_valid', 'hand_boxes'))}
                result = self.coreml.predict(feed)
                out = np.asarray(result[self.coreml_output], np.float32).reshape(size, -1)
                if getattr(self, 'coreml_letter_output', None):
                    letters = np.asarray(result[self.coreml_letter_output], np.float32).reshape(size, -1)
                    out = np.concatenate([out, letters], -1)
                rows.append(out[:size - pad])
                i += size - pad
            return np.concatenate(rows)
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(batch), 64):
                chunk = batch[i:i + 64]
                out.append(self.model(*(torch.from_numpy(np.stack([c[k] for c in chunk])).to(self.torch_device)
                                        for k in range(4))).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)), np.float32)


class SegmentalRuntime:
    """Frame-in, words-out. Indices are absolute observation numbers since the last reset."""

    def __init__(self, boundary, recognizer, config=None, mode='auto'):
        from active.v17.av_boundary_v17 import AVBoundaryStream
        self.config = json.loads(Path(config or CONFIG).read_text()) if not isinstance(config, dict) else config
        dec = dict(self.config['decoder'])
        self.lookahead = int(dec.pop('lookahead'))
        if self.lookahead != boundary.lookahead:
            raise ValueError('decoder lookahead must equal the boundary model lookahead')
        dec['log_theta'] = float(np.log(dec.pop('theta')))
        dec.pop('span_mode', None)
        self.cfg = dict(dec, alpha=1.0)
        stream = self.config['stream']
        self.lag = int(stream['lag'])
        self.context = int(stream.get('context_frames', 2))
        self.early_k, self.early_q = int(stream['early_k']), float(stream['early_q'])
        self.soft_extra, self.hard, self.early_min = int(stream['soft_extra']), .5, 4
        # Glosses whose appearance also begins other signs (e.g. I begins WE) are never shown early.
        self.early_unsafe = set(stream.get('early_unsafe', ()))
        self.boundary = AVBoundaryStream(boundary)
        self.recognizer = recognizer
        has_head = getattr(recognizer, 'letter_head', None) is not None
        # 'words': 100 signs only; 'letters': FS_A..FS_Z only; 'auto': both when a head is attached.
        self.mode = mode if mode != 'auto' else ('both' if has_head else 'words')
        self.letters = self.mode in ('both', 'letters')
        self.labels = list(recognizer.labels) + ([c for c in recognizer.letter_classes[:-1]] if self.letters else [])
        self.letter_gap = int(stream.get('letter_duplicate_gap', 2))
        self.timing = None      # set to {} to collect per-stage wall-clock seconds
        self.raw_cache = None   # optional dict shared with a second decoder: span -> raw logits
        self.reset()

    def _tick(self, name, started):
        import time
        if self.timing is not None:
            self.timing.setdefault(name, []).append(time.perf_counter() - started)
        return time.perf_counter()

    def reset(self):
        self.boundary.reset()
        self.off = 0            # absolute index of self.obs[0]
        self.count = 0          # observations seen
        self.obs, self.hands, self.bio = [], [], []
        self.bio_off = 0        # absolute index of self.bio[0]
        self.scores = {}
        self.origin = 0
        self.last = None
        self.early, self.shown_early = {}, []
        self.last_seconds = -np.inf
        self.preview = None     # best guess for the still-open segment (display only)

    def _span_obs(self, s, e):
        lo = self.obs[s - self.off].seconds - .1
        hi = self.obs[e - self.off].seconds + .1
        ids = [i for i in range(max(0, s - self.off - 3), min(len(self.obs), e - self.off + 4))
               if lo - 1e-9 <= self.obs[i].seconds <= hi + 1e-9]
        return ids

    def _score(self, spans):
        import time
        t = time.perf_counter()
        if self.timing is not None:
            self.timing.setdefault('spans_per_call', []).append(len(spans))
        batch, keys = [], []
        shared = self.raw_cache if self.raw_cache is not None else {}
        cached = [(k, shared[k]) for k in spans if k in shared]
        spans = [k for k in spans if k not in shared]
        for s, e in spans:
            ids = self._span_obs(s, e)
            if len(ids) < 4:
                self.scores[(s, e)] = None
                shared[(s, e)] = None
                continue
            x = self.recognizer.inputs([self.obs[i] for i in ids], [self.hands[i] for i in ids])
            if x is None:
                self.scores[(s, e)] = None
                shared[(s, e)] = None
                continue
            batch.append(x); keys.append((s, e))
        t = self._tick('span_inputs', t)
        logits = self.recognizer.logits(batch) if batch else []
        self._tick('recognizer', t)
        for k, l in zip(keys, logits):
            shared[k] = l
        for k, l in cached:
            if l is None:
                self.scores[k] = None
            else:
                keys.append(k)
        logits = list(logits) + [l for _, l in cached if l is not None]
        words = len(self.recognizer.labels)
        for k, l in zip(keys, logits):
            if not self.letters:
                self.scores[k] = dict(v_raw=l[:words])
                continue
            if self.mode == 'letters':
                p = np.exp(log_softmax(l[words:]))
                keep = p[:-1].max() >= self.recognizer.letter_threshold
                letter_lp = np.log(np.clip(p[:-1], 1e-9, 1)) if keep else np.full(len(p) - 1, -1e9)
                self.scores[k] = dict(v_raw=np.concatenate([np.full(words, -1e9), letter_lp]).astype(np.float32))
                continue
            # One distribution over words and letters: P(word) = P(not a letter) * q(word).
            word_lp = log_softmax(l[:words])
            p = np.exp(log_softmax(l[words:]))
            if p[:-1].max() >= self.recognizer.letter_threshold:
                combined = np.concatenate([word_lp + np.log(max(p[-1], 1e-9)), np.log(np.clip(p[:-1], 1e-9, 1))])
            else:
                combined = np.concatenate([word_lp, np.full(len(p) - 1, -1e9)])
            self.scores[k] = dict(v_raw=combined.astype(np.float32))

    def observe(self, observation, hands=None):
        """Add one Apple Vision observation; return words committed at this frame.

        `hands` lets a second decoder reuse this frame's already encoded hand crops."""
        from active.v17.stage1_window_v17 import raw_observation_features
        words = []
        if observation.seconds - self.last_seconds > .26 and self.count:
            words = self.finish()
            self.reset()
        import time
        self.last_seconds = observation.seconds
        self.obs.append(observation)
        t = time.perf_counter()
        self.hands.append(hands if hands is not None else self.recognizer.frame_hand(observation))
        t = self._tick('hand_crops_encode', t)
        self.count += 1
        raw, _ = raw_observation_features([observation])
        out = self.boundary.update(raw[0], observation.seconds)
        if out is not None and out[0] is not None:
            self.bio.append(np.asarray(out[1], np.float32))
        t = self._tick('boundary', t)
        return words + self._step(final=False)

    def finish(self):
        """Flush at stop/finish: the remaining best path is committed without more lookahead."""
        if not self.count:
            return []
        # Remaining frames are estimated with their unseen future masked, as at a clip end in training.
        pending = self.boundary.flush()
        missing = self.count - (self.bio_off + len(self.bio))
        for row in pending[len(pending) - missing:] if missing > 0 else []:
            self.bio.append(np.asarray(row, np.float32))
        return self._step(final=True)

    def _trim(self):
        # Keep ~2 s behind the decode window so committed letters can be re-scored as one span.
        keep = max(self.origin - 40, 0)
        drop = keep - self.off
        if drop > 0:
            del self.obs[:drop]; del self.hands[:drop]
            self.off = keep
        drop = keep - self.bio_off
        if drop > 0:
            del self.bio[:drop]
            self.bio_off = keep
        self.scores = {k: v for k, v in self.scores.items() if k[0] >= self.origin}

    def _step(self, final):
        T = self.count - 1
        known = self.bio_off + len(self.bio) - 1
        if known < self.origin or not self.bio:
            return []
        origin = self.origin
        sub = np.asarray(self.bio[origin - self.bio_off:known + 1 - self.bio_off])
        spans = [(s + origin, e + origin) for s, e in both_spans(sub)
                 if final or e + origin + self.context <= T]
        missing = [sp for sp in spans if sp not in self.scores]
        if missing:
            self._score(missing)
        local = {(s - origin, e - origin): self.scores.get((s, e)) for s, e in spans}
        segs = segmental_decode(sub, local, list(local), self.labels, dict(self.cfg, merge_duplicates=False))
        horizon = known - self.lag if not final else known
        committed = []
        self.preview = None
        for g in segs:
            if g['end'] + origin <= horizon:
                continue
            st, en = g['start'] + origin, g['end'] + origin
            if any(abs(st - a) <= 2 for a, _ in self.shown_early):
                break
            record = self.scores.get((st, en))
            if record is not None and record.get('v_raw') is not None:
                p = np.exp(log_softmax(record['v_raw']))
                order = np.argsort(p)[::-1][:3]
                self.preview = dict(gloss=self.labels[int(order[0])], score=float(p[order[0]]),
                                    emit=g['gloss'] is not None, start_frame=st, end_frame=en,
                                    top3=[dict(gloss=self.labels[int(i)], model_score=float(p[i])) for i in order])
            break
        gap = self.cfg.get('duplicate_gap', 2)
        if self.early_k and not final:
            nxt = {}
            for g in segs:
                if g['end'] + origin <= horizon or not g['gloss']:
                    continue
                st = g['start'] + origin
                prev = self.early.get(st)
                count = prev[1] + 1 if prev and prev[0] == g['gloss'] else 1
                nxt[st] = (g['gloss'], count)
                if (count >= self.early_k and np.exp(g['log_q']) >= self.early_q
                        and g['gloss'] not in self.early_unsafe and not g['gloss'].startswith('FS_')
                        and g['end'] - g['start'] + 1 >= self.early_min
                        and not any(abs(st - a) <= 2 for a, _ in self.shown_early)):
                    if self.last is not None and self.last['gloss'] == g['gloss'] and st - self.last['end_frame'] <= gap:
                        self.shown_early.append((st, self.last))
                        continue
                    self.last = self._word(g['gloss'], st, g['end'] + origin, T, g['log_q'], early=True)
                    self.shown_early.append((st, self.last))
                    committed.append(self.last)
            self.early = nxt
        new_origin = origin
        pending = horizon + 1   # start of the first segment left undecided at this step
        for g in segs:
            end = g['end'] + origin
            if end > horizon:
                pending = g['start'] + origin
                break
            if not final and self.soft_extra and end + 1 <= known:
                p = np.exp(self.bio[end + 1 - self.bio_off])
                if p[O] + p[UNK] + p[B] < self.hard and end > horizon - self.soft_extra:
                    pending = g['start'] + origin
                    break
            start = g['start'] + origin
            shown = [w for a, w in self.shown_early if abs(start - a) <= 2]
            if shown:
                # The word is already on screen; record where the sign actually ended.
                shown[0]['end_frame'] = max(shown[0]['end_frame'], end)
                shown[0]['end_seconds'] = self.obs[min(max(end - self.off, 0), len(self.obs) - 1)].seconds
                shown[0]['closed'] = True
                new_origin = end + 1
                continue
            if g['gloss']:
                same_gap = self.letter_gap if g['gloss'].startswith('FS_') else gap
                if self.last is not None and self.last['gloss'] == g['gloss'] and start - self.last['end_frame'] <= same_gap:
                    self.last['end_frame'] = end
                else:
                    self.last = self._word(g['gloss'], start, end, T, g['log_q'])
                    committed.append(self.last)
            new_origin = end + 1
        # Rest never needs revisiting beyond the longest candidate span: bound the DP window.
        self.origin = max(new_origin, min(pending, horizon + 1 - 30))
        self.shown_early = [e for e in self.shown_early if e[0] >= self.origin - 2]
        self._trim()
        return committed

    def score_letters(self, start_frame, end_frame):
        """Letter-head (label, probability) for one span of absolute frames, or None."""
        if not self.letters or start_frame < self.off or end_frame - self.off >= len(self.obs):
            return None
        ids = self._span_obs(start_frame, end_frame)
        if len(ids) < 4:
            return None
        x = self.recognizer.inputs([self.obs[i] for i in ids], [self.hands[i] for i in ids])
        if x is None:
            return None
        row = self.recognizer.logits([x])[0]
        p = np.exp(log_softmax(row[len(self.recognizer.labels):]))
        k = int(p[:-1].argmax())
        return self.recognizer.letter_classes[k], float(p[k])

    def _word(self, gloss, start, end, T, log_q, early=False):
        ob = lambda k: self.obs[min(max(k - self.off, 0), len(self.obs) - 1)].seconds
        return dict(gloss=gloss, start_frame=start, end_frame=end, commit_frame=T,
                    start_seconds=ob(start), end_seconds=ob(end), commit_seconds=ob(T),
                    score=float(np.exp(log_q)), early=early)


COREML = dict(boundary=ROOT / 'artifacts/coreml/AVBoundaryStudentV17L6FP16.mlpackage',
              recognizer=ROOT / 'artifacts/coreml/SpanRecognizerV17LocalABatchedFP16.mlpackage')


class CoreMLBoundary:
    """Core ML boundary student with the call signature AVBoundaryStream expects."""

    def __init__(self, path, lookahead, config):
        import coremltools as ct
        import torch
        self.torch = torch
        self.model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL)
        self.output = self.model.get_spec().description.output[0].name
        self.lookahead, self.config = lookahead, config

    def eval(self):
        return self

    def __call__(self, x, valid):
        out = self.model.predict({'features': x.detach().cpu().numpy().astype(np.float32),
                                  'valid': valid.detach().cpu().numpy().astype(np.float32)})
        return self.torch.from_numpy(np.asarray(out[self.output], np.float32).reshape(1, -1))


def build_runtime(config=None, device='mps', image_encoder=None, backend=None, compute_units=None):
    """Load the frozen configuration's boundary student and span recognizer.

    backend='coreml' runs both models from their FP16 Core ML packages (the phone path).
    """
    import torch
    from active.v17.av_boundary_v17 import AVBoundary, FORMAT
    config = json.loads(Path(config or CONFIG).read_text())
    backend = backend or config.get('backend', 'torch')
    compute_units = compute_units or config.get('compute_units', 'ALL')
    payload = torch.load(ROOT / config['boundary'], map_location='cpu', weights_only=False)
    if payload.get('format') != FORMAT:
        raise ValueError('not an Apple Vision boundary student')
    boundary = AVBoundary(**payload['config'])
    boundary.load_state_dict(payload['state_dict'])
    boundary.eval()
    if backend == 'coreml':
        boundary = CoreMLBoundary(COREML['boundary'], boundary.lookahead, boundary.config)
    if image_encoder is None:
        from scripts.live_reel_stage1_v17 import parser
        image_encoder = parser().parse_args([]).image_encoder
    recognizer = SpanRecognizer(ROOT / config['recognizer'], image_encoder, device,
                                coreml=(ROOT / config['coreml_recognizer'] if config.get('coreml_recognizer')
                                        else COREML['recognizer']) if backend == 'coreml' else None,
                                compute_units=compute_units)
    if config.get('letter_head'):
        recognizer.attach_letters(ROOT / config['letter_head'])
        if config.get('letter_threshold') is not None:
            recognizer.letter_threshold = float(config['letter_threshold'])
        if backend == 'coreml' and not recognizer.coreml_letter_output:
            raise ValueError('Core ML recognizer package has no letter output; export it with --letter-head')
    if config.get('letter_boundary'):
        payload = torch.load(ROOT / config['letter_boundary'], map_location='cpu', weights_only=False)
        letter_boundary = AVBoundary(**payload['config'])
        letter_boundary.load_state_dict(payload['state_dict'])
        letter_boundary.eval()
        if backend == 'coreml':
            letter_boundary = CoreMLBoundary(ROOT / config['coreml_letter_boundary'], letter_boundary.lookahead,
                                             letter_boundary.config)
        word_mode = 'both' if config.get('word_decoder_letters') else 'words'
        return DualRuntime(SegmentalRuntime(boundary, recognizer, config, mode=word_mode),
                           SegmentalRuntime(letter_boundary, recognizer, config, mode='letters'))
    return SegmentalRuntime(boundary, recognizer, config)


class SpellingBuffer:
    """Turn committed FS_x letters into spelled words; runs shorter than `minimum` are dropped.

    Letters are kept exactly as recognized (no dictionary correction). A run ends at the next
    non-letter word, after `gap_seconds` without a new letter, or on flush().
    """

    def __init__(self, minimum=2, gap_seconds=2.0, same_letter_seconds=.5, rescore=None,
                 moving=('FS_J', 'FS_Z'), moving_gap_seconds=.4, moving_probability=.6):
        self.minimum, self.gap_seconds, self.same_letter_seconds = minimum, gap_seconds, same_letter_seconds
        # J and Z are a static handshape plus a stroke; the decoder can cut the still part off as
        # its own letter (I/Y before J, X inside Z). Adjacent different letters are re-scored as
        # one span and replaced when that span is confidently a moving letter.
        self.rescore, self.moving = rescore, set(moving)
        self.moving_gap_seconds, self.moving_probability = moving_gap_seconds, moving_probability
        self.letters, self.dropped, self.merged = [], [], []

    def pending(self):
        return ''.join(w['gloss'][3:] for w in self.letters)

    def push(self, word):
        if word['gloss'].startswith('FS_'):
            out = []
            last = self.letters[-1] if self.letters else None
            if last is not None and word['start_seconds'] - last['end_seconds'] > self.gap_seconds:
                out = self.flush()
                last = None
            # One held letter can be committed in two pieces (two decoders or a split hold).
            if (last is not None and last['gloss'] == word['gloss']
                    and word['start_seconds'] - last['end_seconds'] <= self.same_letter_seconds):
                last['end_seconds'] = max(last['end_seconds'], word['end_seconds'])
                last['end_frame'] = max(last['end_frame'], word['end_frame'])
                last['commit_seconds'] = max(last['commit_seconds'], word['commit_seconds'])
                return out
            if (last is not None and self.rescore is not None and last['gloss'] != word['gloss']
                    and word['start_seconds'] - last['end_seconds'] <= self.moving_gap_seconds):
                start = min(last['start_frame'], word['start_frame'])
                end = max(last['end_frame'], word['end_frame'])
                scored = self.rescore(start, end)
                if scored is not None and scored[0] in self.moving and scored[1] >= self.moving_probability \
                        and scored[0] in (last['gloss'], word['gloss'], *self.moving):
                    self.merged.append(dict(pieces=[last['gloss'], word['gloss']], letter=scored[0], p=scored[1]))
                    last.update(gloss=scored[0], start_frame=start, end_frame=end,
                                start_seconds=min(last['start_seconds'], word['start_seconds']),
                                end_seconds=max(last['end_seconds'], word['end_seconds']),
                                commit_seconds=max(last['commit_seconds'], word['commit_seconds']))
                    return out
            self.letters.append(dict(word))
            return out
        return self.flush() + [word]

    def tick(self, seconds):
        if self.letters and seconds - self.letters[-1]['commit_seconds'] > self.gap_seconds + .5:
            return self.flush()
        return []

    def flush(self):
        letters, self.letters = self.letters, []
        if len(letters) < self.minimum:
            self.dropped += letters
            return []
        spelled = ''.join(w['gloss'][3:] for w in letters)
        return [dict(gloss='fs-' + spelled, spelled=spelled, start_frame=letters[0]['start_frame'],
                     end_frame=letters[-1]['end_frame'], start_seconds=letters[0]['start_seconds'],
                     end_seconds=letters[-1]['end_seconds'], commit_seconds=letters[-1]['commit_seconds'],
                     score=float(np.mean([w['score'] for w in letters])), early=False,
                     letters=[w['gloss'] for w in letters])]


class DualRuntime:
    """Word decoder (word boundary, 100 signs) + letter decoder (letter boundary, FS_A..FS_Z only).

    Both see the same frames and share the per-frame hand-crop embeddings and the recognizer. Word
    output is exactly the word decoder's. A letter is dropped when it overlaps a committed word.
    """

    def __init__(self, words, letters):
        self.words, self.letter_rt = words, letters
        self.shared = {}
        words.raw_cache = letters.raw_cache = self.shared
        self.config, self.recognizer, self.lookahead = words.config, words.recognizer, words.lookahead
        self.letters = True
        self.labels = letters.labels
        self._committed_words = []

    @property
    def timing(self):
        if self.words.timing is None:
            return None
        merged = {}
        for part in (self.words.timing, self.letter_rt.timing or {}):
            for k, v in part.items():
                merged.setdefault(k, []).extend(v)
        return merged

    @timing.setter
    def timing(self, value):
        self.words.timing = None if value is None else {}
        self.letter_rt.timing = None if value is None else {}

    @property
    def preview(self):
        return self.words.preview

    def score_letters(self, start_frame, end_frame):
        return self.letter_rt.score_letters(start_frame, end_frame)

    def reset(self):
        self.words.reset()
        self.letter_rt.reset()
        self.shared.clear()
        self._committed_words = []

    def _merge(self, word_out, letter_out):
        """Word-decoder signs always stand. A letter (from either decoder) is dropped when it
        overlaps something already committed, so the two decoders never double a letter."""
        kept = []
        for w in sorted(word_out + letter_out, key=lambda w: w['commit_seconds']):
            if w['gloss'].startswith('FS_'):
                span = max(1e-6, w['end_seconds'] - w['start_seconds'])
                # Overlap relative to the shorter segment, so containment in either direction counts.
                if any((min(b, w['end_seconds']) - max(a, w['start_seconds'])) / max(1e-6, min(span, b - a)) >= .5
                       for a, b in self._committed_words):
                    continue
            elif w not in word_out:
                continue
            kept.append(w)
            self._committed_words.append((w['start_seconds'], w['end_seconds']))
        self._committed_words = self._committed_words[-32:]
        oldest = min(self.words.origin, self.letter_rt.origin)
        for k in [k for k in self.shared if k[0] < oldest]:
            del self.shared[k]
        return kept

    def observe(self, observation):
        word_out = self.words.observe(observation)
        letter_out = self.letter_rt.observe(observation, hands=self.words.hands[-1])
        return self._merge(word_out, letter_out)

    def finish(self):
        return self._merge(self.words.finish(), self.letter_rt.finish())


def split_clauses(words, pause_seconds=1.5):
    """Split an utterance at long pauses (hands resting between sentences).

    Stage 3 was trained on single sentences and drops later clauses when several arrive together.
    A spelled word (fs-...) never starts a clause: signers pause before fingerspelling a name.
    """
    clauses = []
    for w in words:
        if (clauses and not w['gloss'].startswith('fs-')
                and w['start_seconds'] - clauses[-1][-1]['end_seconds'] >= pause_seconds):
            clauses.append([w])
        elif clauses:
            clauses[-1].append(w)
        else:
            clauses.append([w])
    return clauses

