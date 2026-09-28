"""Candidate arbitration probes on existing validation and user replays; no training."""
import copy
import hashlib
import gzip
import pickle
import importlib.util
import json
import sys
from pathlib import Path
from types import MethodType
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from active.v17.segmental_runtime_v17 import build_runtime, SegmentalRuntime, SpellingBuffer, log_softmax
from active.v17.extract_v17 import AppleVisionDetector
from scripts.evaluate_temporal_boundary_v17 import arguments
from scripts.live_stage2_ctc_v17 import observe_stage2_frame

spec = importlib.util.spec_from_file_location('before_correctness', OUT / 'baseline/segmental_runtime_v17.py')
before = importlib.util.module_from_spec(spec)
spec.loader.exec_module(before)
torch.set_num_threads(4)
rt = build_runtime(device='cpu')
args = arguments('data', OUT / 'unused_sessions')
items = json.loads((ROOT / 'artifacts/reports/live_speed_v17_20260928/membership.json').read_text())
for gloss in ('HOME','YESTERDAY','YOUR','PLEASE'):
    for path in sorted((ROOT/'data/local/citizen100_v17/raw/val'/gloss).glob('*.mp4')):
        if not path.name.startswith('._'): items.append(dict(id=gloss+'/'+path.name,path=str(path),expected=gloss))
# Diagnose the word errors first; all selected validation letters follow.
items.sort(key=lambda x: (x.get('letters', False), x['id']))
if '--short' in sys.argv:
    items = [x for x in items if not x.get('letters') or any(s in x['id'] for s in ('G_10.', 'R_11_', 'U_13.', 'N_11.'))]
original_score = SegmentalRuntime._score
original_logits = rt.recognizer.logits
original_hand = rt.recognizer.frame_hand
records = []
(OUT / 'cache').mkdir(exist_ok=True)
for item in items:
    rt.reset()
    cache_path = OUT / 'cache' / (hashlib.sha256(item['id'].encode()).hexdigest()[:16] + '.pkl.gz')
    cap = cv2.VideoCapture(item['path'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert cap.isOpened() and fps > 0, item['id']
    detector = AppleVisionDetector(args.minimum_point_confidence)
    wrists = {'left': None, 'right': None}
    observations, hand_cache = [], {}
    index, deadline = 0, 0.
    blank_frames = {}
    while not cache_path.exists():
        ok, frame = cap.read()
        if not ok: break
        stamps = item.get('stamps', [])
        if stamps and index >= len(stamps): break
        seconds = stamps[index] if stamps else index / fps
        index += 1
        if 'bounds' in item:
            if seconds < item['bounds'][0]: continue
            if seconds > item['bounds'][1]: break
        if seconds + 1e-6 < deadline: continue
        deadline = max(deadline + .05, seconds)
        obs = observe_stage2_frame(frame, seconds, len(observations), detector, wrists, args)
        hand_cache[id(obs)] = original_hand(obs)
        if obs.frame.shape not in blank_frames: blank_frames[obs.frame.shape] = np.zeros_like(obs.frame)
        obs.frame = blank_frames[obs.frame.shape]
        observations.append(obs)
    cap.release()
    if cache_path.exists():
        with gzip.open(cache_path, 'rb') as f: observations, hand_values = pickle.load(f)
        hand_cache = {id(o):h for o,h in zip(observations, hand_values)}
    else:
        with gzip.open(cache_path, 'wb') as f: pickle.dump((observations, [hand_cache[id(o)] for o in observations]), f)
    rt.recognizer.frame_hand = lambda obs: hand_cache[id(obs)]
    logits_cache = {}
    def cached_logits(batch):
        keys = [hashlib.sha256(b''.join(np.asarray(v).tobytes() for v in values)).digest() for values in batch]
        missing = [i for i, key in enumerate(keys) if key not in logits_cache]
        if missing:
            for i, row in zip(missing, original_logits([batch[i] for i in missing])): logits_cache[keys[i]] = row
        return np.asarray([logits_cache[k] for k in keys])
    rt.recognizer.logits = cached_logits
    runs = {}
    for mode in ('baseline', 'defer'):
        rt.reset()
        rt.letter_rt._score = MethodType(before.SegmentalRuntime._score if mode == 'baseline' else original_score, rt.letter_rt)
        rt.words.cfg['defer_letter_segments'] = mode == 'defer'
        traces = []
        def score(self, spans):
            (before.SegmentalRuntime._score if mode == 'baseline' else original_score)(self, spans)
            if self.mode != 'both': return
            for key in spans:
                raw = self.raw_cache.get(key)
                if raw is None: continue
                wp = np.exp(log_softmax(raw[:100])); lp = np.exp(log_softmax(raw[100:]))
                wi, li = int(wp.argmax()), int(lp[:-1].argmax())
                if mode == 'word_rescue' and wp[wi] >= .95 and wp[wi] >= lp[li] + .05:
                    self.scores[key] = dict(v_raw=np.concatenate([log_softmax(raw[:100]), np.full(26, -1e9)]).astype(np.float32))
                traces.append(dict(span=key, word=self.recognizer.labels[wi], wp=float(wp[wi]),
                                   letter=self.recognizer.letter_classes[li], lp=float(lp[li]), none=float(lp[-1])))
        rt.words._score = MethodType(score, rt.words)
        speller_type = before.SpellingBuffer if mode == 'baseline' else SpellingBuffer
        speller = speller_type(minimum=1 if item.get('letters') else 2)
        result, raw_words = [], []
        for obs in observations:
            words = rt.observe(obs)
            for w in words:
                raw_words.append(dict(w)); result += speller.push(w)
            result += speller.tick(obs.seconds) if mode == 'baseline' else speller.tick(obs.seconds, active_hands=any(h is not None for h in obs.assigned.values()))
        for w in rt.finish():
            raw_words.append(dict(w)); result += speller.push(w)
        result += speller.flush()
        runs[mode] = dict(glosses=[w['gloss'] for w in result], words=result, raw=raw_words, traces=traces)
    records.append(dict(id=item['id'], expected=item['expected'], frames=len(observations), runs=runs))
    (OUT / 'defer_probe.json').write_text(json.dumps(dict(records=records), indent=1))
    print(len(records), len(items), item['id'], {k:v['glosses'] for k,v in runs.items()}, flush=True)
