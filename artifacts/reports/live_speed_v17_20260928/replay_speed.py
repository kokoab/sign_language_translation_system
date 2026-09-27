"""Paired old/new inference on identical Vision observations. No training or test split access."""
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import MethodType
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from active.v17 import av_boundary_v17 as av
from active.v17.segmental_runtime_v17 import build_runtime, SpanRecognizer, SpellingBuffer
from active.v17.extract_v17 import AppleVisionDetector
from scripts.evaluate_temporal_boundary_v17 import arguments
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
from scripts import segmental_lab_v17 as lab

spec = importlib.util.spec_from_file_location('speed_old', OUT / 'baseline/segmental_runtime_v17.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
full_features = av.boundary_features
def old_features(raw, times, hand_geometry=True, **kwargs):
    value = full_features(raw, times, hand_geometry)
    return value[-1:] if kwargs.get('last_only') else value

torch.set_num_threads(4)
runtime = build_runtime(device='cpu')
args = arguments('data', OUT / 'unused_sessions')
items = []
for letter in ('U', 'R', 'N', 'G', 'Q'):
    n = 0
    for path in sorted((ROOT / 'data/local/fingerspelling_letters_v17/spans/validation' / letter).glob('*.npz')):
        with np.load(path) as data:
            if 'landmarks' not in data: continue
            meta = json.loads(str(data['meta']))
        items.append(dict(id=letter + '/' + meta['name'], path=str(ROOT / 'data/raw_videos/ASL VIDEOS' / letter / meta['name']),
                          expected='fs-' + letter, letters=True))
        n += 1
        if n == 12: break
for gloss in ('HE', 'HUNGRY', 'HELLO'):
    for path in sorted((ROOT / 'data/local/citizen100_v17/raw/val' / gloss).glob('*.mp4')):
        if path.name.startswith('._'): continue
        items.append(dict(id=gloss + '/' + path.name, path=str(path), expected=gloss))
for row in lab.rows_for('tune')[:4]:
    items.append(dict(id=row['source_item_id'], path=str(ROOT / row['video_path']), expected=row['target_sequence']))
session = json.loads((OUT / 'angelo_history_snapshot.json').read_text())
for lo, hi in ((150, 185), (387, 400)):
    items.append(dict(id='angelo-%d-%d' % (lo, hi), path=session['video'], stamps=session['video_source_timestamps_seconds'],
                      bounds=[lo, hi], expected='user reports repeated ANGELO; boundaries not annotated'))
(OUT / 'membership.json').write_text(json.dumps(items, indent=1))

records = []
for number, item in enumerate(items):
    cap = cv2.VideoCapture(item['path'])
    if not cap.isOpened():
        records.append(dict(id=item['id'], unavailable=True)); continue
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    detector = AppleVisionDetector(args.minimum_point_confidence)
    wrists = {'left': None, 'right': None}
    observations = []
    index = 0
    deadline = 0.
    while True:
        ok, frame = cap.read()
        if not ok: break
        stamps = item.get('stamps', [])
        if stamps and index >= len(stamps): break
        seconds = stamps[index] if stamps else index / source_fps
        index += 1
        if 'bounds' in item:
            if seconds < item['bounds'][0]: continue
            if seconds > item['bounds'][1]: break
        if seconds + 1e-6 < deadline: continue
        deadline = max(deadline + 1 / 20, seconds)
        observations.append(observe_stage2_frame(frame, seconds, len(observations), detector, wrists, args))
    cap.release()
    runs = {}
    embeddings = {}
    for mode in (('before','after') if number % 2 == 0 else ('after','before')):
        runtime.reset()
        runtime.recognizer.frame_hand = MethodType(old.SpanRecognizer.frame_hand if mode=='before' else SpanRecognizer.frame_hand, runtime.recognizer)
        av.boundary_features = old_features if mode=='before' else full_features
        speller = SpellingBuffer(minimum=1 if item.get('letters') else 2)
        words, elapsed, hand = [], [], []
        for observation in observations:
            started = time.perf_counter()
            committed = runtime.observe(observation)
            words += speller.tick(observation.seconds)
            for word in committed: words += speller.push(word)
            elapsed.append((time.perf_counter()-started)*1000)
            hand.append(runtime.words.hands[-1][0].copy())
        for word in runtime.finish(): words += speller.push(word)
        words += speller.flush()
        runs[mode] = dict(words=words, frame_ms=elapsed)
        embeddings[mode] = hand
    max_abs = max((float(np.max(np.abs(a-b))) for a,b in zip(embeddings['before'],embeddings['after'])), default=0.)
    fields = ('gloss','start_frame','end_frame','commit_frame','start_seconds','end_seconds','commit_seconds','score')
    signature = lambda mode: [tuple(w.get(k) for k in fields) for w in runs[mode]['words']]
    records.append(dict(id=item['id'], expected=item['expected'], frames=len(observations),
                        embeddings_max_abs=max_abs, identical=signature('before')==signature('after'), **runs))
    (OUT / 'paired_replay.json').write_text(json.dumps(dict(records=records), indent=1))
    print(number+1, len(items), item['id'], records[-1]['identical'], max_abs, flush=True)
av.boundary_features = full_features
summary = dict(clips=len(records), unavailable=sum(bool(r.get('unavailable')) for r in records),
               changed=sum(not r.get('identical',True) for r in records),
               embeddings_max_abs=max(r.get('embeddings_max_abs',0) for r in records))
for mode in ('before','after'):
    values = [t for r in records if mode in r for t in r[mode]['frame_ms']]
    summary[mode] = dict(frames=len(values), median_ms=float(np.median(values)), p90_ms=float(np.percentile(values,90)), total_seconds=sum(values)/1000)
(OUT / 'paired_replay.json').write_text(json.dumps(dict(summary=summary, records=records), indent=1))
print(json.dumps(summary), flush=True)
