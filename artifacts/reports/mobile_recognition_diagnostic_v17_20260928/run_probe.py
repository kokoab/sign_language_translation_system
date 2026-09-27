"""Read-only model diagnostic: U/R validation rate sensitivity and HE full-span evidence.

No training, tuning, deployment, or official Citizen test access. Rates are offline
sampling interventions, not device throughput measurements or proposed defaults.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from active.v17.segmental_runtime_v17 import build_runtime, SpellingBuffer, log_softmax
from active.v17.extract_v17 import AppleVisionDetector
from scripts.evaluate_temporal_boundary_v17 import arguments
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
from scripts.prefix_confusion_v17 import prefix

OUT = Path(__file__).parent
torch.set_num_threads(4)
runtime = build_runtime(device='cpu')
recognizer = runtime.words.recognizer
args = arguments('data', OUT / 'unused_sessions')
he_only = '--he-only' in sys.argv
records = json.loads((OUT / 'rate_probe.json').read_text())['records'] if he_only else []
selected = []
for letter in ('U', 'R'):
    count = 0
    for path in sorted((ROOT / 'data/local/fingerspelling_letters_v17/spans/validation' / letter).glob('*.npz')):
        with np.load(path) as data:
            if 'landmarks' not in data:
                continue
            meta = json.loads(str(data['meta']))
        video = ROOT / 'data/raw_videos/ASL VIDEOS' / letter / meta['name']
        if not video.is_file():
            raise FileNotFoundError(video)
        selected.append((letter, meta, video))
        count += 1
        if count == 12:
            break

# Identical membership at each rate. Fresh Vision state/cadence for each replay.
for rate in (() if he_only else (20, 10, 30)):
    for letter, meta, video in selected:
        runtime.reset()
        detector = AppleVisionDetector(args.minimum_point_confidence)
        wrists = {'left': None, 'right': None}
        speller = SpellingBuffer(minimum=1)
        cap = cv2.VideoCapture(str(video))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        assert source_fps > 0 and cap.isOpened(), video
        index = processed = 0
        deadline = 0.
        result, raw, frame_ms = [], [], []
        started = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            seconds = index / source_fps
            index += 1
            if seconds + 1e-6 < deadline:
                continue
            deadline = max(deadline + 1 / rate, seconds)
            tick = time.perf_counter()
            obs = observe_stage2_frame(frame, seconds, processed, detector, wrists, args)
            words = runtime.observe(obs)
            result += speller.tick(seconds)
            raw += [dict(w) for w in words]
            for w in words:
                result += speller.push(w)
            frame_ms.append((time.perf_counter() - tick) * 1000)
            processed += 1
        cap.release()
        tail = runtime.finish()
        raw += [dict(w, eof=True) for w in tail]
        for w in tail:
            result += speller.push(w)
        result += speller.flush()
        glosses = [w['gloss'] for w in result]
        records.append(dict(letter=letter, name=meta['name'], session=meta['session'], rate=rate,
                            source_fps=source_fps, processed=processed, glosses=glosses,
                            correct=glosses == ['fs-' + letter], raw=raw,
                            frame_ms_median=float(np.median(frame_ms)),
                            frame_ms_p90=float(np.percentile(frame_ms, 90)),
                            wall_seconds=time.perf_counter()-started))
        (OUT / 'rate_probe.json').write_text(json.dumps(dict(records=records), indent=2))
    print('rate', rate, {l: sum(r['correct'] for r in records if r['rate']==rate and r['letter']==l)
                        for l in ('U','R')}, flush=True)

# Full spans: isolate identity and letter competition from online segmentation.
he = []
for source in ('citizen', 'semlex', 'local'):
    with np.load(ROOT / 'artifacts/generated/unfrozen_phrase_adapt_v17' / (source + '_val.npz'), allow_pickle=False) as data:
        targets = data['targets']
        ids = np.flatnonzero(np.isin(targets, [recognizer.labels.index(g) for g in ('HE', 'HUNGRY')]))
        fields = ('landmarks', 'hand_embeddings', 'hand_valid', 'hand_boxes')
        arrays = {k: data[k][ids] for k in fields}
        item_ids = data['item_ids'][ids]
    for j, item_id in enumerate(item_ids):
        values = tuple(arrays[k][j] for k in fields)
        logits = recognizer.logits([values])[0]
        wp = np.exp(log_softmax(logits[:100]))
        lp = np.exp(log_softmax(logits[100:]))
        top = np.argsort(wp)[-3:][::-1]
        lk = int(lp[:-1].argmax())
        expected = recognizer.labels[int(targets[ids[j]])]
        partial = []
        for fraction in (.35, .5, .65):
            partial_logits = recognizer.logits([prefix(values, fraction)])[0]
            partial_wp = np.exp(log_softmax(partial_logits[:100]))
            k = int(partial_wp.argmax())
            partial.append(dict(fraction=fraction, word=recognizer.labels[k], probability=float(partial_wp[k])))
        he.append(dict(source=source, expected=expected, item_id=str(item_id),
                       word_top3=[(recognizer.labels[k], float(wp[k])) for k in top],
                       prefixes=partial,
                       letter=recognizer.letter_classes[lk], letter_probability=float(lp[lk]),
                       none_probability=float(lp[-1]), correct=recognizer.labels[top[0]]==expected))
summary = dict(rates={str(rate): {l: dict(correct=sum(r['correct'] for r in records if r['rate']==rate and r['letter']==l),
                                         total=12) for l in ('U','R')} for rate in (20,10,30)},
               words_by_source={g: {s: dict(correct=sum(r['correct'] for r in he if r['source']==s and r['expected']==g),
                                     total=sum(r['source']==s and r['expected']==g for r in he))
                                     for s in ('citizen','semlex','local')} for g in ('HE','HUNGRY')})
(OUT / 'rate_probe.json').write_text(json.dumps(dict(summary=summary, records=records, he=he), indent=2))
print(json.dumps(summary), flush=True)
