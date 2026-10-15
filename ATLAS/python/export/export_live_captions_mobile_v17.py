"""Export the desktop live-caption engine for the iPhone app, plus Swift parity cases.

The phone runs the same five models as `scripts/live_captions_v17.py` (two live taggers, the
draft tagger, the clause labeler and the mover) in `LiveCaptions.swift`, with the same rules.
This writes:

  <models>/live_captions_v17.bin    every weight as little-endian float32, back to back
  <models>/live_captions_v17.json   vocabulary, policy, and {model: {tensor: [offset, shape]}}
  <report>/parity_cases.json        timed push/tick/finish streams with the Python captions

Phone policy (desktop Live default): settle the tail after a 1.5 s pause, and start a new
sentence after 3.5 s. The phone speaks only at Finish, so `speak` is not exported.

  venv/bin/python -m ATLAS.python.export.export_live_captions_mobile_v17
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from scripts.live_captions_v17 import DEFAULT_MODEL_DIR, CaptionModels, LiveCaptions

ROOT = Path(__file__).resolve().parents[3]
MOBILE_MODELS = ROOT / 'ATLAS/mobile_app/slt_mobile_app/ios/Runner/LiveReel/Models'
REPORT = ROOT / 'artifacts/reports/mobile_live_captions_20261015'
PHONE_CASES = ROOT / 'experiments/streaming_translation/cases.json'
END_SECONDS, FREEZE_SECONDS, TICK = 1.5, 3.5, 0.05


def write_chunked(path: Path, data: bytes, chunk: int = 1 << 20):
    """Write `data` unless the file already holds exactly it. The repository volume (exFAT through
    FSKit) fails with ENXIO when a large existing file is truncated and rewritten, so a new file is
    written in 1 MB chunks under a temporary name and then renamed into place."""
    if path.exists() and path.stat().st_size == len(data) and path.read_bytes() == data:
        return
    temporary = path.with_name(path.name + '.partial')
    with open(temporary, 'wb') as f:
        for start in range(0, len(data), chunk):
            f.write(data[start:start + chunk])
    temporary.replace(path)


def export_weights(models: CaptionModels, out_dir: Path) -> dict:
    named = {'live_0': models.live[0], 'live_1': models.live[1], 'draft': models.draft,
             'labeler': models.labeler, 'mover': models.mover}
    index, chunks, offset = {}, [], 0
    for name, module in named.items():
        index[name] = {}
        for key, tensor in module.state_dict().items():
            values = tensor.detach().cpu().numpy().astype('<f4')
            index[name][key] = [offset, list(values.shape)]
            chunks.append(values.ravel())
            offset += values.size
    blob = np.concatenate(chunks).astype('<f4').tobytes()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_chunked(out_dir / 'live_captions_v17.bin', blob)
    manifest = models.manifest
    spec = {
        'format': 'slt_live_captions_mobile_v1',
        'source': manifest['name'],
        'source_sha256': manifest['sha256'],
        'weights_sha256': hashlib.sha256(blob).hexdigest(),
        'architecture': manifest['architecture'],
        'policy': {**manifest['policy'], 'end_of_turn_seconds': END_SECONDS, 'freeze_seconds': FREEZE_SECONDS},
        'vocab': models.vocab,
        'tensors': index,
    }
    write_chunked(out_dir / 'live_captions_v17.json', json.dumps(spec, separators=(',', ':')).encode())
    return spec


def stream(models: CaptionModels, glosses, scores, arrivals, finish_after):
    """push at each arrival, tick every TICK seconds in between, then Finish."""
    captions = LiveCaptions(models, end_seconds=END_SECONDS, freeze_seconds=FREEZE_SECONDS)
    events = []
    clock = 0.0
    for g, s, at in zip(glosses, scores, arrivals):
        while clock + TICK < at:
            clock = round(clock + TICK, 4)
            u = captions.tick(clock)
            events.append({'op': 'tick', 'seconds': clock, 'text': None if u is None else u.text})
        clock = at
        u = captions.push(g, s, at)
        events.append({'op': 'push', 'gloss': g, 'score': s, 'seconds': at, 'text': u.text})
    end = clock + finish_after
    while clock + TICK < end:
        clock = round(clock + TICK, 4)
        u = captions.tick(clock)
        events.append({'op': 'tick', 'seconds': clock, 'text': None if u is None else u.text})
    events.append({'op': 'finish', 'text': captions.finish().text})
    return events


def parity_cases(models: CaptionModels, synthetic: int, seed: int) -> list:
    cases = []
    for c in json.loads(PHONE_CASES.read_text()):
        raw = c.get('arrivals') or [1.2 * i for i in range(len(c['glosses']))]   # untimed diagnostics
        scores = c.get('scores') or [0.9] * len(c['glosses'])
        arrivals = [round(a - raw[0] + 0.3, 4) for a in raw]
        cases.append({'id': c['id'], 'events': stream(models, c['glosses'], scores, arrivals, 2.0)})
    rng = random.Random(seed)
    vocab = sorted(models.known)
    letters = [chr(ord('A') + i) for i in range(26)]
    for k in range(synthetic):
        n = rng.randint(1, 14)
        glosses, scores, arrivals, t = [], [], [], 0.3
        for _ in range(n):
            # A spelled name. Never two in a row: after a 3.5 s freeze the desktop engine crashes on a
            # second consecutive name (merge_fingerspelling adds no new slot, so the tagger input is empty).
            if rng.random() < 0.06 and not (glosses and glosses[-1].startswith('fs-')):
                glosses.append('fs-' + ''.join(rng.choice(letters) for _ in range(rng.randint(2, 4))))
            else:
                glosses.append(rng.choice(vocab))
            scores.append(round(rng.choice([rng.uniform(0.25, 0.6), rng.uniform(0.6, 1.0)]), 4))
            arrivals.append(round(t, 4))
            t += rng.choice([0.6, 0.9, 1.2, 1.4, 1.7, 2.6, 4.0])
        finish_after = rng.choice([0.2, 1.0, 2.0, 5.0])
        cases.append({'id': f'synthetic_{k:03d}', 'events': stream(models, glosses, scores, arrivals, finish_after)})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--out-dir', type=Path, default=MOBILE_MODELS)
    parser.add_argument('--report-dir', type=Path, default=REPORT)
    parser.add_argument('--synthetic', type=int, default=300)
    parser.add_argument('--seed', type=int, default=20261015)
    args = parser.parse_args()
    models = CaptionModels(args.model_dir)
    spec = export_weights(models, args.out_dir)
    cases = parity_cases(models, args.synthetic, args.seed)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_chunked(args.report_dir / 'parity_cases.json', json.dumps(cases).encode())
    print(f"exported {spec['source']} ({spec['weights_sha256'][:12]}); {len(cases)} parity cases")


if __name__ == '__main__':
    main()
