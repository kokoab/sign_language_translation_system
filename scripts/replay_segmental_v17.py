"""Replay evaluation videos through the live SegmentalRuntime, frame by frame.

Uses the same Apple Vision observation path as the app shell (observe_stage2_frame at 20 Hz)
and measures per-word latency as (commit frame time - sign end time) + the wall-clock compute
of the committing frame. No training, no defaults changed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='tune')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--backend', choices=('torch', 'coreml', 'tflite'), default=None)
    ap.add_argument('--image-encoder', default=None)
    ap.add_argument('--compute-units', default=None)
    ap.add_argument('--config', default=None)
    ap.add_argument('--no-fingerspelling', action='store_true', help='words only: no letter head / letter decoder')
    ap.add_argument('--detector', choices=('apple', 'mediapipe_full'), default='apple',
                    help='mediapipe_full: Android-family detector (pair with a MediaPipe-trained --config)')
    a = ap.parse_args()
    from scripts import segmental_lab_v17 as lab
    from scripts.evaluate_temporal_boundary_v17 import arguments, edit_counts
    from scripts.live_stage2_ctc_v17 import observe_stage2_frame
    from active.v17.extract_v17 import AppleVisionDetector
    from active.v17.segmental_runtime_v17 import build_runtime
    rt = build_runtime(config=a.config, device=a.device, backend=a.backend, image_encoder=a.image_encoder, compute_units=a.compute_units,
                       fingerspelling=not a.no_fingerspelling)
    args = arguments('data', lab.REPORT / 'sessions')
    if a.detector == 'mediapipe_full':
        from active.v17.mediapipe_full_v17 import MediaPipeFullDetector, MediaPipeFullV17Config
        detector = MediaPipeFullDetector(MediaPipeFullV17Config())
    else:
        detector = AppleVisionDetector(args.minimum_point_confidence)
    rows = lab.rows_for(a.set)
    if a.limit:
        rows = rows[:a.limit]
    records, compute = [], []
    for row in rows:
        rt.reset()
        if a.detector == 'mediapipe_full':
            if detector.calls > 3000:
                detector.renew()  # the macOS GPU leak is per landmarker instance
            detector.reset_sequence()
        cap = cv2.VideoCapture(str(ROOT / row['video_path']))
        fps = cap.get(cv2.CAP_PROP_FPS)
        wrists = {'left': None, 'right': None}
        index = processed = 0
        deadline = 0.
        words = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            seconds = index / fps
            index += 1
            if seconds + 1e-6 < deadline:
                continue
            deadline = max(deadline + 1 / 20, seconds)
            tick = time.perf_counter()
            obs = observe_stage2_frame(frame, seconds, processed, detector, wrists, args)
            processed += 1
            new = rt.observe(obs)
            spent = time.perf_counter() - tick
            compute.append(spent)
            for w in new:
                w['compute_seconds'] = spent
                w['eof'] = False
            words += new
        cap.release()
        tail = rt.finish()
        for w in tail:
            w['compute_seconds'] = 0.
            w['eof'] = True
        words += tail
        # Same post-processing as the app: letter runs -> spelled words, single letters dropped.
        from active.v17.segmental_runtime_v17 import SpellingBuffer
        merge = rt.config['stream'].get('moving_merge', False)
        speller = SpellingBuffer(rescore=getattr(rt, 'score_letters', None) if merge else None)
        final = []
        for w in words:
            final += speller.push(w)
        final += speller.flush()
        letters_dropped = len(speller.dropped)
        for w in final:
            w.setdefault('eof', False)
            w.setdefault('compute_seconds', 0.)
        words = final
        hyp = [w['gloss'] for w in words]
        m = edit_counts(row['target_sequence'], hyp)
        records.append(dict(id=row['source_item_id'], subset=row.get('subset'), reference=row['target_sequence'],
                            hypothesis=hyp, words=words, letters_dropped=letters_dropped, metrics={k: m[k] for k in ('substitutions', 'deletions', 'insertions', 'correct', 'references')}))
        print(row['source_item_id'], row['target_sequence'], hyp, flush=True)

    def summary(recs):
        t = {k: sum(r['metrics'][k] for r in recs) for k in ('substitutions', 'deletions', 'insertions', 'correct', 'references')}
        h = sum(len(r['hypothesis']) for r in recs)
        lat = np.array([w['commit_seconds'] - w['end_seconds'] + w['compute_seconds']
                        for r in recs for w in r['words'] if not w['eof']] or [0.])
        return dict(videos=len(recs), wer=(t['substitutions'] + t['deletions'] + t['insertions']) / t['references'],
                    precision=t['correct'] / max(h, 1), recall=t['correct'] / t['references'], hypotheses=h, **t,
                    latency_median=float(np.median(lat)), latency_p90=float(np.percentile(lat, 90)),
                    latency_max=float(lat.max()), latency_under_05=float((lat < .5).mean()),
                    timed_words=int(sum(not w['eof'] for r in recs for w in r['words'])),
                    eof_words=int(sum(w['eof'] for r in recs for w in r['words'])))
    out = dict(set=a.set, backend=a.backend, fingerspelling=not a.no_fingerspelling, compute_units=a.compute_units, overall=summary(records),
               subsets={s: summary([r for r in records if r['subset'] == s]) for s in sorted({r['subset'] for r in records})},
               long=summary([r for r in records if r['words'] is not None and len(r['reference']) and
                             any(True for _ in [0])]) if False else None,
               compute_ms=dict(median=1000 * float(np.median(compute)), p90=1000 * float(np.percentile(compute, 90)),
                               p99=1000 * float(np.percentile(compute, 99)), max=1000 * float(np.max(compute))),
               config=rt.config, detector=a.detector,
               records=records)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ('overall', 'subsets', 'compute_ms')}, indent=1, default=float))


if __name__ == '__main__':
    main()
