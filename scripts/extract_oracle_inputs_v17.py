"""Freeze the oracle-interval verifier inputs so any PyTorch Stage-1 checkpoint can be scored.

scripts/experiment_oracle_ceiling_v17.py scores the live Core ML verifier on the curated
intervals of the 12 held-out ASLLRP development videos (66.7% at pad 0.05/0.10). Scoring a
new checkpoint that way needs a Core ML export per candidate. This runs the identical live
path once (Apple Vision observations, landmarks_from_observations, hand crops, MobileCLIP2
Core ML image encoder) and saves the unified model's four inputs per (event, pad), plus the
live verifier's own answer so the saved inputs can be checked to reproduce the reference.

Read-only with respect to models; trains nothing and promotes nothing.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_temporal_boundary_v17 import arguments, observations, evaluation_rows
from scripts.live_reel_stage1_v17 import build_components
from scripts.live_isolated_v17 import hand_inputs, landmarks_from_observations
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.schema_v17 import V17Config

REPORT = ROOT / 'artifacts/reports/unfrozen_phrase_adapt_v17_20260925/oracle_inputs'
PADS = (0., .05, .1, .15, .2)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    rows = evaluation_rows(report=REPORT)
    args = arguments(rows[0]['video_path'], REPORT / 'sessions')
    args.no_motion_trim = True
    reel = build_components(args)['classifier']
    reel.args.no_motion_trim = reel.full.args.no_motion_trim = True
    detector = AppleVisionDetector(args.minimum_point_confidence)
    labels = set(reel.labels)

    keep = {k: [] for k in ('landmarks', 'hand_embeddings', 'hand_valid', 'hand_boxes')}
    meta, tick = [], time.perf_counter()
    for row in rows:
        obs = observations(row, args, detector)
        for number, event in enumerate(row['events']):
            gloss = str(event['label'])
            if gloss not in labels:
                continue
            for pad in PADS:
                sel = [o for o in obs if event['start'] - pad <= o.seconds <= event['end'] + pad]
                if len(sel) < 4:
                    continue
                live = reel.full.classify(sel)
                try:
                    features, diag = landmarks_from_observations(sel, V17Config())
                except ValueError as error:
                    meta.append(dict(video=row['source_item_id'], event=number, gloss=gloss,
                                     pad=pad, usable=False, reason=str(error),
                                     live_gloss=live.get('candidate_gloss')))
                    continue
                crops, valid, boxes = hand_inputs(
                    sel, int(diag['trim_start']), int(diag['trim_end_exclusive']))
                keep['landmarks'].append(features.astype(np.float32))
                keep['hand_embeddings'].append(reel.full.encode_hands(crops, valid))
                keep['hand_valid'].append(np.asarray(valid, np.bool_))
                keep['hand_boxes'].append(np.asarray(boxes, np.float32))
                meta.append(dict(video=row['source_item_id'], event=number, gloss=gloss,
                                 pad=pad, usable=True, live_gloss=live.get('candidate_gloss')))
        print('  %-28s %.0fs' % (row['source_item_id'][:28], time.perf_counter() - tick), flush=True)

    np.savez_compressed(REPORT / 'oracle_inputs.npz', **{k: np.stack(v) for k, v in keep.items()})
    (REPORT / 'oracle_inputs.json').write_text(json.dumps(dict(
        scope='curated ground-truth intervals, 12 held-out ASLLRP development videos',
        pads=PADS, rows=meta, reel=reel.provenance()), indent=1) + '\n')
    for pad in PADS:
        rows_pad = [m for m in meta if m['pad'] == pad]
        hit = sum(m['live_gloss'] == m['gloss'] for m in rows_pad)
        print('pad %.2f  n=%d  live verifier %d/%d' % (pad, len(rows_pad), hit, len(rows_pad)))


if __name__ == '__main__':
    main()
