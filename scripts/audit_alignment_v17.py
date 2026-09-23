"""Does the forced aligner recover intervals a human already annotated?

The local phrase clips have no ground truth, so the aligner's output there cannot be
checked directly. ASLLRP contiguous spans DO carry curated per-sign intervals. This runs
the identical pipeline - same teacher, same B-split decoding, same DP, same two-model
verification gate - on those annotated clips and measures the distance to the truth.

If verified intervals land on the annotated ones, the gate means what it claims. If they
do not, the local alignments are not trustworthy no matter how confident the verifier is.

Read-only; trains nothing.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
UPSTREAM = ROOT / 'artifacts/reports/pose_boundary_transfer_v17_20260922'
sys.path[:0] = [str(ROOT), str(UPSTREAM), str(UPSTREAM / 'dependencies')]
from scripts.evaluate_temporal_boundary_v17 import arguments, observations
from scripts.live_reel_stage1_v17 import build_components
from active.v17.extract_v17 import AppleVisionDetector
from scripts.align_local_phrases_v17 import decode_segments, align, CONTEXT
from active.v17.pretrained_boundary_v17 import load_pose, FPS

PREPARED = UPSTREAM / 'prepared_manifest.json'
COMBINED = ROOT / 'data/local/combined_dataset_v17_20260922/manifest.json'
REPORT = ROOT / 'artifacts/reports/local_phrase_alignment_v17_20260923'


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import load_file
    from check import model_from_upstream
    from probe import upstream_helpers
    helpers = upstream_helpers()
    weights = ROOT / 'artifacts/models/pose_boundary_dgs_2026'
    teacher = model_from_upstream(json.loads((weights / 'config.json').read_text())).float().eval()
    teacher.load_state_dict(load_file(str(weights / 'model.safetensors')), strict=True)
    teacher.to('mps')

    prepared = json.loads(PREPARED.read_text())['records']
    combined = {r['source_item_id']: r for r in json.loads(COMBINED.read_text())['records']
                if 'source_item_id' in r}
    rows = [r for r in prepared if r['source'] == 'asllrp_contiguous'
            and r['item'] in combined
            and len(combined[r['item']]['target_sequence']) == len(r['intervals'])]
    print('annotated clips: %d, known intervals: %d'
          % (len(rows), sum(len(r['intervals']) for r in rows)), flush=True)

    args = arguments(combined[rows[0]['item']]['video_path'], REPORT / 'audit_sessions')
    args.no_motion_trim = True
    reel = build_components(args)['classifier']
    reel.args.no_motion_trim = reel.full.args.no_motion_trim = True
    detector = AppleVisionDetector(args.minimum_point_confidence)

    records, tick = [], time.perf_counter()
    for row in rows:
        source = combined[row['item']]
        glosses = source['target_sequence']
        truth = [list(map(float, iv)) for iv in row['intervals']]
        pose = load_pose(row)
        processed = helpers['preprocess_pose'](pose)
        data = processed.body.data.filled(0)[:, 0, :, :3].astype('float32')
        times = np.arange(len(data), dtype='float32') / FPS
        features = np.concatenate([data, helpers['compute_velocity'](data, times)], axis=-1)
        with torch.inference_mode():
            logits = teacher(torch.from_numpy(features)[None].to('mps'),
                             timestamps=torch.from_numpy(times)[None].to('mps'))['sign'][0].cpu()
        segments = decode_segments(logits.argmax(-1).numpy().astype(int).tolist(), FPS)
        try:
            obs = observations(source, args, detector)
        except Exception as exc:
            print('skip', row['item'], str(exc)[:60], flush=True)
            continue
        if not obs or not segments:
            continue
        chosen = align(reel, obs, segments, glosses, {})
        for c in chosen:
            a, b = truth[c['index']]
            overlap = max(0., min(b, c['end']) - max(a, c['start']))
            union = max(b, c['end']) - min(a, c['start'])
            records.append(dict(item=row['item'], gloss=c['gloss'],
                                truth=[a, b], aligned=[c['start'], c['end']],
                                start_error=c['start'] - a, end_error=c['end'] - b,
                                iou=overlap / union if union > 0 else 0.,
                                verifier_score=c['verifier_score']))
        print('  %-28s %d/%d verified  %.0fs' % (row['item'][:28], len(chosen), len(glosses),
                                                 time.perf_counter() - tick), flush=True)

    if not records:
        print('no verified intervals on annotated data')
        return
    se = np.array([r['start_error'] for r in records])
    ee = np.array([r['end_error'] for r in records])
    iou = np.array([r['iou'] for r in records])
    total = sum(len(r['intervals']) for r in rows)
    print('\nverified %d of %d annotated intervals (%.1f%%)' % (len(records), total, 100 * len(records) / total))
    print('start error: median %+.3fs  mean %+.3fs  |median| %.3fs' % (np.median(se), se.mean(), np.median(np.abs(se))))
    print('end error:   median %+.3fs  mean %+.3fs  |median| %.3fs' % (np.median(ee), ee.mean(), np.median(np.abs(ee))))
    print('IoU with the annotated interval: median %.2f  mean %.2f' % (np.median(iou), iou.mean()))
    for tol in (.1, .15, .2, .25):
        both = np.mean((np.abs(se) <= tol) & (np.abs(ee) <= tol))
        print('  both edges within +-%.0fms: %5.1f%%   IoU>=0.5: %5.1f%%'
              % (1000 * tol, 100 * both, 100 * np.mean(iou >= .5)))
    (REPORT / 'audit.json').write_text(json.dumps(dict(
        scope='forced aligner run unchanged on ASLLRP contiguous spans that carry curated '
              'per-sign intervals; measures distance from verified alignments to the truth',
        clips=len(rows), annotated_intervals=total, verified=len(records),
        start_error_median=float(np.median(se)), end_error_median=float(np.median(ee)),
        abs_start_error_median=float(np.median(np.abs(se))),
        abs_end_error_median=float(np.median(np.abs(ee))),
        iou_median=float(np.median(iou)), iou_mean=float(iou.mean()),
        within_200ms=float(np.mean((np.abs(se) <= .2) & (np.abs(ee) <= .2))),
        records=records), indent=1) + '\n')
    print('wrote', REPORT / 'audit.json')


if __name__ == '__main__':
    main()
