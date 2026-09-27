"""Per-stage desktop benchmark of the Core ML segmental Reel and a hypothetical iPhone 13 estimate.

Desktop numbers are measured on this Mac over tuning-pool videos (frame by frame, real Apple
Vision). The iPhone 13 figures are NOT measurements: each stage is scaled by a stated factor for
the hardware block it mostly uses (CPU / GPU / Neural Engine), from published chip figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Approximate A15 (iPhone 13, 4-core GPU) vs M4 ratios, slower-is-larger. Ranges, not measurements.
# CPU: single-core Geekbench-class ratio ~1.6-1.8x; GPU: ~1.4 vs ~4.3 FP32 TFLOPS ~3.0-3.5x;
# Neural Engine: 15.8 vs 38 TOPS ~2.4x; Apple Vision mixes ANE/GPU/CPU ~2-3x.
SCALE = dict(cpu=(1.6, 1.8), gpu=(3.0, 3.5), ane=(2.2, 2.6), vision=(2.0, 3.0))
STAGE_HARDWARE = dict(vision='vision', hand_crops_encode=None, boundary=None, span_inputs='cpu',
                      recognizer=None, decode='cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='tune')
    ap.add_argument('--videos', type=int, default=20)
    ap.add_argument('--recognizer-units', default='CPU_AND_GPU')
    ap.add_argument('--encoder', default=None, help='hand crop image encoder package')
    ap.add_argument('--encoder-units', default='ALL')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    import coremltools as ct
    from scripts import segmental_lab_v17 as lab
    from scripts.evaluate_temporal_boundary_v17 import arguments
    from scripts.live_stage2_ctc_v17 import observe_stage2_frame
    from active.v17.extract_v17 import AppleVisionDetector
    from active.v17.segmental_runtime_v17 import build_runtime
    rt = build_runtime(backend='coreml', compute_units=a.recognizer_units, image_encoder=a.encoder)
    # (DualRuntime merges per-stage timing from its word and letter decoders.)
    if a.encoder_units != 'ALL':
        rt.recognizer.encoder = ct.models.MLModel(str(a.encoder or _default_encoder()),
                                                  compute_units=getattr(ct.ComputeUnit, a.encoder_units))
    args = arguments('data', lab.REPORT / 'sessions')
    det = AppleVisionDetector(args.minimum_point_confidence)
    per_frame = []
    for row in lab.rows_for(a.set)[:a.videos]:
        rt.reset()
        cap = cv2.VideoCapture(str(ROOT / row['video_path']))
        fps = cap.get(cv2.CAP_PROP_FPS)
        wr = {'left': None, 'right': None}
        i = processed = 0
        deadline = 0.
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            i += 1
            if t + 1e-6 < deadline:
                continue
            deadline = max(deadline + 1 / 20, t)
            rt.timing = {}
            tick = time.perf_counter()
            obs = observe_stage2_frame(frame, t, processed, det, wr, args)
            vision = time.perf_counter() - tick
            processed += 1
            tick = time.perf_counter()
            rt.observe(obs)
            total_rt = time.perf_counter() - tick
            stages = {k: float(sum(v)) for k, v in rt.timing.items() if k != 'spans_per_call'}
            stages['decode'] = max(0., total_rt - sum(stages.values()))
            stages['vision'] = vision
            stages['spans'] = float(sum(rt.timing.get('spans_per_call', [0])))
            per_frame.append(stages)
        cap.release()
    names = ['vision', 'hand_crops_encode', 'boundary', 'span_inputs', 'recognizer', 'decode']
    desk = {n: 1000 * np.asarray([f.get(n, 0.) for f in per_frame]) for n in names}
    total = sum(desk.values())
    hardware = dict(vision='vision', hand_crops_encode='gpu' if a.encoder_units in ('ALL', 'CPU_AND_GPU') else 'ane',
                    boundary='gpu', span_inputs='cpu', recognizer='gpu' if 'GPU' in a.recognizer_units else 'ane',
                    decode='cpu')
    estimate = {}
    for bound in (0, 1):
        est = sum(desk[n] * SCALE[hardware[n]][bound] for n in names)
        estimate['low' if bound == 0 else 'high'] = dict(
            median_ms=float(np.median(est)), p90_ms=float(np.percentile(est, 90)), p99_ms=float(np.percentile(est, 99)),
            frames_per_second_sustainable=float(1000 / np.median(est)))
    stats = lambda v: dict(median_ms=float(np.median(v)), p90_ms=float(np.percentile(v, 90)), p99_ms=float(np.percentile(v, 99)))
    result = dict(set=a.set, videos=a.videos, frames=len(per_frame), recognizer_units=a.recognizer_units,
                  encoder=str(a.encoder or 'default FP32'), encoder_units=a.encoder_units,
                  spans_per_frame_mean=float(np.mean([f['spans'] for f in per_frame])),
                  desktop_stage_ms={n: stats(desk[n]) for n in names}, desktop_total_ms=stats(total),
                  iphone13_hardware_assumed=hardware, iphone13_scale_factors=SCALE,
                  iphone13_estimate_ms=estimate,
                  scope='desktop measured on this Mac; iPhone 13 figures are scaled estimates, not measurements')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ('frames', 'spans_per_frame_mean', 'desktop_stage_ms', 'desktop_total_ms', 'iphone13_estimate_ms')}, indent=1))


def _default_encoder():
    from scripts.live_reel_stage1_v17 import parser
    return parser().parse_args([]).image_encoder


if __name__ == '__main__':
    main()
