#!/usr/bin/env python3
"""Export the Apple Vision boundary student to Core ML (one window in, O/B/I logits out).

Parity is checked on tuning-pool windows only (never held-out test clips).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import coremltools as ct
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active.v17.av_boundary_v17 import FRAMES, windows
from active.v17.temporal_boundary_v17 import boundary_features


class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, valid):
        return self.model(features, valid > .5)


def tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--precision', choices=('float16', 'float32'), default='float16')
    ap.add_argument('--parity-clips', type=int, default=40)
    args = ap.parse_args()
    from scripts.train_av_boundary_v17 import load
    from scripts import segmental_lab_v17 as lab
    torch.backends.mha.set_fastpath_enabled(False)   # trace the plain attention graph
    model = load(args.checkpoint, 'cpu').eval()
    wrapper = Wrapper(model).eval()
    dim = model.config['input_dim']
    x = torch.zeros(1, FRAMES, dim)
    v = torch.ones(1, FRAMES)
    traced = torch.jit.trace(wrapper, (x, v), strict=False)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name='features', shape=(1, FRAMES, dim), dtype=np.float32),
                ct.TensorType(name='valid', shape=(1, FRAMES), dtype=np.float32)],
        convert_to='mlprogram', minimum_deployment_target=ct.target.iOS15,
        compute_precision=ct.precision.FLOAT16 if args.precision == 'float16' else ct.precision.FLOAT32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(args.output))
    runtime = ct.models.MLModel(str(args.output), compute_units=ct.ComputeUnit.ALL)
    out_name = runtime.get_spec().description.output[0].name

    mismatch = frames = 0
    max_prob = 0.
    timings = []
    for row in lab.rows_for('tune')[:args.parity_clips]:
        data = np.load(lab.CACHE / 'av_raw' / (lab.key(row['source_item_id']) + '.npz'))
        feats = boundary_features(data['raw'].astype(np.float32), data['times'].astype(np.float64), True)
        xs, vs = windows(feats, model.lookahead)
        with torch.inference_mode():
            ref = torch.softmax(model(torch.from_numpy(xs), torch.from_numpy(vs)), -1).numpy()
        for i in range(len(xs)):
            tick = time.perf_counter()
            out = np.asarray(runtime.predict({'features': xs[i:i + 1], 'valid': vs[i:i + 1].astype(np.float32)})[out_name]).reshape(-1)
            timings.append(1000 * (time.perf_counter() - tick))
            p = np.exp(out - out.max()); p /= p.sum()
            max_prob = max(max_prob, float(np.abs(p - ref[i]).max()))
            mismatch += int(p.argmax() != ref[i].argmax())
            frames += 1
    result = dict(format='slt_v17_av_boundary_coreml_export', checkpoint=str(args.checkpoint),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  parameters=sum(p.numel() for p in model.parameters()), lookahead=model.lookahead,
                  inputs=dict(features=[1, FRAMES, dim], valid=[1, FRAMES]), precision=args.precision,
                  minimum_deployment_target='iOS15', package_mib=tree_bytes(args.output) / 2**20,
                  parity_scope='tuning-pool clips only', parity_frames=frames, parity_top1_mismatches=mismatch,
                  parity_max_prob_abs=max_prob, latency_ms_median=statistics.median(timings),
                  latency_ms_p90=float(np.percentile(timings, 90)), compute_units='ALL')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
