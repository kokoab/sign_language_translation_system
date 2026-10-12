#!/usr/bin/env python3
"""Export a unified span recognizer with enumerated batch sizes (all a frame's spans in one call).

Same graph and weights as export_unified_multimodal_coreml_v17.py; only the batch dimension
changes. Parity: batched Core ML vs PyTorch on Citizen validation clips (never test).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

import coremltools as ct
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active.v17.export_stage1_coreml_v17 import replace_attention
from active.v17.export_unified_multimodal_coreml_v17 import ExportWrapper, load_model, load_pair, directory_bytes

BATCHES = (1, 2, 4, 8, 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--precision', choices=('float16', 'float32'), default='float16')
    ap.add_argument('--letter-head', type=Path, default=None, help='also output letter logits (FS_A..FS_Z, NONE)')
    ap.add_argument('--fixed-batch', type=int, default=0,
                    help='export one fixed batch size (loads on iOS 17; the runtime pads to it)')
    ap.add_argument('--landmark-root', type=Path, default=Path('data/local/citizen100_v17/landmarks/val'))
    ap.add_argument('--hand-root', type=Path, default=Path('data/local/citizen100_v17/hand_mobileclip2_s0/val'))
    args = ap.parse_args()
    model, checkpoint = load_model(args.checkpoint)
    export = copy.deepcopy(model)
    replace_attention(export)
    wrapper = ExportWrapper(export).eval()
    head = None
    if args.letter_head is not None:
        from active.v17.letter_head_v17 import LetterHead, SpanWithLetters
        payload = torch.load(args.letter_head, map_location='cpu', weights_only=False)
        head = LetterHead(**payload['config']).eval()
        head.load_state_dict(payload['state_dict'])
        combined = SpanWithLetters(export, head).eval()

        class Both(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, landmarks, hand_embeddings, hand_valid, hand_boxes):
                return self.inner(landmarks, hand_embeddings, hand_valid > 0.5, hand_boxes)
        wrapper = Both(combined).eval()
    paths = sorted(args.landmark_root.glob('*/*.v17.npz'))
    arrays = []
    for path in paths:
        stem = path.name.removesuffix('.v17.npz')
        arrays.append(load_pair(path, args.hand_root / path.parent.name / f'{stem}.hand_mobileclip2_v17.npz'))
    stacked = [np.concatenate([a[i] for a in arrays]) for i in range(4)]
    sample = tuple(torch.from_numpy(v[:4]) for v in stacked)
    with torch.inference_mode():
        got = wrapper(*sample)
        got = got[0] if isinstance(got, tuple) else got
        manual = float((model(sample[0], sample[1], sample[2] > .5, sample[3]) - got).abs().max())
    if manual > 1e-4:
        raise ValueError(f'batched manual attention parity failed: {manual}')
    # Boolean `~mask` traces to bitwise_not, which Core ML rejects once the batch is not 1;
    # logical_not is the same operation on booleans and converts cleanly.
    invert = torch.Tensor.__invert__
    torch.Tensor.__invert__ = lambda self: torch.logical_not(self) if self.dtype == torch.bool else invert(self)
    try:
        traced = torch.jit.trace(wrapper, sample, strict=False)
    finally:
        torch.Tensor.__invert__ = invert
    shapes = dict(landmarks=(32, 61, 5), hand_embeddings=(16, 3, 512), hand_valid=(16, 3), hand_boxes=(16, 3, 4))
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name=name, dtype=np.float32,
                              shape=((args.fixed_batch,) + tail) if args.fixed_batch else
                              ct.EnumeratedShapes(shapes=[(b,) + tail for b in BATCHES], default=(1,) + tail))
                for name, tail in shapes.items()],
        convert_to='mlprogram',
        minimum_deployment_target=ct.target.iOS17 if args.fixed_batch else ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16 if args.precision == 'float16' else ct.precision.FLOAT32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(args.output))
    runtime = ct.models.MLModel(str(args.output), compute_units=ct.ComputeUnit.ALL)
    out_name = runtime.get_spec().description.output[0].name
    output_names = [o.name for o in runtime.get_spec().description.output]
    names = list(shapes)
    mismatch, max_abs = 0, 0.
    with torch.inference_mode():
        reference = model(*(torch.from_numpy(v) if i != 2 else torch.from_numpy(v) > .5 for i, v in enumerate(stacked))).numpy()
    step = args.fixed_batch or 16
    for i in range(0, len(reference) - step + 1, step):
        if args.fixed_batch:
            out = np.asarray(runtime.predict({k: stacked[q][i:i + step] for q, k in enumerate(names)})[out_name]).reshape(step, -1)
            max_abs = max(max_abs, float(np.abs(out - reference[i:i + step]).max()))
            mismatch += int((out.argmax(1) != reference[i:i + step].argmax(1)).sum())
            continue
        n = min(16, len(reference) - i)
        size = max(b for b in BATCHES if b <= n)
        for j in range(i, i + n, size):
            m = min(size, i + n - j)
            m = max(b for b in BATCHES if b <= m)
            out = np.asarray(runtime.predict({k: stacked[q][j:j + m] for q, k in enumerate(names)})[out_name]).reshape(m, -1)
            max_abs = max(max_abs, float(np.abs(out - reference[j:j + m]).max()))
            mismatch += int((out.argmax(1) != reference[j:j + m].argmax(1)).sum())
    timings = {}
    for b in ((args.fixed_batch,) if args.fixed_batch else BATCHES):
        feed = {k: stacked[q][:b] for q, k in enumerate(names)}
        for _ in range(5):
            runtime.predict(feed)
        t = []
        for _ in range(30):
            tick = time.perf_counter(); runtime.predict(feed); t.append(1000 * (time.perf_counter() - tick))
        timings[b] = dict(median_ms=statistics.median(t), per_span_ms=statistics.median(t) / b)
    print(json.dumps(dict(format='slt_v17_span_recognizer_batched_coreml', checkpoint=str(args.checkpoint),
                          letter_head=str(args.letter_head) if args.letter_head else None, outputs=output_names,
                          precision=args.precision, minimum_deployment_target='iOS17' if args.fixed_batch else 'iOS18', batches=[args.fixed_batch] if args.fixed_batch else list(BATCHES), package_mib=directory_bytes(args.output) / 2**20,
                          parity_samples=len(reference), parity_top1_mismatches=mismatch, parity_max_abs=max_abs,
                          latency_on_this_mac=timings, parameters=model.parameter_count,
                          test_accessed=False), indent=2))


if __name__ == '__main__':
    main()
