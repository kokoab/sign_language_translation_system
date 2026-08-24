"""Stage 2 (CTC): PyTorch baseline → ONNX export → ONNX Runtime validation.

Stage 2 is harder than Stage 1:
  - Variable-length input T (must be multiple of clip_len=32).
  - Forward returns a tuple (logits, inter_logits).
  - Internal reshape over shape-dependent factors.

We wrap the model in a simpler inference module:
  - assumes T % 32 == 0
  - returns only logits (not inter_logits)
  - exports with dynamic `time` axis
"""
from __future__ import annotations
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from _common import load_stage2, ARTIFACTS, ctc_greedy_decode, load_idx_to_gloss

warnings.filterwarnings('ignore', category=UserWarning)


class Stage2Wrapper(nn.Module):
    """Inference-only wrapper: returns only primary logits."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        logits, _ = self.m(x)
        return logits


def main():
    torch.set_grad_enabled(False)
    model, meta = load_stage2()
    wrapper = Stage2Wrapper(model).eval()
    idx_to_gloss = load_idx_to_gloss(meta)
    print(f'▶ Stage 2 loaded: vocab={meta["vocab_size"]}  d_model={meta["d_model"]}  val_wer={meta["val_wer"]:.2f}%')

    # ── Test inputs ──
    feat_dir = ARTIFACTS / 'test_features'
    # Use a mix: single 32-frame clip + 160-frame phrase
    samples = []
    for name in ['HELLO', 'THANKYOU', 'I', 'YOU']:
        paths = sorted(feat_dir.glob(f'{name}__*.npy'))
        if paths:
            samples.append(('iso', name, paths[0]))
    for p in sorted(feat_dir.glob('PHRASE__*.npy')):
        label = p.stem.split('__')[1]
        samples.append(('phrase', label, p))

    # ── 1) PyTorch baseline ──
    print(f'\n▶ PyTorch baseline on {len(samples)} samples:')
    pt_results = []
    pt_logits_list = []
    for kind, label, p in samples:
        arr = np.load(p).astype(np.float32)  # [T, 61, 5]
        T = arr.shape[0]
        x = torch.from_numpy(arr).unsqueeze(0)  # [1, T, 61, 5]
        t0 = time.perf_counter()
        logits = wrapper(x).cpu().numpy()  # [1, S, V]
        ms = (time.perf_counter() - t0) * 1000
        decoded_idx = ctc_greedy_decode(logits, blank=0)
        decoded_glosses = [idx_to_gloss.get(i, f'<{i}>') for i in decoded_idx]
        print(f'  [{kind:6s}] {label:25s}  T={T:3d}  S={logits.shape[1]:2d}  → {decoded_glosses}  ({ms:.0f}ms)')
        pt_results.append({
            'kind': kind, 'label': label, 'T': T, 'S': logits.shape[1],
            'decoded_idx': decoded_idx, 'decoded_glosses': decoded_glosses,
            'ms': ms, 'logits_shape': list(logits.shape),
        })
        pt_logits_list.append(logits)

    (ARTIFACTS / 'stage2_baseline.json').write_text(json.dumps({
        'checkpoint_val_wer': meta['val_wer'],
        'results': pt_results,
    }, indent=2))

    # ── 2) ONNX export ──
    print('\n▶ Exporting Stage 2 to ONNX (dynamic time axis)...')
    onnx_path = ARTIFACTS / 'stage2.onnx'
    dummy = torch.randn(1, 32, 61, 5, dtype=torch.float32)

    export_ok = False
    last_err = None
    export_attempts = [
        dict(dynamo=False, opset_version=17, label='legacy-ts opset17'),
        dict(dynamo=False, opset_version=18, label='legacy-ts opset18'),
        dict(dynamo=True,  opset_version=18, label='dynamo opset18'),
    ]
    for attempt in export_attempts:
        try:
            torch.onnx.export(
                wrapper, (dummy,), str(onnx_path),
                input_names=['landmarks'],
                output_names=['logits'],
                dynamic_axes={
                    'landmarks': {0: 'batch', 1: 'time'},
                    'logits':    {0: 'batch', 1: 'seq'},
                },
                opset_version=attempt['opset_version'],
                do_constant_folding=True,
                dynamo=attempt['dynamo'],
            )
            print(f'  exported via {attempt["label"]}: {onnx_path}  ({onnx_path.stat().st_size/1e6:.1f} MB)')
            export_ok = True
            break
        except Exception as e:
            last_err = e
            print(f'  [attempt {attempt["label"]} failed] {type(e).__name__}: {str(e)[:240]}')
    if not export_ok:
        raise RuntimeError(f'All export attempts failed. Last error: {last_err}')

    try:
        onnx.checker.check_model(str(onnx_path))
        print('  onnx.checker: PASSED')
    except Exception as e:
        print(f'  onnx.checker: FAILED — {str(e)[:200]}')

    # ── 3) ONNX Runtime validation ──
    print('\n▶ ONNX Runtime validation:')
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    parity = []
    for (kind, label, p), pt_logit in zip(samples, pt_logits_list):
        arr = np.load(p).astype(np.float32)[None, ...]  # [1, T, 61, 5]
        t0 = time.perf_counter()
        ort_logit = sess.run([output_name], {input_name: arr})[0]
        ms = (time.perf_counter() - t0) * 1000
        # Compare
        shape_match = pt_logit.shape == ort_logit.shape
        if not shape_match:
            print(f'  [{kind:6s}] {label:25s}  SHAPE MISMATCH pt={pt_logit.shape} ort={ort_logit.shape}')
            parity.append({'kind': kind, 'label': label, 'error': 'shape_mismatch',
                           'pt_shape': list(pt_logit.shape), 'ort_shape': list(ort_logit.shape)})
            continue
        max_abs = float(np.abs(pt_logit - ort_logit).max())
        mean_abs = float(np.abs(pt_logit - ort_logit).mean())
        pt_decoded = ctc_greedy_decode(pt_logit, blank=0)
        ort_decoded = ctc_greedy_decode(ort_logit, blank=0)
        decode_match = pt_decoded == ort_decoded
        pt_glosses = [idx_to_gloss.get(i, f'<{i}>') for i in pt_decoded]
        ort_glosses = [idx_to_gloss.get(i, f'<{i}>') for i in ort_decoded]
        print(f'  [{kind:6s}] {label:25s}  max|Δ|={max_abs:.2e}  mean|Δ|={mean_abs:.2e}  '
              f'decode={"MATCH" if decode_match else "DIFF"}  ({ms:.0f}ms)')
        if not decode_match:
            print(f'      pt : {pt_glosses}')
            print(f'      ort: {ort_glosses}')
        parity.append({
            'kind': kind, 'label': label, 'T': int(arr.shape[1]),
            'max_abs_diff': max_abs, 'mean_abs_diff': mean_abs,
            'pt_decoded': pt_glosses, 'ort_decoded': ort_glosses,
            'decode_match': decode_match, 'ort_ms': ms,
        })

    ok = [r for r in parity if 'error' not in r]
    summary = {
        'onnx_path': str(onnx_path),
        'onnx_size_mb': onnx_path.stat().st_size / 1e6,
        'num_samples': len(parity),
        'decode_match_rate': (sum(1 for r in ok if r['decode_match']) / max(1, len(ok))),
        'max_abs_diff_global': max([r['max_abs_diff'] for r in ok], default=-1),
        'mean_abs_diff_global': float(np.mean([r['mean_abs_diff'] for r in ok])) if ok else -1,
        'details': parity,
    }
    (ARTIFACTS / 'stage2_onnx_report.json').write_text(json.dumps(summary, indent=2))
    print(f'\n  decode-match rate: {summary["decode_match_rate"]*100:.1f}%')
    print(f'  max|Δ| globally:   {summary["max_abs_diff_global"]:.2e}')


if __name__ == '__main__':
    main()
