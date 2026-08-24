"""500-sample Stage 2 test: PyTorch monolith vs ONNX vs CoreML split-model.

On isolated-sign clips (T=32), Stage 2's CTC produces 4 output tokens.
Correct if the greedy-decoded gloss sequence starts with the true class.

Compares:
  1. PyTorch full Stage 2 forward
  2. ONNX (stage2.onnx, dynamic-T)
  3. CoreML split: Stage2A_ClipEncoder → Stage2B_SeqCTC
  4. CoreML fixed: Stage2_fixed32 monolith
"""
from __future__ import annotations
import os, sys, json, time, random, warnings
from collections import Counter
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from _common import load_stage2, ARTIFACTS, REPORTS, ctc_greedy_decode, load_idx_to_gloss

warnings.filterwarnings('ignore')


SEED = 42
N_SAMPLES = 500
DATA_DIR = Path('/Users/frnzlo/Documents/machine_learning/SLT/src_v16/ASL_landmarks_v16')
MANIFEST = Path('/Users/frnzlo/Documents/machine_learning/SLT/src_v16/manifest_v16_files_deep_cleaned.json')


def reproduce_test_split(manifest):
    from collections import defaultdict
    files_by_class = defaultdict(list)
    for fname, cls in manifest.items():
        files_by_class[cls].append(fname)
    test_files = []
    for cls in sorted(files_by_class.keys()):
        cls_files = sorted(files_by_class[cls])
        rng = random.Random(); rng.seed(SEED); rng.shuffle(cls_files)
        n = len(cls_files); val_end = int(n * 0.85)
        for fp in cls_files[val_end:]:
            test_files.append((fp, cls))
    return test_files


class Stage2Wrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        logits, _ = self.m(x); return logits


def p_pctl(vals, p):
    s = sorted(vals); return s[min(len(s)-1, int(len(s)*p))]


def main():
    torch.set_grad_enabled(False)
    manifest = json.loads(MANIFEST.read_text())
    test = reproduce_test_split(manifest)
    rng = random.Random(SEED)
    sample = rng.sample(test, N_SAMPLES)

    print(f'▶ Loading models...')
    m2, meta = load_stage2()
    i2g = load_idx_to_gloss(meta)
    g2i = {v: int(k) for k, v in i2g.items()}
    wrap = Stage2Wrapper(m2).eval()

    import onnxruntime as ort
    ort_sess = ort.InferenceSession(str(ARTIFACTS / 'stage2.onnx'),
                                    providers=['CPUExecutionProvider'])
    ort_in = ort_sess.get_inputs()[0].name
    ort_out = ort_sess.get_outputs()[0].name

    import coremltools as ct
    cm_A = ct.models.MLModel(str(ARTIFACTS / 'coreml' / 'Stage2A_ClipEncoder.mlpackage'),
                             compute_units=ct.ComputeUnit.CPU_AND_NE)
    cm_A_in = cm_A.get_spec().description.input[0].name
    cm_A_out = cm_A.get_spec().description.output[0].name
    cm_B = ct.models.MLModel(str(ARTIFACTS / 'coreml' / 'Stage2B_SeqCTC.mlpackage'),
                             compute_units=ct.ComputeUnit.CPU_AND_NE)
    cm_B_in = cm_B.get_spec().description.input[0].name
    cm_B_out = cm_B.get_spec().description.output[0].name
    cm_mono = ct.models.MLModel(str(ARTIFACTS / 'coreml' / 'Stage2_fixed32.mlpackage'),
                                compute_units=ct.ComputeUnit.CPU_AND_NE)
    cm_mono_in = cm_mono.get_spec().description.input[0].name
    cm_mono_out = cm_mono.get_spec().description.output[0].name

    # ── Preload features ──
    print(f'▶ Preloading {N_SAMPLES} .npy files...')
    inputs = []
    for fname, cls in sample:
        arr = np.load(DATA_DIR / fname).astype(np.float32)
        y = g2i.get(cls)
        if y is None:
            continue
        inputs.append({'fname': fname, 'cls': cls, 'y_true': y, 'x': arr})
    N = len(inputs)
    print(f'  usable: {N}')

    # Container for greedy decode
    results = {name: {'decoded': [], 'times': []} for name in
               ['pytorch', 'onnx', 'coreml_split', 'coreml_mono']}
    pt_logits_all, ort_logits_all = [], []

    # ── PyTorch ──
    print('\n▶ Running PyTorch Stage 2...')
    t0 = time.perf_counter()
    for i, r in enumerate(inputs):
        x = torch.from_numpy(r['x']).unsqueeze(0)
        ts = time.perf_counter()
        logits = wrap(x).cpu().numpy()
        results['pytorch']['times'].append((time.perf_counter() - ts) * 1000)
        pt_logits_all.append(logits)
        dec = ctc_greedy_decode(logits, blank=0)
        results['pytorch']['decoded'].append(dec)
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{N}] elapsed {time.perf_counter() - t0:.1f}s')
    print(f'  total: {time.perf_counter() - t0:.1f}s')

    # ── ONNX ──
    print('\n▶ Running ONNX Stage 2...')
    t0 = time.perf_counter()
    for i, r in enumerate(inputs):
        x = r['x'][None, ...]
        ts = time.perf_counter()
        logits = ort_sess.run([ort_out], {ort_in: x})[0]
        results['onnx']['times'].append((time.perf_counter() - ts) * 1000)
        ort_logits_all.append(logits)
        results['onnx']['decoded'].append(ctc_greedy_decode(logits, blank=0))
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{N}] elapsed {time.perf_counter() - t0:.1f}s')
    print(f'  total: {time.perf_counter() - t0:.1f}s')

    # ── CoreML split (A + B) ──
    print('\n▶ Running CoreML Stage 2 split (A + B)...')
    t0 = time.perf_counter()
    cm_split_logits_all = []
    for i, r in enumerate(inputs):
        x = r['x'][None, ...]  # [1, 32, 61, 5]
        ts = time.perf_counter()
        outA = cm_A.predict({cm_A_in: x})
        tokens = np.array(outA[cm_A_out]).astype(np.float32)  # [1, 4, 384]
        outB = cm_B.predict({cm_B_in: tokens})
        logits = np.array(outB[cm_B_out])  # [1, 4, 311]
        results['coreml_split']['times'].append((time.perf_counter() - ts) * 1000)
        cm_split_logits_all.append(logits)
        results['coreml_split']['decoded'].append(ctc_greedy_decode(logits, blank=0))
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{N}] elapsed {time.perf_counter() - t0:.1f}s')
    print(f'  total: {time.perf_counter() - t0:.1f}s')

    # ── CoreML monolith ──
    print('\n▶ Running CoreML Stage 2 monolith (fixed 32)...')
    t0 = time.perf_counter()
    cm_mono_logits_all = []
    for i, r in enumerate(inputs):
        x = r['x'][None, ...]
        ts = time.perf_counter()
        out = cm_mono.predict({cm_mono_in: x})
        logits = np.array(out[cm_mono_out])
        results['coreml_mono']['times'].append((time.perf_counter() - ts) * 1000)
        cm_mono_logits_all.append(logits)
        results['coreml_mono']['decoded'].append(ctc_greedy_decode(logits, blank=0))
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{N}] elapsed {time.perf_counter() - t0:.1f}s')
    print(f'  total: {time.perf_counter() - t0:.1f}s')

    # ── Metrics ──
    print('\n' + '=' * 72)
    print(f'RESULTS — STAGE 2 CTC on {N} isolated 32-frame samples')
    print('=' * 72)

    def first_token_acc(decoded_list):
        """Correct if first non-blank non-repeat decoded token == y_true."""
        correct = 0
        for r, dec in zip(inputs, decoded_list):
            if dec and dec[0] == r['y_true']:
                correct += 1
        return correct / len(inputs) * 100

    def any_token_acc(decoded_list):
        """Correct if y_true appears anywhere in the decoded sequence."""
        correct = 0
        for r, dec in zip(inputs, decoded_list):
            if r['y_true'] in dec:
                correct += 1
        return correct / len(inputs) * 100

    def decode_match(a, b):
        return sum(1 for x, y in zip(a, b) if x == y) / len(a) * 100

    ft = {name: first_token_acc(results[name]['decoded']) for name in results}
    at = {name: any_token_acc(results[name]['decoded']) for name in results}

    print(f'\n{"Format":<18}  {"1st-tok acc":>11}  {"any-tok acc":>11}  {"Avg ms":>8}  {"P95":>8}')
    print('-' * 72)
    for name, label in [('pytorch', 'PyTorch'),
                         ('onnx', 'ONNX Runtime'),
                         ('coreml_split', 'CoreML split A+B'),
                         ('coreml_mono', 'CoreML monolith')]:
        t = results[name]['times']
        print(f'{label:<18}  {ft[name]:10.2f}%  {at[name]:10.2f}%  {np.mean(t):7.2f}  {p_pctl(t, 0.95):7.2f}')

    print(f'\nDecode-sequence match rate (vs PyTorch):')
    print(f'  ONNX            : {decode_match(results["pytorch"]["decoded"], results["onnx"]["decoded"]):.2f}%')
    print(f'  CoreML split    : {decode_match(results["pytorch"]["decoded"], results["coreml_split"]["decoded"]):.2f}%')
    print(f'  CoreML monolith : {decode_match(results["pytorch"]["decoded"], results["coreml_mono"]["decoded"]):.2f}%')

    # Logit divergence
    def max_abs_diff_per_sample(a_list, b_list):
        return [float(np.abs(a - b).max()) for a, b in zip(a_list, b_list)]
    d_ort = max_abs_diff_per_sample(pt_logits_all, ort_logits_all)
    d_split = max_abs_diff_per_sample(pt_logits_all, cm_split_logits_all)
    d_mono = max_abs_diff_per_sample(pt_logits_all, cm_mono_logits_all)
    print(f'\nLogit max|Δ| vs PyTorch (mean / max across {N} samples):')
    print(f'  ONNX            : mean={np.mean(d_ort):.2e}  max={max(d_ort):.2e}')
    print(f'  CoreML split    : mean={np.mean(d_split):.2e}  max={max(d_split):.2e}')
    print(f'  CoreML monolith : mean={np.mean(d_mono):.2e}  max={max(d_mono):.2e}')

    # Write report
    rep = {
        'n_samples': N,
        'checkpoint_val_wer': meta['val_wer'],
        'accuracy_first_token': ft,
        'accuracy_any_token': at,
        'decode_match_rate_vs_pytorch': {
            'onnx': decode_match(results["pytorch"]["decoded"], results["onnx"]["decoded"]),
            'coreml_split': decode_match(results["pytorch"]["decoded"], results["coreml_split"]["decoded"]),
            'coreml_mono': decode_match(results["pytorch"]["decoded"], results["coreml_mono"]["decoded"]),
        },
        'logit_divergence_max_abs': {
            'onnx': {'mean': float(np.mean(d_ort)), 'max': max(d_ort)},
            'coreml_split': {'mean': float(np.mean(d_split)), 'max': max(d_split)},
            'coreml_mono': {'mean': float(np.mean(d_mono)), 'max': max(d_mono)},
        },
        'latency_ms': {
            name: {
                'mean': float(np.mean(results[name]['times'])),
                'p50': p_pctl(results[name]['times'], 0.50),
                'p95': p_pctl(results[name]['times'], 0.95),
                'max': max(results[name]['times']),
            }
            for name in results
        },
    }
    out = REPORTS / 'test500_stage2_report.json'
    out.write_text(json.dumps(rep, indent=2))
    print(f'\n▶ Report: {out}')


if __name__ == '__main__':
    main()
