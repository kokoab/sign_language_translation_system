"""500-sample Stage 1 test: PyTorch vs ONNX vs CoreML on the real test split.

Reproduces the training-time 70/15/15 split (seed=42 per class) to pull 500
stratified samples from the test portion, runs each format, and reports:
  - per-format top-1 accuracy
  - pairwise agreement rates
  - numerical divergence stats
  - latency distribution
"""
from __future__ import annotations
import os, sys, json, time, random, warnings, platform
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import load_stage1, ARTIFACTS, REPORTS, load_idx_to_gloss

warnings.filterwarnings('ignore')


SEED = 42
N_SAMPLES = 500
DATA_DIR = Path('/Users/frnzlo/Documents/machine_learning/SLT/src_v16/ASL_landmarks_v16')
MANIFEST = Path('/Users/frnzlo/Documents/machine_learning/SLT/src_v16/manifest_v16_files_deep_cleaned.json')


def reproduce_test_split(manifest: dict[str, str]) -> list[tuple[str, str]]:
    """Reproduce train_stage_1_v16's test split exactly.

    Returns list of (filename, class_label) tuples from the test split.
    """
    files_by_class: dict[str, list[str]] = defaultdict(list)
    for fname, cls in manifest.items():
        files_by_class[cls].append(fname)

    test_files: list[tuple[str, str]] = []
    for cls in sorted(files_by_class.keys()):
        cls_files = sorted(files_by_class[cls])
        rng = random.Random()
        rng.seed(SEED)
        rng.shuffle(cls_files)
        n = len(cls_files)
        val_end = int(n * 0.85)
        for fp in cls_files[val_end:]:
            test_files.append((fp, cls))
    return test_files


def load_features(fname: str) -> np.ndarray:
    """Load .npy and normalize to [32, 61, 5] float32 (no vel/acc)."""
    arr = np.load(DATA_DIR / fname).astype(np.float32)
    T, N, C = arr.shape
    assert (T, N, C) == (32, 61, 5), f'unexpected shape {arr.shape} for {fname}'
    return arr


def p_pctl(vals, p):
    if not vals: return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, int(len(s) * p))
    return s[idx]


def main():
    torch.set_grad_enabled(False)

    print(f'▶ Loading manifest: {MANIFEST}')
    manifest = json.loads(MANIFEST.read_text())
    print(f'  entries: {len(manifest)}')
    test_files = reproduce_test_split(manifest)
    print(f'  test split size: {len(test_files)}')

    # Sample stratified 500
    rng = random.Random(SEED)
    sample = rng.sample(test_files, N_SAMPLES)
    print(f'  sampled: {len(sample)} files for 500-sample test')
    class_hist = Counter(c for _, c in sample)
    print(f'  classes represented: {len(class_hist)} / 310')
    print(f'  samples per class: min={min(class_hist.values())} max={max(class_hist.values())} mean={len(sample)/len(class_hist):.2f}')

    # ── Load models ──
    print('\n▶ Loading models...')
    pt_model, meta = load_stage1()
    idx_to_label = load_idx_to_gloss(meta)
    label_to_idx = {v: int(k) for k, v in idx_to_label.items()}

    import onnxruntime as ort
    ort_sess = ort.InferenceSession(str(ARTIFACTS / 'stage1.onnx'),
                                    providers=['CPUExecutionProvider'])
    ort_in = ort_sess.get_inputs()[0].name
    ort_out = ort_sess.get_outputs()[0].name

    import coremltools as ct
    cm_model = ct.models.MLModel(str(ARTIFACTS / 'coreml' / 'Stage1.mlpackage'),
                                 compute_units=ct.ComputeUnit.CPU_AND_NE)
    cm_spec = cm_model.get_spec()
    cm_in = cm_spec.description.input[0].name
    cm_out = cm_spec.description.output[0].name

    # ── Pre-load features ──
    print(f'\n▶ Pre-loading {N_SAMPLES} .npy files...')
    inputs = []
    for fname, cls in sample:
        arr = load_features(fname)
        inputs.append({'fname': fname, 'cls': cls, 'y_true': label_to_idx[cls], 'x': arr})

    # ── Run PyTorch ──
    print('\n▶ Running PyTorch (fp32 CPU)...')
    pt_preds, pt_top5, pt_logits, pt_times = [], [], [], []
    t0 = time.perf_counter()
    for i, r in enumerate(inputs):
        x = torch.from_numpy(r['x']).unsqueeze(0)
        ts = time.perf_counter()
        logits = pt_model(x).cpu().numpy()[0]
        pt_times.append((time.perf_counter() - ts) * 1000)
        pt_logits.append(logits)
        pt_preds.append(int(logits.argmax()))
        pt_top5.append(logits.argsort()[-5:][::-1].tolist())
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{N_SAMPLES}]  elapsed {time.perf_counter() - t0:.1f}s')
    pt_elapsed = time.perf_counter() - t0

    # ── Run ONNX ──
    print('\n▶ Running ONNX Runtime (CPU)...')
    ort_preds, ort_top5, ort_logits, ort_times = [], [], [], []
    t0 = time.perf_counter()
    for i, r in enumerate(inputs):
        x = r['x'][None, ...]
        ts = time.perf_counter()
        logits = ort_sess.run([ort_out], {ort_in: x})[0][0]
        ort_times.append((time.perf_counter() - ts) * 1000)
        ort_logits.append(logits)
        ort_preds.append(int(logits.argmax()))
        ort_top5.append(logits.argsort()[-5:][::-1].tolist())
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{N_SAMPLES}]  elapsed {time.perf_counter() - t0:.1f}s')
    ort_elapsed = time.perf_counter() - t0

    # ── Run CoreML ──
    print('\n▶ Running CoreML (FP16, CPU+NE)...')
    cm_preds, cm_top5, cm_logits, cm_times = [], [], [], []
    t0 = time.perf_counter()
    for i, r in enumerate(inputs):
        x = r['x'][None, ...]
        ts = time.perf_counter()
        out = cm_model.predict({cm_in: x})
        cm_times.append((time.perf_counter() - ts) * 1000)
        logits = np.array(out[cm_out]).reshape(310)
        cm_logits.append(logits)
        cm_preds.append(int(logits.argmax()))
        cm_top5.append(logits.argsort()[-5:][::-1].tolist())
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{N_SAMPLES}]  elapsed {time.perf_counter() - t0:.1f}s')
    cm_elapsed = time.perf_counter() - t0

    # ── Metrics ──
    print('\n' + '=' * 64)
    print('RESULTS — 500-SAMPLE STAGE 1 COMPARISON')
    print('=' * 64)
    y_true = [r['y_true'] for r in inputs]

    def top1(preds):
        return sum(1 for p, y in zip(preds, y_true) if p == y) / len(y_true) * 100
    def top5(top5s):
        return sum(1 for t5, y in zip(top5s, y_true) if y in t5) / len(y_true) * 100
    def agree(a, b):
        return sum(1 for x, y in zip(a, b) if x == y) / len(a) * 100

    acc_pt = top1(pt_preds); acc_ort = top1(ort_preds); acc_cm = top1(cm_preds)
    t5_pt = top5(pt_top5); t5_ort = top5(ort_top5); t5_cm = top5(cm_top5)

    diff_ort = [float(np.abs(p - o).max()) for p, o in zip(pt_logits, ort_logits)]
    diff_cm  = [float(np.abs(p - c).max()) for p, c in zip(pt_logits, cm_logits)]

    print(f'\n{"Format":<12}  {"Top-1":>8}  {"Top-5":>8}  {"Avg ms":>8}  {"P95 ms":>8}  {"Max ms":>8}')
    print('-' * 64)
    print(f'{"PyTorch":<12}  {acc_pt:7.2f}%  {t5_pt:7.2f}%  {np.mean(pt_times):7.2f}  '
          f'{p_pctl(pt_times, 0.95):7.2f}  {max(pt_times):7.2f}')
    print(f'{"ONNX Runtime":<12}  {acc_ort:7.2f}%  {t5_ort:7.2f}%  {np.mean(ort_times):7.2f}  '
          f'{p_pctl(ort_times, 0.95):7.2f}  {max(ort_times):7.2f}')
    print(f'{"CoreML":<12}  {acc_cm:7.2f}%  {t5_cm:7.2f}%  {np.mean(cm_times):7.2f}  '
          f'{p_pctl(cm_times, 0.95):7.2f}  {max(cm_times):7.2f}')

    print(f'\nPairwise top-1 agreement:')
    print(f'  PyTorch ↔ ONNX:   {agree(pt_preds, ort_preds):.2f}%')
    print(f'  PyTorch ↔ CoreML: {agree(pt_preds, cm_preds):.2f}%')
    print(f'  ONNX ↔ CoreML:    {agree(ort_preds, cm_preds):.2f}%')

    print(f'\nLogit divergence (max |PyTorch - other| per sample):')
    print(f'  PyTorch vs ONNX:   mean={np.mean(diff_ort):.2e}  max={max(diff_ort):.2e}')
    print(f'  PyTorch vs CoreML: mean={np.mean(diff_cm):.2e}  max={max(diff_cm):.2e}')

    # Per-class worst performers
    per_class = defaultdict(list)
    for r, pt_p in zip(inputs, pt_preds):
        per_class[r['cls']].append(1 if pt_p == r['y_true'] else 0)
    wrong = [(cls, int(sum(1 for v in acc if v == 0)), len(acc))
             for cls, acc in per_class.items() if any(v == 0 for v in acc)]
    wrong.sort(key=lambda x: -x[1])
    if wrong:
        print(f'\nClasses with PyTorch errors (top 10):')
        for cls, n_wrong, n_total in wrong[:10]:
            print(f'  {cls:<15s}  {n_wrong}/{n_total}')

    # Find samples where formats disagree
    disagreements = []
    for i in range(N_SAMPLES):
        preds_set = {pt_preds[i], ort_preds[i], cm_preds[i]}
        if len(preds_set) > 1:
            disagreements.append({
                'fname': inputs[i]['fname'],
                'true': inputs[i]['cls'],
                'pt': idx_to_label[pt_preds[i]],
                'ort': idx_to_label[ort_preds[i]],
                'cm': idx_to_label[cm_preds[i]],
            })
    print(f'\nSamples where all 3 formats disagreed: {len(disagreements)}')
    if disagreements:
        print('  First 5:')
        for d in disagreements[:5]:
            print(f'  {d["fname"]}  true={d["true"]}  pt={d["pt"]} ort={d["ort"]} cm={d["cm"]}')

    # ── Save report ──
    report = {
        'platform': platform.platform(),
        'torch': torch.__version__,
        'onnxruntime': ort.__version__,
        'coremltools': ct.__version__,
        'n_samples': N_SAMPLES,
        'seed': SEED,
        'classes_represented': len(class_hist),
        'checkpoint_val_acc': meta['val_acc'],
        'accuracy': {
            'pytorch': {'top1': acc_pt, 'top5': t5_pt},
            'onnx':    {'top1': acc_ort, 'top5': t5_ort},
            'coreml':  {'top1': acc_cm, 'top5': t5_cm},
        },
        'agreement': {
            'pt_ort': agree(pt_preds, ort_preds),
            'pt_cm':  agree(pt_preds, cm_preds),
            'ort_cm': agree(ort_preds, cm_preds),
        },
        'logit_divergence_max_abs': {
            'pt_vs_ort':  {'mean': float(np.mean(diff_ort)), 'max': max(diff_ort)},
            'pt_vs_cm':   {'mean': float(np.mean(diff_cm)),  'max': max(diff_cm)},
        },
        'latency_ms': {
            'pytorch': {'mean': float(np.mean(pt_times)), 'p50': p_pctl(pt_times, 0.50),
                        'p95': p_pctl(pt_times, 0.95), 'max': max(pt_times)},
            'onnx':    {'mean': float(np.mean(ort_times)), 'p50': p_pctl(ort_times, 0.50),
                        'p95': p_pctl(ort_times, 0.95), 'max': max(ort_times)},
            'coreml':  {'mean': float(np.mean(cm_times)), 'p50': p_pctl(cm_times, 0.50),
                        'p95': p_pctl(cm_times, 0.95), 'max': max(cm_times)},
        },
        'total_walltime_s': {
            'pytorch': pt_elapsed, 'onnx': ort_elapsed, 'coreml': cm_elapsed,
        },
        'three_way_disagreements': len(disagreements),
        'disagreement_samples': disagreements[:20],
    }
    out = REPORTS / 'test500_stage1_report.json'
    out.write_text(json.dumps(report, indent=2))
    print(f'\n▶ Report written: {out}')


if __name__ == '__main__':
    main()
