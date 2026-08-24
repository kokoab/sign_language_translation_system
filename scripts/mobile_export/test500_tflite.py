"""500-sample TFLite test (runs in the atlas_tflite venv).

Loads the same 500 test-split samples and runs:
  - Stage 1 (Stage1.tflite) vs PyTorch baseline
  - Stage 2 split pipeline (Stage2A + Stage2B_S4) vs PyTorch baseline

Compares to PyTorch in the same venv (torch 2.9.1 here, which is close enough to
2.10.0 for inference parity — the model weights are the same tensors).
"""
from __future__ import annotations
import os, sys, json, time, random, warnings, platform
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

REPO = Path('/Users/frnzlo/Documents/machine_learning/SLT')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src_v16'))
sys.path.insert(0, str(REPO / 'mobile_export' / 'scripts'))

from src_v16.model_v16 import SLTStage1V16, SLTStage2V16CTC  # noqa
from tflite_convert import Stage2Wrapper, Stage2A, Stage2B, ManualMHA, swap_mha_inplace, load_stage1, load_stage2  # noqa

warnings.filterwarnings('ignore')

SEED = 42
N_SAMPLES = 500
DATA_DIR = REPO / 'src_v16' / 'ASL_landmarks_v16'
MANIFEST = REPO / 'src_v16' / 'manifest_v16_files_deep_cleaned.json'
ARTIFACTS = REPO / 'mobile_export' / 'artifacts'
REPORTS = REPO / 'mobile_export' / 'reports'
TFLITE_DIR = ARTIFACTS / 'tflite'


def ctc_greedy_decode(logits, blank=0):
    if logits.ndim == 3: logits = logits[0]
    path = logits.argmax(axis=-1)
    out = []; prev = -1
    for p in path:
        p = int(p)
        if p != prev and p != blank: out.append(p)
        prev = p
    return out


def reproduce_test_split(manifest):
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


def p_pctl(v, p):
    s = sorted(v); return s[min(len(s)-1, int(len(s)*p))]


def main():
    torch.set_grad_enabled(False)
    print(f'▶ Platform: {platform.platform()}')
    print(f'  torch {torch.__version__}')

    from ai_edge_litert.interpreter import Interpreter as LiteInterpreter

    manifest = json.loads(MANIFEST.read_text())
    test = reproduce_test_split(manifest)
    rng = random.Random(SEED)
    sample = rng.sample(test, N_SAMPLES)

    # ── Build label mapping from Stage 1 checkpoint ──
    m1, ck1 = load_stage1()
    idx_to_label = {int(k): v for k, v in (ck1.get('idx_to_label') or {}).items()}
    label_to_idx = {v: int(k) for k, v in idx_to_label.items()}

    m2_raw, ck2 = load_stage2()
    idx_to_gloss = {int(k): v for k, v in (ck2.get('idx_to_gloss') or {}).items()}
    gloss_to_idx = {v: int(k) for k, v in idx_to_gloss.items()}
    m2 = Stage2Wrapper(m2_raw).eval()

    # MHA-swapped PyTorch (this is what TFLite was converted from — should be our baseline)
    import copy
    m1_pt = copy.deepcopy(m1)
    swap_mha_inplace(m1_pt); m1_pt.eval()

    a_mod = Stage2A(m2_raw)
    b_mod = Stage2B(m2_raw)
    swap_mha_inplace(a_mod); a_mod.eval()
    swap_mha_inplace(b_mod); b_mod.eval()

    # ── Preload features ──
    print(f'▶ Preloading {N_SAMPLES} .npy files...')
    inputs = []
    for fname, cls in sample:
        arr = np.load(DATA_DIR / fname).astype(np.float32)
        y = label_to_idx.get(cls)
        yg = gloss_to_idx.get(cls)
        if y is None or yg is None: continue
        inputs.append({'fname': fname, 'cls': cls, 'y_true': y, 'y_true_gloss': yg, 'x': arr})
    N = len(inputs); print(f'  usable: {N}')

    # ─────────────────────────────────────────────────
    # STAGE 1
    # ─────────────────────────────────────────────────
    print('\n' + '=' * 64)
    print('STAGE 1 — PyTorch vs TFLite')
    print('=' * 64)

    # PyTorch reference (post-MHA-swap, same module that was converted)
    print('▶ PyTorch inference...')
    pt_logits, pt_preds, pt_times = [], [], []
    t0 = time.perf_counter()
    for i, r in enumerate(inputs):
        x = torch.from_numpy(r['x']).unsqueeze(0)
        ts = time.perf_counter()
        logits = m1_pt(x).cpu().numpy()[0]
        pt_times.append((time.perf_counter() - ts) * 1000)
        pt_logits.append(logits); pt_preds.append(int(logits.argmax()))
        if (i+1) % 100 == 0: print(f'  [{i+1}/{N}] {time.perf_counter()-t0:.1f}s')

    # TFLite
    print('▶ TFLite inference...')
    interp = LiteInterpreter(model_path=str(TFLITE_DIR / 'Stage1.tflite'))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]; out_det = interp.get_output_details()[0]
    tf_logits, tf_preds, tf_times = [], [], []
    t0 = time.perf_counter()
    for i, r in enumerate(inputs):
        x = r['x'][None, ...].astype(np.float32)
        ts = time.perf_counter()
        interp.set_tensor(in_det['index'], x)
        interp.invoke()
        logits = interp.get_tensor(out_det['index'])[0]
        tf_times.append((time.perf_counter() - ts) * 1000)
        tf_logits.append(logits); tf_preds.append(int(logits.argmax()))
        if (i+1) % 100 == 0: print(f'  [{i+1}/{N}] {time.perf_counter()-t0:.1f}s')

    def top1(preds):
        return sum(1 for p, r in zip(preds, inputs) if p == r['y_true']) / N * 100

    def agree(a, b):
        return sum(1 for x, y in zip(a, b) if x == y) / N * 100

    acc_pt = top1(pt_preds); acc_tf = top1(tf_preds)
    diff = [float(np.abs(p - t).max()) for p, t in zip(pt_logits, tf_logits)]

    print(f'\n{"Format":<10}  {"Top-1":>8}  {"Avg ms":>8}  {"P95":>8}')
    print('-' * 48)
    print(f'{"PyTorch":<10}  {acc_pt:7.2f}%  {np.mean(pt_times):7.2f}  {p_pctl(pt_times, 0.95):7.2f}')
    print(f'{"TFLite":<10}  {acc_tf:7.2f}%  {np.mean(tf_times):7.2f}  {p_pctl(tf_times, 0.95):7.2f}')
    print(f'\nPyTorch ↔ TFLite pred agreement: {agree(pt_preds, tf_preds):.2f}%')
    print(f'Logit max|Δ|: mean={np.mean(diff):.2e}  max={max(diff):.2e}')

    stage1_rep = {
        'top1_pytorch': acc_pt, 'top1_tflite': acc_tf,
        'agreement_pt_vs_tflite': agree(pt_preds, tf_preds),
        'logit_max_abs_diff': {'mean': float(np.mean(diff)), 'max': max(diff)},
        'latency_ms': {
            'pytorch': {'mean': float(np.mean(pt_times)), 'p50': p_pctl(pt_times, 0.5),
                        'p95': p_pctl(pt_times, 0.95), 'max': max(pt_times)},
            'tflite': {'mean': float(np.mean(tf_times)), 'p50': p_pctl(tf_times, 0.5),
                       'p95': p_pctl(tf_times, 0.95), 'max': max(tf_times)},
        },
        'n': N,
    }

    # ─────────────────────────────────────────────────
    # STAGE 2 SPLIT
    # ─────────────────────────────────────────────────
    print('\n' + '=' * 64)
    print('STAGE 2 — PyTorch split vs TFLite split')
    print('=' * 64)

    # PyTorch reference (post-MHA-swap)
    print('▶ PyTorch split inference...')
    pt_s2_logits, pt_s2_dec, pt_s2_times = [], [], []
    t0 = time.perf_counter()
    for i, r in enumerate(inputs):
        x = torch.from_numpy(r['x']).unsqueeze(0)
        ts = time.perf_counter()
        tokens = a_mod(x)             # [1, 4, 384]
        logits = b_mod(tokens).cpu().numpy()  # [1, 4, 311]
        pt_s2_times.append((time.perf_counter() - ts) * 1000)
        pt_s2_logits.append(logits); pt_s2_dec.append(ctc_greedy_decode(logits, 0))
        if (i+1) % 100 == 0: print(f'  [{i+1}/{N}] {time.perf_counter()-t0:.1f}s')

    # TFLite split
    print('▶ TFLite split inference (A + B_S4)...')
    interpA = LiteInterpreter(model_path=str(TFLITE_DIR / 'Stage2A_ClipEncoder.tflite'))
    interpA.allocate_tensors()
    A_in = interpA.get_input_details()[0]; A_out = interpA.get_output_details()[0]
    interpB = LiteInterpreter(model_path=str(TFLITE_DIR / 'Stage2B_SeqCTC_S4.tflite'))
    interpB.allocate_tensors()
    B_in = interpB.get_input_details()[0]; B_out = interpB.get_output_details()[0]
    tf_s2_logits, tf_s2_dec, tf_s2_times = [], [], []
    t0 = time.perf_counter()
    for i, r in enumerate(inputs):
        x = r['x'][None, ...].astype(np.float32)
        ts = time.perf_counter()
        interpA.set_tensor(A_in['index'], x); interpA.invoke()
        tokens = interpA.get_tensor(A_out['index']).astype(np.float32)  # [1, 4, 384]
        interpB.set_tensor(B_in['index'], tokens); interpB.invoke()
        logits = interpB.get_tensor(B_out['index'])
        tf_s2_times.append((time.perf_counter() - ts) * 1000)
        tf_s2_logits.append(logits); tf_s2_dec.append(ctc_greedy_decode(logits, 0))
        if (i+1) % 100 == 0: print(f'  [{i+1}/{N}] {time.perf_counter()-t0:.1f}s')

    def ft_acc(dec_list):
        return sum(1 for r, d in zip(inputs, dec_list) if d and d[0] == r['y_true_gloss']) / N * 100

    def dec_match(a, b):
        return sum(1 for x, y in zip(a, b) if x == y) / N * 100

    ft_pt = ft_acc(pt_s2_dec); ft_tf = ft_acc(tf_s2_dec)
    dm = dec_match(pt_s2_dec, tf_s2_dec)
    diff2 = [float(np.abs(p - t).max()) for p, t in zip(pt_s2_logits, tf_s2_logits)]

    print(f'\n{"Format":<15}  {"1st-tok":>10}  {"Avg ms":>8}  {"P95":>8}')
    print('-' * 48)
    print(f'{"PyTorch split":<15}  {ft_pt:9.2f}%  {np.mean(pt_s2_times):7.2f}  {p_pctl(pt_s2_times, 0.95):7.2f}')
    print(f'{"TFLite split":<15}  {ft_tf:9.2f}%  {np.mean(tf_s2_times):7.2f}  {p_pctl(tf_s2_times, 0.95):7.2f}')
    print(f'\nDecode-sequence match rate (PT vs TFLite): {dm:.2f}%')
    print(f'Logit max|Δ|: mean={np.mean(diff2):.2e}  max={max(diff2):.2e}')

    stage2_rep = {
        'first_token_acc_pytorch': ft_pt, 'first_token_acc_tflite': ft_tf,
        'decode_match_rate': dm,
        'logit_max_abs_diff': {'mean': float(np.mean(diff2)), 'max': max(diff2)},
        'latency_ms_split_pipeline': {
            'pytorch': {'mean': float(np.mean(pt_s2_times)), 'p50': p_pctl(pt_s2_times, 0.5),
                        'p95': p_pctl(pt_s2_times, 0.95), 'max': max(pt_s2_times)},
            'tflite': {'mean': float(np.mean(tf_s2_times)), 'p50': p_pctl(tf_s2_times, 0.5),
                       'p95': p_pctl(tf_s2_times, 0.95), 'max': max(tf_s2_times)},
        },
        'n': N,
    }

    report = {
        'platform': platform.platform(),
        'torch': torch.__version__,
        'ai_edge_litert_version': 'via ai_edge_litert.interpreter',
        'stage1': stage1_rep,
        'stage2_split': stage2_rep,
    }
    (REPORTS / 'test500_tflite_report.json').write_text(json.dumps(report, indent=2))
    print(f'\n▶ Report: {REPORTS / "test500_tflite_report.json"}')


if __name__ == '__main__':
    main()
