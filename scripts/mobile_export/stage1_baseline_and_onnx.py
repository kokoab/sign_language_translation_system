"""Stage 1: PyTorch baseline → ONNX export → ONNX Runtime validation.

Writes:
  artifacts/stage1_baseline.json   (PyTorch predictions on 10 test videos)
  artifacts/stage1.onnx            (exported model, opset 17)
  artifacts/stage1_onnx_report.json (numerical parity report)
"""
from __future__ import annotations
import os, sys, json, time, warnings
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    load_stage1, ARTIFACTS, REPO, load_idx_to_gloss,
)

warnings.filterwarnings('ignore', category=UserWarning)


def main():
    torch.set_grad_enabled(False)

    print('▶ Loading Stage 1 model...')
    model, meta = load_stage1()
    idx_to_label = load_idx_to_gloss(meta)
    print(f'  num_classes={meta["num_classes"]}  d_model={meta["d_model"]}  in_ch={meta["in_channels"]}')
    print(f'  val_acc (checkpoint) = {meta["val_acc"]:.2f}%')

    # ── 1) PyTorch baseline on test videos ──
    feat_dir = ARTIFACTS / 'test_features'
    isolated = sorted(feat_dir.glob('*.npy'))
    isolated = [p for p in isolated if not p.name.startswith('PHRASE__')]
    print(f'\n▶ PyTorch baseline on {len(isolated)} isolated test videos:')

    pt_results = []
    all_pt_logits = []
    for p in isolated:
        cls = p.stem.split('__')[0]
        arr = np.load(p).astype(np.float32)  # [32, 61, 5]
        x = torch.from_numpy(arr).unsqueeze(0)  # [1, 32, 61, 5]
        t0 = time.perf_counter()
        logits = model(x).cpu().numpy()  # [1, 310]
        ms = (time.perf_counter() - t0) * 1000
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()[0]
        top5_idx = probs.argsort()[-5:][::-1].tolist()
        top5 = [(int(i), idx_to_label.get(int(i), str(i)), float(probs[i])) for i in top5_idx]
        correct = top5[0][1].upper() == cls.upper()
        print(f'  {cls:12s} → top1={top5[0][1]:12s} p={top5[0][2]:.3f}  {"OK" if correct else "MISS"}  ({ms:.1f}ms)')
        pt_results.append({
            'class': cls, 'npy': str(p), 'top5': top5, 'correct': correct, 'ms': ms,
            'logits_mean': float(logits.mean()), 'logits_std': float(logits.std()),
        })
        all_pt_logits.append(logits[0])

    acc_pt = sum(1 for r in pt_results if r['correct']) / len(pt_results) * 100
    print(f'  PyTorch top-1 on small test set: {acc_pt:.1f}% ({sum(1 for r in pt_results if r["correct"])}/{len(pt_results)})')

    (ARTIFACTS / 'stage1_baseline.json').write_text(json.dumps({
        'checkpoint_val_acc': meta['val_acc'],
        'small_test_top1': acc_pt,
        'results': pt_results,
    }, indent=2))

    # ── 2) ONNX export ──
    print('\n▶ Exporting Stage 1 to ONNX...')
    onnx_path = ARTIFACTS / 'stage1.onnx'
    dummy = torch.randn(1, 32, 61, 5, dtype=torch.float32)

    # Try multiple export strategies so we don't die on torch 2.10 dynamo quirks.
    export_ok = False
    last_err = None
    export_attempts = [
        dict(dynamo=False, opset_version=17, label='legacy-ts opset17'),
        dict(dynamo=False, opset_version=18, label='legacy-ts opset18'),
        dict(dynamo=True,  opset_version=18, label='dynamo opset18'),
    ]
    for attempt in export_attempts:
        try:
            kw = dict(
                input_names=['landmarks'],
                output_names=['logits'],
                dynamic_axes={'landmarks': {0: 'batch'}, 'logits': {0: 'batch'}},
                opset_version=attempt['opset_version'],
                do_constant_folding=True,
            )
            if attempt['dynamo'] is not None:
                kw['dynamo'] = attempt['dynamo']
            torch.onnx.export(model, (dummy,), str(onnx_path), **kw)
            print(f'  exported via {attempt["label"]}: {onnx_path}  ({onnx_path.stat().st_size/1e6:.1f} MB)')
            export_ok = True
            break
        except Exception as e:
            last_err = e
            print(f'  [attempt {attempt["label"]} failed] {type(e).__name__}: {str(e)[:160]}')
    if not export_ok:
        raise RuntimeError(f'All export attempts failed. Last error: {last_err}')

    # Validate the ONNX graph
    onnx_model = onnx.load(str(onnx_path))
    try:
        onnx.checker.check_model(onnx_model)
        print('  onnx.checker: PASSED')
    except Exception as e:
        print(f'  onnx.checker: FAILED — {e}')

    # ── 3) ONNX Runtime validation ──
    print('\n▶ ONNX Runtime validation:')
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    parity = []
    for p, pt_logit in zip(isolated, all_pt_logits):
        cls = p.stem.split('__')[0]
        arr = np.load(p).astype(np.float32)[None, ...]
        t0 = time.perf_counter()
        ort_logit = sess.run([output_name], {input_name: arr})[0][0]
        ms = (time.perf_counter() - t0) * 1000
        pt_top1 = int(pt_logit.argmax())
        ort_top1 = int(ort_logit.argmax())
        max_abs_diff = float(np.abs(pt_logit - ort_logit).max())
        mean_abs_diff = float(np.abs(pt_logit - ort_logit).mean())
        cos = float(
            (pt_logit @ ort_logit) / (np.linalg.norm(pt_logit) * np.linalg.norm(ort_logit) + 1e-12)
        )
        same_top1 = pt_top1 == ort_top1
        print(f'  {cls:12s}  max|Δ|={max_abs_diff:.2e}  mean|Δ|={mean_abs_diff:.2e}  cos={cos:.6f}  '
              f'top1-match={"YES" if same_top1 else "NO"}  ({ms:.1f}ms)')
        parity.append({
            'class': cls, 'max_abs_diff': max_abs_diff, 'mean_abs_diff': mean_abs_diff,
            'cosine_similarity': cos, 'pt_top1': pt_top1, 'onnx_top1': ort_top1,
            'same_top1': same_top1, 'ort_ms': ms,
        })

    summary = {
        'onnx_path': str(onnx_path),
        'onnx_size_mb': onnx_path.stat().st_size / 1e6,
        'num_samples': len(parity),
        'top1_match_rate': sum(1 for r in parity if r['same_top1']) / len(parity),
        'max_abs_diff_global': max(r['max_abs_diff'] for r in parity),
        'mean_abs_diff_global': float(np.mean([r['mean_abs_diff'] for r in parity])),
        'min_cosine': min(r['cosine_similarity'] for r in parity),
        'mean_onnx_ms': float(np.mean([r['ort_ms'] for r in parity])),
        'details': parity,
    }
    (ARTIFACTS / 'stage1_onnx_report.json').write_text(json.dumps(summary, indent=2))

    print(f'\n  top1-match rate: {summary["top1_match_rate"]*100:.1f}%')
    print(f'  max|Δ| across all samples: {summary["max_abs_diff_global"]:.2e}')
    print(f'  min cosine similarity:    {summary["min_cosine"]:.6f}')


if __name__ == '__main__':
    main()
