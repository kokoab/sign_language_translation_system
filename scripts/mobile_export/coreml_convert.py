"""Convert Stage 1 + Stage 2 to CoreML (.mlpackage) and validate on test videos.

Tries converting from PyTorch traced module first (coremltools recommended path).
Falls back to loading TorchScript if tracing fails.

Environment: coremltools 9.0, torch 2.10 (unsupported combo per ct warning — expect surprises).
"""
from __future__ import annotations
import os, sys, json, time, warnings, traceback
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from _common import load_stage1, load_stage2, ARTIFACTS, ctc_greedy_decode, load_idx_to_gloss

warnings.filterwarnings('ignore')

import coremltools as ct
print(f'coremltools {ct.__version__}, torch {torch.__version__}')

results = {}


# ═══ Helper ═══

class Stage2Wrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        logits, _ = self.m(x)
        return logits


def try_convert(model, sample_inputs, label: str, out_path: Path,
                input_shapes, compute_precision='FLOAT16',
                minimum_deployment_target=None):
    """Try tracing + CoreML conversion. Returns (ok, report)."""
    rep = {'label': label, 'out_path': str(out_path), 'precision': compute_precision}
    try:
        traced = torch.jit.trace(model, sample_inputs, strict=False)
        traced.eval()
        rep['trace_ok'] = True
    except Exception as e:
        rep['trace_ok'] = False
        rep['trace_error'] = f'{type(e).__name__}: {str(e)[:300]}'
        return False, rep

    try:
        ct_inputs = [ct.TensorType(name=n, shape=s, dtype=np.float32) for n, s in input_shapes]
        kw = dict(
            inputs=ct_inputs,
            convert_to='mlprogram',
            compute_precision=(ct.precision.FLOAT16 if compute_precision == 'FLOAT16' else ct.precision.FLOAT32),
        )
        if minimum_deployment_target is not None:
            kw['minimum_deployment_target'] = minimum_deployment_target
        mlmodel = ct.convert(traced, **kw)
        mlmodel.save(str(out_path))
        rep['convert_ok'] = True
        rep['mlpackage_size_mb'] = sum(f.stat().st_size for f in Path(out_path).rglob('*')) / 1e6
        return True, rep
    except Exception as e:
        rep['convert_ok'] = False
        rep['convert_error'] = f'{type(e).__name__}: {str(e)[:500]}'
        rep['traceback'] = traceback.format_exc()[-1500:]
        return False, rep


def validate_coreml(mlpath: Path, pt_model, samples, label, idx_to_name, is_stage2=False):
    """Validate a .mlpackage by running on test features + comparing to PyTorch."""
    print(f'\n▶ Validating CoreML {label}: {mlpath}')
    try:
        mlmodel = ct.models.MLModel(str(mlpath), compute_units=ct.ComputeUnit.CPU_AND_NE)
    except Exception as e:
        return {'load_error': f'{type(e).__name__}: {str(e)[:200]}'}

    # Inspect expected input names
    spec = mlmodel.get_spec()
    in_desc = list(spec.description.input)
    print(f'  inputs: {[(i.name, list(i.type.multiArrayType.shape)) for i in in_desc]}')
    input_name = in_desc[0].name
    output_name = spec.description.output[0].name

    parity = []
    for sample in samples:
        arr = sample['arr']  # [1, T, 61, 5] float32
        pt_logit = sample['pt_logit']
        label_str = sample['label']
        try:
            t0 = time.perf_counter()
            out = mlmodel.predict({input_name: arr})
            ms = (time.perf_counter() - t0) * 1000
            cm_logit = np.array(out[output_name])
        except Exception as e:
            print(f'  [{label_str}] predict failed: {type(e).__name__}: {str(e)[:200]}')
            parity.append({'label': label_str, 'error': f'{type(e).__name__}: {str(e)[:200]}'})
            continue

        if cm_logit.shape != pt_logit.shape:
            # CoreML sometimes adds/removes leading dim
            if cm_logit.squeeze().shape == pt_logit.squeeze().shape:
                cm_logit = cm_logit.reshape(pt_logit.shape)
            else:
                parity.append({'label': label_str, 'error': 'shape_mismatch',
                               'pt_shape': list(pt_logit.shape), 'cm_shape': list(cm_logit.shape)})
                continue

        max_abs = float(np.abs(pt_logit - cm_logit).max())
        mean_abs = float(np.abs(pt_logit - cm_logit).mean())

        if is_stage2:
            pt_dec = ctc_greedy_decode(pt_logit, blank=0)
            cm_dec = ctc_greedy_decode(cm_logit, blank=0)
            match = pt_dec == cm_dec
            pt_g = [idx_to_name.get(i, f'<{i}>') for i in pt_dec]
            cm_g = [idx_to_name.get(i, f'<{i}>') for i in cm_dec]
            print(f'  [{label_str:25s}] max|Δ|={max_abs:.2e} mean|Δ|={mean_abs:.2e}  '
                  f'{"MATCH" if match else "DIFF"}  ({ms:.0f}ms)')
            if not match:
                print(f'      pt : {pt_g}')
                print(f'      cm : {cm_g}')
            parity.append({'label': label_str, 'max_abs': max_abs, 'mean_abs': mean_abs,
                           'match': match, 'pt_decoded': pt_g, 'cm_decoded': cm_g, 'ms': ms})
        else:
            pt_top1 = int(pt_logit.argmax())
            cm_top1 = int(cm_logit.argmax())
            match = pt_top1 == cm_top1
            print(f'  [{label_str:12s}] max|Δ|={max_abs:.2e} mean|Δ|={mean_abs:.2e} '
                  f'pt={idx_to_name.get(pt_top1):12s} cm={idx_to_name.get(cm_top1):12s} '
                  f'{"MATCH" if match else "DIFF"}  ({ms:.0f}ms)')
            parity.append({'label': label_str, 'max_abs': max_abs, 'mean_abs': mean_abs,
                           'pt_top1': pt_top1, 'cm_top1': cm_top1, 'match': match, 'ms': ms})

    return {'parity': parity,
            'mean_ms': float(np.mean([p['ms'] for p in parity if 'ms' in p])) if parity else None,
            'match_rate': sum(1 for p in parity if p.get('match')) / max(1, len(parity))}


def main():
    torch.set_grad_enabled(False)
    coreml_dir = ARTIFACTS / 'coreml'
    coreml_dir.mkdir(exist_ok=True)

    # ═══ STAGE 1 ═══
    print('='*72)
    print('STAGE 1: Squeezeformer d=384, 5ch → CoreML .mlpackage')
    print('='*72)
    model1, meta1 = load_stage1()
    i2l = load_idx_to_gloss(meta1)
    dummy1 = torch.randn(1, 32, 61, 5)

    # Build test sample set (same 10 as PyTorch baseline)
    feat_dir = ARTIFACTS / 'test_features'
    samples1 = []
    for p in sorted(feat_dir.glob('*.npy')):
        if p.name.startswith('PHRASE__'):
            continue
        label = p.stem.split('__')[0]
        arr = np.load(p).astype(np.float32)[None, ...]  # [1, 32, 61, 5]
        with torch.no_grad():
            pt_logit = model1(torch.from_numpy(arr)).cpu().numpy()
        samples1.append({'label': label, 'arr': arr, 'pt_logit': pt_logit})

    s1_path = coreml_dir / 'Stage1.mlpackage'
    ok1, rep1 = try_convert(
        model1, dummy1, 'Stage 1 (fixed [1,32,61,5] FP16)', s1_path,
        input_shapes=[('landmarks', (1, 32, 61, 5))],
        compute_precision='FLOAT16',
        minimum_deployment_target=ct.target.iOS15,
    )
    results['stage1'] = rep1
    if ok1:
        print(f'✓ Stage 1 .mlpackage saved ({rep1["mlpackage_size_mb"]:.1f} MB)')
        results['stage1_validation'] = validate_coreml(s1_path, model1, samples1,
                                                       'Stage 1', i2l, is_stage2=False)
    else:
        print('✗ Stage 1 conversion failed')
        print(rep1.get('convert_error') or rep1.get('trace_error'))

    # ═══ STAGE 2 ═══
    print('\n' + '='*72)
    print('STAGE 2: Stage2 CTC (variable time) → CoreML .mlpackage')
    print('='*72)
    model2_raw, meta2 = load_stage2()
    model2 = Stage2Wrapper(model2_raw).eval()
    i2g = load_idx_to_gloss(meta2)
    dummy2 = torch.randn(1, 32, 61, 5)

    samples2 = []
    for label in ['HELLO', 'THANKYOU', 'I']:
        paths = sorted(feat_dir.glob(f'{label}__*.npy'))
        if paths:
            arr = np.load(paths[0]).astype(np.float32)[None, ...]
            with torch.no_grad():
                pt_logit = model2(torch.from_numpy(arr)).cpu().numpy()
            samples2.append({'label': label, 'arr': arr, 'pt_logit': pt_logit})
    # Phrase sample (T=160)
    for p in sorted(feat_dir.glob('PHRASE__*.npy')):
        label = 'PHRASE_' + p.stem.split('__')[1]
        arr = np.load(p).astype(np.float32)[None, ...]
        with torch.no_grad():
            pt_logit = model2(torch.from_numpy(arr)).cpu().numpy()
        samples2.append({'label': label, 'arr': arr, 'pt_logit': pt_logit})

    # (A) Try fixed-shape [1, 32, 61, 5] (single clip)
    s2a_path = coreml_dir / 'Stage2_fixed32.mlpackage'
    ok2a, rep2a = try_convert(
        model2, dummy2, 'Stage 2 (fixed [1,32,61,5] FP16)', s2a_path,
        input_shapes=[('landmarks', (1, 32, 61, 5))],
        compute_precision='FLOAT16',
        minimum_deployment_target=ct.target.iOS15,
    )
    results['stage2_fixed32'] = rep2a
    if ok2a:
        print(f'✓ Stage 2 (fixed 32) saved ({rep2a["mlpackage_size_mb"]:.1f} MB)')
        # Only validate against single-clip samples
        single_clip_samples = [s for s in samples2 if s['arr'].shape[1] == 32]
        results['stage2_fixed32_validation'] = validate_coreml(
            s2a_path, model2, single_clip_samples, 'Stage 2 (fixed 32)', i2g, is_stage2=True)
    else:
        print('✗ Stage 2 (fixed 32) conversion failed')
        print(rep2a.get('convert_error') or rep2a.get('trace_error'))

    # (B) Try flexible shape [1, RangeDim(32, 320, 32), 61, 5]
    print('\n  Attempting Stage 2 with flexible time dimension...')
    try:
        traced2 = torch.jit.trace(model2, torch.randn(1, 160, 61, 5), strict=False)
        traced2.eval()
        # EnumeratedShapes is more reliable for CoreML than RangeDim with AdaptiveAvgPool etc.
        flex_input = ct.TensorType(
            name='landmarks',
            shape=ct.EnumeratedShapes(
                shapes=[(1, 32, 61, 5), (1, 64, 61, 5), (1, 96, 61, 5),
                        (1, 128, 61, 5), (1, 160, 61, 5), (1, 192, 61, 5),
                        (1, 224, 61, 5), (1, 256, 61, 5), (1, 288, 61, 5), (1, 320, 61, 5)],
                default=(1, 32, 61, 5),
            ),
            dtype=np.float32,
        )
        mlmodel = ct.convert(
            traced2, inputs=[flex_input], convert_to='mlprogram',
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS15,
        )
        s2b_path = coreml_dir / 'Stage2_flex.mlpackage'
        mlmodel.save(str(s2b_path))
        size = sum(f.stat().st_size for f in Path(s2b_path).rglob('*')) / 1e6
        rep2b = {'convert_ok': True, 'mlpackage_size_mb': size, 'label': 'Stage 2 flex'}
        print(f'✓ Stage 2 (flexible) saved ({size:.1f} MB)')
        results['stage2_flex'] = rep2b
        results['stage2_flex_validation'] = validate_coreml(s2b_path, model2, samples2,
                                                             'Stage 2 (flex)', i2g, is_stage2=True)
    except Exception as e:
        rep2b = {'convert_ok': False, 'label': 'Stage 2 flex',
                 'error': f'{type(e).__name__}: {str(e)[:600]}',
                 'traceback': traceback.format_exc()[-1500:]}
        print(f'✗ Stage 2 (flex) conversion failed')
        print(f'  {rep2b["error"]}')
        results['stage2_flex'] = rep2b

    # Save report
    (ARTIFACTS / 'coreml_report.json').write_text(json.dumps(results, indent=2, default=str))
    print(f'\n▶ Report saved: {ARTIFACTS/"coreml_report.json"}')


if __name__ == '__main__':
    main()
