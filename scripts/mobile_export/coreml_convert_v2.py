"""CoreML conversion v2 — monkey-patch MultiheadAttention to unfused form.

coremltools 9.0 doesn't support aten::_native_multi_head_attention (the fused path
used by torch.nn.MultiheadAttention in PyTorch ≥2.x). We replace each MHA module
with an unfused ManualMHA that produces identical outputs, then trace & convert.
"""
from __future__ import annotations
import os, sys, json, time, warnings, traceback
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from _common import load_stage1, load_stage2, ARTIFACTS, ctc_greedy_decode, load_idx_to_gloss

warnings.filterwarnings('ignore')

import coremltools as ct


class ManualMHA(nn.Module):
    """Unfused drop-in replacement for torch.nn.MultiheadAttention (batch_first=True).

    Uses the same weight layout (in_proj_weight [3*D, D], in_proj_bias [3*D],
    out_proj.weight [D, D], out_proj.bias [D]) so weights copy 1:1.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, attn_mask=None, need_weights=False):
        # q, k, v: [B, T, D] (batch_first=True)
        B, T, D = q.shape
        qkv = F.linear(q, self.in_proj_weight, self.in_proj_bias)  # [B, T, 3D]
        qc, kc, vc = qkv.chunk(3, dim=-1)
        H, Dh = self.num_heads, self.head_dim
        qc = qc.view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        kc = kc.view(B, T, H, Dh).transpose(1, 2)
        vc = vc.view(B, T, H, Dh).transpose(1, 2)
        # Manual scaled dot-product attention (no fused call).
        scale = Dh ** -0.5
        attn = torch.matmul(qc, kc.transpose(-2, -1)) * scale  # [B, H, T, T]
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, vc)  # [B, H, T, Dh]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        return out, None


def swap_mha_inplace(module: nn.Module):
    """Replace every nn.MultiheadAttention child with a ManualMHA copying weights."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            m = ManualMHA(child.embed_dim, child.num_heads, dropout=child.dropout)
            with torch.no_grad():
                m.in_proj_weight.copy_(child.in_proj_weight)
                m.in_proj_bias.copy_(child.in_proj_bias)
                m.out_proj.weight.copy_(child.out_proj.weight)
                m.out_proj.bias.copy_(child.out_proj.bias)
            m.eval()
            setattr(module, name, m)
        else:
            swap_mha_inplace(child)


def numerical_sanity(pt_before, pt_after, sample, label):
    """Ensure swap didn't change outputs."""
    with torch.no_grad():
        a = pt_before(sample)
        b = pt_after(sample)
    if isinstance(a, tuple): a = a[0]
    if isinstance(b, tuple): b = b[0]
    max_abs = float((a - b).abs().max())
    print(f'  {label}: max|Δ| before-vs-after MHA swap = {max_abs:.2e}')
    return max_abs


class Stage2Wrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        logits, _ = self.m(x)
        return logits


def try_convert(traced, sample_inputs, label, out_path, input_spec, precision='FLOAT16'):
    rep = {'label': label, 'out_path': str(out_path), 'precision': precision}
    try:
        mlmodel = ct.convert(
            traced,
            inputs=input_spec,
            convert_to='mlprogram',
            compute_precision=(ct.precision.FLOAT16 if precision == 'FLOAT16' else ct.precision.FLOAT32),
            minimum_deployment_target=ct.target.iOS15,
        )
        mlmodel.save(str(out_path))
        rep['convert_ok'] = True
        rep['mlpackage_size_mb'] = sum(f.stat().st_size for f in Path(out_path).rglob('*')) / 1e6
        return True, rep, mlmodel
    except Exception as e:
        rep['convert_ok'] = False
        rep['error'] = f'{type(e).__name__}: {str(e)[:500]}'
        rep['traceback'] = traceback.format_exc()[-1200:]
        return False, rep, None


def validate(mlmodel, pt_model, samples, label, idx_to_name, is_stage2=False):
    spec = mlmodel.get_spec()
    in_name = spec.description.input[0].name
    out_name = spec.description.output[0].name
    parity = []
    for sample in samples:
        arr = sample['arr']
        pt_logit = sample['pt_logit']
        try:
            t0 = time.perf_counter()
            out = mlmodel.predict({in_name: arr})
            ms = (time.perf_counter() - t0) * 1000
            cm_logit = np.array(out[out_name]).reshape(pt_logit.shape)
        except Exception as e:
            parity.append({'label': sample['label'], 'error': f'{type(e).__name__}: {str(e)[:200]}'})
            continue

        max_abs = float(np.abs(pt_logit - cm_logit).max())
        mean_abs = float(np.abs(pt_logit - cm_logit).mean())
        if is_stage2:
            pt_dec = ctc_greedy_decode(pt_logit, blank=0)
            cm_dec = ctc_greedy_decode(cm_logit, blank=0)
            match = pt_dec == cm_dec
            pt_g = [idx_to_name.get(i, f'<{i}>') for i in pt_dec]
            cm_g = [idx_to_name.get(i, f'<{i}>') for i in cm_dec]
            print(f'  [{sample["label"]:25s}]  max|Δ|={max_abs:.2e}  mean|Δ|={mean_abs:.2e}  '
                  f'{"MATCH" if match else "DIFF"}  ({ms:.1f}ms)')
            if not match:
                print(f'      pt: {pt_g}')
                print(f'      cm: {cm_g}')
            parity.append({'label': sample['label'], 'max_abs': max_abs, 'mean_abs': mean_abs,
                           'match': match, 'pt': pt_g, 'cm': cm_g, 'ms': ms})
        else:
            pt1, cm1 = int(pt_logit.argmax()), int(cm_logit.argmax())
            match = pt1 == cm1
            print(f'  [{sample["label"]:12s}]  max|Δ|={max_abs:.2e}  mean|Δ|={mean_abs:.2e}  '
                  f'pt={idx_to_name.get(pt1):12s} cm={idx_to_name.get(cm1):12s}  '
                  f'{"MATCH" if match else "DIFF"}  ({ms:.1f}ms)')
            parity.append({'label': sample['label'], 'max_abs': max_abs, 'mean_abs': mean_abs,
                           'pt_top1': pt1, 'cm_top1': cm1, 'match': match, 'ms': ms})
    return {'parity': parity,
            'mean_ms': float(np.mean([p['ms'] for p in parity if 'ms' in p])) if parity else None,
            'match_rate': sum(1 for p in parity if p.get('match')) / max(1, len(parity)),
            'max_abs_global': max([p.get('max_abs', -1) for p in parity if 'max_abs' in p], default=-1)}


def main():
    torch.set_grad_enabled(False)
    coreml_dir = ARTIFACTS / 'coreml'
    coreml_dir.mkdir(exist_ok=True)

    results = {'coremltools_version': ct.__version__, 'torch_version': torch.__version__}

    # ── Stage 1 ──
    print('='*72)
    print('STAGE 1 via ManualMHA swap')
    print('='*72)
    m1, meta1 = load_stage1()
    i2l = load_idx_to_gloss(meta1)
    # Keep original for sanity check
    import copy
    m1_orig = copy.deepcopy(m1).eval()
    swap_mha_inplace(m1)
    m1.eval()
    sample = torch.randn(1, 32, 61, 5)
    numerical_sanity(m1_orig, m1, sample, 'Stage 1')

    # Build samples
    feat_dir = ARTIFACTS / 'test_features'
    samples1 = []
    for p in sorted(feat_dir.glob('*.npy')):
        if p.name.startswith('PHRASE__'): continue
        label = p.stem.split('__')[0]
        arr = np.load(p).astype(np.float32)[None, ...]
        with torch.no_grad():
            pt_logit = m1(torch.from_numpy(arr)).cpu().numpy()
        samples1.append({'label': label, 'arr': arr, 'pt_logit': pt_logit})

    traced1 = torch.jit.trace(m1, sample, strict=False).eval()
    s1_path = coreml_dir / 'Stage1.mlpackage'
    ok, rep, mlmodel = try_convert(
        traced1, sample, 'Stage 1 fixed [1,32,61,5] FP16', s1_path,
        input_spec=[ct.TensorType(name='landmarks', shape=(1, 32, 61, 5), dtype=np.float32)],
        precision='FLOAT16',
    )
    results['stage1'] = rep
    if ok:
        print(f'✓ Stage 1 saved: {rep["mlpackage_size_mb"]:.1f} MB')
        mlmodel_l = ct.models.MLModel(str(s1_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        print('\n▶ Validating Stage 1 (CPU+NE):')
        results['stage1_validation'] = validate(mlmodel_l, m1, samples1, 'Stage 1', i2l)
    else:
        print('✗ Stage 1 FAILED')
        print(rep.get('error'))

    # ── Stage 2 ──
    print('\n' + '='*72)
    print('STAGE 2 via ManualMHA swap')
    print('='*72)
    m2_raw, meta2 = load_stage2()
    i2g = load_idx_to_gloss(meta2)
    m2 = Stage2Wrapper(m2_raw).eval()
    m2_orig = copy.deepcopy(m2).eval()
    swap_mha_inplace(m2)
    m2.eval()
    numerical_sanity(m2_orig, m2, torch.randn(1, 32, 61, 5), 'Stage 2 32fr')
    numerical_sanity(m2_orig, m2, torch.randn(1, 160, 61, 5), 'Stage 2 160fr')

    samples2 = []
    for lbl in ['HELLO', 'THANKYOU', 'I']:
        paths = sorted(feat_dir.glob(f'{lbl}__*.npy'))
        if paths:
            arr = np.load(paths[0]).astype(np.float32)[None, ...]
            with torch.no_grad():
                pt_logit = m2(torch.from_numpy(arr)).cpu().numpy()
            samples2.append({'label': lbl, 'arr': arr, 'pt_logit': pt_logit})
    for p in sorted(feat_dir.glob('PHRASE__*.npy')):
        arr = np.load(p).astype(np.float32)[None, ...]
        label = 'PHRASE_' + p.stem.split('__')[1]
        with torch.no_grad():
            pt_logit = m2(torch.from_numpy(arr)).cpu().numpy()
        samples2.append({'label': label, 'arr': arr, 'pt_logit': pt_logit})

    # (A) Fixed-shape 32 frames
    traced2a = torch.jit.trace(m2, torch.randn(1, 32, 61, 5), strict=False).eval()
    s2a_path = coreml_dir / 'Stage2_fixed32.mlpackage'
    ok2a, rep2a, mlm2a = try_convert(
        traced2a, torch.randn(1, 32, 61, 5), 'Stage 2 fixed [1,32,61,5]',
        s2a_path,
        input_spec=[ct.TensorType(name='landmarks', shape=(1, 32, 61, 5), dtype=np.float32)],
        precision='FLOAT16',
    )
    results['stage2_fixed32'] = rep2a
    if ok2a:
        print(f'✓ Stage 2 fixed saved: {rep2a["mlpackage_size_mb"]:.1f} MB')
        single_clip = [s for s in samples2 if s['arr'].shape[1] == 32]
        mlm2a_l = ct.models.MLModel(str(s2a_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        print('\n▶ Validating Stage 2 fixed (CPU+NE):')
        results['stage2_fixed32_validation'] = validate(mlm2a_l, m2, single_clip, 'Stage2 fixed', i2g, is_stage2=True)
    else:
        print('✗ Stage 2 fixed FAILED')
        print(rep2a.get('error'))

    # (B) EnumeratedShapes flex
    print('\n▶ Attempting Stage 2 with enumerated flex shapes...')
    try:
        traced2b = torch.jit.trace(m2, torch.randn(1, 160, 61, 5), strict=False).eval()
        flex_in = ct.TensorType(
            name='landmarks',
            shape=ct.EnumeratedShapes(
                shapes=[(1, L, 61, 5) for L in [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]],
                default=(1, 32, 61, 5),
            ),
            dtype=np.float32,
        )
        mlm = ct.convert(
            traced2b, inputs=[flex_in], convert_to='mlprogram',
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS15,
        )
        s2b_path = coreml_dir / 'Stage2_flex.mlpackage'
        mlm.save(str(s2b_path))
        size = sum(f.stat().st_size for f in Path(s2b_path).rglob('*')) / 1e6
        rep2b = {'convert_ok': True, 'mlpackage_size_mb': size, 'label': 'Stage 2 flex enum'}
        results['stage2_flex'] = rep2b
        print(f'✓ Stage 2 flex saved: {size:.1f} MB')
        mlm_l = ct.models.MLModel(str(s2b_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        print('\n▶ Validating Stage 2 flex (CPU+NE):')
        results['stage2_flex_validation'] = validate(mlm_l, m2, samples2, 'Stage2 flex', i2g, is_stage2=True)
    except Exception as e:
        rep2b = {'convert_ok': False, 'label': 'Stage 2 flex',
                 'error': f'{type(e).__name__}: {str(e)[:600]}',
                 'traceback': traceback.format_exc()[-1500:]}
        print(f'✗ Stage 2 flex FAILED: {rep2b["error"]}')
        results['stage2_flex'] = rep2b

    (ARTIFACTS / 'coreml_report_v2.json').write_text(json.dumps(results, indent=2, default=str))
    print(f'\n▶ Report: {ARTIFACTS/"coreml_report_v2.json"}')


if __name__ == '__main__':
    main()
