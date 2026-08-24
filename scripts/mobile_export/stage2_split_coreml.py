"""Prove Stage 2 can be deployed to CoreML by splitting into two fixed/simpler models:

  Stage2A (ClipEncoder):  [1, 32, 61, 5]  → [1, 4, 384]    (encoder + tcn, fixed shape)
  Stage2B (SeqCTC):       [1, N*4, 384]   → [1, N*4, 311]  (seq blocks + ctc head)

iOS/Android code runs Stage2A once per 32-frame clip, concatenates into [1, N*4, 384],
then runs Stage2B once for final CTC logits. Matches the original forward exactly.

Stage2B still has a dynamic time dim but NO B*num_clips reshape, so CoreML can handle it.
"""
from __future__ import annotations
import os, sys, json, time, copy, warnings, traceback
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from _common import load_stage2, ARTIFACTS, ctc_greedy_decode, load_idx_to_gloss
from coreml_convert_v2 import ManualMHA, swap_mha_inplace

warnings.filterwarnings('ignore')
import coremltools as ct


class Stage2A(nn.Module):
    """Per-clip encoder + TCN (fixed [1, 32, 61, 5] → [1, 4, 384])."""
    def __init__(self, parent):
        super().__init__()
        self.encoder = parent.encoder
        self.tcn = parent.tcn

    def forward(self, x):
        # x: [1, 32, 61, 5]
        enc = self.encoder(x)          # [1, 32, 384]
        return self.tcn(enc)           # [1, 4, 384]


class Stage2B(nn.Module):
    """Sequence transformer + CTC head. Accepts pre-pooled tokens [1, S, 384]."""
    def __init__(self, parent):
        super().__init__()
        self.seq_pos_enc = parent.seq_pos_enc
        self.seq_blocks = parent.seq_blocks
        self.seq_norm = parent.seq_norm
        self.ctc_head = parent.ctc_head

    def forward(self, tokens):
        # tokens: [1, S, 384]
        S = tokens.shape[1]
        seq = tokens + self.seq_pos_enc[:, :S, :]
        for block in self.seq_blocks:
            seq = block(seq)
        seq = self.seq_norm(seq)
        return self.ctc_head(seq)     # [1, S, 311]


def full_pipeline(a: Stage2A, b: Stage2B, x):
    """Reference: run A per clip, concatenate, run B."""
    B, T, V, C = x.shape
    assert T % 32 == 0, f'T must be multiple of 32, got {T}'
    nc = T // 32
    clips = x.reshape(B * nc, 32, V, C)
    tokens = a(clips)                         # [B*nc, 4, 384]
    seq = tokens.reshape(B, nc * 4, 384)
    return b(seq)


def validate_coreml_pipeline(mlA, mlB, pt_a, pt_b, pt_full, samples, idx_to_name):
    """Run the 2-model CoreML pipeline and compare to PyTorch full pipeline."""
    specA = mlA.get_spec()
    inA = specA.description.input[0].name
    outA = specA.description.output[0].name
    specB = mlB.get_spec()
    inB = specB.description.input[0].name
    outB = specB.description.output[0].name

    parity = []
    for sample in samples:
        arr = sample['arr']  # [1, T, 61, 5]
        pt_logit = sample['pt_logit_full']  # ground truth from original model
        T = arr.shape[1]
        nc = T // 32
        tokens_list = []
        t_total0 = time.perf_counter()
        for i in range(nc):
            clip = arr[:, i*32:(i+1)*32, :, :].astype(np.float32)
            out = mlA.predict({inA: clip})
            tok = np.array(out[outA])
            tokens_list.append(tok)
        tokens = np.concatenate(tokens_list, axis=1).astype(np.float32)  # [1, nc*4, 384]
        out = mlB.predict({inB: tokens})
        cm_logit = np.array(out[outB]).reshape(pt_logit.shape)
        t_total = (time.perf_counter() - t_total0) * 1000

        max_abs = float(np.abs(pt_logit - cm_logit).max())
        mean_abs = float(np.abs(pt_logit - cm_logit).mean())
        pt_dec = ctc_greedy_decode(pt_logit, blank=0)
        cm_dec = ctc_greedy_decode(cm_logit, blank=0)
        match = pt_dec == cm_dec
        pt_g = [idx_to_name.get(i, f'<{i}>') for i in pt_dec]
        cm_g = [idx_to_name.get(i, f'<{i}>') for i in cm_dec]
        print(f'  [{sample["label"]:25s}]  T={T:3d} nc={nc}  max|Δ|={max_abs:.2e}  '
              f'mean|Δ|={mean_abs:.2e}  {"MATCH" if match else "DIFF"}  ({t_total:.1f}ms)')
        if not match:
            print(f'      pt: {pt_g}')
            print(f'      cm: {cm_g}')
        parity.append({'label': sample['label'], 'T': int(T), 'num_clips': int(nc),
                       'max_abs': max_abs, 'mean_abs': mean_abs, 'match': match,
                       'pt_decoded': pt_g, 'cm_decoded': cm_g, 'total_ms': t_total})
    return {
        'parity': parity,
        'match_rate': sum(1 for p in parity if p['match']) / max(1, len(parity)),
        'max_abs_global': max(p['max_abs'] for p in parity),
        'mean_total_ms': float(np.mean([p['total_ms'] for p in parity])),
    }


def main():
    torch.set_grad_enabled(False)
    coreml_dir = ARTIFACTS / 'coreml'
    coreml_dir.mkdir(exist_ok=True)
    results = {}

    print('▶ Loading Stage 2, building A/B split...')
    m2, meta2 = load_stage2()
    i2g = load_idx_to_gloss(meta2)

    # Build PyTorch reference with original model (full forward)
    def pt_full(x):
        logits, _ = m2(x)
        return logits

    # Separate submodules
    a_mod = Stage2A(m2).eval()
    b_mod = Stage2B(m2).eval()

    # Sanity: A+B pipeline should match the original forward to floating-point precision
    print('\n▶ Sanity check: split A+B vs original forward:')
    for T in [32, 64, 160, 320]:
        x = torch.randn(1, T, 61, 5)
        y_full = pt_full(x).cpu().numpy()
        y_split = full_pipeline(a_mod, b_mod, x).cpu().numpy()
        max_abs = float(np.abs(y_full - y_split).max())
        print(f'  T={T:3d}: max|Δ| = {max_abs:.2e}')

    # Swap MHA in each submodule
    swap_mha_inplace(a_mod)
    swap_mha_inplace(b_mod)
    a_mod.eval(); b_mod.eval()

    # Rebuild PyTorch samples using POST-swap models (so CoreML validation uses matching baseline)
    feat_dir = ARTIFACTS / 'test_features'
    samples = []
    for lbl in ['HELLO', 'THANKYOU', 'I', 'YOU']:
        paths = sorted(feat_dir.glob(f'{lbl}__*.npy'))
        if paths:
            arr = np.load(paths[0]).astype(np.float32)[None, ...]
            with torch.no_grad():
                pt_logit = full_pipeline(a_mod, b_mod, torch.from_numpy(arr)).cpu().numpy()
            samples.append({'label': lbl, 'arr': arr, 'pt_logit_full': pt_logit})
    for p in sorted(feat_dir.glob('PHRASE__*.npy')):
        arr = np.load(p).astype(np.float32)[None, ...]
        label = 'PHRASE_' + p.stem.split('__')[1]
        with torch.no_grad():
            pt_logit = full_pipeline(a_mod, b_mod, torch.from_numpy(arr)).cpu().numpy()
        samples.append({'label': label, 'arr': arr, 'pt_logit_full': pt_logit})

    # ── Convert Stage2A (fixed [1, 32, 61, 5] → [1, 4, 384]) ──
    print('\n▶ Converting Stage2A (ClipEncoder, fixed shape)...')
    tracedA = torch.jit.trace(a_mod, torch.randn(1, 32, 61, 5), strict=False).eval()
    pathA = coreml_dir / 'Stage2A_ClipEncoder.mlpackage'
    mlA = ct.convert(
        tracedA,
        inputs=[ct.TensorType(name='clip', shape=(1, 32, 61, 5), dtype=np.float32)],
        convert_to='mlprogram',
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS15,
    )
    mlA.save(str(pathA))
    sizeA = sum(f.stat().st_size for f in Path(pathA).rglob('*')) / 1e6
    print(f'✓ Stage2A saved: {pathA} ({sizeA:.1f} MB)')
    results['stage2A'] = {'path': str(pathA), 'size_mb': sizeA, 'ok': True}

    # ── Convert Stage2B (flexible [1, S, 384] → [1, S, 311]) ──
    # S takes values from {4, 8, 12, ..., 40} (4 tokens per clip, up to 10 clips = 320 frames).
    print('\n▶ Converting Stage2B (SeqCTC, enumerated flex S)...')
    tracedB = torch.jit.trace(b_mod, torch.randn(1, 20, 384), strict=False).eval()
    flex_in = ct.TensorType(
        name='tokens',
        shape=ct.EnumeratedShapes(
            shapes=[(1, S, 384) for S in [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]],
            default=(1, 4, 384),
        ),
        dtype=np.float32,
    )
    pathB = coreml_dir / 'Stage2B_SeqCTC.mlpackage'
    try:
        mlB = ct.convert(
            tracedB, inputs=[flex_in], convert_to='mlprogram',
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS15,
        )
        mlB.save(str(pathB))
        sizeB = sum(f.stat().st_size for f in Path(pathB).rglob('*')) / 1e6
        print(f'✓ Stage2B saved: {pathB} ({sizeB:.1f} MB)')
        results['stage2B'] = {'path': str(pathB), 'size_mb': sizeB, 'ok': True}
    except Exception as e:
        print(f'✗ Stage2B conversion failed: {type(e).__name__}: {str(e)[:400]}')
        results['stage2B'] = {'ok': False, 'error': f'{type(e).__name__}: {str(e)[:400]}',
                               'traceback': traceback.format_exc()[-1500:]}
        (ARTIFACTS / 'stage2_split_coreml_report.json').write_text(json.dumps(results, indent=2, default=str))
        return

    # ── Validate the two-part pipeline ──
    print('\n▶ Validating two-part CoreML pipeline (CPU+NE):')
    mlA_l = ct.models.MLModel(str(pathA), compute_units=ct.ComputeUnit.CPU_AND_NE)
    mlB_l = ct.models.MLModel(str(pathB), compute_units=ct.ComputeUnit.CPU_AND_NE)
    pipe_rep = validate_coreml_pipeline(mlA_l, mlB_l, a_mod, b_mod, pt_full, samples, i2g)
    results['pipeline_validation'] = pipe_rep
    print(f'\n  match rate: {pipe_rep["match_rate"]*100:.1f}%')
    print(f'  max|Δ|:     {pipe_rep["max_abs_global"]:.2e}')
    print(f'  mean total ms (A*nc + B): {pipe_rep["mean_total_ms"]:.1f}')

    (ARTIFACTS / 'stage2_split_coreml_report.json').write_text(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
