"""TFLite conversion for Stage 1 + Stage 2 (split A/B).

Runs inside the isolated atlas_tflite venv at ~/venvs/atlas_tflite.
Uses ai-edge-torch (now renamed litert-torch) 0.7.2 + torch 2.9.1.

Produces:
  artifacts/tflite/Stage1.tflite
  artifacts/tflite/Stage2A_ClipEncoder.tflite
  artifacts/tflite/Stage2B_SeqCTC.tflite      (one per sequence length in {4, 8, 12, 16, 20})
"""
from __future__ import annotations
import os, sys, json, time, copy, warnings, traceback
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

REPO = Path('/Users/frnzlo/Documents/machine_learning/SLT')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src_v16'))

from src_v16.model_v16 import SLTStage1V16, SLTStage2V16CTC  # noqa

warnings.filterwarnings('ignore')

ARTIFACTS = REPO / 'mobile_export' / 'artifacts'
TFLITE_DIR = ARTIFACTS / 'tflite'
TFLITE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS = REPO / 'mobile_export' / 'reports'

CKPT_STAGE1 = REPO / 'src_v16' / 'output_v16_d384' / 'best_model.pth'
CKPT_STAGE2 = Path('/Users/frnzlo/Downloads/models 2/output_stage2_v16/stage2_best_model.pth')


# ── MHA swap (same pattern as CoreML fix — ai-edge-torch also fails on fused MHA) ──
class ManualMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim; self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads; self.dropout_p = dropout
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, attn_mask=None, need_weights=False):
        B, T, D = q.shape
        qkv = F.linear(q, self.in_proj_weight, self.in_proj_bias)
        qc, kc, vc = qkv.chunk(3, dim=-1)
        H, Dh = self.num_heads, self.head_dim
        qc = qc.view(B, T, H, Dh).transpose(1, 2)
        kc = kc.view(B, T, H, Dh).transpose(1, 2)
        vc = vc.view(B, T, H, Dh).transpose(1, 2)
        scale = Dh ** -0.5
        attn = torch.matmul(qc, kc.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, vc)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out), None


def swap_mha_inplace(module):
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


def _clean_sd(sd): return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}


def load_stage1():
    ckpt = torch.load(CKPT_STAGE1, map_location='cpu', weights_only=False)
    model = SLTStage1V16(num_classes=ckpt['num_classes'], in_channels=ckpt['in_channels'],
                         dim=ckpt['d_model'], depth=4)
    sd = _clean_sd(ckpt.get('ema_shadow') or ckpt['model_state_dict'])
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, ckpt


def load_stage2():
    ckpt = torch.load(CKPT_STAGE2, map_location='cpu', weights_only=False)
    model = SLTStage2V16CTC(vocab_size=ckpt['vocab_size'], stage1_ckpt=None,
                            in_channels=ckpt['in_channels'], dim=ckpt['d_model'],
                            encoder_depth=4, seq_layers=4, out_tokens=4)
    sd = _clean_sd(ckpt.get('ema_shadow') or ckpt['model_state_dict'])
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model, ckpt


class Stage2Wrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        logits, _ = self.m(x); return logits


class Stage2A(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.encoder = parent.encoder
        self.tcn = parent.tcn
    def forward(self, x):
        enc = self.encoder(x)
        return self.tcn(enc)


class Stage2B(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.seq_pos_enc = parent.seq_pos_enc
        self.seq_blocks = parent.seq_blocks
        self.seq_norm = parent.seq_norm
        self.ctc_head = parent.ctc_head
    def forward(self, tokens):
        S = tokens.shape[1]
        seq = tokens + self.seq_pos_enc[:, :S, :]
        for block in self.seq_blocks:
            seq = block(seq)
        seq = self.seq_norm(seq)
        return self.ctc_head(seq)


def convert_and_save(model, sample, out_path: Path, label: str):
    """Run ai_edge_torch.convert → save .tflite. Returns report dict."""
    import litert_torch as ai_edge_torch
    rep = {'label': label, 'out_path': str(out_path)}
    try:
        edge = ai_edge_torch.convert(model, sample)
        edge.export(str(out_path))
        size = out_path.stat().st_size / 1e6
        rep['ok'] = True
        rep['size_mb'] = size
        print(f'✓ {label} saved: {out_path.name} ({size:.1f} MB)')
        return rep
    except Exception as e:
        rep['ok'] = False
        rep['error'] = f'{type(e).__name__}: {str(e)[:500]}'
        rep['traceback'] = traceback.format_exc()[-1500:]
        print(f'✗ {label} FAILED')
        print(f'  {rep["error"]}')
        return rep


def main():
    torch.set_grad_enabled(False)
    print(f'▶ torch={torch.__version__}  numpy={np.__version__}')

    results = {'torch_version': torch.__version__}

    # ── Stage 1 ──
    print('\n' + '=' * 64)
    print('STAGE 1 → TFLite')
    print('=' * 64)
    m1, _ = load_stage1()
    m1_orig = copy.deepcopy(m1).eval()
    swap_mha_inplace(m1); m1.eval()
    sample = torch.randn(1, 32, 61, 5)
    with torch.no_grad():
        a, b = m1_orig(sample), m1(sample)
    print(f'  MHA swap sanity max|Δ| = {(a - b).abs().max().item():.2e}')

    results['stage1'] = convert_and_save(m1, (sample,), TFLITE_DIR / 'Stage1.tflite', 'Stage 1')

    # ── Stage 2 split ──
    print('\n' + '=' * 64)
    print('STAGE 2 → TFLite (split A/B)')
    print('=' * 64)
    m2_raw, _ = load_stage2()
    m2 = Stage2Wrapper(m2_raw).eval()
    a_mod = Stage2A(m2_raw).eval()
    b_mod = Stage2B(m2_raw).eval()
    swap_mha_inplace(a_mod); a_mod.eval()
    swap_mha_inplace(b_mod); b_mod.eval()

    results['stage2A'] = convert_and_save(
        a_mod, (torch.randn(1, 32, 61, 5),),
        TFLITE_DIR / 'Stage2A_ClipEncoder.tflite', 'Stage 2A ClipEncoder',
    )

    # Stage 2B needs one model per sequence length. Produce for S in {4, 8, 12, 16, 20}
    # (covers 1-5 clips = 1-5 seconds signing). Mobile code picks the right file.
    results['stage2B_per_length'] = {}
    for S in [4, 8, 12, 16, 20]:
        sample_b = torch.randn(1, S, 384)
        rep = convert_and_save(
            b_mod, (sample_b,),
            TFLITE_DIR / f'Stage2B_SeqCTC_S{S}.tflite',
            f'Stage 2B SeqCTC S={S}',
        )
        results['stage2B_per_length'][f'S{S}'] = rep

    (ARTIFACTS / 'tflite_convert_report.json').write_text(json.dumps(results, indent=2, default=str))
    print(f'\n▶ Report: {ARTIFACTS/"tflite_convert_report.json"}')


if __name__ == '__main__':
    main()
