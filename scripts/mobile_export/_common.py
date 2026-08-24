"""Shared helpers for the mobile export experiment.

Loads v16 models without touching src_v16/. Provides test-video feature extraction.
"""
from __future__ import annotations

import os, sys, json
from pathlib import Path
import numpy as np
import torch

REPO = Path('/Users/frnzlo/Documents/machine_learning/SLT')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src_v16'))

from src_v16.model_v16 import SLTStage1V16, SLTStage2V16CTC  # noqa: E402

# ── Paths ──
CKPT_STAGE1 = REPO / 'src_v16' / 'output_v16_d384' / 'best_model.pth'
CKPT_STAGE2 = Path('/Users/frnzlo/Downloads/models 2/output_stage2_v16/stage2_best_model.pth')
T5_DIR      = REPO / 'weights' / 'slt_final_t5_model'
ARTIFACTS   = REPO / 'mobile_export' / 'artifacts'
REPORTS     = REPO / 'mobile_export' / 'reports'

ARTIFACTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def _clean_sd(sd: dict) -> dict:
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}


def load_stage1(ckpt_path: Path = CKPT_STAGE1) -> tuple[SLTStage1V16, dict]:
    """Load Stage 1 model in eval mode. Returns (model, meta)."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    num_classes = ckpt.get('num_classes', 310)
    d_model     = ckpt.get('d_model', 384)
    in_channels = ckpt.get('in_channels', 5)
    model = SLTStage1V16(
        num_classes=num_classes,
        in_channels=in_channels,
        dim=d_model,
        depth=4,
        use_pairwise=False,
        use_angles=False,
    )
    sd = _clean_sd(ckpt.get('ema_shadow') or ckpt['model_state_dict'])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f'[stage1] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}')
    model.eval()
    meta = {
        'num_classes': num_classes,
        'd_model': d_model,
        'in_channels': in_channels,
        'idx_to_label': ckpt.get('idx_to_label'),
        'label_to_idx': ckpt.get('label_to_idx'),
        'val_acc': ckpt.get('val_acc'),
    }
    return model, meta


def load_stage2(ckpt_path: Path = CKPT_STAGE2) -> tuple[SLTStage2V16CTC, dict]:
    """Load Stage 2 CTC model in eval mode. Returns (model, meta)."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    vocab_size  = ckpt.get('vocab_size', 311)
    d_model     = ckpt.get('d_model', 384)
    in_channels = ckpt.get('in_channels', 5)
    model = SLTStage2V16CTC(
        vocab_size=vocab_size,
        stage1_ckpt=None,
        in_channels=in_channels,
        dim=d_model,
        encoder_depth=4,
        seq_layers=4,
        out_tokens=4,
        use_pairwise=False,
        use_angles=False,
    )
    sd = _clean_sd(ckpt.get('ema_shadow') or ckpt['model_state_dict'])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f'[stage2] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}')
    model.eval()
    # Re-enable grads (model forcibly freezes encoder on __init__)
    for p in model.parameters():
        p.requires_grad = False
    meta = {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'in_channels': in_channels,
        'idx_to_gloss': ckpt.get('idx_to_gloss'),
        'gloss_to_idx': ckpt.get('gloss_to_idx'),
        'val_wer': ckpt.get('val_wer'),
    }
    return model, meta


def extract_video(video_path: Path, body_3d_interval: int = 4) -> np.ndarray | None:
    """Extract features from a video using src_v16/extract_v16.py.

    Returns np.float32 tensor [32, 61, 5] or None.
    """
    from src_v16.extract_v16 import extract_video_v16
    arr = extract_video_v16(str(video_path), body_3d_interval=body_3d_interval)
    if arr is None:
        return None
    return arr.astype(np.float32)


def ctc_greedy_decode(logits: np.ndarray, blank: int = 0) -> list[int]:
    """Simple greedy CTC decode. logits: [T, V] or [1, T, V]."""
    if logits.ndim == 3:
        logits = logits[0]
    path = logits.argmax(axis=-1)
    out = []
    prev = -1
    for p in path:
        p = int(p)
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return out


def load_idx_to_gloss(meta: dict) -> dict[int, str]:
    i2g = meta.get('idx_to_gloss') or meta.get('idx_to_label') or {}
    # Keys may be int or str depending on pickle
    return {int(k): v for k, v in i2g.items()}
