"""
Roundtrip test: simulate a use_angles=True retrain without actually running it.

Proves that after the audit fixes:
1. make_checkpoint_v16 persists use_angles/use_pairwise/depth
2. inference_v16.py load_models restores them
3. Strict-load check passes (no silent weight drops)
4. Model forward pass produces finite outputs
5. SLTStage2V16CTC loads a Stage 1 encoder trained with use_angles

If this test passes, the Kaggle retrain command `--use_angles` will produce a
checkpoint that inference_v16.py can load cleanly.
"""
import os, sys, tempfile
import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, "src_v16")

from model_v16 import (
    SLTStage1V16, SLTStage2V16CTC, make_checkpoint_v16,
    NUM_NODES, DEFAULT_IN_CHANNELS,
)
from inference_v16 import load_models


def test_stage1_roundtrip(tmpdir, use_angles, use_pairwise, depth=4, dim=384):
    tag = f"angles={use_angles} pairwise={use_pairwise} depth={depth} dim={dim}"
    print(f"\n--- Stage 1 roundtrip: {tag} ---")

    # Build a Stage 1 model with those flags
    s1 = SLTStage1V16(
        num_classes=310, in_channels=9, dim=dim, depth=depth,
        use_angles=use_angles, use_pairwise=use_pairwise,
    ).eval()

    # Fake checkpoint via the real save function
    ckpt = make_checkpoint_v16(
        s1, optimizer=None, scheduler=None, ema=None,
        epoch=0, metric_value=95.0, best_metric=95.0, trigger_times=0,
        label_maps={'label_to_idx': {f'C{i}': i for i in range(310)},
                    'idx_to_label': {i: f'C{i}' for i in range(310)}},
        vocab_size=310, dim=dim, depth=depth, in_channels=9, stage=1,
    )

    # Check: checkpoint has the new fields
    assert ckpt['use_angles'] == use_angles, f"checkpoint dropped use_angles"
    assert ckpt['use_pairwise'] == use_pairwise, f"checkpoint dropped use_pairwise"
    assert ckpt['depth'] == depth, f"checkpoint dropped depth"
    print(f"  ckpt fields OK: use_angles={ckpt['use_angles']}, "
          f"use_pairwise={ckpt['use_pairwise']}, depth={ckpt['depth']}")

    s1_path = os.path.join(tmpdir, f"s1_{tag.replace(' ','_').replace('=','-')}.pth")
    torch.save(ckpt, s1_path)

    # Load via the real inference path (triggers strict-load check)
    loaded, i2l, in_ch, _, _ = load_models(s1_path, None, 'cpu')
    assert loaded.encoder.use_angles == use_angles, \
        f"loaded model use_angles mismatch: {loaded.encoder.use_angles} vs {use_angles}"
    assert loaded.encoder.use_pairwise == use_pairwise, \
        f"loaded model use_pairwise mismatch"
    assert len(loaded.encoder.blocks) == depth, \
        f"loaded encoder depth {len(loaded.encoder.blocks)} vs {depth}"
    print(f"  loaded model matches checkpoint flags")

    # Forward pass produces finite output
    x = torch.randn(1, 32, NUM_NODES, 9)
    with torch.no_grad():
        out = loaded(x)
    assert torch.isfinite(out).all(), "non-finite outputs"
    assert out.shape == (1, 310), f"unexpected shape {out.shape}"
    print(f"  forward pass OK shape={tuple(out.shape)} min={out.min():.3f} max={out.max():.3f}")

    return s1_path


def test_stage2_loads_stage1_encoder(tmpdir, use_angles, depth=4, dim=384):
    tag = f"angles={use_angles} depth={depth}"
    print(f"\n--- Stage 2 loads Stage 1 encoder: {tag} ---")
    s1 = SLTStage1V16(num_classes=310, in_channels=9, dim=dim, depth=depth,
                       use_angles=use_angles).eval()
    ckpt = make_checkpoint_v16(
        s1, None, None, None, epoch=0, metric_value=95.0, best_metric=95.0,
        trigger_times=0,
        label_maps={'label_to_idx': {f'C{i}': i for i in range(310)},
                    'idx_to_label': {i: f'C{i}' for i in range(310)}},
        vocab_size=310, dim=dim, depth=depth, in_channels=9, stage=1,
    )
    s1_path = os.path.join(tmpdir, f"s1_for_s2_{tag.replace(' ','_').replace('=','-')}.pth")
    torch.save(ckpt, s1_path)

    # Stage 2 constructs with flags read from Stage 1 checkpoint
    s1_ckpt = torch.load(s1_path, map_location='cpu', weights_only=False)
    s2 = SLTStage2V16CTC(
        vocab_size=311, stage1_ckpt=s1_path,
        in_channels=s1_ckpt['in_channels'], dim=s1_ckpt['d_model'],
        encoder_depth=s1_ckpt['depth'],
        use_angles=s1_ckpt['use_angles'], use_pairwise=s1_ckpt['use_pairwise'],
    ).eval()
    assert s2.encoder.use_angles == use_angles
    assert len(s2.encoder.blocks) == depth

    # Round-trip Stage 2 ckpt
    s2_ckpt = make_checkpoint_v16(
        s2, None, None, None, epoch=0, metric_value=5.0, best_metric=5.0,
        trigger_times=0,
        label_maps={'gloss_to_idx': {f'G{i}': i for i in range(311)},
                    'idx_to_gloss': {i: f'G{i}' for i in range(311)}},
        vocab_size=311, dim=dim, depth=depth, in_channels=9, stage=2,
    )
    s2_path = os.path.join(tmpdir, f"s2_{tag.replace(' ','_').replace('=','-')}.pth")
    torch.save(s2_ckpt, s2_path)

    loaded_s1, _, _, loaded_s2, _ = load_models(s1_path, s2_path, 'cpu')
    assert loaded_s2.encoder.use_angles == use_angles
    assert len(loaded_s2.encoder.blocks) == depth

    # Forward pass on continuous input (multi-clip)
    x = torch.randn(1, 64, NUM_NODES, 9)
    with torch.no_grad():
        logits, _ = loaded_s2(x)
    assert torch.isfinite(logits).all()
    print(f"  Stage 2 forward OK shape={tuple(logits.shape)}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # Case 1: baseline (matches current production checkpoint)
        test_stage1_roundtrip(tmp, use_angles=False, use_pairwise=False, depth=4)
        # Case 2: use_angles enabled (what the Kaggle retrain will produce)
        test_stage1_roundtrip(tmp, use_angles=True, use_pairwise=False, depth=4)
        # Case 3: use_pairwise (never used but wire-check)
        test_stage1_roundtrip(tmp, use_angles=False, use_pairwise=True, depth=4)
        # Case 4: combined
        test_stage1_roundtrip(tmp, use_angles=True, use_pairwise=True, depth=4)
        # Case 5: non-default depth (would silently mismatch before fix)
        test_stage1_roundtrip(tmp, use_angles=False, use_pairwise=False, depth=6)

        # Stage 2 pairings
        test_stage2_loads_stage1_encoder(tmp, use_angles=False, depth=4)
        test_stage2_loads_stage1_encoder(tmp, use_angles=True,  depth=4)
        test_stage2_loads_stage1_encoder(tmp, use_angles=False, depth=6)

    print("\nAll roundtrip cases passed.")


if __name__ == "__main__":
    main()
