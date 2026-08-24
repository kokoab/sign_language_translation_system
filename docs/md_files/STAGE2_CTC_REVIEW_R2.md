# Stage 2 CTC Training — Round 2 Review (v2.1)

Review of `train_stage_2.py` after implementing S2-1, S2-5, S2-7, S2-8, S2-9.

---

## Fixes Verified

| Fix | Status |
|---|---|
| S2-1 (DropPath restored) | **Correct.** `drop_paths` ModuleList + residual pattern matches Stage 1 exactly. `load_state_dict` will now succeed. |
| S2-5 (Target alignment) | **Correct.** `zip(files, target_glosses)` with paired append. Fallback returns empty targets, filtered by collate. |
| S2-7 (AMP added) | **Correct.** `autocast` wraps forward+loss, `GradScaler` handles backward/step. |
| S2-8 (Frozen params excluded) | **Correct.** `filter(lambda p: p.requires_grad, ...)` saves ~32 MB optimizer state. |
| S2-9 (Checkpoint resume) | **Correct.** Same pattern as Stage 1: load before `.to(device)`, optimizer states moved after. |
| CosineWarmupScheduler | **Correct.** Identical to Stage 1. |
| ModelEMA | **Correct.** `requires_grad` filter, `.to()` method, apply/restore for val. |
| Smoke test | **Correct.** 100/20 samples, 3 epochs. |

---

## New Issues

### R2-1. `clip_grad_norm_` on ALL Parameters Including Frozen — HARMLESS

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # line 481
```

This iterates over all parameters (including frozen encoder) to compute the gradient norm. Frozen params have `grad=None`, and `clip_grad_norm_` skips them internally. So the clip threshold is computed only from LSTM + classifier gradients, which is correct.

**No bug.** Just a wasted iteration over ~4M frozen params. Negligible cost.

---

### R2-2. `y_flat` Not Moved to Device in Val Loop — BUG

```python
# Training loop (line 467):
y_flat = y_flat.to(device, non_blocking=True)  ✓

# Validation loop (line 501-510):
x_pad = x_pad.to(device, non_blocking=True)
# y_flat never moved to device!
# ...
targets.append(y_flat[idx:idx+length].cpu().tolist())  # line 512
```

`y_flat` stays on CPU in the val loop. The `.cpu().tolist()` at line 512 works because it's already on CPU. This is actually fine — you only use `y_flat` for target reconstruction (slicing and `.tolist()`), never for GPU computation. No CTC loss is computed during validation.

**Not a bug.** Accidentally correct. But inconsistent with training loop.

---

### R2-3. `model.train()` Called But Encoder Should Stay in Eval — SUBTLE BUG

```python
model.train()  # line 460 — sets ALL modules to train mode
```

`model.train()` propagates to the frozen encoder, which sets its `Dropout`, `DropPath`, and `BatchNorm`/`GroupNorm` modules to training mode. While the encoder's **parameters** are frozen (no gradient), its **behavior** changes:

- **DropPath** (line 97-101): Randomly drops entire paths during training. With `drop_prob > 0`, the encoder outputs *different features for the same input* across forward passes. This means the LSTM sees noisy, non-deterministic encoder features during training.
- **Dropout in DSGCNBlock** (line 72): Same issue — randomly drops activations.

This isn't catastrophic (it acts as extra regularization), but it means:
1. The encoder is NOT truly "frozen" in behavior, only in weights
2. Validation uses `model.eval()` → deterministic encoder features, but training sees stochastic features → train/val distribution mismatch

**Fix**: After `model.train()`, force the encoder back to eval mode:

```python
model.train()
model.encoder.eval()
```

This keeps LSTM dropout active (training mode) while keeping the encoder deterministic.

---

### R2-4. `temporal_pool` is Trainable But Not Meaningful — INFO

`nn.AdaptiveAvgPool1d` has **no learnable parameters**. It's a pure functional operation. Including it as `self.temporal_pool` is clean for code organization but doesn't affect the trainable param count. Just confirming this is correct — no wasted optimizer states.

---

### R2-5. Augmentation Noise on Mask Channel — MINOR

```python
return x_rotated * scale + torch.randn_like(x_rotated) * noise_std  # line 296
```

`torch.randn_like(x_rotated)` adds Gaussian noise to all 10 channels, including channel 9 (the mask). The mask channel is binary (0/1 indicating hand presence). Adding noise=0.003 to a binary mask is negligible (0.003 vs 0 or 1), but technically corrupts the mask semantics.

If the encoder's `input_norm` (LayerNorm) amplifies this noise relative to the mask's scale, it could slightly affect attention weights downstream. In practice, σ=0.003 on a 0/1 signal is invisible.

**Not worth fixing.** Just noting for completeness.

---

### R2-6. `make_checkpoint` Saves Full `model.state_dict()` Including Frozen Encoder — CORRECT but LARGE

```python
'model_state_dict': unwrapped.state_dict()  # line 373
```

This saves the entire model including ~4M frozen encoder params (~16 MB). On resume, `model.load_state_dict()` restores everything including the encoder (which was already loaded from Stage 1). This is redundant but correct — it guarantees the checkpoint is self-contained.

The alternative (saving only LSTM + classifier state dicts) would save ~12 MB per checkpoint but complicate resume logic. Current approach is the right tradeoff.

---

### R2-7. Val WER With Tiny Numbers Could Mislead — INFO

`calculate_wer` returns per-sample WER as `edit_distance / max(len(reference), 1)`. For a sentence with 2 glosses where the model predicts 1 wrong gloss, WER = 1/2 = 50%. The epoch-level WER averages these per-sample rates.

This is correct (standard WER computation). Just note that with synthetic sentences of 2-6 glosses, the WER granularity is coarse: each error in a 2-gloss sentence is ±50%. Early epochs will show noisy WER. CTC loss is a smoother training signal.

---

## Summary

| Finding | Severity |
|---|---|
| R2-1. clip_grad_norm on all params | Harmless |
| R2-2. y_flat stays on CPU in val | Accidentally correct |
| R2-3. Encoder in train mode (DropPath/Dropout active) | **Medium** — train/val mismatch |
| R2-4. Pool has no learnable params | Info |
| R2-5. Noise on mask channel | Negligible |
| R2-6. Full state_dict in checkpoint | Correct (self-contained) |
| R2-7. Coarse WER on short sentences | Info |

### Recommended Fix Before Running

**R2-3**: Add `model.encoder.eval()` after `model.train()` at line 460. This keeps the frozen encoder deterministic while allowing LSTM dropout to function normally.

### Script Is Otherwise Production-Ready

All previous critical issues (S2-1, S2-5, S2-8) are correctly fixed. AMP, EMA, scheduler, checkpoint resume, and early stopping all follow the proven Stage 1 patterns. The CTC shape flow `[T, B, C]` and `out_lens = num_clips * 4` are verified correct from Round 1.
