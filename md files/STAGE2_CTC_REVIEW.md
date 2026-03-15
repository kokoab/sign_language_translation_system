# Stage 2 CTC Training Script — Review

Review of `train_stage_2.py` focusing on collate, CTC shapes, and adaptive pooling math.

---

## Your Three Fixes — Verdict

| Fix | Verdict |
|---|---|
| 42-node / 10-channel restore | **Correct.** Matches your updated Stage 1. |
| AdaptiveAvgPool1d(4) replacing mean pool | **Good design.** 4 tokens per clip gives CTC 4× ratio over gloss count. Eliminates the interpolation hack. |
| online_augment in training loop | **Correct placement.** Applied to padded batch before forward. |

---

## collate_ctc — CORRECT

```python
x_pad = pad_sequence(xs, batch_first=True)  # [B, max_T, 42, 10]
y_flat = torch.cat(ys)                       # [sum(target_lengths)]
```

- `pad_sequence` zero-pads along dim=0 of each tensor (the time axis). Since each `xs[i]` is `[T_i, 42, 10]`, this correctly pads to `[max_T, 42, 10]` per sample. **Correct.**
- `y_flat` is a flat 1D tensor of all target indices concatenated — this is exactly what `nn.CTCLoss` expects when targets are not padded. **Correct.**
- `x_lens` and `y_lens` are properly computed from original (pre-pad) sizes. **Correct.**

---

## CTC Shape Flow — CORRECT

Tracing the full tensor flow for a batch of 32 with max 6 clips:

| Step | Shape | Notes |
|---|---|---|
| Input `x_pad` | `[32, 192, 42, 10]` | max 6 clips × 32 frames |
| Per-sample slice | `[T_b, 42, 10]` | actual frames for sample b |
| Clip reshape | `[num_clips, 32, 42, 10]` | splits into 32-frame chunks |
| Encoder output | `[num_clips, 32, 256]` | frozen DSGCN |
| Permute for pool | `[num_clips, 256, 32]` | channels-first for Conv1d |
| AdaptiveAvgPool1d(4) | `[num_clips, 256, 4]` | 32 → 4 compression |
| Permute back | `[num_clips, 4, 256]` | time-first again |
| Flatten | `[num_clips × 4, 256]` | continuous token sequence |
| pad_sequence | `[B, max_tokens, 256]` | padded for LSTM |
| BiLSTM output | `[B, max_tokens, 1024]` | hidden=512 × 2 bidirectional |
| Classifier | `[B, max_tokens, vocab]` | projection |
| `.transpose(0, 1)` | `[max_tokens, B, vocab]` | **CTC format: [T, B, C]** ✓ |

The `.transpose(0, 1)` at line 373 correctly converts from batch-first `[B, T, C]` to time-first `[T, B, C]` as required by `nn.CTCLoss`. **Correct.**

---

## out_lens Math — CORRECT

```python
out_lens[b] = num_clips * 4
```

For a sample with L glosses → L clips → L×4 output tokens. CTC requires `input_length >= target_length`. Here `input_length = L×4` and `target_length = L`. The ratio is 4:1, giving CTC ample room for 3 blank positions between each gloss. **Correct and well-designed.**

---

## Issues Found

### S2-1. Encoder DropPath Removed — VERIFY CHECKPOINT MATCH

The Stage 2 encoder omits `DropPath` and `drop_path_rate` from DSGCNEncoder:

```python
# Stage 1 encoder:
for layer, dp in zip(self.transformer_layers, self.drop_paths): h = h + dp(layer(h) - h)

# Stage 2 encoder:
for layer in self.transformer_layers: h = layer(h)
```

Stage 1 has `self.drop_paths` (a ModuleList) and uses the residual-with-DropPath pattern `h + dp(layer(h) - h)`. Stage 2 uses plain `layer(h)`. This means:

1. **`load_state_dict` will fail** if the Stage 1 checkpoint contains `drop_paths.*` keys (which it does — DropPath has no learnable params, but it's registered as a ModuleList member). The `enc_state` dict extraction at line 143 filters for keys containing `'encoder.'`, so `drop_paths` keys will be included. `load_state_dict` will raise `Unexpected key(s)` for those keys.

**Fix**: Either add `strict=False` to `self.encoder.load_state_dict(enc_state, strict=False)`, or restore the DropPath modules to the encoder definition. Since the encoder is frozen and in eval mode (DropPath is identity during eval), either approach produces identical inference.

---

### S2-2. Per-Sample Loop in Forward — SLOW but Correct

```python
for b in range(B):
    valid_x = x[b, :x_lens[b]]
    ...
    enc_out = self.encoder(clips)
```

This runs the encoder B times (once per sample) with varying `num_clips`. For batch_size=32 and avg 4 clips, that's ~32 sequential encoder calls vs one batched call with ~128 clips. The per-sample approach is ~5-10× slower but avoids the padding-in-encoder problem (padded clips would produce garbage features).

**Not a bug** — it's a correctness-over-speed tradeoff. For training with only 8K synthetic sentences, wall-clock time is dominated by the encoder (frozen, no grad), so the total overhead is moderate. If training becomes bottlenecked here, consider batching all clips across the batch into one encoder call and tracking which clip belongs to which sample.

---

### S2-3. Augmentation Corrupts Padded Regions — BUG

```python
x_pad = online_augment(x_pad)  # line 365
```

`online_augment` applies rotation, scaling, and noise to the **entire padded tensor** `[B, max_T, 42, 10]`, including the zero-padded frames. The padded regions (beyond `x_lens[b]` for each sample) get non-zero noise and scaling applied to what was previously zeros.

Inside `forward()`, the slice `valid_x = x[b, :x_lens[b]]` correctly truncates to actual frames, so the padded-and-augmented garbage is never seen by the encoder.

**However**: the `view(num_clips, 32, 42, 10)` at line 178 requires `x_lens[b]` to be exactly divisible by 32. If `online_augment` somehow changes the tensor size (it doesn't — it's element-wise), this would break. Currently safe.

**Verdict: Not a bug in practice** because the slice discards padding before the encoder sees it. The augmented noise on padding wastes a few FLOPs but doesn't affect correctness.

---

### S2-4. `random.choices` Allows Duplicate Glosses — INFO

```python
seq_glosses = random.choices(self.vocab_keys, k=seq_len)  # line 230
```

`random.choices` samples **with replacement**, so a synthetic sentence can contain the same gloss twice: `[HELLO, HELLO, YOU]`. This is actually fine — repeated signs occur in natural signing, and CTC can handle consecutive identical labels (it inserts blanks between them). Just noting it differs from my original design which used `random.choice(..., replace=False)`.

---

### S2-5. `.npy` Load Failure Silently Shortens Targets — BUG

```python
def __getitem__(self, idx):
    files, targets = self.samples[idx]
    arrays = []
    for f in files:
        arr = np.load(self.data_path / f).astype(np.float32)
        if arr.shape == (32, 42, 10):
            arrays.append(arr)
    x = np.concatenate(arrays, axis=0)
    return x, targets  # targets still has original length!
```

If any `.npy` file fails the shape check `(32, 42, 10)`, that clip is skipped from `arrays` but the corresponding target index remains in `targets`. Example: 4 files, file #2 is corrupt → `arrays` has 3 clips (96 frames) but `targets` has 4 glosses. CTC input_length = 3×4 = 12, target_length = 4. CTC constraint `12 >= 4` still holds, but the gloss sequence no longer matches the landmark sequence — gloss #3 now aligns to clip #4's features.

**Fix**: Filter `targets` alongside `arrays`:

```python
valid = []
for f, t in zip(files, targets):
    arr = np.load(...)
    if arr.shape == (32, 42, 10):
        arrays.append(arr)
        valid.append(t)
return np.concatenate(arrays, axis=0), valid
```

**Severity**: Low if all your .npy files are clean `(32, 42, 10)`. High if any are corrupt.

---

### S2-6. `dtype=np.uint8` in WER — Overflow for Long Sequences

```python
d = np.zeros((len(reference) + 1, len(hypothesis) + 1), dtype=np.uint8)
```

`uint8` overflows at 255. For sequences longer than 255 tokens, the edit distance wraps around. With max_len=6 glosses, the max edit distance is 6, so this is safe now. But if you increase `max_len` beyond ~20 or use this function elsewhere, use `np.int32`.

---

### S2-7. No AMP / GradScaler — MISSED OPTIMIZATION

The training loop uses no mixed precision:

```python
loss = ctc_loss_fn(log_probs, y_flat, out_lens, y_lens)
loss.backward()
```

On T4, wrapping the forward pass in `torch.amp.autocast('cuda')` and using `GradScaler` would halve the memory for LSTM activations and speed up the LSTM + classifier by ~1.5-2×. The frozen encoder is already outside autocast (inside `torch.no_grad()`), so AMP only applies to the trainable LSTM + projection.

---

### S2-8. Optimizer Includes Frozen Encoder Params — BUG

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

`model.parameters()` includes the frozen encoder parameters. AdamW will allocate momentum and variance buffers for all encoder params (~4M params × 2 states × 4 bytes = ~32 MB wasted), and weight_decay will be applied to them (though gradients are zero, so no actual update occurs). It won't cause incorrect training, but:

1. Wastes ~32 MB of GPU memory for optimizer states on frozen params
2. The optimizer step iterates over all params including frozen ones

**Fix**: Only pass trainable params:

```python
optimizer = optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3, weight_decay=1e-4
)
```

---

### S2-9. No Checkpoint Save/Resume — MISSING

The training loop runs 50 epochs with no checkpoint saving. On Kaggle, if the session crashes or times out, all progress is lost. Stage 1 has robust checkpoint logic — Stage 2 should mirror it.

---

### S2-10. No Scheduler — MINOR

Flat `lr=1e-3` for all 50 epochs. A cosine warmup scheduler (already available from Stage 1) would likely improve convergence. Not critical for initial experiments.

---

## Summary

| Finding | Severity | Status |
|---|---|---|
| collate_ctc shapes | — | **CORRECT** |
| CTC [T, B, C] transpose | — | **CORRECT** |
| out_lens = num_clips × 4 | — | **CORRECT** (4:1 ratio) |
| AdaptiveAvgPool1d(4) design | — | **CORRECT and well-motivated** |
| S2-1. DropPath mismatch | **Medium** | `load_state_dict` will likely fail |
| S2-2. Per-sample encoder loop | Info | Correct but slow |
| S2-3. Augment on padded region | Info | Safe (sliced before encoder) |
| S2-4. Duplicate glosses | Info | Acceptable |
| S2-5. Shape check skips clip but keeps target | **Bug** | Target-landmark misalignment |
| S2-6. uint8 WER matrix | Info | Safe at max_len=6 |
| S2-7. No AMP | Optimization | ~1.5-2× speedup opportunity |
| S2-8. Optimizer includes frozen params | **Minor bug** | Wasted memory + compute |
| S2-9. No checkpoint save/resume | **Missing** | All progress lost on crash |
| S2-10. No scheduler | Minor | Flat LR for 50 epochs |

### Must-Fix Before Running

1. **S2-1**: Add `strict=False` to encoder `load_state_dict`, or restore DropPath to the encoder class — otherwise checkpoint loading crashes.
2. **S2-5**: Filter targets alongside arrays in `__getitem__` — otherwise corrupted files cause target-landmark misalignment.
3. **S2-8**: Filter optimizer to `requires_grad` params only.
