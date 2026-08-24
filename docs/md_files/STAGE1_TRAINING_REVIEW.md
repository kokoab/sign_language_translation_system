# Stage 1 Training Script — Hardware Review

Review of `train_stage_1.py` across Kaggle GPU tiers.

Constraint: **No changes to GCN/Transformer math or architecture.**

---

## What's Already Done Right

| Feature | Status | Notes |
|---|---|---|
| AMP (mixed precision) | Correct | `torch.amp.autocast` + `GradScaler` — standard A100 pattern |
| `optimizer.zero_grad(set_to_none=True)` | Correct | Avoids memset on gradient buffers |
| GPU preload of full dataset | Correct | 39K × 32 × 21 × 9 × 4 bytes ≈ 950 MB — trivially fits in 80GB. Eliminates all CPU→GPU transfer during training |
| `num_workers=0` when on-GPU | Correct | No point spawning workers when data is already in VRAM |
| `pin_memory=False` when on-GPU | Correct | `pin_memory` is for CPU→GPU transfers — useless when data lives on GPU |
| `non_blocking=True` on `.to()` calls | Correct | Allows overlap of transfer and compute (matters more for val loader) |
| `drop_last=True` on train loader | Correct | Prevents ragged last batch from destabilizing batch norm / GCN stats |
| Gradient clipping | Correct | `clip_grad_norm_` at 5.0 — reasonable for GCN + Transformer |
| WeightedRandomSampler | Correct | Addresses class imbalance across 325 ASL classes |
| Label smoothing | Correct | 0.05 — mild, won't hurt convergence |

---

## Issues Found

### H1. `torch.compile()` Not Used — MODERATE (Free 15-30% Speedup)

The A100 fully supports `torch.compile()` with `mode="reduce-overhead"`, which fuses CUDA kernels and eliminates Python overhead. For a GCN + Transformer pipeline with many small ops (einsum, softmax, layer norm), this is a significant win.

```python
model = SLTStage1(...)
model.to(device)
model = torch.compile(model, mode="reduce-overhead")
```

Why `"reduce-overhead"`: This mode uses CUDA graphs under the hood, which is ideal for fixed-shape inputs like yours (`[B, 32, 21, 9]` — same every batch). The A100's large L2 cache (40MB) amplifies the benefit.

**No quality impact.** Identical gradients, just faster kernel dispatch.

**Caveat**: The first 1-3 epochs will be slower while the compiler traces and optimizes. Subsequent epochs are 15-30% faster. On a 200-epoch run, this is a clear net win.

---

### H2. `batch_size=512` May Be Suboptimal — MINOR (Test 1024)

With 39K samples and 80GB VRAM, your model is tiny relative to the GPU. At `batch_size=512`:

- Model: ~4M params × 4 bytes = ~16 MB
- Batch data: 512 × 32 × 21 × 9 × 4 bytes = ~25 MB
- Activations + gradients: estimated ~200-400 MB

Total: <1 GB out of 80 GB — you're using ~1% of VRAM.

**Recommendation**: Try `batch_size=1024` or `2048`. Larger batches:
- Better saturate the A100's 6912 CUDA cores
- Reduce the number of optimizer steps per epoch (faster wall-clock time)
- Produce more stable gradient estimates

If you increase batch size, scale LR proportionally: `lr = 5e-4 * (new_batch / 512)`. Adjust warmup epochs accordingly. The weighted sampler and label smoothing already handle the class imbalance, so larger batches won't hurt minority classes.

**No quality impact** if LR is scaled correctly. May even improve convergence stability.

---

### H3. TF32 Not Explicitly Enabled — MINOR (Default on A100, but be explicit)

A100 supports TF32 mode, which uses 19-bit mantissa for matmul (vs 23-bit float32). PyTorch enables this by default on A100, but it's good practice to be explicit:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

This is already happening implicitly on your A100, but adding these lines:
1. Documents the intent
2. Ensures it works if PyTorch defaults change
3. Stacks with AMP — TF32 applies to ops that don't get cast to float16 by autocast

**No quality impact.** TF32's 10-bit mantissa precision for the mantissa in matmul is well within noise for your task.

---

### H4. `torch.backends.cudnn.benchmark = True` Missing — MINOR

Since your input shapes are fixed (`[B, 32, 21, 9]` every batch), cuDNN benchmark mode selects the fastest convolution algorithm on first run and caches it. Helps the temporal convolutions in DSGCNBlock.

```python
torch.backends.cudnn.benchmark = True
```

**No quality impact.** Same algorithms, just auto-selected for speed.

---

### H5. Validation Per-Class Loop Is Slow — MINOR

```python
for c in range(num_classes):
    mask = (y == c)
    correct[c] += (preds[mask] == c).sum().cpu()
    total[c] += mask.sum().cpu()
```

This loops 325 times per batch with CPU transfers each iteration. But you only use the *aggregate* accuracy — the per-class breakdown is computed but never reported.

Replace with:

```python
correct = (preds == y).sum().item()
total   = y.size(0)
```

And accumulate across batches. This eliminates the Python loop and 650 `.cpu()` `calls per batch.

**No quality impact.** Same accuracy number, just computed faster.

If you need per-class accuracy later (e.g., for confusion matrix), move it to a separate analysis function after training, not inside the hot eval loop.

---

### H6. Checkpoint Saves Every Epoch — MINOR

```python
torch.save(ckpt, LAST_CKPT)
```

This serializes the entire model + optimizer state to disk every epoch. On Kaggle, this writes to network storage, which is slow. With 200 epochs:

- Each checkpoint is ~16-20 MB (4M params × 4 bytes + optimizer states)
- That's ~4 GB of cumulative writes

**Recommendation**: Save `LAST_CKPT` every 5 or 10 epochs instead. The `BEST_CKPT` save-on-improvement logic is fine as-is.

```python
if epoch % 5 == 0:
    torch.save(ckpt, LAST_CKPT)
```

**No quality impact.** On Kaggle crash-resume, you lose at most 4 extra epochs.

---

### H7. `online_augment` Rotation Applied to ALL 9 Channels — VERIFY INTENT

```python
xr = x.view(B, T, N, C // 3, 3)    # C=9 → 3 groups of 3
xr = torch.einsum('btngi,bij->btngj', xr, R)
```

This rotates all three triplets: `[xyz, velocity, acceleration]`. Mathematically this is correct — if you rotate positions, velocities and accelerations must rotate by the same matrix. This is physically consistent.

**No issue, just confirming the math is right.** The rotation is applied uniformly to position, velocity, and acceleration vectors, which preserves the physical relationship between them.

---

### H8. GradScaler on A100 — HARMLESS but Redundant

On A100, AMP with `dtype=torch.bfloat16` is preferred over `float16` because bfloat16 has the same exponent range as float32, making `GradScaler` unnecessary (no underflow risk).

Your current setup uses `autocast('cuda')` which defaults to float16. This works fine with GradScaler and is not wrong. But switching to bfloat16 would let you remove the scaler entirely:

```python
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    ...
loss.backward()  # No scaler needed
optimizer.step()
```

**Marginal improvement.** Removes 3 scaler-related calls per step. The numerical difference between float16+scaler and bfloat16 is negligible for your model size.

---

## Summary

| Issue | Severity | Quality Impact | Speed Impact |
|---|---|---|---|
| H1. `torch.compile()` | **Moderate** | None | **15-30% faster** after warmup |
| H2. Batch size → 1024+ | Minor | None (if LR scaled) | Better GPU utilization |
| H3. Explicit TF32 | Minor | None | Already active, just undocumented |
| H4. cuDNN benchmark | Minor | None | Faster temporal convolutions |
| H5. Vectorize eval loop | Minor | None | Eliminates 325-iter Python loop per batch |
| H6. Checkpoint frequency | Minor | None | Less disk I/O on Kaggle |
| H7. Rotation on all 9ch | Info | N/A | Confirmed correct |
| H8. bfloat16 over float16 | Minor | None | Removes GradScaler overhead |

### Priority Order for Implementation

1. **H1** — `torch.compile(model, mode="reduce-overhead")` — biggest single win
2. **H4** — `cudnn.benchmark = True` — one line
3. **H5** — Vectorize eval accuracy — quick fix
4. **H2** — Test `batch_size=1024` — measure throughput
5. **H8** — Switch to bfloat16 — cleaner AMP, no scaler
6. **H3** — Explicit TF32 lines — documentation
7. **H6** — Checkpoint every 5 epochs — reduces Kaggle I/O

**None of these change the GCN/Transformer architecture or math.**

---
---

# Round 2 — T4 / P100 Hardware Review

Updated review for Kaggle **Dual T4 (2×16GB Turing)** or **Single P100 (16GB Pascal)**.

Hardware facts:
- **T4**: Turing arch, 2560 CUDA cores, 320 Tensor Cores, 16GB GDDR6, 300 GB/s bandwidth. Supports float16 Tensor Cores. No bfloat16 Tensor Cores. No TF32. `torch.compile` support is fragile.
- **P100**: Pascal arch, 3584 CUDA cores, **no Tensor Cores**, 16GB HBM2, 732 GB/s bandwidth. AMP float16 still helps via reduced memory bandwidth, but no hardware matmul acceleration.
- Both: 16GB VRAM per card vs 80GB on A100 — this is the critical constraint.

---

## What Was Correctly Updated from Round 1

| Change | Status | Notes |
|---|---|---|
| cuDNN benchmark | Applied (line 33) | Correct for fixed-size inputs |
| Vectorized eval loop | Applied (line 280) | Eliminates 325-iter Python loop |
| Checkpoint every 5 epochs | Applied (line 435) | Reduced disk I/O |
| Batch size 1024 + LR 1e-3 | Applied (line 305) | Correct linear scaling from 512/5e-4 |
| float16 + GradScaler kept | Correct | Right call for T4/P100 — bfloat16 and TF32 dropped appropriately |
| `torch.compile` omitted | Correct | Unreliable on Turing, unavailable on Pascal |

---

## T4/P100-Specific Issues

### T1. VRAM Budget Is Now Tight — batch_size=1024 May OOM on P100

On A100 (80GB), batch_size=1024 uses <1% of VRAM. On T4/P100 (16GB), the math changes:

| Component | Size |
|---|---|
| Model params (fp32 master) | ~16 MB |
| Model params (fp16 copy under AMP) | ~8 MB |
| Optimizer states (AdamW: 2× fp32 moments) | ~32 MB |
| Full dataset preloaded on GPU | **~950 MB** |
| Batch activations (1024 × GCN + Transformer) | **~800 MB – 1.2 GB** (estimated) |
| Gradient buffers | ~16 MB |
| CUDA context + fragmentation | ~1-2 GB |

**Estimated total: ~3-4 GB.** Should fit in 16GB, but activation memory is hard to predict exactly with einsum + 5D reshapes in the GCN blocks. AMP reduces some of it, but the peak is during the backward pass when both activations and gradients coexist.

**Recommendation**: Keep 1024, but add an OOM fallback note. If you get CUDA OOM, drop to 512 (and revert LR to 5e-4). On P100 specifically, the lack of Tensor Cores means larger batches don't accelerate matmul — they just reduce optimizer steps per epoch. The benefit is smaller than on T4.

---

### T2. GPU Preload — Still Fine, but Tighter

```python
full_ds.data = full_ds.data.to(device, non_blocking=True)
```

39K × 32 × 21 × 9 × 4 bytes = **~950 MB**. On a 16GB card, this is ~6% of VRAM — still manageable. Combined with the model and activations (~3-4 GB), you're at ~5 GB. Safe margin.

**No change needed.** The preload is still the right call — it eliminates CPU→GPU transfer bottlenecks during training.

---

### T3. P100 AMP: Helps Less Than You'd Think

On T4, AMP float16 hits Tensor Cores for matmul/conv → real 2-4× speedup on those ops.

On P100, there are **no Tensor Cores**. AMP float16 still helps by:
1. Halving memory bandwidth for activations (P100's HBM2 at 732 GB/s is already better than T4's GDDR6 at 300 GB/s)
2. Halving activation memory (lets you fit larger batches)
3. Some CUDA cores can process fp16 at 2× rate on Pascal (via `half2` packing)

But the speedup is ~1.3-1.5× on P100 vs ~2-3× on T4. Don't expect the same gains.

**No code change needed.** AMP + GradScaler is correct for both. Just calibrate expectations.

---

### T4. Dual T4 Multi-GPU — DataParallel vs DistributedDataParallel

If you want to use both T4s on Kaggle, you have two options:

**Option A: `nn.DataParallel` (Easy, Mediocre)**

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

- Splits each batch across 2 GPUs, gathers gradients on GPU 0
- Problem: GPU 0 does all the gradient aggregation → asymmetric VRAM usage. GPU 0 needs ~2× the memory of GPU 1 for gradients
- With your tiny model this probably won't OOM, but you lose ~30-40% of the theoretical 2× speedup to the synchronization overhead
- The `hasattr(model, 'module')` guard in `make_checkpoint` already handles this — good

**Option B: `DistributedDataParallel` (Better, More Code)**

```python
# Requires launching with torchrun or mp.spawn
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

- Each GPU has its own process, symmetric memory usage
- Needs `DistributedSampler` to replace your `WeightedRandomSampler` — this conflicts with your class-balanced sampling
- Needs `torch.distributed.init_process_group()` and `torchrun` launcher
- On Kaggle, the boilerplate setup for DDP on dual T4 is fragile and Kaggle's environment doesn't always play nice with `torchrun`

**My recommendation**: For your model size (~4M params, ~950 MB dataset), **stick to a single T4.** The overhead of multi-GPU synchronization on Kaggle dual-T4 is not worth it for a model this small. A single T4 will train this in minutes per epoch. The dual-T4 approach makes sense for models with 100M+ params or datasets that don't fit in one GPU's memory — neither applies here.

If you still want to try it, use `nn.DataParallel` (Option A) since it's a one-line change and your existing checkpoint code already handles it.

---

### T5. `make_checkpoint` Builds Dict Every Epoch Even When Not Saving — MINOR

```python
ckpt = make_checkpoint(...)  # Runs every epoch

if epoch % 5 == 0 or epoch == epochs:
    torch.save(ckpt, LAST_CKPT)  # Only saves sometimes
```

`make_checkpoint` calls `.state_dict()` three times (encoder, head, full model), which copies all parameters to CPU. This happens every epoch even when you don't save. On T4/P100 where every millisecond counts more than on A100:

```python
# Only build checkpoint when saving
should_save_last = (epoch % 5 == 0 or epoch == epochs)
if val_acc > best_acc or should_save_last:
    ckpt = make_checkpoint(...)
    if should_save_last:
        torch.save(ckpt, LAST_CKPT)
    if val_acc > best_acc:
        ...
        torch.save(ckpt, BEST_CKPT)
```

**No quality impact.** Avoids unnecessary `.state_dict()` copies on non-save epochs.

---

### T6. Label Parsing Fragility — BUG RISK (not hardware-specific)

```python
if '_signer_' in fname:
    label = fname.split('_signer_')[0]
else:
    label = fname.split('_')[0]
```

This appears in both `SignDataset.__init__` (line 196) and the label dictionary builder (line 333). If a label itself contains `_` (e.g., `thank_you`), the `fname.split('_')[0]` fallback returns just `thank` instead of `thank_you`.

Your extraction pipeline saves files as `{label}_{stem}_{hash}.npy`. If any ASL class has an underscore in its name, the first `_` split breaks.

**Check your label set.** If all 325 labels are single words (A-Z + common signs), this is fine. If any label contains underscores (compound words, phrases), this will silently produce wrong labels — the worst kind of bug because the script runs without errors but trains on corrupted labels.

**Recommended fix**: Use a delimiter that can't appear in labels, or store the label map alongside the .npy files during extraction so parsing isn't needed.

---

### T7. `online_augment` Creates 5 GPU Tensors Per Batch — MINOR

```python
R = _batch_rotation_matrices(B, rotation_deg, device)  # Allocates Rx, Ry, Rz, R
scale = scale_lo + torch.rand(B, 1, 1, 1, device=device) * ...
torch.randn_like(x) * noise_std
```

`_batch_rotation_matrices` allocates `angles`, `rad`, `cx`, `sx`, `cy`, `sy`, `cz`, `sz`, `zero`, `one`, `Rx`, `Ry`, `Rz`, and the final `R` — that's ~14 temporary tensors per batch. Plus `scale` and `randn_like`. All small tensors (batch-sized), but on T4/P100 the CUDA allocator overhead per tensor is proportionally more significant than on A100.

This is not a bottleneck worth refactoring — the augmentation is fast relative to the forward/backward pass. Just noting it for completeness.

---

### T8. No `torch.cuda.empty_cache()` After Dataset Load — MINOR

```python
full_ds.data = full_ds.data.to(device, non_blocking=True)
full_ds.targets = full_ds.targets.to(device, non_blocking=True)
```

The `.to(device)` creates new tensors on GPU and lets the CPU originals get garbage collected. But Python's GC may not immediately free the CPU-side pinned memory, and PyTorch's CUDA allocator may hold fragmented blocks. On 16GB cards:

```python
full_ds.data = full_ds.data.to(device, non_blocking=True)
full_ds.targets = full_ds.targets.to(device, non_blocking=True)
torch.cuda.empty_cache()
```

**Marginal.** Frees any fragmentation from the transfer. On 80GB A100 it's irrelevant; on 16GB T4 it could prevent a surprise OOM later during a peak activation allocation.

---

## My Own Suggestions

### S1. Gradient Accumulation for Effective Batch Size Decoupling

If batch_size=1024 OOMs on P100, instead of dropping to 512 and losing gradient stability, use gradient accumulation:

```python
accum_steps = 2  # effective batch = 512 * 2 = 1024
for i, (x, y) in enumerate(train_loader):
    x = online_augment(x.to(device))
    with torch.amp.autocast('cuda', enabled=use_amp):
        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing) / accum_steps
    scaler.scale(loss).backward()
    if (i + 1) % accum_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

This gives you effective batch_size=1024 with 512 actual VRAM usage. Keep LR at 1e-3.

**No quality impact** — mathematically equivalent to batch_size=1024 (ignoring batch norm, which you don't use — you use LayerNorm and GroupNorm, so this is exact).

---

### S2. Log GPU Memory Usage — Essential for T4/P100 Debugging

Add this inside the training loop to catch memory issues early:

```python
if epoch == 1 and train_total == y.size(0):  # After first batch
    log.info(f"GPU Mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

This tells you exactly how much headroom you have. On T4 you want to see <12 GB to have safe margin.

---

### S3. Mixup / CutMix Augmentation — Quality Improvement

Your current augmentation (rotation + scale + noise) operates in coordinate space, which is great. But for a 325-class classification task with only ~120 samples per class (39K / 325), adding **Mixup** can improve generalization significantly:

```python
def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mix, y_a, y_b, lam

# In training loop:
x, y_a, y_b, lam = mixup(online_augment(x), y)
loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)
```

Mixup in landmark space is physically meaningful — it interpolates between two hand poses, creating plausible intermediate shapes. This is different from image-space mixup where interpolation creates ghosting artifacts.

**This does change training behavior** (but not the model architecture). With only ~120 samples per class, regularization beyond label smoothing will likely help. Published GCN-based action recognition papers (ST-GCN, MS-G3D) commonly use Mixup.

---

### S4. Cosine Warmup Schedule — Double Check the Patience Interaction

Your scheduler does cosine decay from peak LR down to `min_lr_ratio * base_lr = 0.01 * 1e-3 = 1e-5`. With `patience=30`, early stopping triggers if val_acc doesn't improve for 30 epochs.

Potential scenario: Model hits a learning rate "saddle" during the cosine decay where the LR is too low to escape a local minimum, val_acc flatlines, and early stopping fires prematurely. This is especially common when the cosine enters its tail (last ~30% of epochs where LR is close to min).

**Two options**:
1. Increase patience to 40-50 if you see early stopping firing in the second half of training
2. Use cosine with restarts (`CosineAnnealingWarmRestarts`) to periodically "kick" the LR back up

Not a bug — just a training dynamics consideration to watch for in your logs.

---

### S5. The `_signer_` Split Problem Is an Extraction-Training Contract Gap

Your extraction pipeline (`extract_augment.py`) saves as `{label}_{stem}_{hash}.npy`. Your training script parses labels with `fname.split('_')[0]` (fallback case). These two must agree exactly.

If the label is `HELLO` and the video stem is `video_001`, the saved filename is `HELLO_video_001_a3f2c1.npy`. Parsing with `split('_')[0]` returns `HELLO` — correct.

But if the label is `THANK YOU` (with a space that got converted to underscore somewhere), it becomes `THANK_YOU_video_001_a3f2c1.npy`, and `split('_')[0]` returns `THANK` — wrong.

**Recommendation**: Save a `manifest.json` during extraction that maps filenames to labels explicitly. Load it in training instead of parsing filenames. This eliminates the parsing problem entirely and makes the contract explicit.

---

## Summary — T4/P100 Specific

| Issue | Severity | Quality Impact | Speed Impact |
|---|---|---|---|
| T1. batch_size=1024 VRAM risk | Watch | None if fits | OOM if doesn't |
| T2. GPU preload still fine | OK | — | — |
| T3. P100 AMP weaker than T4 | Info | None | Expect 1.3× not 3× |
| T4. Dual T4 multi-GPU | Info | None | Not worth it for 4M params |
| T5. Unnecessary state_dict on non-save epochs | Minor | None | Removes redundant copies |
| T6. Label parsing with `_` split | **Bug Risk** | Silent mislabeling if compound labels exist | — |
| T7. Augment tensor allocations | Info | None | Negligible |
| T8. empty_cache after preload | Minor | None | Reduces fragmentation on 16GB |

| My Suggestion | Type | Quality Impact |
|---|---|---|
| S1. Gradient accumulation | Fallback for OOM | None (mathematically equivalent) |
| S2. Log GPU memory | Debugging | None |
| S3. Mixup augmentation | **Quality improvement** | Better generalization with 120 samples/class |
| S4. Watch patience vs cosine tail | Training dynamics | Prevent premature early stopping |
| S5. Manifest file for labels | Robustness | Eliminates parsing ambiguity |

---
---

# Round 3 — Implementation Review of S1-S5 Changes

Final pass on `train_stage_1.py` after user implemented Round 2 suggestions.

Changes claimed:
1. Gradient accumulation (`accum_steps=2`, batch_size=512, effective=1024)
2. `nn.DataParallel` for dual T4
3. Mixup augmentation (`alpha=0.2`)
4. Deferred `state_dict()` generation
5. Manifest-based label parsing with `rsplit` fallback
6. `empty_cache()` + memory logger

Patience bumped from 30 → 40 (addresses S4). Train accuracy logging removed (correct — meaningless under Mixup).

---

## What's Correctly Implemented

| Change | Status |
|---|---|
| Gradient accumulation loop structure | **Correct** — `zero_grad` at epoch start + inside accumulation gate |
| `loss / accum_steps` scaling | **Correct** — mathematically equivalent to full batch |
| `epoch_loss += loss.item() * accum_steps` | **Correct** — reconstructs unscaled loss for logging |
| Tail batch handling `(i + 1) == len(train_loader)` | **Correct** — flushes gradients from residual micro-batches |
| Mixup function | **Correct** — `lam * CE(logits, y_a) + (1-lam) * CE(logits, y_b)` is standard |
| `alpha <= 0` guard | **Correct** — returns identity (lam=1.0), effectively disabling Mixup |
| `model = nn.DataParallel(model)` placement | **Correct** — after construction, before optimizer. Optimizer wraps DP parameters |
| `hasattr(model, 'module')` in `make_checkpoint` | **Correct** — already handled from v1 |
| `empty_cache()` after preload | **Correct** |
| Memory logger at epoch 1, batch 0 | **Correct** |
| Patience → 40 | **Good** — addresses S4 cosine tail concern |
| Train accuracy removed from logging | **Correct** — meaningless under Mixup (soft targets vs hard labels) |
| Manifest-first label parsing | **Correct** — clean fallback chain |
| Deferred `make_checkpoint` | **Correct** — only builds when saving |

---

## Issues Found

### R3-1. `rsplit('_', 2)` Fallback Is Still Ambiguous — MINOR (manifest solves it)

```python
parts = fname.rsplit('_', 2)
label = parts[0] if len(parts) >= 3 else fname.split('_')[0]
```

For `HELLO_video_001_a3f2c1.npy`:
- `rsplit('_', 2)` → `['HELLO_video', '001', 'a3f2c1.npy']`
- `parts[0]` → `HELLO_video` — **wrong**, label is `HELLO`

The `rsplit` slices off the hash and part of the stem, but can't distinguish where the label ends and the stem begins if either contains underscores.

**However**: since the manifest.json path is checked first, this fallback only fires when there's no manifest. And in practice, your extraction pipeline uses directory names as labels (single words like `A`, `B`, `HELLO`), so `rsplit('_', 2)[0]` actually gives a better result than the old `split('_')[0]` for multi-word stems.

**RECOMMENDATION**: Generate the `manifest.json` during extraction so this fallback never fires. Add to `extract_augment.py` `run_pipeline()`:

```python
manifest = {}
# ... inside the save logic of process_single_video:
#     manifest[save_name + '.npy'] = label
# After all processing:
with open(out_path / 'manifest.json', 'w') as f:
    json.dump(manifest, f)
```

Since workers run in parallel, you'd need to collect manifest entries after all workers finish. Simplest approach: build it from the saved filenames + directory structure post-hoc.

---

### R3-2. DataParallel + Gradient Accumulation = Double Sync — MINOR

```python
model = nn.DataParallel(model)  # line 394
...
scaler.scale(loss).backward()   # line 439 — triggers AllReduce on EVERY backward
```

`nn.DataParallel` synchronizes (gathers) gradients after every `.backward()` call. With `accum_steps=2`, this means:
- Micro-step 1: forward on 2 GPUs → backward → **sync gradients across GPUs**
- Micro-step 2: forward on 2 GPUs → backward → **sync gradients across GPUs**
- Then: optimizer step

You're syncing twice per effective step instead of once. The "correct" approach for accumulated gradients with multi-GPU is DDP with `no_sync()` context manager — but DataParallel doesn't support that.

**Impact**: For your 4M-param model, gradient sync takes ~1ms. Doubling it adds ~1ms per step. Over 200 epochs × ~54 steps/epoch = ~11 seconds total. **Negligible.** The gradients are still mathematically correct — you just pay a tiny sync overhead.

Not worth switching to DDP for this. Just noting it.

---

### R3-3. Resume Loads `model_state_dict` Which May Be DataParallel-Wrapped — BUG

```python
model = nn.DataParallel(model)    # line 394 — wraps model, keys become "module.encoder..."
...
ckpt = torch.load(LAST_CKPT, ...)
model.load_state_dict(ckpt['model_state_dict'])  # line 404
```

The checkpoint at line 305 saves via:
```python
unwrapped = model.module if hasattr(model, 'module') else model
'model_state_dict': unwrapped.state_dict()  # Saves WITHOUT "module." prefix
```

But at line 404, you're loading into the **DataParallel-wrapped** model (wrapped at line 394), whose keys have the `module.` prefix. The unwrapped state dict won't match.

**This will crash on resume with dual T4.**

Fix options:
1. **Move DataParallel wrapping after checkpoint loading** (simplest)
2. Or load into the unwrapped model, then wrap

The current code has DataParallel at line 394 and checkpoint load at line 401-409. Just swap the order:

```python
# First: create model and load checkpoint (if any)
model = SLTStage1(...)
# ... load checkpoint into unwrapped model ...
model.to(device)
# THEN wrap
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

---

### R3-4. `np.random.beta` in Mixup Uses Global RNG — MINOR

```python
lam = np.random.beta(alpha, alpha)  # line 263
```

This uses NumPy's global RNG, not a seeded local one. Across DataParallel replicas this is fine (Mixup runs on CPU before the model), but it means Mixup is non-reproducible across runs. For research reproducibility, consider:

```python
lam = torch.distributions.Beta(alpha, alpha).sample().item()
```

Or seed NumPy explicitly at the start. **Not a correctness issue**, just reproducibility.

---

### R3-5. Mixup + Label Smoothing Interaction — INFO (verify intent)

You're applying both:
- Label smoothing: `label_smoothing=0.05` → soft targets → `(1-0.05)*one_hot + 0.05/K`
- Mixup: `lam * CE(logits, y_a) + (1-lam) * CE(logits, y_b)` → already blends targets

Both are regularizers that soften the target distribution. Using both together is common and generally fine, but with `alpha=0.2` (which produces `lam` close to 0 or 1 most of the time due to U-shaped Beta distribution) and `label_smoothing=0.05` (very mild), the interaction is minimal.

If you see training loss plateauing too high (>1.5 after warmup) or val accuracy stalling below expected, try reducing `label_smoothing` to 0.02 or 0.0 since Mixup already provides the regularization.

---

### R3-6. Deferred Checkpoint Logic Has a Subtle `trigger_times` Counting Issue — BUG

```python
if val_acc > best_acc or should_save_last:
    ckpt = make_checkpoint(...)
    if should_save_last:
        torch.save(ckpt, LAST_CKPT)
    if val_acc > best_acc:
        best_acc, trigger_times = val_acc, 0
        ...
    else:
        trigger_times += 1    # ← HERE (inside the outer if)
else:
    trigger_times += 1        # ← AND HERE (outside)
```

The `trigger_times += 1` in the inner `else` (line 479) fires when `should_save_last` is True but `val_acc <= best_acc`. The outer `else` (line 482) fires when both are False. This is actually **correct** — in every case where `val_acc <= best_acc`, trigger_times increments exactly once. But the branching is confusing.

Wait — there's actually a real issue. Consider epoch 5 (a save epoch) where `val_acc > best_acc`:
- Enters the outer `if` (True because `val_acc > best_acc`)
- Builds checkpoint, saves LAST_CKPT
- Enters inner `if val_acc > best_acc` → sets `trigger_times = 0` ✓

Now epoch 6 (not a save epoch) where `val_acc <= best_acc`:
- Outer condition: `val_acc > best_acc` is False, `should_save_last` is False → enters `else`
- `trigger_times += 1` ✓

Epoch 10 (save epoch) where `val_acc <= best_acc`:
- Outer condition: `val_acc > best_acc` is False, `should_save_last` is True → enters outer `if`
- `should_save_last` → saves LAST_CKPT ✓
- `val_acc > best_acc` is False → enters inner `else` → `trigger_times += 1` ✓

**This is correct.** Every non-improvement epoch increments trigger_times exactly once. The logic is just a bit convoluted to read. No bug.

---

## My Own Suggestions — Round 3

### S6. Add `manifest.json` Generation to extract_augment.py

The training script now supports manifest-first parsing, but the extraction script doesn't generate one yet. This is the missing piece. Simplest post-hoc approach since workers are parallel:

```python
# At the end of run_pipeline(), after all workers finish:
manifest = {}
for f in os.listdir(out_path):
    if f.endswith('.npy'):
        # Reconstruct label from the directory structure using done_files logic
        # or simply walk raw_video_dir again and match
        pass
with open(out_path / 'manifest.json', 'w') as f:
    json.dump(manifest, f)
```

Or simpler: since you know the label is `Path(root).name` from extraction, and the saved filename is `{label}_{stem}_{hash}.npy`, just parse with the known label set:

```python
manifest = {}
for f in os.listdir(out_path):
    if not f.endswith('.npy'): continue
    for label in sorted(unique_labels, key=len, reverse=True):  # longest first
        if f.startswith(label + '_'):
            manifest[f] = label
            break
with open(out_path / 'manifest.json', 'w') as f:
    json.dump(manifest, f)
```

---

### S7. Temporal Augmentation — Untapped Quality Win

Your current augmentations operate in spatial dimensions only:
- **Rotation**: 3D rotation of xyz/vel/acc (spatial)
- **Scale**: Uniform scaling (spatial)
- **Noise**: Gaussian noise (spatial)
- **Mixup**: Blends two samples (spatial + temporal, but linear)

You're missing **temporal augmentations**, which are important for sequence models:

**Time Warp** — slight non-linear stretching/compression of the time axis:
```python
def time_warp(x, sigma=0.1):
    B, T, N, C = x.shape
    # Create warped time indices
    t = torch.linspace(0, 1, T, device=x.device)
    warp = torch.cumsum(torch.randn(B, T, device=x.device) * sigma, dim=1)
    warp = warp - warp.mean(dim=1, keepdim=True)  # Zero-mean
    t_warped = t.unsqueeze(0) + warp
    t_warped = t_warped.clamp(0, 1) * (T - 1)
    # Interpolate along time
    idx_lo = t_warped.long().clamp(0, T - 2)
    idx_hi = idx_lo + 1
    alpha = (t_warped - idx_lo.float()).unsqueeze(-1).unsqueeze(-1)
    x_lo = torch.gather(x, 1, idx_lo.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, C))
    x_hi = torch.gather(x, 1, idx_hi.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, C))
    return (1 - alpha) * x_lo + alpha * x_hi
```

This simulates natural speed variations in signing (people don't sign at constant speed). Published in ST-GCN augmentation pipelines and is especially important for 32-frame fixed-length inputs where the temporal resampling already quantizes the original timing.

**Frame Drop** — randomly zero out 1-2 frames (the mask channel handles it):
```python
def frame_drop(x, drop_prob=0.05):
    B, T, N, C = x.shape
    mask = (torch.rand(B, T, 1, 1, device=x.device) > drop_prob).float()
    return x * mask
```

Both are zero-overhead on VRAM and add temporal diversity the model currently doesn't see.

---

### S8. Consider Stratified Split Instead of Random Split

```python
train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size], ...)
```

With 325 classes and ~120 samples each, a pure random 70/30 split could leave some classes with very few validation samples (or none for tiny classes). `random_split` doesn't guarantee per-class proportions.

For a proper stratified split:

```python
from sklearn.model_selection import train_test_split
train_idx, val_idx = train_test_split(
    range(len(full_ds)), test_size=0.30,
    stratify=full_ds.targets.numpy(), random_state=42
)
train_ds = torch.utils.data.Subset(full_ds, train_idx)
val_ds   = torch.utils.data.Subset(full_ds, val_idx)
```

This guarantees every class has exactly 70/30 representation. With the WeightedRandomSampler already handling class imbalance during training, this ensures the validation set is also balanced for fair per-class evaluation.

Adds one import (`sklearn.model_selection`), but Kaggle has it pre-installed.

---

## T4 vs P100 Recommendation

**Use the Dual T4 (single GPU mode)** — meaning select the T4×2 accelerator but only use one GPU.

| Factor | Single T4 | Single P100 |
|---|---|---|
| Tensor Cores | **Yes** (320 Turing) | No |
| float16 matmul speedup | **2-3× via Tensor Cores** | ~1.3× via half2 packing |
| VRAM | 16 GB GDDR6 | 16 GB HBM2 |
| Memory bandwidth | 300 GB/s | **732 GB/s** |
| Your model bottleneck | Compute (einsum, matmul) | Compute (einsum, matmul) |
| AMP benefit | **High** | Moderate |

Your model is **compute-bound**, not memory-bandwidth-bound. The GCN einsum, Transformer attention matmul, and Linear layers all benefit from Tensor Cores. The T4's 320 Tensor Cores crush the P100's brute-force CUDA cores for float16 matmul even though the P100 has more raw cores (3584 vs 2560).

**The P100's advantage (HBM2 bandwidth) only matters for memory-bound workloads** like large embedding lookups, sparse attention, or models with huge batch activations. Your model's activations fit in L2 cache.

Why not use both T4s? `nn.DataParallel` adds sync overhead, and your model is tiny. One T4 will finish each epoch in seconds. The second T4 just wastes your Kaggle quota time.

**Bottom line: Single T4 > P100 > Dual T4 with DataParallel for this specific workload.**

---

## Round 3 Summary

| Issue | Severity | Quality Impact |
|---|---|---|
| R3-1. `rsplit` fallback ambiguity | Minor | None if manifest exists |
| R3-2. DataParallel double sync | Negligible | None (correct gradients, +11s total) |
| R3-3. Resume crash with DataParallel | **BUG** | Crashes on resume. Move DP wrap after load |
| R3-4. Non-reproducible Mixup RNG | Minor | Non-deterministic runs |
| R3-5. Mixup + label smoothing | Info | Reduce smoothing if loss plateaus |
| R3-6. trigger_times counting | OK | Verified correct despite complex branching |

| My Suggestion | Type | Quality Impact |
|---|---|---|
| S6. Generate manifest.json in extraction | Robustness | Eliminates label parsing forever |
| S7. Time warp + frame drop augmentation | **Quality improvement** | Temporal diversity for sequence model |
| S8. Stratified split | **Quality improvement** | Guarantees per-class 70/30 balance |

### Critical Fix (Must Do Before Running)

**R3-3**: Move `nn.DataParallel` wrapping to AFTER checkpoint loading. Current code will crash on resume when using dual T4.

---
---

# Round 3 — Claude's Independent Assessment

My own pass over the same code, confirming what's correct and flagging what the above review missed.

---

## Confirming Existing R3 Findings

| Finding | My Verdict |
|---|---|
| R3-1. `rsplit` ambiguity | **Confirmed.** `"HELLO_video_001_a3f2c1.npy".rsplit('_', 2)` → `['HELLO_video', '001', 'a3f2c1.npy']` → label = `HELLO_video` (wrong). Manifest solves it |
| R3-2. DataParallel double sync | **Confirmed.** Negligible overhead (~11s total). Correct gradients |
| R3-3. Resume crash with DataParallel | **Confirmed. CRITICAL.** unwrapped keys vs `module.`-prefixed keys mismatch |
| R3-4. Unseeded NumPy RNG | **Confirmed.** Minor reproducibility issue |
| R3-5. Mixup + label smoothing | **Confirmed.** Both mild, interaction is fine |
| R3-6. trigger_times counting | **Confirmed correct.** Every non-improvement epoch increments exactly once |
| S6. Manifest in extraction | **Agreed** |
| S7. Temporal augmentation | **Agreed.** Time warp would help |
| S8. Stratified split | **Agreed.** `random_split` doesn't guarantee per-class proportions |

---

## What R3 Above Missed

### R3-7. Early Stopping Doesn't Save LAST_CKPT — MINOR

```python
if trigger_times >= patience:
    log.info(f"🛑 CONVERGENCE REACHED...")
    break  # ← No save before break
```

If early stopping fires at epoch 47 (not divisible by 5), `LAST_CKPT` is from epoch 45. On resume with different hyperparameters, you restart from epoch 45 instead of 47.

Not critical — the BEST_CKPT is always up to date, and the lost epochs had no improvement. But for clean resumability:

```python
if trigger_times >= patience:
    ckpt = make_checkpoint(...)
    torch.save(ckpt, LAST_CKPT)
    log.info(f"🛑 CONVERGENCE REACHED...")
    break
```

---

### R3-8. Gradient Accumulation + Mixup Interaction — VERIFIED CORRECT

This was the user's primary concern, so confirming explicitly:

With `accum_steps=2`, each micro-batch independently samples its own `lam` from `Beta(0.2, 0.2)`. The accumulated gradient is the mean of two micro-batches with different lambdas. This is **mathematically sound** — it doesn't distort the expected gradient direction. It adds a tiny amount of inter-microbatch variance, which is actually beneficial for regularization.

**Key reasoning**: Your model uses `LayerNorm` and `GroupNorm` (not `BatchNorm`). Gradient accumulation is EXACT with LayerNorm because the normalization statistics are per-sample, not per-batch. With BatchNorm, the statistics would differ between micro-batches, making accumulation an approximation. You don't have that problem.

---

### R3-9. Mixup on Already-Augmented Data — Order Is Correct

```python
x = online_augment(x)                              # line 427
x, y_a, y_b, lam = apply_mixup(x, y, alpha=...)    # line 430
```

The augment→mixup order is correct. The alternative (mixup→augment) would apply the SAME rotation to both mixed components, which reduces augmentation diversity. Current order: sample A gets rotation R_A, sample B gets rotation R_B, then they're mixed. This maximizes diversity.

With `rotation_deg=10.0` and `Beta(0.2, 0.2)` (lam near 0 or 1 ~70% of the time), the angular mismatch between mixed samples is small and benign.

---

### R3-10. `head_dropout=0.15` vs ClassifierHead Default `0.4` — INFO

```python
# ClassifierHead.__init__ default:
def __init__(self, d_model=256, num_classes=29, dropout=0.4):

# train() passes:
head_dropout=0.15
```

The effective head dropout is 0.15 (explicitly passed). With Mixup + label smoothing now providing regularization, 0.15 is appropriate. If overfitting persists after 40-50 epochs, try 0.25. If underfitting (train loss > 1.5 after warmup), try 0.10.

---

## T4 vs P100 — My Recommendation

**Use Single T4. Strongly agree with the R3 recommendation above.**

The detailed reasoning:

Your model's forward pass is dominated by:
1. `torch.einsum('knm,btnc->kbtnc', A, x)` — GCN aggregation (matmul-like)
2. `nn.Linear` projections inside GCN blocks (64→192, 192→128, etc.)
3. `nn.TransformerEncoderLayer` with d_model=256, nhead=8, feedforward=1024

All of these are **matmul-heavy** ops. T4's Tensor Cores accelerate fp16 matmul by 2-3× over its own CUDA cores. P100 has no Tensor Cores — it runs fp16 matmul on regular CUDA cores with at most 2× throughput via half2 vector ops.

| | Single T4 | P100 | Dual T4 (DataParallel) |
|---|---|---|---|
| Est. epoch time | **~3-5 sec** | ~6-10 sec | ~4-6 sec (sync overhead) |
| VRAM headroom | ~11 GB free | ~11 GB free | ~11 GB free per card |
| Code complexity | Simple | Simple | Resume bug, double sync |
| Kaggle GPU quota usage | 1 GPU-hour | 1 GPU-hour | **2 GPU-hours** (burns double quota) |

The last row matters: **Kaggle charges quota per GPU.** Dual T4 burns 2× the quota for ~1.2× the speed. Single T4 is strictly better on cost-efficiency.

**Bottom line: Select "GPU T4 x2" on Kaggle (it's often easier to get than P100), but only use one GPU. The `nn.DataParallel` code can stay guarded but you should not deliberately select dual-GPU mode.**

---

## Round 3 Final Verdict

The script is **production-ready for single T4** with one mandatory fix:

1. **R3-3 (MUST FIX if ever using dual T4)**: Move `nn.DataParallel` wrap to after checkpoint loading. Or simply don't use dual T4 — single T4 is better anyway.

Everything else is correct. The gradient accumulation + mixup interaction is mathematically sound. The deferred checkpointing logic (despite complex branching) increments trigger_times correctly. The manifest-first label parsing is robust.

Quality-wise, the augmentation stack (rotation + scale + noise + mixup) is solid for Stage 1. For Stage 2, consider adding temporal augmentation (S7) and stratified splitting (S8).

---
---

# Round 4 — Final Review After R3-3, R3-4, and Strict Manifest Fixes

User applied three fixes from Round 3. This is the final review pass.

---

## Fixes Verified

### R3-3 Fix: DataParallel After Checkpoint Load — CORRECT

```python
# Line 370: Model created
model = SLTStage1(...)
# Line 371: Optimizer created with unwrapped model params
optimizer = optim.AdamW(model.parameters(), ...)
# Line 378-387: Checkpoint loaded into UNWRAPPED model
if LAST_CKPT.exists():
    ckpt = torch.load(LAST_CKPT, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    ...
# Line 389: Model moved to device
model.to(device)
# Line 390-392: Optimizer states moved to device
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor): state[k] = v.to(device)
# Line 395-397: THEN DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

The full sequence is now correct:
1. Create model (unwrapped)
2. Create optimizer pointing to unwrapped params
3. Load checkpoint state dicts into unwrapped model + optimizer
4. Move model to device
5. Move optimizer states to device
6. Wrap in DataParallel

The optimizer state `.to(device)` loop at lines 390-392 is a nice touch — checkpoint loads with `map_location='cpu'`, so optimizer momentum buffers need explicit device transfer. This was not flagged in prior rounds but is correctly handled.

**Status: FIXED. No issues.**

---

### R3-4 Fix: PyTorch Beta for Mixup RNG — CORRECT

```python
def apply_mixup(x, y, alpha=0.2):
    if alpha <= 0: return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam
```

Uses `torch.distributions.Beta` instead of `np.random.beta`. When combined with `torch.manual_seed()` at the DataLoader level, this makes mixup reproducible within PyTorch's RNG stream.

One nuance: `torch.distributions.Beta` creates a new Distribution object every call. This is fine — the object is lightweight (just stores alpha/beta tensors). No caching needed for this call frequency (~54 calls/epoch).

**Status: FIXED. No issues.**

---

### Strict Manifest Loading — CORRECT

```python
# Lines 329-342
manifest_path = Path(data_path) / 'manifest.json'
if not manifest_path.exists():
    raise FileNotFoundError(f"CRITICAL: manifest.json not found in {data_path}. Upload it first!")

with open(manifest_path, 'r') as f:
    manifest = json.load(f)

unique_labels = sorted(list(set(manifest.values())))
label_to_idx = {label: i for i, label in enumerate(unique_labels)}
```

And in `SignDataset.__init__`:

```python
for fname in sorted(f for f in os.listdir(data_path) if f.endswith('.npy')):
    if fname not in manifest:
        skipped += 1
        continue
    label = manifest[fname]
```

All filename-based parsing (`split('_')`, `rsplit('_', 2)`, `_signer_` heuristic) has been completely removed. The manifest is the single source of truth. Files not in the manifest are cleanly skipped with a counter.

The `FileNotFoundError` is the right exception — it's specific, immediately actionable, and stops training before wasting any GPU time. The error message tells the user exactly what to do.

**Status: FIXED. Eliminates T6, R3-1, and S5 permanently.**

---

## Full Script Audit — New Issues

### R4-1. Scheduler Not Saved/Restored in Early Stop Path — MINOR (Inherited from R3-7)

R3-7 noted that early stopping doesn't save `LAST_CKPT`. This is still present:

```python
if trigger_times >= patience:
    log.info(f"🛑 CONVERGENCE REACHED: Early stopping triggered at Epoch {epoch}.")
    break  # No checkpoint save before break
```

If you resume after early stopping (e.g., with increased patience or different hyperparams), `LAST_CKPT` could be up to 4 epochs behind. The `BEST_CKPT` is always current.

**Impact**: Minimal — if you resume, it costs a few redundant epochs. Not fixing this is a reasonable tradeoff for code simplicity.

---

### R4-2. `cache_path` Doesn't Include Manifest Hash — LATENT BUG

```python
cache_path = str(save_dir / 'ds_cache.pt')
```

The dataset cache at `ds_cache.pt` is keyed by `label_to_idx` comparison:

```python
if cache.get('label_to_idx') == label_to_idx:
    self.data, self.targets = cache['data'], cache['targets']
```

If you update `manifest.json` (add/remove files) but the label set stays the same, the cache still passes the `label_to_idx` check — but contains stale data from the old manifest. You'd train on the old dataset without realizing it.

**Fix options** (pick one):
1. Hash the manifest and include it in the cache validation:
   ```python
   manifest_hash = hashlib.md5(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:8]
   # Store in cache and compare on load
   ```
2. Delete cache when manifest changes (manual, error-prone)
3. Include file count in cache validation (crude but catches most changes):
   ```python
   if cache.get('label_to_idx') == label_to_idx and cache.get('num_files') == len(manifest):
   ```

**Severity**: Low on first run (no cache exists). Higher if you iterate on dataset versions on Kaggle across sessions. Option 3 is the simplest guard.

---

### R4-3. `generated_label_map.json` Written Every Run — HARMLESS

```python
with open(save_dir / 'generated_label_map.json', 'w') as f:
    json.dump({'label_to_idx': label_to_idx}, f, indent=2)
```

This overwrites every run, even on resume. No harm since `label_to_idx` is deterministic from `sorted(unique_labels)`, but it means the file's timestamp always reflects the latest run, not the original training. Purely cosmetic.

---

### R4-4. `weights_only=True` for Cache, `weights_only=False` for Checkpoint — CORRECT

```python
# Cache load (line 181):
cache = torch.load(cache_path, weights_only=True)

# Checkpoint load (line 380):
ckpt = torch.load(LAST_CKPT, map_location='cpu', weights_only=False)
```

Cache only contains tensors and dicts → `weights_only=True` is correct and safe.

Checkpoint contains scheduler state (which has non-tensor Python objects like `last_epoch`) → `weights_only=False` is required. This is the correct pattern.

---

### R4-5. `sampler_temperature=0.5` Interaction with Mixup — INFO

The WeightedRandomSampler with temperature 0.5 upsamples rare classes (square root of inverse frequency). Mixup then blends these oversampled rare-class examples with other samples.

This is a benign interaction — Mixup's regularization effect counterbalances the sampler's oversampling bias. With `Alpha=0.2` (U-shaped beta), most samples are only lightly mixed (lam > 0.9 about 40% of the time), so rare-class identity is preserved.

If validation accuracy for rare classes lags despite overall high accuracy, try:
- Increasing `sampler_temperature` to 0.7 (more aggressive oversampling)
- Reducing `mixup_alpha` to 0.1 (less blending of rare-class samples)

Not a bug — just a hyperparameter interaction to watch.

---

## Round 4 Summary

| Item | Status | Severity |
|---|---|---|
| R3-3 Fix (DataParallel after load) | **VERIFIED CORRECT** | — |
| R3-4 Fix (PyTorch Beta RNG) | **VERIFIED CORRECT** | — |
| Strict manifest loading | **VERIFIED CORRECT** | — |
| R4-1. No LAST_CKPT on early stop | Inherited (R3-7) | Minor |
| R4-2. Cache doesn't validate manifest changes | New | **Low-Medium** |
| R4-3. label_map overwritten every run | New | Harmless |
| R4-4. weights_only usage | Correct | — |
| R4-5. Sampler × Mixup interaction | Info | — |

---

## My Suggestions — Round 4

### S9. EMA (Exponential Moving Average) Weights

For small datasets (~120 samples/class), EMA of model weights often outperforms the final/best checkpoint by 0.5-1.5% accuracy. The idea: maintain a slow-moving average of model parameters that smooths out SGD noise.

```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters()}
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            self.shadow[n].mul_(self.decay).add_(p, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {n: p.clone() for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            p.data.copy_(self.backup[n])
```

Usage: call `ema.update(model)` after each optimizer step, `ema.apply(model)` before validation, `ema.restore(model)` after validation. Save `ema.shadow` in the best checkpoint.

Cost: ~16 MB extra memory (one copy of all parameters). Negligible on T4.

**This does not change the architecture.** It's a training technique that produces better weight values within the same model structure.

---

### S10. Validation Top-K Accuracy Logging

Currently you only log top-1 accuracy. For a 325-class problem, top-5 accuracy provides a much better signal of early training progress:

```python
# Inside evaluate():
_, top5_preds = logits.topk(5, dim=1)
top5_correct += (top5_preds == y.unsqueeze(1)).any(dim=1).sum().item()
```

In the first 5-10 epochs, top-1 accuracy might be <5% (hard to distinguish signal from noise), while top-5 might already be 15-20% (clear evidence the model is learning).

One extra line in the eval loop, one extra number in the log. No quality impact — purely informational.

---

### S11. Smoke-Test Mode with 2-3 Epoch Dry Run

Add a `--smoke` or `smoke_test=True` parameter that runs 2-3 epochs on a tiny subset (500 samples) to verify:
- No OOM on target hardware
- Checkpoint save/load round-trips correctly
- AMP/GradScaler doesn't produce NaN
- Manifest loading works

```python
if smoke_test:
    epochs, patience = 3, 3
    # Subsample dataset
    subset_idx = list(range(min(500, len(full_ds))))
    full_ds = torch.utils.data.Subset(full_ds, subset_idx)
```

This is invaluable on Kaggle where a 200-epoch run takes significant quota. A 30-second smoke test catches configuration errors before committing hours of GPU time.

---

## Final Verdict

**The script is production-ready for Kaggle T4.** All critical bugs from previous rounds are fixed. The three remaining items (R4-1, R4-2, R4-5) are minor and don't affect correctness for your first training run.

**Priority if you want to squeeze more quality:**
1. **S8 (Stratified split)** — one `sklearn` import, guarantees per-class balance in val set
2. **S10 (Top-5 logging)** — one line, much better early-training signal
3. **S9 (EMA)** — 0.5-1.5% accuracy boost for free
4. **S11 (Smoke test)** — save Kaggle quota on config errors

The model architecture, loss computation, gradient accumulation, mixup, augmentation pipeline, and checkpoint logic are all verified correct.
