# Stage 2 Inference Review — SLT_test.py vs train_stage_2.py

**Date**: Round 1
**Files compared**:
- `test/SLT_test.py` (inference)
- `src/train_stage_2.py` (training)

**Symptom**: Model hallucinates during inference despite good training WER. One word ("DOCTOR") decoded correctly.

---

## INF-1: CRITICAL BUG — Geo features zeroed out

| | Training (`train_stage_2.py:140-146`) | Inference (`SLT_test.py:82-83`) |
|---|---|---|
| **geo features** | `self._compute_geo_features(xyz)` — 24 real values (5 tip distances + 5 curl ratios + 2 cross features, per hand) | `torch.zeros(B, T, 24)` — all zeros |
| **geo_proj input** | `cat([GCN_out_256, real_geo_24])` → 280-dim | `cat([GCN_out_256, zeros_24])` → 280-dim with dead channels |

The `geo_proj` layer (`Linear(280, 256)`) was trained to use those 24 geometric channels as meaningful spatial signal. Zeroing them shifts the entire encoder output distribution. Everything downstream (temporal pool, LSTM, classifier) receives features it never saw during training.

**This is almost certainly the root cause of hallucination.**

Why "DOCTOR" still works: it's likely a highly distinctive hand shape where the GCN output alone (without geo features) happens to produce a sufficiently unique representation.

**Fix**: Copy `_compute_geo_features`, `_geo_dist`, and all finger-joint constants (`_THUMB_MCP`, `_INDEX_TIP`, etc.) from `train_stage_2.py:88-138` into the inference encoder. Replace lines 82-83 with the real computation.

---

## INF-2: MEDIUM — Non-EMA weights loaded

Training saves checkpoint **after** `ema.restore(model)` (line 519), so `model_state_dict` = raw training weights. EMA shadow is stored separately in `ema_shadow`.

Inference loads `checkpoint['model_state_dict']` — the raw weights.

EMA weights (decay=0.999) are typically smoother and generalize better. This won't cause hallucination on its own, but once INF-1 is fixed, loading EMA weights should give cleaner predictions.

**Fix**: After `model.load_state_dict(...)`, overwrite trainable params from `checkpoint['ema_shadow']`:
```python
ema_shadow = checkpoint.get('ema_shadow')
if ema_shadow:
    for name, param in model.named_parameters():
        if name in ema_shadow:
            param.data.copy_(ema_shadow[name])
```

---

## INF-3: INFO — Adjacency matrix simplified (not a runtime bug)

Inference `build_adjacency_matrices` returns 3 identity matrices (no graph edges). However, `A` is a `register_buffer`, so `load_state_dict` correctly loads the real adjacency from the checkpoint. The identity matrices are immediately overwritten. **No runtime impact**, but the code is misleading.

---

## INF-4: INFO — Input shape `[1, T, 42, 10]` is correct

The inference forward reshapes `T // 32` clips internally, same as training. For single-sample inference with T as a multiple of 32, the shape `[1, T, 42, 10]` is exactly what the model expects. No need for `[1, S, 32, 42, 10]`.

The padding/trimming logic at lines 142-145 correctly ensures T is a multiple of 32.

---

## INF-5: INFO — LSTM packing vs no packing (equivalent for B=1)

Training uses `pack_padded_sequence` → LSTM → `pad_packed_sequence`. Inference feeds the sequence directly. For a single unpadded sample, these are mathematically identical.

---

## Summary: Priority fixes

| # | Severity | Fix |
|---|----------|-----|
| INF-1 | **CRITICAL** | Restore `_compute_geo_features` with all finger-joint constants. This alone should eliminate hallucination. |
| INF-2 | MEDIUM | Load `ema_shadow` instead of `model_state_dict` for trainable params. |
| INF-3 | INFO | Cosmetic — no action needed. |
