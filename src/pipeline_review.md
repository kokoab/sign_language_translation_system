# Pipeline Review: `test_video_pipeline.py`

## 1. `extract_features_from_video` — Feature Extraction Logic

### Frame Collection Loop (Lines 62-91)
Correct. All raw frames are collected into `frames_xyz` and `frames_mask` before any derivative computation. MediaPipe Holistic is used with `model_complexity=2` and both hands are extracted into 42 landmarks.

### Velocity / Acceleration (Lines 98-108)
Correct. Derivatives are computed on the **entire** stacked sequence before windowing:
- `vel[0]` and `acc[0]` are zero-vectors (no prior frame), which is the right convention.
- Final feature shape per frame: `[42, 10]` = xyz(3) + vel(3) + acc(3) + mask(1). Matches Stage 1/2 `in_channels=10`.

### Sliding Window (Lines 110-129)
Correct. `window_size=32`, `stride=16` (50% overlap).
- Short-video fallback (< 32 frames) uses `np.pad(..., mode='edge')` which is reasonable.
- Final return shape: `[1, Num_Windows, 32, 42, 10]` — the `unsqueeze(0)` adds the batch dim. This is correct.

**One note:** tail frames that don't fill a full window are silently dropped by the `range(0, len - 32 + 1, 16)` loop. This is standard and acceptable, but very short residual segments at the end of a video won't be represented.

---

## 2. `run_stage2_recognition` — CTC Decoding (Lines 134-156)

### Model Call Mismatch (BUG)
The training `SLTStage2CTC.forward()` returns a **tuple** `(logits, out_lens)` (line 209 of `train_stage_2.py`). But the pipeline on line 142 does:
```python
out = model(input_tensor, lengths)  # expects single tensor
pred = out[:, 0, :]                 # indexes as [T, B, C]
```
This will fail at runtime. `out` is a tuple `(logits, out_lens)`, so `out[:, 0, :]` will raise a `TypeError`. The fix is:
```python
logits, out_lens = model(input_tensor, lengths)
pred = logits[0]  # [T, vocab_size] for batch item 0
```

### Input Shape Mismatch (BUG)
The pipeline sends `input_tensor` with shape `[1, Num_Windows, 32, 42, 10]`, but `SLTStage2CTC.forward()` expects `[B, T_flat, 42, 10]` where `T_flat = Num_Windows * 32`. The model internally does `valid_x.view(num_clips, 32, 42, 10)`. So the input needs to be reshaped before passing:
```python
B, W, T, N, C = input_tensor.shape
input_tensor = input_tensor.view(B, W * T, N, C)  # [1, W*32, 42, 10]
lengths = torch.tensor([W * T], ...)
```
Without this, the model's `x[b, :x_lens[b]]` and `.view(num_clips, 32, 42, 10)` will fail on the 5D tensor.

### Dictionary Key Lookup (Lines 151-152)
Safe. The dual `.get()` pattern handles both `str` and `int` keys with an `UNKNOWN` fallback:
```python
word = idx_to_gloss.get(str(idx), idx_to_gloss.get(int(idx), f"UNKNOWN_{idx}"))
```
The checkpoint saves `idx_to_gloss` with `int` keys (from `train_stage_2.py` line 411), so `int(idx)` will match. The `str` check is a harmless safety net.

### CTC Blank Suppression (Lines 148-154)
Correct. Consecutive duplicate IDs and blank (index 0) are properly filtered, matching the `decode_ctc` function in `train_stage_2.py`.

---

## 3. `run_stage3_translation` (Lines 161-180)
Correct. Empty gloss list returns a placeholder string. The T5 prompt prefix `"translate ASL to English: "` matches the training prefix `"translate ASL gloss to English: "` — **minor inconsistency** that may slightly degrade quality but won't crash.

---

## Summary of Issues

| # | Severity | Location | Issue |
|---|----------|----------|-------|
| 1 | **Critical** | Line 142 | `model()` returns tuple `(logits, out_lens)`, not a single tensor. Indexing `out[:, 0, :]` will raise `TypeError`. |
| 2 | **Critical** | Line 139-140 | Input shape is `[1, W, 32, 42, 10]` (5D) but model expects `[B, T_flat, 42, 10]` (4D). Needs reshape. |
| 3 | Minor | Line 167 | T5 prompt prefix doesn't exactly match training prefix (`"translate ASL to English"` vs `"translate ASL gloss to English"`). |
