# Stage 2 Inference Review R2 — Continuous Sequence Testing

**Date**: Round 2
**Topic**: Is raw clip concatenation safe for inference?

---

## Question 1: Did training insert blank/transition frames between clips?

**No.** `SyntheticCTCDataset.__getitem__` (train_stage_2.py:238-255) does:

```python
x = np.concatenate(arrays, axis=0)  # line 254
```

That's it. No blank frames, no zero padding between clips, no transition frames. Training concatenated raw 32-frame clips directly back-to-back — exactly what your inference script does.

## Question 2: Is back-to-back concatenation safe for the BiLSTM?

**Yes — your inference exactly matches the training distribution.** Here's why:

The BiLSTM never sees raw frames. It sees **pooled encoder tokens**. The data flow is:

1. `[96, 42, 10]` → reshape to 3 clips of `[32, 42, 10]`
2. Each clip → frozen encoder → `[32, 256]` → AdaptiveAvgPool1d(4) → `[4, 256]`
3. LSTM input: `[12, 256]` (3 clips × 4 tokens each)

The "teleportation" between sign A's last frame and sign B's first frame happens **inside the frozen encoder's 32-frame window**, but the encoder processes each clip independently anyway. The LSTM only sees the 4 pooled tokens per clip — it has no concept of frame-level transitions.

This is exactly how training worked: each clip was processed independently by the encoder, pooled to 4 tokens, then the LSTM saw the concatenated token sequence.

## Question 3: How should the continuous tensor be formatted?

**What you have is already correct.** Specifically:

- Each clip forced to exactly 32 frames: correct (matches `arr.shape == (32, 42, 10)` filter at train_stage_2.py:246)
- Raw concatenation along axis=0: correct (matches train_stage_2.py:254)
- Shape `[1, 96, 42, 10]`: correct (3 clips × 32 frames = 96)
- No transition/blank frames: correct (training had none)

## Verdict

Your continuous inference script is a **faithful reproduction** of the training data distribution. If hallucination occurs on continuous sequences, the issue is NOT in the data formatting — it would be in the model's generalization (e.g., the synthetic training sentences used `random.choices` which allows duplicate glosses, while your test uses `random.sample` which doesn't — but this is minor).

No changes needed.
