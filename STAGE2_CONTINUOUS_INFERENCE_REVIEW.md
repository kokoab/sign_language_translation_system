# Stage 2 Continuous Inference Review — Concatenation Safety

**Date**: Round 1
**Question**: Does the training DataLoader insert blank/transition frames between clips? Is raw concatenation safe for inference?

---

## Answer: Your concatenation is exactly what training did. No changes needed.

### Evidence from training code

**`SyntheticCTCDataset.__getitem__`** (`train_stage_2.py:238-255`):
```python
for f, tgt in zip(files, target_glosses):
    arr = np.load(self.data_path / f).astype(np.float32)
    if arr.shape == (32, 42, 10):
        arrays.append(arr)
x = np.concatenate(arrays, axis=0)   # <-- raw concatenation, no padding between clips
```

**No blank frames. No transition frames. No zeros between clips.** The training dataset concatenates clips back-to-back on `axis=0`, identical to your inference code.

---

## Why "teleportation" between clips doesn't matter

The BiLSTM **never sees raw landmark frames**. Here's the data flow (`train_stage_2.py:186-200`):

1. `valid_x.view(num_clips, 32, 42, 10)` — the concatenated tensor is split into **independent 32-frame clips**
2. Each clip is processed by the encoder **independently** — clip 1 has zero awareness of clip 2
3. Each clip's `[32, 256]` encoder output is pooled to `[4, 256]` via `AdaptiveAvgPool1d(4)`
4. The pooled tokens are concatenated: `[num_clips * 4, 256]`
5. **Only then** does the BiLSTM see the sequence

The BiLSTM operates on **abstract 256-dim tokens**, not raw landmarks. There is no "frame 32 to frame 33 teleportation" at the LSTM level — there's just token 4 (last token of sign A) followed by token 5 (first token of sign B). This is exactly the distribution the LSTM was trained on.

---

## Your inference code vs training distribution

| Aspect | Training | Your inference | Match? |
|--------|----------|---------------|--------|
| Concatenation | `np.concatenate(arrays, axis=0)` | `np.concatenate(clips, axis=0)` | Identical |
| Blank frames between clips | None | None | Identical |
| Clip length | Strict `== (32, 42, 10)` check, skip otherwise | Pad/trim to 32 | Equivalent |
| Sequence length | 2-6 clips (`min_len=2, max_len=6`) | 3 clips | Within range |
| Augmentation | Applied during training, NOT during validation | None | Matches val distribution |

---

## Summary

Your continuous inference tensor `[1, 96, 42, 10]` from 3 concatenated clips is a **perfect match** for the training data distribution. No modifications needed. If the model still hallucinates on continuous sequences, the issue is not in the tensor formatting — look at the CTC decode logic or the model's generalization ability instead.
