# Stage 3 Training Script Review — `src/train_stage_3.py`

---

## Fixed Directly

### 1. Stale safety-net cleanup was deleting valid training data (Critical — fixed)
The regex on old line 27 (`r'^(I LIKE|HE TALK|MY \w+ WHERE)\s'`) was written for the **old** generator which produced broken pairs like `"I LIKE WHERE"`. After R3-R5 rewrites, `build_safety_net` only emits valid noun-class pairs (`"I LIKE BOOK"`, `"MY PHONE WHERE"`, etc.). The regex was now **removing good data**.
**Fix applied:** Removed the regex block. Replaced with a `dropna()` guard.

### 2. `predict_with_generate=True` wasted GPU time every eval epoch (Significant — fixed)
With `predict_with_generate=True` and `generation_num_beams=4`, the Trainer ran full beam-search decoding on the entire eval set (~5k rows) at every epoch. But there was no `compute_metrics` function — the generated text was never scored. Pure wasted compute (~10x slower evals for no benefit).
**Fix applied:** Removed `predict_with_generate=True`. Eval now only computes loss (the metric used for early stopping).

### 3. `MAX_INPUT=48` / `MAX_TARGET=80` oversized (Minor — fixed)
Longest input with prefix: ~22 tokens. Longest target: ~25 tokens. Set to 32/48 — still generous, avoids any truncation, but documents realistic bounds.

---

## No Changes Needed (Verified Correct)

| Setting | Verdict |
|---------|---------|
| `lr=3e-4` + cosine schedule | Standard range for T5-small fine-tuning on synthetic data ✓ |
| `batch_size=64` | Fits Kaggle P100 16GB with fp16 at these sequence lengths ✓ |
| `warmup_steps=100` (~13% of epoch 1) | Reasonable for preventing early divergence ✓ |
| `weight_decay=0.01` | Standard for T5 ✓ |
| `early_stopping_patience=3` | Appropriate — prevents overfitting on synthetic data ✓ |
| `pad_to_multiple_of=8` | Tensor core alignment for fp16, correct ✓ |
| `save_total_limit=2` | Keeps disk usage reasonable on Kaggle ✓ |
| `DataCollatorForSeq2Seq` with `padding=False` in tokenizer | Dynamic padding — efficient ✓ |
| Quick inference test (Section 8) | Good sanity check, test glosses cover key patterns ✓ |
| Zip packaging (Section 9) | Correct for Kaggle download workflow ✓ |

---

## Optional Improvements (Not Applied)

### Add a BLEU `compute_metrics` for richer eval signal
Currently eval only reports loss. Adding BLEU would tell you if the model is producing correct translations, not just low-perplexity ones. If you want to add this later:
```python
import evaluate
bleu = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = [[l for l in label if l != -100] for label in labels]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    return {"bleu": result["score"]}
```
Then re-enable `predict_with_generate=True` and pass `compute_metrics=compute_metrics` to the Trainer. Only do this if eval speed is acceptable (~10x slower per epoch).

### Add gradient accumulation if batch size becomes a bottleneck
If you increase sequence lengths or switch to a larger model, you may need:
```python
gradient_accumulation_steps=2,
per_device_train_batch_size=32,   # effective batch = 64
```
Not needed for current setup on Kaggle P100.
