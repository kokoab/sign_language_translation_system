# Lightweight streaming sequence experiment (v17)

## Verdict

The 26,861-parameter causal head is fast enough, but it is **not accurate enough to
replace the current Stage 2**. It should not be connected to the live default or used
for model claims. The experiment supports keeping the fast Reel/Stage-1 interaction
for isolated signs while retaining the accepted Stage-2 model for optional phrase
confirmation until more signer-disjoint continuous data exist.

No Citizen test, SemLex test, RIT reserved row, or other sealed set was accessed.

## What was tested

Two new architectures were kept separate from all accepted files:

1. A raw 61-node causal depthwise TCN with a 101-way CTC output. A fail-fast training
   screen collapsed to blank (0/109 phrase exact), so a full budget was not spent on a
   model that discarded the already strong Stage 1.
2. A causal depthwise TCN over rolling 100-gloss Stage-1 logits. It preserves the
   existing classifier as a residual, learns blank/timing with CTC, retains per-block
   state, and adds only 26,861 parameters (119 KiB checkpoint).

Training used 434 real phrase clips, 2,975 isolated training clips, and train-only
synthetic compositions. Validation used 109 phrase clips and 1,356 isolated clips.
Local phrases and signer-held-out ASLLRP are reported separately because local signer
metadata are unavailable.

## Results

| Model / validation domain | WER | Exact sequence |
| --- | ---: | ---: |
| New causal head, local phrases (97) | 40.93% | 44.33% |
| New causal head, ASLLRP held-out signer (12) | 62.50% | 0.00% |
| Existing Stage 2, local phrases (97) | 2.70% | 92.78% |
| Existing Stage 2, ASLLRP held-out signer (12) | 45.83% | 16.67% |

The new head retained 94.71% exact accuracy on 378 Citizen validation clips and
83.95% on 978 SemLex validation clips. This is good isolated behavior, but continuous
under-emission remains the decisive failure.

On MPS, a Stage-1 rolling window measured 29.33 ms median and the causal head added
3.06 ms median (32.40 ms combined), excluding landmark extraction. The head is
therefore not the latency problem. It can begin observing after eight frames and
update every four frames, but low latency does not compensate for the accuracy gap.

## Stage-1 retraining and landmark decision

The existing 61-node v17 feature set genuinely helps. Zeroing face/body nodes reduced
Citizen validation from 95.24% to 65.08%, SemLex from 85.28% to 59.51%, and local
phrase-segment accuracy from 63.32% to 44.02%. The present face/body and four lip
points should remain enabled. Adding new live-only MediaPipe mouth nodes was rejected
because the stored training phrases do not contain that schema; doing so would create
a train/live mismatch rather than learned lip evidence.

A separate Stage-1 contextual fine-tune added 92 genuine ASLLRP training segments to
Citizen/SemLex/local replay. The gated epoch improved local phrase-segment validation
from 59.85% to 66.41% and held-out ASLLRP segments from 66.67% to 75.00%, while Citizen
moved from 95.24% to 94.97% and SemLex from 85.28% to 85.17%. It kept the identical
Stage-1 architecture, so inference cost did not grow. However, retraining the causal
head on it made full-sequence results worse (local WER 43.24%, ASLLRP WER 66.67%). The
contextual Stage-1 checkpoint therefore remains an experiment, not a promotion.

## Interpretation

The failure is not primarily model size or landmark count. Stage 1 recognizes complete
segments well, including contextual ASLLRP segments, but rolling windows do not provide
reliable learned token boundaries from the available phrase diversity. Only 44 unique
genuine training sequences and 50 directed bigrams cover 46/100 glosses; the repeated
local clips are concentrated in six phrase identities. The old Stage 2 remains more
accurate because it uses richer frozen multimodal features and whole-window context,
even though its current live policy waits too long and over-assumes phrases.

The next justified experiment is data-facing: obtain signer-disjoint continuous
coverage across substantially more bigrams, then retrain this small causal head (or an
Emformer/RNN-T fallback) with manual or forced alignments. The architecture and paper
review is in [the research report](../lightweight_streaming_stage2_research_v1/README.md).

## Reproduction

```bash
venv/bin/python active/v17/train_stage1_contextual_adapt_v17.py
venv/bin/python active/v17/train_streaming_stage1_head_v17.py \
  --epochs 20 --synthetic-count 500 --phrase-repeats 20 \
  --batch-size 256 --head-device mps --learning-rate 0.001 \
  --output-dir artifacts/models/streaming_stage1_head_v17_experiment_v3
venv/bin/python active/v17/benchmark_streaming_stage1_head_v17.py
venv/bin/python -m unittest test.test_streaming_tcn_ctc_v17 -v
```

Machine-readable evidence is in `latency.json`, `latency_cpu.json`,
`modality_ablation.json`, and the model directories named above.
