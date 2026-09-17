# English text-model review — 2026-09-17

Recommendation: **BART-base is the first English-only replacement worth testing**, while a vocabulary-trimmed copy of the completed mT5 hybrid is the more conservative way to investigate size reduction without throwing away its ASL-adapted weights. Neither is approved for training or deployment. No new training was launched, and the existing run was left unchanged.

The user requires no significant accuracy sacrifice and approval before another training run. This review establishes size and interface feasibility, not accuracy equivalence. The existing 12-sentence development set cannot establish that equivalence reliably.

## Verified size comparison

Counts were computed by instantiating parameter metadata, without allocating model weights or using MPS, from pinned official configurations. Totals include our 6,791,717-parameter Stage-1 network (including its isolated head) and the appropriate linear projection. Decimal MB below are theoretical FP16 parameter bytes only, excluding activations, caches, buffers and runtime overhead. FP16 execution has not been validated here.

| Text component | Total hybrid parameters | Vocabulary entries | Encoder / decoder layers | FP16 parameter MB | Reduction from current |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current Uni-Sign mT5 | 589.39M | 250,112 | 12 / 12 | 1,178.8 | — |
| Current mT5, hypothetical 64K retained vocabulary | 303.52M | 64,000 | 12 / 12 | 607.0 | 48.5% |
| Current mT5, hypothetical 32K retained vocabulary | 254.57M | 32,128 | 12 / 12 | 509.1 | 56.8% |
| Google T5 v1.1 base | 254.57M | 32,128 | 12 / 12 | 509.1 | 56.8% |
| Google FLAN-T5 base | 254.57M | 32,128 | 12 / 12 | 509.1 | 56.8% |
| **Meta BART-base** | **146.41M** | **50,265** | **6 / 6** | **292.8** | **75.2%** |
| Google FLAN-T5 small | 83.88M | 32,128 | 8 / 8 | 167.8 | 85.8% |

Sources: [BART-base](https://huggingface.co/facebook/bart-base), [T5 v1.1 base](https://huggingface.co/google/t5-v1_1-base), [FLAN-T5 base](https://huggingface.co/google/flan-t5-base), [FLAN-T5 small](https://huggingface.co/google/flan-t5-small). Exact revisions, configuration URLs and counts are in `verification.json`; downloaded configs and `inspect_candidates.py` reproduce the audit. FLAN models are not strictly English-only: their instruction tuning includes other languages, although they avoid mT5's 250K vocabulary.

## Why these candidates

**BART-base:** English-pretrained encoder–decoder with six layers per side, 768-dimensional states, and shared token/output embeddings. It can consume our projected visual features and generate English through the existing wrapper. Its pretraining reconstructs corrupted English text; the original paper also demonstrates adaptation to translation. This supports architectural suitability, not ASL accuracy. It has no Uni-Sign ASL training, so changing to BART discards that text-component knowledge. [Model card](https://huggingface.co/facebook/bart-base), [BART paper](https://aclanthology.org/2020.acl-main.703/).

**T5 v1.1 base:** the closest English-pretrained structural alternative to the current mT5, retaining 12 encoder and 12 decoder layers. It was pretrained on C4 without supervised downstream tasks. It is a reasonable fallback if retaining model depth matters, but still loses Uni-Sign adaptation and is materially larger than BART. Identical dimensions do not make tokenizer IDs or embeddings interchangeable. [Google model card](https://huggingface.co/google/t5-v1_1-base).

**FLAN-T5:** useful comparison candidates, but instruction-following performance does not establish visual grounding. Small offers the largest reduction here; given the accuracy requirement, it is not my first replacement experiment. Do not equate fluent English with correct signing interpretation. [Google model card](https://huggingface.co/google/flan-t5-small).

**Vocabulary trimming:** 384,172,032 parameters—65.2% of the current hybrid—are in the shared input embedding and separate output projection. Retaining selected rows in both matrices preserves every transformer layer and the remaining learned rows. Published work supports vocabulary trimming before or after fine-tuning for several NLP tasks, but it is not an ASL result or a guarantee for our checkpoint. [Vocabulary-trimming study](https://aclanthology.org/2023.findings-emnlp.981/).

For an accuracy-first investigation, begin with a 64K target budget rather than forcing 32K. Build the retained set using a broad English corpus plus training references, all required special/prefix tokens, and basic character coverage; never select it from evaluation references or errors. Audit coverage and tokenization before deciding the actual size. Do not reduce to just the 2,945 token IDs seen in our 994 training sentences: that would over-specialize to a tiny corpus. Remap tokenizer IDs and every affected embedding/output row consistently. Removing vocabulary can alter segmentation, softmax normalization and beam ranking even if surviving weights are unchanged. The sizes above are scenarios, not validated token sets.

## What can improve throughput without discarding sign information

The local training-reference audit found mean target length **24.66 tokens**, median **23**, 95th percentile **45**, maximum **70**. Thus **64.8%** of positions in the current fixed 70-token targets are padding. Ignored loss labels do not prevent the standard model from computing their decoder states and vocabulary logits. This is avoidable arithmetic, not a measured 64.8% wall-time saving.

After approval and after the existing job exits, profile a short, disposable training-only run before committing to the full experiment. Use a small number of length buckets rather than unrestricted shape variation, and no truncation. Measure cache-release/synchronization overhead separately; reduce its frequency only if a sustained memory check passes. Keep float32 initially to isolate the model change. Lower precision is a separate numerical-validation question, not a prerequisite or promised speedup. Keep the same batch size initially for comparison.

Do not shorten visual sequences, remove repeated targets, switch to one encoder token per sign, or drop isolated supervision for speed. Those changes can remove the very information this experiment is meant to learn. No timing or iPhone inference benchmark was run in this review; parameter reductions are not speedup factors.

## Concrete proposed experiment, pending approval

1. Let the already-authorized mT5 run finish and inspect its final visual-grounding controls. If it does not use signing meaningfully, do not assume a smaller language model solves that failure; reconsider before the full next run.
2. Evaluate one vocabulary-trimmed copy of its final checkpoint, initially targeting 64K retained tokens, with **no retraining**. Preserve the full checkpoint. This tests whether learned behavior survives vocabulary reduction before spending another training budget. Insufficient coverage means enlarge the set or stop, not silently drop words.
3. Train **one BART-base challenger**, initialized from the pinned English checkpoint and the same original Stage-1 checkpoint as the current pilot. Architecture: Apple features → existing Stage-1 frame sequence → 256-to-768 projection → full BART encoder–decoder; keep the isolated head and loss. No CTC or added intermediate gloss decoder.
4. Use the same 994 paired training utterances, 1,901 isolated replay samples, deterministic full coverage, seed 17111, and translation CE + 0.5 isolated CE. Proposed fixed schedule: 20 epochs, first projection-only, remaining 19 joint, same parameter-group learning rates/Adafactor as the existing pilot. This controls the initial comparison; it is not claimed to be BART's optimal recipe. Freeze tokenizer/special-token handling, non-truncating length buckets and all settings before the run. A short training-only wiring/throughput preflight is discarded and weights reset. Report a failed preflight instead of launching a long run blindly. Detach the full run with exit notification and no polling.
5. Compare final BART, full mT5 and trimmed mT5 using the same inputs, references and decoding policy. No automatic promotion and no official Citizen test access. Do not train every model in the table.

## What “no significant accuracy loss” must mean

There is no accuracy-preservation guarantee from model size or papers. Translation BLEU/chrF are not percentages of correctly recognized signs. The current 12 sentences come from only four sources and two adaptation-held signers, so they are a screening set, not sufficient evidence of equivalence. Inherited Uni-Sign pretraining is not certified signer-disjoint.

Proposed practical margins, to agree before evaluation: no more than **1.0 chrF point**, **0.5 BLEU point**, or **1 percentage point on either isolated-validation subset** below the completed full hybrid. Also retain Stage-1 accuracy against the original pre-adaptation checkpoint; an already-degraded full hybrid is not an acceptable retention target by itself. Review meaning, omissions, invented content, negation and numbers—not only averages. Repetition remains allowed; intended repetition must not be erased.

These are provisional engineering tolerances, not established clinical or linguistic significance thresholds. Twelve paired sentences cannot prove them. A deployment decision requires an expanded, independently annotated evaluation spanning new recordings/sources and signers, with a fixed protocol and uncertainty intervals respecting source/signer clustering. If uncertainty includes an unacceptable drop, the outcome is **inconclusive**, not “no accuracy loss.” Use zero-visual and mismatched-visual controls to test grounding. More data may be needed to establish the claim even if the training dataset is unchanged.

A smaller checkpoint is only a candidate for mobile. Core ML conversion, on-device memory, sustained latency, thermals and accuracy must be measured before any mobile-ready claim. No distillation, quantization or custom decoder is part of this proposal.

## Verification and authorization

`verification.json` records parameter metadata counts, training-target lengths and input-manifest hash. Only official configuration files were downloaded; no new pretrained weights were downloaded. `bart_api_check.json` records a tiny random CPU-only forward/generation compatibility check, not pretrained accuracy. No optimizer steps, new training, live-demo changes or modifications to the running experiment were made by this review.

Approval requested: the single BART-base experiment and the no-retraining trimmed-checkpoint comparison described above, after reviewing the current run's result. The other model options are research alternatives, not additional authorized jobs.

## Subsequent approval

The user approved this comparison on 2026-09-17 and requested results only. Execution and exit reports live in `../english_comparison_20260917/`. The proposed jobs run after the existing baseline exits, with no polling or automatic production promotion.
