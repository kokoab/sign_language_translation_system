# Corrected unified-streaming experiment

## Decision

The experiment supports continuing with a unified online recognizer, but the current
checkpoint is **not ready to replace the stable Reel prototype**. Correcting the
supervision and shortening the trailing observation window produces the first real
signer-disjoint continuous improvement. The remaining continuous set is too small and
the isolated-retention gate is narrowly missed.

No protected test set or external reserved evaluation set was accessed. All figures
below are validation results.

## What changed

1. Stage 1 was adapted on 92 manually aligned training sign cores from the continuous
   corpus, with Citizen, SemLex, and local phrase-segment replay.
2. Incomplete isolated-sign prefixes were removed from hard blank supervision. Only
   genuine inter-sign transitions remain blank targets.
3. Three previously omitted local templates were included with out-of-vocabulary
   glosses mapped to explicit `OTHER`.
4. Exact local and ASLLRP phrase sources, plus Citizen and SemLex replay, were balanced
   during CTC training.
5. The causal head was tested with 32- and 8-source-frame trailing windows. Each
   window is still resampled to the Stage-1 model's expected 32-frame input.

The accepted Reel files and checkpoints were not modified.

## Stage-1 core adaptation

| Validation source | Before | After | Change |
| --- | ---: | ---: | ---: |
| Citizen isolated (378) | 95.24% | 94.18% | -1.06 pt |
| SemLex isolated (978) | 85.28% | 85.79% | +0.51 pt |
| Local phrase sign cores (259) | 63.32% | 89.58% | +26.26 pt |
| ASLLRP manual sign cores (24) | 66.67% | 83.33% | +16.66 pt |

This proves that the manual boundaries are useful and that continuous-domain sign
cores are mostly recognizable. It does not by itself solve streaming segmentation.

## Matched streaming results

| Experiment | Window / stride | ASLLRP exact | ASLLRP WER | Local exact | Citizen isolated | All transitions false emission |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original rolling v1 | 32 / 4 | 0/12 | 62.50% | 85.57% | 73.28% | 12.64% |
| Corrected v2 | 32 / 4 | 0/12 | 62.50% | 85.57% | 86.24% | 1.15% |
| Corrected + source balance | 32 / 4 | 0/12 | 58.33% | 58.76% | 88.62% | 2.30% |
| **Short-window v3** | **8 / 4** | **4/12** | **41.67%** | **80.41%** | **92.59%** | **2.30%** |
| Short-window, dense sampling | 8 / 2 | 4/12 | 41.67% | 55.67% | 89.15% | 2.87% |

The stride-2 run took 120.75 seconds versus 86.91 seconds for stride 4 and did not
improve continuous accuracy. The selected experimental checkpoint is therefore:

`artifacts/models/unified_streaming_ctc_v17_experiment_v3_short_window/best_model.pth`

It remains an experiment, not the default live model.

## Why window length mattered

The twelve aligned validation phrases contain only 21--54 source frames. With a
32-frame trailing window, the Stage-1 classifier saw both expected glosses somewhere
in its raw top-1 window stream for 0/12 clips. A direct matched diagnostic gave:

| Trailing source frames | Clips exposing both target glosses |
| ---: | ---: |
| 32 | 0/12 |
| 24 | 1/12 |
| 16 | 3/12 |
| 12 | 5/12 |
| 8 | 8/12 |

The long window repeatedly mixes the first sign, transition, and short second sign.
The causal head cannot recover a gloss that its pooled Stage-1 evidence never exposes.
An 8-frame trailing crop is noisier, but CTC can suppress much of that noise and gains
four exact continuous sequences without an expected-phrase table or grammar.

## Interpretation

This was not primarily bad annotation and it was not evidence that a larger sequence
model is needed. The dominant problems were mismatched hard-negative labels, source
imbalance, and a whole-sign observation window that was too long for short
coarticulated signs. The 111,406-parameter causal head has enough capacity to show a
real gain once it receives usable evidence.

The best run still misses the intended gates:

- ASLLRP WER is 41.67%, above the provisional 35% target.
- Citizen isolated is 92.59%, below the 93--94% retention target.
- ASLLRP transition false emission is 2/12; the aggregate 2.30% rate hides this small
  source-specific weakness.
- `OTHER`-span known-gloss WER is 89.08%, so arbitrary OOV-context claims are not
  supported.

## Recommendation for the 30-phrase collection

Proceed with the planned collection. Record connected signing with no artificial
neutral pose between glosses. For each phrase and signer, capture two natural-speed
performances and one slower-but-still-connected performance. Preserve the fixed
10-train / 2-validation / 1-sealed-signer split.

Store the exact gloss sequence for every performance. Manually mark sign cores,
transitions, and background for every validation and sealed recording, and for a
representative training subset covering every gloss and transition family. Training
recordings not manually marked can be force-aligned after the first adapted model is
trained, with low-confidence alignments reviewed. Also record 60--90 seconds per
signer of ordinary non-sign hand/face activity for a genuine false-activation set.

After collection, the next experiment should keep the 8-frame / stride-4 causal path,
adapt Stage 1 on annotated cores with isolated replay, retrain the head, and evaluate:

- signer-disjoint continuous exact match and gloss WER;
- isolated top-1 retention;
- transition and ordinary-activity false emissions;
- first-emission and stable-prefix latency; and
- compositional holdouts whose exact phrase combinations were absent from training.

Only after those gates should this head be connected to a separate live prototype.
The stable Reel path should remain the demonstration fallback until then.
