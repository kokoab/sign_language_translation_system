# Downstream stages on the approved phrase split — verified results

Recipe: `active/v17/phrase_segment_recipe_manifest_20261007.json` (train = approved train ∩ lab train, 179 local phrases;
validation = approved validation − lab test, 139; lab tune pool for recognizer selection; the 72-clip lab held-out test
was never trained on, selected on or evaluated). Commands reproduce the August stages exactly (see reproductions/);
only the base checkpoint, isolated caches and phrase split differ. Validation development evidence; no protected test access.

## Accuracy by phase across all validation sets

| Chain | Phase | Citizen (378) | SemLex (978) | Local (2,896) | Mean of 3 | Pooled (4,252) |
|---|---|---:|---:|---:|---:|---:|
| 96.83 chain | Isolated landmark (96.83%) | 96.83% | 87.22% | 64.12% | 82.72% | 72.34% |
| 96.83 chain | Local adaptation | 97.35% | 87.32% | 98.17% | 94.28% | 95.60% |
| 96.83 chain | Multimodal fusion | 97.62% | 88.04% | 98.20% | 94.62% | 95.81% |
| 96.83 chain | Phrase-segment adaptation | 97.62% | 88.04% | 98.03% | 94.56% | 95.70% |
| 96.83 chain | Interval recognizer | 97.09% | 88.34% | 97.76% | 94.40% | 95.53% |
| August chain | Isolated landmark (a7490409) | 95.77% | 85.79% | 60.95% | 80.83% | 69.76% |
| August chain | Local adaptation | 95.50% | 87.93% | 96.34% | 93.26% | 94.33% |
| August chain | Multimodal fusion | 96.30% | 89.06% | 97.10% | 94.15% | 95.18% |
| August chain, approved split | Phrase-segment adaptation | 95.77% | 89.06% | 96.96% | 93.93% | 95.04% |
| August chain, approved split | Interval recognizer | 95.77% | 89.06% | 96.55% | 93.79% | 94.76% |
| August chain, original leaking split | Phrase-activity adapted (reel_v2) | 96.03% | 89.16% | 97.03% | 94.07% | 95.13% |
| August chain, original leaking split | Interval recognizer (local_a) | 95.24% | 89.67% | 96.31% | 93.74% | 94.68% |

## Phrase-segment adaptation and interval recognizer

| Chain | Phrase epoch | Phrase segments | Activity crops | Recognizer epoch | Tune WER (epoch 0 → selected) | Floors (C/S/L) |
|---|---:|---:|---:|---:|---|---|
| 96.83 chain | 22 | 282/375 (75.20%) | 1117/1500 (74.47%) | 4 | 37.61% → 31.42% | 96.62/88.04/97.03 |
| August chain, approved split | 23 | 267/375 (71.20%) | 1080/1500 (72.00%) | 6 | 38.94% → 30.09% | 94.77/89.06/95.96 |

Recognizer training: 179 approved-train videos, 3370 spans (August original: 282 videos incl. 103 approved-validation, 6,254 spans).
Tune WER is the selection metric on the 89-clip tune pool (development evidence, not held-out WER).

## Letter head (head-A recipe retrained on the new recognizer)

New: 26-way top-1 86.23%, recall 83.71% at τ=0.7; head A: 86.62%, recall 86.15% at τ=0.5.

## Core ML conversion (Mac, 378 Citizen validation clips, identical inputs)

| Package | Size (MB) | PyTorch correct | Core ML correct | Top-5 | Agreement |
|---|---:|---:|---:|---:|---:|
| 96.83 chain FP32 | 50.28 | 367 | 367 (97.09%) | 376 | 378/378 |
| 96.83 chain FP16 | 25.51 | 367 | 367 (97.09%) | 376 | 378/378 |
| August FP32 / FP16 (reference) | 50.28 / 25.51 | 360 | 360 / 359 | 374 / 374 | — |

## iPhone 13 timing (preparation + recognition per frame, ms)

| Configuration | Run medians | Median of run medians | Thermal before/after |
|---|---|---:|---|
| august_recognizer_fp16 | 25.58 / 22.92 | 24.25 | 0,0 / 0,0 |
| chain9683_recognizer_fp16 | 24.26 / 23.13 | 23.69 | 0,0 / 0,0 |
| august_recognizer_fp32 | 54.77 / 57.37 | 56.07 | 0,0 / 0,0 |
| chain9683_recognizer_fp32 | 59.56 / 58.75 | 59.15 | 0,0 / 0,0 |

Not claims: tune WER is development selection; the lab held-out test was not run for either rebuilt chain.
Isolated SemLex validation reuses SemLex train signers; local validation is familiar-signer.

## Interpretation notes

- Phone: same iPhone 13, 226 recorded-input example frames per configuration per pass, 20 warm-up frames,
  two passes in forward/reverse order, hand encoder and word boundary FP16, all compute units; only the
  recognizer package changes. Thermal state nominal (0) throughout, low-power off. The two recognizers share one
  architecture and package size, so the FP16 difference (23.69 vs 24.25 ms) is within the observed pass-to-pass
  spread (22.92–25.58 ms) and should be reported as equivalent, not as a speedup. FP16 remains ~2.4× faster than FP32.
  Timing covers frame preparation and recognition, not camera capture, model loading, UI or English output.
  First attempt (phone_recognizer_attempt1_missing_testability/) failed to build because ENABLE_TESTABILITY=YES
  was omitted; sources were restored exactly and production reinstalled in both attempts.
- Letter head: retrained with the head-A recipe so the exported package is internally consistent; letter recall at
  the selected threshold is lower (83.71% vs 86.15%). Letters are outside the recognition comparison.
- The August original phrase stage and recognizer trained on clips from the lab held-out test (51/72) and tune
  pool (73/89) through their parent head and on approved-validation clips; their streaming WERs are not fully
  held out. Both rebuilt chains here exclude those clips, which halves the recognizer's span examples
  (3,370 vs 6,254) and is the likely reason tune WER is higher than the original 27.43%.
