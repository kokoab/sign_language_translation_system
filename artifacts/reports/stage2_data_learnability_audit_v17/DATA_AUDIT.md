# Complete continuous-supervision audit

Date: 2026-09-14 PHT  
Scope: every sample admitted by the frozen O5S5 + ASLLRP supervision manifest, every
isolated replay identity used by the bounded trainer, the original model, both prior
comparators, and all 12 O5S5 checkpoints. The Citizen test was not accessed.

## Finding

The data is not corrupt or uniformly unlearnable, and O5S5 is not faster than the
existing ASLLRP data. The failure comes from three measured problems working together:

1. fixed windows often contain much more context than target signing;
2. the source-balanced replacement sampler does not expose the model to all admitted
   windows; and
3. O5S5 has weak class-level signer coverage and a large held-out-signer gap that remains
   even when exact target boundaries are supplied.

The architecture compounds these problems by forcing one label onto a pooled window and
learning `NO_EMIT` separately. It can trade insertions for deletions, but it cannot learn
a reliable temporal alignment.

## What was validated

| Population | Train | Validation | Total |
|---|---:|---:|---:|
| Context windows | 6,819 | 1,649 | 8,468 |
| Verified background windows | 494 | 16 | 510 |
| All continuous windows | 7,313 | 1,665 | 8,978 |
| Isolated replay clips | 1,901 | 1,356 | 3,257 |

All 1,166 manifest rows and their raw arrays loaded through the production validation
path. No invalid shape, timestamp, nonfinite tensor, protected path, or exact
feature-identical conflicting label was found. Any-hand evidence is present in 98.68%
of O5S5 training window frames and 100% of LG validation window frames. This rules out a
general extraction or empty-window failure.

The original isolated encoder remains strong: 1,900/1,901 training clips are correct,
and validation is 360/378 Citizen plus 834/978 SemLex. The O5S5 epoch-12 encoder still
gets the same 1,900/1,901 training clips and 354/378 Citizen plus 835/978 SemLex. The
full 101-way runtime wrapper scores lower because it sometimes chooses `NO_EMIT`; the
underlying 100-way classifier forgets some Citizen examples but not SemLex. Isolated
classification is therefore not the main failure.

## Signing speed

| Source / split | Events | Median duration | Under 0.27s | Under 0.53s |
|---|---:|---:|---:|---:|
| ASLLRP other, train | 1,199 | 0.267s | 59.55% | 93.24% |
| O5S5, train | 199 | 0.259s | 53.27% | 88.44% |
| ASLLRP other, validation | 284 | 0.267s | 55.63% | 96.83% |
| O5S5 LG, validation | 57 | 0.340s | 29.82% | 84.21% |

O5S5 training signs are only 8 ms shorter at the median than ASLLRP. LG is slower, yet
it is the weakest split. Raw signing speed alone does not explain the result.

Short targets are harder, but the clearest problem is target-to-window mismatch. At
epoch 12, O5S5 training accuracy is 55.85% on 0.27-second windows, 53.42% on 0.53-second
windows, and only 32.45% on 1.07-second windows. ASLLRP-other shows the same collapse:
59.55%, 65.41%, and 38.26%. In the 1.07-second O5S5 windows, only 29.09% of frames on
average belong to the annotated target. The loss still assigns the whole window the
target label. For positive-only O5S5 narratives, the remaining frames may contain
unannotated signing.

On LG, epoch-12 accuracy by target duration is 0/30 for targets under 0.20 seconds,
6/48 for 0.20–0.27 seconds, 18/105 for 0.27–0.40 seconds, 29/66 for 0.40–0.53 seconds,
and 20/69 for targets at least 0.53 seconds. Label and duration are confounded, so this
is diagnostic rather than a causal speed experiment, but very short LG targets are a
real weakness.

## The trainer did not use all admitted windows

The exact seed-17111 sampler was reconstructed for all 12 epochs. It draws 900 context
examples per epoch with replacement and balances the three source names equally.

| Source | Available windows | Total draws | Never drawn | Median draws/window |
|---|---:|---:|---:|---:|
| ASLLRP contiguous | 299 | 3,623 | 3 (1.00%) | 7 |
| ASLLRP other | 5,515 | 3,593 | 3,516 (63.75%) | 0 |
| O5S5 | 1,005 | 3,584 | 154 (15.32%) | 2 |

The 299-window source receives approximately the same total training mass as the
5,515-window source. This is not merely a reporting issue: O5S5 windows drawn at least
once reach 51.47% epoch-12 accuracy, versus 41.56% for never-drawn windows. Coverage
helps, although ASLLRP correlations and class mix mean exposure alone does not explain
every source result.

## Full-pool fit and held-out generalization

The table separates the full 101-way output from the underlying 100-gloss classifier.
`Known top-1` ignores `NO_EMIT` and asks whether the visual classifier chose the right
gloss.

| Split | Original known top-1 | Epoch-12 full accuracy | Epoch-12 known top-1 | Correct in any of 12 epochs |
|---|---:|---:|---:|---:|
| ASLLRP contiguous train | 42.14% | 84.95% | 89.97% | 87.96% |
| ASLLRP other train | 31.41% | 60.18% | 69.48% | 66.46% |
| O5S5 train | 28.26% | 49.95% | 54.03% | 52.84% |
| ASLLRP contiguous validation | 55.07% | 69.57% | 84.06% | 76.81% |
| ASLLRP other validation | 40.65% | 48.26% | 64.74% | 63.23% |
| O5S5 LG validation | 23.90% | 22.96% | 28.62% | 31.13% |

This is partially learnable data. The model learns ASLLRP and O5S5 training windows,
but never fits O5S5 well and does not transfer to LG.

The exact-boundary probe pools only annotated target frames from the already encoded
window. At epoch 12 it reaches 91.97% on ASLLRP-contiguous train, 72.98% on ASLLRP-other
train, and 61.09% on O5S5 train. On validation it reaches 85.51%, 68.70%, and only
27.67% respectively. Removing surrounding frames helps O5S5 training by 7.06 points
relative to known-window classification, but does not fix LG. The LG failure is thus
not only a boundary or emission problem; it includes visual variant, signer, or domain
generalization.

## “Multi-signer” coverage is thin per class

O5S5 provides five training signers overall, but only one of 49 classes appears from all
five. Twenty-three classes appear from one signer, 10 from two, nine from three, and six
from four. LG has 27 target classes; 24 appear in O5S5 training, while `ASK`, `HOW`, and
`MAKE` do not. Among the 24 overlapping LG classes, five have one O5S5 training signer,
five have two, eight have three, five have four, and one has five.

This corpus adds signer variety in aggregate, but it is not a five-signer training set
for each sign. The data cannot by itself establish broad signer-independent learning.

## Root-cause assessment

| Hypothesis | Evidence | Verdict |
|---|---|---|
| O5S5 extraction failed | 98.68–100% any-hand frame coverage; all arrays valid | Rejected |
| O5S5 is uniquely too fast | Train median 0.259s vs ASLLRP 0.267s; LG is slower | Rejected |
| Very short signs are difficult | LG under-0.20s: 0/30; ASLLRP also degrades | Supported |
| Wide fixed windows dilute labels | 1.07s accuracy collapses; target occupies 25–29% of frames | Supported |
| Model saw all admitted data | 63.75% of ASLLRP-other and 15.32% of O5S5 windows never drawn | Rejected |
| O5S5 supplies robust per-class multi-signer evidence | 23/49 classes have one training signer | Rejected |
| Segmentation alone explains LG | Exact-core LG remains 27.67% | Rejected |
| Current model can learn some continuous evidence | ASLLRP exact-core validation reaches 85.51% | Supported |

The data should not be discarded. The safe correction is to use deterministic coverage,
remove whole-window target loss from the 1.07-second mixed windows, and train sequence
alignment jointly with the encoder. O5S5 must remain positive-only until its complete
transcripts are admitted; it can supervise exact cores but not blank regions or complete
sequence CTC.

## Artifacts

- [Machine-readable audit](audit.json)
- [Every window and selected predictions](all_window_predictions.csv)
- [Every admitted known event and duration](event_durations.csv)
- [Model/group metrics](model_group_metrics.csv)
- [Duration, exposure, and signer breakdown](diagnostic_breakdown.json)
- [Exact annotated-core probe](foreground_core_probe.json)
- [Reproducible full audit](audit_all_supervision.py)
- [Reproducible reductions](analyze_audit.py)
- [Reproducible core probe](foreground_core_probe.py)

