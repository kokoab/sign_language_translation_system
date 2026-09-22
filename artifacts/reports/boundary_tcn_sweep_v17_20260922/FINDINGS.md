# Boundary TCN chain — architecture, output formulation, distillation

Run 2026-09-22 into 2026-09-23. Every number below is end-to-end WER through the live path:
`BoundaryStream` frame by frame over Apple Vision observations, intervals through the frozen
Reel cascade with unchanged commit rules, whole-video WER on the **89-clip / 226-sign local
tuning pool**. The 72-video / 186-sign held-out set was **not touched** in any round.

Nothing is promoted. The pinned trainer, recipe, production checkpoint and the DGS teacher
weights were read-only throughout.

## Headline

| | |
|---|---|
| Architecture (width, depth, receptive field) | **no effect** — F(7,8)=1.54, p=0.28 |
| Output formulation (BIO vs start/end edges) | **−2.67 WER**, 95% CI [1.35, 4.00], p<0.001 |
| Distillation from the frozen DGS teacher | **+3.98 WER — significantly worse**, p=0.003 |
| **Held-out replication of the BIO win** | **−3.01 WER**, 95% CI [0.37, 5.66], p=0.059 |
| **Held-out vs the deployed checkpoint** | **−6.13 WER** (56.77% vs 62.90%), 5/5 seeds |

One real improvement was found, and it is not the one the chain was set up to find. It
replicates on data no selection round touched.

## Method notes that make the numbers readable

- **Seed noise is 2.1–2.6 WER points (SD).** Two models identical in architecture, data,
  targets and seed, differing only in the initial draw of a 64→2 convolution's 130 weights,
  landed 6.2 points apart. Any ranking on one or two seeds is mostly noise; rounds 3-5 use
  12–24 seeds per arm.
- **The control reproduces.** `bio_gap` scored 70.98% (n=24, round 3) and 71.35% (n=12,
  round 4) — a 0.37-point difference across separate processes. The harness is stable.
- **The chunk builder was verified against the pinned one**, exact array equality at
  `left_context=30`, before any variant ran. The round-2 control arm's targets were verified
  identical to `boundary_targets` value-for-value on 300 records.
- **`SweptBoundary` is not a re-implementation**: at baseline config it loads the production
  `TemporalBoundary` state dict and returns bit-identical outputs (78,402 params, rf 30).
- **Calibration loss does not predict WER** — r=+0.19 across architectures (round 1), and
  within an arm r=+0.08 (edges) and +0.19 (bio_gap), both n.s. (round 3). There is no honest
  way to select a seed without touching a WER set.

## Round 1 — architecture sweep (8 variants × 2 seeds)

Selected on mean WER. `baseline_h64_d4` is the production architecture.

| variant | params | receptive field | mean WER |
|---|---|---|---|
| h64_d5 | 90,754 | 62 | 68.81% |
| h128_d4 | 255,106 | 30 | 70.13% |
| h192_d4 | 530,114 | 30 | 71.46% |
| **baseline_h64_d4** | 78,402 | 30 | 72.12% |
| h128_d5 | 304,386 | 62 | 72.35% |
| h64_d3 | 66,050 | 14 | 73.01% |
| h96_d4 | 154,466 | 30 | 73.23% |
| h128_d6 | 353,666 | 126 | 75.22% |

Pooled within-variant SD (seed noise) **2.24**; SD of the eight variant means **1.96** —
*less than the noise*. F(7,8)=1.54, **p=0.28**. corr(params, WER)=+0.17; 6.8× the parameters
buys nothing. Receptive fields from 14 to 126 frames change nothing.

**Conclusion: the TCN's capacity is not the binding constraint.** Widening or deepening it
is a dead end.

## Round 2 — output formulation and supervision coverage (4 arms × 2 seeds)

Motivated by the pinned recipe's own declared limitation:

> `supervision: "start/end positives; negatives only inside accepted intervals; all other values ignored"`
> `limitation: "... no supervised outside class."`

Only **39% of each clip lies inside an annotated sign**; the other 61% is never supervised
but is scored at inference, where the decoder proposes ~280 intervals for 226 references.

Blanket "outside = not a sign" would be false supervision: 96% of the 1,121 boundary records
come from `asllrp_other_ctc`, which is not exhaustively glossed. But the gap statistics give
an honest subset — median sign 0.33s, median inter-sign gap 0.33s, **53% of gaps ≤0.4s**, too
short to conceal a sign plus its transitions. Those gaps can be labelled non-sign; longer
gaps, lead-in and tail-out stay unknown.

| arm | what it isolates | mean WER | spread |
|---|---|---|---|
| bio_gap | output formulation | 68.81% | 8.41 |
| edges (control) | — | 71.24% | 6.19 |
| edges_gap_o | outside supervision | 75.66% | 5.31 |
| edges_wide (±150ms band) | positive sparsity | 78.10% | 3.98 |

Two seeds could not resolve these; the spreads are as large as the gaps. Round 3 settles it.

## Round 3 — powered comparison (2 arms × 24 seeds)

| arm | n | mean WER | SD | min | max | intervals proposed |
|---|---|---|---|---|---|---|
| edges (pinned) | 24 | 73.65% | 2.09 | 68.14 | 77.88 | 291 ± 22 |
| **bio_gap** | 24 | **70.98%** | 2.58 | 64.60 | 74.78 | **241 ± 25** |

**edges − bio_gap = +2.67 points, 95% CI [1.35, 4.00], p<0.001.**

The 4-class BIO head genuinely beats the pinned start/end edge head, and the mechanism is
visible: 241 proposed intervals against 226 references, where the edge head proposes 291.
Most of the gain is suppressing spurious intervals.

## Round 4 — distillation from the frozen DGS BIO teacher (3 arms × 12 seeds)

The teacher labels **every** frame, which is exactly what the 61% coverage gap lacks. It is
noncausal and full-video, scored 40.32% WER frozen where the edge student scored 54.30%, and
its poses were already cached for all 1,121 records. `load_pose` resamples to the same 20Hz
clock the student runs on, so teacher frame *i* and student frame *i* are the same instant —
verified on real records (17/17, 23/23, 27/27 frames), and any record whose teacher trace did
not match the student's frame count was dropped rather than stretched.

Teacher gate: **67.8%** frame agreement with the annotation over 42,847 supervised frames;
**75.0%** of annotated sign frames recalled. Epoch selection always used the annotation, never
the teacher, so a student that merely imitates a wrong teacher cannot win on that basis.

| arm | α (teacher weight) | mean WER | SD | intervals proposed |
|---|---|---|---|---|
| bio_gap (control) | 0.0 | **71.35%** | 2.02 | 252 |
| distill_ce | 0.5 | 72.60% | 1.99 | 261 |
| distill | 1.0 | 75.33% | 3.45 | **196** |

- distill − bio_gap = **+3.98**, 95% CI [+1.72, +6.25], **p=0.003**
- distill_ce − bio_gap = +1.25, 95% CI [−0.35, +2.86], p=0.14

**Distillation significantly hurts, monotonically in α.** The failure mode is diagnostic:
`distill` proposes 196 intervals for 226 references — it *under*-proposes, where every other
arm over-proposes. The teacher is noncausal; the student is causal with a 4-frame lookahead.
Asking it to imitate decisions it cannot causally make teaches it to hedge.

## Round 5 — teacher fills only what the annotation lacks (4 arms × 12 seeds)

The teacher speaks only on frames with no hard label (the 61%) and never overrides the 39%
that has one, removing any competition with the annotation.

| arm vs control | Δ WER | 95% CI | p |
|---|---|---|---|
| distill_fill | +1.25 | [−0.75, +3.25] | 0.233 |
| distill_fill_conf (teacher p ≥ 0.9) | +1.59 | [−0.49, +3.66] | 0.151 |
| distill_fill_half | +1.14 | [−0.33, +2.61] | 0.143 |

No arm beats the control. Since removing label conflict does not rescue it, the obstacle is
**causality**, not the teacher overriding the annotation: a noncausal full-video teacher's
decisions are not expressible by a student with a 4-frame lookahead, and imitation teaches
hedging. The `bio_gap` control was bit-identical across rounds 4 and 5.

Rounds 4 and 5 both produced no gated improvement — the chain's stop condition.

## Final — the untouched held-out set (72 videos / 186 signs)

Separation verified before running, not assumed: tuning pool ∩ held-out = 0, `train` split ∩
held-out = 0, `calibration` split ∩ held-out = 0. The 9 ASLLRP eval videos that do appear in
the recipe are all in its `validation` split, which neither training nor epoch selection
reads. Five seeds fixed a priori, since no honest per-seed selection exists.

| arm | mean WER | SD | asllrp12 (24 signs) | local60 (162 signs) |
|---|---|---|---|---|
| deployed incumbent (as it ships) | 62.90% | — | 70.83% | 61.73% |
| edges, retrained here | 59.78% | 2.42 | 70.83% | 58.15% |
| **bio_gap (chain winner)** | **56.77%** | 1.81 | 69.17% | **54.94%** |

**edges − bio_gap = +3.01 points on held-out, 95% CI [0.37, 5.66], p = 0.059** — replicating
the tuning-pool estimate of +2.67 [1.35, 4.00] almost exactly. The effect generalises.

Against the checkpoint actually deployed today, `bio_gap` is **−6.13 points** (56.77% vs
62.90%), and **all five seeds beat it individually**. Roughly half that gap is the BIO
formulation and half is retraining the edge model in this harness.

The gain is concentrated in `local60` (61.73 → 54.94). On `asllrp12` the three arms are
within 1.7 points of each other on 24 reference signs — far too few to resolve anything,
which is the same resolution problem that motivated the expanded evaluation in the first
place.

## What this chain establishes

1. The TCN architecture is not the lever. Measured, not asserted.
2. The output head is a real lever worth 2.67 WER points, and the pinned start/end edge
   formulation is the worse of the two.
3. The teacher's knowledge does not transfer into a causal student by imitation, despite the
   teacher being far stronger in absolute terms.
4. The measurement itself was the hidden problem: at σ≈2.3, the two-seed protocol used before
   this chain could not have detected any of these effects.
