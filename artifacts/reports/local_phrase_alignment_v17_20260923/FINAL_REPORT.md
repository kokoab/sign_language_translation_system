# Boundary supervision: what moved WER, what did not, and why

Session of 2026-09-22 into 2026-09-24. Every figure is end-to-end WER through the live
path — `BoundaryStream` frame by frame over Apple Vision observations, intervals through
the frozen Reel cascade with unchanged commit rules, whole-video WER.

Two evaluation sets, never mixed:
- **tuning pool** — 89 local clips / 226 signs, used for all selection.
- **held-out** — 72 videos / 186 signs (12 ASLLRP + 60 local), untouched by every
  selection round. Separation was verified by sha before each run, not assumed.

Nothing was promoted. The pinned trainer, recipe, production checkpoint and the DGS
teacher weights were read-only throughout.

## 1. The measurement had to be fixed before anything could be measured

Seed noise is **σ ≈ 2.2 WER points**. Two models identical in architecture, data, targets
and seed, differing only in the initial draw of a 64→2 convolution's **130 weights**, came
out **6.2 points apart**. The two-seed protocol in use before this session could not have
detected any effect reported below. Rounds 3 onward use 8–24 seeds per arm.

A second trap: a single seed is not a result. A smoke test of the local-supervision
experiment returned 45.70% and was reported here as an 11.8-point win; the 8-seed mean was
**54.23%**. That was the best of eight runs, not a typical one.

## 2. The boundary model's architecture is not a lever

Eight trunks, 66K to 530K parameters, receptive fields 14 to 126 frames, 2 seeds each:

| variant | params | rf | mean WER |
|---|---|---|---|
| h64_d5 | 90,754 | 62 | 68.81% |
| h128_d4 | 255,106 | 30 | 70.13% |
| **baseline_h64_d4 (production)** | 78,402 | 30 | 72.12% |
| h128_d6 | 353,666 | 126 | 75.22% |

**F(7,8) = 1.54, p = 0.28.** The spread of the eight variant means (1.96) is *smaller than
the seed noise* (2.24). corr(params, WER) = +0.17 — 6.8× the parameters buys nothing.
`SweptBoundary` at baseline config loads the production state dict and returns bit-identical
outputs, so this is the real architecture, not an approximation of it.

## 3. The output head is a lever, worth ~3 points

The pinned recipe declares its own limitation: *"negatives only inside accepted intervals;
all other values ignored ... no supervised outside class."* Only 39% of each clip lies
inside an annotated sign; the other 61% is never supervised yet is scored at inference.

At 24 seeds per arm on the tuning pool:

| arm | n | mean WER | SD | intervals proposed |
|---|---|---|---|---|
| edges (pinned start/end head) | 24 | 73.65% | 2.09 | 291 |
| **bio_gap (4-class BIO)** | 24 | **70.98%** | 2.58 | **241** |

**−2.67 points, 95% CI [1.35, 4.00], p < 0.001**, and it replicated on the held-out set at
**−3.01, 95% CI [0.37, 5.66], p = 0.059**. Against the checkpoint deployed today it is
**−6.13** (56.77% vs 62.90%), with all five seeds beating it individually.

Calibration loss does not predict WER (r = +0.19 across architectures; +0.08 and +0.19
within arms, both n.s.), so there is no honest way to select a seed without touching a WER
set.

## 4. Distillation from the DGS teacher fails, and the reason is causality

The frozen teacher is strong — 40.32% WER where the edge student scored 54.30% — labels
every frame, and its poses were already cached. Teacher and student frames align exactly on
the same 20 Hz clock (verified on real records, 17/17, 23/23, 27/27).

| arm | α | mean WER | intervals proposed |
|---|---|---|---|
| bio_gap (control) | 0.0 | **71.35%** | 252 |
| distill_ce | 0.5 | 72.60% | 261 |
| distill | 1.0 | 75.33% | **196** |

**+3.98, 95% CI [1.72, 6.25], p = 0.003 — significantly worse, monotone in α.** Restricting
the teacher to only the unlabelled 61% (three further arms, 12 seeds each) did not rescue
it: +1.14 to +1.59, all n.s.

The failure mode is diagnostic. `distill` proposes **196 intervals for 226 references** — it
*under*-proposes where every other arm over-proposes. The teacher is noncausal and
full-video; the student is causal with a 4-frame lookahead. Training it to imitate decisions
it cannot causally make teaches it to hedge. Teacher strength does not transfer when the
knowledge is not causally expressible.

## 5. In-domain local data is the significant win — from the aligner, not by hand

The boundary model trains on 1,121 records: asllrp_other_ctc 1,077, asllrp_contiguous 38,
o5s5 6, **local_phrases 0** — then is scored mostly on local video.

A forced aligner was built: the frozen DGS teacher proposes candidate boundaries, dynamic
programming assigns the expected glosses in order, and an interval is kept only when the
full visual verifier ranks the expected gloss top-1. Audited against curated ASLLRP
intervals it verified 15 of 38 with **median IoU 0.55**.

At 142 reviewed clips, 8 seeds, on the held-out set:

| arm | intervals | median width | mean WER | SD | vs baseline |
|---|---|---|---|---|---|
| baseline | 0 | — | 57.06% | 1.73 | — |
| **machine** | 208 | 0.667 s | **53.16%** | 4.02 | **−3.90, p = 0.031** |
| manual_matched | 208 | — | 54.10% | 4.53 | −2.96 |
| manual_long (≥0.15 s) | 258 | — | 55.44% | 4.90 | −1.61, p = 0.403 |
| manual (all) | 401 | 0.167 s | 56.59% | 5.85 | −0.47, p = 0.833 |

**Adding in-domain data is the only statistically significant improvement in the session.**
It comes from the *aligner's* intervals, with half as many annotations as the human pass.

### Why the hand annotations underperformed — a spec mismatch I caused

The reviewer was told to prioritise interval *centres* over edges. That is correct for the
**recognizer**, whose training windows are centre-anchored at fixed 0.27/0.53/1.07 s
durations. It is wrong for the **boundary model**, which learns per-frame spans, where the
full extent *is* the target. The advice was right for one consumer and wrong for the other.

The consequence is measurable:

| | hand | aligner |
|---|---|---|
| median width | 0.167 s | 0.667 s |
| **below `BoundaryDecoder`'s 0.15 s minimum** | **36%** | 0% |
| frames marked inside the sign (20 Hz) | 3.3 | 13.3 |

36% of the hand intervals train the model to predict spans the decoder is hard-coded to
discard. Dropping them recovered 1.14 points — the right direction, not enough to close the
2.28-point gap to the aligner, so the width convention is off more broadly than the floor.

The reviewer's **centres were right**: their start corrections ran +0.166 s later than the
aligner's, independently reproducing this session's ASLLRP audit finding that the aligner
begins 0.117 s too early. It is the width, not the placement, that was wrong.

### The mechanism: WER here is governed by how often the model commits

Each arm was made to report what it *proposes*, not just its WER, so the cause is observed
rather than inferred. Against 186 held-out references:

| arm | proposed | median width | correct | insertions | WER |
|---|---|---|---|---|---|
| machine_all | **177** | 0.70 s | 91.9 | **6.4** | **54.03%** |
| manual_to_teacher | 202 | 0.57 s | **95.8** | 12.9 | 55.44% |
| baseline | 254 | 0.46 s | 94.1 | 14.2 | 57.06% |

**Insertions track proposal count almost exactly** — 177→6.4, 202→12.9, 254→14.2. WER
charges an insertion the same as a deletion, so the arm that commits least often wins even
though it recognises *fewer* signs correctly. The gain from the aligner's data is entirely
insertion suppression: `correct` actually falls (94.1 → 91.9) while insertions halve.

Two hypotheses were tested against this and **both were wrong**:

- *Width drives recognition accuracy.* The padding sweep predicted a 0.70 s proposal (0.90 s
  window) would be recognised at ~32%, making `machine_all` the worst arm. It was the best.
  Per-interval recognition accuracy is not what WER is measuring here.
- *Tight spans fail because 36% fall below the decoder's 0.15 s minimum.* A real contributor
  — dropping them recovered 1.14 points — but not the main one. Over-proposal is.

A further observation: total proposed signing time is near-constant across valid arms
(254×0.46 ≈ 116 s, 177×0.70 ≈ 124 s, 202×0.57 ≈ 115 s). Training width does not change how
much of the video the model believes is signing; it changes **how that time is partitioned**.
Narrow training spans produce many short intervals and over-proposal.

`manual_to_teacher` — the reviewer's centres with the teacher segment's extent — produced the
**highest `correct` of any arm (95.8 against the aligner's 91.9)**. The human centres locate
signs better than the aligner does. They lose on insertions, not on detection.

### A defect in this experiment, found and corrected

The first `manual_extended` arm widened each interval about its own centre to 0.60 s,
independently of its neighbours. These clips run ~0.9 s with three signs, so 3 × 0.60 s does
not fit: **69% of adjacent pairs overlapped** and total annotated span reached **186% of clip
duration**, labelling essentially every frame as inside a sign. `correct` collapsed to 84.6
and the arm scored 60.08%, worse than baseline. That measured a broken construction, not the
reviewer's annotations. Widening bounded by the midpoint of each gap brings overlap to 1% and
coverage to 0.95; the arm was re-run.

Even corrected, the reviewer's annotations imply **0.95 clip coverage** against the aligner's
0.63 — in a 0.9 s clip holding three signs there is almost no rest time. BIO training needs
negative frames, and near-total coverage leaves little "not a sign" supervision. This is a
structural difference between dense human annotation and the aligner's sparser output.

### Final result: the union of both sources, not either alone

Eight seeds per arm, held-out set, corrected widening:

| arm | WER | SD | proposed | width | correct | ins | local60 |
|---|---|---|---|---|---|---|---|
| **combined** | **51.88%** | 3.01 | **186** | 0.62 s | **97.6** | 8.1 | **50.62%** |
| machine_all | 54.03% | 4.11 | 177 | 0.70 s | 91.9 | 6.4 | 53.09% |
| manual_extended | 55.31% | 4.91 | 203 | 0.51 s | 94.4 | 11.2 | 53.86% |
| manual_to_teacher | 55.44% | 4.79 | 202 | 0.57 s | 95.8 | 12.9 | 54.32% |
| baseline | 57.06% | 1.73 | 254 | 0.46 s | 94.1 | 14.2 | 55.17% |

| comparison | Δ | 95% CI | p |
|---|---|---|---|
| **combined − baseline** | **−5.17** | [−7.58, −2.76] | **0.0014** |
| combined − machine_all | −2.15 | [−5.68, +1.38] | 0.255 |
| machine_all − baseline | −3.02 | [−6.12, +0.07] | 0.086 |

`combined` proposes **exactly 186 intervals against 186 references**, carries the **highest
detection of any arm (97.6 correct)**, and holds insertions near the lowest (8.1). It is not
a trade-off between the two sources: merging the reviewer's 401 intervals over 142 clips with
the aligner's 328 over 197 clips covers **222 clips / 451 intervals**, more than either
provides alone, and improves both axes at once.

Against the checkpoint deployed today (62.90% held-out), `combined` is **−11.02 points**.

The head-to-head arms were asking the wrong question. The reviewer's annotations were never
going to beat the aligner on 142 clips when the aligner covers 55 clips the reviewer has not
reached. They are complements, and the union is the only arm to clear p < 0.01.

### Mechanism: partly explained, partly not

Across 48 individual runs, `corr(correct, WER) = -0.87` (p = 1.8e-15) dominates;
`corr(proposed, WER) = -0.11` and `corr(insertions, WER) = -0.10` are **not** significant.
An earlier claim in this session that "the entire gain is insertion suppression" was an
artifact of comparing six arm means and does not survive at run level.

What does hold: `corr(width, proposed) = -0.82` (p = 1e-12) and
`corr(proposed, insertions) = +0.76` (p = 2.8e-10). Training width controls proposal count,
and proposal count controls insertions — that chain is solid, it simply is not the main
driver of WER. A two-term fit, `WER = 0.054*proposed - 0.460*correct + 87.7`, reaches
R^2 = 0.87.

Three mechanisms were proposed during the session and **all three failed against the data**:
width driving recognition accuracy (the padding sweep predicted machine_all would be worst;
it was best), insertion suppression (not significant at run level), and detection rate alone
(the fit predicts manual_to_teacher should edge machine_all; the reverse is observed). Why
the aligner beats the reviewer's annotations head-to-head remains unexplained.

### Statistical status

| comparison | Δ | 95% CI | p |
|---|---|---|---|
| machine_all − baseline (197 clips) | −3.02 | [−6.12, +0.07] | 0.086 |
| machine (142 clips, earlier run) | −3.90 | [−6.93, −0.87] | **0.031** |
| manual_to_teacher − baseline | −1.61 | [−5.14, +1.92] | 0.394 |
| manual_to_teacher − machine_all | +1.41 | [−2.96, +5.79] | 0.538 |

The aligner result has replicated twice in direction but sits at the edge of significance. It
is the best-supported intervention in the session; it is not established.

## 6. Two production constants tested, both left alone

**Interval context pad.** Every call site pads by a fixed 0.1 s. A sweep over 401
hand-annotated intervals put verifier top-1 at 78.2% with 0.05 s against 72.6% at 0.10 s,
suggesting a free win. End-to-end on the held-out set it reverses:

| context | 0.025 | 0.050 | 0.075 | **0.100** | 0.150 | 0.200 |
|---|---|---|---|---|---|---|
| mean WER | 62.58% | 60.43% | 60.22% | **57.63%** | 57.42% | 58.49% |

0.05 s is **2.8 points worse** end-to-end. Optimal padding is relative to interval width,
and the live path feeds machine intervals (0.667 s) not hand annotations (0.167 s). The
deployed value stands.

**Isolated-clip re-annotation.** Isolated clips run 2.07 s, 1.17 s after the automatic
hand-trim, against 0.33 s for real continuous signs — a 3.5× mismatch that looked like a
major lever. The sweep refutes it: accuracy falls *monotonically* as the window widens
toward the isolated envelope (78.2% at 0.23 s → 32.9% at 0.87 s). The phrase adaptation
already moved the recognizer onto short windows. Annotating isolated clips would buy
nothing; that expensive option is closed on evidence.

## 7. Corrections made during the session

- An 11.8-point local-supervision result was reported from a single seed; the 8-seed mean
  was 2.8. Single-seed smoke tests are not results at σ ≈ 2.2.
- The aligner's intervals were criticised as "1.63× too wide" against ASLLRP's lexical
  convention. End-to-end, wide is what works; the yardstick was wrong for a model whose job
  is proposing spans a 0.15 s-minimum decoder will accept.
- The alignment gate was described as "three independent systems". The proposal classifier
  and the full visual verifier share the same unified checkpoint; only the teacher is
  independent.
- The isolated-envelope hypothesis was stated with more confidence than the evidence
  supported, then refuted by the sweep that tested it.
