# Boundary and recognition investigation — part 2

**Date:** 2026-09-22 (evening)
**Continues:** [FINDINGS.md](FINDINGS.md), which ends at the isolated signer probe.
**Scope:** Commit-gate replay, context adaptation, crop-width sweep, hand/video diagnostics,
synthetic phrase construction and validity, a corrected data inventory, a refuted hypothesis,
and multi-source weighted selection. No training succeeded; nothing was promoted.

---

## 1. What changed since part 1

1. **Six more interventions, zero improvements.** Commit-gate rescue, BIO-preserving retrain,
   context adaptation, crop width, synthetic stitching, and an ASLLRP-subtraction hypothesis —
   all negative.
2. **The crop is not the problem.** 100 ms is optimal; every wider setting is monotonically worse.
3. **The video is not the problem.** Local video is darker but has *better* hand detection than the
   isolated clips where the same glosses work.
4. **Synthetic stitching is too easy to be useful** — and in being so, it positively identified
   coarticulation as the mechanism.
5. **Two corrections to earlier claims** (§7): the "44 phrases" figure was wrong, and the
   ASLLRP-harm hypothesis is refuted by an ablation already in the repo.
6. **A 3.01-point improvement was found sitting on disk**, invisible to single-set selection.

---

## 2. Commit-gate replay — failed

Report: [artifacts/reports/reel_rejection_replay_v17_20260922/](../reel_rejection_replay_v17_20260922/)

Part 1 recommended relaxing the commit gate. It was tested and **failed its calibration gate**.

| Policy | WER | Correct /150 | Sub | Del | Ins |
|---|---|---|---|---|---|
| Current | 49.33% | 82 | 16 | 52 | 6 |
| Verifier-aware rescue | 48.67% | 86 | 16 | 48 | **9** |
| Unfiltered verifier (diagnostic) | 56.67% | 102 | 28 | 20 | **37** |

Rescue recovered 4 correct signs and added 3 insertions — not a clean win. Removing acceptance
wholesale is far worse.

**This also corrects part 1's attribution.** First-blocking-stage counts:

```
proposal_rejected_first            42
verifier_rejected_after_proposal   18
commit_score_after_both_accepted   12
```

The commit *score* threshold — the thing part 1 pointed at — is the first blocker in only 12 cases.
Most rejections happen earlier, at the proposal classifier.

The rescued examples show why relaxation backfires: `HELLO HOW YOU` became
`HELLO HOW HOW HOW YOU`. The boundary model's interval over-supply (227 proposals for 162 signs)
means a looser gate admits duplicates. **The gate was partly compensating for over-segmentation.**

---

## 3. Reel context adaptation — selected epoch 0

Report: [artifacts/reports/reel_context_adapt_v17_20260922/](../reel_context_adapt_v17_20260922/)
Recipe SHA `da9fa22ac2c0c84d…`

Trained the proposal classifier and verifier fusion heads on 672 ASLLRP continuous events at
0/100/200 ms context, frozen encoders, isolated replay plus KL to original logits.

| | WER | Correct | Sub | **Del** | Ins | Exact | Retained |
|---|---|---|---|---|---|---|---|
| Baseline (frozen) | **49.33%** | **82** | 16 | **52** | **6** | 14 | 82/82 |
| Best trained | 51.33% | 81 | 17 | **52** | 8 | 11 | 76/82 |

14 epochs in **2.3 seconds** of fitting after 20.7 minutes of preparation.

**Deletions never moved: 52 → 52** (per-epoch 53, 53, 56, 52). 35% of reference signs are never
emitted and head adaptation did nothing to that. Substitutions went 16 → 17, insertions 6 → 8.

The recipe worked mechanically — isolated accuracy held (Citizen 95.2/96.0% → 94.7/96.0%,
SemLex 85.3/89.2% → 85.7/88.8%), so replay and KL prevented forgetting exactly as designed.
Training loss fell 2.65 → 2.23. Nothing transferred.

### The SCHOOL → STOP confusion

In 8 `TOMORROW SCHOOL GO` clips, SCHOOL is never committed:

```
TOMORROW SCHOOL GO  →  ['GO']
TOMORROW SCHOOL GO  →  ['TOMORROW']
TOMORROW SCHOOL GO  →  ['STOP']
TOMORROW SCHOOL GO  →  ['STOP', 'GO']
TOMORROW SCHOOL GO  →  ['TOMORROW', 'STOP', 'GO']
TOMORROW SCHOOL GO  →  ['EAT', 'FEEL', 'WRITE', 'GO']

committed in SCHOOL clips: GO ×4, TOMORROW ×3, STOP ×3, FRIEND, EAT, FEEL, WRITE
```

SCHOOL isn't missed — it is systematically recognized as **STOP**. Both are flat-hand-to-palm
contact signs; SCHOOL is the two-hand clap, STOP the one-hand chop.

---

## 4. Crop-width sweep — 100 ms is already optimal

Script: session scratchpad. Report: [artifacts/reports/context_width_sweep_v17_20260922/](../context_width_sweep_v17_20260922/)

89-clip tuning pool (226 reference signs), frozen BIO intervals, frozen Reel, identical commit
rules; only `classify_interval` context varies.

| Context | WER | Correct | Sub | Del | Ins | Exact |
|---|---|---|---|---|---|---|
| **100 ms** | **47.35%** | **128** | 21 | 77 | **9** | 22/89 |
| 150 ms | 52.65% | 126 | 20 | 80 | 19 | 17/89 |
| 200 ms | 53.10% | 125 | 23 | 78 | 19 | 15/89 |
| 300 ms | 57.96% | 119 | 21 | 86 | 24 | 12/89 |
| 400 ms | 64.60% | 99 | 26 | 101 | 19 | 4/89 |

Monotonically worse; insertions roughly double from 150 ms. **SCHOOL: 0%, 0%, 0%, 8%, 0%** — it
never recovers at any width. The crop is not cutting off SCHOOL's evidence.

Per-gloss, the failures are width-independent: PLEASE stuck at 17–22%, THANKYOU degrading
35% → 6%, while MORNING holds 100% and NAME 94% at every width.

---

## 5. Hand and video diagnostics

### 5.1 Hand crop and detection, 89 local clips

Report: [artifacts/reports/hand_crop_diagnostic_v17_20260922/](../hand_crop_diagnostic_v17_20260922/)
Config: `hand_box_scale 1.70`, `minimum_box_long_side_fraction 0.14`, `minimum_joint_count 5`
([schema_hand_rgb_v17.py](../../../active/v17/schema_hand_rgb_v17.py))

| Phrase | Both hands | No hand | Box clipped | Sign recall |
|---|---|---|---|---|
| PLEASE HELP I | 33.8% | 21.0% | **25.2%** | 47% |
| **TOMORROW SCHOOL GO** | 45.5% | 24.6% | **18.4%** | **29%** |
| GOOD MORNING | 63.0% | 27.5% | 11.3% | 75% |
| HELLO HOW YOU | 27.9% | 15.1% | 8.5% | 58% |
| MY NAME | 43.1% | 29.4% | 6.5% | 79% |
| THANKYOU FRIEND | 45.9% | 22.4% | 2.3% | 55% |

Caveat: "clipped" means the **1.7×-padded box** exceeds the frame, not that the hand is cut off —
an upper bound, not true hand cutoff.

### 5.2 Local video vs isolated video — the hypothesis fails

Report: [artifacts/reports/video_quality_compare_v17_20260922/](../video_quality_compare_v17_20260922/)
Same detector, same settings, same 15 glosses.

| Metric | Local phrases | Isolated (Citizen+SemLex) |
|---|---|---|
| Resolution | 640×480 | 662×488 |
| **Mean brightness** | **91.97** | **118.45** |
| Contrast (std) | 48.84 | 62.60 |
| **Fraction pixels < 40** | **27.2%** | 17.3% |
| Hand size / frame | **0.187** | 0.171 |
| **Hand confidence** | **0.625** | 0.499 |
| **Both hands detected** | **39.5%** | 13.9% |
| **No hand detected** | **22.7%** | 59.5% |

**Local video is darker — and its hands are detected better on every metric.** Darkness is not
preventing detection. The data-quality hypothesis does not explain SCHOOL.

(Confound: isolated clips have rest lead-in/lead-out frames, inflating their no-hand rate. It
cannot rescue the hypothesis — local would need to be *worse*, and it is better throughout.)

---

## 6. Synthetic phrases — built, and too easy

Builder: [scripts/build_synthetic_phrases_v17.py](../../../scripts/build_synthetic_phrases_v17.py)
Data: `data/local/synthetic_phrases_v17_20260922/` (102 MB)
Report: [artifacts/reports/synthetic_phrases_v17_20260922/](../synthetic_phrases_v17_20260922/)

Built at explicit user request for personal experimentation, overriding ASL Citizen's documented
caution against treating concatenated clips as continuous signing.

**139 phrases, 39 distinct signers, 333 reference signs.** Within-signer only — each phrase is one
real person performing every sign, so no identity change mid-phrase. Rest frames trimmed by
frame-difference threshold (58% of source frames kept). Hard cuts, no transition synthesis.
Durations 2.4–4.1s against real local phrases' 4.09s mean.

### Validity check — oracle intervals, frozen Reel

```
WER 18.62%   correct 271/333   sub 18   del 44   ins 0   exact 85/139
ALL:  proposal 90%   verifier 92%   commit 87%   commit_ok 81%
SCHOOL: proposal 94%   verifier 100%   commit_ok 82%
```

| Condition | Intervals | Overall | SCHOOL |
|---|---|---|---|
| Isolated clips | whole clip | 64% commit-ok | 50% |
| **Synthetic phrases** | **oracle** | **81% commit-ok** | **82%** |
| Real continuous (ASLLRP) | **oracle** | 67% verifier-correct | — |
| Real local phrases | BIO-predicted | 47.35% WER | **0%** |

**The synthetic data is easier than the isolated clips it was built from**, because rest-trimming
removed ambiguous lead-in and left only clean canonical articulation.

**Conclusion — coarticulation is positively identified as the mechanism.** With segmentation held
perfect in both cases, synthetic scores 92% verifier-correct and real continuous 67%. Adjacency is
not what makes continuous signing hard; articulation deformation is, and hard-cut stitching cannot
produce it.

It also separates two things that looked alike:
- **SCHOOL is not a weak sign** — 94–100% synthetic, 67% isolated. It fails only under real coarticulation.
- **THANKYOU (46%) and GOOD (64%) are genuinely weak** — poor on synthetic and isolated alike.

**No model was trained on this data.** The validity check is read-only.

---

## 7. Two corrections

### 7.1 "Only 44 continuous phrases" — wrong

That figure is `asllrp_contiguous` train phrases only. The real inventory, from
[confident_supervision.json](../confident_supervision_v17_20260920/confident_supervision.json):

- **1,017 known sign events in continuous video, 65 distinct glosses**
- 753 videos with ≥1 known sign; **147 videos with ≥2** (asllrp_other_ctc 121, asllrp_contiguous 20, o5s5 6)
- Plus `ncslgr_strict` (88 train / 37 validation), a continuous source not catalogued in part 1

Full continuous inventory: asllrp_contiguous 44 + asllrp_other_ctc 879 + ncslgr 88 + local 287.
The distinction that matters is **complete-transcript phrases** (needed for WER training) versus
**individually labelled signs in continuous context** (usable for recognition heads). The latter is
plentiful; `reel_context_adapt` already used 672 of them.

### 7.2 "ASLLRP continuous data is hurting local performance" — refuted

The correlation is real:

| Gloss | ASLLRP continuous events | local60 recall |
|---|---|---|
| FRIEND | 232 | 17% |
| SCHOOL | 52 | 0% |
| TOMORROW | 24 | 67% |
| I, MORNING, MY, YOU, PLEASE | 0 | 50–100% |

```
corr(events, recall)      = -0.66
corr(log events, recall)  = -0.73
with ASLLRP (≥5):  57% mean recall
without (<5):      83% mean recall
```

**But the ablation already exists in the repo and says the opposite:**

| Model | `asllrp_other_ctc` | known WER | exact |
|---|---|---|---|
| `unified_streaming_grounded_ctc_v17_v1` (with) | 879 | **42.02%** | **59/249** |
| `unified_streaming_grounded_ctc_v17_no_asllrp_other` | — | **49.02%** | 47/249 |

Removing it costs **7 WER points and 12 exact phrases**. ASLLRP continuous data helps.

The correlation is confounded by **sign complexity**. The zero-coverage glosses — I, MY, YOU
(pointing pronouns), MORNING (one fixed movement) — are the simplest in the vocabulary and would
score high regardless. And the counterexample was in the table: **PLEASE has 0 ASLLRP continuous
events and scores 50%.** The "SCHOOL is easy so it can't be intrinsically hard" argument also fails,
because that 94% was measured on *Citizen* video, not on local signers' SCHOOL.

---

## 8. Prior art already in the repo

`docs/ground_truth/stage1-architecture/high.md`, **2026-08-16**:

- **v2 contextual adaptation:** 254 signer-held-out contextual signs, v1 37.40% WER → **v2 21.26%**.
  *"confirms within-ASLLRP signer generalization… but does not override the unchanged **50.0% WER on
  12 real ASLLRP phrases**."*
- **v3 synthetic window phase** (isolated signs crossing 32-frame boundaries): regressed to
  **54.17%** phrase WER. **Rejected.**
- Recorded conclusion: *"the remaining limitation is **scarce genuine continuous phrase
  supervision**… the next defensible improvement requires **new fully labelled continuous utterances
  or phrases, not more isolated-window augmentation**."*

Today's synthetic validity result re-derives that conclusion independently, by a different route.

The extraction-level generalization mechanism is `body_relative_normalize`
([geometry_v17.py:121](../../../active/v17/geometry_v17.py#L121)) — coordinates normalized to body
scale and centre, factoring out signer size and camera distance. That is why signer identity is not
the bottleneck and why Citizen isolated sits at 96%.

---

## 9. Multi-source weighted selection

Script: [scripts/score_multisource_v17.py](../../../scripts/score_multisource_v17.py)
Report: [artifacts/reports/multisource_selection_v17_20260922/](../multisource_selection_v17_20260922/)

### Why single-set selection fails

By-source validation of `unified_streaming_grounded_ctc_v17_v1`:

| Source | Samples | Exact | WER |
|---|---|---|---|
| **local_phrases** | 200 | 26.0% | **37.78%** ← easiest |
| asllrp_contiguous | 12 | 25.0% | 45.83% |
| asllrp_other_ctc | 225 | 12.9% | 77.46% |
| ncslgr_strict | 37 | 10.8% | **86.00%** |
| *isolated* | 1,356 | 81.6% | 18.36% |

**Local is the easiest continuous source, not the hardest.** The case for broadening is coverage
(15 of 65 glosses, blind to the 77–86% sources) and resolution (59 clips cannot resolve candidates),
not fairness.

### Design

```
local_phrases 0.40, asllrp_contiguous 0.20, asllrp_other_ctc 0.20, ncslgr_strict 0.20
```

Weighted **macro**, not pooled, so the 12-sample source stays visible. Local carries the largest
single share as deployment domain, but the other three together outweigh it. **Isolated is a
retention constraint, never an aggregate term** — 1,356 samples would swamp 249 phrases. Per-source
gates: no source may lose >3 WER points, isolated >1 point.

### Result — a 3-point improvement was on disk

| Checkpoint | Weighted | Isolated | Gates |
|---|---|---|---|
| **`unified_streaming_aligned_grounded_v17_v1`** | **53.96%** | 82.4% | **PASS** |
| `aligned_grounded_seed17082` | 56.40% | 82.2% | FAIL asllrp_contiguous +8.3pt |
| `grounded_ctc_v17_v1` *(reference)* | 56.97% | 81.6% | — |
| `native_ctc_no_blank` | 57.01% | 83.2% | FAIL local +7.4pt |
| `grounded_ctc_v17_no_asllrp_other` | 58.00% | 84.7% | FAIL local +7.2pt |

The reference model ranks **14th of 26**. `aligned_grounded_v17_v1` passes every gate at **−3.01
weighted points**, with better isolated retention.

**Caveat:** the nine checkpoints scoring 36–51% lack `ncslgr_strict` coverage, so their weights
renormalize over three sources excluding the hardest. They correctly fail on `no coverage`, but
their headline numbers are not comparable. **Only the 4-source models compare directly.**

---

## 10. Recommended recipe

**Free, measured, no training:**

1. **Switch to `unified_streaming_aligned_grounded_v17_v1`** — −3.01 weighted WER, passes all gates.
2. **Use `score_multisource_v17.py` as the selection gate** — stops real gains being rejected on
   59-clip noise, which happened at least twice today.
3. **Boundary: use the frozen backbone with the original BIO head and decoder. Do not adapt it.
   Drop the START/END edge head.**

| Config | Pooled WER (186 signs) | Correct |
|---|---|---|
| Frozen BIO + original decoder | 39.78% | 127 |
| **finetune ep7 + original BIO** | 40.32% | **130** |
| finetune ep7 + edge head | 54.30% | 96 |

The edge head costs 14 WER points at identical weights.

**On a 25% target.** Local sits at 37.78%. ~35% of reference signs are deletions, and that number
did not move under any of the eight interventions across both parts. The oracle bound is binding:
with **perfect** boundaries on ASLLRP12, WER was still 50% (16/24 identified). Segmentation cannot
reach 25%; recognition must. Steps 1–3 plausibly reach the low 30s on local. Closing to 25% requires
the deletions, which require continuous phrase video in the target domain **from more than three
signers** — the one intervention nothing has ruled out.

---

## 11. Artifacts produced in part 2

| Path | Description |
|---|---|
| [scripts/build_synthetic_phrases_v17.py](../../../scripts/build_synthetic_phrases_v17.py) | Within-signer phrase stitcher; 139 phrases, 39 signers |
| [scripts/score_multisource_v17.py](../../../scripts/score_multisource_v17.py) | Weighted multi-source scoring with per-source no-regression gates |
| [artifacts/reports/context_width_sweep_v17_20260922/](../context_width_sweep_v17_20260922/) | Crop-width sweep, per-gloss |
| [artifacts/reports/hand_crop_diagnostic_v17_20260922/](../hand_crop_diagnostic_v17_20260922/) | Hand detection and box clipping per phrase |
| [artifacts/reports/video_quality_compare_v17_20260922/](../video_quality_compare_v17_20260922/) | Local vs isolated brightness, contrast, hand metrics |
| [artifacts/reports/synthetic_phrases_v17_20260922/](../synthetic_phrases_v17_20260922/) | Manifest and validity results |
| [artifacts/reports/multisource_selection_v17_20260922/](../multisource_selection_v17_20260922/) | Ranking of 26 checkpoints |

**Standing limitations:** familiar-signer reused development pools throughout; one seed per
checkpoint; 15 glosses over 6 phrase types on local; cached-pose offline composition, not live
latency; no protected test accessed; nothing promoted; the synthetic set is experimental data that
must never enter a real-data evaluation.

---

## 12. Corpus motion comparison and the pretraining gate (late addition)

### 12.1 Does the ungated data look like the target domain?

Report: [artifacts/reports/corpus_motion_compare_v17_20260922/](../corpus_motion_compare_v17_20260922/)

Hand-velocity statistics from already-extracted v17 61-point landmarks, per-clip
median-normalized so different extractors compare. Low-percentile velocity is the
coarticulation signature: continuous signing rarely comes to a full stop between signs.

| Corpus | v_p10 | v_p25 | near-zero frames | jerk |
|---|---|---|---|---|
| **youtube_asl** | **0.227** | **0.455** | **7.6%** | 0.560 |
| **local_phrases** | **0.212** | **0.451** | **7.0%** | 0.493 |
| asllrp_other_ctc | 0.215 | 0.445 | 6.8% | 0.835 |
| how2sign | 0.269 | 0.497 | 4.5% | 0.404 |

**YouTube-ASL is the closest match to the target domain of any corpus measured** — v_p10
within 7%, v_p25 within 1%, near-zero frames within 0.6 points. How2Sign is measurably
smoother with fewer stops (studio/prepared content). ASLLRP matches on velocity but is
nearly twice as jerky.

This is the property synthetic stitching failed to reproduce (§6), and it is present in
data already on disk — 1,411 clips downloaded, no authorization needed.

Caveat: aggregate velocity statistics, not coarticulation measured at gloss boundaries
(YouTube-ASL has no gloss timing, so that cannot be measured directly). Necessary, not
sufficient.

### 12.2 Schema compatibility audit

YouTube-ASL transition landmarks versus Citizen isolated landmarks:

| Field | Citizen (isolated) | YouTube-ASL |
|---|---|---|
| schema_name / version | `slt_apple_vision_landmarks_v17` / 1 | identical |
| dtype / shape | float16 / `[32, 61, 5]` | identical |
| feature_channels, coordinate_contract | — | identical |
| **all 61 node_names, in order** | — | **identical** |
| `maximum_source_frames` | 96 | **32** |
| `trim_to_hand_activity` | True | **False** |

Layout is directly compatible. The two config differences are real and must be carried
into any fine-tune: YouTube windows sample from fewer source frames and are **not**
trimmed to hand activity, so they retain transition and non-signing frames. For
coarticulation pretraining that is an advantage; for distribution match with Stage 1 it is
a difference to account for.

### 12.3 Why encoder pretraining was NOT launched unsupervised

[model_unified_multimodal_v17.py:95](../../../active/v17/model_unified_multimodal_v17.py#L95):
*"One classifier graph containing frozen landmark/hand temporal encoders."*

The temporal encoders are frozen by architecture. Retraining them invalidates Stage 1, the
Reel proposal/verifier heads, Stage 2 CTC and every CoreML export. That is a reviewed-contract
change, not an overnight run.

### 12.4 Gate experiment: standalone masked-frame reconstruction

Script: [scripts/pretrain_motion_probe_v17.py](../../../scripts/pretrain_motion_probe_v17.py)
Report: [artifacts/reports/motion_pretrain_probe_v17_20260922/](../motion_pretrain_probe_v17_20260922/)

Trains a **standalone** temporal encoder (not the v17 encoder, no existing weights touched)
to reconstruct 6 contiguous masked frames of hand motion, held out **by source video**.

Baselines on the validation split:

```
copy_previous  0.533
linear_interp  0.184   <- the real bar
PASS requires  < 0.147  (20% better than the best trivial baseline)
```

928 windows from 127 source videos, 25 videos held out. Linear interpolation across a
6-frame gap in smooth hand motion is strong; beating it by 20% requires learned dynamics,
not smoothness.

**Interpretation when it lands.** PASS means the corpus carries temporal structure worth
pretraining on, justifying a reviewed contract to pretrain the real encoder. FAIL means the
corpus offers nothing beyond smoothness and the pretraining plan should be dropped — one
night spent instead of a week.

Note: only 127 of the 1,411 downloaded clips have transition landmarks extracted. If the
gate passes, extracting the remainder is the next step.

### 12.5 Gate result — FAIL by the stated bar, but the trend says data-starved

Mask-span sweep, 120 epochs each, 928 windows from 127 source videos, held out by video:

| Mask span | Model best val MSE | `linear_interp` | `copy_previous` | Model vs interp |
|---|---|---|---|---|
| 6 frames (~200 ms) | 0.351 | 0.184 | 0.533 | **−91%** |
| 12 frames (~400 ms) | 0.554 | 0.382 | 0.943 | **−45%** |
| 18 frames (~600 ms) | 0.699 | 0.613 | 1.425 | **−14%** |

The model never beat linear interpolation, so the gate **fails** as stated. But the gap closes
monotonically as the mask widens — −91% → −45% → −14%. At short spans interpolation is
near-optimal on smooth hand trajectories and is effectively unbeatable; at 600 ms it is only 14%
ahead and falling.

Supporting evidence for data starvation rather than absence of structure:
- **train 0.346 ≈ val 0.358** at span 6 — underfitting, not overfitting
- validation was still improving at epoch 120
- 753 training windows for a ~1.8M-parameter transformer

**Verdict: inconclusive, leaning positive on structure, blocked on data volume.** The corpus
plausibly carries learnable temporal structure beyond smoothness; this extraction is too small to
demonstrate it. Do NOT start encoder pretraining on this evidence, and do not abandon the plan on
it either.

### 12.6 Why more data was not extracted

Only 127 of the downloaded clips have v17 transition landmarks. Two possible routes, neither
started:

1. `scripts/acquire_youtube_asl_transition_voices_v17.py` downloads more YouTube video. AGENTS.md
   records "No acquisition is authorized" and `download_status.json` reads
   `paused_by_user … no further downloads authorized now`.
2. The 1,411 already-downloaded LINDAT clips are **keypoint JSON in a different layout**
   (33 pose + 21 + 21 hands + 478 face slots, 553 possible points) — not the 61-point Apple Vision
   v17 layout the transition landmarks use. Using them requires the conversion the 2026-09-21
   report scoped ("map pose 33 + hands 42 + the existing 53-face subset into the shared 128-point
   layout"), which is a build with its own parity checks.

**Next safe action for this thread:** decide between authorizing route 1 or building route 2's
converter, then re-run `scripts/pretrain_motion_probe_v17.py` at `MASK_SPAN=18`. If the −14% gap
turns positive with roughly 10× the windows, encoder pretraining is justified and earns a reviewed
contract. If it stays negative, drop the plan.
