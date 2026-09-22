# Boundary adaptation investigation — computations and findings

**Date:** 2026-09-22
**Scope:** Three boundary fine-tuning runs, an expanded held-out evaluation, a decoder-separation
diagnostic, and a train/held-out fit analysis. No promotion, no deployment, no protected test access.
**Status:** All numbers below are measured, not estimated, unless explicitly marked as a hypothesis.

---

## 1. Summary of conclusions

1. **The boundary model was never the bottleneck.** Three separate adaptation recipes produced no
   reliable improvement over the un-adapted frozen backbone.
2. **The 24-sign evaluation set could not resolve the models.** A gap that looked decisive on 24 signs
   (62.50% vs 70.83% WER) collapsed to 0.5 points on 186 signs.
3. **The largest regression came from the replacement readout, not the weights.** Swapping the
   START/END edge head back to the original BIO head recovered ~14 WER points at identical weights.
4. **Two distinct failure modes, previously conflated:**
   - Boundary model: *underfits* — 58% positive precision on its own training data.
   - Recognizer: *memorizes* — 232/232 exact on training phrases, 30/199 on validation.
5. **Stage 1 isolated is healthy** (87.57% top-1, signer-disjoint) and is not the failing component.
   The failure is isolated → continuous transfer.
6. **The commit gate is the largest single measured loss.** On clean isolated Citizen clips the
   verifier is **96%** correct but only **58%** commit; 27% of correctly-identified clips are discarded
   at a median verifier score of **0.908**. It is controlled by arguments, not learned parameters.
7. **Signer diversity is not the binding constraint.** 96% accuracy across ~19 signers per gloss on
   Citizen. The 88 unused signers would add variation the model already handles.

---

## 2. Training runs observed

### 2.1 `pretrained_boundary_finetune_v17_20260922`

Report: [artifacts/reports/pretrained_boundary_finetune_v17_20260922/](../pretrained_boundary_finetune_v17_20260922/)
Recipe: [active/v17/pretrained_boundary_recipe_20260922.json](../../../active/v17/pretrained_boundary_recipe_20260922.json) — SHA `6e80b22a8be08d9a…`
Trainer: [scripts/train_pretrained_boundary_v17.py](../../../scripts/train_pretrained_boundary_v17.py)

| Field | Value |
|---|---|
| Seed | 17621 (seed 17622 never run) |
| Trainable parameters | 4,727,042 (all 4 attention blocks + edge head) |
| Epochs run | 40 |
| Selected epoch | **7** |
| Calibration loss | **0.6660612386095304** |
| Validation loss | 0.8307688690306854 |
| Elapsed (seed 1) | 4,688.85 s |
| Stop reason | `calibration_plateau_after_minimum40` |

Loss trajectory ([history_17621.json](../pretrained_boundary_finetune_v17_20260922/history_17621.json)):

| Epoch | Train loss | Calibration loss |
|---|---|---|
| 1 | 0.7588 | 0.7503 |
| 4 | 0.6975 | 0.7257 (head-only phase best) |
| 6 | 0.6835 | 0.6807 (attention unfrozen) |
| **7** | **0.6158** | **0.6661 ← best** |
| 10 | 0.4628 | 0.7002 |
| 20 | 0.1450 | 1.3719 |
| 40 | **0.0654** | **2.1353** |

Calibration loss rose monotonically from epoch 8 onward. The 40-epoch minimum in
`should_stop(epoch, best_epoch)` ([active/v17/pretrained_boundary_v17.py:27](../../../active/v17/pretrained_boundary_v17.py#L27))
forced 33 epochs of discarded work per seed.

**Run was stopped manually at 15:51** during seed 17622 epoch 1, after seed 17621's record was written.
`completion.json` correctly records `status: "failed"` with a `KeyboardInterrupt` traceback.
Nothing was corrupted; atomic writes held.

### 2.2 `pretrained_boundary_augmented_v17_20260922`

Report: [artifacts/reports/pretrained_boundary_augmented_v17_20260922/](../pretrained_boundary_augmented_v17_20260922/)
Recipe SHA: `a80d7c7683ce469b…`

Changes vs 2.1: patience 8 with **no 40-epoch floor**, LR plateau patience 8→3, started from original
pretrained weights (not the overfit checkpoint), and acquisition-rate augmentation — 15–20 Hz past-only
sensor sampling plus 0–15% observation dropout with last-value hold, applied to raw poses before
normalization. Two precomputed variants, one drawn per example per epoch. Labels and timestamps never move.

| Field | Value |
|---|---|
| Epochs run | 16 |
| Selected epoch | **8** |
| Calibration loss | **0.652260742764366** |
| Validation loss | 0.8188406560122092 |
| Elapsed (training) | 1,497.21 s |
| Elapsed (total incl. cache + eval) | 2,230.48 s |
| Stop reason | `calibration_plateau` |

Head-to-head loss at matched epochs:

| Epoch | Original cal. | Augmented cal. |
|---|---|---|
| 7 | 0.6661 | 0.6580 |
| 8 | 0.6858 | **0.6523** |
| 9 | 0.6849 | 0.6621 |
| 16 | — | 0.8157 |

Train loss at epoch 9: **0.5552 augmented vs 0.5086 original** — the augmentation measurably slowed
memorization. It delayed the divergence by exactly one epoch; it did not prevent it.

### 2.3 `pretrained_bio_final_v17_20260922`

Report: [artifacts/reports/pretrained_bio_final_v17_20260922/](../pretrained_bio_final_v17_20260922/)
Recipe SHA: `da9fa22ac2c0c84d…`

Trains only the final attention block and the **original BIO head** (1,330,948 parameters), preserving
the original decoder. Selection on full-video WER rather than frame BCE. Cross-entropy on known targets
plus KL to the original BIO distribution as a regularizer.

**Result: selected epoch 0 — the frozen fallback. Training did not qualify.**

| Epoch | Calibration WER | Correct /150 | Exact /59 | Retained /82 |
|---|---|---|---|---|
| 0 (frozen) | **49.33%** | 82 | 14 | 82 |
| 4 | 48.00% | 83 | 14 | 73 |
| 8 | **47.33%** | 84 | 15 | 74 |
| 12 | 48.67% | 83 | 14 | 73 |

Epoch 8 had *better raw WER* than frozen but was rejected on the retention constraint — it lost 8
previously-correct signs while gaining others. Confirmation on the 30-clip set returned identical
results for frozen and selected: 43.42% WER, 46/76 correct, `improved: false`.

---

## 3. Evaluation sets

| Set | Videos | Reference signs | Provenance |
|---|---|---|---|
| `asllrp12` | 12 | 24 | `asllrp_contiguous` role=validation; the frozen comparison set |
| `local60` | 60 | 162 | signer_02 clips still held out by the current versioned split |
| `combined186` | 72 | 186 | the two above, pooled with an explicit label |
| `calibration59` | 59 | 150 | new-run selection set |
| `confirmation30` | 30 | 76 | new-run confirmation set, used once |
| *excluded* | 139 | 375 | signer_02 clips moved into training; Reel-contaminated |

**Contamination audit.** Both boundary checkpoints were trained solely on `asllrp_other_ctc` (1,077),
`asllrp_contiguous` (38) and `o5s5` (6) poses — **zero `local_phrases`**. All 199 local clips are clean
with respect to the boundary models. The 139 moved-to-train clips were excluded because the frozen Reel
classifier may have seen them; Reel is the shared constant across arms, so it cannot bias the
*comparison*, but it would inflate absolute WER.

Held-out set derived from
[artifacts/reports/local_familiar_signer_v17_20260922/split_manifest.json](../local_familiar_signer_v17_20260922/split_manifest.json)
by `experiment_role == 'validation'`. Verified **zero overlap** with the new run's 89 calibration +
confirmation clips.

**Standing limitation:** `local_signer_02` appears in training under a different set of clips. This is a
familiar-signer reused development pool — *not* unseen-signer, not protected test. The split manifest
says so explicitly.

---

## 4. Expanded evaluation results

Script: [scripts/evaluate_boundary_expanded_v17.py](../../../scripts/evaluate_boundary_expanded_v17.py) (written for this investigation)
Report: [artifacts/reports/boundary_expanded_eval_v17_20260922/](../boundary_expanded_eval_v17_20260922/)

**Harness validation:** `asllrp12` reproduces the standalone runs exactly — WER 75.00 / 62.50 / 70.83 and
`boundary_recall_200ms` 0.667 / 0.708 / 0.583. The new code measures the same thing.

| Arm | ASLLRP12 (24) | local60 (162) | Combined (186) |
|---|---|---|---|
| Reel baseline, no boundary model | 83.33% — 4/24 | — | — |
| TCN (`asl_temporal_boundary_v17`) | 79.17% — 5/24 | — | — |
| **frozen_pretrained_bio** | 75.00% — 6/24 | **34.57% — 121/162** | **39.78% — 127/186** |
| finetune_epoch7 (edge head) | **62.50% — 9/24** | 53.09% — 87/162 | 54.30% — 96/186 |
| augmented_epoch8 (edge head) | 70.83% — 7/24 | 51.23% — 93/162 | 53.76% — 100/186 |
| **finetune_epoch7 + original BIO** | 70.83% — 7/24 | 35.80% — 123/162 | 40.32% — **130/186** |
| augmented_epoch8 + original BIO | 75.00% — 6/24 | 37.65% — 118/162 | 42.47% — 124/186 |

Error breakdown, combined 186:

| Arm | WER | Correct | Sub | Del | Ins | Exact /72 |
|---|---|---|---|---|---|---|
| frozen_pretrained_bio | 39.78% | 127 | 13 | 46 | 15 | 17 |
| finetune_epoch7 | 54.30% | 96 | 24 | 66 | 11 | 5 |
| augmented_epoch8 | 53.76% | 100 | 24 | 62 | 14 | 2 |
| finetune_epoch7_original_bio | 40.32% | **130** | 15 | 41 | 19 | **18** |
| augmented_epoch8_original_bio | 42.47% | 124 | 15 | 47 | 17 | 17 |

**Pooled vs unweighted average** — these disagree, and pooled should be preferred because the unweighted
mean gives a 24-sign set equal voice to a 162-sign set:

| Arm | Pooled WER | Unweighted mean |
|---|---|---|
| frozen_pretrained_bio | **39.78%** | 54.78% |
| finetune_epoch7 + original BIO | 40.32% | **53.32%** |
| augmented_epoch8 + original BIO | 42.47% | 56.33% |
| augmented_epoch8 (edges) | 53.76% | 61.03% |
| finetune_epoch7 (edges) | 54.30% | 57.79% |

### Key reading

- **The 24-sign ranking does not survive.** finetune vs augmented: 8.3 WER points apart on 24 signs,
  0.5 points apart on 186. The subset ordering even flips (finetune wins ASLLRP, augmented wins local).
- **Frozen and finetune+BIO are a statistical tie.** finetune+BIO recognizes *more* signs correctly
  (130 vs 127) and has more exact videos (18 vs 17); it loses on WER only via higher insertions.
- **The edge head costs ~14 WER points** at identical weights.

Metric caveat: `curated_manifest.json` has boundary annotations only for `asllrp_contiguous`,
`asllrp_other_ctc` and `o5s5` — **not** `local_phrases`. So `boundary_recall_200ms`,
`wholly_gap_contained_commits` and retention are ASLLRP-only; the local subset carries an explicit
`interval_metrics: unavailable` marker rather than a misleading zero.

Reporting defect found and fixed after the run: `boundary_candidates` accumulated across all 72 videos
while printed under the ASLLRP heading. Recall figures were unaffected (numerator and denominator were
both ASLLRP-only, which is why they reproduce the originals exactly).

---

## 5. Decoder-separation diagnostic

Report: [artifacts/reports/boundary_adaptation_diagnostic_v17_20260922/](../boundary_adaptation_diagnostic_v17_20260922/)

Verified by **exact tensor equality** that both adapted checkpoints retain an unchanged temporal CNN,
input normalization and original BIO head — only attention and the edge head differ. That makes the
readout swap a clean comparison.

**Conclusion: the regression is the readout/decoding path, not the weights.**

Decoder differences:

| | Original BIO | Replacement edge |
|---|---|---|
| Head | `sign_bio_head`, 4-class | `edge`, 2 logits (START/END) |
| Decoding | groups adjacent B/I, accepts 3-frame spans, closes at EOF | requires rising START/END pairs |
| Constraints | — | 0.15–4.0 s duration limits; drops unfinished EOF intervals |

Training optimizes masked weighted frame BCE — not these discrete interval and commit decisions.

---

## 6. Where signs are actually lost

### 6.1 Interval supply and commit accounting (frozen arm)

| Set | Reference signs | Intervals proposed | Committed |
|---|---|---|---|
| asllrp12 | 24 | **19** (under-supply) | 7 |
| local60 | 162 | **227** (40% over-supply) | 148 |

On `local60` the segmenter proposes far more candidates than there are signs. Segmentation is not the
constraint there. On `asllrp12` it under-segments, so it *is* a partial constraint on that set.

Of the 79 rejected intervals on `local60`:
- **51 had verifier `accepted: True`** and were blocked by the commit-score threshold / `VerifiedCommitLock`
- Proposal/verifier gloss agreement among rejected: **35.4%** (58.3% on asllrp12)
- Verifier score: median **0.345** for rejected vs **0.861** for accepted

### 6.2 Accuracy decomposition (ASLLRP12)

| Stage | Correct | Loss attributed |
|---|---|---|
| Stage 1 on isolated clips | ~87% | — |
| + continuous signing, oracle boundaries (`verifier_correct`) | 16/24 = 67% | −20 pts: coarticulation |
| + commit gate | 12/24 = 50% | −17 pts: thresholds |
| + real predicted boundaries | 6/24 = 25% | −25 pts: segmentation |

Oracle figures from
[artifacts/reports/asl_temporal_boundary_v17_20260922/baseline_oracle.json](../asl_temporal_boundary_v17_20260922/baseline_oracle.json).

**Crop context is as valuable as perfect boundaries** — from the same oracle run:

```
oracle_trim_1_context_0    →  6/24 correct, 19 supported intervals, WER 75%
oracle_trim_1_context_100  → 12/24 correct, 24 supported intervals, WER 50%
```

Identical oracle boundaries; doubling the classifier crop context doubled the correct count. Current
evaluations already use `context=0.1`, so that gain is banked. Whether 200–300 ms helps further is
**untested** and is a minutes-long experiment.

### 6.3 Per-gloss recall, local60, frozen arm

| Gloss | Refs | Correct | Recall |
|---|---|---|---|
| GOOD, MORNING, MY, NAME | 6 each | 6 each | 100% |
| I | 18 | 17 | 94% |
| YOU | 18 | 16 | 89% |
| THANKYOU | 6 | 5 | 83% |
| HELP, HOW | 18 | 14 | 78% |
| HELLO | 18 | 13 | 72% |
| TOMORROW, GO | 6 | 4 | 67% |
| PLEASE | 18 | 9 | 50% |
| **FRIEND** | 6 | 1 | **17%** |
| **SCHOOL** | 6 | 0 | **0%** |

False commits: `GOODBYE` ×5, `MORNING` ×4, `HOW` ×4, `WHY` ×2, `READ` ×2 — motion-confusion pattern.

---

## 7. Training-set fit

### 7.1 Boundary models — frame-level, computed from the cached training set

Computed over [artifacts/cache/pretrained_boundary_finetune_v17_20260922/](../../cache/pretrained_boundary_finetune_v17_20260922/)
(train 30,014 / calibration 5,007 / validation 7,593 windows). Positive class = START/END frames,
threshold 0.5, masked frames excluded.

| Arm | Split | Frames | Accuracy | Pos recall | Pos precision | Pos F1 |
|---|---|---|---|---|---|---|
| Frozen | train | 54,153 | 79.64% | 28.45% | 58.97% | 0.384 |
| Frozen | calibration | 9,043 | 79.33% | 28.00% | 56.25% | 0.374 |
| Frozen | validation | 13,640 | 77.17% | 25.30% | 52.07% | 0.341 |
| Finetune ep7 | **train** | 54,153 | **83.17%** | 87.45% | **58.15%** | 0.699 |
| Finetune ep7 | calibration | 9,043 | 80.22% | 81.13% | 53.37% | 0.644 |
| Finetune ep7 | validation | 13,640 | 76.44% | 75.65% | 49.64% | 0.599 |
| Augmented ep8 | **train** | 54,153 | **84.11%** | 88.51% | **59.69%** | 0.713 |
| Augmented ep8 | calibration | 9,043 | 80.47% | 81.23% | 53.77% | 0.647 |
| Augmented ep8 | validation | 13,640 | 76.81% | 76.34% | 50.16% | 0.605 |

**Findings:**
- Train→validation gap is only **6.7 points** (83.17% → 76.44%). This is *not* memorization.
- Fine-tuning tripled positive recall (28.45% → 87.45%) — the frozen model barely fires the randomly
  initialized edge head, as expected.
- **Precision never moved: 58.97% frozen → 58.15% finetuned → 59.69% augmented.** The model learned
  *to fire*, never *where*.
- **58% precision on training data** is the ceiling. Training loss does keep falling (0.6158 at epoch 7
  → 0.0654 at epoch 40), so capacity exists — but calibration loss rises from epoch 8 onward. There is
  **no region where training precision and held-out performance improve together.**

Likely cause (hypothesis): annotations carry a ±50 ms tolerance band while targets are frame-exact, so
near-misses are punished as errors.

Caveat: only best-calibration checkpoints were ever saved, so these figures are the *selected*
checkpoints — by construction the least-overfit points. The memorized epoch-40 state was discarded.

### 7.2 Recognizer — already on record

From [docs/ground_truth/stage2-ctc/log.md](../../../docs/ground_truth/stage2-ctc/log.md),
entry 2026-09-21, seeds 17321/17322:

| Metric | Train | Validation |
|---|---|---|
| Known WER | **2.60% / 0.27%** | **50.27% / 49.38%** |
| Exact sequences | 264/283, 281/283 | 34/211, 31/211 |
| **Local phrases, exact** | **226/232 and 232/232** | **31/199 and 30/199** |
| ASLLRP WER | 13.04% / 2.17% | 58.33% both |

One seed scored **232/232 = 100%** on local training phrases and **30/199 = 15%** on validation —
same phrase types, same signer pool, same recording setup.

Per-gloss, from the same entry — and these **predict today's results exactly**:

| Gloss | Train occurrences | Validation matches (then) | local60 (today) |
|---|---|---|---|
| SCHOOL | 25 | 1/19 | 0/6 |
| FRIEND | 31 | 7/27, 8/27 | 1/6 |
| HOW | 74 | 35/60, 22/60 | 14/18 |

`TOMORROW SCHOOL GO` scored **0/19 exact** then. Today it is 4/6, 0/6, 4/6 on those three glosses.
This is a reproduction of a known, documented failure.

### 7.3 The two failure modes are different

| | Training fit | Generalization | Diagnosis |
|---|---|---|---|
| Recognizer | ~perfect (232/232) | collapses (15%) | **memorization** |
| Boundary model | mediocre (83%, 58% precision) | holds (76%) | **never fits** |

### 7.4 Isolated signer probe — does signer diversity explain the phrase failures?

Report: [artifacts/reports/isolated_signer_probe_v17_20260922/probe.json](../isolated_signer_probe_v17_20260922/probe.json)

**Question.** The phrase model trains on 3 local signers while 88 signers sit unused in isolated
datasets ([§8.3](#83-signer-inventory)). Before building a synthetic phrase-concatenation pipeline,
test whether the frozen Reel classifier actually struggles with signer variation, or with something else.

**Method.** 193 held-out (`role=validation`) isolated clips from citizen/semlex/stem covering exactly
the 15-gloss local phrase vocabulary, 34 distinct signers. Each clip treated as one interval spanning
the whole video, classified through the same `classify_interval` path the phrase evaluation uses.
Frozen Reel, read-only, no training.

Per gloss:

| Gloss | n | Signers | Proposal | Verifier | Commit rate | Commit correct |
|---|---|---|---|---|---|---|
| FRIEND | 18 | 18 | 67% | 72% | 50% | 50% |
| GO | 10 | 10 | 60% | 70% | 50% | 50% |
| GOOD | 19 | 19 | 37% | 42% | 42% | 26% |
| HELLO | 10 | 10 | 80% | 70% | 60% | 60% |
| HELP | 19 | 18 | 68% | 74% | 53% | 53% |
| HOW | 4 | 4 | 100% | 100% | 75% | 75% |
| I | 17 | 16 | 59% | 65% | 47% | 47% |
| MORNING | 13 | 13 | 69% | 62% | 46% | 46% |
| MY | 9 | 9 | 89% | 89% | 56% | 56% |
| NAME | 17 | 17 | 71% | 71% | 65% | 65% |
| PLEASE | 9 | 9 | 78% | 78% | 56% | 56% |
| SCHOOL | 18 | 18 | 67% | 67% | 50% | 50% |
| THANKYOU | 8 | 8 | 25% | 12% | 25% | 0% |
| TOMORROW | 10 | 9 | 60% | 60% | 50% | 50% |
| YOU | 12 | 12 | 67% | 75% | 50% | 50% |
| **ALL** | **193** | **34** | **64%** | **66%** | **51%** | **48%** |

By source — this is the decisive split:

| Source | n | Proposal | Verifier | Verifier accepted | **Commit correct** |
|---|---|---|---|---|---|
| **citizen** | 57 | **96%** | **96%** | 63% | **58%** |
| semlex | 133 | 52% | 54% | 53% | 45% |
| stem | 3 | 0% | 0% | 0% | 0% |

**Findings.**

1. **Signer diversity is not the binding constraint.** On Citizen clips the classifier scores **96%**
   across ~19 signers per gloss. It handles signer variation well. Adding 88 signers to phrase training
   would supply variety the model already copes with.

2. **The commit gate is the single largest measured loss in the pipeline.** On Citizen the verifier
   identifies **96%** correctly but only **58%** are committed. Across all sources, **34 of 127
   correctly-verified clips (27%) were never emitted, at a median verifier score of 0.908.**
   Only 5 of those 34 had `verifier_accepted: True` — the accept criterion itself is rejecting correct
   identifications scoring ~0.9. This reproduces, on clean isolated clips where segmentation plays no
   part, the same pattern as the 51-of-79 blocked intervals on `local60` ([§6.1](#61-interval-supply-and-commit-accounting-frozen-arm)).

3. **The failing phrase glosses are recognizable in isolation.** SCHOOL scores **67% across 18 distinct
   signers** yet 0/6 in local phrases; FRIEND 67% across 18 signers yet 1/6. Neither the sign nor the
   signer count explains the phrase failure — it is context transfer.

4. **No consistent relationship between isolated and phrase performance.** GOOD scores **37% isolated
   but 6/6 (100%) in local phrases**; SCHOOL scores **67% isolated but 0/6 in phrases**. The inversion
   rules out intrinsic sign difficulty and points at the local continuous recordings specifically.

5. **SemLex domain gap: 52% vs Citizen 96%.** SemLex supplies 33 of the 88 signers, so a third of the
   signer pool sits in a domain the classifier handles at roughly half accuracy. This further undercuts
   raw signer count as a lever.

**Conclusion: do not build the synthetic phrase-concatenation pipeline as the next step.** It is a
multi-hour build targeting a variable the evidence shows is not binding, while the commit gate and
context transfer are cheaper and demonstrably cost more.

**Caveats.** 193 clips total. `stem` n=3 and HOW n=4 are not interpretable; THANKYOU n=8. The Citizen
figure rests on 57 clips. Reel operates over the full 100-gloss vocabulary, so these are 100-way
decisions restricted to a 15-gloss target subset, not a 15-way task.

---

## 8. Data composition

### 8.1 Boundary training set — 1,121 poses

Manifest: [artifacts/reports/pose_boundary_transfer_v17_20260922/prepared_manifest.json](../pose_boundary_transfer_v17_20260922/prepared_manifest.json)

| Source | Videos | Events | Signers | Total footage |
|---|---|---|---|---|
| `asllrp_other_ctc` | **1,077** (96%) | 4,142 | 4 | 71.5 min |
| `asllrp_contiguous` | 38 | 58 | 4 | 0.7 min |
| `o5s5` | 6 | 110 | 6 | 21.7 min |

`asllrp_other_ctc` is the CTC **"other"** class. Of its 4,196 confident events, **3,347 (80%) are
`__OTHER__`** — unlabeled/OOV signs. This is correct design: a boundary model needs to learn *where* a
sign starts and ends, not what it is.

Signers: CORY (1,696 events), RACHEL (959), BENJAMIN_JAMES_BAHAN (783), JONATHAN (758) — **four people**
for the entire ASLLRP portion.

### 8.2 Why only 38 of 56 `asllrp_contiguous` videos

Gate: [artifacts/reports/confident_supervision_v17_20260920/confident_supervision.json](../confident_supervision_v17_20260920/confident_supervision.json)

Of 144 events across 56 videos, **58 accepted across 38 videos**. The other 18 videos had every event
rejected and dropped out entirely.

| Rejection reason | Events |
|---|---|
| `source_crop_incomplete` | 48 |
| `fewer_than_four_raw_target_samples` | 41 |
| `fewer_than_six_raw_target_samples` | 18 |
| `outside_raw_clock` | 10 |

```
rejected event durations: median 0.150 s (min 0.033 s)
accepted event durations: median 0.334 s (min 0.234 s)
```

Contract requires `minimum_raw_target_samples: 6` at a **20 Hz** sampling clock. A 150 ms sign — normal
for ASL function words — yields 3 observations. Source video is 29.97 fps, so the downsample is what
starves them. Example: `NIGHT` in `15718738_span00` — hand visibility 1.0, clean crop, 200 ms —
rejected for having 5 samples instead of 6.

**Design mismatch:** `confident_supervision.json` was built as a *classification-confidence* gate
(`public_glosses: 100`, `model_correctness_used_for_filtering: false`). The boundary model predicts
per-frame START/END and needs locatable edges, not six confident interior samples.

**Not worth fixing:** `asllrp_contiguous` is 3.4% of boundary poses. Recovering 18 videos adds ~1.6%,
against a model with no demonstrated adaptation headroom.

### 8.3 Signer inventory

| Source | Signers | Type |
|---|---|---|
| citizen | **37** | isolated |
| semlex | **33** | isolated |
| stem | **18** | isolated |
| o5s5 | 6 | continuous |
| asllrp_segmented | 4 | ASLLRP |
| asllrp_contiguous | 4 | ASLLRP |
| **local_phrases** | **3** | continuous |

**88 signers exist on disk in isolated datasets.** The phrase model trains on 3 local + 4 ASLLRP.

643 of those isolated clips cover the exact 15-gloss local phrase vocabulary (23–46 signers per gloss,
including SCHOOL 58 clips / 44 signers and FRIEND 53 clips / 40 signers). **However,
[§7.4](#74-isolated-signer-probe--does-signer-diversity-explain-the-phrase-failures) shows signer
diversity is not the binding constraint** — the classifier scores 96% across ~19 signers per gloss on
Citizen. Signer count is available but is not the lever.

### 8.4 Pipeline component status

| Component | Status | Performance |
|---|---|---|
| Stage 0 extractor | Frozen — Apple Vision | 93.12% val top-1 (beat MediaPipe 89.95%) |
| **Stage 1 isolated** | **Frozen — test gate consumed** | **87.57% top-1 / 98.64% top-5 / 87.39% macro-F1 on 1,247 signer-disjoint Citizen test clips** |
| Reel (phrase-adapted Stage 1) | `stage1_v17_unified_phrase_activity_adapt_reel_v2` | the failing component |

Stage 1 is **not** memorizing and should not be retrained — doing so would also re-consume a test gate
PROJECT_GROUND_TRUTH explicitly protects.

### 8.5 MediaPipe / Apple Vision split

The live app uses **Apple Vision**
([active/v17/extract_v17.py:232](../../../active/v17/extract_v17.py#L232) — `VNDetectHumanHandPoseRequest`,
`VNDetectHumanBodyPoseRequest`, `VNDetectFaceLandmarksRequest`).

The pretrained boundary backbone consumes **MediaPipe Holistic** reduced-50 keypoints. Shipping it as-is
would require running a second pose extractor per frame — MediaPipe Holistic complexity-1 is typically
15–30 ms/frame, likely more than the boundary transformer itself (~10 ms).

The existing TCN ([scripts/live_boundary_v17.py](../../../scripts/live_boundary_v17.py)) already consumes
Apple Vision features shared with Stage 1/2 — zero extra pose cost.

Measured inference costs ([benchmark.json](../pretrained_boundary_finetune_v17_20260922/benchmark.json)):

| Component | Cost |
|---|---|
| Full forward, CNN + 4 attention (batch 16, no_grad) | 8.05 ms/window |
| — attention stack | ~1.1 ms |
| — `frame_cnn` + `input_norm` | **~6.9 ms (86%)** |
| `window_features` normalization (CPU) | 1.79 ms/window |
| Budget at 20 Hz | 50 ms/frame |

Training cached `project()` away, so **every training timing understates live cost**. Per-frame
projection caching is blocked because `normalize_mean_std` is computed per window — the same frame
normalizes differently in each window.

---

## 9. Corrections made during this investigation

1. **"Let the run finish; seed 2 is not optional."** Wrong. `evaluate()` in
   [scripts/evaluate_pretrained_boundary_v17.py:106](../../../scripts/evaluate_pretrained_boundary_v17.py#L106)
   is a standalone entrypoint that iterates whatever runs exist in `training_results.json`. One seed
   suffices. Stopping after seed 1 saved ~90 minutes.
2. **"The failures are ASLLRP domain mismatch"** (based on a −0.72 correlation between
   `asllrp_segmented` share and per-gloss recall). Revised: the generalization gap appears *within*
   local phrases where no domain gap exists (100% train vs 15% validation). Memorization is primary;
   domain mismatch is at most secondary.
3. **`boundary_candidates` scoping defect** in the expanded evaluation — found and fixed; recall
   figures unaffected.

---

## 10. Open questions and next actions, cheapest first

1. **Commit threshold sweep** — free, minutes, no training. **Now the highest-value action**, on two
   independent pieces of evidence: 51 of 79 `local60` rejections passed the verifier
   ([§6.1](#61-interval-supply-and-commit-accounting-frozen-arm)), and 27% of correctly-identified
   isolated clips were discarded at median score 0.908 ([§7.4](#74-isolated-signer-probe--does-signer-diversity-explain-the-phrase-failures)).
   Sweep `commit_score`, `instant_commit_score`, `commit_hits`, and examine the verifier's `accepted`
   criterion, which rejects correct ~0.9-confidence answers.
2. **Classifier context window beyond 100 ms** — minutes. The oracle data shows 0→100 ms doubled the
   correct count at identical boundaries. 200/300 ms is untested.
3. **Boundary work: stop investing.** Frozen matches every adapted variant. If pursued, change the
   *target formulation* (tolerance-band soft labels) rather than training longer.
4. **Investigate the SemLex domain gap** — 52% vs Citizen 96% on the same glosses and the same
   classifier. SemLex is 33 of the 88 signers; if that domain does not transfer, its contribution to
   any future training mixture needs re-weighting rather than expanding.
5. **Synthetic phrase concatenation from the 88 signers — deprioritized.** [§7.4](#74-isolated-signer-probe--does-signer-diversity-explain-the-phrase-failures)
   shows signer diversity is not the binding constraint. Revisit only after the commit gate and context
   transfer are addressed and the remaining gap is characterized.
6. **External acquisition — deferred.** YouTube-ASL (~2,500 signers, video-ID list, no gate) and
   OpenASL (~288 h, CC BY-NC-ND 4.0, no gate). Neither carries gloss boundaries; both are translation
   corpora. OpenASL's NC-ND terms need a decision before download if this is commercial. Given
   finding 7 in [§1](#1-summary-of-conclusions), signer volume is not the current bottleneck.

---

## 11. Artifacts produced

| Path | Description |
|---|---|
| [scripts/evaluate_boundary_expanded_v17.py](../../../scripts/evaluate_boundary_expanded_v17.py) | Expanded 72-video / 186-sign evaluation, multi-checkpoint, per-subset summaries |
| [artifacts/reports/boundary_expanded_eval_v17_20260922/](../boundary_expanded_eval_v17_20260922/) | `REPORT.md`, `evaluation.json` (3.9 MB — query with `jq`, do not read whole) |
| [artifacts/reports/isolated_signer_probe_v17_20260922/probe.json](../isolated_signer_probe_v17_20260922/probe.json) | Isolated signer probe — 193 clips, 34 signers, per-clip proposal/verifier/commit records |
| [docs/ground_truth/live-streaming/log.md](../../../docs/ground_truth/live-streaming/log.md) | Dated entry, newest first |
| This file | Consolidated computations and findings |

**Standing limitations on everything above:** familiar-signer reused development pools; one seed per
checkpoint with no variance estimate; 15 glosses over 6 phrase types on `local60`; cached-pose offline
composition, not live latency; no protected test accessed; nothing promoted.
