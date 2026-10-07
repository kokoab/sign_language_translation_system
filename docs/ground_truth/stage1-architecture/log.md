# stage1-architecture — log

Measured results, rejected approaches, and progress snapshots. Not read start-to-end —
`rg` this file before re-running an experiment to see if it already failed.

94 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## 2026-10-08 — Citizen-variant local data experiment (96.83 chain rebuilt)

User: for all confused classes use the Citizen variant. Removed local train/val clips of HOME
(=HOUSE), CHILD, GOODBYE, HEAR, WHAT, BIG, SIGN, ASK, COME and non-ME I (letter-I, landmark
pinky/index split visually checked): 1,425 train / 293 val removed. Same Variant C + phrase
recipe. Loader guard for 94 classes now accepts documented derived manifests
(active/v17/extract_hand_rgb_supplement_v17.py); fusion gained opt-in --expected-record-counts.
Recognizer (same inputs: Citizen/SemLex/filtered local): current 367/864/2546 vs new 362/865/2548;
tune WER 31.42 → 30.09%. Citizen −5 = single-clip changes across 9 classes. Affected classes on
public val essentially unchanged (they were already correct there); the benefit can only show on
the user's live Citizen-form signing. Not exported/installed. Report:
artifacts/reports/citizen_variant_local_filter_v17_20261008/REPORT.md.

## 2026-10-07 — Independent paper-claim verification and proposed interpretation

Fresh MPS inference reproduces all15 domain counts across the five new phases and
three counts for the clean August recognizer. verify_paper_claims.py asserts source
membership/hashes (636 RGB/hand files), canonical hash/unchanged gate, train/validation/
tune/test identity separation, phase arithmetic, phone sample medians and strict
checkpoint results. Separate Mac CoreML FP32/FP16 rerun reproduces367correct and378/378
PyTorch top1 agreement. WER selected history arithmetic recomputed31.4159 vs clean
August30.0885. No training/protected test inference. No manuscript changes.
PAPER_CLAIM_REVIEW.md records required corrections: epoch0fixedfusion is not learned
fusion-distillation gain; segment labels use equal-width transcript partitioning;
SemLex is not lower at isolated phase; phone FP32 changes recognizer only (othersFP16),
new59.15/23.69≈2.50; similar timings do not prove equivalence; use clean August comparator.
Scoped manifest membership passes, but claimed span-label/input hashes are missing from
recipe.json (isolated cache hashes are present); keep provenance limitation explicit.
Historical reproduction reused contaminated split, so “test never touched” must be
scoped to newly trained downstream stages. Artifacts under canonical_recognition_
comparison_v17_20261007: verify_paper_claims.py, paper_claim_verification.json,
paper_export_verification.json, PAPER_CLAIM_REVIEW.md. Next: author reviews table/prose
proposal; retain candidate/default distinction and do not promote from this audit.

## 2026-10-07 — 96.83 chain completed through recognizer on approved split; iPhone timed

User authorized overnight completion of phrase adaptation, interval recognizer, export and phone.
Both August downstream recipes reproduced exactly first (reproductions/README.md). Recipe-scoped
manifest active/v17/phrase_segment_recipe_manifest_20261007.json (train179 = approved train ∩
lab train; val139 = approved val − lab test; lab tune selection only; lab test untouched; no
blank/rest/OTHER/NCSLGR/OOV). Results (Citizen/SemLex/local, mean of 3, pooled 4,252):
96.83 chain — isolated 96.83/87.22/64.12 (82.72, 72.34); local 97.35/87.32/98.17 (94.28,
95.60); fusion 97.62/88.04/98.20 (94.62, 95.81); phrase e22 97.62/88.04/98.03 (94.56, 95.70),
phrase seg 75.20%, activity 74.47%; recognizer e4 97.09/88.34/97.76 (94.40, 95.53), tune WER
37.61→31.42%. August chain same split: phrase 95.77/89.06/96.96; recognizer e6
95.77/89.06/96.55 (93.79, 94.76), tune WER 38.94→30.09%. Original leaking local_a
95.24/89.67/96.31, tune 27.43% (not comparable). Letter head A recipe retrained (recall 83.71
vs 86.15%). Core ML 378 Citizen val: FP32 and FP16 both 367/378 = PyTorch, 378/378 agreement;
50.28/25.51 MB (same as August). iPhone 13, 226 frames ×2 passes, nominal thermal: FP16
August 24.25 vs 96.83 chain 23.69 ms (equivalent, within pass spread 22.92–25.58); FP32 56.07
vs 59.15 ms. App sources restored byte-exactly, production rebuilt/reinstalled (attempt1 build
failed: missing ENABLE_TESTABILITY). Report downstream_recipe/REPORT.md. No promotion,
manuscript or default change; no protected test access; lab held-out test not run.

## 2026-10-07 — Corrected 96.83 chain (mild roll): local 97.35/98.17, fusion 97.62

Variant C (CHAIN_PLAN.md; full roll 0, floor361, patience80, user-approved post hoc)
complete; audit_chain.py passes. 80 epochs; selected epoch55 (both candidates):
Citizen368/378 97.35% (epochs45–58 at367), SemLex854/978 87.32%, local masked98.17%,
landmark-roll worst7/378; all gates pass, incl. predeclared365 floor. Raw-pixel roll +
Vision auto-orient: 96/97/86/96/96/95/96/96, mean94.75% (96.83 92.75, a7490409 same
code92.25). Fusion (fresh hash-verified cache, unchanged hand/teacher/seeds/rule): all
seeds select epoch0 = fixed75/25 z-score landmark/hand fusion, 369/378 97.62%,
SemLex861 88.04%, local2844/2896 98.20%; trained distillation epochs peak368/861/2843.
Versus August chain (branch361/860/96.34, fusion364/871/2812): higher Citizen and local,
SemLex −6/−10 clips. Validation-selected development evidence; no test access. Phrase/
interval still blocked (training_ready=false). Report chain_9683_floor361_mildroll/REPORT.md.

## 2026-10-07 — ERROR: 96.83 local replay used mismatched full-roll augmentation

User challenged why 96.83 underperformed downstream. Provenance diff of parents: identical
data/splits/SemLex/sampling/objective/architecture/seed; ONLY difference is augmentation —
96.83 (Kaggle 2026-08-11) predates full-circle roll; a7490409 is the same recipe plus
full roll .35. My local replays copied August's full roll .35, a large input shift for
96.83 only. Local-replay provenance vs August otherwise identical (splits/hashes/sampler).
Evidence: paired fine-tune full-roll arm Citizen 356–362 vs mild arm 363–366; local
replay with roll 356–363; Variant B (floor361, patience80, roll .35) epochs1–9 94.71–95.50,
stopped deliberately (STOPPED.md). 96.83 is not worse as a parent: Citizen366 vs362,
SemLex853 vs839, initial local64.12 vs60.95. The chain_9683 local/fusion results are
confounded and must not be used as the 96.83 chain. Corrected Variant C launched
(CHAIN_PLAN.md): full roll 0 (mild ±12°), floor361, patience80, else unchanged,
PID45792 → chain_9683_floor361_mildroll/. Rule: fine-tune augmentation must match the
parent's training augmentation unless the treatment is the augmentation itself.

## 2026-10-07 — 96.83 chain: auto-orient robust; local replay fails gates; fusion 96.83

User chose 96.83 (5c40b133) as paper isolated result and asked to retrain downstream.
Gates predeclared in CHAIN_PLAN.md (Citizen>=365=parent-1, SemLex>=853, local>parent,
worst landmark angle>=2). audit_chain.py passes. Raw-pixel roll + Vision auto-orient,
same100 val clips: 96.83 96/94/80/96/96/88/96/96 (mean92.75%); a7490409 same code
93/96/85/94/93/91/93/93 (92.25%) vs August record 91.50% — evaluator is deterministic
(96.83 re-run 800/800 rows identical), so difference is evaluator code drift since Aug;
compare only same-code rows. Local replay from 96.83 (exact August recipe, preflight on
a7490409 reproduced 362/1765/13381/sampler weights): Citizen fell to356–363 every epoch,
never >=365; local masked64.12→94.79 at e20, still rising; patience stop at e20 because
Citizen never beat epoch0. Both saved checkpoints = epoch0 original; local gate fails;
no eligible branch. Fusion (unchanged hand/teacher/seeds/rule, fresh hash-verified
caches) on unadapted 96.83, labelled not gate-eligible: seed3407 e4 Citizen366 (96.83),
SemLex862 (88.14), local2447/2896 (84.50); seed5101 had367 Citizen but lower key.
August fusion rerun with fresh cache reproduces 364/871/2812 exactly. Phrase/interval
stages not run (training_ready=false). Files: CHAIN_PLAN.md, run_chain.py,
run_raw_orientation.py, audit_chain.py, chain_9683/, raw_orientation/. Open decision:
report local stage as non-transferring, or a user-approved post-hoc variant (absolute
361 floor / longer patience). No promotion, manuscript or test access.

## 2026-10-07 — No-full-roll Transformer completed; earlier Transformer rotation-robust

User requested waiting; OS process-exit notification received, status complete checked.
New explicit mild-roll Transformer selected epoch47, stopped epoch77 after30 staleepochs.
Fresh CPU strict restore reproduces94.708995%Top1/99.470899%Top5. Current source and
manifest hashes match launch recipe; contiguous history/first-best/earlystop assertions
pass. Existing95.502646%Transformer also independently restored. Identical eight-angle
landmark diagnostics: new mild upright94.71/mean-nonzero32.65/worst1.59; earlier
Transformer upright95.50/mean93.58/worst92.06. Thus earlier Transformer DOES demonstrate
software-landmark rotation robustness; prior uncertainty is resolved. It is comparable
to existinga7490409 Squeezeformer95.77/93.73/92.06 under the same evaluator protocol.
Do not infer precise historical augmentation or a causal training effect from unpinned
historical code. Single-seed development only, no new phone/raw-camera claims, no
protected test. Source/result/rotation logits, hashes and REPORT.md under
artifacts/reports/transformer_mild_roll_v17_20261007; new verify.py passes allassertions.
No active run from this experiment; no promotion/manuscript changes. Next: discuss
verified comparison and any bounded retention experiment from robustSqueezeformer;
no additional training launched.

## 2026-10-07 — Explicit no-full-roll Transformer baseline launched

User requested Transformer under the flat Squeezeformer baseline recipe and explicitly
excluded new full-roll training. Reused existing family train_family through a small
isolated wrapper with full_roll_probability=0, mild12degrees, seed1701, 160epoch
ceiling/patience30, batch64, AdamW3e-4/wd.03, warmup8/cosine, EMA.999 and existing
approved balanced Citizen/SemLex loaders. Historical flat Squeezeformer provenance
records generic augmentation rather than an immutable implementation; this is an
explicit current mild-roll baseline, not a claim of exact historical reproduction.
Current family trainer otherwise inherits full-roll .35; wrapper explicitly overrides it.
Wrapper config preflight passed; detached caffeinate PID36352, startup state running
verified once. No polling; desktop completion/failure notification. Read status before
assuming success. Output artifacts/reports/transformer_mild_roll_v17_20261007/.
Existing finetune audit rechecked successfully; diagnostic rerun already in current state
shows no better robust candidate. No new Squeezeformer training, manuscript or deployment
changes; protected test remains sealed. Next: verify completed Transformer checkpoint
and measure its rotations under identical evaluation, then discuss retention experiments
from a7490409 without assuming 96% is achievable.

## 2026-10-07 — Diagnostic rerun: 20-epoch full-roll warm start dominated by a7490409

finetune_diagnostic/ complete; audit_finetune_diagnostic.py passes (histories bit-identical
to first run, max|ΔTop1|0.00; selected best_model.pth still epoch0=original; diagnostic
epochs/upright match logits; identical378 IDs). Landmark-roll Top1 upright/mean-rotated/
worst: original96.83/35.71/0.53; mild final(e20)96.03/36.02/0.26, best-trained(e1)
96.56/35.41/0.26; fullroll final(e20)94.71/72.07/52.38, best-trained(e2)95.77/43.31/3.97;
from-scratch orientation-robust a7490409 95.77/93.73/92.06 (same evaluator, matches
Aug record). Warm-start full roll trades ~2.1pp upright for partial robustness and is
worse than a7490409 on both axes; mild control gains nothing. No candidate replaces96.83
(upright) or a7490409 (orientation). Single seed, 20-epoch bounded screen; landmark
rotation is not raw-camera/phone evidence. Files: run_finetune_diagnostic.py,
audit_finetune_diagnostic.py, finetune_diagnostic/{REPORT.md,audit.json}. No promotion,
manuscript or production change; no test access. Next: user chooses which pinned
landmark checkpoint parents the matched local/fusion/interval chain.

## 2026-10-07 — Paired fine-tune complete: both arms retained epoch 0 (original weights)

finetune/status.json complete. audit_finetune.py ran: logits-recomputed Top1 matches
saved metrics; identical378 evaluation IDs; both selected best_model.pth are epoch0 and
tensor-identical to5c40b133 (file SHA differs only from re-saved metadata). Audit fixed:
weight-identity check now also requires identical key sets; adds best/final trained-epoch
columns. Upright selected96.83 for original/mild/fullroll; eight-angle landmark scores
identical (0°96.83,17°96.03,37°91.80,73°39.95,90°10.58,123°1.06,180°0.53,270°10.05;
mean rotated35.71). Trained epochs never strictly beat96.83: mildcontrol best96.56
(epoch1), final96.03; fullroll best95.77 (epoch2), final94.71. No orientation improvement
claimed; trained full-roll weights were not saved, so their rotation effect was unmeasured.
Trainer now has opt-in --save-diagnostic-checkpoints (final_model.pth,
best_trained_model.pth; best_model.pth selection unchanged); parser test added,
focused tests pass. Identical-recipe diagnostic rerun launched detached PID30307 into
finetune_diagnostic/ with a7490409 from-scratch orientation reference evaluated by the
same evaluator. No promotion, manuscript or production change; no test access.

## 2026-10-07 — Legacy exact-initialization config fixed; training relaunched

Initialstartupguard rejected96.83checkpoint because oldconfigomits newer disabled
fields. initialize_exact_stage1_finetune now validates/compares resolvedStage1V17Config
values; schema,manifest,labels,sealedtestsandstrictstate-dictguards unchanged.
Newtest proves missingdefaultsaccepted withbit-identicalweights, changedcanonicalization
rejected. Newandexistingexact-init tests2/2pass; actualhistoricalcheckpoint strictinit
passes. Existingrotationtests3/3passedearlier. Changedactive/v17/train_stage_1_v17.py,
newtest/test_v17_finetune_legacy_config.py. Failedrunpreserved; pairedexperiment
relaunched detached PID26292, initialstatusrunning/mild_control verified once.
No trainingprogresspolling. Checkstatus/notificationbeforeassumingcompletion.

## 2026-10-07 — Isolated retiming complete; paired fine-tuning launched detached

Current comparison verified_results.json excludes export-overlapped initialtimings;
three isolatedCPU passes preserve exactpredictions for allsevencheckpoints.
Phone36runs pass parity/nominalthermal assertions. REPORT.md generated. Strict-init,
missing-safeinvertiblerotation andfullcircleaugmentation tests3/3passed; bothtraining
commands andorientationevaluator parse withprotectedtestdisabled. Boundedpaired20epoch
initial launch PID25914 failed before training: legacy config omitted newly added
default fields. Failure preserved underfinetune_initial_config_rejected. No progresspolling. Desktopcompletion/failurenotification configured;
checkfinetune/status.json nextsessionbeforeassumingcompletion. Filesplan,launcher,
results,hashes undercanonical_recognition_comparison_v17_20261007. No manuscript,
productionmodel changes. Laterchaintraining remainspending.

## 2026-10-07 — Current fixes authorized from the preserved 96.83% checkpoint

User asks to apply orientation-robust/current training fixes. Fresh Mac and iPhone
checks reproduce96.8254%, includingFP16; do not describe it as lost. Existing current
Stage1 trainer strict warm-start and rotation tests3/3 pass; CLIpreflight verifies
both paired20epoch commands and eight-angle evaluator, protectedtestdisabled.
Plan/launcher under canonical_recognition_comparison_v17_20261007: mildcontrol versus
fullroll.35 from exact5c40b133checkpoint, sameapproveddata/seed/budget,lr5e-5.
Original remainsfallback; no blind combination of rejectedarchitecture treatments or
local-specific masking into base training. Launch only after isolatedtiming completes,
detached with completionnotification; no trainingpolling. Full downstreamcomparisons
and paperupdates remainpending after these measured gates.
Phonebenchmark nowcomplete36measurements across6configs/6rotatingorders, allnominal,
nochangedpredictions; TransformerFP32/FP16 7.1987/.9540ms, flatSqueeze7.1639/.8866ms,
partwiseSqueeze8.8792/.9917ms. Initialinstallationhitfreeprofilelimit; reusedownprevious
benchmarkbundle, thenfreshderiveddataresolvedstalebundle launch; productionunchanged.

## 2026-10-07 — Selected-checkpoint comparison and phone exports underway

User authorized fast revalidation including flatSqueezeformer versus flatTransformer,
partwise/global improvement and downstream recognition stages; manuscript remains
chat-first. New isolated report root canonical_recognition_comparison_v17_20261007.
Fresh prepared-input checks reproduce flatTransformer95.50, flatSqueezeformer95.77,
partwise96.83, local-adaptedlandmark95.50, unified96.30, phrase-adapted96.03.
Exports additive only, original weights pinned by hash; no training/protected test.
Initialization metadata links local branch to orientation-robust checkpointa7490409,
not directly the96.83checkpoint. Do not invent direct weight inheritance.
Initial CPU timing overlaps later export preparation; preserve it as exploratory,
exclude from final timings and repeat after exports finish without concurrent work.
Phone uses separate benchmark app; no production model selection changes.

## 2026-10-07 — Historical Transformer runs independently revalidated

User authorized fast Transformer improvements and matched downstream stages, then
requested verification of earlier runs first. No new training started. Read-only
CPU restore reproduces Top1/Top5/macroF1 for flat95.50, partwise93.92, conv94.18,
anatomical93.12, compact94.71, Squeezeformer96.30. All histories consistent with
seed1701, recorded data/sampling, 160epoch cosine schedule, first-best EMA selection
and30stale-epoch stopping. Citizen train/val signer and raw-hash separation verified;
SemLex train raw hashes do not overlap Citizen validation. Old SemLex false eligibility
flag triggered initial assertion; September22 explicit admission resolves it and all
1388training feature paths/hashes match admitted manifest. No flags changed.
Checkpoint metadata lacks historical full source/data/optimizer/RNG provenance; current
input hashes archived, not retroactive proof. Shared recipe isn't per-architecture tuning
or identical augmentation realizations. Whole designs and only one seed; part-wise
negative is valid for this run but earlier causal/rejection wording overstates evidence.
Paired correctness: Transformer-only9, Squeezeformer-only12. No protected test loaded.
Report/scripts: `artifacts/reports/transformer_run_audit_v17_20261007/REPORT.md`.
Next: bounded, pinned matched fine-tuning controls plus residual part-wise candidate;
fine-tuning seed repeats must not be presented as independent from-scratch replication.
Full downstream comparison remains authorized and unstarted, not completed by this audit.

## 2026-09-25 — unfrozen-encoder phrase adaptation: controlled test, no promotion

User asked to test unfreezing, with isolated accuracy kept at or above 90. Frozen-head vs
unfrozen arms, same seed and data, from pre-phrase student_v1. Phrase data is approved-v2
train only (276 clips / 2,892 segments). Selection uses local phrase val only; the 12 ASLLRP
val videos are the held-out oracle. Floors vs shipped: Citizen≥95.53, local≥96.53,
SemLex≥89.16 (already <90, zero tolerance).
Oracle inputs frozen and reproduced exactly (live Core ML 12/16/16/12/11 at pads .00–.20).
Unfrozen ep9: Cit 95.77, SemLex 89.16, local 96.69, local phrase val 76.7% (frozen 66.3%),
oracle 13/18/19/11/10 vs frozen 12/16/18/12/12 vs shipped 12/16/16/12/11. Net +2/+1 of 24
held-out events vs control: inside noise. Unfrozen eligible only at ep7/9; later epochs reach
83% phrase val but break the Citizen floor while the oracle stays flat, so it memorizes
templates (5 of 6 val templates are also in train).
Found along the way: the shipped head's old 390-clip split contains 158 approved-v2
validation clips, and the shipped phrase adaptation equals the base on the held-out oracle.
Unfrozen MPS OOM at batch 256 (2.13 GiB cap); fixed with exact gradient accumulation.
Next: a larger held-out continuous-interval set from unseen signers is needed before any
ceiling claim. Report artifacts/reports/unfrozen_phrase_adapt_v17_20260925/REPORT.md.

## 2026-09-10 07:15 PHT — user-requested matched-variant phrase replay

Investigated data/variant/learning question without further training or architecture
changes. Existing accepted selector already combines primary/specialist heads, and
repair adds frozen evidence; another ensemble is not the next diagnostic step.
Exact-variant ASLLRP failures mean label variants alone cannot explain the gap.
Recommended separating raw extraction, isolated sign-core recognition, temporal
sequence decoding and transcript commit behavior before changing data/model capacity.

Found WATER COLD by JONATHAN (ASLLRP30336, 1.368s), already an exact development
validation example despite acquisition folder train_candidate. Verified clip hash,
ASL-LEX codes A_02_031/C_02_068 and SignBank IDs WATER/COLD, decoded four preview
frames. Public parent URL returned HTTP200 video/mp4. Ran the current continuous
Reel with raw clip, --finish-at-eof --no-display --no-speech --naturalizer literal.
Stage2 suggestion WATER COLD is correct (0/2 edits), matching old/new frozen-cache
predictions. Committed hypothesis is empty, so summary exact=false; no utterance
committed and zero stale frames. This distinguishes Stage2 recognition from the
review-only commit path; no new general accuracy or repair improvement claim.
Evidence/command: artifacts/reports/stage2_v17_same_variant_replay_v1/README.md
and its timestamped history.json. No new data acquisition, protected test, model
changes or tuning; CLI --help validated. Preserve this as a positive diagnostic
control and compare failed exact-variant examples without selecting for successes.

## 2026-09-10 13:07 PHT — matched cache complete; online confirmation limitations measured

Completed all1647 live-matched caches in3386.7s:1313 train and334 validation, no zero-feature clips. All12 exact-variant cached final hypotheses match their raw replay baseline. No protected test accessed. Added per-written-frame source timestamps to session histories; old saved15fps recordings remain lossy. Experimental sequence-preview executes CTC during capture and bypasses Stage1 proposals. All334 validation examples fit the8-window runtime context. Prefix-only outputs are incomplete: local139/259 edits, exact22/24;9/97 local and129/225 OTHER-span examples have incorrect committed prefixes. Repeated agreement is not an accuracy guarantee. Paced PLEASE HELP I replay confirms PLEASE and HELP before Finish and the correct full prefix after Finish; HELLO HOW YOU locks KNOW HOW YOU incorrectly. All12 exact-variant paced online replays completed. Initial bounded12-epoch adaptation is still running; early connected improvements fail original phrase retention, so no promotion. Focused52 tests passed; additional context-rollover test passed with the continuous suite13/13.

## 2026-09-10 19:39 PHT — transition run complete; raw adapted-mode validator fixed

Gap+prefix seed17101 completed12epochs in1738s, selectedepoch12, no eligible epochs. Matched local16/259 (6.18%), exact12/24 (50%), connected223/284 (78.52%). Connected S/D/I=43/158/22; prior adaptation56/138/32, so suppression added20deletions despite10fewer insertions. Annotated gap known emissions3/104, same as prior adaptation (baseline13/104). Original local6->12, exact9->14,context43->57 fail; Citizen328/378,STEM16/21. Best/last checkpoint strictreload finite; supervisionhashmatches. Positive-core comparison stillrunning.

Raw adapted replay caught validator rejecting --revisable-transcript unless --sequence-preview also set. Fixed common validator to accept either explicit matched mode with unchanged provenance/input checks; regression failsbefore/passafter. Restarted failedfirstclip in separate raw_transition_retry; originalfailedlogretained. HUDpartialworker edits completed locally, changeablelabel rendering inspected.72 focusedtests pass. No default changes or promotion.

## 2026-09-09 — transition-adaptation Task 1 preparation passed review

The frozen Task-1 experiment manifest now has SHA-256
`2404aa96c03f5949d6cd3af914f1e97381c279abd980b2e4c5a37b7ce516e2de` and records 111
verified STEM intervals (90 train / 21 validation), the exact encoder/vocabulary hashes,
semantic input roles, and the frozen replay inputs. Its audit SHA-256 is
`2b835600f1da2cd590c34fe64897870f1febc55e9174288ea577939d73e52cde`.
It validates ASLLRP and exact/local phrase archives against their manifests, pins the
canonical ASLLRP extraction-manifest hash carried by both multimodal and frozen
archives, checks Citizen item-ID completeness/split isolation/vocabulary indices, and
runs a combined parent-content leakage ledger across video-derived sources. Existing
archive schemas do not contain literal parent/interval fields; that provenance is
therefore explicitly recorded as indirect through the cryptographically pinned
canonical/source-manifest joins rather than misreported as a direct metadata check.

Independent re-review passed. A fresh controller run passed 16 focused preparation and
extraction tests, `py_compile`, and `git diff --check`. The corrected approved host
one-row extraction smoke also passed and wrote a 27-frame archive for inclusive source
frames 138--163 at an exact 30-fps timestamp grid with no geometry transform. The old
Task-1 report statement that Apple Vision Code 9 still blocks extraction is stale: that
failure occurred only in the sandbox. Full 90-row train and 21-row validation
materialization is the next safe action. No official Citizen test data was accessed.

Full Task-1 materialization is now complete. Apple Vision wrote 90 training and 21
validation multimodal archives. A real downstream invocation caught a shallow output
layout (`role/file`) that the existing MobileCLIP2 encoder could not discover; the
dedicated extractor and its regression test now use the repository contract
`role/source/file`. The 111 real archives and the separate smoke archive were moved to
that canonical layout without recomputing their features, and independent re-review
passed with zero direct-role duplicates.

The existing MobileCLIP2-S0 MPS encoder then wrote 111/111 hand-feature archives in
88.84 seconds (batch 16, memory fraction 0.08). The pinned Stage-1 cache wrote 111/111
612-dimensional frozen archives with zero failures in 12.31 seconds (MPS memory
fraction 0.12). A final manifest-to-cache audit matched every source item, role, and
ordered target and confirmed encoder SHA-256
`1caeadf4b3ca620aa9fef00b35c012b39d7c093f67da1ee2f6987d2c2297906b`.
The train extraction, validation extraction, and frozen-cache report SHA-256 values
are `28b093bd3c2b8d1a259026b95714115aece23e53f91dac46e77d8b90fb1cd8cd`,
`faf60d9f4db503d071b3ab6446231f1870624b781e7f747a60ea12c58227184e`, and
`357cf1bb96606a1609114351ea741855f3c1e2120f8962a52abf5fde6698ebb9`.
No Citizen, SemLex, or local protected test data was accessed. Task 2 may now consume
`data/local/stage2_v17_transition_adapt_v1/frozen_features`.

## 2026-09-09 — transition-adaptation Task 2 trainer passed review

`active/v17/train_stage_2_other_ctc_v17.py` now has an opt-in transition mode; its
legacy defaults and execution path are unchanged when no experiment manifest is
provided. The transition path pins the frozen experiment-manifest bytes, warm-start
and accepted-selector hashes, blank/locked/OTHER indices, 102-output student contract,
encoder identity, source/split roles, Citizen pool provenance, and exact STEM numeric
targets and participant membership. `frozen_inputs.json` additionally pins the 111
STEM and 254 contextual validation feature archives by semantic role, resolved path,
directory hash, count, and encoder hash.

The implementation uses the fixed no-STEM/with-STEM masses, class-first and then
participant balancing, the two-epoch projection schedule, MPS-safe CPU CTC loss,
replay-only accepted-selector distillation, and the plan's fixed hyperparameters. It
measures the accepted selector and initialized candidate, saves raw per-example edit
evidence each epoch, applies baseline-relative gates, and writes reloadable per-seed
checkpoints. Independent review passed after the fail-closed provenance fix. A fresh
controller run passed all 18 focused tests, `py_compile`, and scoped `git diff --check`.
No full matched training run has yet been accepted, and the runtime remains unchanged.

## 2026-09-09 — discussion-only diagnosis: STEM label bug invalidates earlier interpretation

Read-only semantic audit found all 111 transition STEM archives store one-based
targets, while `RealPhraseDataset` adds another one for CTC blank. Preparation's
`load_vocab` and manifest construction introduced the first offset; extraction and
caching preserve it. Example: USE trains as MAKE. All 1,104 ASLLRP OTHER archives and
543 existing phrase archives have correct zero-based stored targets. Citizen replay
pool label checks passed for 1,475 train and 378 validation items across all 100
classes. Existing manifest/archive equality checks missed the semantic label error.

The published STEM references therefore are also shifted. Read-only rescoring of
saved predictions against corrected references gives baseline and initialization
16/21, no-STEM seeds 13/21 and 14/21, and with-STEM seeds 9/21 each. This supersedes
the report's STEM accuracy interpretation, including baseline 0/21 and with-STEM
4/21 and 3/21. No model was rerun and the pinned report has not been rewritten.
The experiment does not establish that correctly labeled STEM data is unhelpful.

Citizen isolated replay already occupies 20% of with-STEM training and 30% of
no-STEM training. No-STEM seed 1701 retains aggregate Citizen validation 331/378
(11 previously correct examples lost, 11 gained); it fails phrase retention.
The Stage-1 encoder remains frozen: these are Stage-2 sequence-decoding metrics,
not a reevaluation or degradation measurement of the original Stage-1 classifier.

The existing distillation loss slices away OTHER before softmax. A synthetic
read-only probe confirmed zero distillation loss even when student OTHER probability
approaches one. CTC still penalizes wrong OTHER predictions; this is a blind spot in
the preservation term, not proof of the cause of every retention regression. Also,
the initialized candidate already differs from the accepted composite selector
(local/exact/contextual edits 7/11/56 versus 6/9/43), before training starts.

User requests discussion before changes. No code, datasets, checkpoints, or runtime
changed; no protected tests accessed. Next proposed action, pending discussion: fix
the existing label boundary and semantic check, review the existing preservation
loss and baseline comparability, and retain mixed isolated/new-data training rather
than add another model layer. A new training run is not authorized by this diagnosis.

## 2026-09-09 — STEM label and replay-preservation fixes rerun as v2

User authorized the root-cause fixes and a full matched rerun. STEM preparation now
stores the same zero-based class indices as every existing Stage-2 feature archive;
`RealPhraseDataset` remains the single boundary that adds one for CTC blank. The
trainer now semantically checks each STEM gloss/index against the locked Citizen100
vocabulary, rather than only comparing archives with their generating manifest. The
replay distillation loss now normalizes over all 102 student outputs before comparing
the 101 teacher-supported outputs, so a dominant OTHER logit is penalized. Focused
regressions were written and observed failing before both fixes.

The corrected experiment is versioned under
`artifacts/reports/stage2_v17_transition_adapt_v2/` and
`data/local/stage2_v17_transition_adapt_v2/`; the invalid v1 evidence remains intact.
Preparation produced 90 train and 21 participant-held validation STEM intervals.
All 111 clips were freshly re-extracted, hand-encoded and frozen with encoder SHA-256
`1caeadf4...`. A semantic audit confirmed all 111 frozen targets match the manifest
and canonical zero-based vocabulary. Corrected manifest SHA-256 is
`d120d9747ed01bc7d7cc0d68d2609a25114c75826484536ee942c278820d4af0`.

The complete fixed matched design reran on MPS for both arms and seeds 1701/1702.
Training-result SHA-256 is
`0468e3f491f1872f7be8a915eeb994581801f99006b6bee6e630e2d6f134cda7`.
The accepted selector's corrected held-STEM baseline is 16/21, confirming the old
0/21 report was invalid. Best checkpoint summaries are:

- no-STEM 1701, epoch 19: target 214/284 edits; local/exact/contextual 13/16/54;
  Citizen 332/378; STEM 13/21;
- no-STEM 1702, epoch 15: target 217/284; retention 13/14/52; Citizen 328/378;
  STEM 14/21;
- with-STEM 1701, epoch 18: target 217/284; retention 15/16/57; Citizen 333/378;
  STEM 16/21;
- with-STEM 1702, epoch 18: target 223/284; retention 14/14/51; Citizen 331/378;
  STEM 15/21.

The with-STEM arm retained the isolated Citizen baseline across both seeds (333 and
331 versus baseline 331) and retained held-STEM performance closely (16 and 15 versus
16). All four runs substantially improved connected ASLLRP target edits, but all still
exceeded the frozen local, exact and contextual phrase-retention ceilings of 6, 9 and
43. Therefore selection is null and the accepted selector/runtime remains unchanged;
no Core ML package or Reel integration was produced.

The v2 selector independently verified the four fixed runs and wrote `selection.json`,
`failure.csv`, `per_example_predictions.jsonl` (7,272 rows), and `README.md`.
All 63 focused preparation, extraction, training, selection and Reel lifecycle tests
pass. Four checkpoints reload with 102 finite-output logits; scripts compile;
repository `git diff --check` passes. Verification is pinned in
`artifacts/reports/stage2_v17_transition_adapt_v2/verification.json`. No protected
test data was accessed.

## 2026-09-09 19:56 PHT — v2 regression investigation, code/data review before user context

User requested codebase understanding, then context questions, then web/data research
before deciding improvements. Inspected the current task-relevant diff, v2 report and
verification record, preparation/extraction, frozen Stage-1 encoder, Stage-2 temporal
head and accepted selector, replay loaders/sampling/distillation, training selection,
and saved per-example predictions and all four epoch histories. No model inference,
training, protected evaluation, code/data edits, or runtime changes were performed.
The existing 63-test result was read as historical evidence, not rerun this turn.

Read-only aggregation of v2 training_result.json confirms:
- The accepted composite selector uses a context-adapted primary plus specialist,
  logit blending and blank calibration. The candidate warm-start is the plain
  multivoice_transfer_adaptation_v3 head, extended with OTHER. Before training,
  local/exact/contextual edits are already 7/11/56 versus accepted 6/9/43.
  This starting mismatch must be separated from subsequent adaptation effects.
- Local baseline has 92/97 exact sequences; with-STEM 1701/1702 have 84/97 and
  86/97. Their 15/14 local edits are entirely insertions/deletions, with zero
  substitutions and zero OTHER-containing local predictions. Examples include
  PLEASE HELP I becoming PLEASE HELP I I or PLEASE HELP, and HELLO HOW YOU
  becoming HELLO HOW. This supports investigating emission timing/repeats rather
  than attributing local phrase regression to wrong gloss identities.
- ASLLRP target-only baseline has S/D/I=134/2/467. With-STEM 1701 has
  39/160/18 and 1702 has 37/167/19. Thus the substantial edit reduction also
  hides a large rise in deletions. Target-only exact sequences are 55/225 and
  48/225 versus 4/225; these are development span metrics, not general accuracy.
- Aggregate isolated retention is not per-example invariance: with-STEM 1701
  loses 13 previously correct Citizen examples and gains 15; 1702 loses/gains
  11/11. Stage 1 stays frozen; these are Stage-2 sequence-output measurements.
- None of the saved trained epochs clears all frozen gates. No-STEM 1702 epoch
  1 reaches local 5 and target 381 but exact/contextual remain 11/54; simply
  selecting an earlier saved epoch does not resolve the complete requirement.
- Replay is already substantial: with-STEM masses are 40% OTHER spans, 20% local,
  10% exact phrases, 20% Citizen, 10% reviewed single-sign STEM cores. The context
  adapter's separate contextual training pool is absent from this adaptation mix.

These observations identify diagnostic leads, not a proven complete root cause.
Next action: ask about intended connected-sign use, handling of out-of-vocabulary
signs, and available fresh connected recordings/review; then perform the requested
web research and targeted data/model diagnostics using that context. Preserve v2
artifacts and accepted runtime; do not launch another training sweep.

## 2026-09-09 20:09 PHT — user authorizes diagnosis-driven repair and verification

Active goal: fix v2 connected/phrase regression and complete the previously stated
diagnostics, without another context/approval round. This supersedes prior discussion-
only/no-further-training limits for a bounded evidence-driven repair. Preserve existing
artifacts, immutable splits and runtime until a replacement actually clears the gates.

Implementation/validation plan (canonical handoff location per repository guidance):
1. Cache pinned models' timewise logits on existing train/development inputs; quantify
   blank/OTHER competition, known-class ranking, repeated emissions/window position,
   and the accepted teacher's replay/coverage mismatch. Do not use protected tests.
2. Choose the smallest supported repair in existing Stage-2 model/trainer paths;
   write a failing behavioral regression, implement, then run affected tests.
3. Run a separately versioned two-seed comparison; require original local/exact/
   contextual ceilings 6/9/43, Citizen >=328/378 and target <=542/284, also report
   insertion/deletion/recall and reviewed STEM retention rather than total WER alone.
4. Verify checkpoint reload and inference behavior, review changes, run relevant
   tests/diff checks, record diagnostics and final selection. A failed candidate is
   not a completed fix. Investigate further without relaxing gates or using labels
   or identities to route inference. No new acquisition is currently justified.

## 2026-09-08 15:58 PHT — expanded review progress verified and snapshotted

After the reviewer was closed and reopened, both the live API and expanded queue on disk
confirmed that review saves were intact: 206 total rows, 179 touched, 170 complete by the
UI rule, 106 eligible, 36 pending and 27 wholly untouched. Reopening resets only the
current navigation position because the browser cursor is not persisted; it does not reset
the CSV. A byte-identical recovery snapshot was written to
`artifacts/reports/asl_stem_wiki_manual_expansion_admission_v17/expert_review_queue.backup_20260908_1556.csv`
with SHA-256 `b6cdd20385d60db2a7d167680e8e7231ac4d404e1fc56f5a9e8e465264b03f51`.
The live reviewer remains on port 8765 and continues to save to the expanded queue.

## 2026-09-08 12:59 PHT — local expert-review UI implemented

Added `scripts/review_asl_stem_wiki_manual_v17.py`, a dependency-free localhost UI
for the 71-row ASL STEM Wiki admission queue. It prominently shows the supposed locked
raw gloss, canonical display label and ASL-LEX code, with the source sentence and every
matching Citizen **training-split** reference side by side. It never searches or serves
Citizen validation/test media. All 71 rows have at least one training reference; the
first row has 15.

The UI includes row filters/progress, previous/next and Save-and-next navigation,
reference cycling, source timeline, frame and ten-frame stepping, playback speeds,
pseudo-position jump, start/end sliders and buttons, selection looping, signer/variant
Yes-No-Unsure decisions, notes, keyboard shortcuts and atomic CSV saves. Eligibility is
derived rather than directly editable: signer and exact-variant approval plus a valid
inclusive span of at most 256 frames are all required. Source-excluded L2 rows can never
be admitted. Focused unit tests cover reference isolation, persistence, bounds,
eligibility, L2 exclusion, controls and pending-filter navigation. Real localhost smoke
loaded 71 rows and returned seekable HTTP byte ranges for both source and reference
videos. Launch with `venv/bin/python scripts/review_asl_stem_wiki_manual_v17.py`.

## 2026-09-07 17:22 PST — focus returned to sequential transcription path

User explicitly removed the SignWriting/avatar goal from the active focus and asked
to return to the phrase-improved sequential Squeezeformer transcription path. Read-only
trace confirms the relevant system is the existing Stage-2 general CTC selector, not
the later grounded causal TCN. Stage 1 produces frozen 612-D evidence from its landmark
Squeezeformer, RGB-hand temporal encoder, and fusion scores; `Stage2TemporalHeadV17`
compresses each non-overlapping 32-frame window to eight tokens and runs a four-layer
Transformer CTC sequence model over up to eight rolling windows. The accepted selector
combines the 63-dataset-local-voice context-adapted primary with the signer-transition
specialist using a phrase-agnostic 90/10 blend, +0.30 blank bias, and exact CTC path
comparison for different same-length multi-sign hypotheses.

Existing development evidence remains the reason this path is relevant: relative to
the primary, the selector improved held-out ASLLRP phrase edits 11/24 to 9/24, local
phrase edits 7/259 to 6/259, and held-out contextual-sign edits 44/254 to 43/254. The
separate time-normalized live Stage-2 replay was exact on five familiar local phrases.
These are development/familiar-domain results, not independent continuous accuracy.
Real webcam sessions produced unstable additions and 235-282 ms typical post-landmark
updates because MobileCLIP crop encoding dominates, so `live_reel_stage1_v17.py`
correctly keeps Stage 2 opt-in via `--stage2-arbiter`; `live_stage2_ctc_v17.py` is the
direct sequential-transcription entry point. The newer aligned causal head is faster
but materially less accurate (selected local signer WER 37.04%, NCSLGR WER 78%) and
does not replace this selector. No code, checkpoint, dataset, or protected split was
changed or accessed in this audit. Next transcription work should preserve the proven
selector and address active-window gating/hand-encoder reuse plus new signer-disjoint
continuous data rather than resume avatar generation.

## 2026-09-07 11:52 PST — source-bank smoke passes; full100 extraction started

Implemented scripts/build_avatar_gloss_bank_v17.py and test/test_avatar_gloss_bank_v17.py.
Tests first failed on missing module; strict split/class/raw-gloss/code filtering and
one-handed participation now pass. Candidate ranking uses complete-hand coverage
with P52tie preference; only official train provenance is read for source selection.
Both-hand matching reuses Apple/MediaPipe helpers and coherent palm interpolation.
Missing world observations are preserved in audit masks; classes with no usable
participating hand fail explicitly. YOU/NEED accepted symbolic cores copied exactly;
98others remain source-world estimates with unverified dictionary candidates attached.
Smoke YOU NEED HELLO HOW completed successfully (93368), report avatar_gloss_bank_v17_smoke.
Extended render_signwriting_sequence_v17.py to accept bank format, preserve each
hand's participation/rest state, and verify cache hash. All29bank/rig/mesh tests plus
 git diff --check pass. Started mixed HELLO HOW YOU smoke render and full100bank build
under artifacts/reports/avatar_gloss_bank_v17_v1. Inspect results before claiming
coverage/quality. No classifier test, no training promotion; full objective active.

## 2026-09-06 16:34 PST — YOU hand surface flip reproduced and corrected; joint context measured

The rendered v8 hand was inspected at0/45/90degree views. A second root cause was
confirmed by a failing mesh regression: projecting the palm normal onto each bent
phalange reverses its surface roll when flexion crosses90degrees. Shared renderer
now uses the transverse palm hinge for index/middle/ring/pinky surface orientation;
thumb/palm retain their previous mapping. Regression changed RED to GREEN; all15
mesh/rig tests pass. New source comparison is being rendered; no human acceptance.

Joint context evaluation completed in joint_context_v2.json. At contextweight1,
local184/200exact(92%),3.52%WER versus raw105/200,21.30%WER; Citizen358/378,
SemLex771/978. ASLLRP worsens14to16edits/24tokens; NCSLGR83to77/121. Stronger
weight10 reaches200/200 local but creates additional blank errors and damages
previously correct isolated sequences. This six-template development result is not
unseen-combination accuracy. Do not promote this weighting or remove recording
requirements based on this result alone. Runtime still has the earlier beam-only
context path; candidate-union experiment is not silently integrated.

## 2026-09-06 10:46 PST — mirror and transition-snap regressions fixed; source-world comparison running

Corrected anatomical-left/right mapping consistently in `avatar_rig_v17.py` and
`render_rigged_avatar_v17.py`: source X is no longer reflected, left slots map to
MakeHuman L, right to R. New source-side regression passed after failing. Replaced
transition bone-direction vector blending in `signing_voice_phrase_v17.py` with
shortest-arc 2D angle interpolation over interior samples; opposite endpoints now
rotate instead of snapping180 degrees mid-join. This only fixes the proved angular
artifact, not human plausibility or real3D generation.29 focused avatar/signing-voice
tests passed. Added optional animation-only `hand_world_xyz` retargeting, with a new
foreshortening regression passing after initial failure; all six rig tests pass.

`scripts/compare_avatar_source_v17.py` is extracting the installed, hash-pinned
MediaPipe model's world estimates from the three exact Citizen train source videos,
then rendering Actual source / Rejected avatar / Source-world retargeting audit.
Output `avatar_source_comparison_v17_v2`, log `continuous_rebuild_v17_v1/avatar_source_v2.log`.
True source hand depth is not claimed: these remain monocular detector estimates.
Missing hands have an explicitly unverified neutral fallback; generated joins have
no corresponding source footage and are labeled as such. Source phases align to the
old timeline for diagnosis; real timing restoration and motion plausibility are open.
No Apple recognition model/schema is changed and no synthetic data is admitted.

## 2026-09-06 10:10 PST — inherited phrase exposure found; clean-lineage retrain started

Checkpoint provenance shows that `stage1_v17_asllrp_core_adapt_v1` inherits
`stage1_v17_phrase_adapt_reel_v2`, trained on the original randomly partitioned local
phrase corpus. Re-splitting only the causal-head inputs does not remove that encoder's
prior local-phrase exposure. Therefore the latest grounded local holdout is a
head-training holdout, not proof of fully unseen-signer generalization throughout the
pipeline. Earlier reports must be read with that additional limitation.

The clean restart is the original Citizen+SemLex-only partwise checkpoint at
`artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/best_model.pth`.
Its stored provenance contains 1,475 Citizen and 1,388 SemLex training samples and no
phrase adaptation. A 14-epoch retention-gated run of the existing grounded adaptation
trainer was started with that explicit base, output
`artifacts/models/stage1_v17_grounded_clean_lineage_v1`, log
`artifacts/reports/continuous_rebuild_v17_v1/stage1_clean.log`. Local training uses only
the corrected training identities. Local core partitions remain weak/equal alignment;
only ASLLRP/NCSLGR cores have manual/published timing. No result is available yet.

## 2026-09-06 — authorized implementation started; supervision defect found

The user explicitly authorized the complete generation/avatar/continuous-recognition/
evidence-aware Stage-3 work and requested completion before a final return. The active
thread goal remains open until all requested behavior is verified. A bounded independent
worker owns a realistic rigged-avatar prototype; the main thread owns recognition,
data supervision, contextual decoding, and this handoff.

Further source inspection found that historical `local_phrase_samples` labels a crop
around equal-partition guessed boundaries as blank, while `asllrp_phrase_samples`
includes 35% of each adjacent annotated lexical core in its blank crop. Thus the
historical `transitions_only` setting is not equivalent to pure annotated nonlexical
gaps. The next experiment will obtain negatives only from actual annotated gaps,
retain unaligned local phrases as sequence supervision, and apply identical rolling
observation to isolated replay and continuous samples. A multi-duration evidence path
will be compared with the existing short-window baseline; no protected test is used.
This is an identified code mechanism, not yet a measured correction gain.

## 2026-09-04 11:03 PST — short-window Transformer and natural-video target review

No new model was trained and the paused Kaggle RGB/VideoMAE experiment was not started.
The existing matched 378-clip signer-disjoint benchmark makes the eight-layer flat
landmark Transformer a credible efficiency challenger: 95.50% top-1, 95.16% macro-F1,
6,715,621 parameters, and 2.88/3.00 ms median/p90 batch-one CPU inference, versus the
selected part-wise/global Squeezeformer at 96.30%, 96.14%, 6,791,717 parameters, and
6.15/6.25 ms. The top-1 difference is three clips and comes from one seed, so it is not
yet evidence to replace the selected model. The four-layer flat Transformer gives a
second latency point at 94.71% and 1.76/1.91 ms. A generic convolution adapter and a
part-wise Transformer both performed worse, so adding complexity without a targeted
hypothesis is rejected.

The local `SqueezeformerBlockV17` is Squeezeformer-style rather than a reproduction of
the full speech architecture: it uses two feed-forward residuals, multi-head attention,
and depthwise temporal convolution, but not the original long-sequence Temporal U-Net
or downsampling. At 32 frame tokens, quadratic attention is small; Performer/linear
attention and a full RGB ViT are therefore not selected next steps. The smallest useful
architecture test is a 6-layer, 256-dimensional flat landmark Transformer, followed by
multiple-seed paired comparison against the current eight-layer Transformer and
Squeezeformer. Promotion requires preserved isolated accuracy plus improved continuous
phrase WER/latency on new signer-disjoint recordings.

The reviewed BBC Young Reporter file is a 136.97-second, 1920x1080, 25-fps edited
video. It alternates frontal upper-body signing with tight face shots, title cards, and
frequent cuts; some shots remove one or both hands from view. Its own captions identify
the signing as BSL. BSL and ASL are distinct languages, so the current ASL-labelled
100-gloss model must not be evaluated on it as if it were ground-truth ASL. It is useful
only as a visual/use-case reference unless the project deliberately adds a separate
BSL target and corpus. The feasible near-term goal remains online spotting of the
project's 100 known glosses in unedited, full-upper-body, single-signer video, with an
explicit UNKNOWN/BACKGROUND output; arbitrary BBC-style story translation is outside
the evidence supported by 30 phrases.

The planned 30 phrases x 13 signers x 3 genuine performances yield 1,170 independent
phrase performances. Three simultaneous cameras would yield 3,510 files but remain
correlated views of those 1,170 performances. Preserve the 10 train / 2 validation /
1 sealed signer contract. Capture connected signing without inter-gloss neutral poses,
prefer two natural-speed and one slower-but-connected performance, retain face/hands/
upper torso for every frame, and record synchronized camera/performance IDs. Exact
gloss sequences are required. Manually mark sign cores, transitions, and background for
all validation and sealed recordings plus a representative training subset; use
training-only forced alignment for the remainder and review low-confidence boundaries.

Primary literature supports an online dictionary/window recognizer as the next bounded
continuous experiment: Zuo et al. (EMNLP 2024) train an isolated recognizer with an
explicit background/coarticulation class, boundary-jittered clips, foreground saliency,
sliding windows, and duplicate voting. This is closer to the desired Reel behavior than
full-utterance CTC and is selected as the research blueprint. Backbone comparison is
secondary to train/inference window alignment, background supervision, and boundary
evaluation. No protected test or sealed data was accessed.

## 2026-09-04 10:46 PST — Kaggle VideoMAE execution paused before training

At the user's request, no RGB/VideoMAE training or benchmark job has been submitted.
The repository's isolated Kaggle CLI authenticated as `kokoab`. A private CUDA-only
probe, `kokoab/slt-v17-videomae-gpu-probe-v1`, completed successfully on an actual
Tesla T4 with 15,636,037,632 bytes of VRAM, PyTorch 2.10.0+cu128, and CUDA 12.8. This
resolves the allocation gate that invalidated an earlier MoViNet attempt; it is only a
hardware probe and contains no model or accuracy result.

A local, not-uploaded train/validation RGB archive was prepared at
`artifacts/generated/kaggle_citizen100_rgb_trainval_v1/citizen100_rgb_trainval_v1.tar`.
It is approximately 1.0 GiB, has SHA-256
`3fc7a54b2111e145e59b2217eb358b6ad423bf6b9681fda5681ea6b8758b7060`, and contains
the 1,477 raw train clips, 378 raw validation clips, manifest, and rejections ledger.
An archive listing found no `/test/` member. It has not been uploaded to Kaggle. The
proper speed comparison remains end-to-end: live Apple Vision extraction plus the
landmark classifier versus RGB preparation plus the complete video encoder, while
separately reporting the observation/window delay. Existing measurements do not
support claiming that a full RGB video Transformer is faster than the skeletal path.

## 2026-09-04 07:54 PST — RGB video Transformer scoped as a challenger, not a replacement

A review of the matched local RGB experiments and primary video/sign-Transformer
literature establishes that a pretrained **spatiotemporal** visual model is a genuine
future Stage-1 challenger, but a framewise/plain ViT is not an evidence-backed
replacement for the selected landmark path. Existing visual-only results on the same
378-clip validation set remain 39.68% for frozen full-frame MobileCLIP2, 70.37% for
frozen high-resolution hand crops, 70.63% for the spatial/TSM branch, and 80.69% for
the later balanced hand-crop model, versus 95.50% for the current matched landmark
component and 96.30% for learned landmark/RGB fusion. MobileCLIP2-S0 already uses an
11.4M-parameter FastViT hybrid image tower, but generic image pretraining plus a
temporal head is not equivalent to end-to-end sign-specific video pretraining.

If this direction is executed, the proper bounded experiment is a pretrained small
video encoder (preferably a sign-specialized Hiera/VideoMAE-style checkpoint, otherwise
VideoMAE-Small or Video Swin-Tiny) over identical hand-and-upper-body crops and the
unchanged signer-disjoint split. It must report visual-only accuracy, end-to-end live
latency including crops/decoding, memory, and robustness to signer/background/lighting.
No RGB model should replace landmarks unless it matches or exceeds the landmark result
and satisfies the live-device latency budget. The practical current architecture
therefore remains fast landmark proposal plus selective RGB evidence; no production
model or protected evaluation was changed by this review.

## 2026-09-04 07:50 PST — compact Transformer establishes a credible efficiency frontier

Three additional experiment-only Transformer challengers were trained with the exact
same 2,863-sample balanced training data, 378-clip signer-disjoint validation set,
augmentation, optimizer/schedule, EMA selection, seed 1701, MPS training, CPU restored-
checkpoint validation, and protected-test exclusion as the architecture-family
benchmark.

The four-layer compact flat Transformer has 3,556,581 parameters and selected epoch 52
at 94.71% top-1, 98.94% top-5, and 94.50% macro-F1. It measures 1.76/1.91 ms median/p90
batch-one single-thread CPU latency. Relative to the eight-layer flat Transformer, it
loses only 0.79 validation point while reducing parameters by 47.1% and latency by
39.1%. It is the new compact/latency challenger, while Squeezeformer remains the
accuracy selection at 96.30%.

The 6,916,069-parameter convolution-augmented Transformer adds a gated depthwise
temporal adapter before the unchanged eight-layer global stack. It selected epoch 59
at 94.18% top-1, 99.47% top-5, 93.75% macro-F1, and 3.45/3.54 ms latency; this simple
local-convolution addition does not reproduce Squeezeformer. The 6,854,373-parameter
anatomical-token Transformer converts every frame into left-hand, right-hand, face,
and body tokens, applies spatial self-attention, then the same global temporal stack.
It selected epoch 92 at 93.12% top-1, 99.21% top-5, 92.71% macro-F1, and 3.50/3.60 ms;
four-token compression loses useful detail. A literal ViT is deferred because it would
consume RGB patches and must be compared with the hand-crop RGB branch, not the
landmark table.

The aggregate now contains ten measured architectures; the report and chart identify
Squeezeformer as the accuracy choice, the eight-layer flat Transformer as the closest
same-capacity competitor, and the four-layer Transformer as the compact choice. All
three new checkpoints reproduce their recorded metrics in fresh CPU evaluation; their
two-epoch smoke, compilation, chart inspection, report checks, and `git diff --check`
pass. No production runtime/checkpoint changed and no protected test was accessed.
These remain single-seed validation results; a three-seed comparison plus independent
evaluation is required before promoting the compact Transformer.

## 2026-09-04 05:27 PST — part-wise Transformer does not beat the flat Transformer

The original matched Transformer baseline was confirmed to be flat: all 61 nodes and
66 hand-distance features are concatenated per frame, projected to 256 dimensions, and
passed through eight global Transformer encoder layers. A new experiment-only
`partwise_transformer` challenger now applies one 64-dimensional Transformer layer to
each left-hand, right-hand, face, and body stream, concatenates/fuses them, and then
uses the same eight-layer 256-dimensional global Transformer. It has 6,848,549
parameters, closely matching the flat Transformer's 6,715,621 and Squeezeformer's
6,791,717.

Under the identical seed-1701 training/validation protocol, the part-wise Transformer
selected epoch 105 and stopped at epoch 135. Its restored checkpoint scores 93.92%
top-1, 99.21% top-5, and 93.62% macro-F1 on the 378-clip signer-disjoint validation
set, with 3.53/3.92 ms median/p90 batch-one CPU latency. It is 1.59 top-1 points worse
and 0.69 ms slower than the flat Transformer (95.50%, 2.84 ms), and 2.38 top-1 points
worse than the part-wise+global Squeezeformer (96.30%). This rejects the hypothesis
that anatomical splitting alone explains the selected model's advantage; the evidence
supports the combination of part-wise streams with Squeezeformer's local convolution
and attention blocks. The hybrid's higher 99.21% top-5 indicates ranking ambiguity,
not a top-1 win.

The benchmark aggregate, Capstone table, and accuracy/latency/parameter chart now
contain seven families and label the original baseline `Flat Transformer`. The new
checkpoint reproduces its metrics after a fresh CPU restore; its two-epoch smoke,
compilation, report checks, and `git diff --check` pass. The protected test was never
loaded, and production runtime/model files were not changed. This remains a single-seed
validation comparison.

## 2026-09-04 04:57 PST — matched Stage-1 architecture-family experiment completed

The Capstone revision package no longer uses a literature-only architecture table.
`scripts/benchmark_stage1_families_v17.py` trained real BiLSTM, BiGRU, Temporal CNN,
Transformer, compact ST-GCN, and current part-wise+global Squeezeformer baselines with
the same 2,863-sample class/source-balanced training corpus, exact augmentation,
optimizer/schedule, label smoothing, EMA selection, seed 1701, and the same 378-clip
signer-disjoint validation set. Training ran on MPS; reported metrics were reproduced
from the persisted checkpoints on CPU. The protected test was never loaded.

Validation top-1 / macro-F1 was 89.15/88.06 BiLSTM, 93.92/93.36 BiGRU,
92.59/92.01 Temporal CNN, 95.50/95.16 Transformer, 62.43/61.14 compact ST-GCN,
and 96.30/96.14 Squeezeformer. A matched batch-one, single-thread CPU benchmark over
one real validation tensor (20 warmups, 300 predictions) measured median latency of
3.34, 3.21, 4.32, 2.82, 4.29, and 6.44 ms respectively. The corresponding parameter
counts are 6.10M, 6.14M, 5.90M, 6.72M, 0.64M, and 6.79M. Squeezeformer is the measured
accuracy winner; Transformer is the measured latency winner. The compact ST-GCN result
must not be generalized to all ST-GCN capacities.

An MPS `nn.LSTM` packed-weight cache initially made an in-process EMA swap appear to
score 93.65%, although a fresh restore scored 88.10%. The runner now validates EMA
states in a separate CPU model and requires the persisted checkpoint to reproduce its
reported top-1; the corrected BiLSTM checkpoint selects epoch 47 at 89.15%. All six
checkpoints independently reproduce their recorded top-1/top-5/macro-F1. Raw results
and checkpoints are under
`artifacts/reports/capstone1_v17_revision_checklist_v1/stage1_family_benchmark/`.
The report now includes a measured table and a three-panel accuracy/latency/parameter
chart generated from that JSON. This is a single-seed validation comparison; repeat
three seeds before making publication claims about small differences. Production
runtime code and the selected deployed checkpoint were not changed. The six-family
two-epoch smoke, Python compilation, CLI help, independent checkpoint evaluation,
Markdown image-link check, private-report corpus-name scan, and `git diff --check` pass.

## 2026-09-03 20:41 PST — original Stage-1 chart separated from adaptation; RTMW screening recovered

The Capstone revision report now distinguishes two real training histories. The primary
`stage1-training` chart reads the complete 130-epoch history of the selected v17
part-wise+global landmark Squeezeformer from
`artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/stage1_v17_partwise_v2/history.json`;
it marks the selected epoch 100, its 96.83% validation top-1/99.21% validation top-5,
and separately labels the frozen 87.57% held-out test top-1. It contains no contextual
adaptation epochs. The 30-epoch frozen-encoder fusion adaptation chart remains available
under the new, unambiguous `stage1-final-adaptation` name.

The repository's historical RTMW comparison was recovered from
`docs/md_files/SIMPLIFICATION_TEST_RESULTS.md`. Its strict matched scope is an older
extractor-only screening of 500 videos and 3,960 evenly sampled frames: Apple Vision
5.0 ms/frame and 80.6% frame output, optimized MediaPipe 28.0 ms/frame and 75.9%, and
RTMW-XL 447.6 ms/frame and 100% emitted output. RTMW's value is not hand-detection
accuracy because its top-down whole-body model emits both-hand estimates after person
detection, including absent/occluded-hand guesses. This legacy comparison is now a
separate table/chart and is not combined with the current v17 Apple Vision/MediaPipe
quality or downstream-classifier bakeoff. Conflicting historical ghost-hand percentages
use different denominators/stress protocols and are intentionally not collapsed into
one claim. Chart regeneration, script compilation, Markdown image-link validation,
dataset-brand scan of the user-facing report, and `git diff --check` pass. No runtime
code, checkpoint, dataset, or protected test was modified or accessed.

## 2026-09-02 20:46 PST — always-visible auxiliaries separated from the trained model schedule

The newest completed webcam session,
`artifacts/reports/live_reel_stage1_v17/20260902_201143_747409/`, confirms the optimized
path is substantially smoother: it completed 5,007 landmark observations over 264.09
seconds (18.96 FPS against the 20 FPS target) while committing the familiar live
sequence HELLO HOW YOU exactly. Its button log also disproves a total click-handler
failure: it recorded 45 RESET and two FINISH actions. The apparent lock came from no
visible acknowledgement on an empty RESET, FINISH waiting for an in-flight verifier,
and rapid repeated clicks submitting a second empty action.

Face/body detection was benchmarked and evaluated before promotion. On 230 recorded
640-pixel frames, Apple hand-only detection cost 3.80 ms median / 6.12 ms p90, while
hands plus face and body every frame cost 13.65/16.46 ms; MediaPipe's separate 40-point
lip pass remains about 3.11 ms median. On 29 official Citizen validation clips from the
targeted YOU, SICK, FATHER, MOTHER, GOOD, THANKYOU, HUNGRY, and HELLO classes, feeding
training-sparse versus every-frame Apple face/body landmarks produced zero changed
predictions and the same 26/29 accuracy. No test split was accessed.

The local phrase experiment found a stricter boundary: dense model auxiliaries kept
HELLO HOW YOU exact but changed GOOD MORNING to THANKYOU EASY, whereas always-on
detection with the trained sparse model schedule restored exact GOOD MORNING and kept
HELLO HOW YOU exact. The default is therefore now intentionally split: Apple face/body
detection and the last valid overlay remain visible on every processed frame, but the
Stage-1 tensor receives face/body evidence every eighth frame, matching training.
`--dense-model-auxiliary` retains the experimental dense-input path. Sparse training
frames are boundedly interpolated during feature construction; they do not become
literal discontinuous jumps inside the final resampled tensor.

The reel overlay now uses thin one-pixel white bones and two-pixel white joints for
hands/body, plus thin white lip contours. RESET and FINISH use one shared enlarged
geometry for drawing and hit-testing, debounce rapid duplicate clicks, show immediate
`Display reset.` / `WORKING` feedback, and ignore old naturalizer output after RESET.
An already-running FINISH receives explicit feedback rather than silently queuing an
empty second utterance. An overconfident 0.9977 wrong lip-only GOOD->THANKYOU override
was also exposed by the phrase replay, so the tiny 27-example lip specialist now needs
0.999 confidence to override matching hand/body evidence; the observed genuine
THANKYOU correction at 0.99999999 remains eligible. Full results are under
`artifacts/reports/live_reel_auxiliary_every_frame_v1/`,
`live_reel_dense_auxiliary_v1/`, and
`live_reel_always_display_sparse_model_v1/`. Thirty-eight focused tests pass,
compilation and `git diff --check` pass, and the original isolated behavior is
unchanged.

## 2026-09-02 11:00 PST — low-motion/non-neutral boundaries are cleaner but still fail accuracy

A second separate path, `scripts/live_motion_valley_v17.py`, now closes a sign after a
short low-motion hold at the current hand position rather than requiring return to a
neutral pose. The original neutral/pause path and the overlapping-window experiment
remain available unchanged by default. The motion path keeps the same extractor,
Stage-1 classifier/gates, display, RESET/FINISH, tiny Stage 3, speech, history, and
low-resolution recording. Saved-video segmentation waits for each classification, so
offline frames are not dropped while Core ML is busy.

The initial motion ≤0.006 for 0.12 seconds configuration produced 15 clips and nine
accepted glosses across all nine local development/reference phrases. It scored 1/9
exact overall and 1/5 among fully vocabulary-covered phrases; only MY_NAME was exact.
Median/p90 classification latency was 448.01/529.23 ms. Compared with fixed overlap,
it reduced spurious insertions and recovered more meaningful components, but often
merged neighboring signs.

The nine phrase recordings were then used as intended for weak boundary calibration.
A motion-only sweep found that motion ≤0.008 for 0.16 seconds matches the expected sign
count in all five fully vocabulary-covered phrases. Reclassification of those same five
fitted development videos still scored only 1/5 exact: GOOD_MORNING -> `EAT MORNING`;
HELLO_HOW_YOU -> `HOW NEED`; MY_NAME -> `MY NAME`; THANKYOU_FRIEND -> `NAME`;
TOMORROW_SCHOOL_GO -> `TOMORROW GO`. Twelve clips were formed, nine accepted, with
360.94 ms median and 490.77 ms p90 latency. Better endpoint counts did not solve the
continuous/local classifier-domain errors. This fitted diagnostic is not independent
accuracy evidence.

Therefore phrase data is not needed merely to run a webcam loop, but remains essential
for threshold fitting, transition/domain learning, and honest continuous evaluation.
The existing 780 local recordings should be retained. Their phrase prompts can support
weak sequence training, but missing signer identity, five OOV prompt glosses, and absent
frame boundaries limit claims. Do not delete the phrase corpus, promote either new live
path, or lower gates to make the numbers look better. Full evidence is under
`artifacts/reports/live_motion_valley_v17_local_phrase_eval_v1/` and
`artifacts/reports/live_motion_valley_v17_local_phrase_tuned_v1/`.

## 2026-09-01 21:20 PST — five-phrase grounded scalability gate passes after hand-kinematic repair

The grounded path now passes every locally benchmarkable conversational phrase whose
literal glosses exist in the Citizen 100: `GOOD MORNING`, `HELLO HOW YOU`, `MY NAME`,
`THANKYOU FRIEND`, and `TOMORROW SCHOOL GO`. Batch generation may select a different
train-only signer for each utterance while keeping every utterance internally
same-signer. The selected signers are P50, P11, P37, P50, and P27 respectively. All
12 exact isolated components are correctly recognized by frozen Stage 1 (confidence
range 0.5635--0.9550), and source versus final-rig hand participation matches for
every gloss. Five independent selected-signer `SORRY` audits remain exactly one-handed
and classify correctly.

The first five-phrase artifact (`...scalability_v1`) is rejected despite its older
gates: contact-sheet inspection exposed brief hand-skeleton shrinkage inside several
transitions. The cause was linear interpolation of absolute finger coordinates across
different endpoint poses. `stabilize_transition_hands` now retains the learned wrist
trajectory but reconstructs all finger chains with interpolated endpoint-derived bone
lengths. It preserves the original node/side presence mask, so it cannot add a hand or
create nodes present only inside a transition. A regression test begins with a fully
collapsed transition, restores every right-hand bone above its endpoint floor, and
keeps the source-absent left hand exactly zero.

The replacement output is
`artifacts/reports/stage2_v17_grounded_text_to_sign_scalability_v2/`. Across the five
phrases, p95 speed is 0.927--1.140x, acceleration 0.856--1.150x, and jerk
0.680--1.080x aggregate genuine local train. Transition/gloss p95 speed is
0.594--1.836x; median transition/gloss hand-bone length is 0.707--1.933x. Every
transition has zero handless frames, zero nodes present only within the generated
span, complete render hands when active, and complete face/body on every render
frame. All five machine gates pass. `index.html` provides a native-review page and
`contact_sheet.png` provides the visual audit overview. These remain synthetic
native-review-only artifacts and are explicitly ineligible for training, validation,
or testing.

The train-only Citizen inventory contains 32 signers. P33, P37, and P52 each cover
all 100 glosses; consequently every unordered pair (4,950/4,950) and triple
(161,700/161,700) has at least one same-signer source route. This proves source
coverage for a 50-phrase expansion, not linguistic validity or transition quality.
New phrases must still be curated by native signers and pass the same per-phrase
recognition, motion, bone, presence, render, and native-review gates. No sealed or test
split was accessed.

## 2026-08-22 10:52 PST — synthetic transitions falsified; genuine multi-signer pilot started

The existing feature-space interpolation/style-transfer generator does **not** pass a
natural-human-motion gate.  `scripts/audit_stage2_transition_naturalness_v17.py`
matched generated and genuine train-only ASLLRP spans by exact ordered gloss pair and,
for the strict analysis, by the same signer.  A frozen label-independent temporal
descriptor plus train-fold-only PCA/logistic classifier separated 18 strict paired
examples (36 rows) with 97.2222% balanced accuracy and 0.99691358 ROC AUC.  Two hundred
paired permutations gave a null balanced-accuracy mean of 0.47346 and p=0.004975.  A
broader genuine-signer-held-out analysis over 33 pairs (66 rows) still reached 90.9091%
balanced accuracy and 1.0 ROC AUC.  The complete report is
`artifacts/reports/stage2_v17_transition_naturalness_audit_v1.json`.  Therefore the old
generator remains recognition augmentation only; it must not be described as genuine
coarticulation, timing, or signer-style synthesis.

The phrase-agnostic selector was also distilled into full-model and head-only compact
students across bounded seeds.  The best students reached 11/24 ASLLRP, 7/259 local,
and 43/254 JONATHAN edits.  They retained the contextual-sign benefit but did not
transfer the rare ASLLRP corrections, so neither replaced the selected two-head
research model.  Parameter soups likewise bottomed out at 11/7/43.  The selected
general selector remains 9/6/43.

A genuine-data experiment is now in progress using a bounded, train-only How2Sign
subset.  The official corpus provides continuous signing by 11 signers under CC
BY-NC 4.0; a per-file mirror was pinned at commit
`cfe9b6482aa34d6f6bda1974a7b7cae822c16613`.  Metadata was downloaded and a
deterministic plan selected 1,027 clips (1.6801778 hours), 348 source videos, and all
six signer IDs present in that train shard: signer counts 24, 23, 8, 172, 400, and
400.  Selection is capped at eight clips per source video and includes every eligible
clip from the three rarest voices.  The resumable state and plan are under
`data/local/how2sign_transition_subset_v17/`; at this timestamp 597/1,027 files had
completed with zero failures and about 899 MB downloaded.  How2Sign validation and
test are untouched.

New preprocessing preserves the genuine 32-frame trajectories and all lip nodes:
`scripts/prepare_how2sign_transition_manifest_v17.py` and
`scripts/extract_how2sign_transition_landmarks_v17.py`.  A new
`TransitionInpainterV17` masks a contiguous interval and reconstructs it from visible
context.  Its style representation is the visible-frame mean and variance from the
same clip; it has no signer-ID embedding, so the planned `how2sign:8` signer-held-out
evaluation measures adaptation from context rather than identity lookup.  Training is
capped at 10% MPS allocation with `num_workers=0` and compares spatial, velocity, and
acceleration errors directly against linear interpolation.  Four focused shape,
context-preservation, interpolation, and finite-loss tests pass.  Even if it beats the
linear baseline, that will establish feature reconstruction only; a held-out
discriminator/perceptual gate is still required before any natural-motion claim.

No Citizen, SemLex, local, or How2Sign sealed split, 2M-Flores `devtest`, consumed RIT
test, or JONATHAN data was accessed for these experiments.

## 2026-08-17 00:28 PST — all 155 2M-Flores landmark/RGB archives pass integrity

Long-video Apple Vision landmark and hand-RGB extraction is complete for all 155
selected 2M-Flores sentences with zero failures. Native detector RSS rose on long
4K-origin videos but system memory stayed green; the run was deliberately split into
short processes after row 84 so native allocations were returned between chunks.
Every save is atomic and the resumed workers reused compatible completed archives.

The independent fail-closed audit matches all 155 expected archives with no missing,
unexpected, or schema-incompatible file. It covers 2,718 nonoverlapping temporal
windows, 32 windows without a valid landmark result, and mean hand-view validity of
94.4719%; every archive has usable landmark and hand evidence. Total archive size is
1,395,267,704 bytes under long-video schema fingerprint `277d70d19c5cbb42`. The final
extraction report SHA-256 is
`2e25e187f7de4de2eb967a73532bfe7ebcb4fcb8c39d748cb2d2a9142162e754`; the audit
SHA-256 is `084f84201e3e4dac4d2ff1eec11d0b339ad0dde6a615c48a8908ffe026546a40`.

The dual-head Stage 2 implementation now warm-starts the exact selected v2 locked
100-sign head, extends its positional table from 8 to 40 windows, and adds a
training-only 448-class full-gloss auxiliary CTC head. Its focused shape, warm-start,
long-collation, and validation-priority tests pass. MobileCLIP2 hand encoding has
started in sequential eight-file workers at an 8% MPS memory cap. No evaluation split
was accessed.

## 2026-08-16 13:00 PST — train-only contextual replay features complete

The locked 1,116-row ASLLRP contextual replay set has completed the full Stage 2
preprocessing path. All 1,116 bounded MobileCLIP2 archives were written in 18 short
workers at an 8% MPS memory cap with zero failures; peak MPS driver allocation was
65,765,376 bytes. Its independent audit passes all 1,116 archives, 1,117 windows, and
53,137 valid RGB views under schema fingerprint `fd3110e2db69da2e`. The frozen
selected Stage 1 temporal cache then wrote all 1,116 feature archives with zero
failures in 85.3 seconds, at feature dimension 612 and peak MPS driver allocation
104,677,376 bytes. The frozen Stage 1 checkpoint remains pinned to SHA-256
`1caeadf4b3ca620aa9fef00b35c012b39d7c093f67da1ee2f6987d2c2297906b`.

These rows are training-only: BENJAMIN_JAMES_BAHAN, CORY, and RACHEL are included;
JONATHAN validation and all consumed RIT external rows remain excluded. The next
candidate may be selected only on the unchanged local+JONATHAN validation gate. RIT
must not be rerun, and no Citizen, SemLex, or local sealed test split was accessed.

## 2026-08-14 13:06 PST — local multimodal challenger completed; retain general teacher

All finalized local hand features are complete and exhaustively audited. The bounded
visual-only MobileCLIP2 path produced exactly 13,381/13,381 train and 2,896/2,896
validation embeddings across all 94 admitted classes. The 27-worker bulk-train run
took 6,663.8 seconds with image batch 32, one archive and one worker at a time, a 12%
MPS process ceiling, zero failures, and zero swap. Peak Metal-driver allocation was
1,432,961,024 bytes and maximum RSS was 1,623,146,496 bytes. Run report SHA-256 is
`48bcae48fd187652b5d91f1726d43a8b88968e297dbde929153c4813a61d3213`.
The fail-closed auditor now rejects both missing and unexpected embedding inventory
members in addition to checking every archive's shape, finiteness, unit norm,
explicit missing-view zeros, source item, label, manifest hash, split, eligibility,
and sealed-test provenance. Train and validation audit SHA-256 values are
`db978e2a7b36761af46d22a874b0dbb2a26e898afb48bcd63b42ff2c7b861063`
and `f6334cc448cede0285857b6b95ca0376b0ceec136fd65a03a5ec50a173af2f10`;
both report zero errors and exact crop/embedding valid-view agreement.

The hand replay protocol was locked before optimization at SHA-256
`204281e97d6cb195a7bdcd84ffe46e5428fdbe699d2697b521a2613215e90948`:
exact source checkpoint, seed 1701, 34/33/33 Citizen/SemLex/local replay, 100-epoch
maximum, 20-epoch patience, 5e-5 peak learning rate, four warmup epochs, batch 64,
streamed `--no-cache` data, and Citizen-primary checkpoint selection with local top-1
allowed only on an exact Citizen tie. The exact smoke reproduced the source at
305/378 = 80.69% Citizen and 1,684/2,896 = 58.15% local before optimization. The
definitive run stopped after 20 stale epochs. Its live EMA reached 2,839/2,896 =
98.03% local at epoch 20, but only 284/378 = 75.13% Citizen; the best post-update
Citizen epoch was 299/378 = 79.10%. Therefore every adapted hand state was correctly
rejected. The selected output checkpoint, SHA-256
`7e920d0916842afe4fae284a383aad6c68eced940505a9d66f559f2b983b465e`,
has all 108 model tensors bit-exact to source checkpoint SHA-256
`ec16d1b14a2346fecd993d92b3c92b4965cd1204132f64514d86c109570e6d84`;
its differing file hash comes only from the new provenance/checkpoint envelope. Run
result SHA-256 is
`1ca647a9841ce18bcbed58ffb45bc48266413bee840be9015c31164f6ab9c23d`.

The final fixed-weight evaluations use the promoted mouth-safe local landmark branch,
the retained exact hand branch, and byte-unchanged frozen mouth/lower-face branches.
Per-sample z-score fusion at fixed 0.30 landmark / 0.15 mouth / 0.35 lower face /
0.20 hand reaches 366/378 = 96.83% Citizen and 887/978 = 90.70% SemLex. This is four
Citizen clips below the retained teacher's 370/378 despite improving SemLex over
882/978, so it is **not** the new general teacher. Report SHA-256 values are
`f0b9fe24652b380dfbabdfc097d41b90cabf41212114f0eb931a39df4a283ff4`
and `3d7e011af89d0e4e097c9fa27acb07bb9ba0fa04d518dc1ed762d9f0175c1654`.
On local familiar-signer validation, the separately frozen 75/25 landmark/hand fusion
reaches 2,801/2,896 = 96.72%, 99.65% top-5, and 91.00% all-100 macro F1. It repairs
28 landmark errors while breaking 17, a net gain of 11 clips over the 2,790/2,896
adapted-landmark result. Its report SHA-256 is
`89ddaf26970e860d0b2ce79335fc7ff153d5fab5120ba55f4bae5c211a78c6d6`.
The adapted landmark still clears the existing eight-angle gate with a 356/378
worst-angle floor; the retained hand/face experts were not modified by orientation
training. The existing four-stream teacher at 370/378 Citizen and 882/978 SemLex
therefore remains the accuracy research default, while the completed local challenger
is retained as domain-adaptation evidence and a future independent-capture candidate.

The full affected suite is green: 100/100 focused tests pass, including 17/17 in the
real Apple Vision host environment; both exhaustive hand audits and all generated
JSON reports validate; `git diff --check` passes. The first sandbox-only run produced
three Vision setup errors, then the required host rerun passed all three real Vision
tests; no assertion failed. Citizen test, SemLex test, and local test were never
accessed. Simulator/physical-iPhone benchmarking remains explicitly deferred per the
owner's instruction.

## 2026-08-14 09:33 PST — memory-bounded visual-only encoder proven; local hand validation complete

The local MobileCLIP2 memory regression is fixed and measured. The exact visual-only
loader now instantiates only MobileCLIP2-S0's 11,406,976-parameter FastViT image tower
on the meta device and loads the 784 official `visual.*` tensors from checkpoint
SHA-256 `ab91a1a0c4330d6b1913e24d5035dfdea15423316aaec649610c6b1c6ddd0e95`.
One CPU comparison against a pre-restart MPS archive differed by at most one float16
unit (`6.103515625e-05`), consistent with CPU/Metal kernel rounding; valid masks and
boxes were exact. The bounded MPS supervisor keeps one archive and one worker live,
enforces a per-process memory fraction, records PyTorch/Metal telemetry, and exits the
Metal context after deterministic file shards. The full validation continuation ran
52 sequential 32-clip workers at image batch 16 and an 8% cap with zero failures and
zero swap. Peak Metal-driver allocation was 1,168,441,344 bytes and supervisor/child
maximum RSS was 1,108,426,752 bytes; worker peaks remained flat rather than accumulating.
Its report SHA-256 is
`99573b9d300162318a44c5dbf4ef8e49afab436819e1bc1fe8d40c0b8784119e`.

Exactly 2,896/2,896 finalized local-validation hand embedding archives now exist.
The exhaustive crop+embedding audit passes with zero errors across all 94 classes:
shape, finiteness, explicit zero missing views, unit normalization, manifest hash,
source, split, eligibility, and sealed-test provenance all match. Crop and embedding
valid-view fractions agree exactly at 0.7413458218232044. Audit report SHA-256 is
`f6334cc448cede0285857b6b95ca0376b0ceec136fd65a03a5ec50a173af2f10`.
The untouched selected hand checkpoint (SHA-256
`ec16d1b14a2346fecd993d92b3c92b4965cd1204132f64514d86c109570e6d84`)
scores 1,684/2,896 = 58.15% local top-1, 77.45% top-5, and 51.80% macro F1; this is
the frozen pre-adaptation hand baseline. Metrics SHA-256 is
`557feea178e6c2d89afe59a37b9fff7ba690a8f88b3a9120e6e0865f2c5061e0`.

A batch-32/12%-cap train smoke encoded 128 clips in 61.5 seconds at a 1.225 GiB Metal
peak and zero swap. Batch 48 was rejected despite remaining stable: it saved only 7%
wall time while increasing the Metal peak to 2.123 GiB. Bulk local-train encoding is
therefore pinned to image batch 32, one archive/worker at a time, a 12% hard cap, and
process recycling. Hand replay is additionally fail-closed unless `--no-cache` is
present. The mouth/lower-face experts remain frozen, and Citizen test, SemLex test,
and local test remain untouched.

## 2026-08-14 00:20 PST — mouth-safe landmark replay clears all promotion gates

The definitive exact-checkpoint landmark replay completed on laptop MPS after 22
epochs and the predeclared 20-stale-epoch stop. It used 1,475 Citizen train, 1,388
approved SemLex train, and 13,381 finalized local train archives with exact
34/33/33 class/source-balanced replay. Only the four local lip landmarks were
masked; all other local face anchors and all Citizen/SemLex lips remained visible.
The selected source checkpoint was restored exactly at 362/378 Citizen and
1,765/2,896 mouth-masked local validation before optimization.

The strict Citizen-best checkpoint is epoch 2: 363/378 = 96.03% Citizen and
2,215/2,896 = 76.48% local. The stronger promotion-gate checkpoint is epoch 21:
361/378 = 95.50% Citizen, 2,790/2,896 = 96.34% local, and 90.55% local macro F1.
On the frozen SemLex validation diagnostic, the Citizen-best checkpoint reaches
835/978 = 85.38%, while the promotion-gate checkpoint reaches 860/978 = 87.93%,
exceeding the old selected landmark's 839/978. The promotion checkpoint also improves
the eight-angle landmark stress floor from 348/378 to 356/378: per-angle correct
counts at 0/17/37/73/90/123/180/270 degrees are
361/362/361/357/357/356/356/358. It therefore clears every predeclared landmark gate
without touching Citizen test, SemLex test, or local test and is the landmark branch
candidate for the final multimodal evaluation.

Exact checkpoint SHA-256 is
`12a74a18d71712abf525350e8120e26694d24a62234b4c268639220a37da47a1`;
the conservative Citizen-best SHA-256 is
`d7969867c70335da1455d4b5964ef12e2160eb043f732f805c065d711675544b`.
SemLex reports are under
`artifacts/reports/semlex_citizen100_val_audit/local_deep_clean_replay_ft_*`;
orientation reports are under
`artifacts/reports/stage1_v17_orientation_robustness/local_deep_clean_replay_ft_*`.
The hand-RGB MobileCLIP2 encoding phase is now active; mouth/lower-face experts remain
frozen.

## 2026-08-13 23:01 PST — local trainval corpus finalized and Kaggle upload staged

Fresh current-v17 Apple Vision extraction is complete. Validation retained all
2,896/2,896 clips; train retained 13,381/13,382 clips. The only extraction rejection
is `HE/HE_SHE_6eb9df3e`, whose raw movie produced no hands; there were zero processing
failures. Both final splits still cover all 94 admitted local classes. Every one of
the 16,277 final train/validation archives passes the current v17 integrity audit with
zero schema or invariant errors. Final train/validation manifest SHA-256 values are
`17124e03b59dc3b2fa6031af19e86875eeff75698e62ee2bc531dbc19f2621c7`
and `ab28c5d754133e140cbbcc5a3a8ceaffd083efddc1ebd143f70441786d6dd122`;
finalization summary SHA-256 is
`caed8aaceaabc9d96fd14093772d779b1eb0c8a53eee2717549efc78ad42459a`.

A real MPS optimizer/checkpoint smoke completed through the exact strict loaders:
1,475 Citizen train, 1,388 approved SemLex train, 13,381 local train, 378 Citizen
validation, and 2,896 local validation. Provenance confirms the 0.34/0.33/0.33
class/source-balanced margins, continuous full-circle roll augmentation, current
part-wise architecture, Citizen-only checkpoint selection, and false test-access
flags. Smoke result/provenance SHA-256 values are
`72069114257b62ac359cb268d1a9d052dd82f76b1ee50aa39558310513035a54`
and `c08092f764fcd1b2768c1ee72b9c6c43717bf331b080437f0d47bc3ee248ae80`.
The full affected suite passes 73/73, including six preparation/finalization tests,
50 Stage-1 tests, and 17 extractor tests. `git diff --check` passes.

The fail-closed Kaggle package contains exactly 16,280 regular files: 13,381 train
features, 2,896 validation features, two final manifests, and the finalization
summary. It contains no local test member and no Citizen/SemLex test artifact. The
153 MiB archive SHA-256 is
`7efc93abbfec3f7c8885ecfbe17394db1a26fc7fbf9226b72dbf390154b5c6ca`;
its independently re-extracted tree SHA-256 is
`254019cd2cfa8c9e1e2461aef66708ece0b88b9574f2f915ffa6a3332f0bf56d`.
Private Kaggle datasets `kokoab/slt-v17-local-deep-clean-trainval-v1` and
`kokoab/slt-v17-stage1-local-deep-clean-code-v1` were uploaded. The code dataset is
indexed; the larger 16,280-file feature dataset is still server-side indexing, so the
CUDA kernel is correctly held rather than launched against a partial mount. No test
split was accessed.

## 2026-08-13 21:57 PST — 94-class local deep-clean v17 challenger prepared

The project owner approved a separate model trained with the historical local clean
data together with Citizen train and SemLex train, and explicitly accepted local
signer overlap between train/validation/test. Old tensors and old training code remain
excluded: the historical v16 manifest is used only as a row whitelist back to raw
videos, which are being re-extracted with the current orientation-safe Apple Vision
v17 extractor and will be trained by the current `active/v17` part-wise/orientation
trainer. The selected Core ML checkpoint remains untouched as the control.

The exact clean-data lineage is now resolved. The raw v16 ledger has 66,770 clips, the
first confidence-cleaned ledger has 64,767, and the final deep-cleaned ledger has
62,023 clips across 310 classes. Final deep-clean manifest SHA-256 is
`5cf1e20a48ba2188cf93abf7564676fa5562b9ef2f20051adf6c0a091b9e0f70`.
The owner's remembered smaller snapshot is also real: the older float16 manifest has
57,557 entries and SHA-256
`3bb9072364b124396b041f6b038aa5729f3d07cd8f7c1fa38e94bc145cb5cb02`.
The final 62,023-entry deep-clean list is authoritative for this new experiment.

Ninety current labels overlap the deep-clean ledger exactly. Four more are recoverable
without mixing the two sides of old merged folders: `EAT` from only the `EAT` side of
`EAT_FOOD`, `MAKE` from only the `MAKE` side of `MAKE_CREATE`, `SAME` from the old
`ALSO_SAME` folder stored as `ALSO`, and `HOME` from `HOUSE`. Six current classes have
no traceable local raw class and are not filled with semantic neighbors: `FIND`,
`HAVE`, `HUNGRY`, `LISTEN`, `SICK`, and `TALK`. Citizen and SemLex continue to supply
all 100 classes.

`scripts/prepare_local_deep_clean_v17.py` resolves all relevant historical rows back
to exact raw movies and SHA-256-hashes both raw bytes and historical features. Rows
connected by either identical raw bytes or identical historical features are assigned
to one deterministic content-group split; six cross-label duplicate rows are
quarantined. The resulting 19,111 rows cover 94 classes: 13,382 train, 2,896 local
validation, and 2,833 currently unused local test. Every split covers all 94 local
classes. Signer overlap is explicit and approved; exact duplicate-content leakage is
not. There are 16,098 canonical/raw-text-equal rows and 3,013 traceable but
variant-unverified rows. Preparation summary SHA-256 is
`d9ca1aa84898e8241942f348442b9336ff229cb2786bc945d9502979afe5cad4`;
train/validation manifest hashes are
`b19ff51b4f87ccac1a2f2ea1d191a65016b3d64b3db979e85d50d737aeed7f2f`
and `5e98ab2017ded4eccd4e883c60491016e2010fa8db5629fe44393fb09832ed02`.

The updated v17 trainer now accepts the explicit non-signer-disjoint local validation
manifest and reports its loss/top-1/top-5/macro-F1 every epoch. Checkpoint selection
remains Citizen official validation top-1; local validation is a secondary familiar-
signer-domain diagnostic, so its much larger clip count cannot mask Citizen
regressions. Both local train and validation manifests prove false Citizen/SemLex test
access. Four new preparation tests and the focused local-validation loader test pass,
and affected files compile. Current source hashes are preparation script
`d57095a1e113e51dc618a42d474e9b236d47fb88de6d404f9f7cfbf182666c54`,
trainer `4c50a8ff402b46d77d4181a960113db959a90b4202cffeaaa5b9899be9565a87`,
new tests `4bc1c01b3c906d539a94170b0d352b93976152abeb52ffbe54b0f51e134aab95`,
and updated Stage-1 tests
`9cdd760cbc086452c9c51fd5a0131c2dfeb457bd7611e6adbd82c6045883f276`.
Train/validation Apple Vision v17 extraction is active; local test remains unextracted
and unused.

## 2026-08-13 21:45 PST — local 77-class external diagnostic exposes generalization gap

At the project owner's direction, the next model diagnostic used the existing local
`data/raw_videos/ASL VIDEOS` collection instead of waiting for new phone capture. The
raw directory currently contains 66,875 files in 316 class directories and occupies
about 32 GiB. It is a mixed historical corpus: filenames identify local hash-named and
numbered sessions as well as known scraped sources, but do not contain trustworthy
per-file signer identities. The owner's estimate of about seven new recurring signers
is consistent with the historical description but cannot be mechanically verified or
used to claim a signer-disjoint split.

The evaluation therefore used the already frozen, model-independent quality shortlist
at `data/local/local_citizen100_quality_audit_q82_cap14_exact/`. Its manifest SHA-256
is `45351f760dc8c7e1d064f04296e6055676cf50a900609a2ed3cef3693a9b1a14` and contains
1,021 unique clips across the 77 current classes whose directory text exactly equals
the pinned Citizen raw gloss. Known MS-ASL/WLASL/SignASL filenames, the mixed `I`
folder, non-exact variants, weak decodes, and near-duplicate sessions remain excluded.
All 1,021 Apple Vision v17 archives again pass the integrity audit with zero errors;
prior raw-hash decontamination found zero byte-identical overlap with all Citizen rows
and all retained SemLex-train clips. The other 23 current classes are not silently
filled with aliases or different numeric variants.

The untouched selected orientation checkpoint
`a7490409b3dfd76ba1ff432d2392b5e27df33f12e1b088cd36609fe03c082366`
scores 690/1,021 = 67.58% top-1 and 903/1,021 = 88.44% top-5, with 53.80%
100-class macro F1. Fifty-eight classes meet the existing model-consistency screen,
17 are ambiguous, and two are high-risk. The sharpest mismatch priorities are `COME`
(0/14 top-1 and 0/14 top-5) and `SIGN` (1/14 top-1 and 3/14 top-5). Exact evidence is
under
`artifacts/reports/local_citizen100_quality_audit/orientation_augmentation_only_v1_external_diagnostic/`;
the summary SHA-256 is
`d736d8baee12b87dfb4b37780acbec98eef828dea1aea1d07116bd7bc73a6459`
and the immutable logit ledger SHA-256 is
`33dcdca9d4386b1640913b6144e7d5e3ae1bc18775ec7facd1b33802acc92f33`.

The previously selected Citizen-validation 75/25 landmark/hand-RGB fusion was then
applied unchanged, without choosing weights on this local pool. It improves the
orientation model to 738/1,021 = 72.28% top-1 and 932/1,021 = 91.28% top-5,
rescuing 51 orientation-model errors while regressing only three correct predictions.
The result SHA-256 is
`60eb2a700eed39ea77ace7433d3099c8b01fc0f117ea0e3acbf0f85626e81caa`.
This independently confirms that RGB covers meaningful landmark weaknesses, but the
remaining 280 dual failures and variant uncertainty make blind distillation or blind
use of all 60k files premature.

Decision: this local collection is now the preferred source for the next train-only
v17 supplement, restricted to the current vocabulary and exact pinned variants. It
does not replace a future signer-disjoint benchmark unless anonymous signer/session
membership is first attached to every admitted row and held-out identities are frozen
before training. The 67.58%/72.28% figures are external local-label diagnostics, not
production accuracy. No Citizen or SemLex test split was accessed, and the selected
checkpoint and Core ML package remain unchanged.

## 2026-08-13 17:52 PST — 175 exact variants admitted; controlled retrain running

The official ASLLVD/ASL-LEX acquisition is complete. All 175 selected movies across
52 frozen classes and six named consultants downloaded and passed full video decoding.
Apple Vision v17 extraction produced 175/175 archives with zero no-hand or processing
failures; all 175 pass schema, finite-value, binary-presence, and missing-zero
invariants. The finalized manifest SHA-256 is
`9ebf73276a8e5e12ea33cdbc640169011ebd15170b2bb52e62d34fe6a69a9a8c`.
The audit is `artifacts/reports/ASLLVD_ASLLEX_V17_EXTRACTOR_AUDIT.md` with SHA-256
`776562fc35cad9c7b24002e56b806ea884f593144453996d6d11c8e4db841db1`.
No Citizen or SemLex test data was accessed.

Only the derived v17 landmarks and exact provenance were uploaded to the private
Kaggle dataset `kokoab/slt-v17-asllvd-asllex-features-v1`; raw ASLLVD movies were not
uploaded. A local two-batch smoke proves the strict loader admits exactly 175 samples,
52 classes, and the six known signer identities, and that requested sampler margins
converge to Citizen 0.45 / SemLex 0.45 / ASLLVD 0.10. Private T4 kernel
`kokoab/slt-v17-stage1-orientation-asllvd-v1` is RUNNING with the same seed,
architecture, optimizer protocol, and continuous full-circle augmentation as the
orientation winner. Its only training-data treatment is the new exact-variant source.
Before its result is known, replacement is constrained as follows: it must improve the
unweighted mean of Citizen and SemLex validation accuracy, lose no more than one
percentage point on either clean domain, retain at least the current 348/378 worst
nonzero landmark-roll result, and not reduce the corrected raw-pixel eight-angle mean
or extraction floor. A challenger that fails the clean gates is rejected without
using raw-pixel errors to rescue or tune it.

The first axis-prioritized automatic orientation selector is rejected. On the fixed
100-clip Citizen validation raw-pixel slice it scored only 80/100 upright because
tiny shoulder-axis differences sometimes chose 180 degrees over 0, despite improving
some intermediate-angle decisions. The corrected rule gates by body confidence,
uses an axis tolerance band to distinguish adjacent quadrants, then uses signed face
anatomy to resolve 0-versus-180. Sixteen extractor tests pass, including the real
0/17/37/73/90/123/180/270 correction sequence. A new full 800-condition raw-pixel
run is active; the rejected selector is not bundled for final phone testing.

## 2026-08-12 05:31 PST — one declared attention-score component staged

The higher-risk HTMA-inspired function is implemented without the paper's full CNN
or ambiguous four-convolution variant. Each of the eight part/global Squeezeformer
attention blocks optionally applies one independent depthwise 3x3 convolution to
each head's `T x T` pre-softmax score map. The score residual is zero-initialized,
and CPU RNG is restored after constructing the extra convolution so later baseline
weights retain their initialization sequence. The component starts numerically equal
to ordinary attention and adds only 576 parameters (6,792,293 total).

All 43 focused Stage-1 tests pass, including zero-score equivalence to the underlying
ordinary MHA and the exact 180-parameter delta for the small five-block test model.
A real two-batch/full-validation CPU smoke passed; all eight score mixers moved away
from zero and provenance kept test flags false. Source hashes are
`624e5d611dbf1bb25ee66c4b1af3b7dcccbd860dc675d7ba077c91f03dd816c4` and
`37121efa7ef7da7719daad2f45cf4da4e29f9993e9ea923d1ba472aa09dfa763`.
The staged private overlay archive under
`artifacts/generated/kaggle_stage1_attention_score_mix_overlay_v1/` has SHA-256
`c96ca041c4c55bb13957f572279219817244af4e57db75d0443cdfda511f6631`
and declares no test data. The CUDA-only runner is staged under
`artifacts/generated/kaggle_stage1_attention_score_mix_kokoab_v1/`. Private code
dataset `kokoab/slt-v17-stage1-attention-score-mix-code-v1` is ready and private
kernel `kokoab/slt-v17-stage1-attention-score-mix-v1` version 1 is RUNNING on a T4.
No local heavy process is active.

## 2026-08-12 05:26 PST — masked-pose reconstruction converges but hurts recognition

Private Kaggle kernel `kokoab/slt-v17-stage1-masked-pose-v1` completed on a Tesla T4.
Pretraining ran all 40 epochs / 1,800 steps and reduced the composite reconstruction
loss to 0.05712. It loaded 249 encoder tensors covering 6,591,808 parameters; the
temporary 78,385-parameter decoder was discarded. Pretraining checkpoint SHA-256 is
`6498048e4728ac1af36f01ba84ce061f07c49e7fbaf4c6a45f03c9d57e9d6fbf`.
Provenance confirms no validation or test access during pretraining.

The unchanged 6,791,717-parameter part-wise classifier retained fine-tune epoch 50
and early-stopped at epoch 80. Checkpoint SHA-256 is
`3aef02eaf4bbc3221e07e5a8ef64a7d4fa3613ac4ed3c455978a5d2ab1760b85`.
It scores only 360/378 = 95.24% Citizen validation top-1, 376/378 = 99.47% top-5,
and 94.79% macro F1. SemLex validation is 819/978 = 83.74% top-1,
931/978 = 95.19% top-5, and 80.64% present-class macro F1. Relative to the matched
part-wise control, this loses six Citizen and thirty-four SemLex top-1 clips. The
reconstruction objective therefore converged but learned a materially worse
discriminative initialization. Reject it and do not combine or tune its mask ratio on
the current validation sets. Revisit masked pretraining only with a substantially
larger v17-compatible unlabeled pool or a separately justified pretext objective.

Exact outputs are under
`artifacts/generated/kaggle_stage1_masked_pose_kokoab_result_v1/`,
`artifacts/reports/stage1_v17_masked_pose_finetune_v1_validation/`, and
`artifacts/reports/semlex_citizen100_val_audit/masked_pose_finetune_v1/`. Neither test
split was accessed and no heavy process is active.

## 2026-08-12 04:45 PST — canonical-hand component ready as a matched pair

The next Handshape-GNN-inspired function is implemented off by default without
importing that paper's model or handshape labels. For each hand, it selects three
frames with at least 12 observed joints and pools the existing part-wise features.
The capacity control ranks frames only by mean landmark confidence; the treatment
uses the identical branch but also penalizes normalized inter-frame hand speed, with
a quality-only fallback when no reliable transition exists. Both add the same 33,537
parameters to the 6,791,717-parameter part-wise model (6,825,254 total).

The hand token enters through a zero-initialized scalar residual, so both challengers
are exactly equal to part-wise-only at initialization. This makes quality versus
low-motion the only treatment and prevents a new fusion projection from immediately
perturbing the supported backbone. All 39 focused Stage-1 tests pass, including exact
initial identity, selection behavior, absent-hand zeros, configuration isolation, and
the fixed mobile-size bound. Separate real-data two-batch/full-validation CPU smokes
passed for both modes; each residual scale moved away from zero and provenance kept
both test flags false.

The two source hashes are
`015a84413d4c591b197ccbf88f1bd937ff77f461245d520455d64c8503f6d46f`
and `4915025571521e27906c78d573f5d139de6a726947c63752fe3242d0fcb9b642`.
The staged private overlay archive under
`artifacts/generated/kaggle_stage1_static_hand_overlay_v1/` has SHA-256
`b9876e12fe19567ba577d8f99cb029ec67265aaa0f8464f99f210749ff2fb0af`
and declares no test data. The CUDA-only sequential quality/low-motion runner is under
`artifacts/generated/kaggle_stage1_static_hand_kokoab_v1/`. Private code dataset
`kokoab/slt-v17-stage1-static-hand-code-v1` is ready and private kernel
`kokoab/slt-v17-stage1-static-hand-v1` version 1 is QUEUED for a T4. No local heavy
process is active.

## 2026-08-12 04:21 PST — articulated-distance component staged with required control

The next paper-derived trial is implemented as a component rather than a wholesale
model replacement. A missing-aware articulated hand distance uses length-weighted
bone-orientation differences to mine per-frame triplets from only the approved
Citizen and SemLex training splits. A 168-D wrist-relative hand vector is mapped by
an 84,416-parameter MLP to a normalized 64-D pose embedding and fused into the
replicated part-wise Squeezeformer. The downstream model has 6,958,821 parameters.

The comparison is explicitly capacity-matched: first train the added branch from
random initialization, then train the identical architecture with only that branch
initialized by the distance-preserving triplet task. Thus random versus part-wise
measures extra capacity, while pretrained versus random measures the paper-derived
geometry objective. Neither model receives test data, label-derived pretraining, or
validation feedback during pretraining.

All 37 focused Stage-1 tests pass. They cover translation invariance, missing-hand
masking, articulated distance behavior, forward/configuration isolation, and strict
branch-only preload. A real-data pretraining smoke and separate random/pretrained
two-batch downstream smokes passed with the exact 1,475 Citizen + 1,388 SemLex
training loaders and full Citizen validation. The three source hashes are
`2c6ce33509f22a45b4c5485952e233cf369986f8bee9aace634a8d3bdcef31a6`,
`fe6bddcc02bd1a9201b4c574bc14e479fc8a4fcc8fbb86f12f3829dc4212def7`,
and `00d4fb065fe168354ef85391e46e5e8ecaa7c2ec71ac073d459c5d54da532949`.

The private Kaggle overlay is staged under
`artifacts/generated/kaggle_stage1_articulated_pose_overlay_v1/`; archive SHA-256 is
`44b078f47621987dfdb9a0319f30c030211c6ac463ef8d1787ccd9330440f4cc`.
Its manifest declares no test data. The CUDA-only runner under
`artifacts/generated/kaggle_stage1_articulated_pose_kokoab_v1/` will execute geometry
pretraining, the random capacity control, and the pretrained treatment sequentially
on one T4 while preserving the same seed, data, sampler, optimizer, schedule, and
patience. Private code dataset
`kokoab/slt-v17-stage1-articulated-pose-code-v1` is ready, and private kernel
`kokoab/slt-v17-stage1-articulated-pose-v1` version 1 is RUNNING. No local heavy
process is active. The exhaustive audit's predeclared ladder now records the rejected
temporal gate and the running articulated-distance/control comparison before any
static-hand, masked-pose, or visual-symbol experiment.

## 2026-08-12 04:03 PST — part-wise gain replicated; temporal gate launched

The controlled seed-3407 part-wise replication completed on a Tesla T4, retained
epoch 33, and early-stopped after 63 epochs. Its 6,791,717-parameter checkpoint
SHA-256 is `4dab34bec5fc72684f596f775acfec638312ac88a7cfd3dd29d55084a1515447`.
It scores 362/378 = 95.77% Citizen validation top-1, 375/378 = 99.21% top-5,
and 95.38% macro F1. The fixed SemLex validation diagnostic scores 845/978 =
86.40% top-1, 946/978 = 96.73% top-5, and 84.01% present-class macro F1.

Against the matched seed-3407 flat baseline (358 Citizen, 832 SemLex), part-wise adds
exactly four Citizen and thirteen SemLex clips. This repeats the seed-1701 gains
exactly: 366 versus 362 Citizen and 853 versus 840 SemLex. Part-wise temporal
isolation is therefore a replicated architecture effect and remains the supported
compact landmark backbone; the seed-1701 checkpoint remains the best single run.
Exact replication reports are under
`artifacts/reports/stage1_v17_partwise_seed3407_v2_validation/` and
`artifacts/reports/semlex_citizen100_val_audit/partwise_seed3407_v2/`.

A SKIM-inspired per-keypoint temporal gate is now implemented off by default. It uses
independent depthwise temporal filters over each node's confidence, speed, and
acceleration, and a zero-initialized output makes the initial gate exactly identity.
It adds 732 parameters (6,792,449 total with part-wise), preserves the public
`[32,61,5]` input, and is restricted to an isolated part-wise ablation. All 34 focused
tests pass, and a real two-batch CPU train/checkpoint/full-validation smoke passed on
the exact 1,475 Citizen + 1,388 SemLex training loaders.

Private code dataset `kokoab/slt-v17-stage1-temporal-gate-code-v1` contains only the
two hash-locked source overlays; archive SHA-256 is
`6e128810bf0651d4b6ade9b3c11c3aee17e5a0dc39b6a4163f899bcf3d6a1d02` and it declares
no test data. Private kernel `kokoab/slt-v17-stage1-temporal-gate-v1` version 1 is now
running on a Tesla T4 with seed 1701 and the otherwise unchanged controlled protocol.
No test data was accessed and no local heavy process is active.

The 66,409 legacy files under `data/local/ASL_landmarks_apple_vision/` were also
checked as a possible masked-pretraining pool. They are `[32,61,10]` outputs from the
older extractor: XYZ plus precomputed Savitzky-Golay velocity/acceleration and a
part-level `ever observed` mask. A real sample confirms channel 9 is that mask while
channels 3-8 are derivatives, not v17 presence/confidence. The older pipeline also
interpolates missing frames, uses different normalization, and predates the v17
chirality/schema corrections. These files cannot be sliced or relabeled into the
v17 `[32,61,5]` contract. They may support a separate legacy pretraining study only
after an explicit adapter ablation, or their raw videos may be selectively re-extracted
with v17; they will not be silently mixed into the current model.

A component-level literature pass added one higher-cost geometry candidate from
Sartinas et al. (VISAPP 2026): pretrain a small per-frame MLP to preserve neighborhoods
under a hierarchical articulated bone-orientation distance, then concatenate only its
64-D embedding with the temporal input. Their five-run ablation shows that a random
extra branch explains much of the headline gain, so any v17 trial must include a
capacity-matched random-branch control. This is not selected ahead of the running gate;
it is a later alternative to generic masked reconstruction because it targets the
positive bone signal while differing from the rejected raw angle channel.

A second component-only review covered Varanasi et al.'s CVPRW 2026 HTMA block. The
portable function is a small 2D convolution over each attention head's `T x T` score
map before softmax, not their full 1D-CNN/MediaPipe model. It remains higher-risk:
their attention ablation uses INCLUDE's original split rather than the described
pseudo-signer grouping, and the algorithm claims four score-mixing convolutions while
the hyperparameter table says one. Retain one explicitly declared score-convolution
variant as a later Squeezeformer-attention ablation; do not treat the paper's desktop
latency or nominal mobile claim as iPhone evidence.

## 2026-08-12 03:55 PST — same-seed part-wise replication launched

Private Kaggle kernel `kokoab/slt-v17-stage1-partwise-seed3407-v2` version 1 is now
running on a Tesla T4. The runner compiled locally before launch and fail-closes on
the exact training archive/tree hashes and any test artifact. Its only treatment
relative to the existing seed-3407 flat baseline is `partwise_global` temporal
encoding with part depth 1; data, balanced sampler, optimizer schedule, maximum 160
epochs, patience 30, and seed 3407 are otherwise fixed. The predeclared comparison is
358/378 Citizen validation and 832/978 SemLex validation. No test data was accessed
and no local heavy process is active.

## 2026-08-12 03:54 PST — hand angles rejected; part-wise replication staged

The isolated flat hand-angle run completed on a Tesla T4, retained epoch 86, and
early-stopped after epoch 116 / 30 stale epochs. Its 6,486,501-parameter checkpoint
SHA-256 is `3c8d5a82ea7df1f1015fab4ae3665b9c5ee1bf2e7c085c85bc24f98fedd5ee97`.
It scores 364/378 = 96.30% Citizen validation top-1, 375/378 = 99.21% top-5, and
95.70% macro F1. On SemLex validation it scores 838/978 = 85.69% top-1,
933/978 = 95.40% top-5, and 82.96% present-class macro F1. Relative to flat, angles
gain two Citizen top-1 clips but lose two SemLex top-1 and seven SemLex top-5 clips.
They fail the both-domain gate and will not be combined with part-wise. Exact results
are under `artifacts/generated/kaggle_stage1_hand_angle_kokoab_result_v1/` and
`artifacts/reports/semlex_citizen100_val_audit/hand_angle_v1/`.

The cheap supervised component ladder therefore leaves part-wise-only as the best
single development architecture. A seed-3407 replication kernel is staged under
`artifacts/generated/kaggle_stage1_partwise_seed3407_kokoab_v2/` with the exact same
data/protocol and only the seed changed. It will be compared to the already measured
same-seed flat baseline of 358/378 Citizen and 832/978 SemLex. No test data was
accessed.

## 2026-08-12 03:45 PST — second-seed flat cross-domain baseline established

The existing clean flat seed-3407 checkpoint (358/378 = 94.71% Citizen validation)
was evaluated once on the fixed SemLex validation diagnostic for the forthcoming
architecture replication. It scores 832/978 = 85.07% top-1, 934/978 = 95.50% top-5,
and 82.47% present-class macro F1. The seed-1701 flat reference is 362/378 Citizen and
840/978 SemLex. A seed-3407 part-wise run can therefore be compared against its own
same-seed flat baseline rather than only the selected seed-1701 winner. Exact output
is under `artifacts/reports/semlex_citizen100_val_audit/d256_full_clean_seed3407/`.
No test data was accessed; the hand-angle kernel remains RUNNING.

## 2026-08-12 03:44 PST — per-part supervision mixed; angle trial launched

The corrected per-part auxiliary run completed on a Tesla T4, retained epoch 65, and
early-stopped after epoch 95 / 30 stale epochs. Its 6,818,229-parameter training
checkpoint SHA-256 is
`c1778e5952c71dd0c54f2a6706fedc1ea6398686da9481b1e0373e0448f8a534`.
Provenance confirms the missing-part skip policy, fixed total auxiliary weight 0.20,
unchanged data/sampler/seed, and false test-access flags.

It scores 365/378 = 96.56% Citizen validation top-1, 377/378 = 99.74% top-5, and
96.32% macro F1. The fixed SemLex diagnostic scores 861/978 = 88.04% top-1,
943/978 = 96.42% top-5, and 85.85% present-class macro F1. Relative to part-wise-only,
this loses one Citizen top-1 clip but gains eight SemLex top-1, four SemLex top-5, and
1.41 macro-F1 points. It therefore improves cross-domain robustness but fails the
primary Citizen gate and does not replace the 366/378 part-wise winner. Exact results
are under `artifacts/generated/kaggle_stage1_partaux_w020_kokoab_result_v2/` and
`artifacts/reports/semlex_citizen100_val_audit/partaux_w020_v1/`.

The copied writable Kaggle tree caused the first unfiltered result pull to start
downloading training artifacts; a file-pattern-limited CLI pull retrieved only the
five intended result/log files. The cloud checkpoint was intact; the earlier local
zero-byte partial is not evidence. Future overlay runners should avoid leaving the
copied base tree under `/kaggle/working` at exit or always use filtered pulls.

Private kernel `kokoab/slt-v17-stage1-hand-angle-v1` is now RUNNING. It uses the
unchanged flat d=256/depth=4 Squeezeformer and makes the 30 missing-aware cosine
finger-flexion values its only treatment; part-wise, bone, PartMix, phonology,
contrastive, and auxiliary losses are disabled. No local heavy process or test data
access is active.

## 2026-08-12 03:39 PST — absent-part audit confirms auxiliary masking is required

A complete read of the exact 1,475 Citizen + 1,388 SemLex training features found
587/523 clips with no observed left-hand node, 118/88 with no observed right-hand
node, and 82/159 with no observed body node; every clip has at least one face anchor.
Thus about 39% of training clips lack at least one hand stream, usually because the
sign is genuinely one-handed or that hand was not detected. Training every auxiliary
head on every sample would create a large impossible-label objective from all-zero
inputs. The running v2 kernel correctly skips only those absent part/sample pairs
while retaining global classification loss for every clip. No labels, files, or test
data were changed.

## 2026-08-12 03:35 PST — isolated hand-angle trial staged

Private 22.4 KiB code dataset `kokoab/slt-v17-stage1-hand-angle-code-v1` is ready;
its archive SHA-256 is
`79bf8f102835dde168485538061c3f6437965a127cf8505a2fa706b2cdcfa004`.
It pins only the tested model/trainer sources and the exact angle definition, with no
training data or test artifacts. A fail-closed global-Squeezeformer kernel is staged
under `artifacts/generated/kaggle_stage1_hand_angle_kokoab_v1/`; it will reuse and
reverify the frozen base tree, apply the two-file overlay in the writable ephemeral
volume, and run angles as the only treatment. It has not launched while the corrected
per-part auxiliary kernel is RUNNING. No local heavy process or test data access is
active.

## 2026-08-12 03:32 PST — auxiliary v2 running; hand-angle component ready

Per-part kernel version 1 passed CUDA, the complete base-tree digest, and both overlay
source hashes, then failed before training because the verified base tree was still on
Kaggle's read-only input mount. It produced no checkpoint or metric. Version 2 now
copies that already verified 41 MiB compact tree to the ephemeral working volume,
rechecks the full tree digest there, applies and rechecks only the two pinned overlay
files, and is RUNNING. This is a transport fix only; the experiment and data are
unchanged.

The next paper-derived component is implemented off by default as
`--hand-angle-features`. It derives cosine flexion angles at the three internal joints
of each of five fingers on both hands (30 angles total), zeroing a value unless its
parent, center, and child are all observed. Cosines avoid `acos` instability and add a
scale/translation/rotation-invariant handshape cue without changing the public Apple
`[32,61,5]` archive. The flat model grows by only 15,616 parameters, from 6,470,885 to
6,486,501; the part-wise model grows by 3,904, from 6,791,717 to 6,795,621. All 32
focused tests pass and a real two-batch Citizen+SemLex optimizer/checkpoint smoke
passed. This component is motivated by the Handshape-GNN joint-angle analysis and
SignRep's angle/keypoint/distance prior ablation, not by importing either full model.
It has no full accuracy result and must wait for the running auxiliary trial. No test
data was accessed.

## 2026-08-12 03:31 PST — untuned landmark pair confirms complementary errors

Aligned Citizen logits for the part-wise+bone model reproduce its Kaggle metrics
exactly. A single predeclared, untuned 50/50 per-sample-z-scored ensemble of
part-wise-only and part-wise+bone scores 367/378 = 97.09% Citizen validation top-1,
376/378 = 99.47% top-5, and 96.79% macro F1. On SemLex validation it scores
868/978 = 88.75% top-1, 949/978 = 97.03% top-5, and 84.27% 100-class macro F1.
Relative to part-wise-only this is +1 Citizen and +15 SemLex top-1 clips, proving the
bone model has complementary errors despite its weaker Citizen standalone result.

The pair is an accuracy-oriented research option, not the mobile default: it runs two
6.8M-class landmark networks and still trails the existing four-stream teacher at
370/378 Citizen and 882/978 SemLex. No weights were searched, and no additional
combinations should be mined on these validation sets. Exact reports are under
`artifacts/reports/stage1_v17_partwise_bone_equal_ensemble_validation/` and
`artifacts/reports/semlex_citizen100_val_audit/partwise_bone_equal_ensemble/`. No test
data was accessed; the missing-aware per-part kernel remains RUNNING.

## 2026-08-12 03:29 PST — missing-aware per-part supervision launched

Private kernel `kokoab/slt-v17-stage1-part-auxiliary-w020-v1` is RUNNING. Kaggle
canonicalized the title-derived slug by spelling out `part-auxiliary`; local metadata
now matches the server identity. The runner verifies the unchanged base training tree,
then the ready v2 overlay and both corrected source hashes, and runs the 6,818,229-
parameter part-wise architecture with a fixed total auxiliary weight of 0.20. Each of
the left-hand, right-hand, face, and body heads receives gloss supervision only for
clips where that part has at least one observed node. Bone, PartMix, phonology, and
contrastive objectives are disabled. The auxiliary heads are training-only and normal
inference uses the same 6,791,717-parameter path as part-wise-only. No local heavy
process or test data access is active.

## 2026-08-12 03:28 PST — bone interaction improves SemLex but fails primary gate

The part-wise+bone Kaggle run completed on a Tesla T4, retained epoch 71, and
early-stopped after epoch 101 / 30 stale epochs. Its 6,815,141-parameter checkpoint
SHA-256 is `b5a1a808f3de9cd74e081e816abf1d737cbe448f46d12ea11a4f039e7624f7bf`.
All data/objective/test-isolation provenance matches the declared interaction.

It scores 364/378 = 96.30% Citizen validation top-1, 376/378 = 99.47% top-5, and
96.01% macro F1. On SemLex validation it scores 859/978 = 87.83% top-1,
949/978 = 97.03% top-5, and 85.18% present-class macro F1. Relative to part-wise-only,
bone gains six SemLex top-1 clips and ten SemLex top-5 clips but loses two Citizen
top-1 clips and one Citizen top-5 clip. It therefore fails the predeclared rule that
the interaction must exceed part-wise on the primary Citizen gate and does not replace
the 366/378 part-wise winner. The tradeoff is still useful evidence that bone features
improve cross-domain robustness and may be valuable after independent calibration;
do not choose it solely from SemLex.

Exact artifacts are under
`artifacts/generated/kaggle_stage1_partwise_bone_kokoab_pull_v1/`; the diagnostic is
under `artifacts/reports/semlex_citizen100_val_audit/partwise_bone_v2/`. The corrected
per-part auxiliary objective now skips a part/sample loss when that anatomical stream
has no observed node, preventing legitimate one-handed clips from being punished for
an absent nondominant hand. Normal inference logits remain bit-identical, all 30 tests
pass, and a new real two-batch smoke passed. The ready v2 code overlay SHA-256 is
`2da438ffc49bdf47843802454ef4c25a392f93cda89c85e3bc70d9bfc997e49e`;
the obsolete v1 overlay must not be launched. No test data was accessed.

## 2026-08-12 03:20 PST — two more paper components triaged, not blindly adopted

The literature audit now includes a 2025 arXiv dual-reference model and a 2024
LREC-COLING keypoint-importance study. The portable dual-reference idea is to represent
hand morphology relative to each wrist while retaining wrist trajectory relative to
the body/face. v17 already retains body-relative trajectory and exposes
translation-invariant pairwise hand distances; the positive bone experiment covers
much of the missing local-geometry hypothesis. A direct wrist-relative-coordinate
stream is therefore a later cheap ablation if the current part-wise/bone ladder
plateaus, not a reason to import the paper's 46.3M graph/LSTM/optimal-transport model.

The feature-importance study reports that outer-finger bases/tips dominate while
inner finger joints and coarse face points are often under-used, with occlusion,
missing depth, data imbalance, and insufficient facial detail as plausible causes. Its
own warning is important: low model importance does not imply low linguistic value.
The project will use this only for diagnostics and quality-aware training hypotheses;
it will not delete joints or add dense face landmarks without a measured extractor
bake-off. These decisions and primary links are recorded in the exhaustive audit. The
part-wise+bone Kaggle kernel remains RUNNING; no test data was accessed.

## 2026-08-12 03:18 PST — part-wise plus bone interaction launched

Private kernel `kokoab/slt-v17-stage1-partwise-bone-v2` version 1 is RUNNING on the
same fail-closed CUDA/data-integrity path. The model has 6,815,141 parameters and
combines exactly the two independently positive components: one isolated temporal
layer per left hand/right hand/face/body before the depth-4 global Squeezeformer, plus
internally derived bone vectors/bone motion. Data, seed, sampler, optimizer, schedule,
patience, and classification loss are unchanged; PartMix, phonology, contrastive, and
per-part auxiliary objectives are disabled. This run must beat the part-wise-only
366/378 Citizen and 853/978 SemLex top-1 results to replace it. No heavy local process
or test data access is active.

## 2026-08-12 03:17 PST — bone representation passes both isolated top-1 gates

The isolated bone-only Kaggle run completed on a Tesla T4, retained epoch 50, and
early-stopped after epoch 80 / 30 stale epochs. Its 6,564,581-parameter checkpoint
SHA-256 is `e253b00dd23670960e43a39accc090bf61da1a6a403fe3f47570f4ce58deb4ff`.
Provenance confirms the exact unchanged data/sampler/seed/global architecture, with
bone vectors and bone motion as the only treatment and false test-access flags.

It scores 363/378 = 96.03% Citizen validation top-1, 377/378 = 99.74% top-5,
and 95.56% macro F1. The fixed SemLex diagnostic scores 845/978 = 86.40% top-1,
943/978 = 96.42% top-5, and 84.02% present-class macro F1. Relative to the flat
baseline this is +1 Citizen top-1 clip, +5 SemLex top-1 clips, and +3 SemLex top-5
clips. Bone representation therefore supplies genuine but smaller independent signal;
it does not replace the 366/378 Citizen, 853/978 SemLex part-wise winner.

Because part isolation and bone representation each improved both primary top-1
domains in separate runs, a part-wise+bone interaction is now scientifically
interpretable. Run that combination without PartMix or auxiliary losses before the
already staged per-part-supervision trial. Exact bone artifacts are under
`artifacts/generated/kaggle_stage1_bone_kokoab_pull_v1/`; its SemLex report is under
`artifacts/reports/semlex_citizen100_val_audit/bone_v2/`. No test data was accessed.

## 2026-08-12 03:18 PST — per-part follow-up transport staged without data reupload

The 21.6 KiB private Kaggle dataset `kokoab/slt-v17-stage1-partaux-code-v1` is ready.
It contains only the two updated v17 source files and a manifest—no landmarks, raw
video, validation predictions, or test artifacts. The archive SHA-256 is
`3839501a323b5dd2ca48a0198c317fd0b34bd7c6ab55374066dd41869a91c403`;
the manifest pins `model_v17.py` to
`045564493d189e298b2dda5571a05f2109b19d3cd5afcc0e7e57939d43735aec`
and `train_stage_1_v17.py` to
`df8e259296871c2814e2fe3424f0d371e688ea6f040148a9cb01fb6aa8454894`.

A CUDA-only kernel is staged as
`artifacts/generated/kaggle_stage1_partaux_w020_kokoab_v1/`. Its runner first verifies
the unchanged 3,253-file base training tree, then locates exactly one complete overlay,
checks both source hashes and the no-test manifest, copies only those sources into the
ephemeral Kaggle working tree, and runs part-wise + global training with fixed
per-part auxiliary weight 0.20. It is not launched while the isolated bone kernel is
RUNNING. This avoids uploading the 34 MiB training corpus again and keeps the next
treatment auditable. No local heavy process or test data access occurred.

## 2026-08-12 03:15 PST — fixed multimodal substitution does not inherit the gain

The new part-wise checkpoint's Citizen validation metrics were reproduced locally
exactly and aligned logits were saved under
`artifacts/reports/stage1_v17_partwise_v2_validation/`. Substituting those logits for
the flat landmark member while keeping the previously frozen 0.30 landmark / 0.15
mouth / 0.35 lower-face / 0.20 hand weights produces 368/378 = 97.35% Citizen
validation and 879/978 = 89.88% SemLex validation. The existing fixed flat-landmark
teacher remains better at 370/378 = 97.88% and 882/978 = 90.18% respectively.

This is not a contradiction: the part-wise model is a better standalone classifier,
but its errors and score calibration are less complementary to weights selected with
the old flat member. The current winners therefore remain separate: part-wise for the
single landmark branch, and the old four-stream composition for the multimodal
research teacher. Do not retune weights on SemLex, and do not repeatedly optimize
Citizen weights to manufacture a higher validation number. An independent portrait
set is required before choosing or recalibrating a production ensemble. Exact fixed-
weight reports are under
`artifacts/reports/stage1_v17_partwise_multimodal_teacher_fixed_validation/` and
`artifacts/reports/semlex_citizen100_val_audit/fixed_partwise_teacher_30_15_35_20/`.
No test data was accessed.

## 2026-08-12 03:13 PST — isolated bone-feature challenger launched

Private CUDA kernel `kokoab/slt-v17-stage1-bone-v2` version 1 is RUNNING. It uses the
same verified 3,253-file Citizen/SemLex train/validation tree, seed, sampling,
optimizer, schedule, patience, and unchanged global d=256/depth=4 Squeezeformer as
the superseded flat baseline. Its only treatment is internally derived missing-aware
hand/arm bone vectors plus bone motion (`--bone-features`); PartMix, part-wise temporal
encoding, phonology, contrastive learning, and per-part auxiliary supervision are all
disabled. The runner requires CUDA and rejects any link/path/test artifact before
training. This isolated run determines whether bone representation itself helps; it
does not yet test bone combined with the new part-wise winner. No local heavy process
or test data access is active.

## 2026-08-12 03:12 PST — part-wise encoder becomes the development winner

The isolated part-wise Kaggle trial completed successfully on a Tesla T4. It retained
epoch 100 and early-stopped after epoch 130 / 30 stale epochs. The checkpoint SHA-256
is `5c40b13336b4692d5f7e1e70a9ba430aa2b35ef4e12946952e15fe1f9e54924b`.
Provenance confirms the unchanged 1,475 Citizen + 1,388 SemLex training set,
50/50 class/source-balanced replacement sampling, seed 1701, no PartMix, no auxiliary
loss, 6,791,717 parameters, CUDA, and false Citizen/SemLex test-access flags.

The checkpoint scores 366/378 = **96.83%** Citizen validation top-1, 375/378 =
99.21% top-5, and 96.61% macro F1. The former flat winner scored 362/378 = 95.77%,
378/378 = 100% top-5, and 95.51% macro F1. The fixed SemLex-validation diagnostic
scores 853/978 = **87.22%** top-1, 939/978 = 96.01% top-5, and 84.44% present-class
macro F1, versus the flat winner's 840/978 = 85.89%, 940/978 = 96.11%, and 82.60%.
Thus feature-isolated left-hand/right-hand/face/body temporal modeling adds four
Citizen top-1 clips and thirteen SemLex top-1 clips, with three Citizen and one SemLex
additional top-5 misses. This satisfies the predeclared primary/cross-domain top-1
gate and is the new **development landmark winner**. It has not been evaluated on
either test split and is not yet an independently confirmed production model.

Exact pulled artifacts are under
`artifacts/generated/kaggle_stage1_partwise_kokoab_pull_v1/`; the SemLex report is
under `artifacts/reports/semlex_citizen100_val_audit/partwise_v2/`. The result supports
the diagnosis that early whole-body flattening, not Squeezeformer's global temporal
stack itself, was a real bottleneck.

The literature-derived per-part auxiliary path is also implemented off by default.
When enabled, four training-only gloss heads supervise the isolated left hand, right
hand, face, and body streams before global fusion. Ordinary inference remains the
same fused classifier; the auxiliary heads are unused and can be pruned at export.
The full d=256 training graph adds only 26,512 parameters (6,818,229 total versus
6,791,717), all 30 focused Stage-1 tests pass, and a real two-batch Citizen + SemLex
CPU optimizer/checkpoint smoke passed at fixed auxiliary weight 0.20. This has not had
a full accuracy run. The bone-only challenger remains next so each component's effect
is established before combination. No heavy local process or test data access
occurred.

## 2026-08-12 03:08 PST — component-level literature ladder refined

The user's instruction is to mine useful functions from papers rather than treating
published models as all-or-nothing replacements. The controlled policy is now
explicit: transplant the smallest v17-compatible mechanism, test it in isolation on
the same Citizen-development/SemLex-diagnostic protocol, and retain it only if the
measured cross-domain tradeoff is favorable. No benchmark headline is treated as a
project result.

Two additional primary studies expose relevant components. A 2025 signer-independent
pose study found region decomposition alone gave only a modest gain, while training-
only gloss decoders attached to each hand/lip/torso stream produced the major ablation
gain. This motivates four tiny **per-part auxiliary gloss heads** on the existing
feature-isolated v17 encoder; their logits need not be computed or shipped at mobile
inference. A separate EMNLP 2025 Handshape-GNN study combines full hand dynamics with
a low-motion representative static frame. Its 37-handshape PopSign result and
handshape labels are not directly transferable, but a missing-aware static canonical-
hand token is a legitimate later component. It differs from the rejected global
ASL-LEX phonology objective because it changes the hand-specific representation, not
merely the label penalty on the final pooled token.

The exact updated order is: finish the running isolated part-wise trial; run the
already implemented isolated bone trial; then test per-part supervision; only then
consider a static-hand branch, gated joint/bone interaction, visual-symbol grouping,
or masked-pose pretraining. A CUDA-only bone kernel is staged locally as
`artifacts/generated/kaggle_stage1_bone_kokoab_v2/` against the same verified private
3,253-file train/validation tree, but it has not been pushed while the part-wise GPU
job is active. The part-wise kernel remains RUNNING. No heavy local process or test
data access occurred. The exhaustive audit records the component mapping and primary
paper links.

## 2026-08-12 02:27 PST — sign-specific PartMix challenger launched on Kaggle

The current d=256/depth=4 Squeezeformer remains the best model measured in this
project, not a proof of global architectural optimality. A primary-literature audit
identified three materially distinct skeleton-SLR directions that the rejected v17
graph experiment did not test: SKIM's corresponding-part mixing, P3D/Siformer's
part-wise temporal encoding, and DSTA-SLR's joint/bone/motion streams. The old graph
challenger only applied a per-frame anatomical graph before global temporal modeling.
Siformer's official MIT repository was cloned read-only at commit
`979a14ed15ed0f20afd77d447ad23c4f4107a2c3`; it uses separate left-hand,
right-hand, and body temporal encoders before fusion. DSTA-SLR's official repository
was cloned read-only at commit `e7e5ee225511488039a4f68ab3a146cf8f01312d`;
its released implementation assumes 27 joints/120 frames/CUDA and ensembles four
independently trained joint, bone, joint-motion, and bone-motion streams, so neither
its code nor checkpoint can be silently attached to the fixed Apple v17 contract.
No external weights or datasets were downloaded and no test data was accessed.

The first bounded challenger is SKIM-style one-hand PartMix because it changes only
training and adds zero inference parameters, latency, or mobile preprocessing.
`train_stage_1_v17.py` now optionally replaces exactly one complete 21-node left or
right hand with a guaranteed non-self batch donor and trains that sample with fixed
50/50 primary/donor labels. The default probability is zero, preserving the prior
training path. PartMix is deliberately rejected when contrastive or phonology losses
are enabled so this remains an isolated comparison. Provenance and per-epoch realized
mix fraction are saved. Two focused unit tests cover exact whole-hand replacement,
non-self labels, untouched face/body, missing-value preservation, the zero-probability
identity, and mixed cross-entropy. All 24 Stage-1 tests pass, the real 1,475 Citizen +
1,388 SemLex loader passed a two-batch CPU optimizer/checkpoint smoke at probability
0.5, affected scripts compile, and scoped `git diff --check` passes.

The authenticated Kaggle CLI uploaded a private 27.5 MiB archive as dataset
`francisbatiancela/slt-v17-stage1-partmix-trainval-v1`. Its SHA-256 is
`2c257e0bbc8fc5e198b445edbc472f7b44e9959e3ec60f8d08baf21bb16e9321`;
the archive contains the exact training code, 1,476 Citizen-train archives (1,475
usable after the frozen rejection ledger), 378 Citizen-validation archives, and all
1,388 approved SemLex-train archives. A full member audit found no `/test/` entry.
Private kernel version 1 at
`francisbatiancela/slt-v17-stage-1-partmix-p50` was launched. It verifies the archive
hash, refuses links/path traversal/test members, requires a real CUDA allocation, and
then runs the unchanged d=256/depth=4, seed-1701, 50/50 class/source-balanced protocol
with PartMix probability 0.5 and patience 30. If Kaggle again assigns CPU, it will
fail immediately rather than perform heavy local or mislabeled training.

Kaggle assigned that kernel its CPU-only PyTorch 2.10.0 image despite preserving
`enable_gpu:true` and `machine_shape:NvidiaTeslaT4` in the server-side metadata. The
runner therefore failed closed after 4.8 seconds with zero training batches. The CLI
is authenticated as `francisbatiancela`, whereas the user-supplied private notebook
`kokoab/notebook697ebe7d11` returns `403 kernels.get`; these are distinct Kaggle
accounts/sessions. The official forced OAuth refresh has been opened and is waiting
for browser approval so the CLI can authenticate to the already-open `kokoab`
session. Do not claim a Kaggle GPU run until the refreshed identity and CUDA device
are explicitly verified.

## 2026-08-12 02:34 PST — feature-isolated temporal challenger implemented

OAuth remains unapproved after five minutes and the CLI identity is still
`francisbatiancela`, so no second Kaggle launch was attempted. Work continued on the
next independent architecture rather than heating the laptop with a full run.

The Siformer/P3D evidence is now represented by a controlled v17-native challenger,
not by copying an incompatible upstream model. `Stage1V17Config` and the Stage-1 CLI
accept `temporal_encoder=partwise_global` and `part_depth`. The challenger projects
left hand, right hand, face, and body independently, runs a separate temporal
Squeezeformer on each stream, fuses them, and only then applies the unchanged global
Squeezeformer. Pairwise hand distances are routed only to their matching hand stream.
This directly tests feature-isolated part-wise temporal context; unlike the rejected
graph model, it cannot mix anatomy before each part has been modeled across time.
The default remains the original global architecture.

At d=256/depth=4/part-depth=1 the challenger has 6,791,717 parameters versus
6,470,885 for the winner, a 320,832 / 4.96% increase rather than the d=384 model's
2.22x expansion. A hook-level isolation test proves that changing the right-hand
nodes cannot change left-hand, face, or body pre-fusion projections. A full finite
forward test enforces fewer than eight million parameters. All 26 focused Stage-1
tests pass, affected scripts compile, scoped `git diff --check` passes, and the real
Citizen + full-clean SemLex loader completed a two-batch optimizer/checkpoint smoke.
No test data was accessed. This challenger is ready to package after the PartMix run;
it must not be combined with PartMix in its first comparison.

## 2026-08-12 02:38 PST — explicit bone/motion challenger implemented

The third predeclared sign-specific challenger is now available off by default as
`--bone-features`. It does not alter the public Apple `[32,61,5]` input or extractor.
The model internally derives directed vectors for the 42 hand/arm chains and three
body links, plus their masked temporal differences. A bone is zero whenever either
endpoint is absent; bone motion is zero when either adjacent bone is invalid. Sparse
face anchors are deliberately excluded because they are not a physical face mesh.
The feature is accepted by the flat/global and feature-isolated part-wise paths and
rejected for the graph-replacement path where it would otherwise be silently unused.

The unchanged global d=256 model grows from 6,470,885 to 6,564,581 parameters
(+93,696 / 1.45%); part-wise + bone is 6,815,141. Unit coverage verifies exact bone
direction, missing masking, temporal masking, face exclusion, and the unchanged
public input. All 28 focused Stage-1 tests pass, affected scripts compile, scoped
`git diff --check` passes, and a real Citizen + full-clean SemLex two-batch optimizer
smoke completed. Run bone-only only after the isolated PartMix and part-wise trials;
do not merge these ideas in their first experiments. The consolidated evidence,
literature mapping, bottleneck diagnosis, and experiment ladder are recorded in
`artifacts/reports/STAGE1_V17_SQUEEZEFORMER_EXHAUSTIVE_AUDIT.md`. No test data was
accessed.

## 2026-08-12 02:42 PST — complete Kaggle challenger package staged; OAuth blocked

All safe work that can precede GPU allocation is complete. The v2 Kaggle archive at
`artifacts/generated/kaggle_stage1_challengers_v2/stage1_v17_challengers_trainval_v2.tar.gz`
contains the current PartMix, feature-isolated part-wise, and bone-feature code plus
the same frozen train/validation-only data. Its SHA-256 is
`8a41f26d3393e388d176cf7648b4c3b33797d80de013fa286883508df6c82b79`.
An independent archive-member audit reports 1,476 Citizen train archives (1,475 usable
after frozen rejections), 378 Citizen validation archives, 1,388 approved SemLex train
archives, and zero `/test/` members. `PACKAGE_MANIFEST.json` records these counts.

`kaggle_stage1_challenger_runner_v17.py` is a single fail-closed CUDA runner for the
three isolated experiments. It verifies the v2 archive hash and safe members, rejects
CPU allocation, preserves the seed/data/sampler/schedule/patience protocol, and writes
separate PartMix, part-wise, or bone outputs. It compiles and scoped
`git diff --check` passes.

The official Kaggle OAuth process remains alive after more than twelve minutes, but
the browser callback has not been approved. The CLI still reports username
`francisbatiancela`; `kokoab/notebook697ebe7d11` still returns 403. This is the same
external authentication blocker across three consecutive goal turns. No further
upload, private-notebook mutation, or real GPU experiment can be performed without
the user clicking Kaggle's **Authorize/Allow** action in the Brave tab (or otherwise
authenticating the CLI as the notebook-owning account). No test data was accessed.

## 2026-08-12 02:47 PST — Kaggle OAuth fixed; controlled PartMix run launched

The user approved Kaggle OAuth and the CLI now identifies as `kokoab`. The old private
notebook has no active session, so a fresh fail-closed script kernel was used rather
than mutating stale notebook state. The verified v2 train/validation-only package was
uploaded privately as `kokoab/slt-v17-stage1-challengers-v2`; Kaggle finished indexing
it at 34,060,542 bytes. Its archive SHA/count/test-member invariants remain those in
the 02:42 handoff above.

Private kernel version 1 at `kokoab/slt-v17-stage1-partmix-p50-v2` is RUNNING. It
requests `NvidiaTeslaT4`, requires `torch.cuda.is_available()`, verifies the exact v2
archive hash, rejects unsafe links/paths and every `/test/` member, and only then runs
the isolated PartMix-p=0.5 experiment with the unchanged d=256/depth=4, seed-1701,
50/50 class/source-balanced, patience-30 protocol. No local heavy process is running.
Do not report the allocation or metrics until the kernel log proves CUDA and the
result artifacts are pulled and audited.

## 2026-08-12 02:53 PST — Kaggle v3 passed CUDA and integrity gates; training active

Kaggle automatically expanded the uploaded tarball into a dataset directory rather
than mounting the archive itself. Kernel v1 therefore failed after the CUDA gate but
before training because it looked only for the archive. The dataset-file API and a
fresh download proved Kaggle exposes the archive contents under the prefix
`stage1_v17_challengers_trainval_v2/`; no data content was changed. A deterministic
tree digest was added over every relative path and file SHA-256. The exact extracted
tree contains 3,253 files and hashes to
`990c6045244b00f409d735808b57025132653fe43a37073c34aa4e93ae96fad2`.
The runner accepts either the original archive+archive hash or the Kaggle-extracted
root+tree hash and retains the no-link/no-test checks.

Kernel v2 still assumed the traditional one-level `/kaggle/input/<slug>` mount and
failed after CUDA but before training. Kaggle's current mount is nested more deeply,
so v3 now finds the unique extracted root recursively and then verifies the same full
tree digest. Version 3 has remained RUNNING beyond both prior 7-second failures,
proving the CUDA requirement, dataset resolver, and full 3,253-file integrity gate
passed. The controlled PartMix training is active on Kaggle; the laptop is only
polling status. No test data was accessed.

## 2026-08-12 02:59 PST — PartMix completed and rejected; part-wise run launched

Kaggle kernel v3 completed successfully on a Tesla T4 in about 4.3 minutes. The
PartMix-p=0.5 checkpoint retained epoch 51 and early-stopped after 81 epochs / 30
stale epochs. Its SHA-256 is
`508b8aec656ed5e717a87fe734c451ab66dce83c9f0e5b8d5dbb5845ee06e7f9`.
The 81-epoch realized PartMix fraction ranged 47.61%-52.92% and averaged 50.11%, so
the requested treatment was actually applied. Provenance confirms 1,475 Citizen +
1,388 SemLex training clips, 50/50 expected source exposure, the frozen manifest and
schema hashes, the unchanged 6,470,885-parameter global architecture, CUDA+AMP, and
false Citizen/SemLex test access flags.

PartMix ties the current landmark winner on Citizen top-1 at 362/378 = 95.77%, but
scores 99.47% top-5 and 95.31% macro F1 versus the winner's 100.00% and 95.51%.
The predeclared SemLex-validation diagnostic scores 837/978 = 85.58% top-1,
932/978 = 95.30% top-5, and 83.45% present-class macro F1. The winner has 840/978 =
85.89%, 940/978 = 96.11%, and 82.60%. Thus PartMix improves SemLex macro F1 by 0.86
points but loses three top-1 and eight top-5 clips while providing no Citizen top-1
gain. Decision: reject PartMix as the new default and do not combine it with the next
challenger. Exact pulled artifacts are under
`artifacts/generated/kaggle_stage1_partmix_kokoab_pull_v3/`; the SemLex report is
under `artifacts/reports/semlex_citizen100_val_audit/partmix_p50_v2/`.

The next isolated experiment is now RUNNING as private kernel
`kokoab/slt-v17-stage1-partwise-v2`. It uses the same verified 3,253-file dataset
tree, seed/data/sampler/schedule/patience, CUDA gate, and d=256/depth=4 global stack,
but PartMix is disabled and one isolated Squeezeformer layer per left hand, right
hand, face, and body precedes whole-body fusion. It has 6,791,717 parameters. No test
data was accessed and no heavy local process is running.

## 2026-08-12 02:14 PST — local-landscape conclusion and SHuBERT feasibility gate

Landscape orientation is already handled correctly by the v17 supplemental path:
raw frames retain aspect ratio, hand crops follow v17 landmarks and motion timing,
and face crops use eye alignment rather than stretching. Orientation therefore does
not disqualify a clip, but it also does not make resolution irrelevant: tiny, blurred,
occluded, or off-frame hands still lose usable information. The full immutable local
pool has now been screened, not sampled: 1,021 exact-text candidates received current
landmark, MobileCLIP2 hand, and visual-speech diagnostics. The original 434 Tier-A
clips remain the strong local training set. Exactly 23 additional unused clips across
15 classes pass the conservative landmark+hand/crop-quality upgrade gate; they remain
non-training-eligible until ASL-fluent exact-variant review. The local mouth/lower-face
heads are excluded because they collapse to 2.15%-3.23% folder-label agreement on
this domain. Bulk-admitting the remaining local clips would add label and extraction
noise and over-weight a small number of recording sessions.

The official SHuBERT repository was shallow-cloned at commit
`cc1929326075bfbad7ad73159b2acf84356059bb` for a read-only integration audit; no
weights or datasets were downloaded. The 57 MiB code clone describes a four-stream
pipeline that requires YOLOv8 signer crops, MediaPipe face/hands/body, three
fine-tuned DINOv2 RGB streams, CUDA, and a custom Fairseq SHuBERT-base encoder (12
layers, width 768). It is not input-compatible with Apple Vision v17 or the current
MobileCLIP2 cache, its released inference path hard-codes CUDA and lacks variable-
length batching, and its downstream fine-tuning documentation is still `TODO`.
The README says the project is primarily MIT licensed, but the separately hosted
weight terms are not established in the repository. Decision: do not blindly
download or make SHuBERT the next mobile experiment. Preserve it as a later isolated
Kaggle frozen-teacher probe after checkpoint sizes/hashes/terms are known. Full audit:
`artifacts/reports/shubert_v17_feasibility.md`. No test data was accessed.

The next executable data/model sequence is: obtain exact-variant review for the 23
local candidates; promote only approved clips into a new immutable train-only
manifest; retrain the compact hand branch with the same Citizen/SemLex/local source
balancing; compare on Citizen validation and the unchanged SemLex validation
diagnostic with fixed fusion weights; then freeze the chosen ensemble before the
independent portrait-iPhone capture. More generic lip-reading data is not the next
lever: Auto-AVSR already supplies broad visual-speech pretraining, while exact-vocab
natural-mouthing supervision and portrait-domain validation are missing.

## 2026-08-11 21:34 PST — anatomical/phonological challengers implemented

The current 6,470,885-parameter flat v17 Squeezeformer remains the measured deployable
baseline at 95.77% Citizen validation top-1 and 85.89% SemLex validation top-1. The
larger d=384 run did not improve Citizen validation, and the earlier generic supervised
contrastive-loss experiments materially hurt it. More capacity or a stronger generic
penalty is therefore not the next justified move.

A targeted architecture/data audit found that the baseline flattens all 61 nodes per
frame before temporal modeling. It has no anatomical graph, explicit hand/face/body
part representation, or sign-phonology supervision. The remaining 16 Citizen
validation errors include confident minimal-pair confusions such as THANKYOU->GOOD and
ANSWER->GO; their landmark-presence statistics do not support treating them as simple
extraction failures. The strongest next landmark hypotheses are therefore explicit
spatial structure and auxiliary handshape/location/movement learning, while preserving
the already strong Squeezeformer temporal stack.

`active/v17/model_v17.py` now has two backward-compatible, opt-in capabilities:

- `spatial_encoder=graph_parts` uses a sparse normalized physical graph, an
  input-sensitive global joint-attention branch, and explicit left-hand, right-hand,
  face, body, and whole-body part pooling before the temporal Squeezeformer;
- optional training-only phonology heads predict ten ASL-LEX attributes from the same
  pooled representation. Ordinary `forward()` and the default flat state dict remain
  unchanged.

`scripts/build_v17_phonology_targets.py` joined every one of the 100 frozen manifest
ASL-LEX codes to the local official ASL-LEX 2.0 table and generated
`active/v17/citizen100_phonology.json`. The mapping and both source hashes are frozen.
Coverage is 100/100 classes for handshape, selected fingers, sign type, movement, major
and minor location, contact, repeated movement, and wrist twist; flexion covers 95/100.
Missing annotations use ignored target `-100`, never a fabricated category. The
trainer now supports the graph options and a weighted mean auxiliary phonology loss,
and records their complete provenance.

The focused Stage-1 suite passes 21/21 tests. All flat-phonology, graph-only, and
graph-plus-phonology real-data smokes passed; the combined path also completed a real
MPS optimizer/checkpoint smoke. The pre-change best flat checkpoint strict-loads with
all keys matched and retains exactly 6,470,885 parameters.

The single preregistered flat-plus-phonology run at weight 0.20 completed on the exact
1,475 Citizen + 1,388 full-clean SemLex class/source-balanced recipe. It peaked at
95.50% Citizen validation (361/378), one clip below the 95.77% baseline, and scored
85.07% top-1 / 95.30% top-5 / 81.96% macro F1 on the frozen 978-clip SemLex validation
diagnostic, below the baseline's 85.89% / 96.11% / 82.60%. This exact equal-head,
single-pooled-token auxiliary objective is rejected. It does not prove that all
phonology-aware learning is unhelpful; it proves this simple formulation is not an
accuracy win. The graph-only full run is now active on the same recipe. No Citizen or
SemLex test data was accessed.

The architecture research supports these hypotheses but does not justify copying
headline numbers blindly. SHuBERT is the strongest directly relevant ASL foundation
teacher found (roughly 1,000 hours of self-supervision over face/hand RGB plus pose),
while DSTA-SLR, VSNet, SignBERT+, BEST, MS-MAE, and recent hypergraph work independently
support dynamic spatial structure, part-aware modeling, or masked sign pretraining.
The small PhonSSM repository is **not valid benchmark evidence here**: its claimed
ASL-Citizen/official evaluations use random stratified splitting rather than the
required signer-disjoint split, and its nominal anatomical adjacency becomes dense at
initialization. Only the general graph/phonology ideas may be tested under our frozen
protocol.

The graph experiments are now resolved. A from-scratch graph/part replacement had the
same approximate total parameter count as the baseline but was dramatically slower on
MPS and reached only 78.31% Citizen validation at epoch 30, versus more than 92% for the
flat model at that stage. It was stopped rather than heating the laptop for a plainly
inferior multi-hour trajectory. A better residual formulation was then implemented:
the proven flat frame token is retained, a smaller 32-D/one-layer graph-part branch is
added through a bounded zero-initialized scalar gate, and the exact 95.77% flat
checkpoint is strict-warm-started. Its epoch-0 logits and validation metrics are
bit-for-bit identical to the flat checkpoint. Ten graph-only epochs followed by
low-rate joint fine-tuning never exceeded that epoch-0 result and stopped after the
declared 20 stale epochs. The graph replacement and zero-gated graph residual are both
rejected for accuracy. The best residual checkpoint simply preserves the zero-gate
baseline; it is not a new gain.

`--initialize-from` now enforces format, manifest, schema, and every shared architecture
field, permits only graph/phonology challenger keys to be missing, records the source
checkpoint hash, and saves an epoch-0 fallback. `--freeze-warm-start-epochs` supports
new-branch-only adaptation before joint fine-tuning. The residual equality regression
test raises the focused suite to 22/22, all affected files compile, and scoped
`git diff --check` passes. Neither Citizen test nor SemLex test was accessed. These
results support retaining the flat d=256 Squeezeformer and moving the accuracy ladder
to better mouth/face timing and broader reviewed hand-RGB information rather than
adding landmarks or generic penalties blindly.

The old hand-trimmed 16x96 mouth package remains preserved as prior evidence. A new,
separately fingerprinted full-utterance visual-speech contract is implemented in
`schema_visual_speech_v17.py` and `extract_visual_speech_v17.py`: up to 96 face-only
reference frames across the complete video, a conservative normalized mouth-shape
motion interval with context/minimum-duration and full-utterance fallback, 32 selected
frames, per-frame eye-line alignment, and 112x112 mouth, lower-face, and full-face
views. It never uses the landmark hand-active interval, audio, or any test split.
Reflected padding contains only transformed source pixels and missing views remain
explicitly invalid/zero.

Four focused visual-speech contract/model tests pass. The new corpus is complete under
`data/local/citizen100_v17/visual_speech_rgb`: 1,475 Citizen train plus 378 Citizen
validation clips, zero extraction failures, 32 unique full-utterance-selected frames
per clip, and median 100% validity for mouth, lower-face, and full-face views. Every
clip used the mouth-motion interval rather than the landmark hand-active interval.
The visually inspected contact sheet under
`artifacts/generated/visual_speech_v17_smoke/` shows stable alignment and clear
lip/lower-face/full-face motion instead of background or hand-only timing.

The official Auto-AVSR visual-only research checkpoint is preserved at
`artifacts/model_assets/models/auto_avsr/vsr_trlrs2lrs3vox2avsp_base.pth` (SHA256
`fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb`). Its exact
11,182,784-parameter 3D-stem/ResNet-18 frontend strict-loaded all 120 transferable
keys with no missing or unexpected keys. The official model zoo reports 3,291 hours
of visual-speech pretraining and 20.3% visual-only LRS3 WER for the full model. The
frontend code is Apache-2.0; downstream checkpoint/data terms still require a license
review before product redistribution. The temporal-kernel-one MaxPool3d was replaced
only by its mathematically identical per-frame MaxPool2d form for Apple MPS support.

All six frozen feature caches are complete for mouth, lower-face, and full-face train
and validation views. Identical class-balanced temporal heads achieved 29.63% Citizen
validation top-1 for mouth, 26.72% for lower face, and 23.02% for full face. Mouth is
more than three times the old 8.99% hand-timed MobileNet mouth result, demonstrating
that full-utterance timing and relevant pretraining matter; full face is rejected as a
primary VSR view. Per-sample standardized late fusion of the 95.77% landmark model
with mouth at 0.55/0.45 reached 364/378 = 96.30%, fixing two landmark errors with no
regressions. A coarse development grid over landmark/mouth/lower/full found
0.50/0.10/0.40/0.00 and reached 367/378 = 97.09%, fixing five errors with no
regressions; artifacts are under
`artifacts/reports/stage1_v17_landmark_auto_avsr_mouth_validation/` and
`artifacts/reports/stage1_v17_landmark_auto_avsr_mouth_lower_face_97_validation/`.
This is same-validation-selected research evidence, not an independent production
estimate. A learned mouth/lower-face fusion head is the next controlled experiment.
Citizen test, SemLex test, and audio remain untouched.

The first learned mouth/lower-face head formulation (LayerNorm before each frozen
feature projection) was stopped by patience at epoch 21 after remaining at chance
(1.59% best). A fixed-batch diagnostic proved the model and MPS gradients could learn,
while a full balanced-stream comparison exposed optimization instability. The next
run therefore matches the already proven single-view ordering—linear projection then
LayerNorm—without changing the views, data, validation protocol, or seed. The failed
checkpoint is retained as negative evidence, not reported as a data limitation.

The corrected learned mouth/lower-face model completed 100 epochs and retained its
epoch-82 checkpoint at 31.75% Citizen validation top-1, above mouth alone (29.63%) and
lower face alone (26.72%), with only 3,347,941 trainable parameters over cached
features. Equal per-sample-z-score fusion with the landmark winner reaches 365/378 =
96.56%, fixing three landmark errors and introducing no regressions. A development
weight sweep has the same 365-clip ceiling, so this cleaner learned head does not beat
the validation-tuned separate-mouth/lower ensemble at 367/378. It is retained as the
compact visual-speech candidate; the two-head late fusion remains research ceiling
evidence. The exact artifact is
`artifacts/models/stage1_v17_visual_speech_auto_avsr_mouth_lower_learned/`, with the
equal-fusion report under
`artifacts/reports/stage1_v17_landmark_auto_avsr_mouth_lower_learned_equal_validation/`.

The reviewed hand-RGB expansion is now complete. A strict train-only supplement
extractor resolves exactly 1,388 full-clean SemLex clips across 97 classes and 434
Tier-A dual-top1 local clips across 72 classes; raw paths, v17 landmark trim paths,
selection hashes, source identity, and the sealed-test contract are checked per item.
It also adds a decoded-frame-count fallback for WebM files whose container reports
negative/garbage counts; the already decoded landmark reference count remains the
contract. All 1,822 new archives use the unchanged hand schema fingerprint
`bf6508de2ea851a4`, passed a complete audit with zero corrupt/empty files, and occupy
704 MiB. SemLex left/right/union validity is 43.39%/75.20%/85.34% with 33.24% two-hand
frames; local Tier-A is 75.17%/59.79%/93.99% with 40.97% two-hand frames. Together
with 1,475 Citizen train clips, the planned reviewed RGB training pool is now 3,297.
Compact embedding extraction and a source-balanced hand retrain are the next gate;
the several-gigabyte spatial-map expansion is deferred until compact features prove
that the additional sources help. No validation/test source data was added or opened.

Frozen official MobileCLIP2-S0 compact encoding is complete for all 1,822 new hand
archives: 1,388 SemLex in 700.8 seconds and 434 local Tier-A in 259.2 seconds on MPS.
Every `[16,3,512]` archive passed schema `c54f4edc6f62b08b`, finite-value, explicit
zero-mask, source-provenance, sealed-test, and unit-norm audits; there are zero corrupt
or empty feature archives. The source-balanced trainer now loads selections by their
exact frozen manifests and samples 45% Citizen, 45% SemLex, and 10% local, with equal
class mass inside each available source. Its real 200-train/100-validation optimizer
smoke passed after fixing a smoke-only reduced-dataset/full-sampler index mismatch.
The full 3,297-sample compact hand retrain is the active next measurement.

The source-balanced compact hand retrain completed after 63 epochs and retained epoch
33 at 80.69% Citizen validation top-1 / 94.71% top-5 / 79.32% macro F1. This is a
10.32-point gain over the Citizen-only compact model (70.37%) and a 10.05-point gain
over the old Citizen-only spatial model (70.63%), proving that reviewed cross-source
volume was a major hand-RGB limitation. A fixed 75/25 landmark/hand standardized
fusion is rejected because it regresses one net clip (361 versus 362), despite the
stronger standalone hand score. The hand model's contribution is complementary only
in the broader visual ensemble: a coarse 0.05 development grid at 0.30 landmark,
0.15 mouth, 0.35 lower-face, and 0.20 hand reaches 370/378 = 97.88% top-1 and 97.59%
macro F1, fixing ten landmark errors while regressing two. This is the new research
teacher ceiling, but its weights were selected on the same Citizen validation set and
require independent portrait confirmation. Exact reports are under
`artifacts/reports/stage1_v17_hand_mobileclip2_multisource_balanced_validation/` and
`artifacts/reports/stage1_v17_multimodal_teacher_97_88_validation/`. No test was used.

Before spending roughly another 4 GiB on supplement spatial maps, the next cheapest
controlled experiment warm-starts the already cached Citizen spatial fine-tuner from
the new multisource compact checkpoint. This tests whether pre-pooling adaptation adds
value without re-encoding data or changing the validation protocol.

Both spatial warm-start formulations are now rejected, so supplement spatial-map
extraction remains cancelled. The old hard temporal shift destroyed the multisource
feature geometry: it reached only 66.93%, 68.52%, 72.22%, 73.02%, then 70.90% through
five epochs, with epoch time rising to roughly one minute, and was stopped. A better
zero-gated residual was implemented and regression-tested. Real cached spatial maps
without the shift reproduce the compact checkpoint exactly at 305/378 = 80.69%; its
epoch-0 checkpoint is saved. Conservative 2e-5 joint adaptation nevertheless fell to
75.13% and 73.28% in two epochs and was stopped. This proves the cached pre-pooling
maps are valid while rejecting temporal shift plus joint final-block adaptation as the
next accuracy move. The compact 80.69% hand checkpoint remains selected; no new
supplement spatial maps were created. End-to-end visual-speech pixel fine-tuning is
the next bounded experiment.

The local end-to-end mouth benchmark confirms this experiment belongs on GPU: five
training batches plus full validation took 28.1 seconds on MPS. A private Kaggle
dataset was therefore created by CLI under
`francisbatiancela/slt-v17-visual-speech-pixel-v1`, containing only the 1,475 Citizen
train and 378 Citizen validation visual-speech archives, current v17 code/manifest,
the official Auto-AVSR checkpoint, and the 29.63% frozen-mouth warm start. No test data
is included. The originally supplied private `kokoab/notebook697ebe7d11` is inaccessible
to the authenticated CLI OAuth account (`francisbatiancela`), so a new private GPU
kernel was pushed via CLI at
`francisbatiancela/slt-v17-auto-avsr-mouth-fine-tune`. It strict-warm-starts the proven
mouth checkpoint, saves epoch 0 as a fallback, fine-tunes the complete visual frontend
at 1e-5 and head at 3e-5, and uses patience 10. The kernel is currently running; its
output must be pulled and audited before any accuracy claim.

Kaggle GPU execution is externally blocked despite a correct and fully uploaded
private package. Five CLI kernel versions were audited: script and notebook formats,
explicit `enable_gpu`, `NvidiaTeslaT4`, and generic `Gpu` shapes were tried; the final
notebook syntax is valid and the CLI reports 30/30 GPU hours remaining. Kaggle still
scheduled CPU-only PyTorch every time. The CUDA assertion prevented accidental CPU
training, and logs for every attempt are preserved under
`artifacts/generated/kaggle_visual_speech_pixel_failed_v*/`. This is an allocator
failure, not a model/data failure. The reusable private dataset and kernel remain at
`francisbatiancela/slt-v17-visual-speech-pixel-v1` and
`francisbatiancela/slt-v17-auto-avsr-mouth-fine-tune`.

A bounded local progressive-unfreeze study then resolved end-to-end mouth adaptation.
The initial trainer's random 88-of-112 crop and four-frame deletion collapsed the
warm-start model to 19.31%. A mild regime was added (center-aligned +/-4 pixel jitter,
gentle photometric change, no temporal deletion). A second warm-start issue was also
found: the from-scratch EMA ramp deliberately forgets early weights and therefore
overwrote the pretrained solution. Warm starts now enter the mature fixed-decay EMA
regime. With both fixes, layer-4-only fine-tuning at head 1e-5/frontend 1e-6 is stable
but not beneficial: epochs 1/2/3 reached 29.37%/28.84%/28.57%, and patience retained
epoch 0 at 29.63%. Citizen-only end-to-end visual-speech adaptation is rejected; the
frozen Auto-AVSR mouth/lower-face teacher and learned cached-feature head remain the
selected visual-speech paths. The next accuracy gate is independent portrait-iPhone
confirmation and out-of-fold fusion calibration, followed by a sign-pretrained
video/context RGB teacher or targeted new exact-vocabulary recordings—not further
same-validation weight/model tuning.

## 2026-08-11 21:13 PST — Auxiliary RGB audit fixes the next Stage-1 ladder

The current mouth and hand RGB branches are useful proof-of-signal experiments, not
best-available algorithms. The 1.086M-parameter mouth model uses an ImageNet-pretrained
MobileNetV3-Small, 16 96x96 crops, two shallow temporal blocks, and only Citizen train
data. It reached 8.99% Citizen validation top-1 and corrected two errors from the clean
landmark winner, which confirms complementary information but not a deployable
lip-reader. More importantly, its crops are sampled only within the landmark
hand-activity interval; spoken or mouthed words can occur before or after that interval.

The next mouth experiment must rebuild the crop package over the full utterance using
face alignment and a speech/mouth-motion interval, with roughly 29-32 frames. Audio may
be used offline for VAD/transcription/forced alignment and label-quality auditing, but
the inference branch remains visual-only. Compare aligned mouth/lower-face and
full-face inputs because published VSR evidence shows extraoral facial motion can help.
Use a pretrained visual-speech frontend such as Auto-AVSR or AV-HuBERT plus a stronger
temporal head as an accuracy teacher before designing a mobile student. LRW is useful
for pretraining but is not a direct 100-class supplement: only 17 frozen vocabulary
labels exactly overlap LRW's 500 words, and its official 70 GB distribution requires
the BBC academic data-sharing agreement. LRS3 is sentence-level pretraining/mining
data, not clean isolated examples of all 100 ASL labels. A genuine exact-vocabulary lip
dataset would therefore require targeted recording or rigorously aligned mining and
must be fine-tuned on signing-domain faces rather than treated as equivalent ASL data.

The current hand RGB branch is also not the strongest available design. It was trained
only on 1,475 Citizen clips, starts from generic MobileCLIP image features, and its hard
temporal-shift/final-block fine-tuning reached 70.63% alone. Fixed late fusion with the
clean landmark winner added only one Citizen validation clip (96.03%). Before another
architecture sweep, extract the identical left/right/union RGB schema from the existing
1,388 reviewed full-clean SemLex clips and the 434 Tier-A local clips, then train with
explicit class/source balance. This raises the reviewed hand-RGB training pool from
1,475 to 3,297 without opening either sealed test. After that, evaluate a genuinely
video-native, sign-pretrained RGB teacher (SignRep/Hiera/VideoMAE-family reference) and
retain full upper-body/context information where hand-only crops lose sign location,
contact, or facial evidence. Distillation and mobile optimization remain downstream.

The controlled order is now: (1) collect an independent portrait-iPhone development
set because the 97.09% five-model ensemble weights were selected on the same 378-clip
Citizen validation set; (2) rebuild and retrain the visual-speech branch; (3) expand and
retrain the hand/context RGB branch; (4) calibrate fusion out of fold and accept it only
if both Citizen and the independent portrait set improve without damaging SemLex; and
(5) distill a validated multimodal teacher into a single practical student and measure
it on real iPhones. Do not move the accuracy effort to Stage 2 yet, and do not present
the fragile five-model validation ensemble as production evidence.

## 2026-08-10 17:34 PST — Accuracy ladder opened; local and lip experiments gated

The user authorized continued controlled experimentation toward approximately 97%
Citizen validation accuracy, including the model-screened local pool and a separate
lip-reading-style branch. Citizen test remains consumed and prohibited for tuning;
SemLex test remains sealed. A scan of the existing frozen validation logits showed
that the d=256/d=384 probability ensemble reaches only 364/378 (96.30%), two clips
over d=256 alone but three clips below 97%, while other existing model combinations
are no better. This is not enough to justify doubling mobile inference cost.

The next experiment ladder is deliberately ordered by cost: a second clean d=256 seed;
then Tier-A local supplementation at a bounded source share; then a face/lip-only
landmark branch and fusion; only afterward new mouth-pixel or hand-crop RGB work. The
second clean seed-3407 run is currently active on MPS with the identical 1,475 Citizen
+ 1,388 full-clean SemLex inputs and 50/50 class/source-balanced protocol.

`active/v17/train_stage_1_v17.py` now has a strict local-review loader and explicit
nonuniform source margins. The real three-source contract loads 1,475 Citizen, 1,388
SemLex, and exactly 434 Tier-A local clips and solves to 45%/45%/10% expected source
exposure while retaining exactly 1% expected exposure for every class. Local signer
identity remains unknown, the review manifest remains unmodified, and use requires an
explicit CLI approval flag and records the selected tiers/hashes in provenance.

`Stage1V17Config` also supports `all`, `hands`, `face`, and `mouth` model-visible node
masks inside the network, so a separate lip/face diagnostic cannot accidentally see
hand/body features and remains consistent across training, evaluation, checkpointing,
and export. The cheap branch uses the existing four v17 mouth landmarks first; those
landmarks are only a low-temporal-resolution proxy, so a weak result will trigger a
separate real-pixel mouth-crop experiment rather than a claim that lip information is
useless. Fifteen focused Stage-1 tests pass, including explicit source margins, strict
local-tier loading, and proof that face-only logits are invariant to changed hand
nodes. Relevant primary research reviewed for the ladder includes SWA
(`https://arxiv.org/abs/1803.05407`), model soups
(`https://arxiv.org/abs/2203.05482`), supervised contrastive learning
(`https://arxiv.org/abs/2004.11362`), and SAM
(`https://arxiv.org/abs/2010.01412`).

The clean seed-3407 run subsequently completed after 68 epochs at 94.71% Citizen
validation top-1 (358/378), below the seed-1701 winner's 95.77%. It is rejected as a
standalone candidate and was not evaluated on either test. The controlled Tier-A local
10% run then started from scratch with seed 1701 and the declared 45% Citizen / 45%
SemLex / 10% local source margins. The trainer also now has an opt-in supervised
contrastive objective for a later isolated ablation; its default weight is zero, so it
does not change the active local-only run. Sixteen focused Stage-1 tests pass.

The real-pixel lip route is now schema-gated separately in
`schema_mouth_rgb_v17.py` and `extract_mouth_rgb_v17.py`. It samples 16 actual video
frames only inside each archive's frozen v17 hand-active interval, runs Apple face
landmarks without hand/body requests, makes a 96x96 square lower-face/mouth crop,
stores JPEG pixels plus explicit validity/boxes/source indices, rejects test as a CLI
split, and embeds source/schema provenance. A two-clip real Citizen smoke completed
2/2 with zero failures; all 16 frames in both clips were usable. Visual inspection of
`artifacts/generated/mouth_rgb_v17_smoke/contact_sheet.jpg` shows stable alignment and
clear lip motion rather than accidental full-frame/background content. The three pure
mouth RGB tests and sixteen Stage-1 tests pass. Based on MobiVSR
(`https://arxiv.org/abs/1905.03968`), a small depthwise-separable visual-speech model is
the intended pixel baseline; AV-HuBERT (`https://arxiv.org/abs/2201.02184`) is a much
heavier research reference rather than the first mobile branch. Audio exists in most
sampled Citizen files, but the first branch will remain visual-only so it cannot win by
an audio-label shortcut that would be unavailable to a non-speaking signer.

The matching visual-only classifier is implemented separately in
`model_mouth_rgb_v17.py` and `train_stage_1_mouth_rgb_v17.py`. It is a
depthwise-separable 3D spatial/temporal network with validity-masked attention pooling,
under one million parameters, sign-safe mirror/color/temporal augmentation, EMA,
train/validation-only loading, saved aligned validation logits, and explicit
`visual_only:true`, `audio_accessed:false`, and `test_evaluated:false` checkpoint
provenance. Its forward/missing-frame test passes. This code is prepared but no full
mouth crop extraction or classifier result exists yet; do not quote the smoke as
accuracy. Meanwhile the active local-10% landmark run reached 95.50% Citizen
validation at epoch 41, already ahead of the clean baseline at the same stage but not
yet above the clean run's final 95.77% checkpoint.

The controlled Tier-A local-10% run completed after 84 epochs. Its retained epoch-54
checkpoint ties the clean winner at 95.77% Citizen validation top-1 (362/378), with
99.74% top-5 and 95.31% macro F1. Relative to the clean winner it corrects five clips
and regresses five, leaving eleven clips wrong for both. Probability ensembling the two
reaches only 363/378; the strongest existing d=256/d=384 ensemble remains 364/378
(96.30%). On the 978-clip SemLex validation diagnostic, however, local-10 reaches
86.50% top-1 / 96.22% top-5 / 84.25% present-class macro F1 versus the clean winner's
85.89% / 96.11% / 82.60%. Therefore Tier-A local data provides a genuine secondary-
domain improvement without sacrificing the primary gate, but it does not reach 97%
alone. Citizen and SemLex test remain untouched. The separate d=128/depth-2 face-only
landmark run has started on Citizen plus full-clean SemLex with no local data and no
hand/pairwise visibility.

The face-only landmark proxy completed after 35 epochs at only 2.65% Citizen
validation top-1 and is rejected. The four mouth points are too sparse and mostly
interpolated to support visual speech; this result does not reject real mouth pixels.
Full real-pixel extraction then completed for all 1,475 Citizen train and 378 Citizen
validation clips with zero failures and zero archive/schema/decode errors. Median
validity is 16/16 frames in both splits; the minima are 12/16 train and 15/16
validation. The complete crop corpus is only 62 MiB, and Citizen test was not accepted
or accessed.

The first 115k-parameter depthwise 3D classifier attempt was interrupted before one
epoch after macOS MPS took more than two minutes; this was an operator-throughput
failure, not an accuracy result. The model was changed to a shared depthwise-separable
2D frame encoder plus the same temporal depthwise head/attention. It has 104,785
parameters, uses mobile-friendly 2D/1D operators, completes a warm batch-16 forward+
backward in about 0.084 seconds on MPS, and its first real training epoch completed in
about ten seconds. A noncontiguous-MPS backward edge case found on the initial full run
was fixed with explicit contiguous boundaries and reproduced successfully on a real
augmented batch before restart. The active visual-only mouth run remains Citizen
train/validation-only with audio and test both untouched.

The 104,785-parameter mouth model trained from scratch was stopped after twelve epochs
because train loss fell while Citizen validation remained at 1-2% top-1; its pixels
alone did not provide enough data to learn signer-independent facial features. An
isolated torchvision 0.23.0 target directory was created under
`artifacts/generated/mouth_rgb_torchvision` without changing the validated venv. It
uses the torch-2.8-compatible macOS wheel and the official ImageNet MobileNetV3-Small
weight with SHA-256
`047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`.

The resulting pretrained visual-only MobileNetV3-Small mouth branch has 1,086,789
active parameters and trained for 53 epochs with a lower backbone learning rate. Its
best Citizen validation checkpoint is epoch 41 at 8.99% top-1 (34/378); late top-5
reached 21.96%. It accessed neither audio nor test. It correctly recognizes two of the
clean/local landmark models' sixteen errors: one `THANKYOU -> GOOD` and the
`ANSWER -> GO` clip. However, dense probability/z-score fusion reaches at most 363/378
with the clean model and does not improve the local model; uncertainty-gated top-k
reranking is also neutral. Therefore mouth pixels contain genuine complementary word
signal but this small Citizen-only branch is rejected for runtime/fusion and does not
reach the accuracy goal.

The next controlled run has started from scratch on the clean 1,475 Citizen + 1,388
SemLex d=256 protocol with a 0.05 supervised-contrastive loss weight and temperature
0.10. Architecture, data, 50/50 class/source sampling, augmentation, seed, validation,
and test isolation remain unchanged. This tests explicit cross-domain same-class
embedding alignment independently of local supplementation.

The fixed 0.05 supervised-contrastive run was stopped at epoch 27 after reaching only
90.48% Citizen validation versus the clean run's 93.92% by epoch 25; the persistent
contrastive constraint slowed/displaced classification learning. A second controlled
schedule linearly decayed the same contrastive weight to exactly zero after epoch 12,
preserving its small early advantage and then reverting to plain cross-entropy. It was
also stopped after epoch 22 at 89.95% versus the clean run's 93.39% by epoch 21.
Neither schedule approached the winner, so supervised contrastive learning is rejected
for this class/source sampler and batch regime rather than combined with local data.

The next clean ablation has started with `input_modality=hands`, changing only which
v17 nodes are visible inside the same d=256 model. Both hands, confidence, temporal
derivatives, and pairwise hand-shape distances remain; the chance-level face nodes and
four body nodes are masked. Data, sampler, augmentation, optimizer, schedule, seed,
validation, and test isolation are otherwise unchanged.

The hands-only ablation initially led the all-node baseline by roughly four to seven
points, but lost that advantage as training matured. It was stopped after epoch 23 at
90.21% Citizen validation versus approximately 93.92% for the clean all-node run at a
comparable epoch. This rejects permanent face/body masking: even sparse auxiliary
nodes provide useful late-stage disambiguation.

A clean source-ratio ablation then increased SemLex exposure from 50% to 60% while
holding all other d=256 settings fixed. It was stopped after epoch 24 at 90.74%
Citizen validation, materially behind the baseline curve. The reciprocal 60% Citizen /
40% SemLex run is now active. These are development-validation experiments only;
Citizen test remains consumed and prohibited, and SemLex test remains sealed.

Mirror test-time augmentation was evaluated without training on the retained clean
winner. The original view gets 362/378, the mirrored view 359/378, and both probability
and logit averaging get 360/378 (95.24%). Mirror TTA is rejected because it doubles
classifier inference while losing two correct clips. The active 60% Citizen / 40%
SemLex run reached 91.53% at epoch 22; it remains active because earlier retained runs
made material late-training gains after epoch 40.

The 60% Citizen / 40% SemLex run completed after 86 epochs. Its best checkpoint was
epoch 56 at 95.24% (360/378), 99.47% top-5, and 94.99% macro F1, so the original
50/50 clean model remains better by two clips. Citizen and SemLex test were untouched.

`evaluate_model_soup_v17.py` now performs schema/manifest/config-gated two-checkpoint
weight interpolation on Citizen validation and saves a single-model checkpoint plus
aligned logits. It normalizes legacy/default-equivalent configs before comparison, and
its state-blending unit test passes. Clean/Tier-A-local and clean/60:40 soups at weights
0, 0.25, 0.5, 0.75, and 1 never beat the clean endpoint's 362/378; intermediate soups
fell as low as 359/378. Therefore weight-space averaging is rejected for these
independently optimized trajectories.

Probability ensembles were also recomputed with aligned logits. Clean plus d=384 still
peaks at 364/378. Adding the 60:40 checkpoint reaches 365/378 (96.56%) near weights
0.42/0.28/0.30; a fixed 0.1 simplex grid and 100,000 deterministic Dirichlet samples
found no 366th correction. This remains below the 367/378 target and would require two
d=256 models plus the heavier d=384 model at runtime, so it is research evidence rather
than the mobile selection. A clean 50/50-source d=256 run with label smoothing reduced
from 0.10 to 0.05 is now active; all other settings remain fixed.

The label-smoothing-0.05 run completed after 115 epochs. Its retained epoch-85
checkpoint reaches 95.50% Citizen validation top-1 (361/378), 100% top-5, and 95.18%
macro F1. It is one clip below the 0.10 clean winner and does not increase any tested
landmark-only ensemble ceiling, so 0.10 remains the standalone setting.

The previously audited high-resolution hand-spatial cache was reused without video
re-extraction. The removed OpenCLIP package layer was restored under the isolated
`artifacts/generated/mobileclip2_runtime` target (OpenCLIP 3.3.0, timm 1.0.28 and
small dependencies); the existing exact cached MobileCLIP checkpoint was reused.
Aligned 256-D train/validation features were regenerated for the clean landmark and
70.63% hand-RGB models. A zero-initialized learned residual over the 95.77% clean
model made no validation prediction changes after 21 epochs. Fixed per-sample
z-score late fusion at 75% landmark / 25% hand RGB reaches 96.03% (363/378), a real
one-clip gain but still below target.

A development-only multimodal probability ensemble has now crossed the requested 97%
Citizen validation threshold. The simple rounded weights are 0.15 clean d=256, 0.15
clean d=384, 0.10 60:40-source d=256, 0.10 hand RGB, and 0.50 mouth RGB. It reaches
367/378 = 97.09% top-1, 99.47% top-5, and 97.06% macro F1. Relative to the clean
winner it fixes six clips, regresses one, and leaves ten wrong for both. The exact
reproducible output, member hashes, normalized weights, and aligned scores are under
`artifacts/reports/stage1_v17_multimodal_ensemble_97_validation/`, produced by the new
`evaluate_multimodal_ensemble_v17.py`.

This is a **development validation result, not production-ready evidence**. The weights
were searched on the same Citizen validation split: 1,392/10,000 small perturbations
around the selected vector retained 367 correct, so the result is locally reproducible
but not broadly weight-robust. It also requires three landmark classifiers, the hand
RGB branch, and the mouth RGB branch, which is far too heavy for the current mobile
selection. Citizen test remains consumed/prohibited and SemLex test remains sealed.
Independent confirmation must use a newly collected portrait-iPhone set; until then,
the 95.77% single d=256 clean model remains the honest deployable selection and the
97.09% ensemble remains a research teacher/candidate.

Final focused validation passes: 22/22 Stage-1 and mouth-RGB tests, including the new
model-soup and multimodal alignment/probability checks. All affected Python files
compile, and the scoped `git diff --check` passes. The large pre-existing untracked
repository reorganization remains untouched.

## 2026-08-10 11:07 PST — Local v17 triage yields a 132-clip human-review shortlist

Apple Vision v17 extraction succeeded on all 356 q82 local candidates with zero
no-hand and zero failed clips; all 356 archives passed `audit_v17.py`. Frozen-model
agreement, used only as a mismatch screen, is 51.69% top-1 and 81.18% top-5. The
class-level result is 54 model-consistent, 27 ambiguous, and eight high-risk classes.
The high-risk labels are COME, GOODBYE, SIGN, TAKE, UNDERSTAND, WANT, WHAT, and YOUR.
No class or clip was automatically approved.

`scripts/select_local_citizen100_review_shortlist.py` and three focused tests were
added; all pass. A stricter final review shortlist keeps only pinned-raw-text-equal,
model-consistent classes and clips that are frozen-model top-5 hits with at least 80%
observed hand-frame coverage and 50% face presence, then caps each class at three.
This leaves 132 clips across 49 classes under
`data/local/local_citizen100_quality_audit_q82/review_raw/`, with provenance in
`review_shortlist.json`. They remain `training_eligible: false` and require ASL-fluent
exact ASL-LEX variant review. Trustworthy signer counts remain unavailable, so these
clips can only become a small train-only supplement and never validation/test evidence.

Because only 49/100 classes survive the conservative local review gates and RIT still
requires independent lexical review, external-data coverage is not yet sufficient.
The next action is a further official-source web pass, starting with a bounded,
quality-filtered MS-ASL train-only audit rather than bulk downloading web video.

## 2026-08-10 15:14 PST — Citizen + SemLex augmented Stage 1 completed; baseline retained

The Apple Vision extraction path was already fixed before this run: 1,475 usable
Citizen-train archives, 378 fixed Citizen-validation archives, and all 1,058 selected
SemLex-train archives pass the v17 schema/invariant audit. Training does not create
physical duplicate archives. `augment_v17` produces new random variation online on
every training batch through anatomy-correct left/right reflection, masked isotropic
scale/rotation/translation, masked coordinate noise, nearest-frame temporal warping,
and masked joint dropout. Binary presence and exact missing-zero contracts remain
preserved.

`active/v17/train_stage_1_v17.py` now optionally accepts a provenance-locked SemLex
supplement. `SemLexSupplementV17Dataset` requires a `train_only` manifest, rejects any
non-train clip, validates every label/video/archive/schema, and requires the explicit
`--approve-supplement` run gate. A supplemented run cannot request Citizen test.
Checkpoints and results record both dataset hashes/counts, approval state, online
augmentation, and false test-access flags. The real combined loader reports exactly
2,533 train clips (1,475 Citizen + 1,058 SemLex), 378 Citizen validation clips, 100
classes, and 6,470,885 parameters. A two-batch MPS optimizer/checkpoint smoke passed.

The first full local attempt was stopped at epoch 20 when the user initially redirected
training to Kaggle; its recoverable files are under
`artifacts/generated/stage1_v17_citizen_semlex_interrupted_epoch20/`. No Kaggle kernel
was uploaded or started. After confirming this landmark-only run is short, the user
chose local MPS and a clean seed-1701 run restarted from scratch. It completed in about
8.6 minutes and early-stopped after 67 epochs / 30 stale epochs. The best checkpoint is
epoch 37 at `artifacts/models/stage1_v17_citizen_semlex_augmented/best_model.pth`:
92.86% top-1 (351/378), 100.00% top-5 (378/378), and 92.25% macro F1 on Citizen
validation. Citizen and SemLex test were not accessed.

The primary Citizen-only baseline remains selected: its 93.12% top-1 (352/378) is one
clip better, with 99.47% top-5 and 92.53% macro F1. The augmented challenger corrected
12 former baseline errors but regressed 13 formerly correct clips, so the one-clip net
loss is real rather than identical predictions. The result is promising cross-domain
ranking coverage but not a top-1 win; do not replace the baseline or tune against the
consumed Citizen test. The validation-only augmented report is under
`artifacts/reports/stage1_v17_citizen_semlex_augmented_validation/`.

## 2026-08-10 15:23 PST — v16 96% protocol audited against v17 generalization

The v16 96.00% number is real for its recorded protocol but is not comparable to the
v17 signer-disjoint result. Its deep-cleaned corpus has 62,023 clips and 310 classes,
split by deterministic per-class random clip shuffle into 43,263 train, 9,311
validation, and 9,449 test clips. The loader explicitly says that this mixes every
video source/signer across splits. The local corpus has approximately seven recurring
people/sessions per class, so v16 validation/test mainly measure new clips from familiar
people and capture conditions rather than new-signer generalization.

The v16 cleaning loop also leaks evaluation information. A checkpoint trained on the
random-split corpus scored all 66,770 original samples, including clips later assigned
to validation/test, and removed the bottom 3% by model self-confidence. Statistical
cleaning then removed another 2,744 coordinate/jitter/distribution/low-motion outliers;
4,747/66,770 clips (7.1%) were removed before the final split was recomputed. This can
improve real label/landmark quality, but it also favors the existing model's decision
boundary and allows future evaluation clips to influence corpus selection.

An exact SHA-256 audit of every deep-cleaned v16 `.npy` found 60,999 unique arrays among
62,023 files: 1,024 duplicate pairs remain. Of these, 444 cross split boundaries;
221 validation files and 175 test files have an exact array duplicate in training.
There are 399 cross-split same-label duplicate pairs and 45 cross-split conflicting-
label pairs. Frequent conflicting aliases include ABOUT/WHEN (59 total pairs) and
WORK/WORKER (25). This is direct evidence that the reported 96.45% validation and
96.00% test sets are not independent. The v17 Citizen features have zero exact feature
duplicates across their 3,101 archives/splits.

The v16 pool nevertheless contains useful train-only signal: 18,236 deep-cleaned clips
match 90 of the current canonical names, including 15,653 local hash-named clips. These
are not automatically exact ASL-LEX matches, independent signers, or safe validation
examples. The earlier local audit already found 623 visually strong clips across 89
folders under a seven-per-class cap; 132/209 were conservative review subsets, not the
total usable ceiling.

Generalization evidence favors v17. The frozen v16 checkpoint scored 40.28% top-1 on
72 external Citizen clips from 29 participant IDs, including 29.2% on that audit's
Citizen-test subset. Of 21 exact-variant Citizen-test clips shared with the immutable
v17 test predictions, v16 got 7/21 (33.3%) and v17 got 19/21 (90.5%). Three other audit
clips were intentionally absent from v17 because they are different variants
(`W.H.A.T`, `HOSPITAL2`, and `DRINK2`). This is retrospective reporting from existing
frozen predictions, not a new test run or a tuning signal.

The honest v17 contract remains 1,475 train clips from 32 participants, 378 validation
clips from five disjoint participants, and 1,247 usable test clips from eleven disjoint
participants, with zero participant overlap. Per-class signer ranges are 11-16 train,
3-5 validation, and 10-11 test. Its 93.12% validation and one-time 87.57% test results
therefore measure a harder and more useful question than v16's 96% random-clip score.
The 5.55-point validation/test gap also shows v17 still has a real signer-generalization
problem; it must not be hidden by reverting to the v16 split.

## 2026-08-10 15:42 PST — Balanced Citizen/SemLex challenger wins validation gate

The one predeclared sampling ablation is implemented in
`active/v17/train_stage_1_v17.py`. `class_source_balanced_weights` uses iterative
proportional fitting over existing class/source cells; the resulting replacement
sampler has exactly 1% expected exposure for each of 100 classes and 50/50 expected
Citizen/SemLex source exposure. It creates no files and changes no augmentation,
architecture, optimizer, schedule, seed, train inputs, or validation inputs. Sampling
provenance and min/max per-sample weights are stored in the checkpoint/result. A unit
test verifies both expected margins exactly, and the real-manifest MPS optimizer /
checkpoint smoke passed.

The clean full run used the same 1,475 Citizen + 1,058 SemLex train clips, 378 Citizen
validation clips, seed 1701, d=256/depth=4 model, and patience 30 as the ordinary-
shuffle challenger. It early-stopped after 76 epochs. The best checkpoint is epoch 46
at `artifacts/models/stage1_v17_citizen_semlex_balanced/best_model.pth`: 93.92% top-1
(355/378), 100.00% top-5, and 93.46% macro F1. This is +0.79 top-1 / three clips over
the selected Citizen-only baseline and +1.06 / four clips over ordinary Citizen+
SemLex shuffle. Relative to the Citizen baseline it corrected 13 former errors and
regressed ten, so the net three-clip gain is not an identical-prediction artifact.
Citizen test, SemLex validation, and SemLex test were not accessed. The validation-only
report is `artifacts/reports/stage1_v17_citizen_semlex_balanced_validation/`.

The next agreed gate is now due: acquire SemLex validation for a secondary-domain
diagnostic of the frozen Citizen-only baseline, ordinary augmented checkpoint, and
balanced augmented checkpoint. Google Drive range metadata verifies that file ID
`1VvrbYgNZe_4fWS5ZdSsHyxOuWHmhisGq` is `val.tar.gz`, exactly 8,076,365,890 bytes.
Only exact matched variants should be selectively extracted; the full archive is not
training data at this stage. The SemLex test video archive is the separate 14.66 GB
file ID `1nVjvgJhjo3lILr5S23p_PsMR7yFdQTrS` and must remain untouched.

## 2026-08-10 15:51 PST — Balanced-model local fishing yields 315 priority review clips

The user's concern is confirmed: the current signer-disjoint results establish real
learning but not production readiness. The one-time Citizen test is 87.57%, only 100
isolated classes are covered, and independent portrait-iPhone behavior, OOV rejection,
ASL variant review, licensing, and device measurements remain unresolved. The SemLex
test archive is not additional development volume: its ten signers have zero overlap
with SemLex train/validation and it must remain sealed for one final independent
cross-dataset evaluation. SemLex validation may be used once as the planned secondary
diagnostic and only then folded into a later training version if its diagnostic role is
explicitly retired.

The 623-clip, quality/diversity-capped local pool was rescored with the new balanced
Citizen+SemLex checkpoint, which was never trained on local clips. Its model-assisted
label agreement is 62.12% top-1 / 83.15% top-5, up from the independent Citizen-only
checkpoint's 52.33% / 80.26%. `scripts/select_local_citizen100_consensus.py` now joins
both immutable prediction sets, enforces the existing 80% observed-hand and 50% face
coverage gates, requires exact equality with the pinned Citizen raw gloss, and hashes
all raw files for exact deduplication. All 623 files are hash-unique.

The conservative result is 238 Tier-A clips across 70 classes where both models put
the folder label at top-1, plus 77 Tier-B clips where both include the label in top-5
and one puts it at top-1. The 315-clip A+B priority review pool covers 76 classes with
1-7 clips/class (median four); 35 classes have at least five. Another 41 Tier-C clips
have dual top-5 support only. Quarantine contains 105 extraction-quality failures, 84
clips whose folder/canonical name is not the exact pinned raw-gloss text, and 78 model
disagreements. No file was deleted and no model prediction was written back as a label.

The immutable review manifest and per-clip/class ledgers are under
`artifacts/reports/local_citizen100_quality_audit/consensus/`. They explicitly remain
`training_eligible:false`, train-only after ASL-fluent exact-variant review, because
model agreement is correlated screening evidence rather than label proof and the local
corpus has no trustworthy signer IDs. The v16 model was not used for selection because
it trained on this local domain. Fifteen focused Stage-1/consensus tests pass, both
scripts compile, and `git diff --check` passes.

## 2026-08-10 01:32 PST — Kaggle T4 full run launched

All eight private multipart datasets
`francisbatiancela/slt-v17-movinet-trainval-part-00` through `part-07` now report
Kaggle status `ready`. The official uploader's retry progress was proven unreliable:
connection resets caused it to reread bytes without matching server acceptance. A
bounded GCS-resumable helper was added as
`active/v17/kaggle_resumable_upload_v17.py`. It reused the official CLI's saved upload
sessions, queried authoritative server offsets, transferred 256 KiB chunks, recovered
resets without replaying the whole remainder, and never printed signed URLs. Every
90 MiB part reached the exact byte total `94,371,840`; part 07 reached its exact smaller
total. The official Kaggle CLI then finalized the dataset records, and all eight were
independently queried as `ready`.

Private kernel `francisbatiancela/slt-v17-movinet-end-to-end` version 1 was pushed with
explicit accelerator `NvidiaTeslaT4` and a 43,200-second limit. At launch, Kaggle
reported `KernelWorkerStatus.RUNNING`; the later 09:25 CLI audit above supersedes that
transient status and proves the run failed before training. The attached inputs remain
train/validation-only; the runner verifies every part SHA-256, reconstructs and verifies
the original archive SHA-256, requires CUDA, runs the fixed 5-epoch warmup plus 35-epoch
joint fine-tune, and rejects any result that claims test evaluation.

## 2026-08-10 00:39 PST — Kaggle multipart upload resumed

The original single-stream Kaggle upload was stopped after sustained throttling. The
exact 696 MiB archive was split byte-for-byte into eight private 90 MiB-or-smaller
parts, each with its own recorded SHA-256. Concatenating the local parts reproduces the
original archive SHA-256 exactly. The Kaggle runner and kernel metadata now attach all
eight private datasets, verify every part, reconstruct the archive, and verify the
whole-archive checksum before extraction. This is a transport change only; crops were
not resized, recompressed, or otherwise changed.

Eight parallel uploads reached substantial progress, but the local upload processes
were closed when the active tool call was interrupted by a user status message before
Kaggle created the dataset records. Kaggle showed no completed datasets at that point.
All eight official resumable uploads were immediately restarted. Kaggle confirmed
stored offsets for the restarted streams (for example, part 01 resumed after 27,787,263
bytes and part 07 had approximately 31.4 MiB remaining). Uploads are active again. The
CUDA kernel has **not** launched yet and no training result exists; launch remains gated
on all eight private dataset records becoming queryable.

## 2026-08-10 00:04 PST — Kaggle CUDA runner configured; upload in progress

The official Kaggle CLI 2.2.4 is installed in the isolated Python 3.11 environment
`artifacts/generated/kaggle_cli_env`, and OAuth authentication completed as Kaggle user
`francisbatiancela`. No API token was copied into the repository. A private Kaggle
dataset is being created as `francisbatiancela/slt-v17-movinet-trainval`, and the
private GPU script is configured as
`francisbatiancela/slt-v17-movinet-end-to-end` on an NVIDIA T4.

The upload bundle is
`artifacts/generated/kaggle_movinet_v17/dataset/movinet_v17_trainval.tar`, size 696 MiB,
SHA-256 `699834265f70ae6226b4692a0058b7c1ef2ea325d935941bbdf608af2b9c8bab`.
It contains only the train/validation RGB crops, train/validation Apple feature caches,
official MoViNet-A0 checkpoint, and the exact runner dependencies/code. An archive
listing audit found no `test/` directory or `landmark_test` cache. The private dataset
metadata uses license `other` and explicitly preserves the original ASL Citizen
research/noncommercial terms; the bundle is not published or relicensed.

The fail-closed cloud entrypoint is `active/v17/kaggle_movinet_runner_v17.py`. It verifies
the bundle checksum, rejects unsafe tar paths, installs the pinned CUDA environment,
requires a real CUDA TensorFlow device through the trainer, runs the fixed 5-epoch
head warmup plus 35-epoch joint fine-tune with batch size 4/patience 8, and refuses a
result whose manifest says the test split was evaluated. The Kaggle upload is currently
in progress; this section does not yet establish that the GPU kernel launched or that
full training completed.

## 2026-08-09 18:11 PST — v17 unit and real-Vision regressions

Command:

```bash
venv/bin/python -m unittest test.test_v17_extractor -v
```

Result: 10 tests passed. Coverage includes portrait/landscape isotropic identity,
independent per-joint gap filling, exact zeros for missing nodes, binary masks after
resampling, correct known chirality, temporal assignment for unknown chirality, exact
rotate/unrotate and mirror/unmirror transforms, aspect-preserving 2592x1944 capping,
real Apple Vision feature equivalence across transformed inputs, face detection, and
schema-enforced save/load.

## 2026-08-09 18:15 PST — reproducible v17 archive audit

Command:

```bash
venv/bin/python active/v17/audit_v17.py data/local/ios100_audit/landmarks_v17
```

Result: PASS. All 72 archives loaded with the current schema and passed shape, finite
value, binary-presence, missing-spatial-zero, and missing-confidence-zero invariants.
There were no schema/load errors. Median extraction time was 0.3911 seconds/video;
median detected hand-frame fraction improved from 0.4538 before activity trimming to
0.8806 afterward. Median hand/face/body presence was 0.4277/0.7812/0.4922. Shoulder
normalization was available for 64 videos and palm fallback for 8. Corrected chirality
counts were 1,145 left, 1,825 right, and 0 unknown. Outputs:
`artifacts/reports/V17_EXTRACTOR_AUDIT.md` and
`artifacts/reports/v17_extractor_audit.csv`.

## 2026-08-09 18:16 PST — token-efficient operating guide

`AGENTS.md` now contains the mandatory two-file startup rule, targeted repository map,
focused validation commands, dirty-worktree protection, and current v17/PopSign hard
gates. Project facts remain solely canonical in this file to prevent duplicated truth.

## 2026-08-09 19:47 PST — v17 Stage 1 baseline and landmark-quality gate

`active/v17/model_v17.py` and `train_stage_1_v17.py` now provide a clean Stage 1 path.
The model consumes only the archived `[B, 32, 61, 5]` contract and derives masked XYZ
velocity/acceleration plus valid hand-shape distances internally. Derivatives never
cross a missing observation. The loader uses the official split directories and the
explicit rejection ledger; effective counts are 1,475 train, 378 validation, and 1,247
untouched test archives. Augmentation preserves isotropic geometry and missing zeros,
and reflection swaps anatomical hands, face pairs, shoulders, and elbows. There is no
distillation, aspect stretching, or random video split.

The first capacity choice is d=256/depth=4 (6,470,885 parameters), not d=384/depth=6
(21,154,853 parameters). The older v16 capacity check favored 256, and 21M parameters
is unjustified for only 1,475 training clips. Capacity remains configurable for a
controlled validation ablation.

An initial training attempt revealed that fixed EMA decay 0.999 left validation
dominated by random initialization because there are only about 24 optimizer steps per
epoch. The run was stopped at epoch 10 and preserved under
`artifacts/generated/v17_stage1_aborted_ema999/`. EMA now has a step-count warm start;
the focused regression verifies its first update follows learned weights. A five-epoch
preflight improved from 2.65% to 16.67% validation top-1 under otherwise comparable
conditions.

The corrected baseline early-stopped after 74 epochs. Its best checkpoint is epoch 44
at `artifacts/models/stage1_v17_baseline/best_model.pth`: 93.12% top-1, 99.47% top-5,
and 92.53% macro F1 on 378 clips from the five official validation signers. This is a
validation result, not a final test or portrait-iPhone result. The Citizen test split
has not been evaluated. `evaluate_stage_1_v17.py` requires an explicit `--allow-test`
gate for test access.

The new missingness audit measured all 3,101 archives. Hand-active output-frame
coverage is min/p10/median/p90/max 28.12/81.25/87.50/93.75/100%; zero clips fall below
25%. When a side is active, median joint completeness is 91.67% left and 95.59% right.
Median observed-point confidence is 0.5889. Face-node presence is 78.92%; body/elbow
coverage is lower, but those nodes are auxiliary and explicitly masked.

Visual Apple Vision overlays on the lowest-, median-, and highest-coverage clips show
the detected 21-point hands aligned to visible fingers, including two-handed examples.
The lowest clip, validation SLEEP at 28.12%, is mostly idle with only a short visible
sign; it was nevertheless classified correctly. Validation accuracy by archived
hand-active coverage was 77.78% for the nine clips below 50%, 100% for six clips at
50–75%, 93.33% for 330 clips at 75–90%, and 93.94% for 33 clips at 90%+. The low bin is
small but confirms the coverage tail deserves review; it does not make training
useless. No fake tracking was reintroduced.

Evidence:

- `artifacts/reports/CITIZEN100_V17_LANDMARK_QUALITY.md`
- `artifacts/reports/citizen100_v17_landmark_quality.csv`
- `artifacts/generated/v17_diagnostics/citizen_landmark_overlay_audit.jpg`
- `artifacts/reports/stage1_v17_validation/REPORT.md`
- `artifacts/reports/stage1_v17_validation/predictions.csv`

Focused validation now totals 24 passing tests: the previous 17 plus seven Stage 1
dataset, missing-motion, full-anatomy mirror, augmentation, model-forward, real-count,
and EMA warm-start checks. The MPS optimizer/checkpoint smoke, full validation
evaluation, Python compilation, compatibility CLI help, and scoped whitespace check
also pass.

## 2026-08-09 19:50 PST — mobile algorithm alternatives reviewed

Current primary-source findings:

- Google MediaPipe Hand Landmarker has an official iOS live/video implementation using
  `MediaPipeTasksVision`. It outputs 21 image XYZ landmarks, 21 world-coordinate XYZ
  landmarks, and handedness, and uses tracking in video/live modes to reduce repeated
  palm detection. Google's published full-model Pixel 6 latency is 17.12 ms CPU and
  12.27 ms GPU. This makes it the highest-priority Apple Vision extractor challenger.
- RTMPose-s reports 70+ FPS on Snapdragon 865 for COCO body pose. MMPose also publishes
  an RTMPose-m hand model paired with SSDLite MobileNetV2. It is credible but its hand
  model/mobile iPhone cost is not established by the body benchmark, so it is the
  second extractor challenger rather than an assumed upgrade.
- MoViNet was designed for streaming mobile video with constant-memory stream buffers.
  It is the best direct RGB-video reference. Apple MobileOne reports sub-1-ms backbone
  inference on iPhone 12 for some variants and has the cleaner Core ML path, so a
  MobileOne frame encoder plus small temporal head is the preferred first RGB prototype.
- The existing Squeezeformer family remains competitive for compact temporal landmark
  modeling and has a direct PyTorch-to-Core-ML path. No reviewed evidence currently
  justifies replacing it with a generic Vision Transformer, ST-GCN, or YOLO classifier
  before controlled validation.

Sources:

- `https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/ios`
- `https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker`
- `https://arxiv.org/abs/2303.07399`
- `https://github.com/open-mmlab/mmpose/blob/main/docs/en/user_guides/inference.md`
- `https://research.google/pubs/movinets-mobile-video-networks-for-efficient-video-recognition/`
- `https://machinelearning.apple.com/research/mobileone`
- `https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html`

## 2026-08-09 20:00 PST — MediaPipe extractor challenger implemented

The first controlled challenger is the official MediaPipe Hand Landmarker full float16
task bundle, stored at
`artifacts/model_assets/mediapipe/hand_landmarker.task`. It is 7,819,105 bytes and its
SHA-256 is
`fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.
The source URL is
`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`.
The model binary is part of the feature-schema fingerprint, so features from another
model or threshold configuration cannot be silently mixed.

`active/v17/extract_mediapipe_v17.py` now implements an orientation-safe hybrid
candidate: MediaPipe supplies both hands on every sampled frame and Apple Vision may
supply low-rate body/face auxiliary points without redundantly running its hand request.
Video tracking state is recreated for every clip. MediaPipe world-coordinate Z is
wrist-centered, normalized by world-coordinate palm length, interpolated only across
bounded short gaps, and used only where genuinely observed. It is never extrapolated.
The existing scale-depth proxy remains an explicit fallback elsewhere. This behavior is
implemented by `interpolate_scalar_short_gaps` in `active/v17/geometry_v17.py` and is
covered by tests. Apple archives and behavior remain unchanged.

The candidate has its own contract in `active/v17/schema_mediapipe_v17.py`, named
`slt_mediapipe_hand_apple_aux_v17`, and remains shape-compatible at `[32, 61, 5]` while
being fingerprint-incompatible with Apple archives. MediaPipe Tasks handedness agreed
with corrected Apple anatomical labels on unmirrored Citizen frames, so the Tasks API
labels are not swapped in this pipeline.

Real evidence collected before the bakeoff:

- Citizen validation MAKE: MediaPipe hybrid took 1.4698 seconds versus 0.9364 seconds
  for the existing Apple archive on this M4 host. Both covered every trimmed output
  frame; MediaPipe hand-node presence was 68.75% versus Apple's 67.58%. MediaPipe
  provided genuine world depth for 69.17% of hand-node slots. This single clip is only
  a smoke result, not an extractor verdict.
- A real 16-frame HELLO test was bit-exact after a physical 90-degree pixel rotation
  followed by canonicalization, and after horizontal mirroring followed by anatomical
  unmirroring (`max_abs_diff = 0.0` for both). Genuine world depth was nonzero.
- Three new MediaPipe tests pass: bounded scalar depth interpolation, real
  rotate/mirror equivalence plus missing-value invariants, and schema save/load plus
  mismatched-configuration rejection. The 11 existing v17 Apple extractor tests also
  still pass after the common extraction changes.

No winner is declared from these smoke tests. The official Citizen test split remains
sealed. The next gate is a deterministic 100-class train/validation quality bakeoff;
full MediaPipe extraction and classifier training are allowed only if that evidence is
competitive with Apple Vision.

## 2026-08-09 21:02 PST — full MediaPipe 0.50 train/validation corpus ready

The 300 fingerprint-validated bakeoff archives seeded the full corpus, and resume-safe
extraction processed every remaining official train/validation raw video under schema
`69ae032129a68974`. The test split was not extracted or read.

- Train raw: 1,476. Result: 1,274 newly extracted, 200 validated seeds, two no-hand
  clips, zero failures. One no-hand clip is the already rejected malformed SLEEP clip
  `10667360637258794-SLEEP.mp4`; the only additional omission versus Apple is
  `train/BAD/008134541577476506-BAD.mp4`.
- Validation raw: 378. Result: 278 newly extracted, 100 validated seeds, zero no-hand
  clips, zero failures.
- Effective schema-checked loader counts after applying the existing rejection ledger:
  1,474 train and 378 validation. All 100 classes remain represented; per-class counts
  are 12–18 train and 3–6 validation.

`artifacts/reports/CITIZEN100_V17_MEDIAPIPE_T50_QUALITY.md` audits all 1,852 archives.
Hand-active output-frame coverage is min/p10/median/p90/max
15.62/75.00/87.50/87.50/100%; one clip is below 25%. Pre-trim source hand detection
is 8.93/26.67/41.38/54.55/100%; post-trim is
17.24/73.33/85.98/90.70/100%. MediaPipe emits all 21 joints as a hand or none, so its
reported within-hand completeness is 100% by construction and must not be mistaken for
per-joint ground-truth accuracy. A one-epoch MPS smoke used the exact MediaPipe schema,
loaded 200/100 balanced train/validation samples, completed forward/backward/checkpoint
creation, and kept test evaluation disabled.

The next active gate is an equal Stage 1 run with the same baseline seed, architecture,
augmentations, optimizer, schedule, EMA, and early stopping. Apple remains the selected
extractor unless MediaPipe produces a material signer-disjoint validation advantage
large enough to offset its slower runtime and observed overlapping-hand/false-positive
risks.

## 2026-08-09 21:38 PST — MobileCLIP2-S0 challenger environment verified

The RGB challenger uses the official OpenCLIP model name `MobileCLIP2-S0` and
pretrained tag `dfndr2b`. It is isolated from the validated Apple Vision environment
under `artifacts/generated/mobileclip2_env` with Python 3.10.20, PyTorch 2.13.0,
torchvision 0.28.0, OpenCLIP 3.3.0, and timm 1.0.28. The exact official checkpoint is
cached under `artifacts/model_assets/huggingface`; its SHA-256 is
`ab91a1a0c4330d6b1913e24d5035dfdea15423316aaec649610c6b1c6ddd0e95` and the full
image-plus-text checkpoint occupies approximately 300 MiB. Only the image tower is
part of the Stage 1 experiment and eventual mobile runtime.

The loaded image tower has 11,406,976 parameters, accepts 256x256 RGB input, and emits
a 512-dimensional embedding. A finite CPU forward pass succeeded. An MPS batch-16
smoke measured 8.39 ms per frame on this M4 host after warmup; this is a desktop
development measurement, not Android or iPhone evidence. The official preprocessing is
resize to 256, center crop to 256, tensor conversion, and mean-zero/unit-standard-
deviation normalization. v17 will letterbox each upright full frame to a square before
that transform so landscape hands are not discarded by the center crop.

MobileCLIP2-S0 is architecturally usable on Android after exporting and validating the
image tower through ONNX Runtime Mobile or LiteRT/TFLite, but Apple does not publish a
ready-made Android app/package in the official repository. Android support and speed
must therefore remain unproven until conversion, operator compatibility, numerical
parity, and NNAPI/GPU/CPU benchmarks run on target phones. Apple's published iPhone
latency must not be projected onto Android.

## 2026-08-09 21:58 PST — RGB conclusion narrowed; hand-aware route identified

The user correctly challenged treating a frozen image encoder like a landmark feature
extractor. MobileCLIP2 has no native sign-video classifier: its published objective is
image-text contrastive learning. The completed run froze the visual tower, reduced each
full frame to one globally pooled 512-D vector, and only then modeled time. A temporal
head cannot reconstruct finger articulation or local spatial layout already removed by
global pooling. The head was large enough to drive training loss near its
label-smoothed floor, so the primary failure is representation/domain generalization,
not evidence that Squeezeformer was too small.

The next legitimate RGB experiment, if pursued, is a new hand-aware schema rather than
a replacement temporal head on the old archives:

1. Use Apple Vision only to provide reliable left/right/union hand crop boxes and the
   existing landmark trajectory. Crop actual RGB pixels at high resolution; missing
   detections remain explicitly masked. Short box stabilization may select pixels but
   must never synthesize landmarks or RGB content.
2. Feed shared-weight 224–256-pixel left/right hand crops (and an overlap-aware union
   crop when hands contact) through the MobileCLIP2 visual trunk. Retain spatial feature
   maps or fine-tune the late visual stages instead of freezing and keeping only the
   global image embedding.
3. Add temporal interaction inside the visual backbone using an efficient video method
   such as Temporal Shift Module, or benchmark MoViNet-A0 on the hand-crop sequence.
   Fuse hand appearance with the frozen Apple landmark representation through a small
   gated/cross-attention residual so landmarks retain absolute position and trajectory.
4. Train signer-invariant features with class cross-entropy plus supervised contrastive
   sampling across different signers; use strong background/color/appearance
   augmentation without geometrically corrupting the hands. Fine-tune late stages first
   with a lower backbone learning rate, then unfreeze only if validation supports it.
5. Compare predeclared variants on the same sealed train/validation contract: Apple
   only; frozen hand crops; fine-tuned hand crops with temporal interaction; and
   Apple-plus-hand fusion. Test remains sealed. The RGB branch earns runtime inclusion
   only if it provides a net validation gain and later survives phone profiling.

This direction is supported by sign-specific evidence rather than analogy to ordinary
action recognition. De Coster et al. reported 82.03% validation accuracy for full-frame
VTN, 90.13% after high-resolution hand cropping, and 91.51% after restoring pose-flow
motion. SignRep (ICCV 2025) explicitly addresses the weakness of general visual
pretraining with skeleton-guided sign-specific masked pretraining of a 16-frame Hiera
video model; it is an accuracy reference, not currently a low-end-phone candidate.
Multi-stream sign work likewise reports that local hand/face RGB plus skeleton streams
improve WLASL/MS-ASL recognition. TSM and MoViNet are the relevant mobile temporal
families: TSM introduces feature-level temporal exchange without extra arithmetic or
parameters, while MoViNet is designed for memory-bounded streaming video.

Primary sources:
`https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/De_Coster_Isolated_Sign_Recognition_From_RGB_Video_Using_Pose_Flow_and_CVPRW_2021_paper.html`,
`https://openaccess.thecvf.com/content/ICCV2025/html/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.html`,
`https://arxiv.org/abs/2106.15989`,
`https://openaccess.thecvf.com/content_ICCV_2019/papers/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.pdf`, and
`https://openaccess.thecvf.com/content/CVPR2021/html/Kondratyuk_MoViNets_Mobile_Video_Networks_for_Efficient_Video_Recognition_CVPR_2021_paper.html`.

## 2026-08-09 22:37 PST — Spatial experiment implementation validation

The pre-pooling path, fusion feature exporter, and zero-initialized gated fusion trainer
compile successfully with all v17 modules. A seventh focused hand/RGB unit test now
proves bit-exactly that fusion logits equal the frozen Apple logits before optimization;
all seven focused tests pass. `active/v17/README.md` was corrected to reject only the
measured frozen full-frame/global-pooling design, not MobileCLIP2 or RGB generally. The
full training spatial-map extraction remains in progress and had written 960/1,475
training clips without skips or failures at this timestamp. The test split remains
sealed.

## 2026-08-09 23:12 PST — Hard temporal-shift result

The optimizer/checkpoint smoke passed at batch sizes 8, 16, and 32; batch 16 was used
for the single full run as the fastest safe measured setting on the 24 GiB M4 host.
Training early-stopped after 29 epochs (15 stale). The saved epoch-14 checkpoint at
`artifacts/models/stage1_v17_hand_mobileclip2_spatial/best_model.pth` reached 70.63%
validation top-1, 88.89% top-5, and 69.41% macro F1. This is only +0.26 point over the
70.37% frozen high-resolution hand-crop model and remains -22.49 points behind the
93.12% Apple landmark model. Later epochs were unstable and did not improve top-1;
epoch 26 matched 70.37% with 91.53% top-5. The result shows that processing spatial
maps before pooling contains a small amount of additional sign signal, but hard TSM is
not a selected standalone branch. The checkpoint reports `test_evaluated=false`.

The next and final active gate is zero-initialized feature residual fusion over the
frozen Apple logits. It is accepted only if signer-disjoint validation exceeds the
Apple baseline; otherwise Apple landmarks remain the selected extractor/model. A
possible future RGB ablation is identity-initialized residual temporal mixing, because
hard TSM disrupts the frozen representation at initialization, but it is not required
to decide the present v17 extractor.

## 2026-08-09 23:15 PST — Conservative fusion result does not displace Apple

The first fusion feature export exposed and fixed a real broken interface: the Apple
model supports `return_embeddings=true`, not a `forward_features` method. The failed
attempt wrote no archive. The corrected aligned exports contain finite 256-D features,
identical item IDs/targets, and exactly reproduce 99.59%/93.1217% Apple train/validation
top-1 and 100%/70.6349% spatial-MobileCLIP train/validation top-1.

The zero-initialized gated feature residual is bit-exactly the frozen Apple logits at
initialization. Canonical seed 1701 reached 93.92% validation top-1, 98.94% top-5, and
93.46% macro F1 at epoch 9, versus Apple's 93.12%/99.47%/92.53%. It fixed nine Apple
errors but broke six Apple-correct clips: only +3/378 net. The exact two-sided paired
McNemar/binomial p-value is 0.607, so this is not statistically persuasive. Four
additional diagnostic seeds all exceeded Apple individually (93.65, 93.92, 94.18,
94.44%; five-seed mean 94.02, population SD 0.27), but averaging their logits returned
exactly to 93.12% and produced five fixes versus five regressions. This cancellation
shows that the small residual corrections are seed-sensitive despite favorable
per-seed validation checkpoint selection.

Therefore the fusion branch is retained as research evidence but is **not** the mobile
selection. Adding a 48-view MobileCLIP visual pass for a non-significant, seed-sensitive
net three-clip gain contradicts the low-end offline goal. Apple Vision landmarks plus
the v17 Squeezeformer remains the selected Stage-1 extractor/model at 93.12% signer-
disjoint validation top-1. The official test split remains sealed; none of the fusion
or robustness runs accessed it.

## 2026-08-09 23:19 PST — Final focused validation

All 34 `test_v17*.py` tests pass, including real Apple Vision rotation/mirror
equivalence, real MediaPipe orientation parity, schema separation, missing-value
contracts, real archive counts, hand-crop packing, TSM masking, temporal-model
forward/backward, and bit-exact fusion initialization. Three Citizen downloader tests,
three PopSign downloader tests, and four legacy aspect-correctness tests also pass: 44
focused tests total. All `active/v17`, `src_v17`, affected scripts, and v17 tests compile.
`git diff --check` passes. No hand-RGB, hand-embedding, or hand-spatial files exist under
their Citizen test output directories. Expected Python 3.9 EOL/LibreSSL and MediaPipe
delegate warnings occurred but caused no failures.

## 2026-08-09 23:48 PST — Joint-training correction and sign-specialized MoViNet started

The earlier term “five-seed ensemble” described only a robustness diagnostic. The
canonical MobileCLIP fusion did train one residual head on aligned Apple and RGB
features together, but both base encoders were frozen/cached; it was **not** end-to-end
co-adaptation. MobileCLIP is optimized for image-text semantic alignment, so being a
strong image encoder does not guarantee preservation of subtle finger configuration.
The hand-crop recovery from 39.68% to 70.37% proves that crop scale mattered, while the
remaining gap shows that frozen features still lost sign-specific detail. A true joint
pixel/landmark fine-tune remains a distinct experiment and must not be conflated with
the completed feature-residual result.

A separate MoViNet-A0 experiment is now implemented in
`train_stage_1_movinet_v17.py`; it is explicitly sign-specialized rather than a generic
full-frame Kinetics head. One shared pretrained video backbone processes anatomical
left-hand, right-hand, and union/context sequences from the real Apple-selected crop
corpus. It consumes explicit missing-view masks and all 16 normalized box trajectories,
uses view identity/attention, applies sign-safe mirror/temporal/photometric/view-drop
augmentation, and jointly trains a visual-only auxiliary classifier plus cross-modal
Apple/RGB fusion. The fusion residual is zero-initialized and was verified bit-exactly
equal to the frozen Apple logits before training. The official Citizen test is rejected
by the loader and was not accessed.

The official TensorFlow 2.16.1 / Model Garden 2.16.0 implementation and Kinetics-600
MoViNet-A0 checkpoint were installed in isolated Python 3.10 environment
`artifacts/generated/movinet_env`; the checkpoint archive SHA-256 is
`7bae6c7ef74e2ff4115ad51f1fdad8718247375b420da2e35e8ee7771ac35758`. The environment
is pinned by `movinet_requirements.txt`. TensorFlow Metal 1.2 cannot execute the
official XLA-compiled grouped/depthwise Conv3D graph (`registered platform` failure),
and removing the explicit wrapper still fails in Metal's grouped Conv3D operator.
CPU execution is therefore the reproducible local training path; this host limitation
does not imply a TFLite runtime failure.

Two pure data-contract tests pass: item-ID alignment/test rejection and exact mirrored
left/right involution with missing views remaining zero. The CPU optimizer/checkpoint
smoke then passed official-weight restoration, three-stream forward, two updates,
partial validation, save/reload, and test isolation. It built a 1,867,433-parameter
joint model with a 911,583-parameter MoViNet backbone. Its 87.5% fused result is only
7/8 smoke clips and the 0% visual result is after two updates; neither is accuracy
evidence. The full train/378-clip validation run has not completed. Local CPU throughput
must be benchmarked before committing to a many-hour run or moving it to CUDA/Linux.

## 2026-08-09 — v17 smoke result

The HELLO audit clip produced finite features, 40 observed hand frames, 0.8889 detected
hand-frame fraction after trimming, 0.4346 hand-node presence, 0.875 face presence, and
0.4219 body presence. Apple Vision reported 40 right-hand and 0 left-hand observations;
this specifically verifies the old v16 chirality reversal is corrected.
