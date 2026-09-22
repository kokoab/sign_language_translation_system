# live-streaming — log

## 2026-09-22 evening — six more negatives; coarticulation identified; selection was the flaw

Commit-gate rescue FAILED its gate (49.33->48.67% WER but +3 insertions; unfiltered verifier
56.67%). Blocking stages: proposal_rejected_first 42, verifier_after_proposal 18, commit_score 12 -
the commit score is NOT the main blocker. Rescue admits duplicates (HELLO HOW HOW HOW YOU) because
the boundary model over-proposes; the gate was compensating for that.

reel_context_adapt selected epoch 0. Deletions never moved (52->52). Isolated retention held, so
the recipe worked mechanically and nothing transferred. SCHOOL is systematically committed as STOP.

Crop-width sweep (89 clips): 100ms already optimal; 150/200/300/400ms give 52.65/53.10/57.96/64.60%.
SCHOOL 0% at every width. Crop is not cutting off evidence.

Hand/video diagnostics: local video IS darker (brightness 91.97 vs 118.45, 27.2% vs 17.3% pixels
<40) but hand detection is BETTER (both-hands 39.5% vs 13.9%, confidence 0.625 vs 0.499). Data
quality does not explain the failures.

Synthetic phrases built (scripts/build_synthetic_phrases_v17.py, 139 phrases/39 signers/333 signs,
within-signer, user-authorized over the Citizen caution). Validity: 18.62% WER, SCHOOL 82% - EASIER
than the isolated clips it came from. With segmentation perfect in both, synthetic 92% verifier vs
real continuous 67%. COARTICULATION POSITIVELY IDENTIFIED as the mechanism. No model trained on it.

CORRECTION 1: "only 44 continuous phrases" was wrong - that is asllrp_contiguous only. Real
inventory 1,017 known events in continuous video, 65 glosses, 147 videos with >=2 known signs, plus
ncslgr_strict 88/37.
CORRECTION 2: the -0.73 correlation between ASLLRP continuous coverage and local recall is
CONFOUNDED by sign complexity and REFUTED by the existing ablation - removing asllrp_other_ctc costs
7 WER points (42.02->49.02%) and 12 exact phrases. PLEASE has 0 ASLLRP events and still scores 50%.

Prior art found: 2026-08-16 stage1-architecture/high.md already rejected synthetic window-phase
augmentation (v3, 54.17% WER) and recorded that the limitation is scarce genuine continuous phrase
supervision. Today re-derived that independently.

Multi-source weighted selection built (scripts/score_multisource_v17.py): local 0.40, other three
0.20 each, weighted macro not pooled, isolated as a retention constraint not an aggregate term,
per-source no-regression gates. Local is the EASIEST continuous source (37.78% WER) not the hardest;
ncslgr 86% and asllrp_other 77% hold the headroom. Ranking 26 checkpoints: the reference
unified_streaming_grounded_ctc_v17_v1 ranks 14th; unified_streaming_aligned_grounded_v17_v1 PASSES
every gate at 53.96% weighted, -3.01 points, isolated 82.4% vs 81.6%. Note 3-source checkpoints
renormalize over sources excluding ncslgr and are not comparable.

Next safe action: switch to aligned_grounded_v17_v1; adopt score_multisource_v17.py as the gate;
boundary = frozen backbone + ORIGINAL BIO head/decoder, drop the START/END edge head (14 WER points
at identical weights). 25% is not reachable by segmentation - oracle boundaries still give 50% WER
on ASLLRP12 - it needs target-domain continuous video from more than three signers. Full writeup
artifacts/reports/boundary_investigation_summary_v17_20260922/FINDINGS_PART2.md.


## 2026-09-22 — Reel context head adaptation completed

Finished 14epochs in 0.04min fitting. Selected epoch0; no promotion. Baseline WER49.33%, best trained 51.33%. Full retention/isolated validation gates are in history.json; best trained does not necessarily qualify. Report artifacts/reports/reel_context_adapt_v17_20260922/REPORT.md. Next read report/confirmation before any deployment or further unfreezing.


## 2026-09-22 — Reel preparation/automatic fitting launched

PreparationPID42156; detached native-kqueue supervisorPID46990
waits for process exit without polling, then invokes --train only after cache_manifest
exists and no failure.json. Completion/failure notifications enabled. Exact frozen
verifier landmark/hand tensor lineage check passes; focused independent review found
no blockers. Unit,syntax,diffchecks pass. Preparation measured75/555videos in3min;
estimate20–30min totalpreparation, fitting under5min from measuredheadbenchmark.
Report is explicitly RUNNING, no improvement claim. Automatic completion records
results in currentstate/log and updates artifact index. No defaultchanges.


## 2026-09-22 — authorized Reel context adaptation, preparation started

Added scripts/train_reel_context_adapt_v17.py, one focused runnable regression check,
and artifacts/reports/reel_context_adapt_v17_20260922/PLAN.md. Recipe pins672 reviewed
ASLLRP positive events (47contiguous/625OTHER), occurrence-deduplicated with original
parent/signer roles; OTHER names the source crop, not a training class. Equal local
phrase partitions excluded. O5S5 exact-core-only/unspecified completeness is not silently
reclassified as complete contextual supervision; STEM and other data preserved.
Ruling: heads-only first bounded phase, both proposal classifier and verifier fusion;
all image/temporal encoders frozen, no automatic block-unfreezing or architecture change.
Exact Citizen1475/SemLex1388 replay cache membership,378/978validation retention planned.
89local evaluation clips excluded; frozen BIO saved windows and unchanged acceptance.
Canonical494 verifier passes with generictraining_ready=false unchanged. One test passed
(crop-neighbour overlap and recognition retention guard). Actual runtime first-window
CoreML/PyTorch logit parity passes. CPU128head batch3.6ms measured; fitting projected
under5minutes, preparation separate. Preparation is running, no newtraining or promotion
yet. User permits training polling only if under5minutes. Next: full baseline parity,
source-cache checks and review, then bounded fitting if all pass.



## 2026-09-22 — fixed verifier-aware rejection replay, no live change

User requested tackling rejection losses and reports. New diagnostic-only
artifacts/reports/reel_rejection_replay_v17_20260922/{replay.py,policy.json,calibration.json,
unfiltered_verifier_diagnostic.json,REPORT.md}. Fixed policy before scoring: preserve
all originalcommits; rescue with finalverifier>=.8, actualtop2margin>=.08; bypass only
low_score/low_margin, retain all other rejection safeguards. No threshold sweep,
modelinference/training or productchanges. Reconstructed currentVerifiedCommitLock
with originaldefaults/hits1 reproduces every savedintervalcommit and editcount exactly.
Policyguard assertions and finalsource/codehashchecks pass.

59clips150sign calibration baseline82correct/16S52D6I/49.33%WER. Candidate86correct/
16S48D9I/48.67%,retains82/82,14exactunchanged.7rescues =>4more transcriptcorrect and
3moreinsertions; fails predeclared noextraI guard. Confirmation not evaluated (source
hashed only). Firstblocker72rejections:proposal42,verifierafterproposal18,commitafter
bothaccepted12; not recoverable-correctcounts. Unfilteredverifierdiagnosticonly102correct/
28S20D37I/56.67%,not oracle/deployableupperbound. No annotation gapclaim.
Decision retainpolicy; next discuss continuousidentity/context adaptation rather than
blanketthresholdlowering. Doesnotruleoutallothergatepolicies. Existingdatasets/splits,
checkpoints and live defaults preserved. Individualchangedcases included inREPORT.


## 2026-09-22 — final BIO completed; investigation synthesis corrected

User requested FINDINGS.md review and nextdirection. Completion1081.75s/18.03min,
12epochs,selectedepoch0 frozen. Epoch8calWER47.33% vs49.33%,correct84vs82,retains74/82;
failedretentionguard,not proof of zerolearning. Confirmation selected=frozen46/76,43.42%.
No new training/deployment. Findings read and checked against probe/code/CTClog.
Citizenprobe57clips,55verifiercorrect/33correctcommits,5distinctsigners,not19pergloss.
All193probe clips34signers. 232/232train30/199val belongs to cleanStage2CTC17321/17322,
not demonstrated memorization of currentReel weights. Frozenrandomedgehead precision
is not frozenBIOquality; selectedepoch7precision cannot prove trainingprecisionceiling.
Boundarynever-bottleneck is unsupported: annotationoracle12commits vs6predictedASLLRP.
Moreproposals thanrefs doesnot prove sufficient correctlyaligned intervals.

Concrete decisionpath: live_isolated acceptance tests gate_score/gate_margin, which
can usefastlandmark evidence even when finalverifiermodel_score ishigh; classify_interval
requires proposal.accepted AND verifier.accepted before VerifiedCommitLock. Withhits1,
instant_score is redundant afterminimumscorepass; sweepinghits inone-decisioninterval
evaluation doesn't model live repeatedstability. Existing59cal frozenrejections: verifier
low_score23/low_margin18/insufficienthands9;proposal low_score28/low_margin27/
insufficienthands9;12blockedafterbothaccepted. Categoriesoverlap,not correctsigncounts.

Nextproposedwork: freezeBIO andReelweights; replay existingcalibration decisions with
explicit rejectionattribution and a small verification-aware policy change, score retained
corrects plus substitutions/insertions on wholevideos, then validate actualapp scheduling.
No blanketthresholdlowering. If verifier alreadywrong on reliableintervals, target continuous
identity/context adaptation rather than another boundaryrun. Preserve currentdatasets/splits.

## 2026-09-22 — isolated signer probe: commit gate, not signer diversity, is the largest loss

Read-only diagnostic, no training/promotion/protected test. Ran frozen Reel over 193 held-out
(role=validation) isolated citizen/semlex/stem clips, 34 signers, covering exactly the 15-gloss local
phrase vocabulary; whole clip as one interval through the same classify_interval path as the phrase
evaluation. Script kept in session scratchpad; results
artifacts/reports/isolated_signer_probe_v17_20260922/probe.json.

By source: citizen n=57 proposal 96%/verifier 96%/committed-correct 58%; semlex n=133 52%/54%/45%;
stem n=3 all zero. Overall n=193: 64%/66%/48%.

Findings. (1) Signer diversity is NOT the binding constraint - 96% across ~19 signers per gloss on
citizen. The 88 unused isolated-dataset signers would add variation the model already handles.
(2) The commit gate is the largest single measured loss: citizen verifier 96% correct but 58%
committed; across all sources 34 of 127 correctly-verified clips (27%) never emitted, median verifier
score 0.908, and only 5 of those 34 had verifier_accepted=True - the accept criterion rejects correct
~0.9-confidence answers. Same pattern as 51-of-79 blocked intervals on local60, now shown where
segmentation plays no part. (3) SCHOOL 67% across 18 signers isolated vs 0/6 in local phrases;
FRIEND 67% vs 1/6 - context transfer, not sign or signer. (4) Inversion rules out sign difficulty:
GOOD 37% isolated but 6/6 in phrases. (5) SemLex 52% vs citizen 96% is a domain gap; semlex is 33 of
the 88 signers.

Decision: do NOT build the synthetic phrase-concatenation pipeline as the next step; it is a
multi-hour build against a non-binding variable. External acquisition (YouTube-ASL ~2500 signers
video-ID list, OpenASL CC BY-NC-ND) deferred for the same reason; neither carries gloss boundaries.
Caveats: 193 clips; stem n=3 and HOW n=4 uninterpretable; citizen figure rests on 57 clips; Reel
decides over the full 100-gloss vocabulary.

Next safe action: sweep commit_score/instant_commit_score/commit_hits and inspect the verifier accept
criterion, then test classifier context beyond 100ms. Consolidated writeup
artifacts/reports/boundary_investigation_summary_v17_20260922/FINDINGS.md.



## 2026-09-22 — authorized final planned BIO adaptation launched, no polling

User authorized BIO-preserving training and explicitly forbade polling. Added dedicated
scripts/train_pretrained_bio_final_v17.py,test/test_pretrained_bio_final_v17.py, versioned
active/v17/pretrained_bio_final_recipe_20260922.json and reportPLAN/targets/evalposepins.
Generic phrase verifier remains training_ready=false; dedicated boundary contract only.
Only originalBIOhead +lastattention block train (1,330,948params); CNN/norm/first3blocks
frozen and cached. Originalpretrained initialization, seed17621. Train3019B/21065I/5930
masked windows; Bfirstobserved inside, Ithroughinclusiveend; overlaps/unknown=-1, no
inventedO. ClassweightedCE +2xKL to frozenBIO on same trainingwindow/augmentation; KL
is a prior,not annotated background. Existing clean+2poseaugmentationvariants reused.
59localcalibration selects via fullvideoWER/retention/insertion guards every4epochs;
30confirmation once afterlockedselection. Both reused development,not fresh independent
test. Expanded72 untouched; no training overlap. Max80,12epochswithouteligiblegain stop;
scheduler initialized withbaselineWER so epoch8reduction gets time before12stop.
Selectedepoch0frozen fallback preserved alongside best_trained actualcandidate. No auto
promotion or claim next model must improve. Existing89local reservations preserved.

25focusedtests pass; tests first failed missingnewmodule. Fulltarget/source/hash audit,
realraw->existingCNNcache parity, first3prefix/fullpath parity and finiteMPSbackward pass,
zero preflightoptimizersteps. Independent narrowreview found no blockers. Diffcheckpass.
RecipeSHA256da9fa22ac2c0c84d1cfe0de6b2553dc7d020bb747bf5a98fa09bd836995e6453.
Detached caffeinatePID32926 launched2026-09-22T11:49:17Z. Do not poll. Notification on
completion/failure; reports at artifacts/reports/pretrained_bio_final_v17_20260922/.
Estimated1–2hours remains planningestimate, potentially longer if improvement continues
toward80; single preflightbackward timing is not steady training benchmark. Launch is
not completion. Next read completion/selection/confirmation/REPORT on user update.


## 2026-09-22 — local BIO calibration completed, no improvement

User authorized fast bounded calibration/report. New diagnostic run.py reuses original
BIO and frozenReel;59calibration/150signs,30confirmation/76signs,282unused candidates.
15signer/phrase strata; exacthash disjoint from boundaryfit/expanded72. Video/feature
hashes verified. Familiar3signers/6templates/15glosses; no session/near-duplicate guarantee.
Reserve89clips from future boundaryfit; original roles unchanged. Four predeclared
frozen/adapted x min3/min5 options. Duration variants reuse intervalpredictions.
CalibrationWER49.33/56.00/53.33/57.33percent, correct82/70/81/73 of150, insertions6/4/11/9.
No challenger qualified under lowerWER/nondecreasingcorrect/noextraI/98percentretention.
Frozenmin3 selected before confirmation; confirmation baselineonly43.42%WER,46/76,
3insertions,8of30exact. No challenger test/improvement on confirmation claimed.
Elapsed669.06s/11.15min; no training/deployment. Selection/filter/split checks and final
sourcehash checks pass. Report/manifest/selection/predictions/by_signer under
artifacts/reports/boundary_local_calibration_v17_20260922/. Postreport helper initially
used systemPython lackingcv2, rerun with requiredvenv; inference/results unaffected.
Next discuss BIO-preserving supervision/adaptation; longer minimumsegments lose signs.


## 2026-09-22 — existing local calibration candidates clarified

User challenged two-clip calibration scarcity. Targeted manifest join confirms371
local phrase candidate clips/1006reference signs outside expanded72 and existing
boundary train/calibration hashes. Existing authorized familiar split assigns371train/
60heldout:232originaltrain plus139movedlocalvalidation. Candidates cover3signers
(01=167,02=139,03=65), only6unique phrase sequences. Two complete clips describes the
old boundary calibration loader intersection, not all available local supervision.
No split changed/no fitting/no candidate selection. Inventory at diagnostic report
local_calibration_inventory.json; manifest-level counts, not fresh video/annotation audit.
Next proposal: verify/deduplicate candidates and version a recognition calibration +
separate confirmation partition before tuning BIO candidates. Keep72 untouched as reused
development, not fresh test; preserve originalBIO, score WER/retention/insertions.
Whole transcripts suffice for sequence metrics; timed boundary/gap metrics still need
reviewed intervals. Six phrase templates limit generalization claims.

## 2026-09-22 — controlled original-BIO swaps isolate adaptation regression

User authorized fast diagnosis and smallest justified fix after discussion. No new fitting,
deployment, distillation, data/split changes or old artifact overwrites. Same72videos/186signs,
same frozen Reel. Frozen127correct/39.78%WER/15insertions hypotheses AND interval boundaries
reproduced exactly. First adapted backbone + originalBIO: 130correct/40.32%WER/
19insertions, retains124/127. Augmented+BIO:
124correct/42.47%WER/17insertions, retains120/127.
CNN/inputnorm/BIO tensors unchanged exactly; attention and newedgehead are the only changed
tensors. Major regression is replacement readout/decoder, not wholesale representation loss;
head quality versus decoder policy (including EOF) remains unresolved.

Changed scripts/evaluate_boundary_expanded_v17.py: explicit readout='bio' independent of
checkpoint presence, old defaults preserved. Added scripts/diagnose_boundary_adaptation_v17.py
and OriginalBioReadoutTest in test/test_pretrained_boundary_v17.py. Diagnostic/audit/calibration/
benchmark/verification scripts and results in artifacts/reports/boundary_adaptation_diagnostic_v17_20260922/.
Reconstructed all42614cached targets with independent unknown/edge masks, zero unknown loss
gradients, sampled CPU/raw-cache parity <=5.723e-5; shared20Hz/64frames/target53/500msfuture
contract intact.23focusedtests pass; readout test first failed missingkeyword then passed.
Initial audit matching hit missing source_item_id on unrelated combined records; filtering
that field fixed audit setup only. No training/data changes needed.

Fixed-candidate calibration: only2approved complete locked-vocabulary videos/4signs among
131boundarycalibration records. Original/fine-tuned/augmented BIO all0correct/100%WER;
both adaptededge paths1correct/75%WER/0insertions. Predeclared rule chooses firstedgepath,
not a recoveredBIOcheckpoint. Expanded72never used to override it or tune thresholds/epochs.
This is not sufficient recognition-selection evidence; no candidate promoted, no newtraining.
Bounded experiment was fixed-weight readout comparison with zero gradientsteps.

Replay489.26s; calibration15.88s. Isolated synchronizedMPS
batch1 normalization/CNN/attention/BIO firstadapted median30.19ms,
p9538.97ms, excludesMediaPipe/Reel/camera and intrinsic500msfuture.
Per-stage whole-video timings in REPORT.md; notlive/iPhone evidence. Video/checkpoint/oldsource
hashes and all editcounts/retention recomputed. Next preserve originalBIO reference and resolve
complete recognition scoring inside existing calibration split before another fine-tune.



## 2026-09-22 — augmented completion and user-expanded72video TCN comparison

User requested completed run review, expanded-set confirmation and nativeTCN evaluation.
Augmented run complete2230.48s/37.17min,16epochs/selected8;cal.652261,val.818841.
Cache677.56s,training/validation1497.21s. OldASLLRP7correct versus9priorfine-tune,
so lowerBCE did not establish recognition improvement. No new training launched.

Read scripts/evaluate_boundary_expanded_v17.py and saved72video results; verified all
transcripts/editcounts/aggregateWER against currentsource records.72uniquevideo hashes,
186refs,60local_signer_02clips162refs. No hash overlap with either boundary training/
calibration or localfamiliar split train for heldlocal60. Clip-disjoint reused familiar
signer development,not unseen-signer. No manual relabeling of all videos claimed.
Found savedmembership only12 and mislabeledASLLRPcandidate counts246/261/287 vs19/21/18;
current evaluator source already restricts counts but savedJSON stale. WER unaffected.
Localzero gapcounts meanunavailableannotations,not measuredzerofalse transitions.

Added only artifacts/reports/boundary_expanded_eval_v17_20260922/tcn_comparison/
check.py,audit.json,evaluation_membership.json,evaluation.json,run.log,REPORT.md.
All4existingTCNs evaluated via actualBoundaryRecognizer.observe with sharedfrozenReel
classify_interval/.1context/no wristtrim; sameprovenance verified. Native.2sfuture/unscored
tail preserved vs pretrained.5s/partialEOF. Conditionalfullvideo,not async live scheduling.
All72videos/fourarms complete420.37s; finalsource/checkpoint/provenancehash checks pass.
Audit-only assertion initially compared tuples toJSONlists, correctedserialization; empty
partial subsets skipped. Model/product code and original suppliedreports unchanged.

Expandedcombined frozen127correct/39.78%WER/17exact/15I; firstfine-tune96/54.30%/5/11I;
augmented100/53.76%/2/14I. TCN skeleton17621:89/61.83%/6/18I; skeleton17622:78/66.13%/6/15I;
geometry17621:84/62.90%/3/15I;geometry17622:99/59.68%/7/24I. ASLLRP5/6/7/8correct exactly
reproduced. Geometrynotconsistentlybetter acrossseeds/local60. BestTCN retains80/127 frozen
correct; bothadaptedmodelsretain75/121frozencorrectlocal signs. No promotion/distillation.
Nextdiscussion: frozenpretrained remainsreference; isolate adaptedbackbone versus new
START/END head/decoder before further fitting. Architecturecapacity not provenbottleneck.
Update currentstate/index anddiffcheck; no protectedtest/newdata or seed2restart.

## 2026-09-22 — expanded held-out boundary evaluation; 24-sign ranking does not survive

User judged the 12-video/24-sign set too small to separate checkpoints. Added
scripts/evaluate_boundary_expanded_v17.py (new report dir
artifacts/reports/boundary_expanded_eval_v17_20260922); no training, no promotion,
prior reports/checkpoints untouched. Set = frozen 12 ASLLRP development videos (24 signs)
plus the 60 signer02 clips that local_familiar_signer split_manifest still holds out
(162 signs) = 186 reference signs. The 139 moved-to-train clips were excluded. Verified
both checkpoints were trained only on asllrp_other_ctc/asllrp/o5s5 poses, so no
local_phrases contamination of the boundary models; frozen Reel is the shared constant.
Subsets summarized separately; pooled column explicitly labelled.

Harness validated: ASLLRP12 reproduces the standalone runs exactly (WER 75.00/62.50/70.83,
recall@200ms .667/.708/.583).

local60 WER: frozen_pretrained_bio 34.57% (121/162 correct, 29 del, 17/60 exact);
finetune_epoch7 53.09% (87/162, 53 del, 4/60); augmented_epoch8 51.23% (93/162, 47 del, 2/60).
Combined186: frozen 39.78% (127), finetune 54.30% (96), augmented 53.76% (100).

Conclusions. The 62.50% vs 70.83% finetune/augmented gap measured on 24 signs does NOT
survive: on 186 signs they are 54.30% vs 53.76%, and the subset ordering flips. Do not
draw checkpoint conclusions from the 24-sign set. Separately, the un-adapted frozen
backbone beats both adapted checkpoints on local60 by ~18 points, but the arms use
different decoders (frozen sign_bio_head + upstream segment grouping with EOF close;
adapted edge head + BoundaryDecoder requiring an explicit END, 0.15-4.0s bounds). The
adapted deletion counts (53/47 vs 29) point at decoder failure to close on short clips,
but weights and decoder are not separable from this run.

Familiar-signer reused development pool, not unseen-signer and not protected test. One
seed per checkpoint; no variance estimate. Fixed a reporting defect after the run:
boundary_candidates accumulated across all 72 videos while printed under the ASLLRP
heading (recall unaffected, numerator and denominator were both ASLLRP-only).

Next safe action: separate decoder from weights by running the adapted checkpoints through
the BIO-style decoding path, or the frozen head through BoundaryDecoder, before concluding
that fine-tuning hurt. Seed 17622 remains unrun.



## 2026-09-22 — authorized augmented pretrained boundary fine-tune launched

User authorized next fine-tune, asked to eliminate mismatches and give runtime estimate.
Changed active/v17/pretrained_boundary_v17.py (past-only observation augmentation),
scripts/train_pretrained_boundary_v17.py (separate augmented run, verified clean-cache
reuse, two variants, 80max/patience8/no40floor, LR plateau patience3, real-cache benchmark),
scripts/evaluate_pretrained_boundary_v17.py and evaluate_temporal_boundary_v17.py
(explicit report path), test/test_pretrained_boundary_v17.py (clock/augmentation,
row-label alignment, untouched clean batching, stop and report-destination regressions).
Only one seed17621 from original pretrained weights; head5epochs then all4attention
blocks. Same20Hz/64frames/target53/500mslookahead, float32, masked unknowns, splits and
frozenReel. Observation15–20Hz/dropout0–15% before normalization, two training-only
cached variants plus clean variant, one choice per example per epoch. This is sensor
sampling augmentation, NOT signing-tempo warping. No re-extraction or new data.

Independent review found evaluation_membership would overwrite old temporal report;
fixed shared output parameter and pinned COMBINED/CURATED/baseline and trainer dependency.
COMBINED/CURATED hashes checked against training source manifest. First unlaunched
augmented contract preserved but superseded by _v2; old trained checkpoints/cache/reports
not overwritten. 22 focused tests pass, diffcheck passes. Final recipe SHA256
 a80d7c7683ce469b1042201dbfafe77e15c53c184d88d9de01e262d80508a6e9.
Preflight finiteMPSgradients4,727,042params/zero trainingsteps; benchmark disposable
updates discarded. Clean cache parity max5.722e-6 across3splits, augmentedraw/cache parity.
Measured307.19ms/batch128 +3.72scalibration =>75.91sepoch before paging/I/O/thermals.
Estimate40–75min includingtwo-variantcache/evaluation near15–25epochs, ~3h if80epochs.

Detached caffeinatePID17054 launched2026-09-22T08:24:42Z. Report:
artifacts/reports/pretrained_boundary_augmented_v17_20260922/{PLAN.md,launch.json,
benchmark.json,preflight.json,run.log}; completion/history/evaluation pending. Do not poll.
Notification on success/failure; fullpairedwholevideo evaluation, no auto-promotion.
Seed2 not resumed, defaultReel/genericphrasegate unchanged, protectedtest untouched.
Next read actual completion/retention/false-emission results on authorized update.


## 2026-09-22 — seed 1 review, overfitting audit and synchronized latency measurement

User stopped seed2 and requested discussion/verification before another fine-tune.
Seed1 selectedepoch7 of40,78.15min,calBCE.666061/val.830769. Paired12video24sign
result:9correct,2S13D0I,62.5%WER,retains2of4,1gapcommit,8EOFpartialcommits;
frozenbounded6correct/75%WER/retains3of4. No promotion or distillation recommendation.
Strong overfit; cachedposeaugmentation absent; LR already scheduled. TemporalCNN
neighbor perturbation disproves per-frameprojectionreuse. Patience8 replay stops15,
retains7,saves50.59min. NativeTCN8epochcaps do not establish convergence at7–8.
Added diagnostic-only artifacts/reports/pretrained_boundary_review_v17_20260922/
measure.py,measurements.json,REVIEW.md; no product/trainer/checkpoint changes.
Synchronizedbatch1 normalization+model16.50ms median/21.41ms p95; concurrentReel
p95202.05ms. No MediaPipe/livecamera/iPhone frontend measurement. Trainingmicrobench
~297–367ms/step; warming confounds sync attribution; no measured2xFP16gain.
Disposable optimizer updates discarded. Original checkpoint hash verified unchanged.
Updated current state; seed2 not restarted, new fine-tune not launched. Next discuss
controlled augmented-cache adaptation, then partial-block ablation and retained-sign/
false-output whole-video gates. Full hypotheses, limits and timings are in REVIEW.md.


## 2026-09-22 — authorized pretrained boundary adaptation implemented and launched

User approved nextplan,wanted enough epochs,then requestedunder30minutesifpossible and
explicitly emphasized accuracy over speed. Implemented active/v17/pretrained_boundary_v17.py,
scripts/train_pretrained_boundary_v17.py,scripts/evaluate_pretrained_boundary_v17.py,
test/test_pretrained_boundary_v17.py,newdedicatedrecipe andreportPLAN/benchmarks.
Retained currentdirtyworktree; no unrelated changes reverted or product defaults changed.

20Hzpast-observation sampling,64framewindows,target53/10futureframes,per-window upstream
normalization. Prefix and explicitEOFpadding use zeroconfidence,not fabricated observations;
model still must predictEND. PretrainedCNN/inputnorm frozen; newstart/end headwarmup5epochs,
thenall4attentionblocks adapt. Initial final-block-only speedproposal withdrawn after
useraccuracypriority. All supervisedwindows used:30014train,5007train-parentcalibration,
7593validation. Original889train/232valsourceposes and4310timings unchanged; unknown gaps
masked, no BIO-background invention. Two seeds17621/17622,max120,min40before convergence
stop,patience20,calibrationminimumBCEcheckpoint selection only. AdamWheadLR.001,
attentionLR.00005,batch128,ReduceLROnPlateau,gradientclip1.0. Best checkpoint may precede
stopping epoch; never claim later epochs necessarily improve accuracy.

Fresh tests first failedmissingmodule;17focusedtests nowpass,including boundedfuture
normalization,tailclock,cacheforwardparity,frozenCNN/trainableattention andmin/maxepochs.
Independent review found cachedpose/rawvideo linkage needed explicit checking; added
reusedhashagreement,rawvideoprecheck andafterevaluationhashcheck. Canonical494manifest
verifieswithtraining_ready=false. Dedicatedrecipe121code/weight/dependency pins verified.
MPS preflight finite gradients4,727,042trainableparameters,zerooptimizersteps. Disposable
benchmark models performed timing updates only and were discarded. Actual final configuration
batch1280.28446s/step,~66.85strain/epoch; two-seedestimate~1.5–2hoursnear40epochs,~5hoursnear120.
Earlier25–35minuteestimate concerned withdrawnlast-block-only adaptation. No30minutehardcap.

DedicatedrecipeSHA2566e80b22a8be08d9abd3d0e5229d990d916cdb6f786c4a1e18e3c743abf76b0ed.
Detached caffeinatePID94956 launched2026-09-22T06:25:19Z. It buildsfloat32frozenCNNcache,
trainsbothseeds,evaluatesselectedcheckpointsandfrozenboundedBIOcontrolwithfrozenReel,
writesreports/index andsendscompletion/failurenotification. No polling. Evaluationruns
allframesincludingunknownregions; recordsEOFdependence,WER,retainedcorrectpositions,
regionmatches andguardedgapcommits. Cachedposeconditionalcomposition is NOT live latency;
smallreused12video24sign set is not newgeneralization/hold/repeat/OOVevidence. Existing
Reel/original gates/testseal preserved. Next read completion/results on authorizedcheck.


## 2026-09-22 — all pretrained pose inputs validated; next training step discussed

User received final notification and requested discussion,not an immediate training launch.
preparation_completion.json complete1121records (850resumed/271new);
validation_completion.json passed1121. Full validation.json confirms889train/232validation,
4310intervals,190797frames,zero missing records/errors or recorded parent/signer role
crossings. This is structural input validation,not recognition accuracy. The OS-exit
validation watcher completed successfully; initial kqueue context-manager smoke failed
and was corrected with contextlib.closing before launch. Existing125-record stale current
state removed and replaced with final receipts. No training or default changes.

Next discussion proposal: reuse pretrained representation,masked start/end head adaptation
followed by small-learning-rate fine-tuning; train and evaluate with identical bounded
future windows (initial target about0.5slookahead). Benchmark full bounded inference and
training steps first. Keep Reel fixed and assess all-video retention,false outputs and
latency. Existing12video replay is reused development,not a generalization gate. Dedicated
recipe/input contract still required; generic/prepared training gates remain unchanged.


## 2026-09-22 — requested preparation update: 911/1121

Parallel preparation advanced25records since last check:911/1121(81.3%),
887train/24validation,210remaining. ParentPID38560and allthreeworkers active;
no completion receipt or reported traceback. Manifest remains preparing and
training_ready=false. Training not started; no new accuracy result. No restart.


## 2026-09-22 — user-requested parallel preparation status

886/1121cached (79.0%),235remaining; latest manifest statuspreparing,workers3,
training_ready=false. ParentPID38560elapsed30m44s; allthreeworkers remain CPU-active
(157%,374%,203% in snapshot). No completion receipt. Completion count last advanced
around5min into parallel run; firstthreeunfinished sources are long O5S5 narratives:
007_JAH17972frames/5.0min,027_LR5117frames/2.84min,029_RD14972frames/4.16min.
This explains why per-video progress is much slower than short ASLLRP clips; CPUactivity
alone is not a frame-level progress measurement. No restart,code change,training or
new accuracy result. Wait for completion notification or another authorized status check.


## 2026-09-22 — pose preparation resumed with three video workers


## Parallel extraction update

User requested multiple video workers. Interrupted serial PID69981 with SIGINT after
its850-record checkpoint; the resulting KeyboardInterrupt receipt is preserved as
preparation_completion_serial.json and is an intentional restart,not corrupt input.
Serial code and manifest snapshots preserved. Native per-video extraction is AST-identical.
Three spawned worker processes each keep one sequential tracker per video. Parent alone
writes atomic manifests,now after every completed video. Resume verifies source membership,
video/pose hashes,shape,clock and finite values; completed entries retain their original
source role and extraction provenance. Uncheckpointed tail outputs are re-extracted.

Three-video parallel smoke passed; resume accepted a valid cached record and rejected
changed pose hash,video hash and source identity. git diff --check passed.
Detached PID38560 launched2026-09-22T05:10:00Z with --workers3;850verified-checkpoint
records are eligible for reuse and271remaining videos need extraction. See
prepare_parallel.log and preparation_launch.json. Full completion not yet checked.
No change to frames,resolution,tracking,model,labels or training gate. No measured
threefold speed claim. Training still needs a representative step benchmark; cached
poses eliminate repeated RGB extraction but epoch count determines total runtime.


## 2026-09-22 — user-requested pretrained pose preparation status check

PID69981 remains running after58m54s. Log reports802/1121completed (71.5%);
atomic manifest last checkpoint800records,alltrain so far,statuspreparing,
training_ready=false. No preparation_completion.json yet; no failure receipt.
Training has not started and no new recognition measurements exist. No restart or
poll loop. Next: completion notification/user-requested check, then verify all source
and pose hashes before the bounded-context transfer recipe.


## 2026-09-22 — user-requested preparation status check

Pretrained pose preparation process PID69981 is running after about32minutes.
Latest log progress456/1121sources (40.7%); last saved manifest450records,
status=preparing,training_ready=false. No completion/failure receipt yet and no
traceback/error in the inspected final128KiB of the log. No training has started.
No further polling; next authorized check reads completion and validates all input
hashes before the bounded-context transfer recipe. Default Reel unchanged.


## 2026-09-22 — user-requested pretrained pose preparation status check

On user “update?”, detachedPID69981 remains running (~12m45s elapsed). Last saved
prepared_manifest checkpoint contains125/1121records (~11%),all train-role so far;
source order,not a training/validation admission change. Verified all125pose hashes,
source manifest hash,exact original membership/roles/interval fields and preparation
code pins. No completion receipt yet; no training has started. Keep job running with
its completion notification; no further polling. Next authorized check should inspect
preparation_completion.json before assuming all inputs are ready. No runtime changes.


## 2026-09-22 — pretrained DGS pose boundary verified; real ASL transfer and preparation

User requested continued work and pretrained weights to fine-tune. Downloaded shipped
sign-language-processing/segmentation weights atcommit22ca3a6f63b6f031bfb1c0d717fcb259143ba7db:
11.49MB,147tensors,5.734Mparameters,strict load,CPU/MPSdelta3.815e-5,finite synthetic MPS
gradients,zerooptimizersteps. Isolated source/dependency files stay under report folder;
no product dependency install. Initial GitHubmaster lookup404 corrected to main; synthetic
backward harness inference-tensor issue fixed by fresh normal tensors,then passed.

RealMediaPipe/frontend and upstream normalizer+decoder on the same12ASLLRPdev/24refs,
frozen Reel intervals+100ms/no wristtrim:11correct,3S/10D/0I,54.17%WER,retains4/4baseline,
2/12exact,22segments. Positive pretrained transfer but not promotion: full-sequence attention,
symmetricconv and whole-clip normalization use future frames.12.68svideo,17.52sfrontend,
3.25sboundaryforward includingcoldstarts; Reeladditional. No real-time/iPhone/OOV claim.

Saved decision diagnostic: correct oracle verifiers blocked by proposal acceptance4/24.
Fixed verifier-authority.45counterfactual14correctoracle/retains4of4 but learnedarms6/7/7/7,
with geometryregression/newsubstitution. Removing veto also changes scoreauthority and
removes proposal-boosted commits. No generic gate loosened.

New files: artifacts/reports/pose_boundary_transfer_v17_20260922/{check.py,probe.py,
prepare.py,PLAN.md,REPORT.md,provenance.json,compatibility.json,probe_results.json,REVIEW.json};
artifacts/models/pose_boundary_dgs_2026/{model.safetensors,config.json};
existing boundary report gainsdiagnose.py,decision_diagnostic.json. Currentstateupdated.
Source-verified trainingpose extraction andserializationroundtrip pass. Native50-joint
rawpose preparation for all1121alreadyadmittedrecords launched detached caffeinatePID69981
at2026-09-22T04:06:11Z,notificationoncompletion. No polling,no newdataset,no training.
Outputtraining_ready=false; nextcheckpreparation_completion.json/prepared_manifest.json,
thenbounded-contextnormalization/start-endtransferrecipewithunknownregionsmasked.
No plainBIOgapbackgroundlabels. ExistingReel/phrasegate/protectedtestremainunchanged.


## 2026-09-22 — completed ASL boundary experiment reviewed; no promotion

User requested “check”. All four fixed eight-epoch runs and paired evaluation completed
in 131.44s. Verified eight execution code hashes, recipe SHA256
25b24499633df10cebf69bbe99f49f571c020986a8145f017a65b6917dc057ad and four checkpoint hashes.
Same 12 reused development videos / 24 references: baseline 4 correct, 83.33% WER;
skeleton seeds 5/6 correct, 79.17/75% WER, retain 2/4 baseline positions;
hand-geometry seeds 7/8 correct, 70.83/66.67% WER, retain 3/4. Geometry deletes
16/24; every learned arm has zero exact videos. No candidate promoted.
Boundary recall at ±200ms 37.5–50%; successful-word median end-to-output 0.58–0.65s,
boundary CPU p95 61–80ms exceeds the 20Hz frame budget. Conditional desktop timings
are not sustained phone evidence. Baseline and learned arms have zero wholly-gap-contained
commits; no measured transition false-positive reduction here. No independent hold,
repeat, low-motion or OOV stress coverage. Oracle 12/24 commits and verifier 16/24
still identify downstream errors even with reviewed intervals.

Updated PROJECT_GROUND_TRUTH.md, persistent plan and report STATUS.md. Existing REPORT.md
contains the paired results. Next safe action: inspect saved per-interval proposal,
verifier and commit decisions to separate boundary misses from identity/commit failures
before specifying contextual identity adaptation. No new training, default change,
protected test use or data acquisition. Prior implementation verification: 87 tests pass.

## 2026-09-22 — ASL boundary implementation verified; detached comparison launched

Same authorized plan, no side work. Two independent reviews completed. Corrected decoder
to require predicted END instead of emitting on another START; independently filled
unknown edge target cells (two labels had depended on interval order); asserted both
edges covered by raw samples and exact cached observer/source/video contracts. Inference
checks its recipe/model-code hash and exact observation settings; rejects commit_hits>1
because this candidate makes one identity decision per completed event. Finish drains
completed pending events and records/speaks literal words; reset prevents stale results
from appearing in session history. Default Reel remains unchanged.

87focused tests pass; full git diff --check passes. Final MPS preflight finite loss/
gradients,zero optimizer steps,78402parameters,909train/146train-parent-calibration/
260validationchunks. Final dedicated recipeSHA256
25b24499633df10cebf69bbe99f49f571c020986a8145f017a65b6917dc057ad.
Pre-review recipe snapshots preserved. No training used the preliminary edge labels.

Detached caffeinate PID50241 launched2026-09-22T03:47:13Z. Fixed seeds17621/17622,
skeleton vs added hand-relative geometry,8epochs each; train-parent-calibration selection
only. Orchestrator artifacts/reports/asl_temporal_boundary_v17_20260922/run_experiment.py
calls the dedicated trainer, then shared-runtime whole-video evaluation, report generation
and artifact indexing. It checks code hashes before/after evaluation and sends one macOS
completion/failure notification. No polling. Receipt launch.json; result completion.json,
training_results.json,learned_results.json,REPORT.md. Read outcome next session/after
notification; launch is not evidence of success. Contextual-identity follow-up remains
conditional on reviewed results, not an automatic broader run. No protected test or
acquisition. Persistent plan marks tasks1–5complete; task6launched, final review pending.

## 2026-09-22 — authorized ASL temporal boundary implementation and matched baseline

User approved implementation of the step-back plan, then said continue. Persistent plan:
docs/superpowers/plans/2026-09-22-asl-temporal-boundary.md. Work in current feature branch
to preserve required uncommitted upstream data/app work; no checkout, revert or commit.
Added active/v17/temporal_boundary_v17.py, dedicated training/evaluation/live scripts,
focused test/test_temporal_boundary_v17.py; opt-in app --boundary-checkpoint routing and
explicit --no-motion-trim in existing classifier helper. Default Reel remains unchanged.

Independent data review: start/end-only supervision, no inside/background class because
gaps lack certified physical non-sign truth. 4,310events(3518train/792val),1121raw sequences,
54ASLLRP duplicate occurrences removed by recording+global-frame+label identity with
role/signer agreement. Broader ASLLRP timing used without lexical/OOV/CTC labels; O5S5
limited to110strict accepted/current admitted positive events, all other regions masked.
20Hz Apple raw frontend pinned; no acquisition. New recipe
active/v17/temporal_boundary_manifest_20260922.json is dedicated-runner only; canonical
494manifest verifies and generic training_ready remains false. MPS preflight has finite
loss/gradients,78402parameters,909train/146train-parent-calibration/260validationchunks,
zero optimizer steps. No full training launched at this entry.

Matched replay artifacts: artifacts/reports/asl_temporal_boundary_v17_20260922/.
Actual Reel same12developmentvideos/24known references:4correct,1S/19D/0I,83.33%WER.
Reviewed intervals+100ms:12correct,1S/11D/0I,50%WER,1/12exact,verifier16/24;
retains3/4baseline correct transcript-aligned reference positions. Tight intervals6correct,
75%WER; motion-trim on/off identical per context arm. Conditional oracle commit differs
from live stability scheduler; no causal/oracle deployment claim. Unknown edge fragments
are retained in annotation metadata and omitted only from known-transcript references.
Initial evaluator stopped on their mismatch before replay; fixed reference construction,
failure log preserved. These are reused development figures, no test/phone/low-motion claim.

Ten new focused tests pass including causality, touching edges, repeated identities,
runtime EOF drain, explicit app mode, trim control and retained-correct accounting;
45focused integration tests passed before final added metric/runtime tests. Independent
code review in progress. Next: address review, fixed two-seed skeleton/hand-geometry
boundary experiment detached with completion notification, then same whole-video replay.
Identity/commit errors remain even with reviewed timing; defer conditional adaptation
decision until boundary result, no threshold tuning against reused clips.

## 2026-09-22 — step-back synthesis: boundary learning plus contextual identity

Discussion/research only. User prefers one combined-data direction across available
signers rather than choosing familiar-signer-first; multiple models acceptable if fast
and accurate. Clarified that signer count/mixing alone does not establish generalization:
existing combined adapted arms have local train WER0.48/1.74% versus validation50.84/53.26%.
No new training, acquisition, runtime edits or gate changes.

Read current state, reconciliation, matched Reel, anchored CTC, Renz, SHuBERT, Zuo,
Zhao availability and familiar decoder reports; inspected live candidate activation,
submission and wrist-based trimming. Existing matched cores retain that trimming:
11/17 verifier identity correct versus5/17 conditional commits is not a clean oracle
upper bound. Six correct identities blocked; checked WORK gap fools both recognizers.
SHuBERT reduces correct tight-core commits5->2; Zuo5->0. Protocols differ, tiny reused
development evidence; no encoder ranking or general low-motion evaluation.

Recommendation remains a hypothesis: continuously observe fingers/hands/body, learn
ASL start/end or BIO boundaries with bounded future context, retain Reel recognition as
reference, then adapt identity to real contextual windows if matched oracle-segment
recognition remains weak. Separate rejection of non-sign activity from unknown lexical
identity; do not require pauses or a background frame between adjacent signs. Evaluate
whole-video insertions/deletions/substitutions, retained correct events, repeats/holds,
low-wrist-motion recall and end-to-display latency against unchanged same-input Reel.
First controlled comparison separates current segmentation, reviewed boundaries with
explicit trim policy, and learned boundaries, using identical recognizer/input contracts.
No claim of a final guaranteed fix, deployable foreign blank head or need for a new backbone.

Primary sources rechecked: Zuo https://arxiv.org/html/2401.05336v2 (context augmentation,
foreground saliency, matched short-window training/inference); Renz
https://github.com/RenzKa/sign-segmentation (temporal segmentation); Zhao final paper
https://www.sign-lang.uni-hamburg.de/lrec/pub/26014.pdf (BIO allows adjacent signs without O);
SHuBERT https://github.com/ShesterG/SHuBERT (ASL multimodal representation, not ready boundary
head). Existing background/context/anchor attempts are not new proposals. Proposed
distinction is native temporal boundary supervision and whole-stream evaluation; copied
weights or another interval veto do not establish it. Mix eligible sources by supervision:
isolated identity, verified boundaries, complete transcripts; mask incomplete unknown
regions, preserve current source roles and deduplicate events. Motion-only data has no
invented boundary labels. Next safe action: discuss this design, then prepare coverage and
evaluation contract before any new bounded recipe. Changed only this log and current-state
discussion note in PROJECT_GROUND_TRUTH.md; no experiment or accuracy result generated.

## 2026-09-22 — Zhao author components recovered; end-to-end contract missing

Reviewed final LREC paper and author GitHub inventory. Found ASL-Handshape and
SegmentASLTransformer code/weights, correcting earlier search status. Saved selected
sources, pinned commits, hashes and inventories under zhao_mhb_availability_v17_20260922.
Downloaded model files only (no training/evaluation payloads). Segment strict34-tensor
MPS synthetic check passed, CPU/MPS maxdelta5.722e-6; handshape weights65tensors/88outputs.
Active segment forward ignores handshape labels and requires preseg predictions whose
producer is absent. No sound real-video accuracy test without that contract; zero/random
preseg would not reproduce MHB. No failures in the compatibility check, no live promotion.
Paper/data distinctions and proposed ASL temporal-boundary experiment recorded in REPORT.md.
Next: reviewed source coverage/full-stream evaluation recipe, then bounded in-domain
boundary learning; do not tune reused3gaps or bypass training gates. App unchanged.


Measured results, rejected approaches, and progress snapshots. Not read start-to-end —
`rg` this file before re-running an experiment to see if it already failed.

50 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## 2026-09-22 — actual Zuo online blank transfer completed: rejects correct signs

RecoveredPHOENIXonlinecheckpoint fullyrun withHRNetW48DARKMPS. Same17validationcores+3gaps,
newexplicit16frame25Hzcentrewindows, notSHuBERTpoolingprotocol. Argmaxblank0 ofmean3head
softmax, nofit/tuning. All3gapsrejected; falseconditionalcommits1->0. Correctcorecommits5->0:
4FRIEND+1COLD lost; only4/17corewindowsnonblank.88.86s afterload,73.69pose+11.03recognition.
No livepromotion/WER. Foreign-trainedhead/domain/letterbox differences preventmethodrejection.

Artifacts zuo_twostream_probe_v17_20260922: transfer.py/recipe/results/check/REPORT/DATASETS,
model_smoke.py andpose_smoke.py. All938recognizer tensors strict, HRNetstrict. MPS3DpoolCPU
fallback; CPU/MPSsyntheticprobdelta1.788e-7. Focusedgeometry/membership/vetochecks passed;
independentreviewconfirmedkeypointorder,BGR,ensembleandmetrics. Poseoverlayvisuallychecked.
IsolatedMMPose0.29+MMCV1.7.0 installed; repairedpkg_resources/buildpath/Cython dependencies,
no liveenvironmentreplacement. Initialfailures retained. HRNetCUDA scatteradaptedlocallyforMPS.
No signingdatasetpayload or training; genericgateunchanged. Datasetaccess andauthorreplacement
links preserved. NextZhao/MHBavailability, thenreviewallcandidatesbeforein-domainadaptation.

## 2026-09-22 — recovered Zuo online weights; MPS contract smoke passed

Official SharePoint model links404. Authorissue106/107comments provide replacementDrivefolder
1U-BK7R-fMLmkSDq2M7GpHT9J4aKK9TFa. Listed remoteZIPdirectories; fetchedonlyPHOENIXonline
checkpoint456022647bytes+vocab12551bytes viaRanges/ZIPCRC, no datasetpayload. Onlinearchive
2.7GB includes usablecslr_best; older21.3GBarchivePHOENIX14TteacherS2Gentry iserrorplaceholder.
Sourcepinned38a4f7b00da7a858d59b7fabe5093876a84db8e0. Strict938tensor model load passed;
synthetic16frameMPSforward3.314s,CPU3Dpool fallback,CPU/MPSprobabilitymaxdelta1.788e-7.
This is NOT real-video recognition or accuracy.63HRNetchannels required,1116outputs inclblank.
OfficialHRNetW48DARKwholebodyweights acquired; isolatedlegacy pose runtime setup inprogress.
Savedaccess/inventory/weights/provenance/data notes/model smoke underzuo_twostream_probe_v17_20260922.
No live model changes or training. Next: validate actualHRNet extraction thenpairedrealinputprobe.

## 2026-09-22 — SHuBERT paired decision probe complete: mixed result, no promotion

FrozenMPS extraction45videos/1585frames completed in272.02s; face1576/1585,pose1585/1585,
nozero-facevideos. Same247prior intervals,26fit/7calibration/12validation parents; fixedridge100,
training-onlycalibration, no neural weight update. SHuBERT rejects2/3held-outgaps includingfalse
WORK; conditionalfalsegapcommits1->0. But correcttight-corecommits5->2: threeFRIEND successes
suppressed. Context100ms9->6,250ms5->2. Priorlandmarkgate0/3gapsrejected; DINO/body0/3.
No stationary-wrist-positive samples, fullstreamWER or live evidence; fullclipnoncausalfeatures.

New artifacts/reports/shubert_decision_probe_v17_20260922/{run.py,check.py,recipe.json,PLAN.md,
REPORT.md,results.json,audit.json,cache/}. Prior smoke only extended withsource/output/cropargs.
First scoring aborted beforefit: croppedMP4FPSrounded29.97002997->29.97 excludedendpoints.
Corrected mapping usesoriginalsourceclock; unchangedcachedfeatures admitted explicitly under
revisedrecipe; initialrunner/recipe/failurelog retained. Independent audit verifiesdecodedframe
counts and all247intervalframecounts againstprior evidence. Exactpriorcontrolresults reproduced;
calibration/veto/split/clockchecks passed. No new data or protectedtest; genericphrasegatefalse.
Decision: do notdeploy thisgate or tuneon3reusedgaps. KeepSHuBERTcandidate; nextZuo/TwoStream
comparison beforebroadertraining. Data provenance register retained.

## 2026-09-22 — SHuBERT paired interval comparison started

User authorized continue. New bounded recipe/PLAN/run/check in shubert_decision_probe_v17_20260922.
Reuses247earlier decision-probe intervals across45videos, existing parent-video calibration,
fixedridge100 and thresholds preserving all calibration positives. Three held-out gaps only.
Arms: unchanged/confidence, earlier landmarks, DINO/body, full-context frozenSHuBERT. Encoder
and image weights remain frozen; generic phrase gate remains false. Approved verifier passed.
Refactored earlier smoke entrypoint only to accept source/output/crop parameters; new pooling
and split checks passed. No live changes, new data, protected test, or expanded negative pool.
Results pending; extraction usesMPS and saves cache/source hashes.

## 2026-09-22 — spoken output given a selectable, softer voice

User asked for a Siri-like soft voice. Hard limit found and stated: macOS exposes no Siri
voice to any speech API — nothing Siri-related appears among the185 voices
`NSSpeechSynthesizer` lists, and `say` does not offer one either. The closest reachable is
Apple's neural premium tier, which must be installed by hand in System Settings >
Accessibility > Spoken Content > System Voice > Manage Voices. **This machine currently
has only compact voices**; every English voice installed is `com.apple.voice.compact.*`,
which is the main reason the output sounded robotic.

`LiveSpeaker` was hard-wired to `initWithVoice_(None)` at220 wpm. It now takes a voice and
a rate, resolves the voice through `resolve_voice()` — full identifier, then voice name,
then substring, then a softest-first preference list (premium Ava/Zoe, enhanced Ava/Zoe/
Samantha, compact Samantha) — and falls back rather than failing, so a voice that is not
installed can never stop a session. Default rate is now190 wpm per user choice.

Shared change, per user direction: `make_speaker(args)` replaces the four bare
`LiveSpeaker()` constructions in the reel, isolated, stage2-CTC and streaming scripts, so
all of them improve together, and `--voice` / `--speech-rate` were added to the shared
parser. The resolved voice and rate are printed at startup and recorded in the session
config, with a one-line hint about installing a premium voice when a compact one is in
use. Speech is presentation only and cannot affect recognition; equivalence re-checked
anyway — the same six clips decode identically through pre- and post-change modules, 6/6.

A bug was caught in test: a whitespace-only `--voice` was truthy, so the empty substring
matched the first available voice. A blank request is now treated as no request.

Tests: 231 passing across15 modules (11 new in `test/test_live_speech_voice_v17.py`);
`git diff --check` clean. Nothing here is audible until a premium voice is installed —
that step is the user's.

## 2026-09-22 — SHuBERT published-weight MPS frontend/encoder probe

User requested continuing SHuBERT and documenting its data before choosing combined training.
Downloaded official encoder + face/hand DINO weights, pinned official repositories, saved
hashes and isolated runtime dependencies. New report/probe/check/data register:
`artifacts/reports/shubert_probe_v17_20260922/`. No dataset acquisition or head training.
Strict loads passed. Real approved ASLLRP validation WATER COLD:41frames/1.368seconds,
finite1x41x768features. Uncropped face detection0/41; published YOLO signer crop restored41/41;
pose41/41,hands34/41and37/41. Single-clip timings: crop4.727s,MediaPipe2.715s,
DINOincludingload2.283s,encoder0.456s; not sustained live performance. MPS used for neural
models; MediaPipe separate runtime. CPU/MPS encoder maxabsdifference1.824e-5; check passed.
Initial DINO Python3.9 type-annotation failures retained; postponed annotations in two vendor
files only. OpenCV/in-memory crop adaptation documented, no numerical encoder change.

No recognition/WER or transition-rejection improvement established. Data register distinguishes
YouTube-ASL984hours/caption supervision, downstream datasets, existing local keypoint pilot,
and evaluation overlap. Next: bounded frozen-feature sign/transition comparison under reviewed
recipe and same train/held-out intervals; then other candidates before broader training.
Live app/default Reel unchanged; no protected test accessed. git diff --check passed.

## 2026-09-22 — user orders pretrained comparison before dataset combination

User reiterated: retain Renz dataset for later experiments together; first see SHuBERT and
other candidates before deciding. Saved ordered shortlist, evaluation distinctions, official
Renz dataset URL/schema/archive ID/access caveats in five-step recovery PLAN. Dataset not
downloaded or admitted; do not conflate already-downloaded weights with data. Immediate
Renz fine-tuning deferred in favor of pretrained comparisons. No new model runs this turn.

Also recorded preceding authorized live integration: scripts/live_renz_v17.py, app shell
opt-in mutually exclusive --renz-buffered routing, test/test_live_renz_v17.py. Default remains
Reel. MPS initially failed unsupported3Dpooling; explicit CPU-pooling fallback added. Report
renz_live_v17_20260922 retains failure logs, real-video smoke and accelerator smoke.
60focusedtests passed; actual1.3013second/27observationclip inference10.984seconds;
4second repeated-image throughput28.9999seconds; CPU/MPSembeddingmaxdelta1.82e-6.
Bufferedpreview uses4secondwindows/2secondhop/1secondrightcontext, visible backlogskips,
resetgeneration discards stale results, records decisions/video. No webcam opened by agent.
Current request changes documentation/priority only; no dataset acquisition or training.

## 2026-09-22 — Renz pretrained offline transfer complete, no promotion

12approvedASLLRPval/24refs; strict publishedI3D+MS-TCN loads. Rawvalid16framewindows omit
clipedges ->0tokens/100%WER. Edgepadding restoresonefeature/sourceframe ->6tokens/79.17%.
Publishedmidpointregionextension on samecachedpaddedfeatures ->10tokens,15edits/24=62.50%,
2/12exact. BestregionIoU>=.5eligiblecorecoverage2/17->9/17->13/17; notmatchingprecision.
WORK WHERE->WORK WORK; cannotattributepuretransition fromthattranscript alone.
Conditionalcommit bypasseslivescheduler; nofullReelmatchedstreambaseline, nooverallWERclaim.
SafeCPUadapter,aspectpreservingpadding differsfrompublishedstretch; noncausal,notmobile.
50.18sunpadded/100.39spaddedfullpass; finalreplaycache reuse. No training/livechanges.
Report/scripts/provenance/cachedfeatures/results retainedinrenz_reel_v17_20260922 with
edge_padded andedge_padded_midpoint subdirs. check.py verifiedsourcecontract/checkpoint
hashes,coverage,finiteoutputs,editcounts,threepreservedrunnerhashes;gitdiffcheckpassed.
Next discuss ASLadaptation orSHuBERTreference; neither launched. Data notprovenbad.

## 2026-09-22 — pretrained Renz offline transfer experiment started

User authorized first experiment after research. Downloaded official upstream code commit
29cc10963b41179c09e6fab4e0585c263f4917c9 and official models.zip via repository download
script's Drive ID. Download initially timed out at97%, resumed successfully, ZIP verified.
Extracted defaultBSLCorpus I3D+matchingMS-TCN only; weights_only loading and strictstatekeys
passed. Code artifacts/vendor/renz_sign_segmentation;weights artifacts/models/renz_pretrained_v17.
Dedicated runner/provenance/runlog under artifacts/reports/renz_reel_v17_20260922.
Inference started on12approvedASLLRPvalidation videos; full494approvedmanifest verifierpassed.
No training,newvideos,testaccess orlivechanges. RGB25Hz,16frames,stride1,segment100featurechunks,
threshold.5 perpublished demo; actualcenter timestamps retained. CPU, safeffmpegpipe replaces
upstreamsourceoverwrite. Requiredaspectpreservation addsletterbox before256resize/224crop;
this is an explicit transferfrontenddeviation, notfaithfulstretch reproduction. FrozenReel
classifies predictedclass0spans atnative20Hz schedule; conditionalcommit notfullscheduler.
Models are noncausal offline reference. Results pending; do not claim improvement.

## 2026-09-22 — step-back research: pretrained segmentation and ASL representation alternatives

Discussion only; user challenges pairwise hard-negative patching. Correction: fast ridge fit
with14traininggaps is not a meaningful rejection of general temporal boundary models.
Recommend evaluate pretrained category-agnostic segmentation (Renz I3D+MS-TCN) as an offline
reference before more custom gates; SHuBERT is a stronger ASL representation candidate for
subsequent adaptation, not an off-the-shelf100sign or transition classifier. Published links
verified in official READMEs; weights not downloaded/run and transfer/latency unmeasured.
Renz repo provides302MBmodels archive,I3D andMS-TCN checkpoints/video demo; BSL/PHOENIX
training, code license does not grant separate model/data rights.
https://github.com/RenzKa/sign-segmentation
SHuBERT official repo links encoder+DINOhand/faceweights; ~1000hASLpretraining, feature
extraction available; downstreamfine-tuningREADME stillTODO. Differentfrontend, not direct
Apple-feature-compatible. https://github.com/ShesterG/SHuBERT
Zuo official Online/CSLRREADME links PHOENIX2014T/CSLDailycheckpoints plusTwoStreamteacher;
German/Chineseheads cannot force-align ourASLglosses without adaptation.
https://github.com/FangyunWei/SLRT/tree/main/Online/CSLR
ZhaoMHB authorpreprint hasBIOtemporalSTGCN+canonical87handshapepretraining; adjacent signs
mayhave noOgap. Methods specifyAlphaPose2D+velocity/acceleration despite3Dabstract wording.
Important metric qualification: top180.23–83.30% only boundary-matchedsupportedsegments;
3783/6595groundtruth(57.4%)matched, notoverallWER/randomliveperformance. Random4:1split,
not evidence of oursignerdisjointgate. No officialMHBweights verified in searched sources.
https://arxiv.org/html/2511.19907v1
CommercialSign-SpeakdocumentsASLrecognitionAPI; notverifiedaccuracy/weights/offlineSDK.
https://app.theneo.io/sign-speak/sign-speak-api/api-specifications/asl-production
No acquisitions,contact,training orlivechanges. Preferred generalizing question is where
sign boundaries occur using temporalhandshape/bodyevidence, not cataloging allsignpairs.

## 2026-09-22 — fast temporal Reel decision probe failed; no promotion

User authorized fast test. Reused56approvedASLLRP sources and current frozenReel models.
Approved494membership/hash verifier passed; dedicated recipe pins exact source/script hashes,
raw video hashes checked. Generictraininggate remainsfalse. Four temporal landmark bins plus
proposal/verifier scores -> class-balanced ridge100; fixed linear probe, not full RGB temporal
head. Original44train parents split deterministically for fit/calibration;12validation untouched.
Fit48coresx3+14gaps;cal11coresx3+2gaps;val17coresx3+3gaps. Threshold preservescalpositives.
Bothcalgapsrejected but0/3valgapsrejected; falseWORKstillpasses. Correctconditionalcommits
core5->5,+100ms9->8,+250ms5->4. Confidencebaseline unchangedcommitcounts, also missesWORK.
No promotion/model/livechanges. NotfullstreamWER, no stationarywrist or trueGOODBYE coverage
claim. Linear temporal bins omit learnedRGBembeddings and candidateidentity; cannot dismiss
all decisionmodels from this result. Small negative coverage and feature support unresolved.
run.py, recipe.json, check.py, evidence.json, results.json, REPORT.md, run.log under
artifacts/reports/reel_decision_probe_v17_20260922. Schedule/overlap/linearfit selfchecks and
full previousvalidationprediction/commit parity passed; gitdiffcheck passed. Next audit existing
training-source confusing gaps plus genuineconfusable signs before richer temporalhead;
never tune against or move the validationWORKgap into training.

## 2026-09-22 — researched recommendation: temporal Reel emission decision experiment

User asks whether a decision model can reject transitions. Recommendation only; no training,
weights, runtime thresholds or data roles changed. Freeze current Reel recognition models;
compare existing commit logic, a calibrated score-only rejection baseline, and a small causal
temporal emission head using rolling visual/landmark evidence plus candidate identity.
Train on reviewed training-source false candidates and real contextual positives, including
true GOODBYE/WORK and low-wrist-motion signs. Do not turn every unlabeled gap, O5S5 outside-
positive interval, or CTC blank into transition truth. Preserve mixed sign/context positives;
mask uncertain boundaries. Validation-discovered WORK gap stays diagnostic/validation, never
silently becomes a training example. Split by parent video before overlapping window creation.
Match runtime sampling/normalization and retain temporal handshape/body evidence, not just
softmax confidence or wrist movement. First rejection comparison keeps candidate generation
fixed; separately evaluate fixed-cadence, untrimmed candidates for low-motion recall, because
an added veto cannot recover windows the current wrist gate never supplies.
Evaluate full stream insertions, deletions, correct-sign recall, WER, repeated signs and
emission latency; choose threshold on separate calibration sources. Proposed practical target
(not achieved): halve transition insertions with <=2 percentage-point correct-sign recall loss,
plus latency measurement. Three diagnostic gaps are not a sufficient promotion benchmark.
Failure at matched recall means do not stack another gate; inspect representation/label support.

Research: Zuo et al. EMNLP2024 Table5 background alone devWER62.6->49.1; adding contextual
augmentation24.4, fullrecipe22.2. These are paper-specific numbers, not projected ASL results.
https://arxiv.org/html/2401.05336v2
Zhao et al. LREC2026 abstract supports separate boundary detection using 3D handshape and
skeletal dynamics followed by recognition; no claim its offline method supplies live latency.
https://aclanthology.org/2026.signlang-1.52/
Kong/Ranganath2014 separates SIGN/movement-epenthesis before recognition; feasibility evidence,
not a performance guarantee on our data. https://doi.org/10.1016/j.patcog.2013.09.014

## 2026-09-22 — executed matched Reel core/context/gap comparison

User authorized doing comparison. CurrentCoreMLproposal+fullverifier re-extracted12approved
ASLLRPdevelopmentvideos; hashesverified. Same17cores: verifier11/17tight,11/17+100ms,
7/17+250ms; correctconditionalcommits5/17,9/17,5/17; wrong1/0/1. Modestcontext helpssome
commits; largercontext harmsidentity. Gap842935(0.5205–0.8475s): proposalWORK.5943/verifier
WORK.5173, bothaccepted+conditionalcommit;1/3checkedgaps pass. Sixcorrectcoreverifiertop1
blocked; oneFRIENDproposal overridden toANSWER passescommit. No single threshold diagnosis.
Conditionalcommit bypasses activation/proposalstability/history; not actualstreamWER.
All17cores passwithin-windowwriststart; staticwrist behavior notcovered.

First replay incorrectly advanceddeadlinefromcurrenttime (effectively15Hzon30fpsinput).
Preservedinitialoutputs, fixed to actualReelmax(previousdeadline+1/20,currenttime), added
selfcheck, reranall12clips and assertedpaired17identities. Only correctedresults count.
Recent5appMP4unreadable, so no exactuserHELLO→MY replay. Report/script/results:
artifacts/reports/reel_matched_windows_v17_20260922/.

Separateboundedrecordingfix: SessionRecorder.finish_video() releasesMP4 beforemodelcleanup;
appnativewindowclose signalsquit. Historicalmissingmoov/unfinishedhistory shows nofinalization,
notproof ofexactexitroute. Worker and parent independently ran89focusedtests, allpass.
No modelweights,thresholds,training or dataset roleschanged. Next hard-negative+truepositive
comparison and replayableliveinput; preserveReeldefault.

## 2026-09-22 — conditional twenty-minute Zuo request feasibility

User authorizes execution ifunder20minutes. Checked priortrainer andmeasuredruntime:
O5S5window12epochs108.43sMPS, but this omits prep/implementation/evaluation. Existing
traineralreadyhasbackground,contextforegroundCE,balanced sampling andteacherKL, so
rerunning/addforegroundalone is not newZuo-styletest. Missing exactpaperloss/grouping,
currentrecipecontract andend-to-endvalidation remainunprepared/untimed. No defensible
under20min total estimate; no newtraininglaunched. Feasibility recorded in
artifacts/reports/reel_streaming_research_v17_20260922/TIME_FEASIBILITY.md.

## 2026-09-22 — Zuo replication feasibility and YouTube pilot clarified

User asks if study can be replicated without more acquisition and what happened to
YouTube keypoints. Re-read paper sections3.1/3.2/A.1/A.3, dataset reconciliation and both
motionpilot reports. Distinguish faithful RGB+keypoint TwoStream/S3D reproduction from
method adaptation using currentReel architecture. Paper's dictionary derives from a
strong offlineCTC teacher plus known gloss transcripts, not unlabeledmotion alone.
Existing ASLLRP/O5S5/STEM timedpositives and rawparentcontext support a bounded adapted
replication without newdownloads. Positive-only labels do not establish surrounding
background; intervalcoverage across100 and ambiguousgapreview still required. Additional
annotation of existingvideo may be necessary; sufficient100signcontextcoverage unproven.
Earlier background/window failures are partialattempts, not evidence of fullfaithfulreplication.

YouTube retained1411JSON,1191countconsistent(220quarantined), separate46nodeXY/presence
branch. Masked6frame spans/32framewindows, transferonly3causalCTCblocks; AppleStage1frozen.
Correctedphrase rerunlocalWER48.52→43.15 and50.19→44.44, butseed17321ASLLRP54.17→66.67,
seed17322deletions52→130, holds/repeatsnotimproved. No promotion; not adopted by current
livecandidate/Reel. Earlier reconstruction failed causalcarryforward baseline. This is
mixed evidence for a particularSSLobjective/transfer, not proof keypointsuseless.
Do not count unlabeledYouTube as semantictransitiontargets or restart acquisition.
Recommend explicit paper-component/data-coverage audit before anotherboundedreplication;
retainYouTube for optionalcontrolledpretraining, not primarysignboundarytruth.
Sources: https://arxiv.org/html/2401.05336v2 and
artifacts/reports/youtube_motion_pretrain_fixed_phrases_v17_20260921/REVIEW.md.
No runtime/training/data changes.

## 2026-09-22 — research: retain Reel UX/reference, replace wrist-only segmentation hypothesis

User asks whether CSLR target is realistic, reports HELLO→MY falseGOODBYE. Reviewed
EMNLP2024onlineCSLR and AT4SSL2023linguisticfeatures paper, current wrist activation/
trim and disablednoemit, and earlierwindow/background failures. Recommendation is
continuous sign spotting using current trusted Reel evidence plus learned emission,
not freezing current wristgates or promising a freshhead solves everything. Preserve
holds/fingerchanges, use reviewed real hard-negative transitions and positiveGOODBYE,
measure arbitrarycompositions/insertions/deletions/stablelatency. Existingwindowmodel
alreadyfailed316.20%WER/770insertions,16/16transitionemissions; background class alone
notnovel or sufficient. First collect replayable commoninput/modeltraces, diagnose,
then bounded frozenencoder comparison before broadertraining. No runtime/trainingchange.
Report: artifacts/reports/reel_streaming_research_v17_20260922/REPORT.md.

## 2026-09-22 — user live familiar failure on I GO; Reel history confirms emissions

User reports candidate recognizes familiar greeting but misses I GO; requested Reel history
comparison. Reviewed candidate sessions074639/074748 and Reelapp074918Sep22. Reel committed
I27.17/41.44s andGO29.54/42.95s (verifiersI.915/.962,GO.771/.768). Candidate logs emitted neither;
mostlyHELLO/HOW/YOU/OTHER/TOMORROW. These are distinct runs, not pairedgroundtruth/WER.
Training coverage: I28isolated and115phraseclips allPLEASE HELP I;GO23isolated and33phraseclips
allTOMORROW SCHOOL GO;zeroI GO pairs. Selectedcandidatecachedisolatedval I15/17headexact,
GO9/16; labelsnotmissing. CandidateusesolderasllrpcoreStage1, not currentReelproposal/verifier.
No liveLMenabled. Six-template19.14%validation did not establish user's arbitrary signing.

Livecandidate13.34/16.90observedfps with30Hzduplicatedfeatureticks and causalnormalization
remainpossibleinput-domain causes, not demonstrated attribution. Candidate saved onlydecoded
events, no video/features/logits: instrumentation insufficient to diagnose missedrealattempts.
Tried bounded same-recording replay via AppleVision, but latestReelMP4 lacksmoovatom and cannot
open;no replayheadprobabilities. No running app process atcheck; nonfinalizationcauseunknown.
New report/replaydiagnostic: artifacts/reports/live_familiar_diagnosis_v17_20260922/.
No thresholds/model/training changed. Keepcandidate experimental; next safeaction capture
replayable commoninput and per-window base/head evidence before trainingorLMpromotion.

## 2026-09-22 — familiar candidate available through app shell

User requested live command and concurrent decoder experiment, then explicitly selected
scripts/app_shell_v17.py. Added opt-in --familiar-ctc-checkpoint branch to existing shell
warm/start; default Reel path preserved. New active/v17/familiar_live_v17.py verifies
reviewed candidate hash and original base/head provenance, restores familiar state, uses
8sourceframes/stride4 and bounded15evidence-step causal history with greedy CTC collapse.
No wrist-motion emission gate, no language prior, no English completion. Existing
scripts/live_continuous_v17.py supports prebuilt model/shell presentation; navigation
pauses extraction and resets model state, literal single-line transcript, Enter clears,
Q quits. Camera normalization is past-only and differs from cached clip normalization;
batch parity does not establish live accuracy. Candidate events saved under continuous_live_v17;
not yet integrated into app HISTORY's SessionRecorder format.

Command: venv/bin/python scripts/app_shell_v17.py --familiar-ctc-checkpoint
artifacts/models/local_familiar_ctc_v17_20260922/seed_17521_familiar.pth

Validation: real checkpoint CPU streaming logits match batch including residual endpoint
and boundedhistory (>15steps), CTC tokens match, reset/nonfinite rejection checked. MPS
live forward finite. 95 focused tests pass across new runtime, app pages/shell and causal
vision; CLI help pass; diffcheck pass. No camera opened by agent, no protectedtest access,
no live default promotion. Offline decoder comparison running separately on CPU; initial
weak bigram gains require uniform-prior control to distinguish a length penalty effect.

Offline decoder comparison COMPLETE, fixed1735validationrecords/two familiar heads,
beam8 and priorweight0.1,51distinct trainingphrase transcripts only. Local60WER:
17521 greedy19.14/beam17.90/uniform16.67/bigram16.05%;17522 22.22/22.84/21.60/20.99%.
Uniform prior isolates constant extension length penalty: learned pair preference adds
only one fewer edit (0.62percentagepoint) and one more exact clip per seed beyond uniform.
Most improvement is not demonstrated linguistic-context gain. All60local targetsequences
seen in training;9/12ASLLRPsequences novel and noWERchange(33.33%). No liveLM promotion.
Report/results/scripts/tests under familiar_decoder_v17_20260922 and
scripts/evaluate_familiar_decoder_v17.py. Decoderfocusedtests pass; no hypothesis weights
tuned on validation. CPUbigramdecode2.90/2.92msperexample averages, not camera latency.
Next safe action is user live app-shell evaluation of visually decoded candidate; use
novel combinations to assess reliability before promoting any language prior.

## 2026-09-22 — familiar-signer trial completed; language-prior research

Reviewed all4completedruns: checkpoint hashes/earliest-best selection/paired baselines/
coverage verified. Same60localclips: control35.80%WER(epoch0both), familiar19.14%(epoch6)
and22.22%(epoch5); exact20→33/27of60. LocaltrainWER4.87/6.26%familiar. AllS/D/Ifell.
ASLLRP12knownWER33.33%both; O5S5still77–79%. Familiar-signer/six-template reused development
results, not arbitrary live/unseen-signer proof. FrozenStage1 means no evidence of new
encoder coarticulation learning. Added139clips also changes data amount/composition.
Previous standalone211evaluator countsOTHER; this trainer removesOTHER for knownWER;
localmetricsunaffected, do not compare cross-report pooledscores without normalization.

User asks next action and next-word research. Read2025 Sign Spotting Disambiguation using
Large Language Models: visual candidates+gloss conditional probabilities via beam search,
top1 latefusionWER47.24→44.38/44.73%, top5notdeployabletop1. Recommended candidate streaming
parity/live stress evaluation first; add separate offline greedy/beam/weaksmoothedglossprior
comparison with training-only transcripts, no forced phrase completion, unseen combinations
and latency/insertions checks. Then actual-context jitter as separate visual intervention.
No live promotion/newtraining/LM implementation. Full report:
artifacts/reports/local_familiar_signer_v17_20260922/RESULTS_REVIEW.md.

## 2026-09-22 — older matched evaluation; authorized familiar-signer experiment prepared

User requested evaluation, relaxing local-phrase signer separation, and research on
transition augmentation. Evaluated older aligned-grounded causal pipeline on exact current
211 approved validation phrases: WER39.0374%, local199WER38.7337%, ASLLRP12WER45.8333%;
55/211 exact. Earlier pipeline beats recent50.98–53.65% on this development benchmark;
multiple pipeline components differ, so this is not a head-only causal comparison.
No old-head training source-item overlap; broader historical base exposure not fully audited.
Report: artifacts/reports/previous_ctc_approved_v17_20260922/REPORT.md; evaluator
scripts/evaluate_previous_ctc_approved_v17.py. No live promotion.

Reviewed source labels and three papers (EMNLP2024 online CSLR, IberSPEECH2021 synthetic
sentences, ICCV2021 VAC). Report: artifacts/reports/coarticulation_research_v17_20260922/REPORT.md.
2863 training singles carry clip identity only; 1401 have interval-derived provenance.
Current single-sign training does not supply uniform entry/core/exit targets. Actual
continuous-context jitter and smooth synthetic transitions are supported hypotheses;
transition-only noise is not established as sufficient. No augmentation implemented here.

Prepared split_manifest.json and CONTRACT.md under local_familiar_signer_v17_20260922:
139 previously-validation local signer02 clips moved to experimental training, 60 held out;
raw-video hashes separate; other roles unchanged. User-authorized local exception only,
not a waiver of Citizen splits. Same6421 records, control4547/familiar4686train;
common1735validation incl60local+12ASLLRP+1663singles. These are reused familiar-signer
held-out clips, not independent/unseen-signer results. Historical211benchmark retires for
new candidate selection after139clips enter training.

Dedicated scripts/train_local_familiar_ctc_v17.py, test/test_local_familiar_ctc_v17.py,
active/v17/local_familiar_ctc_manifest_20260922.json: frozen older Stage1, cached evidence,
control/familiar arms, seeds17521/17522,6epochs,headLR1e-4,WD1e-3,epoch0 allowed.
Two focused tests passed. Review caught pre-launch report-directory, FP32 cache, dependency
pinning, correlation weights, known-WER/OTHER and failure notification issues; corrected.
MPS zero-step preflight and independent review completed.
Launch verified after final equality guard and repinned zero-step preflight:
6421records/15430windows, cache/direct maxdelta4.7684e-6, finite/nonzeroheadgradients,
basegradientsabsent,2focusedtestspass. Detached caffeinatePID78465 launched
2026-09-21T23:31:43.656638UTC (Sep22PHT), recipeSHA
 a91c756a3f8b04eda63f7d7185f373288efbfcc079243cb20908503f1e9437c9.
Completion not yet observed; no training polling. Read status/results next session, compare
both arms on common60local clips and source retention before deciding. Index regenerated;
git diff --check passed. Allfive recovery steps retained; liveUI/model unchanged.


## 2026-09-22 — transition-anchor comparison complete; no improvement over control

User requested update. All4runs completed3epochs; checkpoint hashes, earliest-best selector
includingepoch0, saved validation and4264/536single/phrase exposure verified. PhraseWER
control50.98/52.41%, anchored50.98/52.41%; exact29/24versus29/22of211. Seed17421bothselect
unchangedepoch0; seed17422controlselectepoch2andanchoredepoch1. Anchoredseed2source-balanced
selector worse(.57228vs.55237). All-sourceWER27.56/31.25%controlvs27.56/31.47%anchored.
No benefit demonstrated from this bounded timed-anchor refinement. No promotion or extra
training. Review artifacts transition_anchors_v17_20260922/REVIEW.md,REVIEW_SUMMARY.json.
Scope remainslimited: frozenencoder,3epochs,0.25loss,59traincores/27gaps; does not establish
transition supervision is impossible. Nextdecision must revisit intervention/data support,
not automatically repeat or extend the failed recipe. Live model/application untouched.

## 2026-09-22 — requested interim timed-anchor update

User requested status; one snapshot shows state running, three of four runs saved with
all3epochs each. Seed17421control and anchored both selectedepoch0: neither beats initial
validation selector; both retain combinedphraseWER50.98%. Seed17422control selectedepoch2,
combinedphraseWER52.41% (initial53.65%), all-sourceWER31.25%. Seed17422anchored has no saved
history/result in this snapshot. No overall anchor benefit established; do not infer final
result or promote from partial results. No trainingparameters changed or backgroundpolling.
Next: completion notification / next requested status; then full paired comparison.

## 2026-09-22 — timed sign/transition head comparison launched

User asked for faster work and actual transition learning. Independent review cleared
anchor prep/contract and runner0cdb5b345116b24ef949b2df17fa83dc12f14963dd1848ebda97fa1445212bad.
Focused anchor-loss test passed, including blank0/known1..100, ignored tokens and finite
nonzero gradients. ActualMPSpreflight passed6421records/7367richwindows perseed; cached
versus direct logits maxdifference0.0both; each gradientprobe included4known/2blank regions,
allheadgradientsfinite/nonzero, basegradientsabsent, zerooptimizersteps. gitdiffcheckclean.

Launched detachedcaffeinate PID40294 at2026-09-21T22:57:08.955694UTC (Sep22PHT).
Two selected adapted initializations17421/17422; control and anchored arms;3epochs each.
Stage1frozen, richfeaturescachedonceperseed. Anchored loss adds0.25intervalCE to unchanged
CTCloss, central signcores plus bounded internalgapblank hypothesis. Genericgateunchanged.
Recipe SHA03b2fac300e3b8fb45a5be62fa8147c20b406867007c530e21649f55e503fe74;
preflightSHA8cb563bc79287a142b3508c4d88c4751776dba87fffe5b696ccfffb00bf8b8a6.
Newfiles: scripts/train_transition_anchors_v17.py, test/test_transition_anchors_v17.py,
scripts/prepare_transition_anchors_v17.py, active/v17/transition_anchors_manifest_20260922.json;
reports under artifacts/reports/transition_anchors_v17_20260922. Earlier matched diagnostic
script/report and sourceinterval audit retained. No trainingpolling or completionclaim.

Nextsafeaction: readnewstatus/results nextsession, compare anchored vscontrol acrossboth
seeds includingepoch0, phraseWER/exact, single-source retention and selectedtrainfit.
No livepromotion based solely on auxiliary loss falling; step5stilldepends on quality,
holds/repeats and latency. Currentapp/livecode/model defaults untouched; no protectedtest,
acquisition or blanket extraction. Allfive recovery steps retained.

## 2026-09-22 — matched diagnostic completed; timed-anchor refinement prepared

User asked faster progress and explicit transition learning. Replayed all56approvedASLLRP
contiguous clips with all4selectedcheckpoints in9sMPS; checkpoint hashes and whole-sequence
exact counts match prior saved results. Contextual token-classifier core evidence on59train/
17validation eligible intervals: frozen30/59and10/17 bothseeds; adapted50/59,12/17 and51/59,
9/17. Adapted whole validation exact0/12both. FRIEND MAYBE: frozen can have both corelabels
correct but outputSCHOOL/blank; adapted outputsOTHER MAYBE MAYBE orSCHOOL MAYBE and its
FRIEND core evidence isYEAR/WE. Both identity and emission problems remain; not proof that
blank-only training fixes them. Gap-located spikes are timing observations, not certified
false-positive signs. Core evidence is not independent isolated classification.

Newdiagnostic scripts/diagnose_combined_transitions_v17.py and
artifacts/reports/combined_transition_diagnostic_v17_20260922/{REPORT.md,results.json}.
The diagnostic initially compared one-based predictions to zero-based annotation IDs;
corrected before interpreting results. This was diagnostic-only, not an old training bug.

Prepared scripts/prepare_transition_anchors_v17.py and recipe-scoped anchor JSON:
59train central sign-core regions/27internal gaps,17validationcores/3gaps across41admitted
clips. Known targets use lockedlabel+1; blanks0only internalgaps between consecutiveeligible
known events with0.05s margins and no overlap with ANYannotation. Verified no native-frame
subsampling inall56clips; originalwindowextractor disables trimming. Independent contract/
anchor reviewclear; source/feature/annotation hashes pinned. Blankalignment is a bounded
hypothesis, not physicalrest truth. Oldgeneric gate untouchedfalse.

NewCONTRACT.md and blocked active/v17/transition_anchors_manifest_20260922.json define
3epochs perheadarm/seed, selectedadaptedStage1frozen, cache richevidence once forspeed,
CTConlycontrol versusCTC+0.25timedintervalCE; epoch0selection allowed. Newrunner/test in
review; no newtraining launched yet. Need focusedchecks, finiteMPSpreflight and reviewed
codepins before detachedlaunch. No live/application changes or newdatasetacquisition.

## Aggregate clarification — 2026-09-22

Previously reported60.89/66.85% frozen and50.84/53.26% adapted WER referred only to199local phrase validation clips. Token-weighted combined phrase validation (211clips,561reference signs): frozen60.96/66.49%, adapted50.98/53.65%. All admitted validation sources combined (1874records,2224reference signs): frozen38.67/39.03%, adapted27.56/32.69%. Single-sign validation alone (1663records): frozen31.15/29.77%, adapted19.66/25.62% CTC WER. Aggregate computed as total substitutions+deletions+insertions divided by total reference signs, not mean source WER. It describes only this recipe's validation membership, not all datasets in the repository. The aggregate is dominated by single-sign examples and cannot stand in for continuous performance.

Next-step recommendation is a bounded matched identity-versus-sequence diagnostic, not an established unique best intervention. Existing gap proves weak generalization under this evaluation, not its cause. Compare whole-sequence versus annotated-core outputs with the same selected weights, documenting any frontend/resampling differences; review whether failures follow source/input conditions before choosing new training or extraction. No additional training authorized by this clarification alone beyond the existing recovery scope.

## 2026-09-22 — combined comparison complete, no promotion

User requested update. Saved status complete; all four checkpoint SHA256 hashes match,
12epochs/arm, earliest-best selection matches saved validation, all epochs4264unique
single visits/536phrase visits. Selected epochs frozen9/11, adapted8/3. Local validation
WER frozen60.89/66.85%, adapted50.84/53.26%; adapted local trainingWER0.48/1.74%, exact
validation29/199 and22/199. Strong remaining generalization gap; not a working continuous
model. ASLLRP phrase validation adapted54.17/62.50%WER, zero exact of12both. Citizen
pooled Stage1 validation95.24%initial→94.71/94.18%; ASLLRPsegmented53.54%→86.61/82.68%.
These are validation-source metrics, not protected-test or globally signer-disjoint claims.
Review artifacts: combined_frozen_joint_v17_20260922/REVIEW.md and REVIEW_SUMMARY.json.
No live promotion/retraining. Next: selected-checkpoint per-record identity/alignment
analysis on existing annotated examples; do not simply add epochs. All five steps retained.

## 2026-09-22 — navigation centred and its backing bar removed

User request. `nav_rects` now lays the pills out through the shared `row_rects` helper
with `align="center"` instead of its own left-aligned loop, so there is one layout
routine rather than two, and `clicked_nav` follows automatically because it has always
read the same rect table. `draw_nav` no longer paints the full-width panel behind the
bar.

Without that panel the labels sit directly on whatever is beneath them, which on the live
page is a camera frame that may be bright. Inactive labels are therefore drawn with
`draw_shadowed_text` in white at alpha205 rather than flat MUTED; the active page keeps
its filled ACCENT pill with INK text. Checked against a dark backdrop, a bright poster
grid and a lit camera frame.

One test had pinned the old geometry by hard-coding `clicked_nav(30, 28, 1280) ==
"HOME"`; it now derives the point from `nav_rects`, and a second test asserts the row is
centred within two pixels rather than against the left edge. `nav_height` is unchanged, so
`top_inset` and every page's content offset are untouched. Tests:218 passing across14
modules; `git diff --check` clean.

## 2026-09-22 — EXPERIMENTAL badge removed from the app at user request

User asked for the badge gone everywhere. Removed: the home-screen footer line and the
word from the practice setup footer. No other UI text carries it. The app now states
nothing about promotion status.

This is a UI label only. The repository's own record is unchanged and still governs:
`PROJECT_GROUND_TRUTH.md` continues to mark Live/streaming as "Nothing promoted", and
session records still carry `score_semantics: "uncalibrated Stage-1 softmax with temporal
agreement"`. Two footers still note that scores are uncalibrated — the history page and
the practice result page — since the request named the experimental badge specifically.

While making the change, `scripts/app_pages_v17.py` was found edited outside this session
with `_footer` stripped of its `note` parameter while its `draw_text` call still expected
the text argument and four call sites still passed one; every page render raised
`TypeError: _footer() takes 5 positional arguments but 6 were given`. Restored the
parameter and dropped the home call site instead. Tests: 217 passing across 14 modules;
`git diff --check` clean.

## 2026-09-22 — practice stripped to one sign at a time; home mosaic rejected

User direction: remove reset, buffer and finish triggers from practice; add a skip
button; remove the home-screen photo mosaic, which looked bad.

`ReelHud.draw` gained `minimal: bool = False`. In minimal mode it omits the committed
gloss rail, the naturalised sentence, the running CTC hypothesis (in both places it was
drawn) and the RESET/FINISH control row, and extends the lower panels into the freed
space. The default is byte-identical to before, asserted in test. A practice round judges
one sign, so a running buffer is not just clutter, it is the wrong model of the task.

Finish has three trigger paths and all three are now closed during a round: the buttons
are not drawn and clicks on the practice page never route to `clicked_reel_control`; `r`
and `f` are swallowed by the shell; and the ten-finger gesture is gated by a new
`shell.allow_finish`, checked in `scripts/live_reel_stage1_v17.py` beside the existing
`args.no_finish_gesture`. `NullShell` reports `allow_finish=True` and `minimal_hud=False`,
so the standalone script is unchanged. After every judged commit the shell queues an
internal reset, so each attempt starts from an empty buffer without exposing a control.
Verified end to end on a real clip: score1/1, **zero utterances** (no finish fired) and
one reset event in the session history.

Skip added as a visible "SKIP THIS SIGN" button where the reel's controls used to sit,
sharing one code path with the `n` and space keys; it counts an attempt, records the sign
as missed and clears the buffer. `overlay_live` now returns its rect table like every
other page, so pointer and paint cannot disagree — the same discipline that the invisible
nav bar earlier showed is necessary.

Home mosaic removed. Replaced with a drawn aurora: three soft colour fields drifting over
a dark vertical wash, composited at 1/12 scale and upscaled, so there is no corpus
imagery on the home screen at all and nothing to re-license if it is screenshotted.
Cost5.77ms per frame after switching the upscale from cubic to linear, which is
indistinguishable on a smooth field 12x up (9.93ms before).

`run()` changed only by the `allow_finish` gate and the `minimal` pass-through;
equivalence re-checked — the same six clips decode identically through pre- and
post-change modules, 6/6, on every recorded prediction field. Tests:218 passing across14
modules (21 new); `git diff --check` clean. A live camera run remains the only
unexercised path.

## 2026-09-22 — user-requested partial training update

One requested status snapshot, not background polling: status=running; seed17421 frozen
finished12epochs, selectedepoch9. Adapted arm latest savedepoch2/12; secondseed has no
saved history yet. Frozen selected-checkpoint local phrase knownWER train9.19% versus
validation60.89% (13/199exact); ASLLRPcontiguous validation62.50% (0/12exact).
Citizen validation pooled Stage1top1 stays95.24% while CTC single exact84.92%.
Frozen data mixture alone has not solved phrase recognition; adaptation comparison is
incomplete. No causal claim against old unmatched recipes and no deployment change.
Next: await completion notification, then review both seeds and selected-checkpoint
train/validation retention. No further polling performed for this update.

## 2026-09-22 — gallery switched to local recordings; scroll and home screen fixed

User direction: prefer their own recordings in the gallery, fix a dead scroll wheel, and
make the home screen less plain.

Gallery source changed to the exact-variant local audit pool
(`data/local/local_citizen100_quality_audit_q82_cap14_exact`), chosen over the wider
`q82_cap7` pool (89 classes) because it is restricted to classes whose canonical label and
pinned raw gloss agree — the safeguard against displaying the wrong articulation. Result
is **77 local /23 ASL Citizen train**, still100/100 covered, rebuilt in8.2s. All four
local audit pools were already free of scraped msasl/signasl/wlasl clips; those were
excluded when the audits were made. No local pool reaches100: `I` is quarantined in every
one of them because that folder visibly mixes fingerspelled I with ME. Citizen keeps its
train-only guard, its hash and its landmark-diagnostic ranking; local clips are ranked by
the audit's own quality score and rendered whole, since they have no hand-activity window.

Scroll wheel was never broken by the backend — `_on_mouse` returned early on anything that
was not `EVENT_LBUTTONUP`, so wheel and motion events were discarded before anything saw
them. Now handled, but OpenCV's macOS Cocoa backend cannot be relied on to emit them at
all, so scrolling also has clickable ▲▼ buttons on the scrollbar, arrow keys, w/s, k/j and
[ ] for page jumps. The arrows are the guarantee; the wheel is a bonus.

Home screen is now an animated poster mosaic: a wall larger than the window built once
(67ms) from the100 gloss frames, with each frame taking a drifting crop of it, one tile
cross-fading to a different sign every~2.2s, a staggered ease-in for title and buttons,
and a hover halo. Costs7.28ms per frame, comfortably inside the browse loop's budget.
PRACTICE was added to the home menu as well as the nav. A bug was caught in test: with no
entry timestamp the ease-in left the whole page fully transparent; absent an entry time it
now renders settled.

Provenance was removed from the gloss detail page per user choice — the gallery mixes
project recordings with corpus footage, and the audit states no trustworthy local signer
IDs exist, so quoting one would have been invention. The page shows gloss, category and
clip length only. The gallery footer no longer claims "ASL Citizen train split".

`run()` was not touched this round; equivalence re-checked anyway — the same six clips
decode identically through pre- and post-change modules, 6/6, on every recorded prediction
field. Tests:197 passing across14 modules (19 new); `git diff --check` clean. A live
camera run remains the only unexercised path.

## 2026-09-22 — combined comparison MPS preflight passed; detached launch

First preflight exited134 on Apple MPS matmul dtype assertion. Verified isolated/windowed
caches are float16 and positive-core cachesfloat32. New runner had cast only CTC chunks;
fixed make_records once to widen all cached inputs tofloat32, covering preflight/identity/
evaluation paths without geometric normalization. Failed attempt JSON/log preserved.
Second preflight passed all6421records,7367full-model windows per arm; largest32single
records use41windows and4phrases21windows. Frozen base gradients absent; adapted base
and both heads finite/nonzero. Zero optimizer steps. Four focused tests including dtype,
ID collision, repeated CTC/OTHER and normalized weighting passed; independent final
review clear, git diff --check clean. These preparation defects do not establish causes
of earlier trained-model failures.

Pinned recipe SHA b077e6939c6dc71c581cea2ee669348e2b4cc1d1cedef50eb7f8a9fd85c4ee64;
runner3c1ea6c7cfb4e7698eace0839d9a70de897a329335819f4400b65b4fd7b7a9fb;
preflight3ce024fca29bbb43ce3765270efeea7a11b520d432d0d055fc433850cae76999.
Launched dedicated --train detached under caffeinate, PID98647,2026-09-21T17:45:51.436154UTC
(2026-09-22PHT). Two seeds, two arms,12epochs each. Completion/failure notifications wired.
No training polling or completion claim. Reports/models use combined_frozen_joint_v17_20260922.
Current ground truth and all-five checklist updated; source datasets/live defaults and
old generic gate unchanged. Next safe action: read status/results next session, compare
selected-best training versus validation and single-sign retention before any deployment.

## 2026-09-22 — bounded combined frozen/adapted comparison prepared

User continued the authorized five-step recovery. New recipe and TRAINING_CONTRACT.md
pin the actual Reel landmark proposal checkpoint, cleaned6421-record combined manifest,
unchanged normalized32-frame inputs, unpooled temporal CTC, and single-sign identity replay.
Two seeds17421/17422, frozen/adapted arms,12epochs, same initialization/sampling/dropout
policy,4264single visits and536phrase visits per epoch. This is a cleaned-data comparison;
prior experiments already used joint adaptation/unpooled tokens and this initialization.
No blanket re-extraction, live replacement, protected-test access or acquisition.

New files: scripts/train_combined_frozen_joint_v17.py,
test/test_combined_frozen_joint_v17.py,
active/v17/combined_frozen_joint_manifest_20260922.json and recovery TRAINING_CONTRACT.md.
Recipe-specific training readiness requires matching preflight before launch; the old
approved494manifest remains training_ready=false, verified SHA a821cae8d443e1bf3bc6649deb82dbea7a3e03d217b163360bffda6168390d1c.

Pre-launch review caught/fixed target-length CTC normalization, OTHER-preserving exact
scoring, per-arm RNG reset, selected-best metric provenance and repeated O5S5 signer IDs
that would otherwise overwrite190examples. These were new-runner defects caught BEFORE
any training, not evidence explaining historical model failures. Unique feature paths
now key records. Added complete input/dependency pins, finite full-model/gradient checks,
atomic saves, per-epoch component losses/coverage and selected-checkpoint full-train metrics.
Four focused tests pass; pending actual MPS preflight and final review. All five recovery
steps remain tracked in PLAN.md; successful preparation is not a continuous-model result.

## 2026-09-22 — app startup fixed, practice drill added; three shipped bugs corrected

User feedback after trying the app: starting the camera felt like launching a new
process, and there was no back button. Both were real defects, not perception.

Nothing ever spawned a process. The delay was measured at **10.96s of model construction
inside `run()`**, paid on every entry into the live page, while the warm-up thread only
pre-imported the module (1.12s) — it warmed the cheap half and left the expensive half in
front of the camera. Fixed by factoring construction into `build_components(args)` in
`scripts/live_reel_stage1_v17.py` and adding `run(..., prebuilt=None)`. The shell now
builds models on a background thread behind an already-interactive home screen, then
enters the recognition loop **once** and never leaves it: every page is a paused state
inside that loop. Measured after: warm-up6.52s off the critical path, entering the
loop→attach0.22s, →first frame0.42s; page switches are state changes only. `run()` is
still not refactored.

Second defect: the nav bar was never painted on the live page. `present()` showed the HUD
frame unchanged while `top_inset` reserved an empty56px strip, and `clicked_nav` was still
consulted on mouse-up — navigation was functional but invisible, which is worse than
broken. `pages.overlay_live()` now paints it. A related off-by-one surfaced in test: PIL's
`rounded_rectangle` includes its bottom edge, so the bar covered57 rows against a
reserved56 and ate the first row of frame content; corrected, with a test asserting
nothing below the strip moves. Third bug, also caught in test: advancing to the next sign
cleared the verdict, so a celebration would have rendered a tick with no gloss; the
verdict is now kept and expires on its own timer.

Added per user direction: PRACTICE as a nav page running a scored drill (choose5/10/20,
one sign at a time, result screen naming the misses), the reference clip as a corner
overlay looping continuously with no pacing throttle, a full-frame celebration on a
match, and a chime via `afplay` with silent fallback (`--no-practice-sound`). A tile's
SIGN THIS NOW starts a one-sign round. Practice pages needing the camera report
themselves unpaused; the rest still skip all detection and inference.

Recognition output re-verified after this second round of changes to `run()`: the same six
clips decode identically through pre- and post-change modules on every recorded prediction
field — 6/6. A scripted practice round through the real models recognised EAT and scored
1/1. Tests:178 passing across14 modules (18 new this round); `git diff --check` clean. A
live camera run remains the only unexercised path.

## 2026-09-22 — five-step recovery authorized and first diagnostics executed

User approved starting all five recovery steps and useful parallel work, including
evidence-based re-extraction/retraining. Persistent PLAN.md and measured REPORT.md are
under artifacts/reports/continuous_recovery_v17_20260922. Steps1–2started;3–5remain
dependent, not forgotten or declared complete. Two read-only routine workers reviewed
paired audits/input contracts after fast_scan model proved unavailable.

Fresh unchanged-loader verification passed6421records/7367windows,6158raw video hashes
and7source-manifest pins in9.766s. General494manifest passes integrity, remains blocked
for old recipes. Phrase train covers44known labels; single-sign replay100. No proven
contract defect; no blanket extraction. Native1280px caches and prior failed ablation
were checked before proposing reuse.

Created lossless FFV1 oracle crops from4annotated sources (3train/1validation), with
pixel-identical roundtrip and pinned source hashes.12unchanged live-CTC full/core probes
all exited0with identical model provenance; logits/features/times saved. Train controls
NOW READ/FAMILY IMPORTANT/FAMILY SIGN and six cores exact. Validation source selected
before outputs: FRIEND MAYBE→FRIEND STOP; cores→FAMILY/STOP. One reused development
example, not an accuracy score or unseen-exposure claim. No parameter tuning.

Current Reel oracle probes on8cores use default20fps/640px observations, bypassing live
activation/duration/commit gates. MAYBE proposal/verifier top labels correct, verifier
rejects low score/margin. FRIEND proposal SAME, verifier COLD/rejected; other identity
and rejection errors recorded. Different weights/frontends from legacy CTC: no head-only
causal claim. Boundaries alone do not repair all errors; raw identity, acceptance and
final emission need separate measurement. Runtime/source/model files unchanged by us.

Added recovery plan/report, input/coverage/verification JSON, probe inputs/results/status,
logs, lossless diagnostic cores, replay histories/videos and traces. Updated current
state and this log; regenerate artifact index after outputs. Next: matched-current-Stage1
input/objective recipe with combined roles, plus real hold/repeat/low-motion evidence
before training/live promotion. No acquisition, training, protected test, split changes
or old-gate bypass. Concurrent app-shell changes preserved.

Handoff check detected a concurrent Reel script hash change after CTC verification.
Repeated only8Reel oracle probes with before/after code pins:8/8proposal/verifier labels
and acceptance decisions reproduced, code stable during recheck. Original evidence
retained; reel_oracle_recheck.json is the reproducible reference. git diff --check passes;
artifact index regenerated. No product code changes by the recovery task.

## 2026-09-22 — demo app shell complete; recognition output proven unchanged

All six planned phases of the user-directed demo app are done. `scripts/app_shell_v17.py`
is a new entry point with home, a 100-sign gallery, session history and the live feed as
pages of one window, plus "sign this now" practice. It does not fork or reimplement
recognition: the reel script keeps its own loop and now calls `shell.present()` once per
frame, so the shell only decides what that frame becomes. The whole change to
`scripts/live_reel_stage1_v17.py` is 42 insertions/18 deletions in 2,139 lines: a
`shell=None` parameter defaulting to `NullShell` (the old single-page behaviour), window
and pointer ownership moved to `shell.attach`, the display tail replaced by
`shell.present`, `top_inset=shell.top_inset`, and a paused branch that skips detection,
inference and recording while another page shows but still services language and speech
so a finish in flight completes. `run()` was deliberately not refactored.

No accuracy claim changes and nothing is promoted. Proven, not asserted: six gallery
clips decoded through the pre-change and post-change modules produced identical
`hypothesis` and identical per-prediction `gloss`/`committed_gloss`/`model_score`/
`accepted`/`margin`/`gate_score`/`end_seconds`/`frames` — 6/6 identical. HUD pixel hashes
at three geometries are unchanged from the phase-1 baseline
(`0481a012…`/`9d0fd6f8…`/`40580520…`); render cost1.270ms median versus1.258ms before.
An integration test navigates to gallery and history mid-session and back, and the clip
still decodes to DRINK.

Gallery examples: `scripts/build_gloss_examples_v17.py` scores all1,476 ASL Citizen
*train* clips from their v17 landmark diagnostics alone, decoding nothing, in0.8s, then
renders only the100 winners (11.9s,27MB). Coverage is100/100 with2 edge-flagged
(THINK, HUNGRY) and0 missing;18 distinct participants; every row carries participant and
sha256. `rejections.csv` is honoured explicitly because the rejected SLEEP clip is still
present in the train pool, while `quarantine/w_h_a_t/` already sits outside `raw/`.
Validation and the sealed test split are never read, asserted by a test that greps the
source. The selection total ranks candidates within one gloss only: hand_presence counts
both hands, so one-handed signs floor near0.5 and totals are not comparable across
glosses.

Sessions write to `artifacts/app_sessions/` with an mtime-keyed index, deliberately not
`artifacts/reports/`, so demo runs are never mistaken for experiment provenance. That
directory and `artifacts/app_assets/` are gitignored; Citizen footage is not ours to
redistribute. The live page keeps an EXPERIMENTAL badge and the history page states that
scores are uncalibrated.

Tests: 160 passing across14 modules (74 new). `git diff --check` clean. Added
`scripts/app_shell_v17.py`, `scripts/app_pages_v17.py`, `scripts/app_sessions_v17.py`,
`scripts/build_gloss_examples_v17.py` and their tests; changed
`scripts/live_reel_stage1_v17.py`, `scripts/reel_hud_v17.py`, `.gitignore`. Next safe
action is a manual camera run, the only path not exercised headlessly. Note: this file's
"50 entries" header is stale (137 dated entries present); left untouched because other
sessions are appending concurrently.

## 2026-09-22 — product clarification and read-only streaming diagnosis

User confirms locked100 recognition of individual/naturally connected signs, no required
pauses/resets, stable visible output;0.5–1s delay acceptable, faster preferred. English
rendering may use additional words but must not invent meaning. Reported Reel low-wrist-
motion misses and continuous-Reel transition insertions; separate legacy Stage2 live
trial also unsatisfactory. Discussion only; no model/runtime edits or training.

Inspected current Reel/continuous wrapper and Stage2 input/output paths. Reel's candidate
activation uses wrist-only motion, submit requires active=True, and motion trimming uses
the same observation motion. This can prevent or distort low-wrist-motion evidence;
individual reported misses are not yet timestamp-attributed. Ordinary Reel scores are
normalized over known labels, and temporal stability alone is not transition rejection.
Stage1 uses temporal attention/convolution before pooling. Clean phrase baseline used
pooled-window frozen evidence, whereas legacy CoreML Stage2 retains unpooled temporal
features; neither the blanket claim of no sequence modeling nor pooling as the sole
failure cause is supported. Prior joint/contextual/4-state experiments already exist.

Targeted JSON analysis of live_stage2_ctc_v17/20260922_005306_195364/history.json:
117 predictions,75 accepted/non-stale,41 insufficient-hand-evidence rejections,24 resets.
Epoch7 output at26.01/27.06/28.15s is HELLO / HELLO TAKE / KNOW HAVE TIRED.
Configured window span1.0667s; accepted processing median169.50ms, excluding capture
and window accumulation. No saved emission trace or timestamped human reference: no
session WER, exact failure attribution, or evidence that all41 rejections lost signs.
The UI displays the revisable context hypothesis, separately from stable speech logic.

Research revisited original Squeezeformer (arxiv.org/abs/2206.00888), VAC
(arxiv.org/abs/2104.02330), and EMNLP2024 online CSLR
(aclanthology.org/2024.emnlp-main.619/). The latter already informed prior project work;
its contextual-window/background approach must not be presented as an untried solution.
Its benchmark results are not ASL/iPhone evidence; learned alignments are not certified
physical transition labels. Combined6421manifest is still untrained mixed supervision,
not6421independent phrases. Historical native-rate/no-blank and joint-CTC failures remain
relevant; no claim that unfreezing, a new head, or threshold relaxation alone fixes this.

Next safe action: discuss a controlled comparison separating complete-sign recognition,
stream localization/emission and signer transfer, reusing existing diagnostics/recordings
before any new recipe. Define stable-output latency from sign evidence through display,
keep product evaluation distinct from exact boundary scoring and isolated retention,
and preserve protected test sealing and explicit supervision roles. Files changed only:
this log and PROJECT_GROUND_TRUTH.md (user requirements/current discussion scope).

## 2026-09-22 — reel HUD promoted to the canonical app toolkit; no model change

User-directed demo work: the live path gains a home/gloss-gallery/history shell, and
`scripts/reel_hud_v17.py` is the single HUD every new file draws through. Phase 1 is
additive only. The private primitives (`_panel`, `_chip`, `_text`, `_shadowed`, `_wrap`,
`_dot`, `_scale`, `_text_width`, `_chip_size`) were renamed to public names with short
aliases retained, so no existing call site changed. Added: `canvas`/`frame_canvas`/
`to_frame` (one composite path whether or not a camera frame is behind the page), `hit`
plus `row_rects`/`grid_rects`, and `NAV_PAGES`/`nav_rects`/`draw_nav`/`clicked_nav`.
`ReelHud.draw` gained `top_inset: int = 0` so a shell can reserve the nav strip.

Nothing here reads or changes model state; no checkpoint, feature contract, threshold or
decoder path was touched, and no accuracy claim changes. Verified by pixel hash at three
geometries (720x1280 `0481a012…`, 481x641 `9d0fd6f8…`, 400x700 `405805207…`), identical
before and after. Render cost median1.258ms before versus1.255-1.315ms across five runs
after: within run-to-run noise, no regression. `top_inset=0` is byte-identical to the
previous render and an inset leaves the control row untouched, both asserted in tests.
Tests: 16 HUD (12 new), 38 reel, 35 other live paths, all passing; `git diff --check`
clean; the reel CLI still resolves.

Next safe action is the phase-2 gloss-example generator, which reads
`data/local/citizen100_v17/raw/train/` only and never val or the sealed test split.
Deliberately deferred: `run()` in `scripts/live_reel_stage1_v17.py` stays unrefactored;
the app will drive it through a per-frame shell hook rather than extracting an engine
from its ~1,100-line closure body. Changed: `scripts/reel_hud_v17.py`,
`test/test_reel_hud_v17.py`.

## 2026-09-21 — matched no-blank CTC ablation failed promotion

Precheck found that the existing aligned grounded CTC run already used source-rate
1280px ASLLRP, native30fps signer-disjoint local phrases, an8-frame causal window,
stride4, the exact-core-adapted Stage1 initialization, locked100+OTHER outputs and timed
alignment. The audit's proposed native-rate CTC was therefore not repeated.

The only missing controlled change removed standalone transition clips labeled blank.
It completed18 MPS epochs in106s and selected epoch8. Local held-out-signer performance
regressed from25.5% exact/37.04% WER to16.0%/45.19%; ASLLRP contiguous remained33.33%/
41.67%; NCSLGR regressed5.41%/78% to0%/84%. Isolated exact improved slightly from
82.37% to83.19%. The checkpoint fails promotion and runtime remains unchanged. These
results reject blank clips as the dominant cause and close another decoder/supervision
patch. Citizen test and external reserved evaluation stayed sealed. Report:
`artifacts/reports/native_ctc_no_blank_v17_20260921/`.

## 2026-09-21 — data-path audit identifies the next minimal Stage-2 experiment

The source videos and annotations remain usable; the continuous preprocessing contract
does not preserve them faithfully. The continuous observer caps detection at640px and
20fps even though all audited ASLLRP videos are1280px on the long side at29.97fps. The
current confident filter then rejects6,020/11,936 events under its four/six-observation
floors. This compounds the already measured context-window defect where all6,641 known
windows ended before the sign end. Exact-core held-out ASLLRP accuracy of72.95%/82.35%
on other/contiguous data confirms that sign identity survives better than localization.

The next experiment is one small causal frame-sequence CTC head with an exact-core
classification auxiliary, trained from a native-rate1280px ASLLRP cache plus the
existing signer-disjoint local phrase split. Short signs stay in through masks rather
than arbitrary sample floors. O5S5 remains positive-core supervision only. Do not add
another boundary state machine, slow/interpolate cached sequences, or acquire replacement
data before evaluating this corrected input contract. Audit:
`artifacts/reports/stage2_data_path_audit_20260921/`.

## 2026-09-21 — boundary tolerance was strict, but it was not the detector's only failure

The source/Luna review shows that±100ms exact-edge scoring should not be used to declare
ASLLRP annotations bad. The official source interval and the single-reviewer Luna
interval differ partly in semantics, especially final holds. However, the coherent
decoder still reached only40.46% F1 at±200ms and0/20 intentional-repeat probes, so the
failed promotion is unchanged. Review:
`artifacts/reports/luna_boundary_annotation_pilot_20260921/comparison.html`.

## 2026-09-21 — frozen coherent decoder improved insertions but still failed

Three user-requested low-effort Luna agents independently audited the failed segment
head, existing activity code and timing. A Luna implementation then froze the checkpoint,
averaged overlapping-window state evidence on absolute source timestamps, applied the
existing v16-style hand-presence outer range and required coherent
OUTSIDE→START→SIGNING→END paths with END→START repeats. Edge bias was selected only on
888 training sources; the unchanged0.56 known gate and232 validation sources were used
once. Predictions overlapping excluded annotations were ignored consistently.

The evaluation improves±100ms boundary F1 from14.44% to18.79% and±200ms F1 from24.41%
to40.46%. Visible WER improves204.69%→103.65%, mainly because insertions fall261→33,
but deletions rise66→121; substitutions are45 over192 references. Precision/recall is
20.96%/17.04% at±100ms and45.12%/36.67% at±200ms. Synthetic probes worsen to2/20 held
once and0/20 repeated twice. Thus incoherent peak pairing caused many insertions, but the
learned boundary evidence itself remains inadequate. Do not promote or patch further.
Verification confirms D+I+S arithmetic, train-only selection, sealed Citizen test and
unchanged frozen checkpoint. Report:
`artifacts/reports/segment_first_coherent_decode_v17_20260921/REPORT.md`.

Next authorized action is a small Luna-reviewed offline boundary-annotation pilot. Use
dense train-only frame strips, require independent agreement and admit only consensus
edges before deciding whether this can scale. Luna output is provisional supervision,
not ground truth, until agreement is measured.

The user then required exactly one Luna per clip. The completed24-clip pilot produced19
medium/high, non-censored provisional intervals, but only6/19 match both existing edges
within100ms and14/19 within200ms; median absolute end disagreement is128ms. With one
reviewer per clip there is no independent consensus, and the disagreement is not a
consistent timing offset. Do not train or scale these pseudo-labels as truth yet. Pilot:
`artifacts/reports/luna_boundary_annotation_pilot_20260921/`.

## 2026-09-21 — segment-first confident-only experiment completed and rejected

The corrected detached run completed eight full-coverage MPS epochs in5m05s. An
initial18-second startup attempt failed before epoch1 because the runner required every
SemLex replay class even though SemLex train lacks CHILD/TAKE/THEY; the failed evidence
is preserved under `failed_attempt_01/`. The loader now follows the established rule:
all100 Citizen classes are mandatory and available SemLex classes are optional. The
expanded precheck exercises the exact replay loaders and counts500 Citizen train,480
SemLex train,378 Citizen validation and978 SemLex validation examples.

The model fails promotion. Across232 held-out complete ASLLRP sources, absolute boundary
F1 is14.44% at±100ms and24.41% at±200ms. At±100ms precision is9.44% and recall30.69%;
at±200ms precision is15.96% and recall51.89%. Visible locked100 WER is204.69% with66
deletions,261 insertions and66 substitutions over192 reference glosses. Exact-core
known/unknown balanced accuracy is63.34%; exact-core gloss is169/226=74.78%, including
ASLLRP contiguous81.82%, ASLLRP other81.22% and O5S538.24%. Matched predicted-segment
gloss is76/106=71.70%. Isolated validation is Citizen92.33% and SemLex83.23%. Synthetic
time-warped probes pass6/20 held-once and2/20 intentional-repeat cases.

Verification passes3/3 focused tests, full coverage and64/64 CPU/MPS gloss and gate
decisions. Citizen test stayed sealed. Do not promote the checkpoint or change runtime.
The result confirms that clean whole-sign identity is substantially more learnable than
automatic boundary localization; this four-state frame head does not solve segmentation.
Report: `artifacts/reports/segment_first_v17_20260921/REPORT.md`.

## 2026-09-21 — segment-first confident-only experiment launched

User authorized the minimal segment-first experiment with one exit notification and no
polling. Precheck passed the exact builder over all5,331 confident events:7,788 training
boundary windows,4,303 continuous training segment cores,1,797 validation boundary
windows and1,028 validation segment cores. It verified the locked100 vocabulary,
signer-disjoint roles, questionable-annotation masking, zero O5S5 background targets,
real MPS execution and a tiny fit from1.613 to0.198. Focused tests pass3/3.

Detached PID42246 is training for eight full-coverage epochs from the plain Stage-1
checkpoint. The model has one OUTSIDE/START/SIGNING/END linear head, one internal
KNOWN/UNKNOWN head and the existing100-gloss classifier. The final report will measure
absolute boundary F1, rejection, predicted-segment gloss accuracy, visible WER,
isolated retention and held/repeat probes. No CTC, alphabet expansion, gap-derived
transition target, runtime promotion or Citizen-test access is part of this run.
Report directory: `artifacts/reports/segment_first_v17_20260921/`.

## 2026-09-20 — strict whole-sign audit completed: ASLLRP identity learnable, localization fails

Detached PID55540 completed in85.7s. It evaluated8,367 strict whole-sign cores and
42,111 complete-ASLLRP endpoint windows without training. Verification passes;12
failure/control videos and a contact sheet were generated and inspected. Citizen test
remained sealed.

The adapted encoder recognizes strict ASLLRP-other cores at901/1,040=86.63% train and
178/244=72.95% held-out, versus the starting encoder31.15%/51.23%. Limited neighbouring
context lowers adapted held-out accuracy to67.21%, so missing context is not the current
identity bottleneck. Contiguous strict-core adapted accuracy is88.14%/82.35% train/
held-out. This establishes meaningful learnability for clean ASLLRP sign evidence.

O5S5 remains a source/signer failure: adapted strict-core accuracy is101/133=75.94%
train versus10/43=23.26% LG held-out; cleaning incomplete/overlapping events does not
close it. Boundary localization remains inadequate: train-selected phase threshold
has43.14% held-out known recall and79.10% non-known recall (61.12% balanced), original
three-way phase accuracy49.79%, and correct gloss+phase detects152/261=58.24% eligible
held-out ASLLRP events. Gloss confidence alone is59.57% balanced. Gaps remain annotation
gaps, not certified physical transitions.

Decision: keep locked100; reject checkpoint. Use strict whole-sign manifest for identity
work and separately reviewed boundary evidence for localization. Do not infer a physical
TRANSITION class from annotation gaps. Reports and playable examples:
`artifacts/reports/clean_boundary_subset_20260920/`.

The follow-up confident-only manifest further restricts supervision to at least six raw
target observations,80% hand visibility, maximum1.20s duration and maximum80ms target
timestamp gaps. It retains5,331 events, including1,017 known and4,314 unknown, across
61 train classes. Use this manifest for the next controlled experiment; do not silently
fall back to the combined supervision manifest. Report:
`artifacts/reports/confident_supervision_v17_20260920/REPORT.md`.

## 2026-09-20 — strict whole-sign recognition/localization audit authorized and prechecked

User authorized the corrective report-only experiment with exit notification and no
polling. It will evaluate the unchanged phrase-adapted encoder and failed boundary-phase
checkpoint on one strict derived subset. ASLLRP rows with lost crop-completeness are
excluded; every retained event is non-overlapping, inside the raw clock, has at least
four raw target observations, and covers both annotated edges within60ms. O5S5 supplies
exact sign cores only, never gap/boundary evidence. Annotation gaps remain named gaps,
not asserted physical transitions. Train-only thresholds will be applied unchanged to
held-out signers.

Full dry run retained8,367/11,936 annotated cores:1,536 known and6,831 unknown, plus
42,111 fixed0.27s endpoint windows from complete ASLLRP crops. Exclusions include3,122
events with fewer than four target observations,60 outside the raw clock,141 within
incomplete source crops,357 overlapping annotations and65 without both-edge coverage;
reasons may overlap. Real MPS precheck and100-label/output contracts pass. The worker
will report whole-core and whole-core-plus-context recognition, known/unknown/gap
rejection, event localization, and selected playable failures. No training, protected
test access or automatic promotion. Runner/precheck:
`artifacts/reports/clean_boundary_subset_20260920/`.
Detached MPS worker PID55540 launched with one completion/failure notification; do not
poll it.

## 2026-09-20 — direct boundary-data audit corrects earlier mismatch claim

User requested concrete diagnosis of cropped signs, learnability, and alphabet expansion.
Compared all 1,166 manifest rows with original ASLLRP CSV/crop provenance, inspected
sampled frames from six examples, and created ten original-speed source/input excerpts.
No training ran. Re-evaluated final checkpoint on all 38,043 train and 9,380 validation
context windows; validation known counts reproduce 708/1,275. ASLLRP-other known gloss
train/held-out accuracy is 88.14%/59.90%, phase 69.33%/51.04%; O5S5 known 68.39%/32.02%.

21 ASLLRP source crops were previously marked incomplete (22 clipped occurrences, all
OTHER); materialization did not preserve that flag. All 5,366 known train windows end
before sign completion, 2,349 have less than half target observations, 31 known events
produce no window, 46 O5S5 window endpoints overlap another annotation, and two FEEL
windows contain zero raw target samples. These are not physically cropped exact cores.
11,428/11,731 TRANSITION windows overlap preceding signs: valid in principle for an
endpoint task, but annotation gaps were not visually certified as physical transitions.
Training uses selected event/gap moments; replay samples continuously and fuses durations.
Thus previous claims of no remaining mismatch and architecture failure were too strong;
corrected original report and current state, preserving model and metrics.

ASLLRP train OTHER has 5,070 lexical occurrences (83.26%) and 377 fingerspelled (6.19%)
out of 6,089. Adding alphabet classes would not resolve ordinary unsupported vocabulary.
73 known classes occur in continuous train annotations; 16 have one signer. Data has
learnable signal, but no proof of reliable unseen-signer continuous recognition. Next
safe action: curate complete and unambiguous evidence with reviewed boundaries, then
separate recognition from localization on that fixed subset before further training.
Report, measured fit, source audit, videos, and verification:
`artifacts/reports/boundary_data_audit_20260920/`. Citizen test remains sealed.

## 2026-09-20 — clean boundary-phase experiment completed and rejected

Detached MPS worker PID95985 completed all12 full-coverage epochs in24m52s. The fixed
0.27s/0.53s trailing-window model used39,944samples and625updates per epoch. Its locked
100-gloss plus separate KNOWN/UNKNOWN/TRANSITION contract and saved CPU/MPS decisions
passed64/64 checks; Citizen test remained sealed.

The result fails runtime use: validation phase accuracy50.23%, known-core gloss55.53%
(708/1,275), and complete-ASLLRP online WER106.82% with86 deletions,110 substitutions,
133 insertions and55/237 empty transcripts. Isolated Citizen/SemLex fell from the
starting checkpoint's95.24%/85.28% to93.39%/84.76%. Therefore the earlier failures
cannot be attributed only to replacement sampling,1.07s dilution, combined NO_EMIT
competition, or train/live window mismatch. Short independently classified windows do
not yet generalize reliable phase boundaries and gloss identity. Reject checkpoint;
default runtime unchanged. Report:
`artifacts/reports/boundary_phase_v17_20260920/REPORT.md`.

## 2026-09-20 — boundary-phase fixed-window experiment launched

User authorized the researched non-CTC experiment with one exit notification and no
polling. It keeps the phrase-adapted 100-gloss Stage-1 model and adds one three-way
endpoint phase head: KNOWN, UNKNOWN or TRANSITION. Both training and online replay use
the shared 32-frame normalization at0.27s/0.53s; phase is defined at the trailing-window
endpoint. Held identical glosses stay one run, and the same gloss may repeat only after
a predicted transition. UNKNOWN is internal and cannot enter the visible transcript.

Strict routing is executable: all1,160 complete ASLLRP rows can supply known cores,
explicit OTHER intervals as UNKNOWN, and guarded interior gaps as TRANSITION; the six
incomplete O5S5 rows can supply known cores only. Full normalization dry run produced
38,043train and9,380validation context windows, with no O5S5 negative target. Every
admitted training sample will appear exactly once per epoch; no replacement sampler,
1.07s target or CTC loss is used. All1,166 raw archives,100 label indices, source and
signer gates, model output shapes, and real-MPS tiny-fit passed. Citizen test remains
sealed; no automatic runtime promotion. Runner/precheck:
`artifacts/reports/boundary_phase_v17_20260920/`.
Detached MPS worker PID95985 is running with one completion/failure notification;
do not poll it.

## 2026-09-20 — Finish-time CTC completed: lower WER masks deletion collapse

Detached MPS run completed20 epochs in10m16s with full coverage each epoch. The frozen
Stage-1 + one400,174-parameter bidirectional GRU CTC head reaches83.80% connected and
88.03% familiar WER versus repaired causal98.24%/107.72%, but deletes215/284 and
191/259 reference glosses. It emits only74/284 connected and68/259 familiar tokens;
160/225 connected outputs are empty and familiar exact is0/97. Contiguous WER83.33%,
LG core2/57, isolated Citizen344/378 and SemLex812/978, verified blanks16/16 empty.
Six adjacent duplicate outputs remain versus three references; this is not an expert
held-once/twice evaluation. Locked100 visible-label contract verified; blank/UNKNOWN
internal, protected test untouched, no promotion.

Root-cause review found the initial reporter summed alignment `match` operations into
WER. Added a failing regression, centralized the three edit operations, regenerated
evaluation/report and independently recomputed every source. Model/checkpoint/predictions
were unchanged. Ten focused tests and compilation pass. Report/correction/verification:
`artifacts/reports/finish_ctc_v17_20260920/`. Decision: keep current runtime; the simple
frozen-head recipe underfits continuous training (best71.34%, final74.05% WER), so the
next work must improve continuous evidence/coverage rather than add decoder heuristics.

## 2026-09-20 — Finish-time bounded CTC experiment authorized and ready to launch

User chose the researched bounded continuous direction and authorized the experiment
with exit notification and no polling. The fixed spike freezes the Stage-1 encoder and
trains one bidirectional GRU CTC head on the existing complete phrases, isolated clips,
verified positive cores and verified blank clips using one objective. Internal outputs
are blank + locked100 + UNKNOWN; only the100 glosses can reach the visible transcript.
Seed17201,20 full-coverage epochs, fixed final evaluation against the repaired causal
CTC on the same334 development recordings; Citizen test remains sealed and no runtime
promotion is automatic. Nine focused tests, compilation and real MPS packed-GRU
forward/backward pass. Runner/precheck:
`artifacts/reports/finish_ctc_v17_20260920/`. Final action is a detached launch with one
completion/failure notification; do not poll.

## 2026-09-20 — direct-translation root cause established: caption memorization

Completed a read-only audit of frozen data, features, losses, outputs and pairing
controls. BART reproduces a 994-row training caption exactly on 11/12 held-out clips;
mT5 has median nearest-training-caption similarity0.803 and six of12 at least0.80.
BART zero-visual emits one exact training caption for all12 rows. Correct pairing does
affect chrF, so the visuals weakly route caption choice, but neither hybrid composes a
translation. The994 pairs cover1.63h/four signers; two supply80.4%. Extraction is not
globally broken:6/4,813invalid windows,98.7%median hand presence,72/994downsampled.
Eight rows exceed six English words/s and are viewer flags, not proven bad labels.
Root cause is the isolated-window encoder/linear bridge trained end-to-end with an
oversized decoder and insufficient aligned continuous data. Do not rerun another LM
swap on this recipe. Keep gloss-free work separate; current Stage2 needs one temporal
blank+100-gloss head and signer-disjoint hold/repeat/rest/transition supervision.
Report/viewer: `artifacts/reports/direct_translation_failure_diagnosis_20260920/`.
No training or protected test access. Fresh verification passes report self-check,
Python compilation,44video-panel HTML audit, JSON parsing, artifact indexing and
`git diff --check`.

## 2026-09-20 — English comparison complete; independently verified negative translation result

User requested update. No active workers. All20 BART epochs completed,10552.19s.
Recomputed BART/trim/mismatched BLEU+chrF from saved outputs: exact agreement.
Trimmed mT5 keeps21/21 predictions identical,303.52M vs589.39M parameters (-48.5%).
BART146.41M: BLEU1.061/chrF17.464 vs mT5 1.076/18.989; fails1-point chrF margin.
Mismatched BART BLEU1.287 exceeds paired1.061, weakening grounding claim. Inspected
predictions contain unrelated content. Isolated Citizen358/378, SemLex841/978.
No deployment promotion; no new training. Appended interpretation to REPORT and
updated canonical state. Next safe action: discuss alignment/generalization diagnosis,
not another unexamined model swap. Twelve correlated sentences insufficient for
accuracy equivalence; official Citizen test untouched.

## 2026-09-20 — English comparison exit

English model comparison completed; see english_comparison_20260917/REPORT.md. No model promotion; review predictions and controls.

## 2026-09-20 — approved English comparison resumed after storage available

User asked to continue. Baseline already completed epoch20 with poor translations;
no active worker. Internal free58.5GiB now exceeds startup guard. Verified4,295
frozen input/code/model/tokenizer hashes and completed baseline checkpoint SHA.
No missing assets or recipe changes. Prior failed queue records archived under
english_comparison_20260917/failed_attempt_05; precheck JSON records verification.

Proceed with approved trim evaluation and one BART pilot, internal storage only;
existing completed mT5 is a weak comparison baseline, not a deployment candidate.
No model/code modifications or unnecessary training rerun. Final action: detach
comparison worker with terminal notification, no assistant polling. Need final
predictions, visual controls and isolated retention before interpreting results.

## 2026-09-18 — final hybrid results verified; English comparison storage-blocked

User-requested update: baseline completed epoch20, no active processes. Independently
recomputed saved initialized/final/zero-visual BLEU and chrF, exact matches. Final
1.076/18.989 vs unchanged Uni-Sign8.206/38.059; training loss0.16688. Citizen isolated
360/378 unchanged; SemLex832/978 vs834 initially. Inspected translations contain
unrelated details. Automated weak descriptive-check pass is not translation success.
Appended review to REPORT.md and replaced stale epoch13 current-state text.

English comparison failed in startup checkpoint-space guard before trim evaluation
or BART training. Internal free7.7GiB (<8GiB); SSD not mounted. Do not lower safety
threshold or restart blindly. Next: adequate storage and assessment of poor baseline
before progressing the approved comparison. No checkpoint changes/test-gate access.

## 2026-09-18 — English comparison exit

English model comparison failed; see english_comparison_20260917/FAILURE.md. No model promotion; review predictions and controls.

## 2026-09-18 — direct-translation worker exit

Stage 1 direct translation completed. See stage1_direct_translation_20260917/REPORT.md. No production promotion. Inspect saved predictions, training coverage, retention and controls before deciding the next action.

## 2026-09-18 — user-requested update: baseline saved epoch14; comparison waiting

One requested status check confirms live MPS baseline worker, completed/saved epoch14
of20, all994 translation/1901 isolated samples covered that epoch; translation loss
0.34381 (epoch13 0.41154). Current provenance says recovery fromepoch13, checkpoint
root internal artifacts/models, consistent with current ground truth. No new failure
record. Comparison supervisor45994 waits on baseline45878; BART has not started.
Corrected stale comparison status.json from failed to queued/waiting; canonical
state was already queued. No running code, model or training changes; no polling.

## 2026-09-18 — epoch13 saved; external EIO at epoch14; verified internal recovery

User requested update. Saved epochs8–13 after file-handle repair; SSD returned EIO
at epoch14 save/flush. No workers active and no BART training. Epoch13 training
loss0.411538/isolated0.004517, not validation scores. Copied SSD latest to internal
epoch_13_before_internal_resume.pth; source/copy hashes, all ZIP CRCs, and epoch13
mmap metadata verified. All valid source checkpoints retained; no SSD writes in retry.

Both runs now target internal artifacts/models. Internal free14GiB before2.37GB
recovery copy. Split storage gates:8GiB startup for model loading versus4GiB running
checkpoint free space (known2.37GB maximum), plus1GiB internal report reserve.
Added regression for phase-specific limits, archived failed attempt/source manifests,
recorded internal_recovery.json and EPOCH13_RECOVERY_REPORT.md, refreshed code hashes
without data/architecture/optimizer/recipe changes. Next: focused checks, epoch14
resume and comparison event queue; no polling. Final evaluation still pending.

## 2026-09-18 — English comparison exit

English model comparison failed; see english_comparison_20260917/FAILURE.md. No model promotion; review predictions and controls.

## 2026-09-18 — direct-translation worker exit

Stage 1 direct translation failed. See stage1_direct_translation_20260917/FAILURE.md. No production promotion. Inspect saved predictions, training coverage, retention and controls before deciding the next action.

## 2026-09-18 — reproduced SSD native-writer failure; full-size file-handle alternative passed

User requested update. No live workers. Baseline reached epoch8 save then rename
failed ENOENT; temporary file now exists at exactly1,879,048,192 bytes but invalid
ZIP. Epoch7 remains valid; BART never started. Disk free14GiB internal/280GiB SSD.
Reproduced direct-filename torch.save failure using synthetic2.4GB CPU storage test.
Same-size Python binary file-handle save + flush/fsync/rename passed full ZIP CRC
verification (2,400,001,577 bytes,27.80s). Evidence JSONs and SSD_WRITE_REPORT.md saved.
Root PyTorch/ExFAT defect not identified; observed API-path difference verified.

Shared baseline/BART checkpoint writer now uses tested file-handle path and distinct
latest.filehandle.tmp.pth (old corrupt output retained). All valid checkpoints kept.
Archived attempt04 baseline and03 comparison; refreshed code/baseline hashes without
data/recipe changes. Compilation/diff checks pass. Next: resume fromepoch7 and queue
comparison, with notifications and no polling.

## 2026-09-18 — English comparison exit

English model comparison failed; see english_comparison_20260917/FAILURE.md. No model promotion; review predictions and controls.

## 2026-09-18 — direct-translation worker exit

Stage 1 direct translation failed. See stage1_direct_translation_20260917/FAILURE.md. No production promotion. Inspect saved predictions, training coverage, retention and controls before deciding the next action.

## 2026-09-18 — SSD launch failed on internal ENOSPC before training; reserve raised

User requested update. No live worker; last verified epoch remains7. SSD recovery
loaded model then failed writing provenance and failure/status due internal ENOSPC.
Checkpoints on SSD alone did not prevent internal exhaustion during model loading;
swap/memory pressure is a likely contributor, not separately instrumented. Comparison
never trained. Current internal free space observed16GiB, SSD282GiB. Archived attempt
03 baseline and attempt02 comparison, keeping all checkpoints.

Raised internal-space gate from512MiB to8GiB; external8GiB/mount guards unchanged.
This is conservative headroom, not a guarantee against concurrent disk consumption.
Refreshed frozen code/baseline-manifest hashes, data unchanged. Next: verify focused
recovery checks and relaunch from epoch7, requeue comparison without polling.

## 2026-09-18 — English comparison exit

English model comparison failed; see english_comparison_20260917/FAILURE.md. No model promotion; review predictions and controls.

## 2026-09-18 — direct-translation worker exit

Stage 1 direct translation failed. See stage1_direct_translation_20260917/FAILURE.md. No production promotion. Inspect saved predictions, training coverage, retention and controls before deciding the next action.

## 2026-09-18 — approved corrupt-temp cleanup and verified SSD recovery

User explicitly permitted deleting corrupt latest.tmp.pth and using attached SSD.
Deleted only that invalid ZIP (2,063,536,128 bytes). All valid local checkpoints
retained. Verified writable ExFAT SSD /Volumes/secret (UUID recorded), ~283GiB free.
Copied epoch7 to SSD; SHA matches original. Actual mmap load and small atomic
checkpoint replacement verified on SSD.

Changed baseline runner for external checkpoint root, verified recovery record,
epoch7 optimizer restoration, mount/path guards, >=8GiB checkpoint free-space and
>=512MiB internal report-space checks before work/epochs/saves. Comparison runner
uses external root and recovered baseline launch record. Added test_ssd_recovery_v17;
red tests exposed original epoch1-only restore and absent space guards, then five
focused recovery/queue/label tests pass. Archived previous scripts/manifests/failed
queue terminal files. Updated frozen code hashes in both manifests while asserting
all input hashes unchanged. Recovery report/record/verification preserve provenance.
Next: detached epoch8 resume and event-based requeue; no polling. Keep SSD attached.

## 2026-09-17 — disk exhaustion identified; verified epoch-7 recovery checkpoint

User reported failed comparison. Baseline actually failed saving epoch 8, then
errno28 prevented status rename; stale status/GT said training/running. No active
baseline/comparison process. Comparison stopped on failed prerequisite and never
started BART or trim evaluation. Assistant missed the prerequisite failure when
queuing. Verified latest.pth ZIP CRCs and mmap metadata: epoch7, history1–7,
543 optimizer states. latest.tmp.pth is an invalid ZIP (~1.9GiB); free disk ~3GiB.
No files deleted. Archived failed status/logs under failed_attempt_02 and corrected
canonical/status files. Added comparison/DISK_FAILURE_REPORT.md and diagnosis JSON.
Next: obtain explicit permission for corrupt temporary checkpoint deletion per
AGENTS.md, prepare epoch7 recovery with disk-space checks, and requeue comparison.
Keep all valid checkpoints. Current resume helper is hard-coded to epoch1 and
must not be blindly rerun. No final translation-quality result yet.

## 2026-09-17 — English comparison exit

English model comparison failed; see english_comparison_20260917/FAILURE.md. No model promotion; review predictions and controls.

## 2026-09-17 — English comparison prepared and reviewed; queue launch next

Prepared pinned BART-base weights/tokenizer, 64,000-entry mT5 tokenizer and explicit
row mapping from 500 Brown documents plus TRAIN-only requirements. All protected
training/prefix/basic-character token sequences and decoded strings preserved.
BART target mean 21.56, maximum 63, no truncation. Frozen assets/code/data manifest
and corpus SHA recorded under english_comparison_20260917. No new GPU work yet.

Added vocab_trim.py, run_comparison.py and focused tests; reused existing fixed
baseline loading/training/evaluation without modifying its source. CPU check covers
retained mT5 logits exactly; queue test uses a real short child process exit event.
Review corrected supervisor exit-code propagation and canonical-status updates;
second tokenizer/integration review found no further high/medium issues. BART repeat
suppression explicitly disabled. PRECHECK_REPORT.md explains approval, recipe and
limits. Next action after final checks: detach queue on existing baseline supervisor
exit; no polling. Stop/report if baseline fails or fails the zero-visual grounding
screen. Otherwise trim-evaluate, preflight/reset, train one BART and report results.

## 2026-09-17 — user approved English comparison; preparing event-based queue

User approved the documented BART-base pilot and no-retraining mT5 vocabulary
comparison, requested results only and no polling. Current baseline still training,
last checked epoch 5; do not overlap GPU jobs. Preparing new report-local runner
`artifacts/reports/english_comparison_20260917/run_comparison.py`; reuse existing
frozen-input loader/train/eval helpers without changing their source or live defaults.
Queue waits for baseline supervisor process exit with kqueue NOTE_EXIT, then verifies
successful completion and baseline zero-visual grounding before full new training.
Focused queue/bucket tests failed first for absent runner, then uncovered Python 3.9
kqueue lacks a context manager; fixed with contextlib.closing, now 2/2 pass.
Pinned BART download started; no new GPU computation/optimizer steps yet. BART's
released no_repeat_ngram_size=3 must be overridden to 0 to honor allowed repetition.
Independent trim helper/test and bounded runner review underway. Next action: verify
CPU preparation and freeze hashes, then detach comparison queue; no polling.

## 2026-09-17 — English text-model alternatives researched; approval pending

User requested a review before any further training. Added report-local
`english_text_model_review_20260917/{REPORT.md,inspect_candidates.py,verification.json}`,
official pinned configs, inspection log and `bart_api_check.json`. Parameter counts
use meta allocations only; train-target statistics use the 994 frozen TRAIN
references. No evaluation references used for vocabulary design and no vocabulary
actually trimmed. No new pretrained weights, GPU benchmark, optimizer step or new
training run. Existing run and its files unchanged.

Verified current text parameters 582.40M; embeddings/output 384.17M (65.2% of the
589.39M hybrid). BART-base hybrid 146.41M; T5 v1.1 and FLAN base 254.57M; FLAN small
83.88M. Hypothetical 64K / 32K mT5 trims 303.52M / 254.57M. Target mean 24.66,
median 23, p95 45, maximum 70, padding 64.8%. Tiny random CPU BART compatibility
forward/generation passes through existing DirectTranslation; this is not accuracy
evidence. Official model cards and vocabulary-trimming research cited in REPORT.

Recommendation for approval: one BART-base English challenger plus no-retraining
trimmed-checkpoint comparison after completed mT5 results. Full recipe and proposed
retention margins documented; 12 existing sentences cannot establish equivalence.
FLAN is not strictly English-only. No speedup or mobile-readiness claim. Next action
is user review/approval, not launch. Preserve all existing training and data gates.

## 2026-09-17 — user-requested resumed-run throughput check

Saved provenance confirms MPS, float32, 589,390,373 parameters. Resumed worker
exists and no completion/failure file is present. Epochs 2/3/4 took approximately
1,003/1,132/1,123 seconds (497 updates each, 2.02–2.28 seconds/update), versus
142 seconds for projection-only epoch 1. Full encoder/text training, batch two,
fixed masked padding and twice-per-step MPS synchronization/cache release explain
why this is substantially heavier than warmup; individual overhead contributions
have not been profiled. At recent throughput, 16 remaining epochs take roughly
five hours plus final evaluation, not a guaranteed completion time. No training
changes or restart; continue detached and inspect results after exit notification.

## 2026-09-17 — epoch-2 MPS failure diagnosed; checkpoint resume verified

Initial run completed epoch 1 (994 paired / 1,901 isolated examples, 497 steps,
translation loss 2.902774, isolated loss 0.065100, 142.043 seconds), then failed in
Adafactor with MPS memory exhaustion. Train-only tiny-fit and gradient checks passed;
no final translation-quality result. Preserved original run artifacts under
`stage1_direct_translation_20260917/failed_attempt_01/` and epoch-1 checkpoint as
`epoch_01_before_resume.pth`.

Changed `active/v17/direct_translation_v17.py`, its focused test and the report-local
runner: fixed masked source/target padding, unused MPS cache release around optimizer
updates, checkpoint alias preservation, verified epoch-1 resume and stale completion
archiving. Two real-network tests pass (including padded/unpadded loss equivalence).
Four maximum-length real joint train-only updates passed; updates discarded.
Observed driver allocations at measurement points <=10.33 GB before / 8.78 GB after
cleanup; not continuous peak measurements. Inputs, architecture, objective and
20-epoch recipe unchanged; all non-code manifest fields match original. Epoch 2
restarts with seed + epoch, not the interrupted partial epoch RNG state. Added
RESUME_REPORT.md and hashed resume_provenance.json. No test-gate access or promotion.
Next action: detached resume, then inspect exit report after notification; no polling.

## 2026-09-17 — direct-translation worker exit

Stage 1 direct translation failed. See stage1_direct_translation_20260917/FAILURE.md. No production promotion. Inspect saved predictions, training coverage, retention and controls before deciding the next action.

## 2026-09-17 — approved Stage-1 direct-translation pilot prepared

User authorized the hybrid experiment, accepted repetition for this stage, and
requested results with detached execution/no polling. Kept the existing experiment
branch and all unrelated changes. Added `active/v17/direct_translation_v17.py`,
`test/test_direct_translation_v17.py`, the report-local `run_experiment.py` and
`docs/superpowers/plans/2026-09-17-stage1-direct-translation.md`.

Architecture preserves unpooled, ordered Stage-1 frame tokens, projects them to
the released ASL mT5 input, and directly generates English. The existing isolated
head is trained with auxiliary CE. No CTC, gloss decoding or repeat heuristics.
Full text-component state is required to load strictly from the previously hashed
Uni-Sign How2Sign checkpoint. This is a hybrid pilot, not native Uni-Sign replication.

Prepared 994 paired TRAIN utterances (How2Sign signers 3/5/8/11), excluding 32 clips
from adaptation-held signers 1/2 and one previously failed feature extraction.
Reused existing Apple continuous features. Extracted 21 evaluation videos under
the same schema; corrected reused extractor bookkeeping to mark validation access.
Retained verified 1,901 isolated TRAIN / 1,356 validation entries. All 4,272 input
hashes check; max source tokens 256, max English target tokens 70, no target
truncation. No official Citizen test access. External mT5 pretraining may include
evaluation signers; whole-system signer disjointness is not claimed.

Two real-network checks failed first for the missing new module, then pass after
implementation: variable sequence masks, encoder/projection/text gradient flow,
isolated-head gradient, target-free generation, and deterministic full coverage.
Initial data preparation rejected SemLex's different train/val root convention;
corrected source-specific path checks and reran successfully before training.
Added an external subprocess wait to report fatal worker crashes without polling.

Fixed 20 epochs (one projection warmup, 19 joint), seed17111, Adafactor, translation
CE + 0.5 isolated CE, all admitted samples used each epoch. Full-model train-only
gradient/tiny-fit preflight runs inside the worker and resets before main training.
Compare final epoch with initialization, unchanged Uni-Sign and zero-visual input;
report isolated retention, losses, coverage and all translations. Repetition is
not a failure gate. Final action is detached launch; next action after notification
is to inspect REPORT/FAILURE. No production replacement or model-quality claim.

## 2026-09-16 — clarify Uni-Sign adaptation potential during discussion

User observed that some translations are close and asked about fine-tuning,
architecture lessons and isolated-sign failures. Rechecked official model/training
code and paper plus saved predictions. Some full-sentence outputs preserve useful
content; exact-match count must not be interpreted as semantic accuracy. Rejection
applies to unchanged deployment, not fine-tuning feasibility. The architecture has
separate downstream isolated-recognition fine-tuning; the evaluated checkpoint is
the How2Sign sentence translator. Short-clip failure is confounded with domain,
signer and capture differences. Added this interpretation to REPORT.md. No training
or implementation started. A proposed adaptation comparison needs correctly paired
target-domain data and unseen-signer evaluation; keep the native architecture first.

## 2026-09-16 — reviewed Uni-Sign results; reject drop-in replacement

On the user's status request, verified completed status and all 21 predictions;
independently recomputed BLEU 8.206457 and chrF 38.059096 over the 12 paired
development utterances. Strict checkpoint loading succeeded. Nine diagnostics
remain unscored for English accuracy because they lack expert English references.
Outputs include invented details (“I really like that” → “And, I like that from a
spoon”) and a brush sentence on the HELLO HOW YOU annotated clip. Webcam EAT gives
“I'm sorry, I'm sorry.” These results do not support replacement of Stage 2/Reel
or a claim of corrected repetition. They do not identify the dominant error cause
or disprove gloss-free translation generally. Updated REPORT.md and current state.
No production changes or training. Next safe action remains verified target-domain
hold/repeat examples before another model change. No new background job launched.

## 2026-09-16 — Uni-Sign worker exit

Detached Uni-Sign baseline completed; see `artifacts/reports/unisign_asl_baseline_20260916/REPORT.md`. Measured 12 paired development utterances and nine diagnostics: BLEU 8.21, chrF 38.06.
No training or production promotion. Next action: inspect outputs and limitations before deciding whether this challenger helps.

## 2026-09-16 — authorized pretrained Uni-Sign comparison prepared

User approved one inference-only gloss-free ASL comparison and asked to return with
results, retaining the earlier detached-work/no-polling instruction. Added the
report-local `unisign_asl_baseline_20260916/run_baseline.py`. Reuses existing download
and hashing helpers, official model/pose code at eed438bcb49e30405cd6ccdfcccca330c134e830,
and the released How2Sign checkpoint at eab251b7fe7e8521afc0e67be98add670ea40a0d.
Full strict state loading is required; mT5 is initialized from its configuration
before loading the complete checkpoint, avoiding an unused base-weight download.

The available local How2Sign subset is training data. Selected instead the pinned
validation mirror at 1231830fc1e8d77555a245ca22353b144e168111: first four eligible
source names in raw validation shard001, at most two sources per filename signer,
first three sorted sentence IDs of duration 2–12s each. Metadata/remote ZIP-directory
precheck confirms 12 utterances, four sources, signer IDs 1 and 2. This is a small
development slice, not signer-disjoint generalization. Raw clips will be cut at
manually realigned CSV boundaries; original sentence clips have different timing.
Nine prior difficult recordings have no expert English targets and stay qualitative.

Installed ONNX Runtime 1.19.2 and its missing logging dependencies in a separate
`data/local/tools/unisign_native_deps` target; main environment unchanged. Native
pose-only AST adapter excludes unused CUDA training imports without changing the
selected definitions. Self-check passes shape/direct-output equivalence, confidence
masking and empty inference targets; compilation and git diff --check pass.
No model-quality result yet. Final action is detached launch, with REPORT/results/
summary/provenance or FAILURE, completion record, history update and macOS exit
notification. No assistant polling, training, runtime replacement, or Citizen test
access. Next safe action after notification: inspect measured outputs and limitations.

## 2026-09-16 — replay result and decision; repeat errors survive beam decoding

Confirmed completion of all 40 runs over 10 recordings. Rollover correction changes
zero final outputs; corrected live/offline differ in three long-context runs.
Window-origin sensitivity affects both count and identity; original I I did not
reproduce, and saved-video re-extraction cannot recover original camera features.

Following the user's Stage-3 cleanup suggestion, ran one fixed existing beam8/topk12
control on the new saved emission matrices (no tuning/training). It changes 2/40
outputs, leaves all EAT/WHEN repeat examples unchanged, and increases labelled
diagnostic edit errors 32→34 across 32 reference tokens. These 16 labelled runs
are four correlated phases of four selected clips. Exact phase-zero WHEN sequence
log probabilities: single -6.460, double -0.00621; the repeated preference is in
the model emissions. Greedy outputs were checked against all saved report rows.

Added report-local `check_beam.py`, `beam_control.json`, `DECISION.md`; updated
current ground truth. No production, model, annotation or training changes.
Stage 3 may improve wording, but equal strings alone cannot establish intended
event count. Do not implement blanket repeat removal or promote beam decoding.
Next comparison chosen: one released pose-only Uni-Sign ASL checkpoint at utterance
level with native preprocessing and fixed paired development references, before
fine-tuning. That comparison is not launched; no new background job exists.

## 2026-09-16 — boundary correction and detached diagnostic replay prepared

User authorized the proposed diagnosis/correction sequence and requested ending
the turn after launching long work, with completion/failure notification and no
polling. Created branch `codex/stage2-held-sign-diagnostics` in the existing checkout,
preserving all pre-existing changes. No worktree/data/checkpoint duplication.

Added a regression demonstrating that a continuing raw CTC run is emitted again
at a rolled context start. It failed before implementation. `collapse_ctc_path`
now accepts the preceding raw token; both live callers retain that token from the
exiting window and clear boundary state on Finish/reset. 51 focused CTC/Reel tests
pass, including the actual Reel event-loop fixture spanning multiple rollovers
and preservation of blank/OTHER-separated repeats. This is a narrow runtime
correction, not proof of held-sign accuracy or a fix for early short-context errors.

Changed `scripts/live_stage2_ctc_v17.py`, `scripts/live_reel_stage1_v17.py`,
`test/test_live_stage2_ctc_v17.py`, `test/test_live_reel_continuous_v17.py`.
Opt-in `--ctc-trace-dir` records compressed emissions, frozen window features and
source observation times, with exclusive file creation and trace latency included.
Reel events include raw path, boundary token, context offset and emission file path.

Prepared report-local `run_replay.py`, `manifest.json`, timestamp sidecars and a
long webcam excerpt in `stage2_held_sign_diagnostics_20260916`. Frozen 10 video
hashes/readable streams/timestamp arrays verified; 40 runs compare legacy/corrected
collapse on identical logits and final single-context/stitched output. Four corpus
references are scored; partial O5S5/webcam remain diagnostics. This re-extracts
saved low-resolution video and cannot recreate original features or true live latency.

The worker records REPORT/results/summary/provenance plus completion JSON, or a
failure report, and sends one macOS notification on exit. It does not poll or
automatically reopen the chat. Launch is the final action of the turn; inspect
its exit notification/completion report in the next interaction. Training is not
launched: independent human hold/repetition labels and a justified objective are
still missing. Plan: `docs/superpowers/plans/2026-09-16-stage2-held-sign-diagnostics.md`.

## 2026-09-16 — held-sign diagnosis, video evidence, and gloss-free research

Completed the user's research request without another runtime patch or training run.
Read the saved `live_reel_stage1_v17/20260915_220852_978390` session and verified the
general selector SHA. It predates/does not load the repaired joint epoch-42 model.
Sampled-frame review finds one chest-point episode decoded as I I at two accepted
windows (token positions 6,10), before Finish or eight-window rollover. Two separate
salutation episodes produce HELLO HELLO and are retained as a repetition control.
Gesture descriptions are provisional, not expert ASL labels. Recorded Stage 3 also
changes I to “Is that?” and I I to “Is it?”.

Production CTC helper assertions verify held A A A collapses once and A blank A
twice. An independent synthetic counterexample proves prefix rollover can count a
single ongoing A run twice. This is not the cause of the early two-window duplicate.
76/130 session sequence updates are rejected; only accepted windows enter context.
No raw logits were saved, so the reason the model fragmented the early gesture
cannot be established retrospectively. Current continuous wrapper defaults differ;
the report describes both ordinary Finish review and revisable sequence mode.

Created `artifacts/reports/stage2_research_review_20260915/`: standalone nine-video
viewer, exact annotation/event rows and provenance JSON, report-local builder/checks,
contact sheets, seven full primary papers and selected official Uni-Sign source.
The viewer separates recorded live predictions, research continuous CTC predictions,
and pooled exact-core predictions. LG HELLO (0.46s) and WHEN (0.88s) are misclassified
as LESS/SIGN even with exact core boundaries. Existing partial/unverified annotations
are labelled honestly. Webcam frame indices map through saved source timestamps;
two trailing unmapped frames are excluded. No protected test clips were accessed.

Reviewed GFSLT-VLP, Sign2GPT, FLa-LLM, Uni-Sign, SHuBERT, simultaneous SLT and the
2026 sentence-finalization preprint. Screened SignLlama's abstract and the standardized
comparison's abstract/official code notes; comparison PDF requests timed out. Released
Uni-Sign pose-only ASL checkpoints are the closest structural challenger, but their
133-keypoint preprocessing/mT5 stack is not compatible with Apple v17 features.
Gloss-free requires video/English pairs and would replace Stages 2+3; it does not
automatically resolve online event counting or demonstrate iPhone readiness.

Decision: first freeze a signer-disjoint hold/true-repeat/rest/OTHER evaluation and
capture framewise logits/absolute time; isolate model emission errors from rollover
ownership. Retain one encoder/one temporal head as the simple recognition candidate;
evaluate a released gloss-free ASL model separately before considering replacement.
No promise that speed changes, bulk data, or a new architecture alone will solve it.
Verification: report builder passes its production-helper counterexamples, selected
source/model hashes, nine valid media exports and webcam frame-count checks. Gallery
links, embedded annotations, JavaScript syntax, Python syntax and research-checkpoint
hash pass. Contact sheets inspected; browser interaction and expert linguistic review
remain unperformed. Large-artifact index regenerated; git diff --check passes.
Verification details are saved in the report's `verification.json`.
Next safe action: deliver the annotated viewer and report for discussion.

## 2026-09-15 — repair handoff checks complete

Fresh final checks pass:33focused tests,23Python files compiled, report consistency
assertions, code whitespace andgitdiffcheck. Large-artifact index regenerated.
Completion audit maps the authorized training-collapse repair to measured full-training,
held-out, provenance and boundary evidence. Results remain a failed-promotion research
checkpoint; general continuous recognition and LG remain unsolved. All requested repair
experiments/results are complete; no additional training or deployment action remains.
Artifacts:joint_ctc_aligned_v17_20260915/{README.md,verification.json,handoff_checks.json,
completion_audit.json}. Next action:return the verified result candidly.

## 2026-09-15 — CTC repair verified; training collapse resolved, generalization still fails

Finalepoch42evaluated once on all334frozen development recordings after training-only
corrections. TRAIN1,088/1,291known matches,32.77%WER,1,901/1,901isolatedCTC;263/334
development recordings emit known signs. Connected WER98.24% (65S/90D/124I), familiar
107.72% (112S/74D/93I), contiguous50% (3S/6D/3I). ActualCitizen/SemLex validationCTC
339/378 (89.68%)/762/978 (77.91%); pooled95.77%/85.28% retained. LG pooled15/57 but
actualCTC4/57with42empty; trainingcores199/199bothoutputs. Generalization remains weak.

Verified4,520input hashes,54checkpoints, all frozen recipes/full schedules/optimizer
states, recomputed finaltraining and developmentcounts/gates. CPU/MPS628/628tokens
agree,63tokenprefix consistent.33focused tests pass. Source-clock latency misses
158/296signs; runtime excluded. Failedpromotion gates:connecteddeletions,familiarWER,
medianruntime delay,runtimeinclusion. No Citizen test, confirmation, export or promotion.
Finalreport:joint_ctc_aligned_v17_20260915/README.md; canonicalcurrentstate updated.
Decision: training-collapse repair complete, no more tuning from these held-outerrors.
Next product work must address signer/domain generalization and sequenceprecision;
this checkpoint is not a deployment candidate. Final handoff: compile/whitespace,
report consistency and large-artifact index, then return verified results.

## 2026-09-15 — final alignment completed; held-out evaluation started

All six fixed epochs37–42completed. Final TRAIN known matches1,088/1,291 (84.28%),
169deletions,34substitutions,220insertions,32.77%WER; isolated CTC1,901/1,901 (100%).
Compared with the original failed joint2/1,291known matches, the CTC training collapse
is substantially repaired; training sequence accuracy is still imperfect. Final model
has4,620updates across42epochs; the rejected first12epochpositive-only run is separate.
Launched the predefined single finalepoch42evaluation on the frozen development suite.
No held-out outputs informed loss corrections or checkpoint selection. Next: inspect
all334recordings/isolated/LG results, independently verify provenance and metrics,
record limitations and report the completed repair result without promoting failures.

## 2026-09-15 — verified-frame phase completed; final alignment launched

All12epochs25–36completed. Final TRAIN:1,263/1,291known matches (97.83%),3deletions,
25substitutions,1,370insertions,108.29%WER; isolatedCTC1,897/1,901 (99.79%). The
sign evidence is learnable, but extra outputs still dominate. Launched the predefined
six-epoch37–42alignment continuation with original target-normalized CTC restored,
retaining all verified supervision and optimizer state. Loss routing verified directly;
no held-out evaluation has informed any correction. Next: complete final alignment,
evaluate fixedepoch42once, verify provenance/metrics and report measured limitations.

## 2026-09-15 — final alignment curriculum prepared from training-only evidence

Verified-frame epoch28 recognizes1,186/1,291known training signs, but2,387insertions
remain. A controlled original-CTC+CE optimization stays blank from a low-logit start
(p=0.062706) but preserves a strong nonblank solution from the warm start(p=0.999520).
This supports temporary frame normalization to escape collapse, followed by restoring
original sequence alignment strength once sign evidence has been learned.

Prepared a fixed six-epoch37–42continuation after currentepoch36, preserving all
verified-frame/anchor/pooled/background objectives, model/optimizer/learning rates/data.
Only original target-normalized complete/positive CTC functions are restored. No
held-out outputs or decoder changes inform the choice. Minimal runner/plan:
`joint_ctc_aligned_v17_20260915/`; it reuses the verified-frame training loop. Next:
complete current run, execute the final alignment pass, then evaluate fixedepoch42
once and verify all results. No further tuning from held-out errors is planned.

## 2026-09-15 — balanced continuation completed; verified frame refinement audited

All12continuation epochs13–24 completed. Final TRAIN:1,117/1,291matches (86.52%),
25deletions,149substitutions,2,336insertions and194.42%WER. Isolated CTC1,891/1,901
(99.47%). Blank suppression is removed, but sequence precision still fails.
No held-out outputs were inspected. The next dense-frame audit admits150,876training
tokens:23,454known,103,251OTHER,24,171guarded blank;86,112tokens remain ignored.
The exact0.10s interior-gap guards are reused; clip edges/overlap/unguarded contexts
remain unsupervised. Independent reviewer hit a usage limit before this review;
root directly checked helper/loop routing, global frame population scaling, resume,
freeze/evaluation and copied imports. No blocking issue found. Next: focused checks,
then fixed25–36refinement followed by final held-out evaluation and full verification.

## 2026-09-15 — emission locations expose missing continuous-context supervision

Epoch16 TRAIN location probe samples32recordings per source. ASLLRP OTHER produces
129known emissions for39references:35inside known intervals,52inside OTHER intervals,
42in gaps. These are timestamp-location diagnostics, not event-alignment accuracy.
The current recipe teaches one sequence point per sign and blank on standalone
background windows; it leaves most OTHER motion and short guarded gaps without direct
sequence-frame supervision. Existing complete annotations can supply those labels.

Prepared `joint_ctc_frame_supervision_v17.py` and two regression tests: known/OTHER
intervals,0.10s guarded interior gaps, incomplete-annotation refusal, overlap/padding
masking, gradients and partition-invariant population normalization. Both pass.
The new recipe adds only this sequence CE after the current fixed epoch24 completes,
retaining prior objectives/optimizer for25–36. Runner/plan:
`joint_ctc_frames_v17_20260915/`. No new data, unverified blanks or held-out tuning.
Next: audit frame coverage/review; complete current run, then execute fixed refinement.

## 2026-09-15 — first frame-normalized epoch confirms blank suppression mechanism

After one unchanged-data continuation epoch, known TRAIN matches rise330→879/1,291,
deletions fall817→65, and blank steps fall235,498→50,515/236,988. This directly
supports the loss-scale diagnosis on real training data, beyond the toy counterexample.
However insertions rise65→2,608 and TRAIN WER is233.93%; emission recovery alone is
not accurate recognition. Isolated CTC remains1,830/1,901. The fixed epoch13–24 run
continues; no held-out outputs or decoder tuning informed this result. Next: judge
final precision/deletions jointly, then evaluate the fixed candidate and verify.

## 2026-09-15 — per-frame CTC correction reviewed and launched

Reviewer cleared nonrecursive loss replacement, complete/positive normalization,
epoch12 optimizer continuation and provenance/evaluation routing. All31focused tests
pass. Launched fixed epochs13–24 from the completed anchor checkpoint, changing only
CTC NLL normalization from targets to valid output frames. All prior objectives,
learning rates and full coverage remain. Next: verify full-training recognition,
then evaluate the fixed final candidate and report measured limits.

## 2026-09-15 — anchor run completed; loss-scale counterexample identifies remaining blank basin

All12anchor epochs completed. Final TRAIN:330/1,291known matches,817deletions,
65insertions,79.47%WER,1,858/1,901isolated CTC correct (97.74%). No development
results were inspected. A64-training-sequence epoch10 probe finds blank at87/106
known anchors; removing blank would identify62/106. Both competition and label
confusion remain, so blank suppression would not solve recognition.

A controlled32frame single-target optimization reproduces a concrete scale failure:
target-normalized CTC plus positive CE settles at p(target)=0.062706 with blank greedy
output; frame-normalized CTC plus the same CE reaches0.995391/nonblank. New shared
`joint_ctc_loss_balance_v17.py` reuses existing CTC validation/autograd and normalizes
by valid input length. Two red/green tests verify the opposing gradients and padding/
length scaling;17focused CTC tests pass. A bounded continuation preserves anchor12
model/optimizer and changes only CTC normalization for epochs13–24. Plan/runner:
`joint_ctc_balanced_v17_20260915/`. Next: review, train, evaluate and verify full results.

## 2026-09-15 — verified anchors produce earlier nonblank fitting

Corrected anchor epoch4 recovers53/1,291known continuous training signs and correctly
decodes1,259/1,901isolated training clips, versus0/1,291 and9/1,901 at epoch4 of the
preceding positive-CTC-only repair. This is an intermediate training-only result;
it is not yet adequate full-data fitting or evidence of held-signer generalization.
The original fixed12-epoch run continues without recipe/decoder changes. Next:
finish the run and evaluate the final candidate before claiming a repair.

## 2026-09-15 — corrected anchor objective reviewed and launched

Reviewer cleared fixed population weighting and its epoch/source scaling; all29
focused tests pass. The corrected fresh seed17111 run is launched for12epochs with
full-training recognition measured each epoch. All1,291known training targets have
anchors. No held-out evaluation or decoder adjustment has occurred. Next: finish
training, evaluate the final candidate, then independently verify checkpoints,
full-data coverage, greedy CTC retention and reported development metrics.

## 2026-09-15 — review corrected epoch-level known/OTHER anchor weights

Review found minibatch means would overweight OTHER-only batches. Stopped the new
run before any checkpoint; preserved its recipe, freeze and log under
`joint_ctc_anchor_v17_20260915/interrupted_batch_weighting/`. The repaired objective
uses fixed full-training per-source known/OTHER anchor populations and sum reductions,
so batch partitioning cannot change total weighting. New partition-invariance test
passes alongside both anchor safety/gradient checks. Anchor audit covers 7,379/7,380
training events, including all 1,291 known targets; one ambiguous OTHER is skipped.
Manifest timing SHA is verified against the original preparation audit. Next: final
bounded review and run corrected frozen recipe; no held-out outputs inspected.

## 2026-09-15 — first positive repair completed; verified-anchor correction prepared

All 12 positive-CTC epochs completed with complete coverage. Final isolated TRAIN CTC
is 1,780/1,901 (93.63%), but continuous TRAIN known recall is 183/1,291 (14.18%),
974 deletions and 87.84% WER. No held-out evaluation was used to choose the correction.
The missing positive route is repaired; sequence alignment still fits poorly.
New `joint_ctc_anchors_v17.py` uses one unambiguous token within each verified event,
and event-balanced CE directly teaches the actual CTC head. Repeated targets remain
separate; overlapping/unsampled intervals receive no invented target. Two new red/green
regression tests pass. The next bounded recipe adds these anchors and positive midpoint
CE while retaining prior objectives, data, architecture, original initialization and
12-epoch schedule. Report/plan: `joint_ctc_anchor_v17_20260915/`. Next: inspect anchor
coverage and review, run the frozen training recipe, then evaluate and verify results.

## 2026-09-14 — train-only probability diagnosis separates CTC marginal and greedy collapse

First direct-positive repair is still training; epoch4 recovers0/1,291known continuous
training signs and9/1,901isolated train clips. A16-training-clip epoch3 probe measures
mean blank0.9660/target0.02745; correct transcript marginal beats empty8/16, but greedy
is empty16/16. A controlled stationary32-frame CTC calculation has a blank-dominated
local basin near target probability0.03 (loss0.95464); uniform increase to0.10 raises
loss1.98916. These are training-only diagnostics, not held-out selection or a decoder
change. Evidence in joint_ctc_repair_v17_20260914/DIAGNOSIS.md and corresponding JSONs.
Next action: complete fixed first repair and judge full training recognition, not loss.

## 2026-09-14 — positive-CTC repair reviewed and training launched

Independent review clears training loss routing, pooled-forward equivalence and
optimizer recovery. Added the requested pre-evaluation recipe/checkpoint identity
validation before freezing the run. Corrected joint training is now running for
12epochs from the original seed17111 initialization; data, architecture, learning
rates and full-coverage schedules remain unchanged. New direct replay/core CTC losses
are the sole objective correction. Full-training known-sign recovery and all1,901
isolated CTC training outputs are evaluated every epoch; held-out results remain
uninspected until this bounded training run completes. Next action: inspect training
fitting, then evaluate or continue training-only diagnosis if collapse persists.

## 2026-09-14 — failed-checkpoint probe confirms missing positive CTC gradient route

On the final failed joint checkpoint and16training replay clips, pooled CE has no
CTC-head gradient. Direct one-token CTC gives blank-bias gradient+0.054630 (gradient
descent lowers blank), while verified-background CE gives-0.017992 (raises blank).
Training transcripts contain1,291known and6,089OTHER targets across236,988tokens.
This establishes a missing direct positive route, not that it is the sole cause.

Added `active/v17/joint_ctc_supervision_v17.py`, two red/green regression tests and
report-local `joint_ctc_repair_v17_20260914/run_repair.py`. Twelve focused CTC tests
pass. Repair retains original losses and adds single-token CTC on replay and exact
cores, sharing encoding with pooled CE. Checkpoints now retain optimizer state and
recover an interrupted summary write; previous frozen files are untouched. Next:
independent bounded review, then fixed12epochs with full training recognition measured.

## 2026-09-14 — user authorized repairing CTC blank collapse

Code trace identifies a candidate missing route: pooled replay/core losses update
Stage1 only, whereas verified-gap CE directly teaches the CTC head blank. Measure its
gradient consequences before changing the objective. New bounded plan:
`artifacts/reports/joint_ctc_repair_v17_20260914/PLAN.md`; reuse frozen data, architecture,
seed/schedules/learning rates, add direct single-token CTC on isolated/core positives,
and require full-training recognition diagnostics. No held-out target changes,
decoder bias tuning or previous artifact mutation. Next action: gradient probe/tests.

## 2026-09-14 — both joint-CTC arms complete; blank collapse fails every epoch

Completed12frozen+12joint seed17111 epochs,1,320updates each,8,016 evaluations across
the same334 recordings. Every epoch yields no known development signs: connected100%
WER/284deletions, familiar100%WER/259deletions, contiguous24/24deletions. Zero verified
gap emissions is silence, not useful rejection. Final joint pooled isolated accuracy
95.50%Citizen/85.99%SemLex retains the baseline95.24%/85.28%. O5S5 raw-core train
accuracy rises74/199→186/199 (37.19%→93.47%); LG stays12/57 (21.05%).

Post-run final-checkpoint inference on every complete ASLLRP training sequence finds
frozen0/1,291 and joint2/1,291known signs recovered. Joint236,988training token decisions
are236,974blank,10OTHER,4known (two collapsed signs). Both final models choose blank
on every ASLLRP validation token. Separate single-clip CTC diagnostic scores frozen
4/378Citizen,18/978SemLex; joint0/378,0/978. Pooled retention does not imply CTC retention.

Verified all4,520input hashes,24checkpoint hashes, unchanged frozen/changed joint
encoders, identical starting head/encoder and per-epoch schedules/coverage. Each epoch
covers923complete sequences,199cores,1,901replay,494background. Recomputed recording edit
counts/gates;24focused tests pass. Final CPU/MPS628/628tokens agree per arm; max logit
error4.77e-6. Successful training724.62s excludes preparation/evaluation and the
preserved8-epoch Metal-aborted attempt. Complete retry used unchanged original recipe.

No confirmation, runtime/device gate, export or promotion ran. All296temporally scored
signs missed; cached CPU timings exclude extraction/scheduling and are not live latency.
Decision: next model work must diagnose training CTC blank collapse before another
generalization comparison; more data alone is not a measured fix. Signer/variant/phone
coverage remains necessary separately. Do not scale the same failed recipe or tune
blank suppression from LG. Result/report scripts and verification are in
`artifacts/reports/joint_ctc_v17_20260914/`; PROJECT_GROUND_TRUTH.md updated. Next safe
action: return the completed negative result. Final handoff checks passed:24tests,
compilation, tracked/new-file whitespace, report assertions and large-artifact index.
Binding evidence promoted to live-streaming/high.md; no required experiment work remains.

## 2026-09-14 — Metal runtime aborted frozen arm after epoch8; recipe unchanged

Authoritative process inspection shows no surviving training process. The frozen log
ends with Apple M4 Metal command-buffer Internal Error(00000001), after eight complete
epoch checkpoints/evaluations; joint has not launched. All eight froze isolated accuracy
and emitted no known connected signs (100%WER,284deletions), failing gates. This is not
a completed experiment. Preserve partial files as frozen_interrupted and restart the
same arm from original initialization because checkpoints lack optimizer state. No
hyperparameter or data change is permitted from the observed development results.
Next action: MPS smoke check, caffeinate-protected same-recipe retry, then joint arm.

## 2026-09-14 — final preflight passed; matched CTC training launched

Final preparation retains all923train/237validation complete ASLLRP sequences,
199train/57LG cores (four static one-observation cores), 1,901/1,356 isolated replay,
494/16 verified backgrounds and all334 development recordings. All4,520 inputs match
their frozen hashes. Final9,239 chunks have max duration0.5005s, max sequence528tokens.
Train-only preflight again reaches4/4 exact at70updates/loss0.04293; encoder gradient
L1>0, frozen gradients absent, initialized CPU/MPS token predictions agree256/256.
Independent reviewer reports no remaining blocking issue;24focused tests and diff
check pass. Recipe and dependency hashes are sealed in training_freeze.json.

Frozen then joint12-epoch arms launched sequentially; no inference-mode encoder cache
used for sequence training. Current experiments remain research-only and no checkpoint
is eligible until all gates pass. Raw-core original-base probe gives74/199train and
12/57LG versus contextual-core52/199 and16/57 on one result per original event.
These differ from previous overlapping-window metrics and are not interchangeable.
Next action: collect all24 epoch evaluations and compare matched coverage, retention,
deletions and event-level LG; do not alter recipe after observing development results.

## 2026-09-14 — reviewed chunk-duration correction before full training

Independent review found wall-grid chunk boundaries could exceed0.53s when observation
times undershoot successive boundaries. Reproduced with a failing30-frame10Hz test,
then anchored each next endpoint to the preceding observed endpoint+0.53s. Ten focused
CTC tests now pass, including full-sequence/available-prefix equality and timing bound.
Prepared caches before this correction are retained as data_initial195.pt and
data_initial_grid.pt; final cache is being regenerated and will report maximum duration.
The corrected199-core preflight had already repeated4/4 tiny-fit success at step70;
it will rerun after timing correction. CPU/MPS initialized-token predictions agree256/256,
max logit difference2.86e-6. All16 validation background feature arrays exactly match
the frozen baseline gate. No full arm has started; next action final freeze/preflight.

## 2026-09-14 — joint CTC preflight fits train-only sequences; one-frame cores retained

Added `active/v17/joint_ctc_v17.py`, four focused contract tests, and report-local
`joint_ctc_v17_20260914/run_experiment.py`. Eight new/existing CTC tests pass after
observed red tests for the new module and single-observation core helper. First cache
contains 923/237 train/validation complete sequences, 1,901/1,356 replay clips,
494/16 verified backgrounds and all 334 development recordings. Sequence encoding
has 8,875 chunks, max 528 owned tokens; no sequence rejection occurred.

Train-only preflight: four distinct transcripts fit exactly by update70, CTC loss
0.04293; encoder gradient L1 162621.20, frozen encoder gradients absent. Initial
preparation excluded four one-observation O5S5 cores; corrected to repeat their observed
static pose, with no invented trajectory, so all199 events can participate. Zero-frame
cores still reject. Initial195 cache/reports retained under renamed paths; corrected
preparation and preflight will rerun before full training. Both arms explicitly use
eval-mode encoders so gradient enablement, not encoder dropout, differs. Independent
review underway. Next action: validate corrected preparation, then matched12-epoch arms.

## 2026-09-14 — frozen-versus-joint experiment authorized and recipe fixed

User explicitly requested execution through measured results. Plan is frozen before
training at `artifacts/reports/joint_ctc_v17_20260914/PLAN.md`: existing causal CTC
blocks, differentiable frame encoding, timestamped <=0.53s chunks, complete ASLLRP
sequence CTC, positive-only O5S5 cores, isolated retention and verified-gap blank loss.
Compare frozen and joint arms for 12 seed-17111 epochs with matched coverage/updates,
after gradient/alignment/tiny-fit preflight. Reuse existing development gates and
frozen replay identities; no official Citizen test, default changes or acquisition.
Free disk at start: approximately 12 GiB. Next action: focused tests and implementation.

## 2026-09-14 — independent next-step review qualifies the CTC recommendation

Fresh 14-test run passes. Streamed saved CSV reductions reproduce 8,978 windows,
positive-window exposure deficits, 29.0891% long-window O5S5 foreground, 199/57
O5S5 train/LG events, and 23/49 one-signer classes; combined manifest hash matches.
Stored core metrics were checked, not rerun inference. Initial verification assumptions
about globally unique IDs and one model_summaries row per model failed: the latter is
a grouped metric list; isolated has 15 model entries. Background IDs collide in 52
groups/57 extra rows because per-gap ordinal resets; features differ, targets agree,
and positive context IDs are unique. Include gap/timestamp in future coverage identities.

Code inspection qualifies the prior recommendation: core pooling follows full-window
noncausal encoding, so it is not a raw-core oracle. The encoder has unmasked attention,
symmetric convolution and a 32-position table. Causal head alone does not establish
causal streaming. Existing UnifiedStreamingCTCHeadV17 already provides shallow dilated
blocks; its trainer caches inference-mode features. Existing window training also
already has foreground CE and replay KL. CTC needs 102 outputs, including OTHER.

Decision: next model experiment should first pass gradient/alignment/timing and tiny-fit
preflights, then compare reuse of that head with frozen versus joint encoder training
under matched corrected supervision/coverage. Defer multi-level/shared-head additions.
Fresh portrait-iPhone signer groups and targeted per-class coverage remain necessary.
Primary-source research supports shallow temporal/visual auxiliary learning, but the
2025 pose paper tests unseen sentences, not this project's ASL signer/iPhone setting.
CVF direct fetches failed 403; official indexed paper text supplied reviewed details.

Changed report: `artifacts/reports/stage2_data_learnability_audit_v17/NEXT_STEP_REVIEW.md`;
current-state frontier/actions clarified in PROJECT_GROUND_TRUTH.md. No pipeline code,
checkpoint, runtime, dataset or external message changed; Citizen test stayed sealed.
Next safe action is the bounded preflight/comparison described in the report, not a
new architecture sweep. Validation: large-artifact index regenerated (23,525 bytes),
review/current-state/history reference assertions pass, and git diff --check passes.

## 2026-09-14 — exhaustive supervision audit separates window dilution from LG generalization

Audited every 8,978 continuous window and all 3,257 isolated replay clips against the
original model, all 12 O5S5 checkpoints and prior no-O5S5 comparators. Data extraction
is globally healthy (98.68% O5S5-train and 100% LG any-hand frame coverage), exact
feature conflicts are absent, and O5S5 is not uniquely fast: median target duration is
0.259s versus 0.267s for ASLLRP OTHER; LG is slower at 0.340s but remains weak.

The training contract is the measured bottleneck. Across 12 epochs the source-balanced
sampler never selected 3,516/5,515 ASLLRP OTHER windows or 154/1,005 O5S5 windows.
O5S5 targets occupy only 29.1% of a 1.07s window on average. Exact-core pooling raises
epoch-12 O5S5-train accuracy from 54.03% known-window to 61.09%, but LG remains
27.67%, so segmentation is only part of the failure. O5S5 per-class signer support is
thin: 23/49 train classes have one signer, and three LG classes are absent from O5S5
train. The underlying isolated classifier remains 87.68% on its full validation replay
at epoch 12, confirming that continuous recognition is the main failure while Citizen
retention still regresses.

Decision: do not repeat pooled 1.07s target classification or call unverified O5S5
surrounding signing blank. Next candidate is a shared frame encoder with isolated
auxiliary CE and a shallow temporal CTC head, deterministic full-pool coverage and
exact-core O5S5 supervision. No model, runtime or default changed; Citizen test was not
accessed. Reports: `artifacts/reports/stage2_data_learnability_audit_v17/`.

## 2026-09-14 — O5S5 final verification passed; negative result sealed

`o5s5_augmented_v17_20260914/verification.json` records 14 passing focused tests,
clean `git diff --check`, identical recomputed selection, unchanged original manifest,
4,520 verified frozen inputs and 12 checkpoint hashes. All 4,008 recording evaluations
(86,544 scheduled predictions) and held-out LG checks completed. The first-run gates
failed; confirmation was not launched. Report and CSV comparisons are complete.
Next safe action remains retention/deletion analysis under a separately bounded scope;
no further run is queued by this experiment.

## 2026-09-14 — O5S5 augmentation completed; no eligible candidate

Frozen manifest SHA256 `6b506552f7d35014e539e5df9f5e9d8d40a8e227544974ddb407f8d52cae4178`.
Fresh original-base seed17111, paired with the prior data-only comparison, completed
all12epochs/564updates in108.43s. Added1,005 O5S5 context windows while retaining
1,901 isolated training replay clips and494 verified ASLLRP background windows.
All12checkpoints evaluated on334 development recordings/7,212windows each, plus318
LG positive windows. No Citizen test access, acquisition or MediaPipe conversion.

Best connected checkpoint epoch3:123.5915% WER,113S/42D/196I over284 signs, versus
frozen CTC168.3099%,144S/11D/323I. WER improves26.57% relative, but deletions fail.
Familiar WER111.9691% versus56.3707%; Citizen93.3862% versus95.2381% own start;
SemLex84.2536% versus85.2761%. Both isolated drops exceed1point. Verified gap
emissions improve15→6/16. Prior no-O5S5 best was146.1268% WER/33D/258I.
Only epoch1 preserves both isolated accuracies, but fails connected311.6197% WER,
768 insertions, familiar128.5714%, and16/16gap emissions. All epochs fail measured
gates. Runtime-inclusive latency remains unverified; raw/phone replay was not run
after accuracy rejection. No confirmation17112, export, promotion or default changes.

LG positive-window accuracy: original base76/318=23.90%, prior epoch1 73/318=22.96%,
new epoch1 82/318=25.79%, new best-connected epoch3 67/318=21.07%. No full-narrative
LG WER because annotations are incomplete; windows overlap and are not independent.
CPU/MPS labels agree636/636 across LG epochs1/3. Fourteen focused tests passed.
All4,520 frozen raw/replay hashes,12checkpoint hashes and exact isolated replay
identities reverified; encoder/classifier updates confirmed. Initial sandbox MPS
launch failed before optimizer updates; authorized MPS run succeeded.

Artifacts: `artifacts/reports/o5s5_augmented_v17_20260914/` contains README, frozen
manifest, development freeze, all-epoch evaluations, LG predictions, selection,
CSV comparisons/error examples, parity and provenance. Checkpoints:
`artifacts/models/stage1_window_o5s5_v17_seed17111/`. Changed only report-local
evaluation orchestration, reports/index and current/history documentation.
Next safe action: retain rejected artifacts and accepted defaults. Do not run
confirmation merely on aggregate WER improvement; the retention/deletion gate failed.

## 2026-09-14 — all12 O5S5 epochs completed; streaming evaluation underway

Fresh original-base seed17111 completed all12epochs in108.43s, saving every checkpoint
under `artifacts/models/stage1_window_o5s5_v17_seed17111/`. Training audit confirms
6,819 contextual positives and494 verified ASLLRP background windows, with no O5S5
background. Epoch1 reaches Citizen94.9735%/SemLex85.4806%; epochs2–12 fail at least
Citizen's94.2381% retention floor. First LG comparison: original base76/318 (23.90%),
prior no-O5S5 epoch1 73/318 (22.96%), new epoch1 82/318 (25.79%). These are correlated
positive-window diagnostics, not full-narrative WER. Full334-recording evaluation is
running for all12epochs. Next: freeze complete error/gate results; do not confirm
merely on LG window gains. Citizen test remains untouched.

## 2026-09-14 — O5S5 training launched on MPS after sandbox failure

Initial sandbox attempt failed at `model.to(mps)` before optimizer updates or checkpoint
creation. Original failure log retained. Authorized escalated command started normally;
first five epochs show isolated-retention regression after epoch1. Added report-local
`evaluate_experiment.py` reusing the frozen full-stream evaluator and MPS adapter, with
assertions for all334 recordings/284 connected tokens and318 LG positives. No production
code changed. A compile check hit the system Python cache sandbox restriction and was
rerun with the existing approved compile permission. Next: finish all12epochs and
evaluate every checkpoint; no gate/threshold changes.

## 2026-09-14 — bounded O5S5 augmentation frozen before training

User authorized one fresh development run and conditional confirmation, without further
acquisition or Citizen test access. Frozen byte-identical combined supervision under
`artifacts/reports/o5s5_augmented_v17_20260914/`; original manifest unchanged.
Pinned all 4,514 prior raw/replay inputs plus six O5S5 archives. Prior baseline artifacts
are unchanged; trainer differs only by previously implemented positive-only loading.
Use fresh original-base initialization and paired seed17111, unchanged 12-epoch
50/30/20 recipe. O5S5 adds 1,005 train positives; LG's 318 positives are validation-only.
Incomplete LG annotation prohibits full-narrative WER; report positive-window accuracy.
Fourteen focused training/selection/O5S5 tests passed. Next: train all12 epochs, evaluate
334 existing recordings plus LG, then apply unchanged gates before any confirmation.

## 2026-09-12 — Stage-1 window final handoff verification passed

Final verification.json records 61 passing focused tests, successful compilation and
clean git diff --check; all frozen inputs/artifacts and 12 checkpoint hashes reverified.
All 12 complete evaluations and 16 actual replays are finished; no task processes remain.
The final comparison report includes all epochs, exact failure gates, timestamped
examples, provenance, timing/misses, missing coverage and a runnable experimental Mac
command. Defaults and accepted artifacts are unchanged. Outcome: completed bounded
experiment, no eligible model. No additional training or export follows this handoff.

## 2026-09-12 — bounded Stage-1 window experiment completed without promotion

All12 seed17111 checkpoints have complete334-recording evaluations. Lowest connected
WER is146.1268% at epoch4 (258 insertions,33 deletions), versus168.3099% CTC (323/11).
Epoch4 fails deletions, familiar WER117.3745%, Citizen93.3862%, SemLex83.6401%.
Epoch1 alone passes isolated retention but fails connected WER/insertions, familiar
WER and transition emissions. Selection.json records no eligible candidate; full-pool
latency remains unverified, independently of these measured failures. No seed17112,
Core ML export, protected-test access or default promotion occurred.

Report README.md, all-epoch evaluation JSON, timestamped_candidate_examples.csv,
checkpoint_provenance.json and runtime summaries are under
artifacts/reports/stage1_window_v17. Runtime measured15/15 raw/cache transcript agreement
and7/15 recognized signs on the verified timing subset; missing coverage and zero
observed spontaneous online corrections are explicit. All16 replay processes completed,
including partial tails and the112-second HUNGRY source. Its Finish pass took46.25s
under contended CPU execution; this is not mobile/realtime readiness.

Final independent gate review found that runtime evidence was not bound to the frozen
annotation pool. Added a regression that reproduced false eligibility, then fixed
selector checks to require matching manifest SHA and exact annotated identities.
Re-review closed the issue. New selector/test/report-side analyzer changes do not alter
any training-frozen code or checkpoint. All12 checkpoint hashes bind the unchanged
freeze, and encoder/classifier changes were verified. The first review agent could not
run because its model was unavailable; a supported routine reviewer completed the check.
Next safe action: inspect the rejected research artifacts or obtain fresh independent
annotations under a new explicit scope; do not silently restart training or promote.

## 2026-09-12 — paced runtime verified; CPU/MPS parity enables remaining evaluation

Epoch1 actual runtime completed 16 recordings (12 exact ASLLRP, three connected,
diagnostic HUNGRY). All 15 comparable final transcripts exactly match cached results.
Verified alignment is available for nine clips/15 signs: seven recognized, eight
missed (53.33%); median first-correct delay 0.304s, p95 0.681s including runtime and
Finish. This small subset does not satisfy full-pool latency verification. Per-window
median/p95 is 58.14/106.47ms during concurrent CPU evaluation. HUNGRY took153.92s
capture processing plus46.25s Finish;453/453 no-hand-detected windows emitted known
labels. No whole-session WER is claimed. Report artifacts include runtime_summary,
hungry_candidate_diagnostic and timestamped_candidate_examples.csv.

CPU evaluation epochs1–4 completed. To avoid repeated slow CPU evaluation, the remaining
epochs use the same frozen evaluator with only model/input device transfer to MPS.
Epoch1 full parity verified all7,212 window labels,334 final transcripts and16 transition
labels before this change in execution device. CPU/MPS probabilities need not be bitwise
equal. No checkpoint, feature, decoder or training change was made. Scripts in the report
folder reproduce this evaluation and timing summary. Next: finish all12, apply selector,
record final gates and verify code/provenance.

## 2026-09-12 — first complete Stage-1 window evaluation fails streaming gates

Epoch 1: connected 316.20% WER (125 substitutions, 3 deletions, 770 insertions),
familiar 129.34%, exact subset 54.17%; transition false emissions 16/16. CTC connected
baseline is 168.31% WER (144/11/323), older-Reel familiar 56.37%, transition 15/16.
Epoch 1 alone retains both isolated accuracies within 1 point; epochs 2–12 already
fail isolated retention. No epoch can therefore qualify, even before full latency
measurement. All 12 complete-stream evaluations remain required for the report.

The serial evaluator was stopped during epoch 2 after preserving completed epoch 1;
remaining epochs use three independent CPU processes, unchanged code/data/checkpoints.
This is evaluation scheduling only, not retraining. Paced runtime epoch 1 covers all
12 exact-development recordings, three connected recordings selected by sorted identity,
and diagnostic-only HUNGRY. Startup-inclusive delay and CPU contention will be disclosed.
A final test invocation initially named a nonexistent test module; corrected invocation
passed all 60 tests. Added selector check covers missing epochs/runtime evidence and ties.
Next: finish evaluations/replay, freeze failure report and retain defaults.

## 2026-09-12 — Stage-1 window baselines frozen and bounded seed completed

Older-Reel paced familiar replay completed all 97 recordings: 56.370656% WER
(127 deletions, 18 substitutions, 1 insertion / 259 reference words). The matched
CTC transition check emitted on 15/16 checked background windows. Development freeze
pins 334 stream identities, 4,514 raw/replay inputs, model hashes, code and decoding.
Seed 17111 completed all 12 authorized epochs in 111.22 seconds; all checkpoints are
saved separately under artifacts/models/stage1_window_v17_seed17111. No protected
test was accessed. Sixty focused tests passed before launch. Complete-stream evaluation
of every epoch is running; no candidate is selected and no confirmation seed or export
is authorized yet. Next: apply every frozen gate, measure actual runtime replay,
and report failure if none qualifies. The immutable freeze records prelaunch state.

## 2026-09-11 — training gates and complete-stream evaluator added

Implemented scripts/evaluate_stage1_window_v17.py for all frozen raw development
windows, Finish tails, source-time latency with explicit misses, and fail-closed
promotion gates. Cached/batched latency is explicitly ineligible as runtime latency.
Added preparation of a common16-window guarded-background comparator; this evaluates
both classifiers on identical support rather than comparing incompatible output rates.
Training now refuses unfinished/changed baseline freezes, uses exact1,500replay/
900context/600background samples per epoch, and limits frozen-teacher consistency to
known replay. All12epochs remain separately saved and unselected.

Corrected Finish decoding across timestamp gaps so identical predictions on separate
capture runs do not collapse together. Recent prediction corrections are restricted
to the last2s, and frame-mapping/exclusion validation rejects malformed input.
60focused tests and git diff --check passed before the last metadata/comment-only
edits. Existing defaults unchanged. Matched older-Reel familiar replay continues with
all completed processes successful; no candidate training launched yet.
Next: finish97baseline replays, freeze raw validation/transition inputs and baseline
results, train seed17111 once, then evaluate every epoch and report actual failed gates.

## 2026-09-11 — raw preparation passed; experimental runtime smoke verified

Materialized1,160 checked all-sign videos through the original live observation
contract; raw feature archives preserve actual sampled timestamps and per-source
hash/observer provenance. Preparation:train923 clips,5,814 contextual windows at
.27/.53/1.07s (2,528 on exact.53/.13 schedule),494 guarded background windows;
validation237 clips,1,331context/16background. Source signers remain disjoint3/1.
Coverage report finds173short known events,14rapid known pairs,1,560OOVevents;
zero identical-sign repetitions and zero known signs longer than1.07s in validation.
No independent portrait-phone/hold/repetition claims are supported.

Shared window conversion now normalizes only within the requested time slice,
resamples continuous channels by timestamps, uses binary nearest-neighbor presence,
and zeroes missing nodes. Runtime receives raw frames, uses only landmark classifier,
retains compact arrays for Finish, and handles partial tails. An untrained emission
head smoke produced2provisional windows and3Finish windows including the final partial;
this proves integration only, not accuracy. Unit checks cover explicit CLI/defaults,
timestamp invalidity/gaps, repeated outputs, correction of a recent prediction tail,
missing-node zeroing, and no future normalization leakage. Training sampler now uses
exact1,500/900/600 per3,000 draws with source/class balancing.

Frozen development identities334 (225connected,97familiar,12exact); older Reel paced
familiar baseline is running. Existing revisable CTC cached baseline and raw parity
reports are pinned for reuse. No candidate trained yet; defaults/checkpoints unchanged.
Next: complete matched baseline freeze, run seed17111 once for12epochs, evaluate all
complete streams and retention gates, then conditionally confirm seed17112.

## 2026-09-11 — matched HUNGRY diagnostic and checked-window audit

Four fixed period-origin diagnostic comparisons completed; evidence/hashes/settings:
artifacts/reports/stage1_window_v17/hungry/{diagnostic_freeze,diagnostics,summary}.json.
Shared frames:1,392 decoded,285 retained at source90–111s,640x360; damaged final
frame excluded. Timestamp sidecar actually has1,393 entries (earlier recovery prose
said1,394); no verified transcript or sign-boundary truth, therefore no WER/latency
accuracy claims. Fixed origins0,.25,.5,.75 of32/30s are engineering diagnostics, not
an exact reconstruction of session resets or original1280px pixels.
At phase.5,105.08–106.05s HUNGRY ranks1 with proposal.763/verifier.788; at phase.75,
105.35–106.32s ranks1 with.809/.847 yet CTC emits EAT in its recent hypothesis.
Window placement substantially changes both paths; compared CTC evidence before/after
OTHER suppression does not explain these HUNGRY failures. No reproducible extraction
or suppression repair yet meets gates; proceed with bounded candidate preparation.

Full all-sign annotation audit:train923 items/3signers/53known signs/41known pairs,
validation237/1/36/8; estimated guarded0.53s background windows494train/16validation.
These are annotation-time estimates pending raw-observation materialization, not
training-ready samples. Existing resampled Stage2 caches cannot establish the new
raw-clock window contract. Added shared raw Vision conversion and per-window
normalization so future frames cannot influence window scale. Existing defaults remain.
Next: complete raw preparation/runtime integration and freeze matched baseline results
before seed17111. No training or protected tests accessed.

## 2026-09-11 — Stage-1 contextual-window implementation started

User approved the bounded plan saved at artifacts/reports/stage1_window_v17/PLAN.md.
Existing dirty reorganization and runtime defaults are preserved. Work is split into
verified-data/trainer, opt-in timestamped runtime, and matched diagnosis/evaluation.
Recovered HUNGRY preview has 1,393 frames for 1,394 timestamps; exclude its damaged
final frame and disclose reduced resolution. Original timestamp mapping includes an
initial ~12.5s gap; constant-rate preview playback cannot reproduce source timing.
No causal regression finding yet. Required supervision will fail closed rather than
convert equal-duration local phrase partitions or unreviewed gaps into ground truth.
Next: freeze comparator hashes/identities/settings, diagnose shifted window origins,
and audit checked foreground/background before authorizing seed17111 training.

## 2026-09-11 08:13 PHT — architecture research for Stage-1-first continuous recognition

User requested research and discussion before further fixes: whether Squeezeformer
makes CTC/Stage 2 unnecessary and whether Stage-1 fine-tuning is preferable. No model,
runtime, data, or test changes were made. Reviewed current model/trainer code and prior
unified-streaming v2 report rather than repeating the experiment.

Verified distinctions: Stage-1 forward pools temporal features into one clip label;
encode retains frame tokens. Current Stage-2 retains unpooled frozen Stage-1-derived
features, then applies its own temporal Transformer and CTC output. Latest live
adaptation explicitly records stage1_frozen=True. Earlier unified streaming trainers
also cache Stage-1 evidence under inference_mode before optimizing a separate causal
head, so those failures do not establish failure of joint encoder sequence training.
Separate prior Stage-1 core adaptation improved local core accuracy63.32->89.58% and
ASLLRP cores66.67->83.33% (24 samples), Citizen95.24->94.18% on that experiment's
validation baseline. Short-window downstream CTC improved ASLLRP exact0/12->4/12 and
WER62.50->41.67%, but failed promotion. These are historical development results,
not the current multimodal baseline or the protected official Citizen test.

Primary research:
- Squeezeformer uses CTC in its speech implementation; encoder architecture and
  sequence alignment objective are complementary:
  https://arxiv.org/abs/2206.00888 and https://github.com/kssteven418/Squeezeformer
- CTC maps unsegmented input to variable-length labels with a no-emission symbol;
  it does not require this repository's separate Stage-2 Transformer:
  https://www.cs.toronto.edu/~graves/icml_2006.pdf
- EMNLP2024 online CSLR trains isolated recognition on continuous-derived contextual
  crops, includes background, foreground saliency supervision, and online duplicate
  removal. Its dictionary uses a CTC segmentor; it is not evidence that isolated
  dictionary training alone solves streaming. Different benchmarks, not our ASL gate:
  https://aclanthology.org/2024.emnlp-main.619.pdf
- VAC supports supervising visual representations as well as alignment, not only the
  downstream decoder: https://arxiv.org/abs/2104.02330
- Streaming Conformer constrains context during training and caches activations;
  supports both CTC and RNNT, so replacing CTC is not a prerequisite for streaming:
  https://arxiv.org/abs/2312.17279

Recommendation for discussion, not an approved new training specification: prioritize
existing Stage-1 recognition adapted to real contextual windows and independently
checked foreground/background evidence; preserve isolated replay and revisable UX.
Use matched HUNGRY Stage-1/Stage-2 evidence to localize the immediate regression before
choosing an implementation. CTC remains a valid baseline; direct Stage-1 online
classification is a credible alternative but requires emission/repetition handling.
No evidence yet justifies a new backbone, ensemble, or RNNT migration. Do not repeat
prior short-window/head-only runs under a new name or treat incomplete signs/OOV
signing indiscriminately as background. Existing data coverage/alignment should be
measured before requesting more videos. Next action: discuss this recommendation.

## 2026-09-11 08:06 PHT — user HUNGRY regression inspected; recording recovered

User reports revisable mode worse than earlier Reel and identifies HUNGRY near the
end of session 20260911_075501_769734. Recovered 1,393 frames from unfinalized mp4v
mdat using a locally generated 640x360/15fps codec header; original recording remains
untouched, final frame damaged. Inspected timestamped ending contact sheets. Visible
repeated chest movements cross sequence-window boundaries near 102.6 and 105.8s;
context hypotheses add GOOD, then MY/COME, with no HUNGRY in 41 accepted updates.
This does not establish lexical-variant correctness, per-sign accuracy, or causality.
Session uses repaired seed1702, not rejected adapters; Stage-1 predictions are empty.
Code confirms revisable mode bypasses older Reel's phrase-adapted Stage-1 classifier.
Median accepted total processing 378.51ms, hand image encoding 339.43ms, plus collection
of roughly one-second windows. Logs lack evidence to distinguish weak visual scores
from CTC/OTHER suppression. Diagnostic files and recovery limitations:
artifacts/generated/revisable_hungry_diagnosis/README.md. Next safe experiment is a
matched Stage-1/Stage-2 comparison and window-origin sensitivity check on these motions;
640px recording cannot reproduce original 1280px retained input exactly. No runtime
changes, training, or protected tests. User requested discussion before further changes.

## 2026-09-11 07:53 PHT — completion revalidated and condensed handoff corrected

User requested continuation before final report delivery. Verified the completed revisable
experiment against current files: all recorded source hashes and report SHA match; both
12-epoch runs have zero eligible epochs, 72 focused tests are recorded passing, and all45
raw replays completed with12/12 exact-subset raw/cache agreement per model. No experiment
rerun or runtime change. Corrected PROJECT_GROUND_TRUTH.md to distinguish the earlier
immutable-prefix experiment from implemented revisable transcription, preserve the user
choice, and record completed transition/core failures. Full evidence remains in
artifacts/reports/stage2_v17_revisable_v1/{README.md,verification.json}. Next safe action:
user review of the report; further alignment/recall work requires its own evidence-based
experiment rather than repeating failed losses.

## 2026-09-10 07:02 PHT — retention repair verified and enabled in continuous Reel

Completed the v2 retention-regression repair using existing data. Both final packages
pass all original frozen development gates plus STEM retention. Seed1701 target
485/284 (S/D/I139/13/333); selected seed1702 target470/284 (137/12/321), down22.1%
from accepted603. Both retain local/exact/contextual6/9/43, Citizen331/378 and
STEM16/21, with zero per-example prediction changes across those retention pools.
This preserves accepted behavior while reducing connected false insertions; it does
not retain v2's217/223 total edits or establish general continuous recognition.
The development-tuned run policy/margin and rejected experiments are disclosed.

Final self-contained package hashes supersede the earlier packaging hashes above:
- seed1701: 5656d21941462fc1b284d6e048a4e37ec0f4bb2ab1dea03e45d53a5ac8dda76b
- seed1702: d7bdc71d68f3093311fe85feef85e9b0b90c838f6d4a11a903afb596248d5685
Review identified unpinned selector configuration and frozen encoder provenance;
both were fixed with observed failing/passing regression checks. Runtime pins now
cover configuration, selector checkpoint, primary/specialist Core ML packages,
frozen encoder and upstream hand-image encoder. Both final packages reload.

Final validation:82 focused tests pass; compilation and git diff --check pass.
Actual Core ML selector plus CPU evidence matches checkpoint decoding on987/987
frozen development samples, zero mismatches. Mac cached-feature p95 is13.38ms
versus7.44ms baseline (added evidence p956.06ms); this excludes extraction and is
not sustained iPhone/full-Finish evidence. Two paired raw-video smoke tests preserve
old/new suggestions and selected transcripts with zero stale frames. Existing live
suggestions KNOW HOW YOU and PLEASE HELP I I remain erroneous; cached-development
retention does not fix those pre-existing live errors.

scripts/live_reel_continuous_v17.py now defaults to selected seed1702. Roll back
with --no-stage2-other-preservation. Legacy entrypoint defaults remain unchanged;
Stage2 remains review-only and does not overwrite the trusted transcript. No camera
session was automatically launched, no new data acquired, and no protected Citizen,
SemLex or local test was accessed. No git commit was made.

Final evidence: artifacts/reports/stage2_v17_transition_repair_v3/README.md,
validation.json, verification.json, tests.log and replay_comparison.json. Builder:
scripts/build_stage2_other_preservation_v17.py; evaluator:
scripts/evaluate_stage2_other_preservation_v17.py --runtime. The completed scope is
the measured v2 retention regression. Remaining live recognition errors and mobile
readiness are separately unresolved; do not claim them fixed or rerun protected
tests for development. User can run the continuous Reel command with this repair.

## 2026-09-10 08:48 PHT — live-lock diagnosis in progress: session and paired core evidence

User requested a completed diagnosis report of latest webcam locking difficulty,
temporal learning and live extraction. Examined session20260910_071844_645159:
66 Stage1 probes,40 accepted,14 stable proposals,10 commits;4 keyboard Finishes.
Stage2 runs at Finish and is review-only; it cannot rescue the live locks. First
Finish temporarily decodes HELLO HOW YOU, then repeats YOU; last temporarily gives
I READ TOMORROW before repetition/prefix changes. Intended user transcripts remain
unverified; do not assign a webcam WER. Read-only independent code/history analysis
confirms growing multi-sign windows and verifier disagreements, not simply high
commit thresholds. Raw history and lowres video inspected; fixed15fps recording
omits original timestamp/gap timing, so it cannot exactly recreate capture timing.

New diagnostic artifacts only under artifacts/reports/stage2_v17_live_lock_diagnosis_v1.
All12 exact-variant dev phrases replayed through current runtime (no cherry-picking).
Matched24 annotated cores via ASLLRP occurrence/frame provenance;20 have existing
valid archives,4 are2–3-frame cores excluded by the existing4-frame minimum.
Isolated frozen Stage1 gets16/20, repaired CTC18/20. On8 complete pairs/16 tokens:
full cached phrase5 edits, stitched independent CTC cores2 edits, oracle core-window
concatenation2 edits. Cropping/window resampling changes as well as boundaries;
this supports temporal/window sensitivity without proving every upstream cause.
A first probe stopped on missing short-core archives, then explicitly recorded those
exclusions rather than padding or silently shrinking the denominator.

Controlled landmark-resolution/sampling intervention now running, holding cached
hand evidence and window edges fixed. Runtime/checkpoints/data splits untouched;
no protected evaluation or new training. Final report pending this experiment and
reconciliation of evidence. No more model ensemble proposed.

## 2026-09-10 08:53 PHT — completed continuous-lock diagnosis report

Completed requested latest-webcam history, temporal and live-pipeline diagnosis.
Canonical report: artifacts/reports/stage2_v17_live_lock_diagnosis_v1/README.md;
verification.json hashes all quantitative evidence. No runtime/model/training-data
change. Latest webcam remains20260910_071844_645159:66 probes,40 accepted,14 stable
proposals,10 commits; hand detection coverage mean99.54%. Commit-only threshold
reduction to0.25 would add DAY for a COME proposal, not recover either SORRY proposal;
these are fixed-proposal counterfactuals, not full changed-policy accuracy tests.
Stage1 isolated growing crops drive locks; Stage2 runs at Finish and is review-only.
User intended transcripts unverified; no webcam WER or expert lexical claim made.

All12 exact-variant JONATHAN development phrases (24 tokens) replayed successfully:
cached Stage2 WER37.50%,4/12 exact; raw Stage2 WER45.83%,3/12 exact; committed
transcript WER79.17%,0/12 exact. Raw/cached match7/12. Six committed outputs empty.
No cherry-picking, no protected tests. Manual core probe evaluates20/24 cores:
frozen Stage1 correct16/20; CTC18/20. Four2–3-frame cores explicitly unavailable.
Eight complete pairs: cached WER31.25%, isolated-core CTC/concatenated-core-window
CTC both12.50%, with different failure identities. This supports temporal/context
sensitivity; manual boundaries also alter duration resampling/extraction context.

Controlled frontend test retains cached hand evidence and source window edges:
cached inputs through CoreML and fresh1280px source-rate landmarks reproduce all12
cached predictions (max encoder/cache difference0.001947). Resolution640px alone
changes9→11 edits. Reducing samples to20Hz makes one tail3frames/unusable. Shared
11-clip/22-token cohort: cached/fresh1280-source8 edits;640-source10;1280-20Hz9;
640-20Hz10. Failed first probe and unavailable tail explicitly retained. Actual
live hand crops, detector state, auxiliary phase and timing remain jointly measured,
not individually causally isolated. Do not present640px as the only live cause.

42 focused existing tests pass; diagnostic scripts contain cohort/provenance/result
assertions; report links/hashes and all12 replay exits/history outputs reconciled.
No additional ensemble or blanket threshold change recommended. Next priorities:
continuous provisional/commit design, matching training/live extraction contracts,
then temporal adaptation on existing connected sequences including short signs and
endings/repetitions. New targeted timestamped/transcript-confirmed capture only if
needed for remaining domain gaps. Recorder currently writes fixed15fps without
preserving source timestamps/gaps, so lowres webcam replay is not exact timing.
This completes diagnosis/report scope; live recognition is not claimed fixed.

## 2026-09-10 18:54 PHT — user confirms revisable sequential transcription; research discussion

User prefers continuous transcription and explicitly accepts revisions to recent displayed words. Earlier proposed hold-until-Stage1-lock experience is not the product target. User requests discussion/research first, not implementation. Transition/coarticulation false emissions are a priority; small residual mistakes may be corrected at phrase completion, but final recognition should retain visual evidence rather than rely on grammar alone.

Primary-source research: AWS documents revisable partial transcripts and optional last-few-word stabilization, with accuracy/latency tradeoff (https://docs.aws.amazon.com/transcribe/latest/dg/streaming-partial-results.html). Google distinguishes interim stability from finality (https://docs.cloud.google.com/speech-to-text/docs/reference/rest/v2/StreamingRecognitionResult); streaming transducers condition on accumulated signal and output history (https://www.research.google/blog/an-all-neural-on-device-speech-recognizer/). Two-pass deliberation uses both original signal and first-pass hypotheses (https://research.google/pubs/deliberation-model-based-two-pass-end-to-end-speech-recognition/). EMNLP2024 Towards Online Continuous Sign Language Recognition and Translation uses sliding-window isolated recognition trained with background/coarticulation, surrounding-clip augmentation and saliency supervision (https://aclanthology.org/2024.emnlp-main.619.pdf); evidence is on its own sign-language benchmarks, not validation on our ASL runtime.

Decision: separate transcription UX from recognizer architecture. Keep existing visual/temporal learning as a starting point; investigate transition-aware training and revisable live sequence output before choosing any transducer replacement. No training or runtime changes made in this discussion.

## 2026-09-10 19:32 PHT — revisable runtime verified; transition experiments running

Implemented --revisable-transcript in continuous Reel: no prefix locks, changeable HUD, per-revision events, disk-retained visual windows and Finish decode over overlapping8-window chunks with absolute-time logit stitching/one CTC collapse. Baseline raw replay completed all12 exact-variant clips,2 local clips and user saved recording. User recording retained22 windows;4 Finish chunks covered all of them;12 live revisions. No verified intended transcript/source timing, therefore no user WER. 39 focused tests passed after updating HUD mock signature; screenshot hud_preview.png rendered and inspected. Worker usage limit interrupted HUD follow-up, main completed remaining integration and checks.

Annotation supervision payload:1609 guarded gap bins (1505train/104development),2407 eligible nonempty prefix targets,1128 full alignment matches. Prefix OTHER collapse and feasibility checked; per-span frame rate used. Second independent supervision artifact adds1595 guarded known cores (1299train/296development) to counter silence; original running-training artifact unchanged. Paired12-epoch training jobs live, gap+prefix seed17101 and same plus.25core CE. No epoch eligible so far. Previous adaptation error decomposition is material: OTHER-domain insertions323->32 but deletions11->138. Final beam8/topk12 control reuses existing CTC decoder; it leaves exact errors unchanged, changes connected478->477 and worsens local19->20, so no default beam change justified. Finite beam input floors run-veto negative infinity to-10000; no language model rescoring.

## 2026-09-10 19:46 PHT — revisable transcription experiment complete; no promotion

Completed authorized runtime, transition-aware training experiments and final report artifacts/reports/stage2_v17_revisable_v1/README.md. Evidence verification.json, training_verification.json, raw_verification.json and tests.log. --revisable-transcript is opt-in, changeable live hypotheses, recent8windowcontext and whole retained utterance overlapping visual Finish re-decode. Existing defaults/checkpointunchanged. No transducer/ensemble replacement or grammar-only lexical repair.

Both12epoch runs completed: gap+prefix1738s selectedepoch12; plusknowncore1821s selectedepoch7. Zero eligible epochs across24. Matched core variant connected210/284 (73.94%WER), local17/259 (6.56%), exact12/24 (50%). Connected S/D/I41/153/16, versusbaseline144/11/323 and precedingadaptation56/138/32. Thus56.07% fewer totalconnectederrors thanbaseline still deletes53.87% knownreference signs. Corevariant isolatedCitizen316/378 (83.60%) versus331/378 (87.57%); originalphrase retention local16vs6,exact13vs9,context56vs43 allfail. STEM16/21 unchanged. Gap knownemissions baseline13/104,precedingadaptation3/104,newgap3/104,newcore4/104: no new demonstrated gap-suppression gain. No promotion.

45 successful rawpaced replays completed (baseline,gap,core each12exact+2local+userdiagnostic), excluding retained initialfailedadaptedflagattempt. Eachmodel exactraw/cacheagreement12/12. RawexactWER45.83%baseline versus50%bothadapted; nonemptyupdatesbeforeFinish1/12baseline and0/12adapted, so latency unresolved. All3userreplays retain22windows/fourFinishchunks and finalvisualoutput may reviseearlierprefix. No verifiedusertranscript/timing, no userWER. LongerlocalHELLO correctedbynewmodels; PLEASE HELP I stillduplicatesHELP. Wholemetricsgovern decision.

72 focusedtests pass includingprovenanceflagregression;4best/lastcheckpoints reloadfinite,supervisionhashesmatch. Fixedinputcontract anddefaultbaselineSHAunchanged. HUDscreenshotrendered/inspected. gitdiff--checkpasses. Protectedtestneverused,unrelatedworktreepreserved. Limitations explicitlydocumented: annotationbins approximate,sparsegapcoverage,noexplicitnewholdaugmentation,noiPhonemeasurements ortranslationaccuracytest. Goalexperimentcomplete; robustnaturalcontinuousASLnotclaimedfixed. Nextsafeaction:userreviewreport anddecide whethernewindependentlyverified boundary/hold datajustifyfurtheralignment/recallwork.

## 2026-09-09 20:02 PHT — existing videos reproduce learned v2 emission regressions

User reports no additional observed live failure, challenges requests for more data,
and wants discussion of what is actually needed. Decision: do not request new
recordings or launch another acquisition; current evidence supports diagnosis with
existing videos, labels, frozen features and checkpoints. Recommend locked100
connected recognition with unfamiliar combinations and tolerance of unsupported
signing as the next bounded target; user has not yet adopted a new product scope.

Inspected 20 timestamped sampled frames each from two existing local familiar-domain
validation videos, then ran bounded CPU inference with accepted selector, initialized
candidate and both with-STEM best checkpoints. All four model results reproduce
the saved predictions. Both accepted/initialized models correctly emit PLEASE HELP I
and HELLO HOW YOU. Both adapted seeds emit PLEASE HELP I I and HELLO HOW. The extra
I is a new spike at CTC step 36 after I at step 28, with blanks between; YOU at step
21 disappears in both adapted models. These are output steps, not video timestamps.
Sampled frames show a prolonged final chest-pointing posture in the first clip and
a final forward-pointing gesture in the second. This is not expert ASL semantic
validation or full-motion playback. The two initially correct examples establish
an adaptation effect separate from the aggregate initialization/selector mismatch.

Artifacts: artifacts/reports/stage2_v17_transition_diagnosis_v1/README.md,
model_probe.json and two contact-sheet JPGs. No code/dataset/checkpoint/runtime
changes, training, protected evaluation or additional test-suite run occurred.

Web research checked primary CTC (Graves 2006), CTC spike-guidance (Kurata/Audhkhasi
2019), sequence-level KD (Huang 2018), VAC CSLR (Min 2021), and official Citizen use
guidance. These support testing temporal alignment/preservation objectives; they do
not prove a specific loss change will fix this model. Links and limits are recorded
in the diagnostic README. Next safe action: inspect timewise known/blank/OTHER
scores and source/replay coverage on existing development inputs, distinguishing
model discrimination, emission calibration, held poses/window sensitivity and
teacher disagreement before proposing a bounded training change.

## 2026-09-09 20:40 PHT — posterior diagnostics support conditional-emission preservation

Cached accepted/plain-initial/with-STEM seed outputs on 987 unique development
examples and accepted/with-STEM train outputs on 3,994 examples (including 1,116
previously omitted contextual training cores; all embedded encoder hashes match).
Reports under stage2_v17_transition_diagnosis_v1 include posterior_diagnostics.json,
window_diagnostics.json, counterfactual_diagnostics.json, training_detection.json.

YOU remains the top known sign at its old emission step, but blank probability rises
from 0.385 accepted to 0.962/0.982 adapted. Its probability falls 0.614 to 0.037/0.018;
OTHER is negligible. Held final I gets an extra step-36 spike: accepted I probability
0.0067, adapted 0.522/0.758. Restoring the old blank odds recovers seed1701's local
6 edits; restoring the full old conditional distribution recovers local 6, contextual
43/44 and STEM 16/21 but still fails exact retention (12). These controlled output
interventions locate the local mechanism; they do not isolate every training cause.
Removing the final cached window removes the extra I but seed1702 then repeats HELP,
so blind trimming is not a safe fix. A fixed majority-OTHER router failed exact and
contextual gates (10;46/45), and was rejected, not integrated.

Selected repair experiment: keep accepted known/blank conditional logits frozen and
fit only the OTHER output on frozen v2 temporal representations using original CTC
labels plus previously omitted context negatives. Reuse all accepted-model behavior;
no identity, phrase, source or duration routing at inference. This is a larger research
composite, not model compression or an iPhone-ready claim. A behavioral regression
for preserve_known_ctc_logits was observed failing, then passed after adding the
shape-checked concatenation helper to model_stage2_v17.py. Training/validation caches
are under artifacts/models/stage2_v17_other_preservation_v1; bounded 20-epoch, two-
seed linear-head fitting now runs on CPU (no Stage-1 or temporal-backbone training).
No protected test accessed; existing runtime/artifacts unchanged.

## 2026-09-07 20:20 PHT — continuous Reel experiment delivered; metadata candidates acquired

Completed the authorized separate experiment. Run:
`venv/bin/python scripts/live_reel_continuous_v17.py`.
Amber GLOSS? is tentative; F/button finishes one utterance. Incoming frames newer
than a committed clip survive verification. Finish drains eligible new tail evidence
once, then computes Stage 2 over private disk-spooled exact observations. Its candidate
is displayed separately for review and cannot auto-replace or speak over the Stage-1
transcript. No accept-suggestion control is implemented. Speech is finished-sentence
only. Reset, quit without Finish, reset during CTC, and a second Finish during prior
naturalization have regression coverage. Original Reel defaults and frozen100 retained.

Final paired paced development run: both lanes 1/5 exact with 5/12 token edits;
there is no aggregate Stage-1 accuracy improvement. Continuous retained55observations,
first tentative output median0.904s (range0.539–6.311s; early guesses can be wrong),
Finish decode median1091ms (range794–1267ms). Sequence review candidates5/5exact on
the same familiar short recordings. Baseline itself varied from the initial2/5pass;
async scheduling changes candidate windows. A capture-start outlier and interruption
between pairs prevent claims of controlled hardware timing. Webcam FPS, human repeat
attempts, and unseen-signer accuracy were not measured. Default tiny-naturalizer smoke
also completed: Stage1 HELLO HOW -> "How is you?", candidate HELLO HOW YOU,
Finish total1321ms. This confirms integration while exposing remaining recognition
and language quality limits. Do not promote this prototype or its5/5candidate result.

Reports: `artifacts/reports/continuous_reel_v17_experiment_v1/README.md`,
`final/comparison.json`, runnable `run_replays.py`, and visually checked preview/finished
PNGs. Runtime changes: `scripts/live_reel_continuous_v17.py`, opt-in hooks in
`scripts/live_reel_stage1_v17.py`, provisional rendering in `scripts/reel_hud_v17.py`,
and `test/test_live_reel_continuous_v17.py`. All41focused Reel/CTC/HUD/cache tests pass;
syntax checks and git diff --check pass. Core ML/syntax validation required approved
unsandboxed execution due to GPU/model compilation and Python bytecode-cache paths.

Completed bounded public metadata audit after delegated audit failed with usage limit.
`artifacts/reports/continuous_reel_v17_web_audit_v1/` contains reports, hashes, official
NCSLGR index and MoLo listing/sample EAF. NCSLGR index564370bytes,1887uniqueutterances,
38collections; excluding existing166leaves1721. Conservative case-sensitive exact
Citizen raw-label matching found77textual contiguous-span candidates in76parents,
42distinctsequences/39rawclasses. These lack per-sign timing and participant IDs and
are NOT variant-approved training data. Current DAI requires login for XML downloads.
MoLo public API successfully lists32files including16EAFs. One inspected timed EAF
has299right/124leftannotations;34right/13left exact raw-label occurrences across13
classes, with two-hand duplicates not resolved. No video acquisition or training.
Sources: https://www.bu.edu/asllrp/ncslgr-for-download/download-info.html ,
https://dai.cs.rutgers.edu/dai/s/daioriginal , https://ida.gallaudet.edu/molo/ ,
https://api.osf.io/v2/nodes/9uevh/files/osfstorage/ .

Next safe work: user webcam comparison of the separate prototype and exact-variant,
two-hand/timing/signer audit of these candidate annotations before selective video
acquisition. No model retraining, threshold tuning, protected split access, baseline
replacement, or new continuous-accuracy claim. All work from this request is left
reviewable in the existing worktree; no commit or cleanup of unrelated changes.

## 2026-09-07 18:02 PHT — initial matched results and Finish review corrections

Five initial paired paced replays finished: both commands 2/5 exact with 4/12 token
edits, so buffer preservation alone did not improve aggregate Stage-1 accuracy.
Continuous lane retained 59 pending observations; first tentative display median
1.006s versus baseline first commit 1.726s (different semantics, some guesses wrong).
Review-only Stage-2 candidates were 5/5 exact; median Finish decode 805ms. These are
familiar short recordings, not independent/generalization evidence. THANKYOU FRIEND
regressed in Stage 1 while TOMORROW SCHOOL GO improved; do not hide individual changes.

Independent review identified two valid Finish edge cases: buffered new evidence
could be discarded on Finish, and a second utterance's Finish could be rejected during
prior naturalization. Added failing real-loop regressions and fixed both behind the
experiment behavior: Finish may probe an eligible newer tail once (never duplicate
stability hits on identical evidence); Stage 2 waits for that finite drain. A second
Finish is sealed while the previous language job completes; timestamps belong to each
utterance context. Also cleared stale uncommitted previews at candidate reset.
Review suggestion to disable baseline ten-finger controls was rejected: that control
was pre-existing user work, not introduced in this experiment. Original defaults stay.
Reviewer delivered findings before a usage-limit error; no further agent review run.

Final five-pair replay rerun is underway under
`artifacts/reports/continuous_reel_v17_experiment_v1/final/` because Finish behavior
changed. Reset-during-CTC test additionally checks stale result exclusion. No training,
threshold edits, model promotion, or protected split access.

## 2026-09-07 17:56 PHT — continuous Reel smoke passed; paced comparisons in progress

First real Core ML smoke completed after sandbox compilation/GPU access failed and
the user approved an escalated saved-video run. Unpaced HELLO HOW YOU replay returned
Stage 1 HELLO HOW and review candidate HELLO HOW YOU; candidate did not overwrite the
transcript. Added optional --realtime-video to pace saved sources for comparison;
new parser regression failed first and all 32 relevant tests then passed. Rendered
and visually inspected preview.png and finished.png in
`artifacts/reports/continuous_reel_v17_experiment_v1/`: amber HELLO? appears centrally
before commit, and final sequence suggestion is explicitly marked for review.

`run_replays.py` in that report folder is running five already-used familiar sources
against original/continuous commands, with identical thresholds, 20fps observations,
literal naturalizer, no display/speech/control-gesture filtering, and timestamp pacing.
It logs capture elapsed time, first tentative/committed output, verifier timing, retained
frames, final edits and Finish decode latency; no protected data is read. Early results
include an ~8s initial capture stall in one continuous HELLO replay despite ~10ms
proposal calls, so do not promise uniformly fast webcam response. Early tentative
glosses can be wrong (GOOD MORNING first previews MORNING); first-preview speed is not
first-correct-gloss latency. Matched results/report and independent async-state review
remain pending. No model weights or thresholds changed.

## 2026-09-06 21:16 PST — actual-video causal v2 improves to5/6 with context, SCHOOL still omitted

Six-video replay completed under session10885; no background process remains from
this turn. Result: continuous_rebuild_v17_v1/video_causal_v2/result.json. Raw exact
4/6,13.33%WER; existing guarded beam context5/6,6.67%WER. Prior same-video v1raw2/6,
46.67%WER and context3/6,33.33%WER. HELLO HOW YOU and MY NAME now exact raw;
context recovers FRIEND after THANKYOU. TOMORROW SCHOOL GO still omitsSCHOOL in
raw and corrected. These are six development templates, not unseen-combination or
live-camera accuracy. Per-frame p95 about104–344ms on this run, so do not claim
sustained30Hz real-time performance from these desktop/video measurements.
Initial aggregation import used wrongmodule; corrected to existing streaming edit
helper, no duplicate distance implementation. Saved targets, predictions, sourcepaths
and timing. No official test accessed; no checkpoint promoted by this smoke.

Added a guard rejecting degenerate palm-plane finger direction in the symbolic
handshape helper; focused18 tests pass and both v3rendered trajectories were verified
bit-identical after guard, reporthash updated. Source-match test passed separately
in19-test aggregate before guard. git diff --check passes. Human review of v3YOU/NEED
is pending. Full SignWriting motion generation, continuous joins, all100combinations,
real transfer from synthetictraining and robust continuous translation remain unmet.
Next useful actions: implement/test explicit symbol orientation/movement for this
bounded pilot; use pending fluent handshape feedback; evaluate motion-proposal fallback
against the omittedSCHOOL and OOV controls before optionalruntime integration.

## 2026-09-06 16:26 PST — pointing review identifies detector curl; lexical extension audit and causal model finish

User's v6 feedback: pointing YOU finger is not good. Root inspected raw source hand
closeups and estimated3D chains across source frames13–20: index bends fluctuate
35–75deg to120–150deg, confirming detector geometry is unstable. A tight Apple-box
MediaPipe crop probe also yields unstable curls, so crop-only replacement is rejected.
Existing pinned ASL-LEX annotations explicitly identify YOU as handshape1, selectedi,
FullyOpen; NEED isBent. Added opt-in lexical extension constraints for FullyOpen
selected fingers only, with a regression verifying bent entries/nonselected fingers
are untouched. Ten rig tests pass. Source comparisonv8 is rendering with lexiconhash
and explicit annotation provenance; these are generated constraints, not observations.

Causal local extraction completed287train/200validation. Matched-camera continuousv2
finished25epochs with duration1.5x/2x training replays. The selected metrics are in
artifacts/models/continuous_evidence_v17_causal_v2/result.json; finalepoch localWER
26.11%, not yet the selected metric or runtime score. Training a matching Stage3
context model now; original context hash cannot be silently attached to a newrecognizer.

Stage3's earlier cyclic-shift control was flawed: adjacent sorted rows often have the
same label. Corrected to different-label same-source near-duration controls; five
context tests pass. Proposal-only v1 exact on local183/200 vs0/200 with different-label
motion, Citizen357/378 vs0/378, SemLex758/978 vs0/978. This demonstrates motion use on
these development sets, not general compositional translation. Reranking still relies
heavily on language patterns (mismatched local106/200 vs matched112/200). Existing
reports are preserved; corrected controls are context_mismatch_control.json. No
runtime promotion or claim that all new recordings can be eliminated.

## 2026-09-06 16:15 PST — automatic utterance-end integration verified; causal train loader validated

Created a diagnostic fixture from validation PLEASE_HELP_ME/09d3e2c0 plus60 black
frames (not training data). Current live script finalized PLEASE HELP I at5.433s,
rendered Please help me once, then processed another.933s with an empty next utterance.
Thus automatic no-hand-gap finalization happens before EOF and does not lock input.
Artifact: continuous_rebuild_v17_v1/utterance_end_smoke;fixture is marked by its path.
This tests integration only, not natural pause detection reliability.

Actual causal training loader check passed:4509total training examples,861local
(287original plus1.5x/2x duration replays), unchanged target/signer identities and
expected frame counts, unique sample identities. Saved causal_loader_check.json.
Validation remains unaugmented by construction; its full load still awaits extraction.
The extraction job is live at320/487completed examples. All44 focused tests passed
before the subsequent isotropic-body fix; nine rig tests passed after that fix.
V7 HOW source-side preview was inspected and uses correct anatomical side and metric
body scaling. Pending fluent review still concerns source-comparisonv6, not blanket
approval of all later variants.

## 2026-09-06 10:38 PST — causal runtime equivalence checked; camera path and guarded context added

Eight development clips match full-sequence vs frame-by-frame final CTC tokens;
maximum raw-logit difference <9e-6. Cache FP16 rounding contributes <.003 logits.
Report: `continuous_rebuild_v17_v1/runtime_equivalence.json`. This isolates the
remaining recognition errors from incremental convolution/CTC implementation.

Added `continuous_vision_v17.py` (past-only 32-observation normalization; missing
hands remain absent; no activity trim) and `scripts/live_continuous_v17.py` (camera
or video, continuous partial/stable display, Enter finalizes an utterance, asynchronous
English rendering, optional completed-utterance speech). Two causal camera tests
passed after initial import failure; CLI help works. Actual video replay is running
on validation PLEASE_HELP_ME/41b5e8c9. It is a diagnostic: camera normalization differs
from cached whole-window normalization, and elapsed 30-Hz duplicate ticks do not
represent additional camera observations. No measured live/iPhone claim.

Optional context checkpoint now pins recognizer hash/vocabulary and ranks final
utterance candidates only. Blank/OTHER-leading hypotheses bypass correction because
the observed open-vocabulary regression makes rewriting them unjustified. The new
regression passed after failing before implementation; all three context tests pass.
Partial recognition remains independent; context is opt-in, not promoted as default.

Avatar v5 also rotates hand surfaces with the palm plane instead of using bone axes
alone; eight avatar tests pass. The v5 contact sheet was inspected: solid human mesh,
visible hands, but planar handshape/retrieved-rest naturalness still needs fluent
assessment. A concrete MP4 review question is pending with the user; this is quality
assessment, not implementation permission. Synthetic training eligibility remains false.
`git diff --check` passed before the latest camera additions; rerun before handoff.

## 2026-09-04 12:19 PST — corrected short-window streaming experiment shows real gain

The recommended bounded follow-up was completed without modifying the accepted Reel
path or accessing protected/reserved tests. A new Stage-1 checkpoint was adapted on 92
manually aligned ASLLRP training sign cores with Citizen, SemLex, and local phrase-core
replay. At selected epoch 9 it changed Citizen isolated top-1 from 95.24% to 94.18%,
SemLex from 85.28% to 85.79%, local phrase cores from 63.32% to 89.58%, and the 24
ASLLRP validation cores from 66.67% to 83.33%. This supports the manual annotations and
continuous-domain adaptation while passing the configured isolated-retention gates.

The corrected CTC experiment removed incomplete isolated prefixes from hard blank
supervision, added 260 full local phrase trajectories containing known-plus-`OTHER`
targets, and balanced exact ASLLRP/local phrases plus Citizen/SemLex replay. With the
old 32-frame trailing window, isolated retention improved substantially and transition
false emissions fell, but ASLLRP remained 0/12 exact. A direct observation audit found
that 32-frame pooled Stage-1 windows exposed both expected glosses in 0/12 validation
clips; 8-frame trailing windows exposed both in 8/12. These clips are only 21--54
source frames, so long windows mix the first sign, transition, and short second sign.

The selected 8-frame-window / four-frame-stride causal run reached 4/12 ASLLRP exact
and 41.67% WER, versus 0/12 and 62.50% for the original/corrected 32-frame runs. It also
reached 80.41% exact on 97 local phrase clips, 92.59% on 378 Citizen isolated clips,
82.21% on 978 SemLex clips, 96.15% exact on 52 added local `OTHER` phrases, and 2.30%
aggregate transition false emission. A matched two-frame-stride run tied ASLLRP at
4/12 / 41.67% but reduced local exact to 55.67%, Citizen to 89.15%, and took 120.75
seconds instead of 86.91; it is rejected.

The short-window result is the first genuine signer-disjoint continuous gain, but it
is not promoted: ASLLRP WER remains above 35%, Citizen retention is below the desired
93--94%, ASLLRP transition false emission is 2/12, and ASLLRP `OTHER` known-gloss WER
is 89.08%. Keep the stable Reel prototype. Proceed with the planned 30 phrases x 13
signers x 3 connected performances, using two natural-speed and one slower connected
take, exact gloss targets, manual boundaries for all validation/sealed clips and a
representative training subset, plus ordinary non-sign activity. Full methods and
matched results are in
`artifacts/reports/unified_streaming_ctc_v17_experiment_v2/README.md`.

## 2026-09-03 11:33 PST — exact-input cached Reel experiment passes offline regression gate

The current user session was measurably slower than the earlier stable Reel session:
effective landmark processing fell from 19.2 to 16.5 FPS, camera-frame dropping rose
from 10.1% to 38.5%, landmark proposals rose from 12.2 to 26.9 ms median, and full
verification rose from 224.7 to 362.9 ms median. Stage 2 was disabled. The hand image
encoder consumed 268.8 ms median and remained the dominant modeled bottleneck. A
standalone component benchmark measured 10.46 ms hands-only Apple Vision, 6.01 ms live
lips, 3.72 ms HUD, and negligible Finish geometry, implicating full-verifier/system
contention rather than the ten-finger control. The machine also had 23/24 GiB memory in
use, 10 GiB compressed, and 12.75 GiB swap occupied, but no thermal warning.

A separate `scripts/live_reel_cached_stage1_v17.py` path now reuses MobileCLIP
embeddings only for byte-identical RGB crops through a bounded 512-entry LRU. It keeps
all 16 time points, three views, landmarks, boxes, and validity masks. Twelve/eight
time-point and reduced-view alternatives were rejected because each lost at least one
Citizen validation prediction. On a real two-window overlap benchmark, cached and
baseline embeddings were bit-exact and median encoding improved 784.07 -> 563.86 ms
(1.39x). Across matched five-video local replays, both lanes produced the same five
final sequences and the same 2/5 exact count. Fourteen verifier calls per lane measured
359.00 -> 304.15 ms median total and 272.51 -> 215.15 ms median hand encoding, with an
observed 15.52% cache hit rate. This passes an offline no-regression gate but is not yet
a webcam FPS or independent accuracy result. Evidence is in
`artifacts/reports/live_reel_cached_stage1_v17_experiment_v1/README.md`. The original
Reel entry point retains the original verifier; only small dependency-injection hooks
were added so the separate command can reuse its loop. No test or sealed split was
accessed.

## 2026-09-03 08:07 PST — tiny causal sequence head is fast but fails the accuracy gate

A new non-destructive streaming experiment compared (1) a raw 61-node causal TCN and
(2) a 26,861-parameter causal depthwise TCN/CTC head over rolling logits from the
accepted phrase-adapted Stage 1. The raw model collapsed to CTC blank in its fail-fast
screen and was rejected before a full training budget. The Stage-1 evidence head is a
119 KiB checkpoint and retains per-block causal state. On this Mac/MPS, Stage 1 measured
29.33 ms median and the head added 3.06 ms median, excluding landmark extraction.

The selected development-only head reached 44.33% exact / 40.93% WER on 97 local
phrase clips and 0/12 exact / 62.50% WER on signer-held-out ASLLRP phrases. It retained
94.71% Citizen and 83.95% SemLex isolated exact accuracy. This misses the existing
Stage-2 validation reference of 92.78% exact / 2.70% WER local and 16.67% exact /
45.83% WER ASLLRP, so the new head is **not promoted or connected to live inference**.
Its speed proves the head is not the bottleneck; learned continuous boundaries and
domain coverage are.

A controlled modality ablation confirms that all existing v17 landmarks are needed:
all-landmark versus hands-only accuracy was 95.24% vs 65.08% Citizen, 85.28% vs 59.51%
SemLex, and 63.32% vs 44.02% on local phrase segments. New live-only MediaPipe mouth
nodes were not added because stored phrase archives lack them and would create schema
mismatch. A separate Stage-1 contextual adaptation added 92 genuine ASLLRP training
segments with Citizen/SemLex/local replay. Its gated epoch improved local segment
accuracy 59.85% -> 66.41% and ASLLRP held-out segment accuracy 66.67% -> 75.00%, with
Citizen 95.24% -> 94.97% and SemLex 85.28% -> 85.17%, without changing architecture or
latency. But the downstream sequence head became worse (43.24% local and 66.67%
ASLLRP WER), so this checkpoint also remains experimental.

The user's new fast learned-emission histories confirm the previously documented
accuracy tradeoff: sessions `20260903_073822_141602` and `20260903_074154_998790` did
not load the RGB hand verifier and contained many low-confidence rejects; the later
`20260903_074617_667378` comparison did load it via `--full-visual-verifier`. No default
was silently changed again during this experiment. Full model/data/latency evidence is
in `artifacts/reports/streaming_stage1_head_v17_experiment_v1/README.md`. No protected
test or external-reserved sample was accessed.
Eleven focused streaming/emission tests, compilation of all new entry points, and
`git diff --check` pass.

## 2026-09-03 07:29 PST — emission live bottleneck removed; lightweight streaming direction researched

The user's first learned-emission webcam session at
`artifacts/reports/live_reel_emission_stage1_v17/20260903_071450_141909/`
confirmed an implementation bottleneck distinct from model latency. Its 130 learned
landmark proposals took 37.11 ms median / 48.95 ms p90, while 26 subsequent hand-image
verification calls took 745.07 ms median / 979.22 ms p90 (maximum 1,088.00 ms). The
103-second session produced 1,273 landmark observations but dropped 1,604 stale camera
frames. Repeated Core ML hand-crop encoding was the dominant source of the reported
lag, not the 101-logit emission model.

`scripts/live_reel_emission_stage1_v17.py` now defaults to a landmark-only fast commit:
after two stable learned-emission proposals, it reuses that accepted result and does
not load or call the expensive hand-image verifier. `--full-visual-verifier` restores
the prior comparison path. The accepted `scripts/live_reel_stage1_v17.py` behavior is
unchanged because the new switch is enabled only by the separate wrapper. An 18-second
saved-video smoke processed all 270 frames, produced 60 proposals at 37.84 ms median,
and made no additional verifier inference. It establishes removal of the 0.65--1.09
second operation, not webcam FPS or accuracy; those require a new live session. The
speed/accuracy tradeoff is deliberate because the image verifier may reject or correct
some landmark proposals.

A primary-source review found no compatible downloadable checkpoint for the fixed
100-ASL v17 vocabulary. The direct EMNLP 2024 online CSLR model is conceptually similar
to Reel but uses two S3D streams and reports 5.1 GB V100 memory. Skeleton CSLR evidence
from CoSign and ICCVW 2025 supports grouped hand/body/face/mouth keypoints, short 1D
temporal convolution, and CTC supervision. The recommended next experiment is therefore
a sub-million-parameter causal landmark TCN with cached per-frame state and a 101-way
CTC output (blank plus 100 glosses), trained on genuine phrase sequences plus isolated
augmentation. It is a new streaming sequence recognizer, not the existing whole-phrase
Stage-2 live policy. Full paper/model assessment is in
`artifacts/reports/lightweight_streaming_stage2_research_v1/README.md`. Twenty-seven
focused tests, compilation, and `git diff --check` pass.

## 2026-09-03 00:48 PST — learned Stage-1 emission helps but does not replace sequence recognition

A complete separate Stage-1/Reel experiment now exists in
`active/v17/train_stage_1_reel_emission_v17.py`,
`active/v17/model_reel_emission_v17.py`, and
`scripts/live_reel_emission_stage1_v17.py`. Neither accepted Reel checkpoint nor its
default entry point was replaced. The selected model freezes the phrase-adapted
100-gloss classifier and learns a tiny ordered-temporal `__NO_EMIT__` head from
complete signs, incomplete prefixes, and between-sign transitions. The original 100
gloss logits are preserved bit-for-bit. Training uses permitted Citizen/SemLex replay,
all local phrase caches, and manually aligned ASLLRP training phrases; JONATHAN remains
signer-disjoint ASLLRP validation. No test or reserved RIT data was accessed.

At its validation-selected 0.77 threshold, the temporal head accepts 97.68% of 1,639
complete validation windows and catches 83.56% of 1,813 incomplete windows. The held-out
ASLLRP slice is materially weaker: 20/24 complete accepted and 16/36 incomplete caught.
The safer live threshold 0.95 accepts 99.02% complete and catches 70.60% incomplete
overall; its ASLLRP figures are 21/24 and 7/36. A first linear-head attempt is retained
as a failed baseline: it caught only 13.7% incomplete at a 95.5% complete-accept gate.

The FP16 Core ML export is 13.48 MiB and measured 11.93 ms median / 14.04 ms p90 on this
Mac. Across 378 Citizen validation clips it had zero top-1 and zero emission-decision
mismatches against PyTorch. The separate live preset probes at 0.32 seconds, every 0.08
seconds, and requires two agreeing proposals. It tied the accepted Reel timing on five
local phrases at 3/5 exact and four token edits, but committed more slowly (0.93 versus
0.67 seconds median). On the same 12 fast ASLLRP clips, it remained 0/12 exact but
improved from 20 to 17 token edits, emitted 8 rather than 5 glosses, and reduced median
committed duration from 0.67 to 0.57 seconds.

Therefore this model remains an experimental auxiliary gate, not the new default. It
demonstrates that Stage-1 temporal fine-tuning can reduce partial-window emissions, but
it cannot recover unrestricted natural sequences: the fast ASLLRP clips still lack
enough stable probes and retain classification/domain errors. A future natural stream
needs a temporal sequence model with explicit blank/boundary supervision, but does not
need to reuse the current slow whole-phrase Stage-2 live policy. Full evidence and the
run command are in
`artifacts/reports/stage1_v17_reel_emission_experiment_v1/README.md`. Twenty-five focused
tests, compilation, checkpoint/Core ML load smoke, and `git diff --check` pass.

## 2026-09-02 22:23 PST — Reel local-phrase gains do not transfer to fast ASLLRP clips

The user's newest completed webcam history at
`artifacts/reports/live_reel_stage1_v17/20260902_213908_704489/history.json` is genuinely
better under deliberately clearer articulation. Stage 2 was disabled. It produced five
nonempty FINISH sequences, including `I GO DOCTOR TOMORROW MORNING I FEEL SICK`, and the
five provisional `LESS` appearances never committed. This supports the Reel interaction
for the current signer, but expected labels were not recorded for every attempt and some
suspicious `CHILD` commits remain, so the session is not an accuracy benchmark.

A new development-only external check replayed all 12 vocabulary-covered ASLLRP
contiguous validation-cache clips that were excluded from the local phrase adapter. The
underlying rows are still `train_candidate` material, not sealed test data. With default
Reel timing and Stage 2/lips/speech disabled, exact sequence accuracy was 0/12 and total
WER was 20/24 = 83.33%. Only five glosses committed: four aligned correctly and one was a
wrong `WRITE`; deletion/under-emission dominated because these short, fast videos yielded
only two to four probes.

A model-only equal-segment comparison separated classification from live emission. On 24
ASLLRP gloss crops, original versus adapted top-1 was 10/24 versus 8/24 for the landmark
proposal and 13/24 versus 12/24 for the full verifier; full-verifier top-5 tied at 18/24.
Thus the local phrase/activity adaptation did not transfer to this small ASLLRP domain.
The failure combines signer/domain mismatch with a Reel policy too conservative for fast
clips. Full details are in
`artifacts/reports/live_reel_asllrp_external_adapted_v1/README.md`. Sixteen focused Reel
tests pass, and no protected test split was accessed.

## 2026-09-02 21:21 PST — Reel delay reduced and unsafe lip override disabled

Read-only diagnosis of the user's completed Reel session at
`artifacts/reports/live_reel_stage1_v17/20260902_205147_630595/history.json` confirmed
that the reported delay was real. Across 422.75 seconds, extraction completed 6,674
landmark observations (15.79 FPS), down from the preceding optimized session's 18.96
FPS, and discarded 4,638 of 12,180 total captured frames as stale. The 150 activity
candidates split evenly into 75 commits and 75 failures; 48 timed out. Successful
candidates required 1.00 seconds median / 1.88 seconds p90 from detected activity,
while failures occupied 2.37 seconds median. The cheap proposal itself remained fast
at 17.81 ms median, but full verification rose to 337.32 ms median and the proposal
and verifier disagreed 61/159 times.

The every-frame Apple face/body experiment caused avoidable contention. The Reel path
now detects Apple face/body on the training-matched every-eighth-frame schedule and
holds the last valid points continuously for display. Hands and MediaPipe lip markers
remain live. The optional `--dense-model-auxiliary` experiment remains available;
the isolated path is unchanged.

The 27-example GOOD/THANKYOU lip specialist was unsafe on this signer. Of 19 targeted
evaluations, it disagreed with the landmark proposal 11 times and with the original
full-verifier top class 15 times, commonly at essentially 1.0 confidence. The previous
0.999 threshold therefore did not calibrate it. Lip-based classification is now off by
default; `--lip-marker-verifier` explicitly enables it. Even then, lips may only break
a disagreement where both the landmark proposal and accepted full verifier already
select GOOD/THANKYOU. It cannot override agreement, a rejected verifier, or an
unrelated class. The raw model candidates are preserved in diagnostics.

A development-only timing sweep used five existing local phrase recordings with
unchanged confidence/margin gates and no Stage 2 or lip verifier. The old 0.62/0.14
second candidate/probe defaults scored 1/5 exact with five token edits and 1.07-second
median committed candidate duration. The selected 0.50/0.12 preset scored 3/5 exact
with four edits and 0.67 seconds median. A more aggressive 0.45/0.10 preset also scored
3/5 and four edits but added a false STOP, so it was rejected. This is fitted local
development evidence, not independent accuracy. The compact report is
`artifacts/reports/live_reel_stability_sweep_v1/README.md`; generated histories/videos
remain local. Thirty-nine focused tests, compilation, and `git diff --check` pass.

## 2026-09-02 20:09 PST — reel proposal and verifier decoupled from the display loop

The user's no-Stage-2 session at
`artifacts/reports/live_reel_stage1_v17/20260902_195337_949818/` confirmed a second,
more direct latency bug. Of 112 proposals in about 107 seconds, 83 fell through the
supposed landmark cascade into a full MobileCLIP classification; 65 stable proposals
then ran the same full classifier again synchronously on the UI thread. Proposals used
24.60 seconds and the second verifier passes used 18.21 seconds. The final candidate
contained only 21 observations across 1.91 seconds (10.49 observed FPS), and its
synchronous verifier froze the UI for 777 ms. Disabling Stage 2 therefore could not
fix the remaining lag.

`scripts/live_reel_stage1_v17.py` now keeps every provisional probe landmark-only and
schedules the full visual/hand verifier asynchronously only after two consistent cheap
proposals. The expensive verifier never runs inside the capture/display callback. One
verified hit now commits, replacing two repeated visual passes after the new two-hit
landmark stability gate. The default extraction rate is 20 FPS at a 640-pixel detector
input so the newest-frame display can refresh independently; full Stage 2 is now
opt-in with `--stage2-arbiter`. Lip landmarks remain enabled. Full nested JSON printing
is opt-in with `--verbose-predictions`, while complete structured results are still
written to the session history. Adjacent identical commits are suppressed to prevent
a transient proposal from turning one held sign into duplicate spoken glosses.

In a 16-second live-camera smoke with no person deliberately signing, the 13 landmark
proposals measured 14.3 ms median and 30.0 ms maximum with zero hand-image encoding,
versus 235.9 ms median and 501.4 ms maximum for the prior no-Stage-2 session. The smoke
was terminated externally and therefore has no valid display-FPS footer; the user must
confirm perceived camera smoothness in the actual window. Normal-speed saved-video
replays retained exact GOOD MORNING. HELLO HOW YOU produced the intended four commits
`HELLO HOW HOW YOU`; the new adjacent-duplicate guard collapses the duplicate HOW.
Twenty-two focused reel/lip/Stage-2 tests pass, compilation and `git diff --check`
pass. No original isolated source/model, sealed split, or Citizen test was touched.

## 2026-09-02 19:46 PST — first real reel-path session exposes duplicated visual inference

Read-only diagnosis of the user's first webcam run at
`artifacts/reports/live_reel_stage1_v17/20260902_194002_860837/` confirms genuine
throughput loss. During 96.96 seconds, the camera supplied approximately 28.09 FPS,
but the main loop completed only 20.86 landmark observations per second and discarded
700 stale frames. The reel path therefore lost about one quarter of captured frames;
this is not merely a display impression.

MediaPipe lips are not the primary bottleneck. A 240-frame replay of the recorded
640x360 session measured 3.11/3.49/4.84 ms median/p90/max for the 40-point lip tracker.
The architectural cost is duplicated visual inference. The 38 Stage-1 proposals used
4.40 seconds total; 27 subsequent full Stage-1 verifications used another 9.33 seconds
(277.0 ms median), including 6.89 seconds of MobileCLIP hand encoding. In parallel,
30 accepted Stage-2 windows used 9.15 seconds (221.1 ms median, 732.8 ms p90, 974.4 ms
maximum). Stage 2 attempted 78 windows in total because the current reel script feeds
it every elapsed-time window, including idle/background periods, rather than only an
active utterance. The Stage-1 verifier and Stage-2 arbiter also load separate instances
of the same hand-image encoder and can contend while the detector/display loop runs.

Prediction latency is additionally increased by policy: every stable Stage-1 proposal
runs a full verifier, and weak proposals require two verified hits. This protects
against transition fragments but can add two 0.2-0.7 second verifier passes after the
initial 0.62-second candidate. The isolated prototype feels smoother because it stops
landmark extraction while its one classification future is running and continues to
read/display camera frames; it does not run full Stage 2 every 1.067 seconds.

The live session also invalidates promotion based only on the five local replays. It
committed HELLO, YES, HOW, HELLO, LESS, YOU, HOW, CHILD, HELLO, HEAR, HOW, I, I, while
Stage 2 produced unstable sequences such as WHY COME and HELLO MORNING HOW. The next
fix should not tune the lip tracker. The minimal architectural correction is to keep
the fast Stage-1 landmark proposal and lip overlay live, remove continuous full CTC
from the display-critical path, and run full Stage-2 arbitration only for buffered
active signing/FINISH (or use the 11 ms landmark preview live). Full Stage-1 hand-image
verification should be reserved for genuinely ambiguous proposals rather than every
commit attempt. No source code, model, or dataset was changed, and no sealed/test split
was accessed.

## 2026-09-02 19:34 PST — separate reel path now uses visible lip markers and Stage-2 final arbitration

The accepted pause-delimited prototype `scripts/live_isolated_v17.py` and its models
remain unchanged. A separate `scripts/live_reel_stage1_v17.py` experiment now performs
frequent Stage-1 landmark proposals, invokes the full unified classifier as a verifier,
uses a two-hit commit lock for weak transition fragments, and feeds the already-trained
Stage-2 CTC model asynchronously as a final multi-sign sequence arbiter. Stage 2 never
replaces a one-gloss result unless a stable multi-gloss CTC sequence contains the
Stage-1 evidence, so isolated signs are not automatically expanded into phrases. For
utterances beyond the Stage-2 model's eight-window context, emissions leaving the
window are frozen into a prefix instead of silently losing the sentence beginning.

The new lip path is no longer display-only. `active/v17/lip_marker_v17.py` extracts 40
MediaPipe FaceMesh points from the outer and inner lip contours on every processed
frame, immediately discards RGB, and normalizes face position, scale, and roll before
computing shape and temporal-change features. The exact same 40 points are now drawn
on screen as white markers with black outer and inner contours. The tiny closed-pair
model at `artifacts/models/lip_marker_good_thankyou_phrase_crops_v17/model.npz` is only
allowed to choose GOOD versus THANKYOU after Stage 1 proposes one of that pair; it
cannot introduce an unrelated class. On permitted validation it scored 22/27 overall,
including 16/20 phrase crops and 6/7 Citizen clips. This helps when mouthing differs,
but deliberately does not claim that neutral-mouth GOOD and THANKYOU are separable by
lips alone.

The phrase/activity-adapted Stage-1 model is separate at
`artifacts/models/stage1_v17_unified_phrase_activity_adapt_reel_v2/best_model.pth`.
Its selected validation results were 96.03% Citizen, 89.16% SemLex, 97.03% local
isolated, 66.41% equal phrase segments, and 69.79% activity crops. Its FP16 Core ML
export is 23.58 MiB, measured 8.64 ms median after warm-up, and had zero top-1
mismatches across 378 permitted validation archives. A more aggressive pair-targeted
checkpoint improved phrase/activity accuracy to 83.01/82.43% but regressed the SemLex
GOOD/THANKYOU pair to 38.1%; it is preserved as an experiment and is not the default.

Five normal-speed held-out local phrase replays now finish exactly through the hybrid
router: HELLO HOW YOU, GOOD MORNING, MY NAME, PLEASE HELP I, and THANKYOU FRIEND. In
the last THANKYOU FRIEND replay, the raw Stage-1 proposal was GOOD and the lip marker
specialist corrected it to THANKYOU at confidence 0.99999999; a later transition
fragment was still READ, while Stage 2 supplied `THANKYOU FRIEND FRIEND`, adjacent
duplicate collapse produced `THANKYOU FRIEND`, and FINISH selected the exact sequence.
This is useful replay evidence, not an independent signer or production-accuracy claim.
Fourteen focused reel/lip unit tests pass, Python compilation passes, and
`git diff --check` passes. No Citizen test or sealed split was accessed.

## 2026-09-02 14:34 PST — newest live Stage-2 session confirms utterance-stream over-emission

Read-only inspection of the user's newest webcam session,
`artifacts/reports/live_stage2_ctc_v17/20260902_142833_531729/history.json`, confirms
that the reported extra words are model hypotheses rather than a HUD-only artifact.
Examples include `HELLO -> HELLO LIKE -> HELLO LIKE WHY -> HELLO LIKE WHY WHO ->
HELLO LIKE WHY WHO MAYBE`, and a later stream revision from `HELLO` to
`NEED HELLO GOODBYE` after its third accepted window. The first attempted stream also
resolved as `HELLO HOW ASK`, rather than the expected familiar `HELLO HOW YOU` pattern.
Across the session, 39 windows were accepted and 11 rejected; accepted post-landmark
inference latency was 282.17 ms median, with hand-image embedding still dominant.

The behavior is partly the intended Stage-2 contract and partly a live-design/data
failure. The script treats every accepted 1.067-second window between RESET/FINISH as
another part of one open continuous utterance, gives each window eight CTC time slots,
and has no explicit one-sign endpoint or idle/no-sign class. Therefore a hold,
transition, partial sign, or incidental hand motion can extend or revise the entire
phrase hypothesis and can legitimately decode two or three nonblank tokens. Greedy CTC
collapse itself is operating as implemented; the unsuitable assumption is using this
continuous utterance decoder as the default reel-like single-sign interaction.

The selected UX direction is consequently a separate Stage-1 isolated-lock loop for
the reel behavior: frequent rolling complete-sign candidates, immediate provisional
labels, stability/debounce plus duplicate suppression before committing a chip, and
RESET/FINISH defining the sentence buffer. Stage 2 remains optional for explicit
continuous-utterance mode or later confirmation, not required for the fast default.
The current Stage-1 model was trained on completed clips, so raw frame-by-frame labels
must not be committed directly; an endpoint/stability gate is still required, and the
previous wrist-motion valley alone is not reliable enough. No source code, model, or
dataset was changed during this diagnosis, and no test or sealed split was accessed.

## 2026-09-02 14:18 PST — Stage-2 cascade is fast and safe for provisional/final routing, not hard early lock

The separate landmark Stage-2 preview retraining completed in 320.66 seconds on MPS.
The selected seed 9811 epoch 6 checkpoint is
`artifacts/models/stage2_v17_landmark_cascade_preview_v1/best_model.pth` (SHA-256
`b4cce35376858d774652f39665e985fde12f408e9db54cf383c8fcce4ed6d484`). It has
[ASLLRP contiguous phrase, local phrase, held-out ASLLRP segmented] validation edit
counts [18, 17, 47], so it is weaker than the full selector and is not a replacement.

A new validation-only cascade sweep selected minimum greedy nonblank emission
probability 0.978. It used the landmark preview for 3/12 ASLLRP phrase rows, 101/254
held-out ASLLRP contextual rows, and 29/97 local phrase rows. Relative to the accepted
full selector, per-domain edits changed 9->9, 43->42, and 6->6. Thus it provides 133/363
(36.6%) fast coverage without an observed domain regression on the selection data.
This same-validation selection is optimistic and must be confirmed on new independent
portrait recordings before promotion.

The combined landmark encoder and CTC preview was exported to the new
`artifacts/coreml/Stage2LandmarkCascadePreviewV17FP32.mlpackage`. Across 109 permitted
phrase validation archives Core ML had zero decode mismatches versus PyTorch, maximum
absolute logit difference 1.62e-5, and measured 11.56 ms median / 18.13 ms p90 after
warm-up on this Mac. The FP32 package is 39.90 MiB. This proves that a reel-like
provisional label can be computed quickly after landmark extraction.

Hard early locking remains unproven. Partial-window probes at 8/16/24/28/32 model
frames showed a false lock even at 0.999 confidence. Requiring the same strict
hypothesis extension across four consecutive probes removed observed false locks but
locked only 11 tokens in 9/109 rows and zero complete phrases before their final probe.
Accordingly, the supported architecture is immediate **provisional** landmark output
followed by permanent-chip/full-Stage-2 confirmation. It is not safe to speak or
permanently append every high-confidence partial prediction.

The genuine multi-gloss training cache contains only 44 unique sequences, 50 unique
directed bigrams, and 46/100 glosses in any multi-gloss sequence. Only six local phrase
identities have repeated coverage; most of the 38 ASLLRP multi-gloss sequences have a
single example. The 1,116 ASLLRP segmented train spans add contextual isolated evidence,
not 1,116 genuine transitions. Additional phrases are not needed to fix the current
timebase or demonstrate familiar phrases, but they are needed for useful hard-lock
coverage, unseen combinations, and research generalization. The existing 20-30 phrase,
10 train + 2 validation + 1 sealed native signer plan with three genuine performances
per phrase is a meaningful next pilot if it maximizes new bigrams and low-motion
pronoun/confusion transitions; it is not enough for a general 100-gloss continuous-ASL
claim. Simultaneous cameras are correlated views, not independent performances.

New source files are `train_stage_2_landmark_cascade_v17.py`,
`evaluate_stage_2_landmark_cascade_v17.py`,
`export_stage2_landmark_cascade_coreml_v17.py`, and
`evaluate_stage_2_early_lock_v17.py`. Existing isolated and motion-valley source files
were not edited for this experiment. Seven focused live Stage-2 tests pass and
`git diff --check` passes. Full findings are in
`artifacts/reports/stage2_v17_landmark_cascade_preview_v1/README.md`. No sealed or test
split was accessed.

The code, tests, ground-truth handoff, and compact reports were committed as
`e027f72` (`Add time-normalized Stage 2 cascade experiments`). Local webcam histories,
session videos, trained checkpoints, and Core ML packages remain uncommitted/ignored;
no personal recording was added to git.

## 2026-09-02 14:02 PST — motion-valley YOU errors are boundary-dependent; landmark CTC retraining started

The user's latest motion-valley session is
`artifacts/reports/live_motion_valley_v17/20260902_133345_224087/history.json`.
Read-only inspection confirms the classifier can recognize YOU, but the wrist-motion
boundary produces inconsistent pieces of repeated attempts. In the concentrated
318-336 second interval, clips labeled NEED, YOU, HE, NEED, NEED, THEY, NEED, NEED,
NEED, YES, YOU, YES ranged from 7 to 17 motion-trimmed observations. Elsewhere YOU
was correctly accepted many times, including high-confidence clips. Across all
post-250-second candidates in the confusion family there were 29 YOU, 12 NEED, 9
TELL, 8 THEY, and 2 UNDERSTAND clips; median motion-trimmed lengths varied from 13
frames for YOU to 19 for UNDERSTAND. This disproves a simple missing-YOU-class
explanation and supports the user's cutoff diagnosis. Low-motion lexical signs and
low-motion inter-sign transitions are structurally ambiguous under this trigger, so
further global threshold tuning is not the selected path.

A new, separate `active/v17/train_stage_2_landmark_cascade_v17.py` experiment was
added without modifying the isolated or motion-valley scripts. It trains a Stage-2
CTC preview from the landmark-token slice of the existing approved caches, reconstructs
the 612-D contract as 256 landmark tokens + 256 zero hand features + 100 frozen
landmark logits, and distills the accepted full multimodal selector while applying
supervised CTC. A 24-sample smoke completed on MPS and improved validation in one
epoch. The first full run reached a best [ASLLRP phrase, local phrase, ASLLRP
contextual] edit vector of [17, 17, 48] at epoch 5, then encountered a non-finite MPS
CTC batch before packaging. The new script now uses the project's existing bounded
non-finite-batch discard policy and the safer 5e-6 learning rate; this failed partial
run is not a promoted artifact. No test or sealed split was accessed.

## 2026-09-02 12:27 PST — elapsed-time Stage 2 stays 5/5 exact at live-like 15 FPS

The separate `scripts/live_stage2_ctc_v17.py` experiment now partitions model input
by 1.067 seconds of source time rather than by 32 successful detector calls. Webcam
capture is drained on a background thread and stale frames are discarded, so slow
feature work cannot build a seconds-long camera backlog. The fixed eight-window stop
was replaced by a rolling eight-window CTC context: emissions leaving the context are
locked into a prefix while the recent context remains revisable. Apple Vision face
features are now computed only at the trained every-eighth-observation interval;
MediaPipe FaceMesh supplies display-only moving lip landmarks on each processed frame.
It never enters model features. Seven focused unit tests pass, including elapsed-time
partitioning, CTC emission positions, rolling-prefix behavior, CTC score parity, and
stable-prefix speech.

Five genuine familiar-domain phrases were replayed at only 15 extracted observations
per second, close to the webcam's measured 14.47 FPS. All remained exact: GOOD MORNING,
HELLO HOW YOU, MY NAME, THANKYOU FRIEND, and TOMORROW SCHOOL GO. Most full model
windows contained 16 observed frames spanning one second and were resampled to 32,
instead of incorrectly spanning about two seconds. This confirms the timebase repair
for familiar recordings; it is not independent signer/generalization evidence.

The 16 accepted updates still had variable post-landmark cost: 488.70 ms median,
949.41 ms p90, and a 2045.51 ms maximum in this sequential replay. MobileCLIP hand
embedding remained dominant. Therefore time normalization addresses inability to sign
at a natural pace, while a separately trained landmark-first Stage-2 preview/gate is
still needed to target reel-like immediate feedback. Evidence is under
`artifacts/reports/live_stage2_ctc_v17_timebase15_eval_v1/`. No sealed or test split
was accessed.

## 2026-09-02 12:18 PST — reference reel is an isolated-lock UX, not evidence of streaming CTC

The user supplied the local 720x1280, 30 FPS, 17.62-second copy of the previously
linked Instagram reel at `/Users/frnzlo/Downloads/What if AI could bridge the gap
between sign language and spoken EnglishThis prototype uses Medi.mp4`. Frame-level
inspection shows a MediaPipe-style hand overlay, a transient single-word label, and a
separate row of committed word chips. The visible sequence is TECHNOLOGY, USE,
IMPROVE, LIFE, WAR, NOT; transient labels change during motion, but a word is appended
only after it persists. A deliberately separate FINISH sign then advances the UI from
Hand Tracking/Emotion Detection to LLM Interpretation and Voice Output and produces
“Let's use technology to improve lives, not war.” The reel itself does not expose its
source, weights, timing thresholds, evaluation set, or accuracy and therefore is not
evidence of a continuous CTC architecture.

Its useful target behavior is a fast isolated-sign lock/debounce loop with rolling
gloss chips and an explicit utterance terminator—not frame-by-frame translation and
not necessarily Stage 2. This matches the project's cascade/isolated direction more
closely than the first Stage-2 live prototype. The current Stage-2 experiment can
still offer coarticulation-aware correction, but must first fix its observed timebase
and eight-window ceiling. No dataset or sealed evaluation set was accessed.

## 2026-09-02 11:51 PST — first webcam Stage-2 session exposes a live-timebase mismatch

The user's first real webcam run of `live_stage2_ctc_v17.py` is preserved at
`artifacts/reports/live_stage2_ctc_v17/20260902_114604_186832/`. No code was changed
in response; this entry records the read-only diagnosis. The Stage-2 experiment does
not use the isolated path's landmark-first cascade. Every accepted Stage-2 window
encodes all valid crops among 16 temporal samples x left/right/union views, then runs
the frozen multimodal encoder and both CTC heads. In this session MobileCLIP hand
embedding measured 151.43 ms median, 378.61 ms p90, and 465.93 ms maximum. The full
post-landmark update measured 179.04/411.07/497.58 ms median/p90/max. The isolated
cascade can skip hand embedding when its landmark evidence is strong; the current
Stage-2 graph has no equivalent pre-hand decision point.

More importantly, the requested 30 processed FPS was not achieved. Across 44 full
windows, the median 32-frame wall-time span was 2.143 seconds (p90 2.583), equivalent
to only 14.47 processed FPS. The selected Stage-2 training/replay contract uses
32 source frames, and the exact local HELLO HOW YOU reference is 30 FPS / 3.567
seconds. Therefore the successful offline replay fed about 1.067 seconds per full
window, while the webcam fed about twice as much real motion into the same 32-frame
tensor. Signing more slowly compounds this mismatch rather than helping: a lexical
sign can occupy multiple CTC windows and be emitted twice.

After the final reset, the webcam hypothesis evolved HELLO -> HELLO HOW -> HELLO HOW
HOW -> HELLO HOW HOW YOU. FINISH received `HELLO HOW HOW YOU`; tiny Stage 3 rendered
`Hello, how are you?`, but the recognition buffer itself was not exact. Stable-prefix
speech adds another full update before speaking a new prefix, so at the measured rate
it adds roughly 2.1-2.6 seconds. YOU was delayed further by an intervening rejected
zero-hand window. Nine accepted windows in the full session had at most ten detected
hand frames, and eight had hand presence below 0.2; these sparse windows can consume
context or preserve unstable hypotheses. The user reset eleven times.

The fixed eight-window checkpoint limit is also a hard UX ceiling: once eight accepted
windows are stored, the prototype clears incoming frame buffers and stops extracting
until RESET or FINISH. This is unsuitable for an open-ended conversational live loop.
The priority is now timebase correction and streaming-state design, not threshold
tuning: decouple capture from expensive extraction, form model windows by elapsed time
at the training-equivalent rate, avoid making slower signing the workaround, and then
measure whether a display-only lightweight lip tracker can retain continuous mouth
motion while Apple Vision face features remain sampled at their trained interval. A
Stage-2 cascade cannot be enabled by toggling the existing isolated cascade; it would
require a separately validated landmark-only preview/gate or a conditional Stage-2
encoder. No sealed test or held-out dataset was accessed.

## 2026-09-02 11:42 PST — separate live v17 Stage-2 CTC experiment passes familiar phrases

The accepted v17 phrase-agnostic general CTC selector was found in the existing logs
and connected to a new, separate `scripts/live_stage2_ctc_v17.py` path. The working
`live_isolated_v17.py`, failed fixed-overlap experiment, and motion-valley experiment
remain unchanged. The new path uses the parity-validated Core ML frozen multimodal
encoder and primary/specialist CTC heads, then mirrors the saved general selector's
90/10 blend, +0.30 blank calibration, and exact specialist CTC path-score rule in
NumPy. It processes non-overlapping 32-source-frame windows, matching Stage-2 training,
and supports at most the checkpoint's eight-window context. It does not use beam search
or the stride-8 overlap previously shown to degrade accuracy.

The live camera continues extracting and drawing face landmarks every processed frame,
while only every eighth face sample enters model features to match the v17 training
contract. The UI updates the current greedy CTC hypothesis after each completed window.
Only the prefix surviving a following update is eligible for immediate gloss speech;
FINISH closes a 4-31-frame tail, naturalizes the latest full hypothesis, clears queued
gloss speech, and speaks the finished sentence. RESET clears only current visible
state while keeping JSON/video evidence. The frozen encoder and both CTC heads run in
Core ML; no large PyTorch research graph is loaded in the live path.

Five genuine, vocabulary-covered local development/reference recordings were replayed
through the full path. All five final hypotheses were exact: GOOD MORNING, HELLO HOW
YOU, MY NAME, THANKYOU FRIEND, and TOMORROW SCHOOL GO. This is familiar-domain
execution/development evidence, not independent accuracy: those phrases are inside the
model's established local development domain. Across 16 accepted updates, median/p90/
maximum post-landmark latency was 235.77/291.89/351.34 ms. The CTC selector itself took
roughly 3-5 ms; MobileCLIP hand-image embedding dominated. At 30 processed FPS the
first full-window update still requires about 1.07 seconds of signing context before
that inference cost. GOOD MORNING first displayed an unstable THANKYOU, then revised
to GOOD and finally GOOD MORNING; stable-prefix speech correctly withheld the unstable
first guess.

A separate EOF-FINISH smoke on MY NAME logged one completed utterance and rendered
`My name.` through the deterministic literal path. Four focused tests cover exact
NumPy/PyTorch CTC-score parity, repeat collapse, stable-prefix speech, and isolated
defaults. Full evidence is under
`artifacts/reports/live_stage2_ctc_v17_local_phrase_eval_v1/` and the FINISH smoke is
under `artifacts/reports/live_stage2_ctc_v17_finish_smoke_v1/`. No sealed test,
Citizen test, SemLex test, local test, 2M-Flores devtest, or consumed RIT row was
accessed.

## 2026-09-02 11:08 PST — motion-valley live trigger made more responsive

After the user's first real live motion-valley session, the experimental defaults were
changed from hybrid, motion ≤0.008 for 0.16 seconds to cascade, motion ≤0.010 for 0.12
seconds. The latest session had six clips lasting 1.38–3.12 seconds and 238–489 ms
classification latency; it accepted HELLO twice and HOW once. The higher motion cutoff
is less vulnerable to small wrist jitter, the shorter hold saves one processed frame,
and cascade can skip expensive hand-image encoding when landmark evidence is strong.
Uncertain signs still use the unified fallback. This explicitly trades some boundary
precision for faster response and may split internal holds; the neutral/pause path is
unchanged. The user must validate the new live behavior before any accuracy claim.

## 2026-09-02 10:46 PST — no-pause path fails the nine-local-phrase accuracy gate

The unchanged 1.2-second/0.2-second-stride streaming experiment was evaluated on all
nine project-owned representative phrase recordings from the genuine local-motion
reference report. These are development/train-source diagnostics, not a new
signer-disjoint test. Citizen validation/test, SemLex test, and the local test partition
were not accessed.

All nine runs completed and wrote independent history/video evidence. Across 97 windows,
69 passed the existing Stage-1 gates. Median classification latency was 548.88 ms and
p90 was 1,334.46 ms. No stale-window backlog accumulated. Nevertheless, exact sequence
accuracy was 0/9 overall and 0/5 among the phrases entirely covered by the locked 100.
Outputs were: GOOD_MORNING -> `EAT MORNING`; HELLO_HOW_YOU -> `HELLO HOW`;
I_WANT_FOOD -> `UNDERSTAND WANT`; MY_NAME -> `NAME`; PLEASE_HELP_ME ->
`STOP HELP I`; SORRY_I_LATE -> `SORRY`; THANKYOU_FRIEND -> `BAD NAME DOCTOR`;
TOMORROW_SCHOOL_GO -> `TOMORROW LESS`; YESTERDAY_TEACHER_MEET -> `ANSWER`.

Four prompts are structurally impossible to recover fully with the current vocabulary:
FOOD, ME, LATE, TEACHER, and MEET are absent class labels. More importantly, even the
five fully covered prompts fail because fixed windows include partial signs/transitions
and short signs may not survive the two-window agreement gate. The mechanics pass but
the accuracy gate fails. Do not replace the working pause-delimited prototype or simply
accept every window; the next serious no-pause experiment needs a learned boundary or
framewise/Stage-2 sequence model. Full evidence is under
`artifacts/reports/live_streaming_v17_local_phrase_eval_v1/`.

## 2026-09-02 10:36 PST — separate no-pause overlapping-window live experiment added

The working pause-delimited path in `scripts/live_isolated_v17.py` remains intact. A
separate `scripts/live_streaming_v17.py` experiment now continuously extracts Apple
Vision landmarks, classifies the newest 1.2-second window at most every 0.2 seconds,
and never queues stale windows. It defaults to the fast cascade, keeps landmark/lip
sampling aligned with Stage-1 training, preserves the white-point/black-bone display,
records low-resolution video and full JSON evidence, and retains RESET, FINISH, local
tiny Stage 3, and native speech. It does not require a neutral pose or explicit pause.

Two consecutive accepted windows must agree before a gloss enters the visible buffer.
The same prediction run emits once; a different stable label or two rejected windows
rearms it. This suppresses overlap duplicates but deliberately cannot distinguish two
adjacent repetitions of the same sign. The approach is overlapping-window isolated
Stage 1, not CTC Stage 2, so windows can still contain transition motion or pieces of
neighboring signs and no continuous-accuracy claim is justified yet.

Four focused stabilizer/default tests and direct CLI/compile checks pass. An end-to-end
smoke used only the quarantined Citizen training clip
`6226330398612929-W.H.A.T.mp4`; Citizen validation/test were not accessed. It processed
six windows, emitted one stabilized gloss, wrote history/video, and measured roughly
341–406 ms per fallback classification without building a backlog. It stabilized as
TELL rather than the quarantined folder label WHAT. This is successful execution and
latency evidence, not accuracy evidence; the next meaningful check is a labeled live
session of naturally connected locked-100 signs.

## 2026-09-02 09:46 PST — explicit RESET/FINISH utterance UX and local Ollama rephrasing implemented

The live laptop prototype now implements the preferred endpoint-delimited path:
isolated Stage-1 predictions accumulate in a visible bottom gloss buffer; `RESET`/`R`
clears all visible utterance state without deleting predictions, video, or events; and
`FINISH`/`F` waits for an in-flight classifier, closes a genuinely active sign if one
exists, consumes the accepted non-UNKNOWN gloss buffer, and sends that one utterance to
the installed `llama3.2:1b`. A reset epoch prevents an old classifier or Ollama result
from reappearing or speaking after RESET, although the result remains in the audit log.
This is still pause/endpoint-delimited Stage 1 plus text rendering, not continuous
recognition and not a replacement accuracy claim for Stage 2.

Ollama uses the local `/api/generate` endpoint on a separate single-worker queue and is
warmed asynchronously with a 30-minute keep-alive. The prompt requests one short
meaning-preserving sentence and JSON containing the exact input gloss audit. The result
is accepted only when `used_glosses` exactly matches the input sequence and the sentence
is nonempty/bounded; API, JSON, or gloss-audit failure uses the existing reviewed
Stage-3 template when available and otherwise literal gloss-preserving English. This
guard detects dropped/reordered/replaced glosses but cannot prove that arbitrary natural
language is semantically faithful, so all prompt/response/fallback evidence is retained
and LLM output must not be treated as linguistic ground truth.

Native macOS speech is now queued rather than stop/restarted: each accepted gloss is
spoken after its HUD update and the finished sentence is queued behind those glosses.
The finished gloss sequence remains visible while its final sentence is displayed or
spoken. `history.json` format version 2 records predictions across resets, buffer/display
epochs, reset/finish events, raw utterance glosses, prompt, raw model response, fallback
decision, model latency, final sentence, and speech queue/start times. The low-resolution
session video behavior is unchanged.

A real local `llama3.2:1b` integration probe returned `I feel sick.` for
`[I, FEEL, SICK]` with an exact gloss audit. Cold asynchronous warm-up was 5.11 seconds
and the subsequent generation was 2.47 seconds on this Mac; that generation is outside
the camera/extractor thread. Eleven focused live-script tests pass, including control
hit-testing and accepted/rejected Ollama audits. A no-display/no-speech/no-Ollama replay
of a quarantined Citizen training clip completed end to end and wrote the version-2
history/video; its rejected WHAT prediction is smoke evidence only, not accuracy
evidence. No Citizen validation or sealed test data was accessed in this change.

## 2026-09-02 09:20 PST — live SICK/FATHER confusion is a one-hand variant and local coverage gap

The latest live session `20260902_090932_943567` contains 47 predictions, including a
long user-described SICK/FATHER/MOTHER/FEEL comparison sequence. Exact intended labels
were not stored per event, so the session must not be silently relabeled for training.
The observed outputs nevertheless form a coherent learned confusion: SICK, FATHER,
WHY, KNOW, and THINK repeatedly occupy the top candidates. One early attempt was
accepted as SICK; later attempts variously put SICK first but reject it, or rank FATHER
or WHY above it. Hand/face coverage is generally present and the cascade calls the
unified fallback on the ambiguous clips, so this is not merely a missing-hand or
boundary failure.

The training-cache audit explains the domain failure. Current Citizen train/validation
support for SICK, FATHER, MOTHER, and FEEL is respectively 14/4, 14/3, 14/3, and 15/4,
and the unified head gets all fourteen focused Citizen validation clips correct.
SemLex train/validation support is 23/18, 10/6, 14/6, and 16/12, with only one SICK
validation error. In contrast, the local cache has **zero SICK** train or validation
examples but 85/18 FATHER, 135/26 MOTHER, and 153/27 FEEL. Moreover, 92.9% of Citizen
and 92.0% of SemLex SICK landmark archives show both hands, whereas the user's SICK is
the one-handed reduction and the FATHER/MOTHER/FEEL training examples are overwhelmingly
one-handed. This is a domain/lexical-reduction coverage gap, not a class-index mismatch.

Three already-downloaded PopSign SICK audit clips are genuinely one-handed. The
landmark primary predicts two as SICK with scores 0.829 and 0.889 and misses one as
SLEEP at 0.268. They are useful reviewed supplement candidates, but three clips alone
are not enough for durable adaptation and do not change PopSign's non-primary status.
The current Citizen-only mouth and lower-face teachers are also not safe fixes for this
set: mouth validation is SICK 1/4, FATHER 2/3, MOTHER 3/3, FEEL 1/4; lower-face is
SICK 0/4, FATHER 2/3, MOTHER 1/3, FEEL 1/4. Do not activate them on this confusion
without retraining and a focused gate.

The live cascade itself demonstrates the intended UX advantage but also a transfer
gap. Across 44 cascade predictions, 28 (63.6%) invoked the unified fallback versus
11.4% on Citizen validation. Primary-only median post-clip work was 21.75 ms, fallback
median was 276.65 ms, and actual within-clip processing cadence was 14.69 FPS median.
The preferred prototype architecture is therefore pause/endpoint-delimited Stage 1 ->
token accumulation -> explicit FINISH -> Stage 3/speech, with Stage 2 retained as an
optional research/continuous-sign path rather than the default UX. This remains a
controlled-signing interface: without separable endpoints it cannot decode fully
coarticulated continuous signing.

Retraining is warranted, beginning with reviewed one-handed SICK plus balanced hard
negatives FATHER, MOTHER, FEEL, WHY, KNOW, and THINK, while retaining two-handed SICK.
Handshape and hand-to-face location are the primary signal. Detailed Apple Vision lip/
face motion should be a separate supplementary expert, promoted only if it improves the
focused confusion gate without reducing all-class signer-disjoint validation. The
current landmark augmentation mirrors, rotates, scales, warps time, and drops a few
random nodes, but it never models the complete one-hand SICK reduction. A targeted
label-preserving augmentation can use the existing two-handed SICK clips: retain the
hand closest to the face and mask the lower/stomach hand on only a subset of repeats,
while keeping the unmodified two-hand form in training. This provides substantially
more one-hand evidence than the three real PopSign candidates without fabricating
handshape or motion. Before
using new live captures for supervision, the collector must save intended gloss, exact
v17 tensors, hand crops, detailed lip landmarks, and event boundaries; the current
low-resolution session MP4 is reference evidence, not safe training input. Full numeric
evidence is recorded in
`artifacts/reports/live_isolated_v17_sick_confusion_audit_v1/report.json`. No sealed
test split was accessed.

## 2026-09-02 09:05 PST — default/cascade live A/B mode added as clickable controls

The laptop isolated-sign UI now exposes two clickable controls without replacing the
default. `DEFAULT` retains the existing unified-first hybrid. `CASCADE` runs the
13.29-MiB `Stage1OrientationV17` landmark model first and invokes the unified
landmark+hand model only when the primary softmax score is below the validation-selected
0.70 threshold. Existing targeted GOOD/THANKYOU real-pixel visual reranking remains
available, as does the YOU/NEED landmark reranker after a unified fallback. The selected
mode is snapshotted at classification submission, so clicking during an in-flight
classification safely changes the next sign. `--mode cascade` starts directly in the
experimental mode, while the ordinary command still starts in `hybrid`/`DEFAULT`.

The cascade avoids hand-crop extraction and MobileCLIP2 encoding on confident primary
clips rather than merely running both models and choosing afterward. A real saved
Citizen validation THANKYOU replay produced the correct accepted gloss with primary
score 0.9039, no unified fallback, zero hand-image encoding, and the expected targeted
GOOD/THANKYOU visual reranker. Forcing the threshold to 1.0 on the same clip exercised
the fallback and remained correct, with `cascade_fallback_used=true` and 219.34 ms of
hand-image encoding. The ordinary-threshold and forced-fallback histories are under
`artifacts/reports/live_isolated_v17_cascade_smoke/20260902_090447_087648/` and
`artifacts/reports/live_isolated_v17_cascade_smoke/20260902_090521_494952/`.
A second ordinary-threshold Citizen validation replay recognized HELLO correctly with
primary score 0.9129, no hand or visual fallback, 16.23 ms classifier work, and 25.38 ms
total post-clip processing. Its history is under
`artifacts/reports/live_isolated_v17_cascade_smoke/20260902_090730_513829/`. These
single-clip timings are implementation smoke evidence, not an end-to-end live latency
distribution or accuracy estimate.

MediaPipe Face Mesh was not added. Apple Vision already exposes fuller outer/inner-lip
regions, but the locked v17 schema intentionally samples four mouth anchors among its
15 face nodes. Adding more Apple or MediaPipe points to the classifier would change the
input schema and require training a new face-motion branch; untrained extra points do
not improve accuracy. For the present prototype, every-frame Apple lip anchors provide
continuous non-RGB tracking/display while the already-trained genuine mouth/lower-face
pixel teachers provide targeted supplementary evidence. The previously mentioned
"fresh signer-disjoint" set means only a small untouched portrait laptop-camera
transfer gate, not another large training collection; the existing signer-disjoint
corpora remain the basis for model fitting and offline validation.
Eight live-prototype tests plus all seventeen extractor tests pass (25 total), including
button hit-testing. Both cascade primary-only and forced-fallback saved-video paths pass;
Python compilation, report JSON parsing, CLI help, and `git diff --check` pass.

## 2026-09-02 08:56 PST — cascade measured; continuous lip landmarks enabled without changing Stage 1 inputs

A Citizen validation-only cascade study measured the lightweight landmark/orientation
model at 362/378 (95.77%) and the unified landmark+hand model at 364/378 (96.30%). A
landmark-primary score threshold of 0.70 invoked the unified fallback for 43/378 clips
(11.38%) and matched the unified model's 364/378 result; the existing targeted
GOOD/THANKYOU visual reranker raised the simulated result to 366/378 (96.83%). The
two-model oracle ceiling was 367/378 (97.09%). The threshold was selected on the same
validation split and some primary errors are high-confidence, so the cascade was not
made the default. It first needs a fresh portrait, signer-disjoint development gate.
No Citizen test or other sealed split was accessed.

An extraction benchmark on 90 frames of one Citizen validation video, with a
same-aspect maximum side of 720 pixels and body every eight frames, measured hands plus
sparse face at 8.06 ms median / 22.60 ms p90 and hands plus face every frame at 18.44 ms
median / 26.70 ms p90. The latter stays below the 33.33 ms per-frame 30-FPS budget on
this Mac; it is not an iPhone or sustained-thermal claim.

The live prototype now requests genuine face landmarks on every processed frame, so
lip movement remains continuously visible without requiring RGB display. A new
`face_for_features` observation flag keeps only every eighth face sample in Stage 1's
landmark tensor, preserving the sparse training-time input contract while allowing the
UI and quality tracking to use all intervening face/lip observations. Real RGB remains
available only to the targeted visual tie-break when invoked. The measurement and
decision record is
`artifacts/reports/live_isolated_v17_fast_cascade_validation_v1/report.json`.
Seven live-prototype tests plus all seventeen v17 extractor tests pass (24 total),
including the new regression test that every detected face can reach display while
only scheduled faces reach Stage 1. Python compilation and `git diff --check` pass.

## 2026-09-02 08:47 PST — reel architecture reviewed; fast cascade recommended, not implemented

The referenced Instagram reel `DXJ0G8ADPc1` publicly describes MediaPipe tracking 21
points per hand and 468 face points, sign recognition/token accumulation, a local LLM
for emotional/contextual English rewriting, and ElevenLabs speech. It publishes no
vocabulary size, signer-disjoint split, held-out recognition accuracy, WER, end-to-end
latency, or sustained-device measurements. Its apparent demo accuracy/speed must not be
treated as benchmark evidence; it may be a small scripted vocabulary and signer-specific
demonstration. The LLM and speech layers do not improve the upstream gloss recognizer.

The comparable prototype does not require replacing v17. The recommended next accuracy/
speed experiment is a cascade: `Stage1OrientationV17` landmark Core ML as the immediate
primary, followed only for low-score/low-margin or predeclared hard-confusion cases by
the unified hand-RGB model and targeted face/landmark specialists. The existing primary
package is 13.29 MiB, has 95.77% Citizen validation top-1, exact 378/378 Core ML/PyTorch
top-1 parity, and a recorded 16.31 ms median / 21.67 ms p90 classifier-only latency on
this Mac. The unified student has 96.30% Citizen validation top-1 but adds the 43 MiB
MobileCLIP image encoder and per-crop work. A new local probe observed 8.68 ms median /
12.94 ms p90 for the orientation package, but the canonical recorded benchmark remains
the reported figure. Neither figure includes camera extraction, boundary time, or
iPhone thermals. No sealed test split was accessed.

A separate explicit FINISH gesture is viable later as a control channel: it commits the
accumulated gloss buffer to Stage 3/LLM and speech, and should not be part of the 100
lexical outputs. It only marks utterance end; it does not segment adjacent signs inside
a continuous phrase. Without pauses or per-sign commit gestures, Stage 2/online temporal
segmentation is still required. Per the user's priority, no finish gesture, LLM, cloud
voice, or cascade code was added in this discussion turn; recognition accuracy and
camera-to-gloss latency remain the next gate.

## 2026-09-02 08:39 PST — YOU/NEED specialist, live scheduling, synchronized speech

The newest live history `20260902_082833_471330` contains seven accepted events. Two
intended YOU attempts were strongly classified NEED with YOU second (NEED 0.960 versus
YOU 0.007, then NEED 0.935 versus YOU 0.022), while another performance was strongly
YOU (0.953 versus NEED 0.007). The recorded frames show the difficult domain case: the
straight index points nearly into the camera, so 2-D foreshortening resembles NEED's
bent-index silhouette. This is not a label-map error.

That session also revealed the nominal 30-FPS loop was only observing about 12-13 FPS.
When the loop was even slightly late, its deadline advanced a full interval from the
current time and systematically skipped the next camera frame. The deadline now catches
up without adding that extra interval. Apple Vision is also paused while the post-clip
Core ML classifier owns the hardware; previously concurrent Vision submissions caused
hand-encoder latency to spike from the roughly 0.2-0.4-second offline range to 1.2-2.3
seconds, with total live classification reaching 2.65 seconds. New histories record
the actually observed processing FPS. A fresh interactive session is still required to
measure sustained live FPS and confirm the contention fix on the camera.

Default hybrid inference now invokes the existing selected landmark-only Stage-1 model
only when unified Stage 1's top two are exactly YOU and NEED, using it to rerank only
that pair. This specialist is appropriate for the user's straight-index distinction and
keeps all other classes unchanged. The landmark checkpoint and the final targeted rule
both retain 4/4 YOU and 3/3 NEED on Citizen validation. This seven-clip check is a small
development gate, not a new accuracy estimate; no sealed test data was accessed. A raw
YOU saved-video smoke remains correct and accepted at 29.289 observed FPS with 351.9 ms
post-clip classification.

Speech no longer launches a new `say` process before the HUD is painted. One native
`NSSpeechSynthesizer` is retained in memory; the accepted word is drawn, `waitKey`
commits that frame, and the same result then starts speech. Any older utterance is
stopped rather than queued. The main overlay now draws black hand/body bones and white
hand/body/face landmark points. Six live-prototype tests, including an exact pixel-color
rendering check, plus all seventeen extractor tests pass (23 total); Python compilation,
the native speech API probe, and `git diff --check` also pass.

## 2026-09-02 08:26 PST — live/Stage-1 mismatch fixed; fast-first hybrid gated

The user's first live sessions exposed systematic `THANKYOU` predictions for intended
GOOD/HUNGRY and a head scratch accepted as HOME. The principal implementation mismatch
was concrete: body/face requests were scheduled on the global camera frame counter and
then filtered a second time by each clip's unrelated local frame index. Seven of eight
recorded live clips consequently had zero shoulder coverage, while only 19.0% of 378
Citizen validation archives have zero shoulder coverage. The second filter is removed;
any globally scheduled auxiliary detection is retained. A regression test now proves a
body frame at a nonzero clip offset survives and selects shoulder-width normalization.

Live defaults now restore the training-side temporal/image contract where practical:
30 processed FPS, fixed body/face interval 8, 1280-pixel real RGB crop frames, and a
same-aspect 720-pixel Vision detection copy. Face detection is no longer run on every
frame. Clips are trimmed to their moving interval before the 32-frame resample, and the
neutral close delay is 0.20 seconds instead of 0.55 seconds. The 720-pixel detector was
retained because replaying the 640-pixel reference recordings lost the hand detections;
speed is obtained from sparse auxiliary requests and fast-first classification, not by
sacrificing the primary hand signal.

Default mode is now `hybrid`: selected unified landmark+hand Core ML runs first. Frozen
real-pixel mouth/lower-face teachers run only when the fast model's top two are exactly
GOOD and THANKYOU, and only rerank that pair. HUNGRY and all other classes remain
hand/body-led. On the existing Citizen validation logits, the unified model got 9/11
GOOD/THANKYOU/HUNGRY clips correct (GOOD 4/4, THANKYOU 1/3, HUNGRY 4/4); the targeted
tie-break corrected both THANKYOU errors while retaining GOOD 4/4 and HUNGRY 4/4.
This 11-clip development check is not a new accuracy estimate. No sealed test data was
accessed.

Protective prototype rejection is active by default: score 0.25, margin 0.08, maximum
2.5-second motion interval, and an explicit insufficient-hand-evidence rejection. A
rejected result is `UNKNOWN` and is never spoken. Pair-reranked clips use combined fast
GOOD+THANKYOU evidence against the third class for this gate, rather than applying a
threshold across incompatible logit scales. This remains a heuristic closed-set gate;
reliable open-set rejection still requires independently held-out nonsign/background
data and likely an explicit OTHER/no-sign training class.

Fresh saved-video smoke results at the live 720/1280 settings are correct and accepted:
GOOD in 660.4 ms, THANKYOU in 752.2 ms, and HUNGRY in 271.2 ms after clip close. With
the 0.20-second neutral close, these correspond to approximately 0.86, 0.95, and 0.47
seconds before speech launch on this Mac, excluding ordinary camera scheduling jitter.
The user's saved head-scratch reference is now `UNKNOWN` for insufficient hand evidence
instead of HOME. The low-resolution 640-pixel session recordings are references rather
than exact extractor replays; the user's three earlier ambiguous sign segments also lost
hand evidence when re-extracted from those recordings, so a fresh live signer retest is
still required before claiming the reported mistakes are solved.

`test/test_live_isolated_v17.py` now also covers motion trimming and global/local
auxiliary scheduling. Its five tests plus all seventeen focused v17 extractor tests pass
(22 total); Python compilation, CLI help, and `git diff --check` pass. The usage guide
documents hybrid behavior, rejection, timing, and the extraction contract.

## 2026-09-02 07:56 PST — laptop live isolated 100-gloss prototype implemented and saved-video gated

`scripts/live_isolated_v17.py` is now the runnable laptop prototype. It uses the
locked checkpoint label map, built-in/OpenCV camera input, Apple Vision hands+face on
each processed frame and body at a time-adjusted auxiliary interval, one shared
detection pass for segmentation/landmarks/real-pixel crops, a neutral-return plus
low-motion automatic boundary, top-three model scores and margin, hand/face/motion
quality, boundary progress, accepted history, nonblocking macOS `say`, and a
single-worker classification queue. It saves an atomic `history.json` after every
prediction and a 640-pixel-wide 15-FPS MP4 without aspect distortion. Live frame
history is bounded to five seconds so long camera sessions do not retain unbounded
full-resolution frames. Saved `--video` development files are intentionally treated
as one isolated clip and classified once at EOF; only the webcam path uses automatic
segmentation.

The default `--mode lip-aware` reproduces the frozen per-sample-zscore four-stream
teacher weights: 0.30 landmark, 0.15 mouth, 0.35 lower face, and 0.20 hand. Both
visual views use their actual face-aligned pixels. Their identical frozen Auto-AVSR
frontend is shared in memory and receives mouth+lower-face as one two-view batch;
this retained the exact `HELLO` output scores while reducing the post-clip latency
from 1,140.2 ms to 736.8 ms on the same run setup. `--mode fast` uses the selected
unified landmark+hand Core ML model and therefore exposes only the four lip
landmarks, not full visual lip reading. Core ML/MPS lazy compilation is paid before
the camera opens. The initial un-warmed 9.07-second measurement is not representative
and is retained only as an artifact; all reported runtime measurements below are
after explicit warm-up.

Two Citizen signer-disjoint **validation** videos passed the saved-video gate in both
architectures. Final representative results are: lip-aware `HELLO -> HELLO`, score
0.5801, margin 0.5141, mouth/lower validity 100%, 736.8 ms after clip close; lip-aware
`THANKYOU -> THANKYOU`, score 0.5460, margin 0.5164, both visual views 100% valid,
755.6 ms before the exact batching optimization; fast `HELLO -> HELLO`, score 0.8934,
margin 0.8893, 375.4 ms; and fast `THANKYOU -> THANKYOU`, score 0.8329, margin
0.8180, 177.4 ms. These are pipeline smoke cases, not a new accuracy estimate. The
low-resolution reference output was probed as 640x480, 15 FPS, 25 frames / 1.667 s
for the final `HELLO` run. Final lip-aware evidence is under
`artifacts/reports/live_isolated_v17_lip_validation/20260902_075549_892643/`; the
latest fast evidence is under
`artifacts/reports/live_isolated_v17_fast_validation/20260902_075436_955044/` for
`HELLO` and `20260902_075342_952405/` for `THANKYOU`.

`test/test_live_isolated_v17.py` covers motion-to-rest emission, EOF fallback, and
the critical rule that a static hold away from the learned neutral pose must not end
a sign. The three new tests plus all seventeen focused v17 extractor tests pass (20
total); Python compilation and targeted `git diff --check` pass. Usage and explicit
scope/score caveats are in `artifacts/reports/live_isolated_v17/README.md`. The
MacBook camera index 0 opens successfully and returned one 1920x1080 frame. The live
UI has not been signer-tested in this noninteractive run, so the next gate is a user
session with neutral -> one normal-speed sign -> neutral.
No Citizen test, other sealed split, Stage 2, or naturalization model was accessed.

Research basis: Apple's live Vision examples support per-frame pixel-buffer requests
with explicit prediction backpressure; Apple also provides sequence request handling
and live face tracking. The 2024 EMNLP online CSLR work supports isolated-dictionary
recognizers as an online baseline but identifies the offline-CTC/short-window mismatch
that prevents calling this continuous recognition. Manual/nonmanual fusion evidence
supports retaining mouth/face input. Raw neural softmax is not generally calibrated,
so the HUD and JSON deliberately call these values `model_score`, not confidence.
Primary references: Apple Vision hand pose, live object recognition, gesture sample,
and face tracking documentation; Sincan et al., EMNLP 2024, *Towards Online
Continuous Sign Language Recognition and Translation*; Gueuwou et al., LREC 2020,
*Sign Language Recognition Using Neural Network*; and Guo et al., ICML 2017,
*On Calibration of Modern Neural Networks*.

## 2026-08-26 23:07 PST — live-camera v17 inference integrated into the Flutter app

The original Flutter design at
`/Users/frnzlo/Documents/machine_learning/mobile_app/slt_mobile_app` is now a
functional iOS-first record/stop/translate app rather than a UI mock-up. It uses the
official Flutter `camera 0.12.0+2` plugin, defaults to the front camera, records a
complete clip without audio, and sends the saved file through the proven Apple Vision
v17 orientation/aspect-ratio-safe extractor. The locked Stage-2 Core ML CTC model
naturally returns one gloss for a single recognized sign or multiple glosses for a
recognized phrase. The bounded Stage-3 renderer exposes the gloss sequence, reviewed
English when an exact template exists, and the literal gloss-preserving fallback
otherwise. The UI supports portrait and landscape layouts and includes clear status,
failure, retry, and conversation actions. Android native inference remains untouched
and explicitly unavailable.

A persistent Settings switch enables device benchmarking; it is off by default and
normal users see no benchmark panel. An enabled run performs one extraction, five
warmups, and 20 timed Stage-2 inferences, then shows extraction time, median/p90 model
time, before/after resident memory, and thermal state. It writes and shares an atomic
JSON report pinned to the Stage-2 candidate/checkpoint/package/vocabulary hashes and
Stage-3 manifest hash. Simulator reports set
`hardwarePerformanceClaim=false` and `thermalsInterpretable=false`; physical-device
runs identify themselves separately. Normal inference skips the five benchmark
warmups. Camera captures are marked end-to-end camera-to-gloss evidence, while no
physical-iPhone performance or accuracy result is claimed until the app is run on the
phone.

The app bundles exact copies of the three selected Core ML packages and exact
manifests. Manifest SHA-256 values are unchanged: vocabulary
`3a665bda8d2b916c504406be815e601eeb55badfe62afcec42c7869885eab7cf`,
Stage-2 mobile contract
`342101de35b0c3065d730b44172e112b348281234b6e870f421ffd069c32adfa`,
and Stage-3 naturalizer
`68c7ce67632f66ee70fa3b3d36eb8df33ad72dc674edbf3b720e93c1240f84a6`.
The copied Stage-2 Swift runtime differs from the proven benchmark source only by a
configurable warmup count.

Validation passed: `flutter analyze` has zero issues, both focused Flutter tests
pass, both plist files lint cleanly, and the unsigned iOS Release build succeeds for
arm64 with iOS 17.0 minimum and bundle ID `com.kokoab.sltMobileApp`. The resulting
`Runner.app` is 126.2 MB and contains exactly the three compiled model bundles plus
the three pinned manifests. Large local model packages are ignored by the mobile
app's `.gitignore`. No Citizen, SemLex, local, ASLLRP, or 2M-Flores test split was
accessed.

## 2026-08-10 09:25 PST — Streaming MoViNet baseline completed; Kaggle failure reconciled

The eight private train/validation multipart datasets and the private offline Model
Garden wheel dataset are all complete and `ready` on Kaggle. The assembled archive is
696 MiB with SHA-256
`699834265f70ae6226b4692a0058b7c1ef2ea325d935941bbdf608af2b9c8bab`; the test split
is absent. Kaggle kernel versions 1–5 fixed, in order, recursive input discovery,
offline package installation, the archived v17 package initializer, and explicit GPU
selection. The final server metadata records `enable_gpu: true` and an exact
`NvidiaTeslaT4` or `NvidiaTeslaP100` machine shape. The account API reports six unused
GPU hours, but every batch worker still had no `/dev/nvidia*`, no `nvidia-smi`, a
CPU-only PyTorch build, and no TensorFlow GPU. Separate script and notebook probe
kernels reproduced the same scheduler failure even with embedded Kaggle accelerator
metadata. No Kaggle run produced model metrics.

The local host is an M4 MacBook Air with 24 GiB memory. A clean Python 3.11 TensorFlow
2.16.1 / Model Garden 2.16 / Metal environment confirmed that the official TensorFlow
MoViNet graph still fails at its XLA-compiled Conv3D stem. Its CPU path passed but took
45.5 seconds for one training batch plus one validation batch, so it remains unsuitable
for the complete schedule.

A second, faithful execution path is now implemented in
`active/v17/train_stage_1_movinet_torch_v17.py`. It uses the MIT-licensed
Atze00/MoViNet-pytorch streaming A0 architecture with 2+1D convolutions and converted
official Kinetics-600 weights. Source is pinned to commit
`c2d1edf48fc6c5259707f9d833f22171b4f63493`; the A0 stream weight SHA-256 is
`447c0554daa6bebdcf6fc69b2651b25b29cc69e003da4e6ff56f9a2488f403cf`. PyTorch Metal
runs its convolutions and falls back to CPU only for the small unsupported AvgPool3D
operation. A real end-to-end smoke passed source/weight verification, bit-exact Apple
initialization, forward/backward, checkpoint save/reload, and test isolation. The model
has 2,489,393 parameters, including a 1,533,543-parameter MoViNet backbone. A ten-batch
unfrozen benchmark passed; batch 4 was selected because batch 8 doubled work without a
throughput gain. Benchmark subset scores are not accuracy evidence.

The first detached launch was terminated by the execution shell before Python started;
PID `28083` and the empty log are not evidence of a run. The complete signer-disjoint
experiment was immediately relaunched locally on Apple Metal; its output directory is
`artifacts/models/stage1_v17_sign_movinet_stream_fusion/`. The fixed protocol is five
frozen-backbone warm-up epochs plus at most 35 end-to-end epochs, batch size 4, patience
8. The untouched Apple validation checkpoint is saved at epoch 0 before training, so a
degraded fusion cannot become the reported best model. The official Citizen test stays
sealed. The full epoch-0 audit reproduced 93.12% top-1 on all 378 validation clips.
Completed full-validation rows are:

| Epoch/phase | Loss | Fused top-1 | Visual top-1 | Mean gate | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 / protected Apple baseline | n/a | 93.12% | 1.06% | 0.124 | n/a |
| 1 / warm-up | 2.1673 | 93.12% | 1.32% | 0.007 | 381.7 |
| 2 / warm-up | 2.0973 | 93.12% | 4.23% | 0.012 | 384.9 |
| 3 / warm-up | 1.9617 | 93.12% | 7.41% | 0.007 | 393.5 |
| 4 / warm-up | 1.8844 | 93.12% | 10.58% | 0.003 | 405.8 |
| 5 / warm-up | 1.8277 | 93.12% | 13.76% | 0.009 | 414.6 |
| 6 / joint fine-tune | 1.8794 | 93.12% | 8.20% | 0.010 | 812.7 |
| 7 / joint fine-tune | 1.8626 | 93.12% | 9.52% | 0.010 | 814.6 |
| 8 / joint fine-tune | 1.8511 | 93.12% | 6.88% | 0.009 | 806.4 |
| 9 / joint fine-tune | 1.8692 | 93.12% | 7.14% | 0.010 | 814.2 |
| 10 / joint fine-tune | 1.8654 | 93.12% | 7.41% | 0.011 | 825.4 |
| 11 / joint fine-tune | 1.8687 | 93.12% | 7.41% | 0.011 | 852.6 |
| 12 / joint fine-tune | 1.8566 | 93.12% | 7.67% | 0.011 | 837.0 |
| 13 / joint fine-tune | 1.8422 | 93.12% | 7.41% | 0.012 | 792.8 |

Epoch 13 was the eighth consecutive non-improving joint epoch, so patience 8 stopped
the controlled run after 13 total epochs. The protected epoch-0 Apple checkpoint
remains best: 93.12% fused top-1, 99.47% top-5, and 92.53% macro F1 on all 378 official
validation clips. An independent strict checkpoint load reproduced those three fused
metrics exactly. The auxiliary visual path at epoch 0 is randomly initialized and its
standalone scores/gate vary under nondeterministic MPS kernels; because the saved
residual classifier is exactly zero, this cannot change the protected fused logits.
The full history has 13 sequential rows, every row covers 378 validation samples,
`last.pt` records `joint_stale: 8`, and both result and checkpoint metadata record
`test_evaluated: false`. No fused epoch improved on Apple-only, while the best RGB-only
validation top-1 was 13.76% during warm-up and ended at 7.41%. This completes and
rejects this exact three-stream MoViNet-A0 fusion challenger; it does not displace the
frozen Apple Vision plus v17 Squeezeformer selection. The completed local Python
process remained idle after all final artifacts were closed and was terminated normally
to release host resources; no training or result file was open when it was stopped.

The authenticated Kaggle CLI was rechecked after completion to resolve where the epoch
logs originated. `francisbatiancela/slt-v17-movinet-end-to-end` is `ERROR`, and its
server log ends after 57 seconds: no `nvidia-smi`, TensorFlow reported zero GPUs, CUDA
initialization failed with error 303, and the fail-closed launcher stopped before any
epoch. The separate `slt-v17-kaggle-gpu-probe` is also `ERROR`; it saw no NVIDIA device,
CPU-only PyTorch 2.10, and zero TensorFlow GPUs. Therefore none of epochs 1–13 ran on
Kaggle; they came from the local MPS process. No Kaggle model metrics exist. The latest
Kaggle source/metadata and retained partial working output were downloaded for audit to
`artifacts/generated/kaggle_movinet_v17/kernel_pulled/` and `kernel_output/` (about
32 MiB); these are transport/debug artifacts, not model evidence.
