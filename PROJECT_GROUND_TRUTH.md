# SLT Project Ground Truth

**Last updated:** 2026-09-22 PHT (+0800, Asia/Manila)

This file is the current state of the project, and it is the only file you must read
before changing the pipeline. It states what is true and what binds you now.

Everything that has ever happened is in `docs/ground_truth/` — 354 dated entries split
by topic. **Do not read that archive start-to-finish.** `rg <term> docs/ground_truth/`
when you need history; the map is at the bottom of this file.

After any material decision, experiment, dataset action, or validation result: append an
entry to the matching `docs/ground_truth/<topic>/log.md`, and update the section here
that it changes. If it changes nothing here, it does not belong here.

---

## Product goal

Build a fully offline, iOS-first ASL translator that retains the highest practical
accuracy while remaining viable on low-end and medium-spec iPhones. Accuracy and
generalization come first; distillation and aggressive compression are deferred until
the best accurate baseline exists.

The isolated-sign vocabulary is **100 signs**, locked. Splits must be signer-disjoint.
Target minimum is 20 training and 5 test signers, ideally 5 clips per person per sign.
The old seven-person dataset is not acceptable evidence of generalization.

## Pipeline state

**Reel context adaptation COMPLETE 2026-09-22:** 14epochs/0.04min fitting, selectedepoch0. No promotion. Report `artifacts/reports/reel_context_adapt_v17_20260922/REPORT.md`. Proposalclassifier/verifierfusion heads; frozenencoders/BIO/acceptance. Next inspect retention, isolatedvalidation and confirmation before deployment.



**Reel rejection replay COMPLETE 2026-09-22 — no live change:**
59clip150sign savedcalibration, exactcurrentdecision reproduction. Onefixedverifier-aware
rescue policy (finalscore>=.8/finalmargin>=.08; onlyconfidenceveto bypass, otherguardskept)
recovers4correct butadds3insertions:82->86correct,6->9I,49.33->48.67%WER,retains82/82.
FailsnoextraIguard; confirmation not evaluated. Unfilteredverifier102correct but37I/
56.67%WER. No clean gate-onlyfix demonstrated; not proof allgatepolicies fail.
Report `artifacts/reports/reel_rejection_replay_v17_20260922/REPORT.md`. No inference,
training or deployment. Next discuss continuous-sign identity/context adaptation,
keep frozenBIO and currentlive acceptance until a validated improvement exists.


**BIO-preserving final planned adaptation COMPLETE 2026-09-22 — frozen retained:**
12epochs,18.03min. Epoch8improvedcalibrationWER49.33->47.33%,82->84correct but
retained74/82; failed98%retentionguard. Selectedepoch0frozen; confirmation43.42%WER,
46/76correct. best_trained.pth preservesactualtrainedcandidate; no promotion.
Report `artifacts/reports/pretrained_bio_final_v17_20260922/REPORT.md`.
Investigationreview: Citizenprobe96% rests on57clips/5signers,not19pergloss;232/232
memorization evidence belongs to olderStage2CTC,not currentReel. Frozenrandomedge
precision doesn't measure originalBIO. Boundarynotbottleneck claim tooabsolute.
Next proposed focus: rejectionattribution and verification-aware decisionpolicy with
frozenBIO/Reel, using existingcalibration and checking extraoutputs/retainedsigns,
then actualruntime scheduling. No newtraining or policychange launched.


**Local BIO calibration COMPLETE 2026-09-22 — no improvement/no promotion:**
59calibration/150signs +30confirmation/76signs from371local candidates;282unused.
Reserve89clips from future boundary fitting. Familiar3signers/6templates; exacthash
checks pass, expanded72 and boundaryfit excluded. Session/near-duplicate independence
unproven. Frozen/adapted BIO x min3/min5 calibrationWER49.33/56.00/53.33/57.33percent.
No challenger qualified; frozenmin3 selected before confirmation. Confirmationbaseline
43.42%WER,46/76correct,3I,8/30exact; no claimed head-to-head improvement. Runtime11.15min.
No training/defaultchange. Report: `artifacts/reports/boundary_local_calibration_v17_20260922/REPORT.md`.
Next discuss BIO-preserving adaptation; duration filtering alone loses correct signs.


**Boundary adaptation regression isolated 2026-09-22 — no promotion:**
Controlled72video/186sign swaps reproduce frozenBIO127correct/39.78%WER/15I exactly.
Firstadapted+originalBIO=130correct/40.32%WER/19I,
retains124/127; augmented+BIO=124/42.47%/17I,
retains120/127. CNN/norm/BIO tensors unchanged; major loss
comes from replacing readout/decoder, not wholesale representation damage. Offline evaluator
now permits originalBIO independently of checkpoint; defaults unchanged.42614target rows and
mask/clock/cache audit pass;23focusedtests. Existing calibration has only2complete in-vocabulary
videos/4signs: BIO0correct, adaptededges1. It cannot support robust recognition selection;
no expanded-set checkpoint selection, no newtraining/deployment/distillation. Next resolve
complete calibration scoring and preserve originalBIO in future adaptation comparisons.
Local candidate inventory now confirms371clips/1006signs outside boundaryfit and expanded72,
from3signers but only6phrase templates. The2clip limit belongs to the old loader;
new versioned59/30calibration/confirmation partitions are complete above. No blanket re-admission.
Report: `artifacts/reports/boundary_adaptation_diagnostic_v17_20260922/REPORT.md`.


**Augmented boundary run COMPLETE / expanded evaluation audited 2026-09-22:**
Augmented seed17621 completed16epochs,selected8:37.17min total (11.29mincache,
24.95mintraining/validation),calBCE.652261,val.818841. Old12videos7correct vs9first
fine-tune,retains2of4/1gapcommit; no promotion. Recipe_v2 and weights preserved.
User-expanded72video/186sign set confirmed:12ASLLRP/24 +60local_signer_02/162;
no boundary train/calibration video-hash overlap, all WER/counts recompute. Reused
familiar-signer development,not unseen-signer. Local interval/gap labels unavailable.
Original expanded membership lists12only; saved ASLLRPcandidate counts incorrectly
includeall72 (246/261/287 instead19/21/18), without changing WER. Original files preserved.
Full72membership and corrections: boundary_expanded_eval_v17_20260922/tcn_comparison/.

SamefrozenReel conditionalwholevideo comparison: untouchedpretrained127/186correct,
39.78%WER/17exact; firstfine-tune96/54.30%/5exact; augmented100/53.76%/2exact.
All4nativeTCNs freshly evaluated: skeletonseed17621=89correct/61.83%WER,
seed17622=78/66.13%; geometryseed17621=84/62.90%,seed17622=99/59.68%.
StrongestTCN99correct/7exact/24insertions vsaugmented100/2exact/14insertions.
TCN uses native200msfuture/unscoredtail; pretrained500ms/partialEOF. No claim of a
controlled architecture-only comparison, defaultReel improvement onlocal60, or live/iPhone
readiness. Comparison420.37s; originalASLLRP TCN5/6/7/8counts reproduced exactly.
Audit/result/report: `artifacts/reports/boundary_expanded_eval_v17_20260922/tcn_comparison/REPORT.md`.
No newtraining,distillation or defaultchanges;seed2staysstopped. Next discuss frozenBIO
versus adaptedreadout/backbone diagnostic before specifying another fine-tune.


**Pretrained ASL boundary adaptation reviewed 2026-09-22 — no promotion:**
Seed17621 completed40epochs/78.15min, selectedepoch7; seed17622 interrupted by user
(KeyboardInterrupt), not restarted. CalibrationBCE .666061, validationBCE .830769.
Same12reuseddevvideos/24signs: frozenboundedBIO6correct/75%WER/retains3of4;
adapted9correct/62.5%WER/retains2of4,1guardedgapcommit,8EOFpartialcommits,1of12exact.
Bestcheckpoint retained despite late overfit. Patience8 replay selects sameepoch7,
stops15 and saves50.59min;40epochfloor was unnecessary. No nextfine-tune launched.
CachedCNN features omit pose augmentation; LR already ReduceLROnPlateau, not flat.
CNN is temporal, so per-frameprojectionreuse is invalid. Actual train subset758poses,
3019events; total1121poses/4310events includes calibration/validation.
SynchronizedMPS batch1 normalization+model median16.50ms/p9521.41ms; concurrentfrozenReel
p95202.05ms, excludesMediaPipefrontend and intrinsic0.5sfuture. Not verifiedreal-time.
No demonstrated2xFP16cachegain; warmcache confounds syncmicrobenchmark. Report:
`artifacts/reports/pretrained_boundary_review_v17_20260922/REVIEW.md` and measurements.json.
Original recipe/hash, checkpoint and evaluation preserved in pretrained_boundary_finetune
report/model directories. DefaultReel and genericphrasegate unchanged; no protectedtest.
Augmented follow-up is complete and reviewed above; whole-video retention/false-output gates
remain required before deployment or teacher distillation.


**Pretrained pose boundary transfer verified / preparation fully validated 2026-09-22:**
User requested continued work and pretrained fine-tuning. Author DGS segmentation weights
(sign-language-processing/segmentation,commit22ca3a6f63b6f031bfb1c0d717fcb259143ba7db)
now local:11.49MB,5.734Mparameters,147strictlyloadedtensors,CPU/MPSforward and synthetic
MPSgradients pass,zerooptimizersteps. Same12reusedASLLRPdev/24signs with actualMediaPipe
and frozenReel:11correct,3S/10D/0I,54.17%WER,retains4/4baseline,2/12exact. Offline only:
full-sequence attention and whole-clipnormalization;17.52sfrontend+3.25sboundary for12.68svideo,
Reeladditional. No promotion or real-timeclaim. Gate-only savedcounterfactual is mixed,
not a safe universalfix. Report: `artifacts/reports/pose_boundary_transfer_v17_20260922/REPORT.md`.
Native MediaPipe preparation and full structural validation completed:1121/1121records,
889train/232validation,4310intervals,190797frames. No missing records,hash/pose/clock
errors or recorded signer/parent role overlaps. Final validation receipt passed; this
is input validation,not a new model-accuracy result. Raw50jointposes remain unnormalized.
Input validation preceded the completed/interrupted dedicated adaptation reviewed above.
Proposed next recipe: preserve pretrained representation,learn masked start/end targets,
then gently fine-tune with the same bounded-future window at training and inference.
Benchmark the complete bounded path before choosing the run budget; keep Reel frozen
and measure whole-video retained signs,false outputs and end-to-output delay.
Prepared training_ready=false pending that recipe; no uncertified gaps become BIO
background. Default Reel and generic phrase gate unchanged. Full evidence:
`artifacts/reports/pose_boundary_transfer_v17_20260922/validation.json`.


**ASL temporal boundary comparison completed and reviewed 2026-09-22:**
Implementation and bounded two-seed skeleton/hand-geometry experiment are complete;
no promotion. Same 12 reused development videos / 24 known signs: Reel 4 correct,
83.33% WER; skeleton 5–6 correct, 75–79.17% WER, retains 2/4 baseline correct;
hand geometry 7–8 correct, 66.67–70.83% WER, retains 3/4. All learned arms have
0/12 exact videos; geometry still deletes 16/24 signs. Reviewed timing +100ms gives
12 correct / 50% WER, verifier 16/24: identity/commit errors also remain.
Baseline and learned arms have zero wholly-gap-contained commits, so this replay
cannot demonstrate reduced transition false positives. Successful-word median delay
0.58–0.65s; boundary CPU p95 61–80ms exceeds the 20Hz frame budget. No independent
low-motion/hold/repeat/OOV or iPhone evidence. Eight code hashes, dedicated recipe hash
and four checkpoint hashes verified; previous 87 focused tests passed. Default Reel
and generic phrase training gate unchanged. Persistent plan:
`docs/superpowers/plans/2026-09-22-asl-temporal-boundary.md`.
Results: `artifacts/reports/asl_temporal_boundary_v17_20260922/REPORT.md`.
Next safe action: use saved per-interval proposal/verifier evidence to isolate missed
boundaries versus identity versus commit rejection before specifying a bounded contextual
identity intervention. No further training launched; no architecture replacement justified.

**Step-back discussion 2026-09-22:** User prefers combined eligible data across signers;
multiple models acceptable if fast and accurate. Mixing alone has not established
generalization (existing combined-run results below). Proposed next direction: ASL
continuous boundary learning plus a controlled contextual-identity check using Reel as
reference; full-stream retained-sign/false-emission/latency evaluation required. This
remains discussion, not a guaranteed fix or new training/runtime authorization. Details:
latest `docs/ground_truth/live-streaming/log.md` synthesis entry.

**Zhao component review COMPLETE 2026-09-22:** Author-owned ASL-Handshape and
SegmentASLTransformer repositories contain weights. Segment checkpoint strictly loads
34 tensors and runs on MPS (synthetic CPU/MPS max difference 5.722e-6); no accuracy test.
Active code ignores handshape inputs and requires external preseg predictions with no
producer in the inspected tree. Handshape checkpoint has 88 outputs, mapping unresolved.
Not a verified complete MHB release or live candidate. Sources/data register and proposed
ASL temporal-boundary next step: zhao_mhb_availability_v17_20260922/REPORT.md.
No dataset acquisition, broad training or default live changes.

**Zuo online transfer COMPLETE / no promotion 2026-09-22:** Recovered author's online
PHOENIXcheckpoint from replacementDrive afteroriginal404links. Strict938tensorload; realHRNet
MPSfrontend. Directblankhead on17core+3gapcentres rejects3/3gaps, falseconditionalcommits1->0,
but correctcorecommits5->0 (4/17coresnonblank).88.86s diagnostic, notliveWER.16frame25Hz
centreprotocol differsfromSHuBERTreadout; letterbox/poseprovenance deviations documented.
Dataregister/weights/verification: zuo_twostream_probe_v17_20260922/REPORT.md. No dataset
payload or training. Zhao component availability reviewed above; next prepare ASL boundary adaptation recipe.

**Stage 3 ASL-order renderer IS NOW THE LIVE DEFAULT 2026-09-22 (user instruction):**
`artifacts/models/stage3_v17_asl_order_v1` replaced the legacy checkpoint as
`DEFAULT_STAGE3_TINY`. Input format is read from `stage3_input_contract.json` beside the
weights, so `--stage3-encoding auto` resolves evidence/plain per checkpoint and a caller
that never heard of the flag (scripts/live_continuous_v17.py) cannot be fed the wrong
format. Rollback is `--stage3-checkpoint artifacts/models/stage3_v17_t5_efficient_tiny_locked100_v1`,
which auto-resolves to plain. Final: test BLEU 92.74 vs 43.58, exact 0.905 vs 0.260,
noise suppressed 0.992 vs 0.589, negation 1.000 vs 0.990, genuine glosses dropped 0.21%
vs 3.09%, words invented from no gloss 1/1261 vs 57/1261. Reordering 89.43 vs 7.53;
SVO regression guard 96.42 vs 55.38. 26ms/sentence CPU.
Running it live exposed two defects that held-out BLEU did not, both now fixed and in the
probe set. (1) Session 20260922_122517_239185: `I SICK MY HUNGRY` at 0.97/0.89/0.91/0.52
rendered "I am sick, and my family is hungry" — MY scored 0.91, so no threshold could
catch it; a stranded possessive with no noun made the model invent one. Corpus now has a
stranded-determiner class whose confidence comes from the GENUINE band, so omission is not
learnable as a function of score; possessive-before-noun is untouched. (2) `WHO YOU` ->
"Who do you see?" — verbless wh+noun-phrase questions were absent although they dominate
real sessions; the model invented a verb instead of supplying the missing copula.
STILL NOT VALIDATED BY A FLUENT SIGNER. BLEU measures agreement with model-written English
on rule-generated sequences. Two long garbled multi-clause buffers still fail. The mobile
naturalizer manifest `active/v17/stage3_mobile_naturalizer_manifest_v17.json` is deliberately
UNCHANGED and still states "never delete, replace, reorder"; it governs the separate mobile
bounded renderer, which did not change, so the desktop live default now diverges from it.
Report artifacts/reports/stage3_v17_asl_order_v1/README.md.

**Stage 3 ASL-order retrain COMPLETE / no promotion 2026-09-22 (superseded by the entry above):** User reported live SVO
errors and rendered noise glosses. Deployed renderer proven to be a monotone function-word
inserter: 11,745 of 11,840 matchable training rows preserve gloss order, only 95 reorder,
and only 742 of 15,843 rows are inside the locked 100. It drops 3.03% of genuine glosses,
keeps 57.50% of noise, and 231/958 outputs fail a gloss-to-word content check (20 lose the
subject — the reported `I TIRED` class). User authorized reordering plus low-confidence
omission, rule-generated ASL syntax with model-written English, per-gloss confidence input,
and BLEU as the gate. New candidate `stage3_v17_asl_order_v1` trained fresh from the
grammar-correction base on 13,047 generated rows (DeepSeek V4 Flash via OpenRouter, $0.109,
all 50 rejects were invented content). Test opened once: BLEU 92.26 vs deployed 41.10,
exact 0.893 vs 0.221, noise suppressed 0.992 vs 0.550, negation 1.000 vs 0.986, genuine
glosses dropped 0.28% vs 3.03%, content-check failures 5/1367 vs 364/1367. Reordering slice
89.89 vs 5.72; SVO regression guard 99.07 vs 53.98, so working cases did not regress.
26.1ms per sentence on CPU. Fixes the 2026-09-16 lost-subject defect recorded below:
deployed `I`->"Is that?", `I SICK`->"Is I sick?"; retrained "I.", "I am sick." BLEU measures
agreement with model-written English on rule-generated sequences, NOT human-judged
translation quality; the corpus is not genuine ASL and no fluent signer reviewed it.
Live opt-in only, default unchanged: `--stage3-encoding {plain,evidence}` with
`--stage3-checkpoint artifacts/models/stage3_v17_asl_order_v1`. Reel now carries per-gloss
commit scores to Stage 3. No recognition model, checkpoint or threshold changed; no
protected split accessed. Report artifacts/reports/stage3_v17_asl_order_v1/README.md.
Next: user judges live signing, then decide promotion and whether a fluent reviewer should
validate a sample of the generated English.

**SHuBERT paired decision probe COMPLETE / no promotion 2026-09-22:** Frozen45videoMPS
features,247fixedintervals,train-onlycalibratedridge. Rejects2/3held-outgaps includingfalseWORK;
conditionalfalsegapcommits1->0, but correcttight-corecommits5->2 (threeFRIEND lost),100ms9->6,
250ms5->2. Landmark/DINO-bodygatesreject0/3gaps. No low-motion-positivecases or fullstreamWER;
fullclipnoncausalfeatures and tinyreuseddevelopmentset. Source-clockroundingbug fixedbeforefit;
allintervalcounts/priorcontrols verified. Report shubert_decision_probe_v17_20260922/REPORT.md.
Zuo comparison completed above; do notdeploy/tuneon3gaps. No broadertraining/dataacquisition.

**SHuBERT MPS probe COMPLETE 2026-09-22:** Official encoder/face/hand weights strictly
loaded; approved1.368sASLLRPclip ->41x768finitefeatures. Published signer crop restored
face detections0/41->41/41; do not omit frontend. Encoder0.456s, frontend/DINO/crop separately
~9.724s including loads; not live throughput or accuracy. CPU/MPS encoder maxdelta1.824e-5.
No transition head/WER result or live promotion. Data sources/overlap/local YouTube relation
saved in shubert_probe_v17_20260922/DATASETS.md; dataset acquisition/training deferred.
Frozen-feature comparison completed above; broader SHuBERT adaptation remains deferred.

**User priority 2026-09-22 — compare before training:** SHuBERT evaluation, then other shortlisted
pretrained approaches (Zuo/TwoStream; Zhao subject to weights availability). Retain Renz's
BSL/PHOENIX I3D-feature/boundary dataset for later joint experiment design; archive not
downloaded/admitted. Do not jump to combined training before reviewing comparisons.
Details and dataset links saved in continuous_recovery_v17_20260922/PLAN.md.

**Renz buffered live mode available:** app_shell_v17.py --renz-buffered --device mps.
MPS convolutions with CPU 3D-pooling fallback; 60 focused tests and real-video smoke passed.
Approximately29seconds per4seconds of repeated-image input in throughput smoke; slow
experimental preview with visible delay/backlog skips, not real-time. Default Reel unchanged.

**Renz pretrained offline transfer COMPLETE / no promotion 2026-09-22:** OfficialBSLCorpus I3D+MS-TCN strictlyloaded,12approvedASLLRPval/24signs. Rawvalidwindows100%WER; edgepadding79.17%; edgepadding+publishedmidpointextension62.50%(10tokens,2/12exact). FinalsegmentbestIoU>=.5covers13/17eligiblecores; WORK WHERE->WORK WORK. Conditionalofflinecomposition, notlive/all-datasetWER or demonstrateddefaultReelimprovement. Aspectpreservingfrontend differsfromupstreamstretch; noncausal. Checkspassed; no training/livechanges. Report artifacts/reports/renz_reel_v17_20260922/REPORT.md.

**Fast Reel decision probe FAILED 2026-09-22:** Frozen recognizers + four temporal landmark
bins/scores, linear ridge gate. Fit48cores/14gaps; calibration11cores/2gaps; validation17cores/
3gaps (three context representations per core). Rejects2/2calibration gaps but0/3validation;
falseWORK remains. Correctconditionalcommits5/9/5 ->5/8/4. Confidencebaseline also missesWORK.
No promotion/livechanges. Small linear interval probe, not full learned visual head or WER.
Next audit existing training hard negatives and true confusable signs; preserve held-out gaps.
Report: artifacts/reports/reel_decision_probe_v17_20260922/REPORT.md.

**Matched Reel comparison COMPLETE 2026-09-22:** 12approveddevelopmentvideos,17matchedcores:
verifier11/17tight,11/17+100ms,7/17+250ms.1/3checkedgaps foolsbothmodels(WORK.594/.517)
andpassesconditionalcommit. Sixcorrectcoreverifierpredictionsblocked; no singlethresholdfix.
ActualReel20Hzscheduleverified afterinitialdiagnosticcorrection. No fullstreamWER or
exactuserattempt replay. Report reel_matched_windows_v17_20260922. Recorder finalization/
appnativeclosefix added separately; no model/threshold changes.

**Reel direction researched 2026-09-22:** Retain appUX/current signrecognition baseline;
proposed continuous spotter must remove wrist-only candidate/trim dependence and learn
complete-sign vs non-emission on actual confusing transitions. Priorbackground/window
and timedanchor failures remain binding evidence; no claim a newhead alone fixes this.
First commoninput/replayable traces, then controlled hard-negative/hold study. No change
launched. See reel_streaming_research_v17_20260922/REPORT.md.

**Live familiar failure investigated 2026-09-22:** User reports I GO omissions. LatestReel
history confirms I/GO commits twice; candidate sessions emit neither. Different signing
runs, no pairedWER. I/GO exist in isolated supervision; continuous training contains I
onlyinPLEASE HELP I,GOonlyinTOMORROW SCHOOL GO,zeroI GO pairs. Candidate olderStage1differs
fromworkingReel. NoliveLM. Do not treat19.14% six-templatevalidation as arbitrarylive success.
Candidate recorded no video/features/logits; latestReelMP4 lacksmoovatom, replayblocked.
Encoder/preprocessing/head attribution unresolved. Keepcandidate experimental, capture
commoninput/per-windowevidence before anothertrainingrun. Report live_familiar_diagnosis_v17_20260922.

**Older matched evaluation / familiar-signer preparation 2026-09-22:** Older aligned-grounded
pipeline evaluated on current211approvedphrases:39.04%WER (199local38.73%,12ASLLRP45.83%),
55/211exact, versus latest adapted50.98–53.65%. Development benchmark, not protected test;
no live promotion. User explicitly authorizes local phrase signer sharing: new versioned
split moves139signer02clips into training, keeps60different localclips held out. Citizen and
other official roles unchanged. Dedicated local_familiar_ctc recipe compares old frozen-base
control/familiar heads (2seeds,6epochs,LR1e-4), common60local+12ASLLRP+1663singlevalidation.
This is familiar-signer reused development evaluation; old211benchmark becomes historical
once139clips train. Preparation/review passed; detached PID78465 launched Sep22PHT (23:31:43UTC Sep21),
now COMPLETE and reviewed. Same60localclips: control35.80%WER(epoch0both), familiar19.14/22.22%
(epoch6/5); exact20→33/27of60. Checkpoint hashes, selector, initial equality and coverage verified.
Familiar-signer six-template reused development result; no live promotion or arbitrary-signing
claim. ASLLRP12knownWER33.33%both; O5S5still77–79%. Scoring removesOTHER unlike older standalone
211evaluation; do not compare pooledmetrics across conventions. See local_familiar_signer_v17_20260922/
RESULTS_REVIEW.md. Next: candidate streaming parity/live stress evaluation, then independent
weak gloss-language-prior decoding comparison; actual-context augmentation separate.
Candidate live opt-in now available through scripts/app_shell_v17.py
--familiar-ctc-checkpoint artifacts/models/local_familiar_ctc_v17_20260922/seed_17521_familiar.pth.
95focusedtests pass plus MPS smoke; camera normalization remains an unvalidated domain difference.
Literal greedy transcript, no language completion; events under continuous_live_v17 (not app HISTORY).
Offline decoder comparison complete: local60greedy19.14/22.22%, beam17.90/22.84%,
uniform-prior16.67/21.60%, bigram16.05/20.99%. Learned pairs add only0.62pp per seed
beyond uniform length bias; no live LM promotion. Report familiar_decoder_v17_20260922.

**Timed-anchor head comparison COMPLETE / no promotion 2026-09-22:** Matched56clip diagnostic found
both contextual identity and CTC event errors; adapted eligible-core scores12/17and9/17
validation butwholephraseexact0/12both (limited contextual evidence, not isolatedaccuracy).
New recipe active/v17/transition_anchors_manifest_20260922.json pins samecombineddata,
59traincore/27internalgap intervals, originalroles, selectedadaptedStage1frozen. Two seeds,
CTConlycontrol vsCTC+0.25timedintervalCE,3epochs/head, cachedrichevidence forspeed.
Independentreview/focusedtest/MPSpreflightpassed;7367windows/seed finite, cache/directdelta0,
headgradientsfinite/baseabsent, zerooptimizersteps. DetachedPID40294 launched22:57:08UTC
Sep21 (Sep22PHT), now complete and verified. Reports/models transition_anchors_v17_20260922.
PhraseWERcontrol50.98/52.41%, anchored50.98/52.41%; exact29/24vs29/22of211.
No improvement over matchedcontrol; seed17421bothselectepoch0. No promotion/extraepochs.
See REVIEW.md; revisit intervention/data support before another training run.
Blankgap supervision is a bounded alignmenthypothesis; notallmovementisblank. No livechanges.

**Five-step recovery STARTED 2026-09-22:** User approved beginning diagnosis, combined-data
preparation, evidence-based Stage1 adaptation, continuous recognition, then stable/fast
display; useful parallel work explicitly authorized. Persistent checklist:
`artifacts/reports/continuous_recovery_v17_20260922/PLAN.md` (retain all five steps).
Fresh verification:6421records/7367windows and6158raw paths hashed; phrase train covers44
known labels, single-sign train100. No proven manifest defect or blanket re-extraction need.
Twelve unchanged legacy-live-CTC full/core replays on3train+1validation sources: train
controls exact; validation FRIEND MAYBE→FRIEND STOP, cores→FAMILY/STOP. Current Reel
oracle MAYBE top label correct but verifier rejects low score/margin. Different models/
frontends: not a head-only comparison or new benchmark. Details/provenance in REPORT.md.
**Bounded comparison COMPLETE / reviewed 2026-09-22:** two seeds17421/17422, frozen/adapted Stage1,
12epochs/arm, current Reel proposal initialization, combined supervised data. Dedicated
recipe active/v17/combined_frozen_joint_manifest_20260922.json (SHA b077e6939c6dc71c581cea2ee669348e2b4cc1d1cedef50eb7f8a9fd85c4ee64).
MPS preflight passed all6421records/7367windows per arm and worst-case gradients with zero
optimizer steps. First preflight aborted on float16/float32 MPS mismatch; fixed once at
record loading, four focused tests pass, failure evidence retained. Detached caffeinate
PID98647 launched2026-09-21T17:45:51UTC; all four12epoch runs now complete and hashes verified.
Reports/checkpoints: artifacts/{reports,models}/combined_frozen_joint_v17_20260922.
Local phrase validationWER frozen60.89/66.85%, adapted50.84/53.26%; adapted training
WER0.48/1.74%. Citizen Stage1 validation95.24%→94.71/94.18%. No promotion: large
generalization gap remains. See REVIEW.md; next per-record identity/alignment diagnostics. Steps3–4in progress, step5still pending;
no protected test, acquisition or live default edits. Original generic phrase gatefalse.

**Product clarification / discussion 2026-09-22:** User wants the locked100 recognized
individually or naturally connected, without pauses/resets between signs or an assumed
phrase. English rendering may use additional words without inventing meaning. Stable
visible words with roughly0.5–1s delay are acceptable; faster Reel-like output preferred.
User reports low-wrist-motion misses in Reel and transition false positives in Reel
continuous; latest separate Stage2 trial also unsatisfactory. Initial discussion-only
scope was superseded by the five-step start authorization above. See latest
`docs/ground_truth/live-streaming/log.md` entry for saved-session evidence and limits.

**Supplement finalization 2026-09-22 — current admission:** User permits shared/missing
signer IDs; preserve existing roles, no signer-based exclusion, label evaluations honestly.
Five pinned supplemental manifests:5,927records=4,264train/1,663val, under
`artifacts/reports/supplement_finalization_v17_20260922/REPORT.md`.
Citizen1475/378,SemLex1388/953,ASLLRPsegmented1116/254,reviewedSTEM90/21,O5S5cores195/57.
Recovered3tails; excluded1knownCitizenreject,6missingSemLexfeatures,25exactvalidation
copiesoftrainvideos,4too-shortO5S5cachedcores. No signer exclusions;SemLex951validation
rows sharetrainIDs. All5,927records/5,937windows pass frozenStage1MPSfiniteforward and
pinnedfilechecks. Plus494baseline=6,421records(4,547train/1,874val),notindependentphrases.
**Combined manifest created:** `data/local/combined_dataset_v17_20260922/manifest.json`.
All6421records/7367windows loaded and verified;4547train/1874validation, including431localphrases.
Builder/loader: `scripts/build_combined_dataset_v17.py`; report:
`artifacts/reports/combined_dataset_v17_20260922/REPORT.md`.
Dedicated combined frozen/adapted recipe launched; see recovery status above.
Preserve roles, three feature representations, and139sharedASLLRPparent records.
This supersedes prior1Citizenextra/3tailunresolvedcounts andglobal signer-compatibility
blocker for these supplements. Other lexical/representation restrictions remain.


**Full source reconciliation 2026-09-22 — supersedes narrow inventory interpretation:**
494 is approved whole-phrase/subspan baseline size, NOT all usable data. Existing exact
ASLLRPsegmented1370(1116train/254val),O5S5positives256(199/57),reviewedSTEM111(90/21)
were omitted from that run. Fresh ASLLRP/STEMloaderchecks:1ASLLRP+2STEMtrain tailsblocked;
remaining supplementalrecords1734=1402train/332val, not deduplicated independent videos.
Citizen/SemLex4220currentarchives retain1extraCitizen provenance discrepancy. Local
auxiliary16277(13381/2896,NOTsigner-disjoint) andASLLVD175officiallinkedclips exist with
separate schema/signer restrictions. Flores155(141completeOTHERcache),NCSLGR166(125strict
phrasecache) preserved, lexical identity links stillconditional. Motion-onlyHow2Sign1027,
YouTube128raw +separate1411keypointfiles,OpenASL3; no supervised-label inflation.
The baseline mixed local+ASLLRP in each epoch; it did NOT mix all established sources.
Authoritative scope/status/count reconciliation:
artifacts/reports/dataset_reconciliation_v17_20260922/REPORT.md and evidence.json.
Next: preserve prior source admissions, deduplicate/pin a compatible mixed-supervision
recipe (ASLLRP/O5S5/STEM/isolated plusphrases), resolve3tailcaches/1Citizenextra and global
splitcompatibility. No automatic newtrainingallowlist/promotion. No downloads/training.

**Clean baseline fit diagnosis 2026-09-21:** MPS saved-checkpoint evaluation shows strong
training fit but poor held-out generalization: seeds17321/17322 trainWER2.60/0.27%,
exact264/283(93.29%)/281/283(99.29%),versus validationWER50.27/49.38%,exact34/211/31/211.
Local trainingexact97.41/100% versus validation15.58/15.08%; within-source gap persists.
Stage2train44knownlabels,val26; GOOD/MORNING/BAD lack Stage2training targets (not absent
from Stage1vocabulary). Seen-label-only190validationclips stillWER46.24/45.47%;182clips
have training-seen target sequences but exact31/30. SCHOOLmatched1/19both, TOMORROW
SCHOOL GOexact0/19both. Strong overfit/generalization evidence, not proof of one cause.
All original validation predictions reproduced exactly; no training/test/acquisition.
Report: artifacts/reports/clean_phrase_baseline_v17_20260921/FIT_DIAGNOSIS.md.
Next: history-informed controlled generalization intervention; do not simply add epochs
or change the held-out split. Keep missing-label coverage separate from signer transfer.

**Clean phrase baseline COMPLETE / reviewed 2026-09-21:** User reported completion;
verified both saved checkpoints and all211predictions/18epochs per seed. MPS training
12.40seconds combined. Seeds17321/17322 selectedepochs8/11; overallknownWER50.27/49.38%,
exact34/211(16.11%)/31/211(14.69%),deletions119/114of561tokens,blank-only15/8clips.
LocalWER49.91/48.98%;ASLLRPcontiguousWER58.33%both(12clips). Weak clean-data baseline;
no promotion, no UNKNOWNrecall/test/live claim. Manifest/base/checkpoint hashes and
selection verified; Stage1unchanged. No further training performed for review.
Report: artifacts/reports/clean_phrase_baseline_v17_20260921/REVIEW.md.
Next: saved-checkpoint training-vs-validation diagnostics and phrase/class errors before
choosing fitting versus coverage changes. Launch notes below are historical; run is done.

**Clean phrase baseline launched on MPS 2026-09-21:** User authorized two-seed phrase-only
baseline and explicitly required MPS. Dedicated runner scripts/train_clean_phrase_baseline_v17.py
launched detached under caffeinate, PID71737, at15:51:26UTC. Completion not checked; do not poll.
Recipe-scoped active/v17/clean_phrase_baseline_manifest_20260921.json authorizes only this
runner (hash661eac1b318f54b80e13a45848be2d9225447479eefb33cc107227ee20eaab3d).
283train/211validation, Stage1 frozen, CTC-head seeds17321/17322,18epochs, batch16;
MPS encoder/head with CPU CTC loss because aten::_ctc_loss is unsupported on installed MPS.
Cross-device backward verified. MPS evidence preparation12.716s, finite forward CTC,
zero preflight optimizer steps;42focused tests plus MPS-gradient test pass.
No isolated/blank/rest/Flores/NCSLGR replay or test access; no unknown/live readiness claim.
General data manifestv2 remains blocked for older recipes. Notifications on completion/failure.
Reports: artifacts/reports/clean_phrase_baseline_v17_20260921/{PLAN.md,preflight.json,launch.json}.
Next session read status/results; do not infer completion from launch or compare unmatched
old scores as controlled results. Original CPU preparation was interrupted before caching
when user requested MPS; it performed no training.

**Canonical phrase manifest v2 / conservative salvage 2026-09-21:**
`active/v17/approved_phrase_manifest_20260921_v2.json` and
`data/local/approved_phrases_v17_20260921_v2/` supersede v1 as current defaults.
494 approved clips =283train/211validation: previous493 plus one verified WHEN→OTHER
training subspan from an excluded ASLLRP clip. Re-extracted11original frames; complete
adjoining official annotations; observed-hand fraction1.0;2CTC steps for2targets.
507 resolved candidate runs screened, only2gap-free; the other failed temporal CTC
feasibility. Gapped runs remain uncertified under this conservative salvage rule, not
proven unusable. Original full-clip exclusions preserved. No new independent video/signer.
41focused tests and actual494-file loader pass; v2 hashes and disjoint splits verified.
Training stays blocked; new asllrp_verified_span source must be handled by future recipe.
No acquisition/training/protected test use. This recovery pass stopped at the evidence
limit; report: `artifacts/reports/verified_phrase_salvage_v17_20260921/REPORT.md`.

**Inventory count 2026-09-21:** Approved phrases493 plus currently structurally valid
isolated archives4220 =4713available clips across current input roots (3146train/1567val).
This is not a fully approved next-run total. Citizen train1476 differs from historical
base provenance1475; resolve that extra file and pin isolated inputs before admission.
No event-core/window counts or sealed test clips added to this total.

**Prior phrase admission v1 (superseded by v2 above) 2026-09-21:**
`active/v17/approved_phrase_manifest_20260921.json` is the authoritative whole-phrase
allowlist. Approved symlink view: `data/local/approved_phrases_v17_20260921/{phrases,other}`.
493 clips admitted (282 train / 211 validation; 56 recovered), 1,365 excluded with reasons;
originals preserved. Local known 232/199, ASLLRP contiguous 44/12, ASLLRP OTHER 6/0.
Flores/NCSLGR whole sequences and ambiguous OTHER clips are excluded, not relabeled.
Both unified CTC run functions and Flores/motion run/preflight launchers verify admission
before model work; old-root overrides and supplements fail. Checkpoint/result writers
record manifest provenance. Run `venv/bin/python -m active.v17.approved_phrase_data_v17`.
**training_ready=false:** excluded-source-dependent metrics/recipe, auxiliary blank/rest
inputs and unseen-OOV evaluation need resolution before a new manifest permits training.
Do not bypass this gate or toggle the flag alone. Other historical trainers are not
retroactively rewritten; AGENTS.md requires the same contract for any future trainer.
40 focused tests pass; actual phrase loader reads all 493. No training or acquisition.
Report: `artifacts/reports/approved_phrase_manifest_v17_20260921/REPORT.md`.

**Tail recovery / exposure audit 2026-09-21:** User authorized continuation. Rebuilt
159 incomplete caches from hash-verified existing videos in 64.5 seconds under
`data/local/phrase_tail_recovery_v17_20260921/`; 1,699 unchanged archives are symlinked.
All 1,858 pass structure/timing/stride4-window8 CTC feasibility; all 125 NCSLGR alignments
pass. Originals, targets and signer roles unchanged; one rebuilt local-validation
window has insufficient hand detections, explicitly flagged (not blank truth).
37 focused tests pass. Active defaults remain on original roots; recovered files are
landmark-only. Exposure check: 259/276 OOV cores confirmed seen; remaining 17 cores /
13 identities unresolved, with Flores raw-label candidates for EYES/NEAR/BOTH.
Zero globally unseen identities certified. No acquisition, training or promotion.
Next: trusted event supervision/admission and evaluation contract; defer step 5.
Report: `artifacts/reports/phrase_tail_recovery_v17_20260921/REPORT.md`.

**Identity/OTHER audit 2026-09-21:** Existing-data continuation produced an annotation
ledger:9,104ASLLRP crop-associated occurrences =1,397known/1,844confirmed lexical OOV/
5,863unresolved under official identity links. Unresolved is not automatically OTHER.
Prepared248known+276OOV validation cores (97OOVidentities, one reused signer);259OOV
cores have ASLLRP-training identity exposure,17have unresolved other-source exposure.
Four saved Flores-arm core diagnostics emit exactlyOTHER on only1–40/276OOV cores;
211–253/276are blank-only. This is not an unseen-sign or live-stream benchmark.
No acquisition/training/promotion; sidecars do not change active training manifests.
Recover159incomplete caches and resolve supervision/evaluation contracts before step5.
Report: `artifacts/reports/annotation_identity_audit_v17_20260921/REPORT.md`.

**Existing-data repair 2026-09-21:** User stopped all new acquisition and authorized
steps1–2only. NCSLGR alignment now uses loaded source-frame counts; auxiliary padding
is ignored and length mismatches rejected. Shared unified-CTC phrase loader enforces
schema/vocabulary/ranges/full-coverage/CTC feasibility. Fresh checks:1,699/1,858pass;
159incomplete caches now block loading (no silent skipping or source rewrites).
All113complete NCSLGR alignments pass;34focused tests pass. No training or accuracy
measurement. Recommend mapping review(step3), then existing-data OOV evaluation(step4);
defer step5until cache recovery/admission is resolved. Report:
`artifacts/reports/phrase_contract_repair_v17_20260921/REPORT.md`.

**Public-data search 2026-09-21:** User requires no-account public downloads and
established mappings; no ASL-fluent reviewer available. DePaul's public NCSLGR ELAN
archive verified (870EAFs;684additional10-series EAFs beyond166local); video ZIP
public range endpoint works. This qualifies the prior DAI-account expansion blocker,
but conversion timing, signer roles and exact lexical mappings remain unadmitted.
Citizen train/val metadata identifies2,623nonlocked ASLLEXcodes as OOV candidates;
no protected test read. Repair/OOV-data work is proposed only. Evidence:
`artifacts/reports/public_dataset_search_v17_20260921/SEARCH.md`.

**Dataset annotation audit 2026-09-21:** Fresh checks cover1,858current phrase archives.
Twelve NCSLGR caches (8train/4validation) have1–3missing tail frames relative to alignment
metadata; the current auxiliary collator can label one padded step as blank in the8train
cases. Another147local/ASLLRP caches explicitly drop tails, unlike Flores admission.
No annotation extends beyond cached end in the12NCSLGR cases; review tail semantics
before exclusion. Current OTHER evaluation measures known-sign false rejection, not
independent unseen-OOV detection. Loader schema/range enforcement gaps reproduced;
checked archives themselves have compatible Apple schemas and no cross-role exact hash
overlap. Findings only, no pipeline/data changes. Report:
`artifacts/reports/dataset_annotation_audit_v17_20260921/REPORT.md`.

**Acquisition override — user paused YouTube-ASL downloads:** 1,411 manifest-matched
JSON files retained; no downloader process found. Frozen subset:
`artifacts/reports/free_continuous_asl_alternatives_20260921/acquired_manifest.csv`.
Use this subset for a controlled pretraining pilot before acquiring more. Earlier
download session/status claims below are historical and superseded. Motion pretraining
is a hypothesis, not established boundary supervision; MediaPipe-to-Apple feature
compatibility must be resolved before transfer.
**Motion pilot 2026-09-21:** Corrected-phrase paired run complete and reviewed. Local WER baseline→pretrained48.52→43.15% /50.19→44.44%; only1/2 overall gates pass (ASLLRP regression in seed17321). Seed17322 deletions52→130 despite lower WER; hold/repeat diagnostics do not improve. No promotion or further YouTube acquisition. Reports: `artifacts/reports/youtube_motion_pretrain_fixed_phrases_v17_20260921/REVIEW.md`.
**Flores OTHER 2026-09-21:** Higher-cap MPS retry completed;0/2 paired gates. ASLLRP/NCSLGR WER improves both seeds, but local exact42→14/200 and43→29/200; deletions68→204 and52→126; isolated accuracy falls (seed17322−2.29pp). No promotion. Checkpoints, matched data/weights and edit counts verified. Reports: `artifacts/reports/flores_other_mps_retry_v17_20260921/REVIEW.md`.
**Local phrase review 2026-09-21:** User confirms all 685 videos in PHRASES FIXED were checked against folder phrases. Exact-content audit: 95 originals removed, none relabeled/edited. Corrected admitted cache has 232 train / 200 validation (55 training removals; validation unchanged), signer/content disjoint. No GOOD_MORNING phrase training examples remain. Prior training-label conclusions remain provisional; corrected matched rerun uses user-reviewed retained clips. Audit: `artifacts/reports/local_phrases_fixed_audit_20260921/`; corrected cache: `data/local/stage2_v17_grounded_phrases_fixed_20260921`.
All1,411 files are structurally valid (168,531 frames);220 have actual/manifest count
disagreements and are quarantined, leaving1,191 count-consistent clips before quality
filtering. Direct Apple feature substitution remains inadmissible. The pilot uses a
separate46-node2D source adapter and transfers only the existing causal CTC temporal
blocks, keeping Apple Stage1 frozen. Two paired seeds compare the same supervised
recipe/decoder. Real-sequence WER/errors, conditional ASLLRP emission delay and isolated
CTC retention are measured; synthetic held/repeat probes are diagnostic only.
User requires audit/preparation in-session and **only training detached**, no polling.
The retired audit runner's CLI resume failed due to an active thread writer; do not
retry that mechanism. Training sends a macOS notification on completion/failure.

| Stage | Selected artifact | Status | Best measured result |
|---|---|---|---|
| **Stage 0** extractor | Apple Vision, `active/v17/extract_v17.py` | **Frozen** | Beat MediaPipe 93.12% vs 89.95% val top-1 |
| **Stage 1** isolated | v17 Squeezeformer | **Frozen — test consumed** | 87.57% top-1 / 98.64% top-5 / 87.39% macro F1 on 1,247 Citizen test clips |
| **Stage 2** continuous | general primary/specialist CTC selector | Accepted runtime | ASLLRP contiguous 9/24 edits; local phrases 6/259; ASLLRP contextual 43/254 |
| **Stage 2** repair | `stage2_v17_transition_repair_v3/seed_1702.pth` | Passes dev gates | local/exact/contextual 6/9/43, Citizen 331/378, STEM 16/21 |
| **Stage 3** translation | `stage3_v17_asl_order_v1` | **Live default 2026-09-22 at user request** | 92.74 BLEU vs 43.58 paired control; reordering 89.43 vs 7.53 |
| **Stage 3** legacy | `stage3_v17_t5_efficient_tiny_locked100_v1` | Rollback target; monotone/SVO defect | 93.36% exact on its own synthetic test; 43.58 BLEU on ASL-order rows |
| **Live / streaming** | — | **Nothing promoted** | See below |

### Live and continuous work — current frontier

A complete data-path audit establishes that the source videos are not globally broken.
All1,160 audited ASLLRP clips are1280x720 or
1280x960 at29.97fps, and all produced features; ASLLRP-other training signs have99.85%
median hand-node presence and92.48% at p10. The continuous observer nevertheless caps
detection at640px and samples20fps, unlike the isolated extractor's1280px cap. In a
controlled11-clip/22-token frontend probe,1280px/source-rate achieved36.36% WER versus
45.45% at640px/source-rate;20Hz also lost one exact sequence. Resolution and sampling
matter, but do not fully explain failure. The stronger defect is supervision loss:
6,020/11,936 annotated events fail the current four/six-observation floors, including
legitimate short signs, and the earlier6,641 known context windows all ended before the
annotated sign end. Slowing cached landmarks only duplicates observations and adds no
information. The old local phrase manifest also leaks all three signers across train
and validation; the existing signer-disjoint split gives25.5% exact and37.04% WER on
200 held-out-signer clips and covers only15 glosses. A post-audit precheck corrected the
next-step interpretation: source-rate1280px ASLLRP caches and native30fps local caches
already exist, and `unified_streaming_aligned_grounded_v17_v1` already evaluated the
proposed8-frame causal CTC path on them. Do not repeat extraction or that training run.
The matched no-standalone-blank ablation selected epoch8 in106s and regressed local
held-out WER37.04%→45.19% and exact25.5%→16.0%; ASLLRP contiguous stayed41.67% WER,
NCSLGR worsened78%→84%, and isolated exact rose82.37%→83.19%. It fails promotion and
shows that questionable standalone blank clips are not the dominant error. Current
evidence now supports obtaining materially broader signer-disjoint connected coverage
rather than another CTC/boundary-head patch. Audit/report/videos:
`artifacts/reports/stage2_data_path_audit_20260921/`.
Matched ablation: `artifacts/reports/native_ctc_no_blank_v17_20260921/`.

A demo app shell now wraps the reel live path: `scripts/app_shell_v17.py` presents home,
a100-sign gallery, session history and the live feed as pages of one window. It changes
no model, checkpoint, threshold or decoder and promotes nothing; the live page carries an
EXPERIMENTAL badge. The reel script was not refactored — it keeps its loop and calls
`shell.present()` per frame (42 insertions/18 deletions), with `NullShell` preserving the
old single-page behaviour. Equivalence is measured, not assumed: six clips decode
identically through the pre- and post-change modules on every recorded prediction field,
and HUD pixel hashes are unchanged. Gallery examples come only from ASL Citizen *train*
(`scripts/build_gloss_examples_v17.py`,100/100 classes, chosen from landmark diagnostics
without decoding); validation and the sealed test split are never read. Demo sessions go
to `artifacts/app_sessions/`, never `artifacts/reports/`, so they cannot be mistaken for
experiment provenance. Both app directories are gitignored.

Frozen coherent decoding confirms that the failed segment head cannot be rescued by a
better state decoder alone. Averaging rolling evidence, applying a v16-style hand gate
and enforcing OUTSIDE→START→SIGNING→END raises held-out online boundary F1 to18.79% at
±100ms and40.46% at±200ms and lowers visible WER204.69%→103.65% by cutting insertions
261→33, but deletions rise to121/192 references. Synthetic held/repeat probes are2/20
and0/20. The checkpoint stays rejected. A follow-on train-only Luna annotation pilot
is described below. Report:
`artifacts/reports/segment_first_coherent_decode_v17_20260921/`.

The user-directed one-Luna-per-clip pilot is complete on24 train-only known signs. Each
clip had a separate blind Luna-low reviewer. Nineteen medium/high, non-censored rows are
provisional, but just6/19 agree with both source edges within100ms and14/19 within200ms;
median absolute start/end differences are55ms/128ms. Four reviews are edge-censored and
one is low confidence. The comparison review establishes that this disagreement does
not show broadly wrong ASLLRP annotations: the published convention excludes setup and
release motion and may annotate final holds separately, while Luna reviewers often
selected a fuller visible articulation. Therefore±100ms was too strict as an annotation
quality verdict, although the model still fails at±200ms. Confirmed defects are in the
derived supervision:21/1,160 crops clip an annotation (22 occurrences, all OTHER), and
the prior context objective ended every known window before the annotated sign ended.
Single-reviewer Luna intervals remain unapproved training truth. Comparison/report:
`artifacts/reports/luna_boundary_annotation_pilot_20260921/comparison.html`.

The segment-first confident-only experiment completed in5m05s and failed promotion.
Its frame-level OUTSIDE/START/SIGNING/END head reaches only14.44% held-out online
boundary F1 at±100ms (9.44% precision/30.69% recall) and24.41% at±200ms. Visible
locked100 WER is204.69% with261 insertions over192 references. Exact-core recognition
remains much stronger: ASLLRP contiguous81.82%, ASLLRP other81.22%, O5S538.24%, and
matched predicted-segment gloss71.70%. The known/unknown gate is63.34% balanced.
Citizen/SemLex isolated validation is92.33%/83.23%. Synthetic probes pass6/20 held-once
and2/20 intentional-repeat cases. Verification passes, Citizen test stayed sealed and
default runtime is unchanged. Do not promote this checkpoint; the simple four-state
frame head does not solve boundary localization. Report:
`artifacts/reports/segment_first_v17_20260921/`.

The corrective whole-sign/localization audit completed without training. It evaluated
8,367 strict cores and42,111 endpoint windows after excluding incomplete crops,
overlaps, insufficient observations and missing edge coverage. Adapted ASLLRP-other
strict-core accuracy is86.63% train/72.95% held-out; contiguous is88.14%/82.35%.
Limited context reduces held-out ASLLRP-other to67.21%. Thus clean ASLLRP sign identity
is learnable. O5S5 remains a signer/source failure at75.94% train versus23.26% LG.
Localization still fails: train-selected phase threshold gives43.14% held-out known
recall,61.12% balanced accuracy and152/261 events detected with correct gloss; three-way
phase accuracy49.79%. Keep locked100, reject checkpoint, and separate strict whole-sign
identity training from boundary evidence that is visually reviewed rather than inferred
from annotation gaps. Report/videos:
`artifacts/reports/clean_boundary_subset_20260920/`.

The derived confident-only continuous manifest remains provenance evidence, but it is
not suitable unchanged for the next Stage-2 experiment. It accepts5,331/11,936
annotated intervals:1,017
known and4,314 explicit unknown, covering61/100 training classes. Accepted events are
complete/non-overlapping, cover both annotation edges, have at least six raw target
observations, at least80% hand visibility, maximum1.20s duration and no target clock gap
above80ms. The six-observation floor at20Hz systematically removes short signs and must
be regenerated against the native-rate cache with masks. Model correctness was not a
filter. O5S5 supplies positive cores/edges only;
annotation gaps are never transition targets. Original data/manifests remain unchanged.
Report/manifest: `artifacts/reports/confident_supervision_v17_20260920/`.

The clean boundary-phase fixed-window experiment completed and **failed promotion**.
It separately trained100-gloss identity and KNOWN/UNKNOWN/TRANSITION endpoint phase on
matched32-frame0.27s/0.53s windows, with complete coverage and strict routing: incomplete
O5S5 supplied known cores only, while negative phases came from fully annotated ASLLRP.
After12 epochs, validation phase accuracy is50.23%, known-core gloss55.53%, and online
complete-ASLLRP WER106.82% with86 deletions,110 substitutions,133 insertions and55/237
empty transcripts. Citizen/SemLex isolated retention fell from95.24%/85.28% to
93.39%/84.76%. Contract checks and64/64 CPU/MPS decisions pass; Citizen test stayed
sealed. A subsequent direct data audit qualifies the interpretation: shared tensor
contracts did not resolve all supervision/sampling mismatches. All known context
windows end before the annotated sign ends; 43.78% of training windows contain less
than half target observations, 31 known training events yield no window, and two FEEL
windows contain no raw target observations. The source crop-completeness flag was lost
for 21 ASLLRP rows (22 clipped OTHER occurrences). Gap-derived TRANSITION targets are
not independently verified physical transitions. Fresh training/held-out ASLLRP-other
known-gloss accuracy is 88.14%/59.90%; phase accuracy is 69.33%/51.04%. This demonstrates
partial learnability plus fitting/generalization problems, not proof that the data or
encoder is unusable. Of 6,089 ASLLRP training OTHER occurrences, 83.26% are lexical and
6.19% fingerspelled; adding A–Z will not cover most OTHER. Correct supervision and
review complete-sign evidence before another run. Audit and annotated videos:
`artifacts/reports/boundary_data_audit_20260920/`. Default runtime remains unchanged.
Original experiment report:
`artifacts/reports/boundary_phase_v17_20260920/REPORT.md`.

The fixed Finish-time bounded CTC experiment completed in10m16s and **failed
promotion**. A frozen Stage-1 encoder plus one400,174-parameter bidirectional GRU CTC
head, trained for20 full-coverage epochs with one objective, lowered connected/familiar
WER versus the repaired causal CTC to83.80%/88.03%. This is deletion collapse rather
than usable improvement:215/284 connected and191/259 familiar reference glosses were
deleted;160/225 connected recordings were empty, familiar exact was0/97, and LG core
CTC was2/57. Citizen/SemLex isolated CTC exact was344/378 and812/978; all16 verified
blank clips stayed empty. Six adjacent duplicate outputs remained versus three in the
references. The visible output contract was verified as exactly the locked100 glosses;
blank/UNKNOWN stayed internal and Citizen test remained sealed. A reporting-only WER
bug that counted matches was corrected and independently recomputed; checkpoint and
predictions were unaffected. Do not promote or add suppression. Report:
`artifacts/reports/finish_ctc_v17_20260920/REPORT.md`.

Earlier streaming experiments completed and **failed promotion**. A bounded Stage-1
window experiment has completed its bounded training; nothing new is promoted:

- Unified streaming CTC (rolling 32-frame, stride 4): 76.15% exact local, but 0/12 exact
  on signer-disjoint ASLLRP and it drops Citizen isolated from 95.77% to 73.28%.
- Causal continuous v2: 53% local exact, 0/12 ASLLRP. Not promoted.
- Live-matched v1 temporal adaptation: no epoch passed both retention and matched gates.
- Earlier immutable-prefix experiment: only 29.90% local confirmed exact; stable
  agreement does not guarantee correctness.
- Revisable transcription is implemented behind `--revisable-transcript`: recent words
  may change, and Finish re-decodes all retained visual features in overlapping chunks.
  The user selected this experience; holding each sign until a Stage-1 lock is not the target.
- Transition-aware gap/prefix and positive-core training each completed 12 epochs; neither
  passed promotion. Best new connected WER was 73.94%, but 153/284 reference signs were
  deleted (53.87%), and familiar/isolated retention failed. Annotated-gap emissions did
  not improve over the preceding adaptation. Report: `artifacts/reports/stage2_v17_revisable_v1/README.md`.
  Verification: 72 focused tests, 45 raw replays, exact-subset raw/cache agreement 12/12
  per model. Short-phrase first-output latency remains unresolved.

The approved Stage-1 window experiment completed all 12 seed 17111 epochs and failed
promotion. Best connected WER was 146.13% (epoch 4) versus 168.31% CTC, but deletions
increased 11→33; familiar WER 117.37%, Citizen 93.39%, and SemLex 83.64% fail retention.
Epoch 1 alone passes isolated retention but fails connected/familiar/transition gates.
No confirmation seed or Core ML export ran. The opt-in `--transcript-backend stage1-window`
backend is available for research; existing defaults remain unchanged.

The frozen comparison covers 334 recordings and 4,514 raw/replay inputs. Actual epoch 1
replay matched cached transcripts 15/15. On nine verified-boundary clips, 7/15 signs were
recognized: median delay 0.304s, p95 0.681s, misses 53.33%; full-pool latency remains
unverified. CPU/MPS parity covered 7,212 window labels/334 transcripts/16 transition
labels. Sixty-one focused tests pass. Missing independent repetitions, long holds,
background and phone coverage prevents reliability claims. Recent-tail replacement is
tested, but no useful spontaneous online corrections were demonstrated. Matched HUNGRY
diagnostics found strong window sensitivity and no OTHER-suppression explanation.
Report: `artifacts/reports/stage1_window_v17/README.md`.

The bounded O5S5-augmented follow-up completed all 12 fresh original-base seed 17111
epochs and failed promotion. Best connected WER is 123.59% (epoch 3), versus 168.31%
CTC and 146.13% prior no-O5S5 best, but deletions rise 11→42, familiar WER is 111.97%,
and Citizen 93.39%/SemLex 84.25% fail isolated retention. Only epoch 1 retains isolated
accuracy; its connected WER is 311.62%. All 12 checkpoints were evaluated on 334
recordings plus 318 held-out LG positive windows. LG reaches 25.79% at epoch 1 versus
23.90% original base, but falls to 21.07% at best-connected epoch 3. LG annotation is
incomplete, so no full-narrative WER is claimed. No checkpoint passed; runtime latency
remains unverified, and no confirmation seed, export or iPhone test ran. Report:
`artifacts/reports/o5s5_augmented_v17_20260914/README.md`.

The exhaustive post-run audit covers every 8,978 continuous supervision window and all
3,257 isolated replay clips across the original model, all 12 O5S5 checkpoints and the
prior no-O5S5 comparators. It rejects a global extraction failure and rejects O5S5 as
uniquely too fast: O5S5 train targets have 0.259s median duration versus 0.267s for
ASLLRP OTHER, while held-out LG is slower at 0.340s yet remains weak. The actual
seed-17111 sampler never used 63.75% of ASLLRP OTHER windows or 15.32% of O5S5 windows.
The 1.07s target windows contain only 29.1% annotated O5S5 foreground on average, and
23/49 O5S5 training classes have only one training signer. Exact-core pooling raises
O5S5-train top-1 to 61.09% but leaves LG at 27.67%, separating window dilution from a
held-signer/domain generalization failure. No exact feature/target conflicts were found.
Next model work should expose the Stage-1 frame sequence to a shallow temporal CTC head,
retain isolated classification as an auxiliary loss, cover all admitted windows before
repeats, and supervise O5S5 only on verified positive cores. Do not repeat 1.07s
whole-window target CE or label unverified O5S5 context as blank. Reports:
`artifacts/reports/stage2_data_learnability_audit_v17/`.

Independent review qualifies that recommendation: the exact-core probe pools tokens
after noncausal full-window encoding; it is not a raw-cropped-core oracle. The existing
causal dilated CTC head can be reused, but its current trainer caches inference-mode
Stage-1 features. First verify differentiable sequence training and timestamped
32-frame chunk assembly, then compare frozen versus joint encoder training with the
same corrected supervision and coverage. A causal head alone does not make the encoder
causal. Preserve all 102 CTC outputs (blank, 100 glosses, OTHER); defer extra multi-level
heads. Background sample IDs currently collide across gaps, so future coverage keys
must include gap/timing identity; positive-window exposure figures remain valid.
Review: `artifacts/reports/stage2_data_learnability_audit_v17/NEXT_STEP_REVIEW.md`.

The authorized frozen-versus-joint CTC comparison completed both12-epoch seed17111
arms and failed promotion. Every epoch emits no known signs on all334 development
recordings: connected100%WER/284deletions, familiar100%WER/259deletions. Final joint
raw-core train top-1 improves74/199→186/199 (37.19%→93.47%), but LG stays12/57 (21.05%).
Pooled Citizen/SemLex retention passes at95.50%/85.99%, while the CTC output itself
collapses to blank: only2/1,291 known ASLLRP training signs recovered at final joint
epoch; single-clip CTC gets0/378Citizen and0/978SemLex. Thus the next model gate is
full-data CTC fitting, not an assumption that more signer data alone will fix this run.
All24 checkpoints/8,016 recording evaluations, matched1,320updates per arm, complete
923sequence/199core/1,901replay/494background epoch coverage and4,520input hashes were
verified. Preflight4/4 tiny fit and24focused tests pass; final CPU/MPS agrees628/628
checked tokens per arm. Actual emission latency remains unverified (all296timed signs
missed); no confirmation, export or promotion ran. An initial Metal-aborted8-epoch
frozen attempt is preserved separately; the complete retry kept the original recipe.
Report: `artifacts/reports/joint_ctc_v17_20260914/README.md`.

The authorized CTC repair completed and resolves the training blank-collapse failure,
but **fails continuous promotion**. Direct positive CTC/verified anchors, temporary
per-frame CTC normalization, verified sequence-frame CE and a final original-CTC
alignment pass produce1,088/1,291known TRAIN matches (84.28%),32.77%training WER and
1,901/1,901isolated TRAIN CTC exact. The final42-epoch model uses existing data and
architecture;54repair epochs including a rejected12epochbranch are preserved.

Final fixedepoch42 emits known signs on263/334development recordings. Connected WER
98.24% has90deletions/124insertions; familiar WER107.72%, contiguous WER50.00%.
Actual Citizen/SemLex validation CTC reaches89.68%/77.91%, while pooled accuracy retains
95.77%/85.28%. LG remains a domain failure: pooled15/57 (26.32%), actualCTC4/57 (7.02%),
42/57empty, versus199/199training cores through both outputs. No full-narrative O5S5
WER is valid. All choices used training evidence; only finalepoch42was evaluated.

Verified4,520input hashes,54checkpoints, complete epoch coverage and saved optimizer
states;33focused tests pass, CPU/MPS628/628checked tokens agree. Source-clock timing
misses158/296signs and excludes runtime; no mobile readiness or promotion is claimed.
Failed gates: connected deletions, familiar WER, runtime median delay and runtime
inclusion. Citizen test remains sealed; default runtime unchanged. Report:
`artifacts/reports/joint_ctc_aligned_v17_20260915/README.md`.

Default checkpoints are unchanged. `--sequence-preview`,
`--stage2-live-checkpoint` and `--revisable-transcript` are explicit experimental flags.

The September 16 research review traced saved webcam session `20260915_220852_978390`
to the older general selector, not the repaired epoch-42 checkpoint. A single visible
chest-point episode becomes `I I` after two accepted windows, before rollover/Finish;
two separate salutation episodes correctly remain `HELLO HELLO`. Intended glosses
are not expert-annotated. A separate executable counterexample shows CTC prefix
rollover can count one ongoing run twice. Raw logits are absent, so the model cause
of the early duplicate remains unresolved. Stage 3 also renders `I` as “Is that?”.
Nine annotated video examples and a primary-source gloss-free review are available
at `artifacts/reports/stage2_research_review_20260915/{videos.html,README.md}`.
Next diagnostic gate: frozen signer-disjoint hold/repetition/rest/OTHER examples
with framewise emissions and absolute-time replay. A released Uni-Sign ASL model is
a separate gloss-free challenger, not an approved Apple Vision replacement. No
runtime or training changes were made by this review.

Following user authorization, CTC rollover now carries the last raw token across
the cut in both live callers and clears it on Finish/reset. A failing regression
was reproduced first; 51 focused tests pass, including the Reel event loop across
multiple rolls and blank/OTHER repeat cases. Optional `--ctc-trace-dir` saves logits,
frozen features and observed source times. This corrects the proven boundary case,
not the short-context webcam duplication or generalization failure. The pinned
10-recording/four-phase replay completed all 40 runs: no final output changed from
the rollover correction, and three long-context live/offline outputs differed.
Window origin changes EAT repeats and sign identities; I I did not reproduce.
A fixed existing beam8/topk12 control leaves EAT/WHEN repetitions unchanged and
increases labelled diagnostic edits 32→34 over 32 reference tokens. These are
correlated selected examples, not a generalization benchmark. Model probabilities
strongly favor repeated WHEN. No decoding or Stage-3 string-dedup change is justified.
The authorized released pose-only Uni-Sign ASL comparison is complete and reviewed.
Decision: reject this checkpoint as a drop-in Stage-2/Reel replacement. Outputs
invent details and fail the existing annotated HELLO HOW YOU clip; fluent English
does not demonstrate corrected recognition or repetition. This does not reject
gloss-free translation generally or isolate the underlying model/preprocessing cause.
Some sentence predictions preserve useful meaning. Fine-tuning remains an untested
candidate; short-clip failures do not isolate duration from task/domain differences.
Runner and reports: `artifacts/reports/unisign_asl_baseline_20260916/`.
Fixed scope: released How2Sign pose-only checkpoint with strict loading; native
lightweight Wholebody preprocessing; 12 realigned validation utterances from four
source videos (filename signer IDs 1 and 2), plus nine prior qualitative clips.
No training or automatic promotion. Corrected annotations require raw-video cuts;
the local How2Sign training subset is unsuitable for this checkpoint evaluation.
All 21 outputs verified; paired development BLEU 8.21 and chrF 38.06 independently
recomputed. No numerical webcam accuracy: those clips lack expert English targets.
The user subsequently authorized the direct-translation training pilot below;
repetition is acceptable during this experiment and is not a rejection gate.
Earlier diagnostic reports and decision:
`artifacts/reports/stage2_held_sign_diagnostics_20260916/`. Long work must run
detached with an exit notification and no assistant polling. No new training ran.

### Authorized September 17 direct-translation pilot

Direct-translation experiment status: completed; review its REPORT.md.
Report/runner: `artifacts/reports/stage1_direct_translation_20260917/`.
Architecture: runtime Stage-1 Squeezeformer frame sequence -> linear projection ->
complete pretrained ASL mT5 encoder-decoder; retain the isolated classifier with
auxiliary CE. No CTC, gloss commitments or repeat suppression. This adapts the
released Uni-Sign text component to our Apple features, not its native pose encoder.

Frozen data: 994 realigned How2Sign TRAIN utterances from filename signer IDs
3/5/8/11, 1,901 isolated TRAIN examples; 12 paired development utterances (signers
1/2), nine qualitative clips and 1,356 isolated validation examples. Verified
4,272 input hashes. Inherited mT5 pretraining is not certified signer-disjoint.
Recipe: seed17111, 20 epochs, one projection-only warmup then joint training,
complete epoch coverage, translation CE + 0.5 isolated CE, Adafactor. Final epoch
is fixed before evaluation, with initialized hybrid, unchanged Uni-Sign and
zero-visual controls. No automatic deployment; Citizen test remains sealed.

Final epoch20 completed; translation training loss0.16688. Independently recomputed
12-sentence development BLEU1.076/chrF18.989 versus initialized0.745/16.652,
zero-visual0.780/16.009 and unchanged Uni-Sign8.206/38.059. Isolated development
retention: Citizen360/378 (95.24%) unchanged; SemLex832/978 (85.07%), down2 clips.
Translation remains poor: inspected outputs introduce unrelated content. Passing
the pilot's weak descriptive checks is not successful translation or deployment.
Final checkpoint is internal; SSD EIO prompted recovery from epoch13 before final
completion. Space guards:8GiB at startup,4GiB checkpoint saves,1GiB reports.
No active training process. Citizen official test was not accessed.


### September 17 English text-model comparison — approved

User requested smaller English-focused alternatives with no significant accuracy
sacrifice and explicit approval before another training run. Review only:
`artifacts/reports/english_text_model_review_20260917/REPORT.md`.
Pinned-config parameter counts: BART-base hybrid 146.41M (75.2% fewer than current
589.39M), T5 v1.1/FLAN base hybrids 254.57M, FLAN small 83.88M. Hypothetical
64K/32K mT5 vocabulary trims yield 303.52M/254.57M while retaining transformer
weights. Actual retained token sets and accuracy remain untested. Current target
padding is 64.8% of positions (994 train texts; mean 24.66 tokens versus pad 70).
User approved one BART-base challenger and a no-retraining trimmed
mT5 comparison after current results. Separate runner/report:
`artifacts/reports/english_comparison_20260917/`.
English comparison status: completed.
English comparison completed20 epochs on September20; no active training. BART
146.41M parameters: BLEU1.061/chrF17.464 vs full mT5 1.076/18.989, failing the
proposed1-point chrF retention margin. Training10552s (~2h56m); isolated validation
Citizen358/378 (94.71%), SemLex841/978 (85.99%). Trimmed mT5 303.52M retains identical
predictions on all21 checked clips (48.5% fewer parameters), but preserves poor
translation. BART outputs also introduce unrelated content. Mismatched visual BLEU
1.287 exceeds correctly paired1.061; higher paired chrF alone is weak grounding
evidence. Do not promote either hybrid. Metrics independently recomputed; twelve
correlated reference sentences cannot establish accuracy equivalence or mobile readiness.
Full report: artifacts/reports/english_comparison_20260917/REPORT.md.

Direct-translation failure diagnosis completed September20; no new training. BART
copies a994-row training caption exactly on11/12 held-out rows; mT5 median nearest-
training-caption similarity is0.803 with6/12 at least0.80. Zero-visual BART emits one
exact training caption for all12 clips. Correct pairing weakly changes caption choice,
but neither hybrid composes translations. The994-pair set is1.63h/four signers, with
80.4% from two; extraction shows6/4,813invalid windows and98.7%median hand presence,
so raw speed/global landmark loss is not the main cause. Eight high English-word-rate
rows are annotation-risk flags; How2Sign sentence/context noise is secondary to the
measured memorization and visual-text alignment failure. Do not rerun another LM swap
on the isolated-window/linear-bridge recipe. Current mobile Stage2 should remain a
bounded blank+100-gloss temporal problem with targeted signer-disjoint hold/repeat/
rest/transition supervision; keep full-data gloss-free translation separate. Report:
`artifacts/reports/direct_translation_failure_diagnosis_20260920/REPORT.md`.

The follow-up connected-data search selects ASL-Homework-RGBD as the next acquisition,
not another decoder experiment. It is the only identified source combining broad
signer coverage (45 people:24 fluent/21 learners),1920x1080 RGB, stable participant IDs
and human ELAN sign onset/offset glosses plus an explicit Signing Happening tier. Full
data require authorized Databrary volume1249 access; only its F13 sample is public and
already local. The signed-in Brave account currently sees zero session files at the
Authorized Users release level; no annotations or recordings were downloaded and no
access request was sent. Resume with EAFs and demographics after volume-owner access is
granted, audit exact locked100 coverage, freeze fluent signer-disjoint roles, then
acquire only matching RGB. The official full
NCSLGR index/database bundle is now local and verified:1,887 utterances/38 collections;
59 of76 screened target-bearing parents have reachable front video, but per-sign XML
still requires a free DAI account and the corpus has only8 participants. Do not bulk
download How2Sign or pseudo-annotated sources for this supervised gap. Report:
`artifacts/reports/continuous_asl_gap_search_20260921/README.md`.

Databrary access is no longer a blocker for **transition pretraining**. The public
LINDAT YouTube-ASL keypoint release was independently indexed: 390,547 unique
sentence-level 2D keypoint JSONs across ten range-addressable ZIPs, with keypoints for
389,474/391,494 provided train/dev annotations (99.48%). Its train/dev source-video
IDs are disjoint (8,479/943), but no signer identities are supplied. The files expose
553 possible MediaPipe points per frame, not the 208 stated in the repository prose.
Use a selectively downloaded train subset for masked temporal/motion pretraining, then
fine-tune the closed blank+100 decoder only on trusted labeled data. English captions
are not gloss boundaries and must not expand or directly supervise the output
vocabulary. The frozen pilot manifest now selects2,000 clips from2,000 unique source
video IDs (240,585 frames), estimated at1.91GB compressed; reserve about6GB including
extracted JSON and working space. How2Sign landmarks are the controlled secondary source. Report:
`artifacts/reports/free_continuous_asl_alternatives_20260921/README.md`.
The earlier managed transfer was externally terminated after22 validated clips. The
same resumable2,000-clip transfer is now running under macOS launchd label
`org.slt.youtube-asl-pilot`; existing atomic outputs are retained and success/failure
notifications remain enabled. Do not poll it; inspect status only after notification or
an explicit user request.
After session78480 hit a transient LINDAT DNS failure, the downloader was hardened to
reopen a shard up to12 times with bounded backoff and to checkpoint every completed
clip; previously validated files remain resumable.
Session32891 later stalled inside a network read at493 clips. Range reads now time out
after30 seconds, and the resumable job runs outside the tool lifecycle in detached
screen session `11217.slt_youtube_asl_pilot`. A failed LaunchAgent attempt was removed.
LINDAT throttled that session after496 clips. The current detached screen session is
`15719.slt_youtube_asl_pilot`; exactly one Python worker is active. A surviving orphan
from the earlier screen had caused two concurrent workers to request identical members
and amplify throttling. All stale workers and partial files were removed. Keep this
endpoint single-worker; it uses a one-second delay and indefinite network recovery.
Foreground verification observed five consecutive new validated outputs,497→502, with
the worker remaining in `running` state on shard2.
Screen/Terminal launch experiments were removed. The original managed-execution method
is restored as session40900 with one worker; a second monitored gate passed502→507.
Current acquisition is stopped at515/2,000. Direct LINDAT throughput measured only
363,558B/s, and three fresh connections produced one additional file; reconnecting does
not reset the current repository/IP throttle. No downloader is running. Resume only
after cooldown or on a different public network/IP, and verify five exact-range files
before leaving it unattended.
The user enabled a VPN. The persistent-session exact-range downloader then passed two
live gates,515→520 and520→538, with CRC, uncompressed-size and JSON validation. The
full resumable acquisition is active as managed session18670 with one worker.

## Locked decisions

1. **Never rerun or tune against the official Citizen test.** It was consumed once for
   the frozen Apple Vision + v17 Squeezeformer selection. Do not select checkpoints from
   its errors. Do not describe the 93.12% validation score as test accuracy.
2. **ASL Citizen is the sole primary training dataset**, using its official
   signer-disjoint split. Never random-split videos or mix signer identities across
   splits. Never use aspect-ratio distortion.
3. **Per-class floor is 10 train / 3 validation / 5 test signers.** Pin one exact raw
   gloss plus one ASL-LEX code per class. Do not merge numeric variants by normalized
   label. Citizen has 35/6/11 signers overall — do not claim every class has 20.
4. **Apple Vision is the locked v17 extractor.** MediaPipe stays a separately
   fingerprinted challenger; never mix its archives with Apple archives, and never
   replace Apple without new signer-disjoint evidence.
5. **v17 is a clean schema boundary.** The 96% v16 checkpoint is not compatible with v17
   features. Do not silently connect them or report v17 accuracy from a v16 model. v16
   scored 40.28% top-1 on an external Citizen audit and had reversed Apple chirality,
   fake zero-valued gap fills, and unconditional presence masks — the specific defects
   are in `docs/ground_truth/stage1-architecture/high.md`. Its aspect-distortion
   augmentation must never be carried into v17 training.
6. **PopSign is one-handed smartphone signing.** A PopSign-only model is a one-handed
   isolated-sign recognizer, never a general two-handed ASL translator. It is not the
   primary v17 dataset. Do not resume the paused audit download unless portrait
   one-handed auditing is specifically useful.
7. **Portrait is the canonical capture mode**, but the extractor must accept portrait,
   landscape, square, rotation-tagged, explicitly rotated, and mirrored input without
   geometric stretching.
8. **Do not distill or compress yet.** Mobile readiness requires measured Core ML size,
   memory, cold start, sustained latency, thermals, and accuracy on real iPhones.
   Desktop parameter count or MPS timing is not evidence.
9. **Do not use standard YOLO Pose** as the hand extractor — its keypoints lack the 21
   finger joints ASL needs.
10. **CTC blank is index 0.** `OTHER` is a Stage-2-only class (101 nonblank, `OTHER` at
    index 101) removed after collapse. Blank is a no-emission symbol, not a transition
    class. Never label unverified surrounding signing as blank.
11. **Synthetic and generated phrases are training-only.** They must never enter
    validation, sealed testing, or model-selection truth. Rejected synthetic motion is
    barred from every dataset and renderer.
12. **Rylo is a reference, not a corpus.** Its hosted `.pose` outputs may be qualitative
    comparators only — never training, validation, or test labels. Do not scrape its
    media. SignBank+ is CC BY-NC 4.0 notation/text, not continuous signer video.
13. **ASL Citizen is research/noncommercial data.** It cannot be assumed to license a
    commercial shipping model.
14. Mirror/TTA must swap hand indices `0-20 <-> 21-41` with X-axis sign flips on
    coordinate and motion channels.

## v17 extractor state

**Location:** `active/v17/` — `schema_v17.py`, `geometry_v17.py`, `extract_v17.py`,
`audit_v17.py`, plus `src_v17/` wrappers and `test/test_v17_extractor.py`.

Feature tensor: **`[32 frames, 61 nodes, 5 channels]`**, float16.

- Nodes: 21 left hand, 21 right hand, 15 face samples, 4 upper body.
- Channels: body-relative X, body-relative Y, relative log-scale depth proxy, binary
  presence, confidence.
- Missing spatial/depth/confidence values are exactly zero.
- Every archive embeds a schema fingerprint; a config mismatch is rejected on load.
- `window_stride` is part of the Stage-2 feature config. Stride-32 keeps the original
  fingerprint; stride-8 archives get `44e9f97c67a003c4`.

Default extraction: 32 output frames from at most 96 uniformly sampled source frames;
long side capped at 1280px without upscaling or distortion; body/face every 8 sampled
frames, hands every frame; minimum joint confidence 0.15; hand gaps up to 3 frames and
auxiliary gaps up to 16 interpolated only when bounded by real observations;
leading/trailing inactivity trimmed with 2 frames context; at least 2 hand frames
required.

Orientation: OpenCV honors rotation metadata; `--rotation 0|90|180|270` overrides it;
`--input-mirrored` flips stored mirrored pixels exactly once before Vision. Vision always
receives upright, unmirrored pixels. Coordinates use isotropic geometry from the longest
image side.

## Data and storage state

- The bounded O5S5-augmented continuous-development experiment completed and failed
  promotion. Broader 100-class coverage and independent phone recordings remain open. See
  `artifacts/reports/continuous_asl_acquisition_20260912/DATA_NEEDED.md`.
  Three corpus access inquiries are prepared in the same report folder as
  `ACCESS_REQUESTS.md`; none sent. New MoLo interview and Daily Moth EAF checks
  do not establish full gloss targets;
  do not equate English translation tiers with sign annotations.
- Continuous ASL acquisition completed 2026-09-13 under
  `data/local/continuous_asl_acquisition_20260912/`; inventory and verification:
  `artifacts/reports/continuous_asl_acquisition_20260912/README.md`.
  Épée provides 1,200 timed sequences from six source signer IDs, with 68/100 exact
  raw-label matches, but is MediaPipe-only with no raw video; it is incompatible
  with the Apple Vision input contract. MoLo adds 17.29 minutes of original raw
  video, two annotated signers (1,517 hand annotations), and verified signer crops.
  RIT adds a 6.5-second public fluent-coded sample. These remain outside training
  pending annotation completeness, alignment and exact visual-variant review;
  unannotated/OOV spans are not background. No phone-generalization claim follows.
- Better no-account continuous data was acquired 2026-09-13 under
  `data/local/open_asl_alternatives_20260913/`; measured report:
  `artifacts/reports/open_asl_alternatives_20260913/README.md`. O5S5 contributes six
  frontal paired raw-video/EAF narratives, 21.73 minutes, six signers and 3,959 timed
  hand-tier annotations. Exact equality from each frozen Citizen ASL-LEX code through
  official `SignBankAnnotationID` to the O5S5 ID gloss admits 256 deduplicated positives
  across 53 classes. Real Apple Vision replay produced 26,079 observations and detected
  hands in 256/256 targets. LG is validation-only; the other five signers are train-only.
  Combined ASLLRP+O5S5 loading yields 6,819 train context windows across 73/100 classes;
  O5S5 supplies 1,005 across 49 classes. Its gaps are never background because transcript
  completeness is unproven. Ready manifest and audits:
  `artifacts/reports/o5s5_citizen100_v17/`. RWTH-BOSTON-104 remains low-resolution
  auxiliary sequence data because its official split reuses all three signers. SoMe ASL
  is excluded by user visual review and must not enter training.
- Raw/local datasets go only under `data/local/`. Generated reports under
  `artifacts/reports/`, disposable outputs under `artifacts/generated/`.
- Never commit or delete datasets, checkpoints, reports, or metrics without an explicit
  request.
- PopSign is ~1.1 TB in full — never download wholesale. One sign/split archive at a
  time, check free space before transfer, preserve license provenance.
- Free space was ~7.1 GiB after the 2026-09-13 acquisitions.

## Environment

- Host: macOS on Apple Silicon. Project Python: **`venv/bin/python`** (3.9).
- System Python lacks the PyObjC Vision/Quartz bridge — real extraction and tests must
  run through `venv/bin/python`.
- Isolated research envs: `artifacts/generated/mobileclip2_env`,
  `artifacts/generated/movinet_env` (CPU-only; TF Metal unsupported for its Conv3D graph).
- MPS is correct for training but wrong for token-by-token live decoding: live Stage 3
  measured 137.96 ms on CPU versus 1,678.02 ms on MPS. Live Stage 3 defaults to CPU.

## Immediate next actions

1. The frozen-versus-joint comparison is complete and negative. Before another model
   comparison, diagnose full-data CTC blank collapse on training sequences: alignment/
   blank pressure, token-to-target density and direct CTC supervision from isolated/
   positive-core examples are untested hypotheses. Require known-sign training emissions
   before claiming a signer-generalization experiment. Do not extend the failed recipe
   or tune suppression on LG; retain all existing promotion gates and test sealing.
2. Collect a new portrait-iPhone signer-disjoint evaluation set. This is the only valid
   dataset for measuring future model changes without contaminating the consumed Citizen
   test.
3. Measure Core ML package size, memory, cold start, sustained latency, and thermals on
   real low/medium-spec iPhones.
4. Design UNKNOWN / out-of-vocabulary rejection and evaluate it on independently held-out
   nonsign clips before presenting the classifier as an app feature.
5. Obtain ASL-fluent review of the raw-gloss/ASL-LEX mappings and frequent confusions.

## Known boundary

A v17 classifier is trained with a one-time official signer-disjoint test result of
87.57% top-1. **Not established:** independent portrait-iPhone accuracy, UNKNOWN
rejection behavior, ASL variant review, real-device performance, and any continuous or
conversational translation claim. Stage 3 is a bounded synthetic/reviewed renderer, not
a general ASL translator. Do not claim production mobile readiness, continuous sign
recognition, or end-to-end conversational translation.

---

## Archive map

Full history: **`docs/ground_truth/`** — 354 entries, 805 KB, nothing deleted.
Search it with `rg`, do not read it. Detail and conventions: `docs/ground_truth/MAP.md`.

| Topic | high | log | Covers |
|---|---:|---:|---|
| `stage1-architecture/` | 19 | 94 | Encoder, extractor bakeoffs, component ablation ladder |
| `text-to-sign/` | 6 | 67 | SignWriting, avatar rendering, motion generation |
| `live-streaming/` | 10 | 45 | Reel path, streaming CTC, commitment and latency |
| `data-sources/` | 8 | 33 | Acquisition, licensing, admission audits, split policy |
| `stage2-ctc/` | 9 | 17 | Continuous recognition, CTC contracts, selectors |
| `signing-voice/` | 2 | 17 | Style transfer, coarticulation, transition synthesis |
| `mobile-deployment/` | 7 | 11 | Core ML export, orientation contract, iPhone/Flutter |
| `stage3-translation/` | 2 | 3 | Gloss-to-English models and their gates |
| `capstone-paper/` | 0 | 4 | Paper revisions, benchmarks, repository hygiene |

`high.md` holds the dated evidence behind the constraints stated above. `log.md` holds
measured results and rejected approaches — `rg` it before re-running an experiment, so
you do not repeat one that already failed.
