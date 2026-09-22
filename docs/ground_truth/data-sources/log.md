# data-sources — log

Measured results, rejected approaches, and progress snapshots. Not read start-to-end —
`rg` this file before re-running an experiment to see if it already failed.

45 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## 2026-09-22 — combined dataset manifest created and fully loaded

User requested creation immediately. Created data/local/combined_dataset_v17_20260922/manifest.json
from all five finalized supplement lists plus all494baseline records, including431localphrases.
6421records=4547train/1874validation; source roles, hashes, representations, targets and
139sharedASLLRPparent records preserved. Source manifests/vocabulary pinned. No file copying,
source mutation, downloads, test features, or training. Manifest SHA256:
089da8b13943b65663691cc7e23c8bd1dcb279fceca56ab1675fa768d19c95f2.
Added scripts/build_combined_dataset_v17.py with representation-aware load_features.
Initial check failed because isolated positive-core per-window hand floor was applied to
approved full phrases with natural rest windows; corrected shared feature_check to allow
explicit require_hands=False only for phrase supervision. Structural checks unchanged;
existing supplement callers retain defaultTrue. Full rerun saved/reopened6421records and
verified7367windows plus feature/raw/source/evidencehashes andCTCtargets. Focusedtest passes.
Report artifacts/reports/combined_dataset_v17_20260922/REPORT.md and verification.json.
Next: dedicated combined recipe consuming this manifest; no automatic legacy trainer
migration or training launch. Validation never becomes training due to signer policy.


## 2026-09-22 — existing supplements finalized, signer overlap no longer an admission gate

User explicitly permits shared/missing signer IDs and inclusion of local phrases; keep
existing roles and describe metrics honestly. Finalized five per-source lists under
artifacts/reports/supplement_finalization_v17_20260922/: 5,927 records (4,264train/1,663val):
Citizen1475/378, SemLex1388/953, ASLLRPsegmented1116/254, reviewedSTEM90/21, O5S5cores195/57.
Three incomplete tails safely recovered into separate landmark-only archives from pinned
raw video (STEM exact reviewed intervals); 252 exact O5S5 cores materialized. Originals kept.
36 exclusions:1knownCitizenreject,6missingSemLexfeatures,25exacttrain/valSemLexduplicates,
4O5S5intervals with fewer than2cachedobservations. No signer-based exclusions. SemLex951
retainedvalidationrows share training signer IDs; descriptive familiar-signer metrics.
Five manifests pin exact labels/codes/features/raw hashes and source evidence; oldSemLex
approvalflag resolved by frozenbase explicitapproval and matching selectionhash.
Scripts finalize_supplements_v17.py / verify_finalized_supplements_v17.py and focused
feature-contract test added. Full frozenStage1MPS verification:5,927records/5,937windows,
finite100classoutputs,zerooptimizersteps; no exactrawhashoverlap with494baseline. ASLLRP
parent utterance overlap tracked, not independent-event count. No protectedtestfeatures,
download,training,orcombinedmanifest. 494+5927=6421records(4547train/1874val),not allphrases.
Next: combined recipe/manifest using these pinned lists and existing494 (includinglocal),
with explicit representation loading and parent-aware sampling. No learnability guarantee.


## 2026-09-22 — broad existing-source reconciliation corrects narrow audit scope

User challenged omittedFlores/O5S5/etc and required complete trustworthycounts. Confirmed
lastbaseline mixedlocal/ASLLRPwithin epochs, notonecorpusatime, butomittedotherestablished
supervision. Prior494count wasneverwholeprojectusableinventory. No sourcefilesdeleted.

Restored explicit prioradmissions:ASLLRPsegmented1116train/254val;O5S5officialpositive
intervals199/57across53classes(sixrawApplecachesexist);STEMfinalreview111spans31classes,
features90train/21val. Freshvalidator over1370ASLLRP+111STEM:1ASLLRPtraintail and2STEM
traintailsblocked;1369+256+109=1734supplementalrecords(1402/332), notuniquevideos.
ExistingCitizen1476/378,SemLex1388/978;Citizenextra1stillunresolvedagainstbase1475.
Localdeepclean13381/2896exists withownerlabelapproval/familiar-signer restriction;
ASLLVD175officialexacttrain-onlyfeatures exist, schema/global signer compatibilitypending.
Flores155/141completeOTHERcache;NCSLGR166/125strict-targetphrasecache remainconditional
lexicalsupervision, notblanketbadcorpora. Motion-onlyrawpathsverifiedHow2Sign1027,
YouTube128,OpenASL3,NCSLGR166;phone5. Reviewed prior acquisition restrictions forRWTH201,
Epee1200,MoLo/RIT,SoMeexcluded,PopSignpaused. Do notcountcachedwindowsasnewvideos.

RWTHconflict explicit:olderprosepermitsauxiliaryuse butacquisitionaudittrainingeligiblefalse
pendingexactidentity/signerreview;keptconditional. Do notsilentlyrevoke existingpositive
O5S5/STEMadmissionsbecausenotwholephrases;do notpromoteunresolvedmappingsforlarger totals.
Outputartifacts/reports/dataset_reconciliation_v17_20260922/{REPORT.md,evidence.json};
evidencepinshashes/currentcounts/blockedpaths. AGENTS/currenttruthcorrected. Nextsafeaction
iscompatiblemixed-supervisionregistry withsource-event/global-signer dedup,threecacherepairs
andCitizenextra reconciliation;no new traininglaunched, noacquisition/protectedtestaccess.

## 2026-09-21 — one excluded ASLLRP subspan salvaged; v2 canonical

User requested repair where genuine, no compromise, stop at evidence limit. Added
scripts/salvage_verified_phrases_v17.py and test/test_salvage_verified_phrases_v17.py;
missing-module test failed first, then41focused tests passed. Scanned excludedASLLRP
ledger:507established complete/nonoverlapping runs≥2signs with≥1known, only2gap-free.
Ruling: require adjoining annotation intervals for this salvage pass; unannotated gaps
are not certified absence of signs. Cost: conservative exclusion of505potential runs;
not evidence they are bad, not a new global rule invalidating earlier approved phrases.

Recovered WHEN/NOT annotations343321/343322 from originalcrop23882926, sourceframes[5,16).
Official codesC_03_086/E_01_100 map WHEN/OTHER. Hash-verified rawvideo; native11frames,
originalorientation/frozenAppleconfig; observedhandfraction1.0;2stride4/window8CTCsteps.
STILL/HAVE annotations386202/386203 rejected:7frames→1observation cannotfit2targets.
No artificial extension, relabeling, joined footage, acquisition or new reviewer.

Version2manifest active/v17/approved_phrase_manifest_20260921_v2.json pins prior manifest,
annotation evidence and salvage script; view data/local/approved_phrases_v17_20260921_v2
preserves493priorclips and adds1training subspan, total494=283train/211validation.
Source asllrp_verified_span is explicit. Shared defaults andAGENTSupdated; future recipe
must handle new source. Originalfullclips remainexcluded and preserved. All494load;
source-signers/video hashes cross-role disjoint;41tests pass. Current inventory total
increases by1derivedclip, not1new independentvideo; isolated inventory unmodified.
Report/verification: artifacts/reports/verified_phrase_salvage_v17_20260921/.

Training_ready remainsfalse. Stop this recovery pass at the evidence limit. Do not
claim independent unseenOOVbenchmark, broadcoveragegain or thatremainingclipscannever
be recovered. Next safe work is separately justified annotation coverage/event-core
supervision and compatible recipe; no training performed. Compile/diffchecks pass;
largeartifactindex refreshed.

## 2026-09-21 — current usable-inventory count clarification

User asks total remaining data. Read canonical phrase manifest:493 approved whole clips
(282train/211validation). Counted locked100 *.v17.npz archives using the isolated loader's
four current roots and checked every features array for finite (32,61,5) shape:
Citizen1476train/378validation;SemLex1388train/978validation;all4220 structurally pass.
Combined available isolated + approved phrases =4713 (3146train/1567validation), not a
fully approved next-run manifest: isolated membership remains unpinned. Historical base
provenance has1475Citizen training samples, one fewer than current1476; reconcile this
before admission. Excludes sealed Citizen test, duplicate/overlapping event cores/windows,
and additional corpora outside these current inputs. No training/dataset changes.

## 2026-09-21 — approved phrase manifest installed and enforced

User authorized versioned manifest, exclusions and training enforcement. Added
active/v17/approved_phrase_data_v17.py, approved_phrase_manifest_20260921.json,
scripts/build_approved_phrase_manifest_v17.py and test/test_approved_phrase_manifest_v17.py.
Updated both unified CTC training entry points and Flores/YouTube-motion run/preflight
launchers, plus AGENTS.md/current truth. No unrelated edits reverted.

Built non-destructive data/local/approved_phrases_v17_20260921/{phrases,other} symlink view:
493 admitted, 1,365 excluded from 1,858 repaired/current archives. Admission counts:
local train232/validation199; ASLLRP contiguous44/12; ASLLRP OTHER6/0. 56 admitted archives
use recovered evidence. Exclusions:1 hand-detection quality flag;266 unresolved cross-
corpus (125NCSLGR+141Flores);1,098 unresolved ASLLRP sequence annotations. Whole sequences
excluded, never targets removed with corresponding video retained. This does not discard
trusted event cores within excluded sequences. No unseen-OOV evaluation admitted.

Ruling: keep training_ready=false because both old recipes depend on excluded source
validation metrics; blank/rest/auxiliary inputs and unseen-OOV roles remain unapproved.
Cost: a recipe/input-contract update is required before next training; this avoids
silently reintroducing excluded sources or pretending the reduced corpus passes old gates.
The manifest is specifically phrase admission, not an audit of every isolated/RGB dataset.
Do not flip the flag alone or bypass via a different legacy trainer.

Verifier pins exact file membership/hashes and admission-evidence/locked-vocabulary hashes.
Run gates reject stale roots and supplemental datasets before model loading. Checkpoint
and result writers carry manifest path/hash. Historical experiment launchers guard before
pretraining/preflight backward operations. Default roots updated. Preserved original files.

Test-first missing-module failure followed by 40 focused passes. Tampered/missing membership,
modified file/evidence hashes, old roots and early entry-point/launcher guards tested.
Actual shared loader reads all493 with stride4/window8; source-level signer splits and
cross-role video hashes checked at construction. Compile/diff checks pass. Report:
artifacts/reports/approved_phrase_manifest_v17_20260921/{REPORT.md,verification.json}.
Next safe action: define the reduced-source recipe and approved auxiliary supervision,
then issue a new manifest version; no training/acquisition/protected test access occurred.

## 2026-09-21 — versioned tail recovery and cross-source OOV exposure check

User authorized continuation, no new acquisition or training. Added
scripts/recover_phrase_tails_v17.py, scripts/audit_oov_exposure_v17.py and
 test/test_recover_phrase_tails_v17.py. Recovery range test failed before implementation,
then passed. Ruling: rebalance the final 32-frame window plus its 1–3-frame tail into two
16–18-frame windows because the existing extractor requires at least four frames;
retain earlier windows exactly. Cost: changed normalization/resampling in final windows,
so future matched experiments must use the new version in both arms.

All 159 videos existed and matched hashes. Recovered 315 missing source-frame intervals
in 64.5 seconds; 159 landmark-only replacements and 1,699 unchanged symlinks at
 data/local/phrase_tail_recovery_v17_20260921/. Original archives/video unchanged.
New metadata records lineage, range policy and actual landmark schema; no stale RGB
claims. One rebuilt local-validation window fails hand-detection minimum and is explicit
missing evidence, not labeled rest/blank. All 1,858 pass structural/timing and stride4/
window8 CTC checks; all 125 NCSLGR alignments pass. Original hashes, repaired targets,
roles/signers and unaffected window equality verified. 37 focused tests pass.

Exposure report verifies four saved-head/base/initial hashes and Citizen/SemLex base
manifest hashes, checks admitted 88 NCSLGR and 141 Flores training annotations. Of 276
OOV development cores, 259 confirmed seen in ASLLRP; 17/13identities unresolved. EYES,
NEAR and BOTH have Flores raw-string candidates, not certified identity links. Initial
head files only hold config/state. No globally unseen identity is certified; reused
validation signer and ambiguous cross-corpus/blank/incidental signs prohibit that claim.

Reports: artifacts/reports/phrase_tail_recovery_v17_20260921/REPORT.md, inventory.json,
summary.json, verification.json; annotation_identity_audit_v17_20260921/exposure_audit.json.
Existing historical reports remain pinned to original caches. Active training defaults
not redirected. Next safe action: prepare trusted event supervision/admission, leave
unresolved annotations out of OTHER truth, and define seen/unseen/rest evaluation before
step5. No acquisition, training, protected-test access or promotion. git diff --check
passes; large-artifact index regenerated for handoff.

## 2026-09-21 — existing-data identity ledger and OTHER core diagnostic

User says continue after steps1–2. Continued mapping audit and development evaluation
only; acquisition/training remain stopped. Added scripts/audit_annotation_identities_v17.py,
scripts/evaluate_annotation_cores_v17.py and two focused test files. New reports:
artifacts/reports/annotation_identity_audit_v17_20260921/; source data/manifests unchanged.

Mapped9,104ASLLRP crop-associated annotation occurrences via exact official SignBank→
ASLLEX identity and occurrence convention:1,397known/1,844OOV/5,863unresolved. OOV
requires lexical status and a distinct documented code/lemma; missing links and uncertain
categories are not OTHER truth. Missing/ambiguous links5,228; occurrence mismatches604;
nonlexical31. Ledger retains raw labels/timing/signers/reasons. All1,858phrase archives
have admission sidecars; strict full-sequence rule leaves6ASLLRP OTHER training clips,
not an instruction to discard corpora. Local user-reviewed/ASLLRP exact-known provenance
retained; Flores/NCSLGR cross-corpus mapping unresolved for this stricter proposal.

Prepared524unique complete, nonoverlapping cores on existing validation signer JONATHAN:
248known+276OOV/97identities,≥2hand frames/≥80%frame hand visibility. Of OOV cores,
259have ASLLRP-train identity exposure;17have unresolved exposure through other sources.
No globally unseen or independent benchmark claim. Existing validation reused.

Fixed four saved Flores-arm CPU core CTC evaluations (no tuning/training): without/with
seed17321known exact84/85of248, OOV exactlyOTHER40/32of276, OOV blank211/216; seed17322
known82/72, OOV exactlyOTHER1/34, OOV blank253/226. OOV false-known counts25/28/22/16.
Core protocol differs from full-sequence training; not live streaming or Stage1accuracy.
Blank-only output does not establish UNKNOWN recognition. Checkpoint/ancestor/cache/
manifest hashes saved;524unique cores and all prediction counts independently checked.
36focused tests pass, git diff --check passes, large-artifact index regenerated.

Next safe action: recover159incomplete caches from existing video into versioned caches;
prepare trusted event-core supervision and verify OOV identity exposure. Do not start
step5until admission, rest/mixed-stream truth and evaluation roles are defined. No new
acquisition, protected test access, promotion or runtime change.

## 2026-09-21 — authorized steps1–2 repaired;159existing caches blocked

User stopped all new acquisition and approved timing/padding plus loader checks only,
requesting a recommendation on steps3–5. No training or additional download performed.
Modified active/v17/train_unified_streaming_{ctc,aligned_grounded}_v17.py and added
test/test_phrase_data_contract_v17.py. Existing unrelated edits preserved in place.

Test-first reproduction exposed missing input checks and padding-length acceptance.
NCSLGR now aligns against loaded frame counts and rejects out-of-evidence events;
collation requires matching per-sample lengths and ignores padding. Shared phrase
loader enforces Apple schema, finite tensor shape, integral contiguous/nonoverlapping
ranges, full source coverage, locked target mapping and CTC feasibility including repeats.
Both training entry points verify checkpoint vocabulary against frozen100mapping.
Frame-level feasibility uses actual encoder steps. No semantic alias changes.

Fresh actual-loader checks:1,858archives examined,1,699pass,159reject solely for missing
tails (147previously explicit+12NCSLGR). All113complete NCSLGR alignments pass. Files
unchanged; no silent filtering. Existing mixed-root training commands now fail clearly
until affected caches are recovered or explicitly excluded in a versioned manifest.
34focused tests pass, including zero auxiliary padding gradient and real encoder
frame/window feasibility; git diff --check passes. No accuracy claim or protected tests.

Report and per-file blockers: artifacts/reports/phrase_contract_repair_v17_20260921/.
Recommend step3next, step4using only confirmed already-local OOV evidence, and defer
step5until cache admission and evaluation roles are frozen. Recovery uses existing
raw video, not new acquisition; no metadata-only claims of complete evidence.

## 2026-09-21 — public-only search finds NCSLGR ELAN route; repair remains proposed

User restricts new data to public/no-account downloads and established mappings;
no ASL-fluent reviewer is available. No implementation or training authorized by
this discussion. Proposed repairs and source evidence:
artifacts/reports/public_dataset_search_v17_20260921/{SEARCH.md,evidence.json}.

Verified DePaul publisher's public elanBUcorpus.zip by unauthenticated GET:
5,335,358bytes, CRC pass,870parseable EAFs, main and nondominant gloss tiers in all.
850ncslgr10-series EAFs include166already-local10a–10d files and684additional EAFs;
these are not684admitted clips. No missing/reversed alignable times; media sync remains
unverified. Public video.zip Range GET returns206/ZIPmagic,4,108,493,785bytes total.
No full video archive/clip downloaded. Publisher flags ali, DSP Ski Trip and roadtrip2
conversion/media defects. Annotation archive retained under
data/local/dataset_metadata/ncslgr_depaul_public_20260921/; checksum in evidence.
This public conversion qualifies the prior statement that expansion annotations
require DAI access. Same underlying corpus, not new independent signer diversity.
Exact lexical identities, catalog signer roles and media timing must pass admission.

Fresh Citizen train/val metadata-only audit finds2,623ASLLEXcodes outside frozen100:
38,678train rows/35signers and9,926validation rows/6signers. Propose bounded UNKNOWN
training/validation with disjoint OOV identities, official signer roles and whole-lineage
exposure checks. These are candidates, not downloaded/admitted videos. No official test
metadata/video read. Already-local approved STEM/O5S5 remain useful within existing
limits; no new broad fully admissible corpus established. No access requests sent.

Next safe action: user reviews repair/OOV/admission proposal. Keep original annotations
and experiment lineage; fix alignment/masking before changed-data training, distinguish
unresolved mappings from confirmed OOV, and avoid deleting target tokens while retaining
their video. No code/runtime/training-manifest changes, no model runs, no acquisition
resumption. Search source links and measured endpoint checks are in the report.

## 2026-09-21 — v17 annotation-to-training audit, discussion findings

User requests broad local v17 dataset audit, especially annotations/extraction/training,
and genuine out-of-100 OTHER recognition. Read current state/relevant logs and traced
current Flores/aligned-loader paths; no training, dataset changes or protected test access.
New report/evidence: artifacts/reports/dataset_annotation_audit_v17_20260921/.

Fresh structural inspection of all1,858phrase archives across corrected grounded,
ASLLRP OTHER and Flores OTHER roots: finite tensors, valid target indices/name mapping,
feasible CTC lengths, matching embedded Apple fingerprint, contiguous non-overlapping
ranges and no cross-role exact stored video-hash/source-item overlap. These checks do
not certify semantic labels, global signer separation or near-duplicate separation.
Corrected local train covers13glosses versus15validation; Flores141clips/93labels.

New confirmed mismatch:12NCSLGR caches (8train/4validation) omit1–3tail frames relative
to annotation metadata, yielding one extra aligned target step. The actual collator
copies labels to batch width and can assign a blank CE target to padding on all8train
cases. Reproduced; no source annotation extends beyond cached end in these12cases.
Additional147non-Flores caches explicitly record dropped tails (293frames), whereas
Flores excludes such caches. Total observed incomplete caches159/1,858; this is not
proof159clips cut lexical signs. Need event-aware completeness review, not blanket deletion.

Current rejection check measures OTHER on known isolated signs, not unseen-OOV recall.
Known WER removes OTHER; full sequence exact retains it, but selection has no explicit
OOV gate. No OTHER-only validation examples in checked roots; one Flores training case.
Flores/NCSLGR exact text mappings still do not prove cross-corpus lexical equivalence.
Loader probe accepts wrong fingerprint; overlapping-range probe duplicates timeline,
but current archives are compatible and non-overlapping. All-zero windows: local9train/
61validation; Flores32train. Unknown rest-versus-detection-loss cause, not semantic evidence.

Existing tests passed:2Flores mapping/device plus13extraction/ASLLRP preparation.
Public-source spot-check found no verified new immediately admissible corpus beyond logs.
No downloads, model changes, runtime changes, or running-experiment polling. Next safe
action: discuss findings, then repair sample-length alignment/masking, define common
completeness admission and independent OOV evaluation before another changed-data run.

## 2026-09-21 — Flores OTHER MPS retry completed and reviewed

All four arms completed; completion notification returned0. Higher MPS cap avoided
the previous observed allocation failure. Checkpoints exist, original source weights
and validation sources match across arms, only141Flores training sequences are added,
and all saved behavior edit counts were independently recomputed successfully.

Flores improves ASLLRP WER in both seeds54.17→50.00% and62.50→41.67%, and NCSLGR
92→84% and98→86%. Local WER worsens48.52→55.56% in17321, improves50.19→48.33%
in17322. Isolated exact83.92→83.41% and83.41→81.12%. No paired gate passes (0/2).
The stronger behavior warning is local deletions68→204 and52→126, while exact phrases
fall42→14/200 and43→29/200. Lower insertion counts do not establish better recognition.
Synthetic hold exact10→7/20 and6→8/20; repeat7→7/20 and6→4/20. Known isolated signs
with OTHER and no expected sign3→3/1356 and0→8/1356. This diagnostic does not prove
that OTHER caused all phrase deletions; it does not identify a semantic data defect.

No promotion. Flores contains useful supervision for some development domains, but
this exact OTHER-span/10%sample-weight recipe sacrifices local completeness and isolated
retention. These results neither establish unusable Flores data nor justify expanding
acquisition. Existing development sets were reused, not a fresh unbiased test. Protected
test/devtest data remain untouched. Any future change needs a separate matched recipe.


## 2026-09-21T22:47:43+08:00 — Flores OTHER completion

Flores OTHER experiment complete; reports: artifacts/reports/flores_other_mps_retry_v17_20260921; no promotion or protected test access.

## 2026-09-21T22:38:45+08:00 — higher-cap MPS retry launched

Flores OTHER MPS retry running, PID 98619; first optimizer step acknowledged. Memory fraction .35 (~6.2GiB). Same141clips, two seeds, both arms restarted,18epochs. Reports: artifacts/reports/flores_other_mps_retry_v17_20260921/. Only training detached; no polling; completion/failure notification enabled.

## 2026-09-21 — higher-cap MPS stress check passed

Full batch32longest Flores sequences passed3optimizer steps with finite loss and gradients at .35MPSmemory fraction;289behavior sequences loaded. CPU override/mapping tests, compilation and whitespace checks passed. Launch higher-cap matched training; preserve failures, no data changes.

## 2026-09-21 — user directs MPS retry with increased memory

CPU preflight completed successfully:32longest real Flores sequences,3optimizer steps,
finite loss/gradients and289behavior input sequences. No CPU training was launched.
User explicitly requests MPS and use of more available memory. Current memory_pressure
reports64% free. Raised bounded MPS fraction .12→.35 (approximately2.13→6.2GiB);
no unbounded allocator setting. MPS full-batch stress preflight underway. New report/model
roots flores_other_mps_retry_v17_20260921 preserve the failed original and CPU checks.
All data, batch32,18epochs, two matched seeds, initial weights and10%supplement mass fixed.
Both arms restart to avoid mixing devices or partially completed experiments.
Next action: verify higher-cap full-batch check, launch only training, stop monitoring.


## 2026-09-21 — Flores OTHER failure diagnosed; CPU retry prepared

User reported failure. status.json and training.log show the without-Flores17321 arm
completed18epochs and its behavior report, then with-Flores completed1epoch before
MPS allocation failure at layer normalization:1.09GiB tensor+1.08GiB other allocation,
2.13GiB cap, failed33KiB request. No complete paired result exists. This proves memory
exhaustion under the .12 per-process cap, not bad Flores labels or negative transfer.
The prior longest-single-sequence and tiny end-to-end tests missed sustained full-batch
memory behavior. Failed outputs and completed baseline are preserved; stale running
REPORT.md replaced with failure explanation. Failure handler now writes report too.

Added explicit device/report/model flags to runner. CPU override regression first
failed then passed; both comparison arms will restart CPU/two threads with identical
batch32,18epochs, seeds, initial states,141Flores clips,10%sample-weight mass and decoder.
CPU preflight expanded to32longest real Flores sequences and3optimizer steps. New roots:
artifacts/{reports,models}/flores_other_cpu_v17_20260921. No unbounded GPU allocator.

Other report findings:14Flores caches omit1–3tail frames, recoverable through complete
re-extraction but excluded here to preserve the controlled dataset; local correction
already removed55training clips. Historical ASLLRP short-sign/context truncation issues
belong to earlier preparation paths; native-rate caches already exist and are used.
Source-local Flores signer IDs cannot prove cross-corpus signer separation. No new
semantic label defect is established by this resource failure. No protected test access.
Next action: complete CPU full-batch/integration checks, detach training only, notification.


## 2026-09-21T22:27:17+08:00 — Flores OTHER completion

Flores OTHER experiment failed; reports: artifacts/reports/flores_other_v17_20260921; no promotion or protected test access.

## 2026-09-21 — corrected-phrase motion run reviewed

All four checkpoints exist; all saved behavior edit counts independently recomputed.
Each arm used the corrected232local training clips and identical4554total training /
2182validation samples. Local evaluation includes200clips/540reference glosses.

Pretraining improves local WER in both seeds (48.52→43.15%;50.19→44.44%), mean
49.35→43.80% (5.56 percentage points). This is more consistent local evidence than the
original mixed-label experiment, but it is not a clean overall win. ASLLRP regresses
54.17→66.67% in seed17321; only1/2 predefined gates pass. Local phrase exact matches
42→46/200 and43→42/200. In seed17322, insertions90→31 but deletions52→130, so the
lower WER includes a substantial shift toward omitted signs. Duplicate output pairs
62→66 and75→38 do not establish held-sign correctness. Synthetic hold exact10→6/20
and6→6/20; repeat exact7→6/20 and6→5/20. No real adjacent-repeat references exist.
Conditional matched-token ASLLRP median delay changes0→−33ms and0→33ms on only8–11
matched events; this is neither an all-token latency score nor hardware latency.

Conclusion: promising local motion transfer, insufficient overall behavior evidence;
no promotion and no additional YouTube acquisition. Existing development sets were
reused; no protected test was accessed. Completion notification returned0. The separate
Flores experiment is unaffected and was not polled during this review.


## 2026-09-21T22:24:40+08:00 — Flores OTHER training launched

Flores OTHER matched experiment running, PID 84890; first optimizer step acknowledged. Two seeds with/without 141 complete Flores dev clips, 18 epochs per arm. Only training detached; no polling. Reports: artifacts/reports/flores_other_v17_20260921/. Completion/failure notification enabled; no protected test access or promotion.

## 2026-09-21 — Flores OTHER full preflight passed

Real longest admitted Flores sequence completed finite CTC backward and optimizer step
on capped MPS. Shared behavior inputs loaded289 sequences. Additional in-session tiny
integration run exercised the actual optional supplement path:18 training samples
including2Flores,16validation samples, one epoch, strict checkpoint reload, behavior
report and isolated OTHER rejection computation all passed. Temporary smoke checkpoint
was discarded; smoke accuracy is not an experiment result. Mapping/weight tests,
compilation and whitespace check passed. Source/initial-state hashes saved in provenance.
Only full paired training is now eligible to detach; no more audit work remains.


## 2026-09-21 — parallel Flores OTHER experiment audited and prepared

User authorized the proposed with/without-Flores experiment in parallel with the
corrected YouTube motion comparison. Implemented main-thread, no agent or LLM job.
New scripts/train_flores_other_v17.py maps raw full gloss transcripts by exact uppercase
label match with outer sentence punctuation only; numeric variants, fingerspelling
and aliases are not normalized into known labels. Consecutive unsupported glosses
collapse to one OTHER span, known repeats remain. All original raw transcripts retained.
155 source videos hash-checked and all landmark archives structurally checked;141
complete caches admitted,14 missing final1–3 frames excluded. Admitted set has476 known
occurrences covering93 locked labels and462 OTHER spans. Full frame ranges are contiguous,
arrays finite, schema matches existing Apple b872fa3dcc16aab5, and CTC alignments feasible.
No exact content overlap with existing phrase train/validation caches. Cross-dataset
Flores signer identity remains unknown; no signer-disjoint Flores generalization claim.
Old Flores frozen features use a different Stage1 and are deliberately not reused.

Existing aligned trainer gained optional audited Flores supplement support: original
source weights stay fixed, Flores total sample-weight mass equals10% of original.
Two paired baseline-initialized seeds17321/17322,18epochs per arm, same corrected phrases,
other supervised data, frozen Stage1, selection/decoder. No YouTube SSL in either arm.
Separate output roots artifacts/{reports,models}/flores_other_v17_20260921 and
 data/local/stage2_v17_flores_other_20260921 prevent collisions with the running pilot.
Existing process already imported its trainer; no ongoing training state modified.
Two CPU threads and .12 MPS memory cap planned (24GiB machine,43% free at initial check).

Focused mapping/CTC-feasibility/weight test first failed for absent module, then passed
following implementation. Compilation and git diff --check pass. Full audit passed;
real-device training/evaluation preflight is in-session before detached launch.
Report metrics include known-sign OTHER rejection, WER, isolated accuracy, synthetic
hold/repeat and conditional emission delay. No protected test/devtest access or promotion.
Next safe action: complete real preflight, launch only training with first-step event,
then stop monitoring and rely on completion/failure notification.


## 2026-09-21T22:17:01+08:00 — motion pilot completion

Motion pilot training complete at 2026-09-21T22:17:01+08:00; reports: artifacts/reports/youtube_motion_pretrain_fixed_phrases_v17_20260921. No runtime promotion or protected test access.

## 2026-09-21T22:10:52+08:00 — corrected phrase training launched

Corrected-phrase paired training running, PID 71589; first optimizer step acknowledged. Only training detached; no polling. Reports: artifacts/reports/youtube_motion_pretrain_fixed_phrases_v17_20260921/. Completion/failure notification enabled.

## 2026-09-21 — user-reviewed PHRASES FIXED audited; corrected matched rerun ready

User confirms every retained video was checked against its folder phrase. All 685
fixed videos match original SHA256 content; 95 removed, zero new/edited/relabeled.
The admitted cache retains 232 local train / 200 validation, removing 55 training
clips and no validation clips. Train signers local01/03 and validation local02 remain
disjoint, with no content overlap. Retained landmarks/ranges/targets verified bitwise
equal. No GOOD_MORNING phrase training clips remain (20 validation clips remain).
253 fixed videos are outside the existing admitted cache; no aliases or classes added.
This records user semantic review, not independent machine transcript verification.

New script scripts/prepare_fixed_local_phrases_v17.py produced
 data/local/stage2_v17_grounded_phrases_fixed_20260921 and audit reports under
 artifacts/reports/local_phrases_fixed_audit_20260921. Runner now accepts corrected
phrase/report/model roots and reuses all four exact saved initial heads; only temporal
blocks differ between paired initializations. Trainer records source counts and emits
startup acknowledgement after its first optimizer step. Two seeds, 18 epochs per arm,
same frozen Stage1, decoder and all other sources. No SSL retraining or test access.
ASLLRP, NCSLGR, Citizen and SemLex were already included. Flores was previously tested
on 155 dev sentences with mixed transfer and no promotion; adding it here would change
the controlled question. O5S5 partial intervals need separate supervision, not blank gaps.

Validation: source hashes and cache equality checked; self-check and real MPS motion/CTC
backward passed; strict checkpoint loads passed; behavior path checked on 289 sequences
including 200 local and 20 each synthetic hold/repeat, edit counts independently checked.
Python compilation and diff whitespace checks passed. Reports/plan:
artifacts/reports/youtube_motion_pretrain_fixed_phrases_v17_20260921.
Next action: detach training only after first optimizer step; completion/failure macOS
notification, no polling, no runtime promotion. Prior pilot results remain historical.


## 2026-09-21 — local phrase semantic-label audit gap identified

User reports local phrase folders may mix different phrases. Traced label provenance:
`audit_stage2_phrase_sources_v17.py` assigns phrase=parent folder after ffprobe/hash
checks; `prepare_stage2_training_manifest_v17.py` assigns every clip the static
LOCAL_TARGETS sequence for that folder; `prepare_grounded_streaming_data_v17.py`
changes signer roles while retaining target_indices. Found source integrity, landmark,
face-cluster signer audits and a nine-clip rendering reference, but no evidence of a
corpus-wide per-video semantic transcript audit comparable to isolated cleanup.
This establishes unverified semantic labels, not yet the number of mislabeled clips.
Local WER/training conclusions are provisional; do not interpret this pilot as a
definitive negative pretraining result. Added report/current-state caveat. No data,
checkpoints or labels changed. Next safe action: clip-level visual transcript audit
before any new supervised phrase run or label-based evaluation.

## 2026-09-21 — motion pilot reviewed: mixed transfer, no promotion

All four runs completed and macOS notification command returned0. Local WER
baseline→pretrained46.85→48.89% (seed17321) and49.07→44.44% (seed17322);
ASLLRP54.17→62.50% and62.50→45.83%; isolated CTC exact84.22→84.00% and
82.96→84.00%. Only1/2 paired gates passed. Local duplicate output pairs58→82
and86→87. Synthetic hold exact6→6/20 and4→7/20; synthetic repeat exact4→6/20
and3→2/20. Conditional matched ASLLRP median sign-end-relative emission delay
0→0ms and0→66.7ms; these are not real-device latency measurements.
Masked reconstruction lost to causal last-observation control in both seeds on the
same endpoint-observed subset (0.01795/0.01661 vs0.01322/0.01322).
Verified saved connected error totals and temporal-only initialization differences.
Found and fixed a reporting-only adjacent-repeat bug: five OTHER-separated equal-gloss
pairs became adjacent after filtering. Full references contain zero true adjacent-repeat
examples. Added focused regression assertions; corrected behavior/comparison/REPORT files
without rerunning models or changing predictions/checkpoints/WER.
Files: `scripts/train_youtube_motion_pilot_v17.py`, report directory comparison, behavior,
REPORT and verification files; PROJECT_GROUND_TRUTH current decision updated.
Decision: no consistent benefit established; no promotion or further acquisition.
Next safe action: inspect objective/transfer limitations using retained data before a
new run. Official Citizen test stayed sealed.

## 2026-09-21T21:50:48+08:00 — motion pilot completion

Motion pilot training complete at 2026-09-21T21:50:48+08:00; reports: artifacts/reports/youtube_motion_pretrain_v17_20260921/. No runtime promotion or protected test access.

## 2026-09-21 — matched motion training launched

Foreground audit/preparation and real-device preflight completed. Behavior reporting
passed on289 sequences with independent edit-count checks; combined connected-source
train/validation signer sets are disjoint. Quality filtering retained6584 windows
from1166 clips (5811 SSL train/773 source-video-held-out windows). Detached only
training after an explicit first-optimizer-step handshake. PID 44365.
Two paired seeds17321/17322,8 SSL epochs and18 CTC epochs per arm. Completion/failure
reports and macOS notification are configured; no polling and no CLI resume retries.
See `artifacts/reports/youtube_motion_pretrain_v17_20260921/training_launch.json`.

## 2026-09-21 — foreground audit corrected; causal temporal transfer prepared

User corrected execution: do not detach audit, only training. Checked completed first
audit:1191 accepted/220 frame-count mismatches; notification command passed but CLI
resume failed with active-writer conflict. Removed obsolete detached audit runner.
The original audit stopped validating a clip at its first count discrepancy, so a
foreground re-audit now separates structural validity from temporal provenance:
1411/1411 structurally valid,168531 actual frames. Of220 count mismatches,203 are-1,
three are-2, one is+1 and13 are larger deficits. Quarantine all220 without modifying
raw data; do not claim corruption from the count mismatch alone.
Upstream PoseEstimation code confirms pixel XY geometry and reveals different single-
versus-two-hand assignment rules. New separate46-nodeXY+presence adapter reassigns by
pose wrists, omits ambiguous assignments, uses one torso scale and masks missing data.
No face/confidence/depth or frame rate is fabricated. Only the128-wide causal CTC
temporal blocks transfer; Apple Stage1 stays frozen. Use its pre-local-adaptation
orientation-robust ancestor to avoid inherited local signer leakage.
Main agent wrote `scripts/train_youtube_motion_pilot_v17.py`; shared aligned trainer
adds an optional strictly validated full-head initialization, preserving defaults.
Foreground MPS SSL/real-CTC backward and temporal-only transfer checks passed; PyTorch
CTC falls back to CPU. Phrase source signer splits checked disjoint. Two paired
seeds17321/17322,8SSL epochs then18identical supervised CTC epochs per arm; report
real-sequence errors and conditional delay, with synthetic held/repeat diagnostics.
No automatic promotion, acquisition, or protected test use. Training not yet launched
at this entry; consult the launch handshake and status artifact after notification.

## 2026-09-21 — connected-motion pilot authorized; audit runner prepared

Primary outcome is connected-sign recognition with isolated retention. User requested
detached execution, no polling/sleeps, reports, completion notification and session
resume. After user preferred self-written code, stopped the single audit subagent and
wrote the audit and runner in the main session. Files:
`scripts/audit_youtube_motion_pilot_v17.py`, `scripts/run_youtube_motion_pilot_v17.py`,
and `artifacts/reports/youtube_motion_pretrain_v17_20260921/PLAN.md`.
Audit missing-hand/nonfinite checks and runner report self-check passed; three real
clips validated275 frames. Both scripts compiled; `git diff --check` passed.
Direct MediaPipe substitution is unsupported: confidence, timing, chirality and face
mapping are not established. Apparent-scale depth is derivable, not true detector depth.
A separate source adapter and compatible temporal-block transfer remain conditional,
not disproven. No training or protected evaluation ran in this preparation phase.
Next action: full detached audit, then completion-triggered same-session review using
GPT-6 Astra medium; no polling, no automatic acquisition or promotion. Consult status
and completion artifacts only after notification or explicit user request.

## 2026-09-21 — downloads paused by user; 1,411-clip pilot frozen

Stopped managed session47842 (exit130); subsequent process inspection found no Python
download worker. Saved acquired_manifest.csv containing1,411 locally present members
and marked download_status.json paused_by_user. No more acquisition until the pilot
shows value. Proposed experiment compares matched supervised baselines with/without
masked-motion pretraining; first validate landmark quality and MediaPipe/Apple schema
compatibility. Captions supply no human sign boundaries. No training started.

## 2026-09-21 — public YouTube-ASL keypoints replace Databrary for transition pretraining

Found and verified an immediately downloadable CC BY 4.0 sentence-level ASL keypoint
corpus at LINDAT. Its ten range-addressable archives contain390,547 unique JSON
sequences (347.98GiB compressed). The train/dev annotations contain391,494 rows and
389,474 join to keypoints (99.48%);8,479 train and943 dev source-video IDs are
disjoint. Saved both metadata files, the complete compressed archive-member index and
three selectively extracted samples locally without downloading the full corpus.

The source solves connected-motion availability, not exact gloss timing: supervision
is English sentence translation, signer IDs are absent and the files are 2D MediaPipe
keypoints. Use it for masked temporal/motion pretraining on a selective train subset,
then fine-tune only the locked blank+100 decoder on trusted labeled data. Do not treat
caption words as aligned glosses. How2Sign landmarks remain the controlled secondary
source. Rejected the purported 3D Continuous ASL corpus because only2,000/98,196
referenced tensors are present. Report:
`artifacts/reports/free_continuous_asl_alternatives_20260921/README.md`.

Froze the selective pilot manifest at2,000 clips from2,000 unique train source-video
IDs. Each chosen clip has a non-empty caption,50–250 frames and is the candidate nearest
120 frames for its source video; video IDs are selected in deterministic SHA-256 order.
The pilot has240,585 frames across all ten shards. Expected transfer is1.91GB compressed
(1.78GiB) from the full-corpus mean; reserve about6GB for extracted JSON and workspace.
The earlier managed execution was externally terminated after22 validated clips
(55MB), before its first25-clip status checkpoint, so its Python failure handler did not
run. Restarted the same resumable transfer under macOS launchd label
`org.slt.youtube-asl-pilot`; the22 atomic outputs are retained. It prevents system sleep
and sends a macOS notification on ordinary success or failure. Do not poll it.
Managed session78480 later exited on a transient LINDAT DNS resolution failure. The
shared HTTP helper retried only three times over a few seconds. The pilot downloader now
reopens each shard up to12 times with bounded backoff, updates status after every clip,
and resumes all atomic outputs before retrying.
Managed session32891 reached493 clips but stopped advancing after18:27 while blocked in
a network read; its stale `running` status was incorrectly reported as live. Added a
backward-compatible30-second range-read timeout. A LaunchAgent attempt reported
`spawn failed` and was removed. The transfer now runs independently in detached native
screen session `11217.slt_youtube_asl_pilot`, resuming493 validated files.
That session remained alive but advanced only to496 before the remote range endpoint
stalled again. This confirms server throttling rather than local process loss. Replaced
the finite retry loop with indefinite network recovery capped at a five-minute cooldown
and added a one-second inter-clip delay. Restarted from496 atomic files in detached
screen session `13538.slt_youtube_asl_pilot`.
Root cause of the repeated apparent stall was then confirmed locally: the Python child
from screen11217 survived after its screen was closed, so it and screen13538 downloaded
the same output concurrently and amplified LINDAT throttling. Terminated every stale
Python/caffeinate worker, removed partial files, and started exactly one verified Python
worker under detached screen `15719.slt_youtube_asl_pilot`. Do not parallelize this
range endpoint; shard concurrency would recreate the failure.
Per explicit user request, monitored the repaired single-worker transfer through five
new atomic outputs. It advanced497→502 without retry/failure; status remained `running`
on `raw_keypoints_2.zip`. This establishes current forward progress before detaching.
Subsequent attempts to move the job into screen and Terminal proved unreliable and were
removed. Restored the original working managed-execution method as session40900 with
exactly one worker. After the remote endpoint's initial pause, continuous monitoring
verified five further atomic outputs,502→507, all on shard2 with `running` status.
Measured repository throughput directly: an8MiB sequential LINDAT range took23.07s
(363,558B/s), implying about28.5h for one37GB shard. Files1–493 had arrived in37m37s;
files507–514 then took11m25s after throttling. A user-requested three-reconnect burst
test moved only514→515, proving reconnects do not reset the present repository/IP
limit. A persistent-session exact-member prototype could not finish even the shard
central-directory lookup under the active throttle and was stopped; no downloader is
currently running. Further acquisition needs repository cooldown or a different public
network/IP, followed by exact-range verification before resuming.
After the user enabled a VPN, the persistent-session exact-range downloader passed a
five-file CRC/size/JSON gate515→520. The full resumable run then passed a second gate
520→538 in about30s after ZIP metadata opened. Acquisition is active in managed session
18670 with one worker; this confirms the prior public IP was the immediate throttle.

## 2026-09-21 — Databrary volume 1249 files are inaccessible to current account

Checked volume 1249 in the signed-in Brave session. Databrary lists session files at
the Authorized Users release level, but shows zero accessible session files for this
account; its participant view does not expose downloadable demographics. No files
were transferred and no access request was sent. The F13 public sample is already
local. Continue the EAF/demographics-first acquisition after the volume owner grants
the account access.

## 2026-09-21 — connected-ASL gap search selects ASL-Homework-RGBD

Current public and local sources were rechecked against the actual Stage-2 gap: RGB,
human per-sign onset/offset glosses, stable signer IDs and enough signers for a real
held-out split. ASL-Homework-RGBD is the only identified direct match:935 continuous
recordings from45 people (24 fluent,21 learners),1920x1080 RGB and ELAN tiers for
Signing Happening, exact gloss timing, nonmanuals and errors. Full data are restricted
to authorized Databrary users at volume1249; the public F13 sample is already local.
No legitimate anonymous mirror was found. Acquire EAFs/demographics before media and
prioritize fluent signers with verified locked100 overlap.

Downloaded and hash-verified the official full NCSLGR index and legacy SignStream
database bundle without media. The index has1,887 utterances/38 collections. Of76
previously screened target-bearing parents outside the local166-clip subset,59 front
video endpoints currently respond, representing50 unique videos/223,521,002bytes.
Media transfer was deliberately skipped because per-sign XML and complete participant
metadata still require a free DAI account; NCSLGR has only8 corpus participants and is
secondary to ASL-Homework. Existing local ASL STEM review already supplies111 approved
spans/31 glosses/18 participants, but only18 glosses have at least3 participants.
How2Sign, FLEURS-ASL, OpenASL and YouTube-ASL do not provide the required public human
locked-gloss timing. Report:
`artifacts/reports/continuous_asl_gap_search_20260921/README.md`.

## 2026-09-21 — source data retained; continuous preprocessing must be rebuilt

Audited the complete local Stage-2 path across1,160 ASLLRP clips, the confident-event
manifest, existing frontend ablations, all780 unique local phrase videos and the
signer-disjoint local phrase evaluation. ASLLRP is not low-resolution: every audited
clip is1280x720 or1280x960 at29.97fps, all produced features, and ASLLRP-other training
signs have99.85% median hand-node presence and92.48% at p10. The local phrase videos
are640x480 at30fps and remain usable, although landmark coverage varies by recording.

The current continuous observer discards evidence by detecting at a640px maximum and
sampling20fps. A controlled11-clip/22-token ablation improved WER from45.45% at
640px/source-rate to36.36% at1280px/source-rate;20Hz also lost an exact sequence.
Moreover,6,020/11,936 annotated events fail the current four/six-observation floors,
which disproportionately excludes short signs. Slowing cached sequences doubles frame
rows but not the69 independent source observations, so it cannot repair the loss.

The prior local phrase manifest is invalid as generalization evidence because all three
signers occur in both train and validation. The existing signer-disjoint split trains on
signers01/03 and validates on200 signer02 clips, yielding25.5% exact and37.04% WER; it
covers only15 glosses. Keep the source ASLLRP annotations and local phrases, re-extract
ASLLRP at source timing with1280px detection, preserve short events with masks, and use
only signer-disjoint validation. Citizen test stayed sealed and no training ran. Report:
`artifacts/reports/stage2_data_path_audit_20260921/`.

## 2026-09-21 — source/Luna comparison corrects the boundary-quality interpretation

Built a24-clip side-by-side HTML review with the original video, source and Luna
intervals, proportional timelines, seek/play controls, signed edge differences, the
exact24-frame Luna sheet and agreement filters. All24 video paths resolve and the page
renders correctly in a local browser.

The review and ASLLRP's published annotation convention correct an earlier overstatement:
the pilot does not establish that the source annotations are wrong. ASLLRP excludes
preparatory movement and release into the next sign from its linguistic sign interval
and may annotate final holds separately. Luna reviewers sometimes included a fuller
visible articulation or hold. The±100ms band was too strict to serve as an annotation
quality verdict, although the coherent detector still failed at±200ms, so tolerance
alone does not explain the model failure.

Confirmed defects remain downstream:21/1,160 derived ASLLRP crops clip at least one
annotation (22 occurrences, all mapped to OTHER), and the earlier context-window
objective ended every known training window before its annotated sign end. Comparison:
`artifacts/reports/luna_boundary_annotation_pilot_20260921/comparison.html`.

## 2026-09-21 — one-Luna-per-clip boundary pilot completed

Following user direction,24 train-only known ASLLRP signs were assigned to24 separate
Luna-low runs, exactly one visual reviewer per clip. Each reviewer saw a blind24-frame
timestamped contact sheet and the target gloss but not the source boundary timestamps.
All24 returned valid intervals:23 medium confidence and1 low. Four touch the first or
last review frame and are censored. After excluding low-confidence and censored rows,
19 remain as provisional annotations.

Against the reserved source timestamps, all24 have median absolute start/end differences
of68ms/145ms; only6/24 agree at both edges within100ms and16/24 within200ms. The19
provisional rows have55ms/128ms median differences,6/19 agreement within100ms and14/19
within200ms. Signed differences are inconsistent, so a global offset cannot reconcile
them. These single-reviewer Luna outputs are pseudo-labels, not verified ground truth;
do not scale them directly into training without a stronger reference or agreement gate.
Citizen test stayed sealed. Report and per-clip annotations:
`artifacts/reports/luna_boundary_annotation_pilot_20260921/`.

## 2026-09-20 — confident-only continuous supervision frozen

Created a derived manifest without modifying or copying source data. Starting from all
11,936 annotated events and the strict whole-sign audit, accepted5,331 mechanically
high-confidence intervals:1,017 known and4,314 explicit unknown. Each accepted interval
is source-crop complete when ASLLRP, non-overlapping, inside the raw clock, covered at
both annotated edges, at most1.20s, has at least six raw target observations, at least
80% target-frame hand visibility, and no target timestamp gap above80ms. Model
predictions were never used for filtering.

Continuous training coverage is61/100 locked classes;17 retain only one continuous
training signer. All65 classes appearing across train/validation are indexed in the
coverage map. O5S5 may provide exact positive cores and boundary edges but never
negative/background regions because its narrative annotation is incomplete. Annotation
gaps are explicitly not transition targets. The manifest records every accepted row
and a decision/exclusion reason for every source annotation; train/validation signers
remain disjoint, all paths resolve outside protected splits, and Citizen test stayed
sealed. Verification and report:
`artifacts/reports/confident_supervision_v17_20260920/`.

## 2026-09-18 — local storage inventory; no data mutation

Inspected all top-level `data/local/` entries, `data/raw_videos/` roots, and the
storage-heavy `artifacts/` trees by size, metadata/manifests, current code references,
and the relevant data-source, Stage-1, Stage-2, live-streaming, deployment, and
translation reports. `data/local/` occupies 52.84 GiB and `data/raw_videos/` 35.98
GiB. No files were deleted, moved, or rewritten.
`data/local/ASL_continuous_synthetic` is explicitly v17-incompatible and
provenance-only; `open_asl_alternatives_20260913/some` was rejected as predominantly
one-handed; `popsign_v17_archives` contains only an unfinished paused partial with no
extracted video; and `stage2_v17_grounded_signer_split_v2_auto` duplicates the original
dataset's 668 NPZ archives byte-for-byte. These are cleanup candidates pending explicit
per-path approval. `data/raw_videos/ASL VIDEOS` is 32.10 GiB and remains a source for
local-deep-clean preparation and legacy extractors; `PHRASES` is 1.27 GiB and remains
an active local phrase source. `openpose_output` is 2.55 GiB with no current v17 code
reference, and `NUMBERS` is 39 MiB with no current code reference; both are candidates
pending approval. Canonical Citizen, ASLLRP, 2M-Flores, STEM, SemLex, phone, portrait,
and current Stage-2 inputs remain preserved. Artifact storage was also classified:
`models` 18.20 GiB contains the accepted runtime plus failed/research checkpoints;
`model_assets` 11.47 GiB contains current v17 extractors and legacy v16 assets;
`generated` 6.51 GiB contains reproducibility caches, packaged datasets, build/env
outputs, and failed probes; `reports` 2.24 GiB remains provenance/evidence; `archives`
1.16 GiB contains legacy recovery bundles; `coreml` 367 MiB contains current v17
runtime packages; and `mobile_export` 970 MiB is legacy export evidence. None was
mutated or deleted.

Next safe action: ask the user to approve exact candidate paths, including whether
legacy v16 compatibility/recovery and failed-experiment reproducibility must be kept;
perform no deletion until each path is approved.

---

## 2026-09-20 — second experimental/legacy sweep; candidates and recovery sources only

Performed a second read-only sweep of `data/local/`, `data/raw_videos/`,
`artifacts/models/`, `artifacts/generated/`, `artifacts/model_assets/`,
`artifacts/archives/`, and `artifacts/mobile_export/`. The sweep used directory sizes,
top-level manifests, source/provenance records, fixed-string references from active
code/tests, and the experiment READMEs. No dataset, checkpoint, report, archive, cache,
or environment was deleted, moved, or rewritten. A candidate below is not approval to
delete it; reports remain evidence and exact-path approval is still required.

The current v17 runtime boundary remains protected: Citizen100, local-deep-clean,
ASLLRP, O5S5/RWTH, 2M-Flores, STEM/SemLex, phone/portrait inputs, current Stage-2
inputs, accepted v17 checkpoints/Core ML packages, and all reports were not added to the
cleanup list. `stage1_direct_translation_20260917` is also retained for the current
English comparison: keep `epoch_01_before_resume.pth`, `epoch_13_before_internal_resume.pth`,
and `latest.pth` until that experiment is explicitly closed.

### Previously identified data candidates, now with recovery locations

These are the earlier candidates carried forward with exact recovery notes:

- **D01 — legacy synthetic arrays, about 3.8 GiB:**
  `data/local/ASL_continuous_synthetic/*.npy`. The v17 source manifest marks this
  `[32,61,10]` corpus `v17_compatible: false`, `role: provenance_only_do_not_train_v17`.
  Keep `manifest.json` if the history is wanted; the arrays are not a valid v17 input.
  The old corpus was generated by `scripts/generate_continuous_training.py` from
  `data/raw_videos/ASL VIDEOS`. A future valid replacement must be regenerated from
  current train-only v17 isolated archives through `data/local/stage2_v17_synthetic`,
  not downloaded or relabeled from these arrays.

- **D02 — visually rejected SoMe ASL, about 2.2 GiB:**
  `data/local/open_asl_alternatives_20260913/some/`. The user's review rejected it as
  predominantly one-handed; it must not enter a training manifest. Recovery is pinned
  by the local per-video `*.source.json` files and the public OSF nodes:
  transcripts `https://osf.io/5ws8h/`, recordings `https://osf.io/4k95c/`, with each
  downloaded recording's exact `https://osf.io/download/<file-id>/` URL preserved locally.

- **D03 — unreferenced OpenPose/How2Sign output, about 2.55 GiB:**
  `data/raw_videos/openpose_output/`. The local metadata includes
  `how2sign_realigned_test.csv`; no current v17 code consumes this root. Recovery is
  through the official How2Sign page `https://how2sign.github.io/`, the local source
  clone `artifacts/generated/how2sign_official_repo/`, and the bounded v17 acquisition
  source card `https://huggingface.co/datasets/martinctl/how2sign-asl-clips`.
  The exact URL that produced every old OpenPose JSON was not preserved, so this remains
  conditional until that provenance is no longer needed.

- **D04 — paused PopSign partial, about 1.05 GiB:**
  `data/local/popsign_v17_archives/`. It contains incomplete archive parts and no
  extracted video. Recovery is disk-safe, one sign/split archive at a time from
  `https://signdata.cc.gatech.edu/data/popsign_v1_0/game/{split}/{sign}.tar`, with
  metadata at `https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/` and the
  CC BY 4.0 license. Do not resume a wholesale PopSign download.

- **D05 — duplicate signer-split feature archives, about 17 MiB:**
  `data/local/stage2_v17_grounded_signer_split_v2_auto/`. Its 668 NPZ archives are
  byte-identical to `data/local/stage2_v17_grounded_signer_split/`; only the manifest
  lineage differs. Recovery is the original local directory and the report/history
  entry at `docs/ground_truth/stage2-ctc/log.md`.

- **D06 — unreferenced raw number clips, about 39 MiB:**
  `data/raw_videos/NUMBERS/`. No exact per-file provenance or current code consumer was
  found. The broad legacy recovery tool is `src/download_asl.py`, whose known metadata
  sources are WLASL (`https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json`),
  MS-ASL (`https://raw.githubusercontent.com/iamgarcia/msasl-video-downloader/master/MSASL_classes.json`),
  and SignASL (`https://www.signasl.org/sign`); exact recreation of these number files
  is not guaranteed without the original per-video URLs.

- **D07 — rejected Uni-Sign/How2Sign challenger material, about 3.65 GiB:**
  `data/local/unisign_asl_baseline_20260916/` and its retained report
  `artifacts/reports/unisign_asl_baseline_20260916/`. The report rejects this released
  checkpoint as a Stage-2/Reel replacement and records no training change. Recovery is
  the official code `https://github.com/ZechengLi19/Uni-Sign`, released weights
  `https://huggingface.co/ZechengLi19/Uni-Sign`, and validation mirror
  `https://huggingface.co/datasets/aipieces/How2Sign`. Keep the report if the challenger
  comparison is part of the project history.

- **D08 — small unreferenced validation/hand-cache probes:**
  `data/local/stage2_v17_online16_validation_multimodal/` (~36 MiB),
  `data/local/stage2_v17_overlap32_validation_multimodal/` (~11 MiB),
  `data/local/stage2_v17_online16_validation_hand/`,
  `data/local/stage2_v17_online16_validation_frozen/`,
  `data/local/stage2_v17_overlap32_validation_hand/`,
  `data/local/stage2_v17_overlap32_validation_frozen/`,
  `data/local/stage2_v17_asllrp_other_hand_mobileclip2/` (~189 MiB),
  `data/local/stage2_v17_2m_flores_hand_mobileclip2/` (~112 MiB), and
  `data/local/stage2_v17_asllrp_segmented_train_hand_mobileclip2/` (~30 MiB).
  These are derived caches, not source datasets. Recovery is from the parent ASLLRP/
  2M-Flores landmark roots with the existing MobileCLIP2 extraction scripts and
  `artifacts/generated/mobileclip2_env`; do not remove the parent raw/landmark sources.

- **D09 — unreferenced legacy clone/sample:**
  `data/local/SAM-SLR-v2-main/` (~0.5 MiB) is recoverable from
  `https://github.com/kokoab/sign_language_translation_system.git`; `data/local/demo_npy/`
  (~34 MiB) has no reliable source URL or current consumer and should remain conditional
  until its provenance is either accepted as disposable or documented.

### Newly identified experimental outputs

- **E01 — failed joint-CTC family, about 7.2 GiB including its data cache:**
  `artifacts/models/joint_ctc_v17_20260914/`,
  `artifacts/models/joint_ctc_repair_v17_20260914/`,
  `artifacts/models/joint_ctc_frames_v17_20260915/`,
  `artifacts/models/joint_ctc_balanced_v17_20260915/`,
  `artifacts/models/joint_ctc_anchor_v17_20260915/`,
  `artifacts/models/joint_ctc_aligned_v17_20260915/`, and
  `artifacts/generated/joint_ctc_v17_20260914/` (~2.17 GiB). All failed the continuous
  promotion gates; the reports explicitly preserve the failure and checkpoint
  provenance. Recovery is from the report-local plans/runners, current ASLLRP/O5S5/
  Citizen train-only caches, and `active/v17/joint_ctc_*.py`; no external dataset is
  required for replay. This is the largest new conditional cleanup group.

- **E02 — failed Stage-1 window checkpoints and derived observations, about 0.68 GiB
  of checkpoints plus ~53 MiB of local observations:**
  `artifacts/models/stage1_window_v17_seed17111/`,
  `artifacts/models/stage1_window_o5s5_v17_seed17111/`,
  `data/local/stage1_window_v17/`, and
  `data/local/stage1_window_o5s5_v17/`. No checkpoint passed the promotion gates.
  The reports, manifests, and evaluation JSONs remain usable evidence. Recovery is from
  `active/v17/train_stage1_window_v17.py`,
  `scripts/evaluate_stage1_window_v17.py`, the ASLLRP/O5S5 source caches, and the
  report commands; no new public download is required.

- **E03 — unpromoted live/revisable/adaptation checkpoints, about 0.5 GiB:**
  `artifacts/models/stage2_v17_live_adapt_v1/`,
  `artifacts/models/stage2_v17_revisable_core_v1/`,
  `artifacts/models/stage2_v17_revisable_v1/`,
  `artifacts/models/stage2_v17_transition_adapt_v1/`,
  `artifacts/models/stage2_v17_transition_adapt_v2/`,
  `artifacts/models/stage2_v17_other_preservation_v1/`, and
  `artifacts/models/stage2_v17_other_preservation_v2/`. These are not the accepted
  `stage2_v17_transition_repair_v3` runtime; reports say to retain the current default
  and treat these as opt-in research. Recovery is from the report-local commands and
  `data/local/stage2_v17_live_matched_v1/` or
  `data/local/stage2_v17_transition_adapt_v1|v2/`.

- **E04 — failed/duplicate visual-speech and RGB packages, about 2.45 GiB:**
  `artifacts/generated/kaggle_visual_speech_pixel_dataset/` contains a base Auto-AVSR
  checkpoint and frozen mouth checkpoint duplicated byte-for-byte by canonical files
  under `artifacts/model_assets/models/auto_avsr/` and
  `artifacts/models/stage1_v17_visual_speech_auto_avsr_mouth_frozen/`, plus a package
  tar. `artifacts/generated/kaggle_citizen100_rgb_trainval_v1/` is an unuploaded
  train/validation-only RGB package. The failed Kaggle attempt remnants
  `artifacts/generated/kaggle_visual_speech_pixel_failed_v1/` through `_v5/`,
  `.../kaggle_visual_speech_pixel_pull_v4/`, and
  `.../kaggle_visual_speech_pixel_kernel/` are metadata/small leftovers.
  Unreferenced visual-speech ablation checkpoints are
  `artifacts/models/stage1_v17_visual_speech_auto_avsr_full_face_frozen/`,
  `artifacts/models/stage1_v17_visual_speech_auto_avsr_mouth_layer4_finetuned/`,
  `.../mouth_layer4_mild/`, `.../mouth_layer4_mild_fixed_ema/`, and
  `.../mouth_layer4_mild_fixed_ema_patience3/`; keep the canonical mouth/lower-face
  teacher and learned checkpoints. Recovery is the Auto-AVSR source
  `https://github.com/mpc001/auto_avsr`, the retained Citizen train/validation data,
  and the private Kaggle package IDs recorded in
  `docs/ground_truth/stage1-architecture/log.md`.

- **E05 — unreferenced Stage-1 ablation checkpoints, approximately 0.6 GiB total:**
  `artifacts/models/stage1_v17_asllrp_core_adapt_smoke_v1/`,
  `stage1_v17_asllrp_core_local_replay_smoke_v1/`,
  `stage1_v17_citizen_semlex_augmented/`,
  `stage1_v17_citizen_semlex_balanced/`,
  `stage1_v17_citizen_semlex_full_clean_balanced_d384/`,
  `stage1_v17_citizen_semlex_full_clean_balanced_seed3407/`,
  `stage1_v17_citizen_semlex_full_clean_flat_graph_residual/`,
  `stage1_v17_citizen_semlex_full_clean_graph_parts/`,
  `stage1_v17_citizen_semlex_full_clean_hands_only/`,
  `stage1_v17_citizen_semlex_full_clean_labelsmooth005/`,
  `stage1_v17_citizen_semlex_full_clean_phonology020/`,
  `stage1_v17_citizen_semlex_full_clean_source40_60/`,
  `stage1_v17_citizen_semlex_full_clean_source60_40/`,
  `stage1_v17_citizen_semlex_full_clean_supcon005/`,
  `stage1_v17_citizen_semlex_full_clean_supcon005_decay12/`,
  `stage1_v17_citizen_semlex_local_tiera10_balanced/`,
  `stage1_v17_face_only_landmarks_d128/`,
  `stage1_v17_hand_spatial_from_multisource_compact/`,
  `stage1_v17_hand_spatial_residual_from_multisource/`,
  `stage1_v17_local_deep_clean_face_masked_mps_v1/`,
  `stage1_v17_local_deep_clean_mps_v1/`,
  `stage1_v17_mouth_rgb_citizen/`,
  `stage1_v17_mouth_rgb_mobilenet_citizen/`,
  `stage1_v17_phrase_adapt_reel_smoke_v1/`,
  `stage1_v17_reel_emission_v1/`, `stage1_v17_reel_emission_v2/`, and
  `stage1_v17_unified_phrase_adapt_reel_v1/`. These are conditional candidates only:
  reports preserve their measured comparisons, and the source data can be rebuilt from
  `data/local/citizen100_v17/`, `data/local/semlex_citizen100_train_audit/`, and
  `data/local/local_deep_clean_v17/` using the existing v17 trainers.

- **E06 — unreferenced Stage-2 ablation checkpoints, approximately 1.0 GiB total:**
  `artifacts/models/stage2_v17_2m_asllrp_mix_0p25_baseline/`,
  `.../stage2_v17_2m_asllrp_mix_0p50_baseline/`,
  `.../stage2_v17_2m_asllrp_mix_0p75_baseline/`,
  `.../stage2_v17_2m_asllrp_other_ctc_combined_v1/`,
  `.../stage2_v17_2m_asllrp_other_ctc_conservative_v1/`,
  `.../stage2_v17_2m_asllrp_other_ctc_mix0p5_v1/`,
  `.../stage2_v17_2m_flores_aux_ctc_cpu_diagnostic/`,
  `.../stage2_v17_2m_flores_aux_ctc_pilot/`,
  `.../stage2_v17_2m_flores_aux_ctc_pilot_cpu/`,
  `.../stage2_v17_2m_flores_partial_ctc_conservative_diagnostic/`,
  `.../stage2_v17_2m_flores_partial_ctc_cpu_diagnostic/`,
  `.../stage2_v17_2m_flores_partial_ctc_pilot/`,
  `.../stage2_v17_2m_flores_temporal_pretrain_smoke/`,
  `.../stage2_v17_accuracy_repair_mps_smoke/`,
  `.../stage2_v17_accuracy_repair_smoke/`,
  `.../stage2_v17_accuracy_repair_smoke_cpu/`,
  `.../stage2_v17_asllrp_other_ctc_conservative_single_v1/`,
  `.../stage2_v17_balanced_multivoice_pilot_a/`,
  `.../stage2_v17_balanced_multivoice_scratch_v1/`,
  `.../stage2_v17_balanced_multivoice_smoke/`,
  `.../stage2_v17_general_selector_distill_boost_smoke/`,
  `.../stage2_v17_general_selector_distill_pilot_v1/`,
  `.../stage2_v17_general_selector_distill_smoke/`,
  `.../stage2_v17_identity_anchored_smoke/`,
  `.../stage2_v17_landmark_cascade_preview_smoke/`,
  `.../stage2_v17_multivoice_adaptation_v1/`,
  `.../stage2_v17_multivoice_conservative_v2/`,
  `.../stage2_v17_signer_voice_context_adapted_pilot_v1/`,
  `.../stage2_v17_signer_voice_context_adapted_w0p75_pilot_v1/`,
  `.../stage2_v17_signer_voice_context_adapted_w1p0_pilot_v1/`,
  `.../stage2_v17_signer_voice_context_adapted_w1p25_pilot_v1/`,
  `.../stage2_v17_signer_voice_context_adapted_w1p5_pilot_v1/`,
  `.../stage2_v17_temporal_context_student_v1/`,
  `.../stage2_v17_temporal_replay_confirm_synth10/`,
  `.../stage2_v17_temporal_replay_confirm_synth10_seed12703/`,
  `.../stage2_v17_temporal_replay_screen_synth00/`,
  `.../stage2_v17_temporal_replay_screen_synth10/`,
  `.../stage2_v17_temporal_replay_screen_synth20/`,
  `.../stage2_v17_temporal_replay_screen_synth30/`,
  `.../stage2_v17_temporal_selector_oversample_synth10_v1/`,
  `.../stage2_v17_unified_ctc_smoke/`,
  `.../stage2_v17_unified_ctc_v2_smoke/`,
  `.../stage2_v17_unified_ctc_v3_phase/`, and
  `.../stage2_v17_unified_ctc_v3_phase_smoke/`. These are not current defaults;
  recovery is from the retained ASLLRP/2M-Flores/Citizen train-only features and the
  corresponding Stage-2 report commands. Current selector/repair/runtime directories
  are excluded.

- **E07 — failed causal evidence v2 output:**
  `artifacts/models/continuous_evidence_v17_causal_v2/` and
  `artifacts/generated/continuous_evidence_v17_causal_v2/` (~142 MiB cache). The live
  default remains `continuous_evidence_v17_v1`; recovery is the existing
  `active/v17/train_continuous_evidence_v17.py` plus the retained v1 cache and current
  live development inputs.

### Legacy artifact bundle and rebuildable generated material

- **L01 — legacy v16 feature/data bundle, about 5.16 GiB:**
  `data/local/ASL_landmarks_apple_vision/`,
  `data/local/ASL_landmarks_float16/`,
  `data/local/ASL_hand_crops_av/`,
  `data/local/ASL_phrases_apple_vision/`,
  `data/local/ASL_phrases_apple_vision_fixed/`, and
  `data/local/ASL_phrases_reextracted/`. These use the old `[32,61,10]` contract and
  are incompatible with v17. Recovery is from the retained
  `data/raw_videos/ASL VIDEOS/` and `data/raw_videos/PHRASES/` using
  `scripts/extract_apple_vision.py`, `src/extract.py`, and the legacy v16 scripts.
  `src/download_asl.py` documents the mixed WLASL/MS-ASL/SignASL/YouTube sources; it
  does not preserve an exact per-file source ledger.

- **L02 — legacy v16 model/export/recovery bundle, conditional on dropping v16:**
  `artifacts/model_assets/models/output/`, `output_joint/`,
  `output_stage2_v13/`, `output_stage2_v14/`, `output_stage2_v15/`,
  `output_stage2_v15_fixed/`, `output_stage2_v15_reextracted/`, `output_v12/`,
  `output_v13/`, `output_v14/`, `output_v15_clean/`,
  `stage1_joint.pth`, `stage1_velocity.pth`, `stage1_bone.pth`,
  `stage1_angle.pth`, `stage2_ctc.pth`,
  `artifacts/model_assets/weights/SLT_Stage1_Results/`,
  `artifacts/model_assets/weights/results/`, and
  `artifacts/mobile_export/artifacts/`. Recovery is either from the retained
  `artifacts/model_assets/models/all_models.tar.gz` or by retraining/re-exporting the
  old pipeline from `data/raw_videos/ASL VIDEOS/`. The recovery archives themselves,
  `artifacts/archives/src_v16.zip` and
  `artifacts/archives/ASL_landmarks_float16.zip`, are separate candidates; do not
  remove an old directory and its only archive in the same batch unintentionally.
  Keep `artifacts/model_assets/weights/slt_final_t5_model/`, which is still referenced
  by the v17 Stage-3 reference replay, and keep current Auto-AVSR, MediaPipe, MoViNet,
  and MobileCLIP2 assets.

- **G01 — rebuildable upstream clones, about 230 MiB total:**
  `artifacts/generated/dsta_slr_upstream/` →
  `https://github.com/hulianyuyy/DSTA-SLR.git`,
  `artifacts/generated/shubert_upstream/` →
  `https://github.com/ShesterG/SHuBERT.git`,
  `artifacts/generated/how2sign_official_repo/` →
  `https://github.com/how2sign/how2sign.github.io.git`,
  `artifacts/generated/siformer_upstream/` →
  `https://github.com/mpuu00001/Siformer.git`, and
  `artifacts/generated/lgf_slr_upstream/` →
  `https://github.com/MeiqiZhang7/LGF-SLR.git`. These are source clones, not training
  datasets; each local Git remote was recorded above for recovery.

- **G02 — rebuildable build/environment/comparison outputs:**
  `artifacts/generated/build/` (~200 MiB PyInstaller build),
  `artifacts/generated/portrait_capture_v17_orientation_obj/` (~132 MiB Swift/Xcode
  object cache), `artifacts/generated/portrait_capture_v17_orientation_build/`,
  `artifacts/generated/portrait_capture_v17_derived/`,
  `artifacts/generated/coreml_v17_comparison/`,
  `artifacts/generated/kaggle_cli_env/`,
  `artifacts/generated/msasl_download_env/`, and
  `artifacts/generated/mobileclip2_runtime/`. These are not source datasets. Recovery
  is from the project build/setup scripts, `scripts/build_portrait_iphone_eval_v17.py`,
  the existing MobileCLIP2 environment, or the corresponding downloader script.
  Keep `artifacts/generated/mobileclip2_env/`, which is still the active extractor
  environment; the orientation simulator release/input trees remain protected until
  the portrait-iPhone gate is closed.

- **G03 — unreferenced Kaggle challenger/probe outputs:**
  The material top-level candidates are
  `artifacts/generated/kaggle_stage1_challengers_v2/`,
  `kaggle_stage1_challengers_v2_download_probe/`,
  `kaggle_stage1_partaux_w020_kokoab_pull_v2/`,
  `kaggle_stage1_partaux_w020_kokoab_result_v2/`,
  `kaggle_stage1_articulated_pose_kokoab_result_v1/`,
  `kaggle_stage1_static_hand_kokoab_result_v1/`,
  `kaggle_stage1_masked_pose_kokoab_result_v1/`,
  `kaggle_stage1_partmix_dataset/`,
  `kaggle_stage1_partwise_bone_kokoab_pull_v1/`,
  `kaggle_stage1_orientation_asllvd_pull_v1_superseded/`,
  `kaggle_stage1_orientation_asllvd_pull_v2_superseded/`,
  `kaggle_stage1_orientation_asllvd_pull_v3/`,
  `kaggle_stage1_orientation_canonical_pull_v1/`,
  `kaggle_stage1_attention_score_mix_kokoab_result_v1/`,
  `kaggle_stage1_temporal_gate_kokoab_result_v1/`,
  `kaggle_stage1_partwise_seed3407_kokoab_result_v2/`,
  `kaggle_stage1_bone_kokoab_pull_v1/`,
  `kaggle_stage1_hand_angle_kokoab_result_v1/`, and
  `kaggle_stage1_partmix_kokoab_pull_v3/`, plus their small overlay, pull, kernel,
  metadata, and failed-probe siblings under `artifacts/generated/`. These packages are
  reproducible from the canonical Citizen/SemLex train-only inputs and the Kaggle
  preparation scripts. The current local-deep-clean package and its code/kernel
  directories are protected and are not in this candidate group.

The following were deliberately not marked as cleanup candidates despite being
experimental: current text-to-sign/transition packages, current MoViNet/MobileCLIP2
assets, external RIT evaluation material, continuous acquisition source material,
English comparison inputs, and reports. They still have active future-use or provenance
references. No deletion or storage mutation was performed.

Next safe action: review candidate IDs D01–D09, E01–E07, L01–L02, and G01–G03 one
path at a time. Before any deletion, recheck the exact path, preserve the report and
recovery source, and keep at least one approved recovery copy for legacy v16 material.

Validation: `git diff --check` passed after the sweep; this turn changed only this log
and did not mutate any dataset, checkpoint, report, archive, cache, or environment.

## 2026-09-20 — measured cleanup-candidate storage; no deletion

Measured the logged candidate paths directly with `du -sk` (GiB = 1,048,576 KiB):

- D01–D09 data candidates: **13.751 GiB**.
- E01–E04 and E07 experimental outputs: **11.287 GiB**.
- E05 Stage-1 ablations: **0.616 GiB**.
- E06 Stage-2 ablations: **1.046 GiB**.
- L01–L02 legacy data/model/export directories: **12.937 GiB**.
- L02 separate recovery archives `artifacts/archives/src_v16.zip` and
  `artifacts/archives/ASL_landmarks_float16.zip`: **1.155 GiB**.
- G01–G02 rebuildable clones/builds/environments: **0.741 GiB**.
- G03 Kaggle challenger/probe material, excluding protected local-deep-clean dirs:
  **0.699 GiB**.

The combined candidate total is **44,285,248 KiB / 42.234 GiB**. This is a maximum
conditional cleanup estimate, not an approval: reports, accepted v17 runtime assets,
current source datasets, and the protected local-deep-clean Kaggle package remain
excluded. The first aggregation wrapper had a shell-quoting parse error; the direct
path-by-path measurement completed successfully with no filesystem mutation.

Next safe action: approve exact paths or IDs before any deletion; keep reports and at
least one explicitly approved recovery copy for legacy v16 material.

---

## 2026-09-14 — O5S5 exact positives materialized with Apple Vision

Resolved the O5S5/Citizen label gate without English normalization or visual guessing.
O5S5 `RightHand_IDg`/`LeftHand_IDg` values use ASL Signbank IDs; the frozen Citizen
ASL-LEX codes resolve through the official ASL-LEX table to `SignBankAnnotationID`.
Exact equality admits 386 hand-tier events across 53 locked classes. One-to-one
cross-hand overlap pairing removes 130 duplicate two-hand copies, leaving 256 unique
positive occurrences. Nine frozen classes have no ASL-LEX Signbank ID and are rejected.
LG is reserved as a whole validation signer; CK, Doug Ridloff, JAH, LR and RD are
training-only.

Replayed all six original videos through the existing live Apple Vision observer and
wrote 26,079 timestamped raw observations under
`data/local/stage1_window_o5s5_v17/raw_observations/`. Apple Vision hands occur in all
256 exact targets. The initial long-video run retained every resized pixel frame and
reached an 18.6 GB footprint; added opt-in pixel discard after detection, preserving
frame geometry and reducing the run to roughly 0.5 GB. RD reports 14,972 container
frames but OpenCV decodes 14,965; the existing seven-frame tail exclusion is pinned,
and the last annotation ends 0.91 seconds before file duration.

Stage-1 context loading now accepts incomplete rows as positive-only while continuing
to derive background solely from `all_signs_annotated=true` rows. The ready combined
ASLLRP+O5S5 manifest yields 6,819 training context windows across 73 classes plus 494
verified background windows. O5S5 contributes 1,005 train windows across 49 classes
and 318 signer-held-out validation windows across 27 classes, with zero O5S5 background.
All preparation, Apple Vision, loader and provenance reports are in
`artifacts/reports/o5s5_citizen100_v17/`. No model training or protected-test access.

---

## 2026-09-13 — better no-account continuous ASL acquired and verified

User visually rejected SoMe ASL as predominantly one-handed and unsuitable; it is
now explicitly excluded from training while its downloaded files remain preserved.
Acquired every O5S5 narrative with a currently public ELAN transcript: six paired raw
videos/EAFs, six signers, 2,586,024,974 video bytes, 21.73 minutes, 60,944 fully
decoded frames from 540p through 4K, and 3,959 timed hand annotations (2,601 right,
1,358 left). The EAFs contain 591 distinct gloss strings; 34 exactly match frozen
Citizen raw-label strings across 290 tier events. All annotation endpoints fit their
paired video durations. Visual QA confirms frontal upper-body framing and both hands
across all six sources. Gallaudet publishes O5S5 open access under CC BY-NC-SA 4.0;
it is a research/noncommercial auxiliary-temporal training source, while direct
locked-head use still requires exact ASL-LEX visual-variant review.

Also acquired all 201 official RWTH-BOSTON-104 MPEG videos plus 42 corpus, lexicon,
language-model and readme files. All 15,746 frames decode; the videos total 80,759,696
bytes and 525.78 seconds. Sentence annotations contain 888 tokens and 113 gloss
strings from three signers; 19 frozen raw-label strings occur 148 times. The official
161/40 split reuses all three signers and the 336x312 grayscale video is low-resolution,
so RWTH is supplemental training-only sequence supervision, not generalization evidence.

Created a 290-row O5S5 variant-review manifest and a 201-row RWTH sentence manifest.
Initial whole-archive RWTH transfer was stopped after finding that the 685 MB archive
mostly duplicated published frames; its partial is preserved, and the 80.8 MB official
individual videos were verified instead. Initial serial O5S5 transfers were slow;
public byte-range resumes completed the same official files. Two guessed Drive IDs
returned clean 404s before IDs were extracted from the official folder listing; no
incorrect payload was retained. Final acquisition audit, hashes, timing checks,
full-frame decodes, manifest counts, Python compilation and visual contact sheet all
pass. Report: `artifacts/reports/open_asl_alternatives_20260913/README.md`.

## 2026-09-13 — resumed blocker threshold reached (3/3)

Revalidated goal status, missing evidence and unsent inquiries. Third consecutive
resumed turn with no new access, send authorization or account. Previous turn was
no progress; no live acquisition requires waiting. Marked goal blocked again under
the fresh resumed audit. Full data remains unacquired; preserve all files and resume
when external access or user authorization changes. No messages or purchases made.

## 2026-09-13 — resumed access blocker unchanged (2/3)

Rechecked active goal and unsent access requests. No user authorization/account or
new corpus access arrived. Previous resumed turn was no progress, not a verified
wait; no acquisition process is pending. Same external-access/evidence blocker,
second consecutive resumed turn. No outgoing messages or repeated downloads.

## 2026-09-13 — resumed acquisition, access blocker unchanged (1/3)

Goal service reports active after resumption. Checked the saved requirements and
unsent access inquiries; no new authorization, account or corpus access was supplied.
Previous turn was no progress toward acquisition, not a live-process wait. The
remaining step still requires external access; no messages sent or downloads restarted.
Fresh resumed blocker audit starts at one; do not mark blocked again before three.

## 2026-09-13 — full data-sufficiency goal blocked on external access and evidence

Revalidated DATA_NEEDED.md and the three unsent ACCESS_REQUESTS.md inquiries.
No acquisition or verification process remains live. The same missing external
access, exact-variant review and independent phone evidence has persisted across
three consecutive goal turns. The previous turn made progress by preparing concrete
requests; this turn found no further authorized action that resolves these gaps.
An automatic continuation is not permission to send email or institutional access.
Marked the goal blocked, not complete, to stop unproductive automatic continuations.
Resume when user supplies send authorization/account or authorized corpus access;
full completion still requires actual files and the remaining coverage evidence.
Preserved all acquired files and the rejected partial download. No messages sent,
training started or protected tests accessed. Documentation diff check passed.

## 2026-09-13 — concrete corpus access inquiries prepared, not sent

Rechecked remaining public source leads. How2Sign CVPR2021 supplementary material
explicitly describes human ELAN annotation collection, but current public download
page does not expose those files. This justifies a targeted availability request;
it does not establish access or completeness. Current RIT lab downloads page gives
matt.huenerfauth@rit.edu as the CUNY corpus contact, superseding old CUNY email.
Prepared three exact inquiries in acquisition report `ACCESS_REQUESTS.md` and a
verified 100-row `access_request_vocabulary.csv` containing frozen raw labels and
ASL-LEX codes. Initial CSV draft used a nonexistent alias; corrected to manifest's
citizen_asl_lex_code and asserted every value before handoff. No messages sent,
accounts created, institutional affiliation asserted, terms accepted or purchases.
Next available access step requires explicit send authorization and a user-owned
sending account; institutional eligibility, expert variant review and independent
phone recordings remain unresolved. Goal active, not complete. git diff --check
passes. Previous goal turn classified as progress, not a repeated no-progress turn.

## 2026-09-13 — broader acquisition: two annotation leads fail completeness check

User asks for all needed data and MediaPipe conversion. Added acquisition report
`DATA_NEEDED.md` with full requirements and explicit remaining gaps. Épée XYZ
cannot recover Apple detector confidence/failures; no fake conversion was created.
Current MoLo provider exposes 26 MP4s; exact names match systems and interview EAFs.
Started official 2.14 GB interview transfer, then tier inspection found only one
hand gloss across four EAFs (mostly English translations). Stopped verified process
19664; partial file preserved and excluded from acquisitions. Initial aggregate
check failed on empty hand tiers; direct XML inspection established the cause.
Downloaded Daily Moth research EAF: 32 first-person-reference marks, not full gloss
labels. Raw sample downloaded (75,053,731 bytes); publisher MD5 matches and local SHA256
is recorded in its sources.json. All 11,166 frames decoded; annotation times fit the 372.57-second video. Neither
lead fills complete targets. Final inventory assertions and git diff --check pass.
How2Sign raw training clips alone exceed free disk and official annotations are
English translations. Re-fetched RIT/CUNY official access pages (HTTP 200); CUNY
explicitly requires a publisher email request. Saved both pages in the report.
Full RIT/CUNY access restrictions remain. Goal is active;
next action is to verify pending sample transfer and pursue complete timed labels,
exact variants and independent phone recordings. No training/test access occurred.

## 2026-09-13 — continuous ASL acquisition completed and verified

Acquired files under `data/local/continuous_asl_acquisition_20260912/`; inventory,
source URLs, hashes, licenses and runnable checks are in
`artifacts/reports/continuous_asl_acquisition_20260912/README.md`.
Épée: 1,200 MediaPipe arrays plus 1,200 timed JSONs, six source signer IDs,
70.86 minutes, 3,788 gloss tokens, 68/100 exact frozen raw-label strings,
83 repeated-label clips and 244 segments over 1.07 seconds. All 2,400 publisher
hashes match the pinned revision; arrays/timings pass structural checks.
These are pose-only data, not Apple Vision input or visually approved variants.
MoLo003 systems: official OSF original MP4 (926,530,042 bytes, 17.29 minutes),
two matching original EAFs with 1,517 timed hand annotations, and two 640x360
signer crops. Original and both crops fully decode to 31,086 frames; all packet
timestamps match within one microsecond. Crop hashes are in `signer_crops.json`.
RIT: public fluent-coded F13 MOV/EAF sample, 6.5 seconds, 195 decoded frames,
nine timed glosses. Full RIT and CUNY remain access-restricted, not acquired.
Native MoLo/YouTube downloads returned 403; public official OSF succeeded.
HF anonymous rate limit resolved by respecting reset and resuming normal download.
Crop verification initially failed on relative-path bookkeeping; resolving paths
in the shared report helper fixed it, and both complete decode checks then passed.
Final inventory assertions and `git diff --check` passed. No training, protected-test
access, production changes or dataset deletion. Next safe action: review raw-video
alignment, annotation completeness and exact lexical variants before creating any
Apple Vision training manifest. Do not label unannotated/OOV spans as background.

## 2026-09-12 — authorized targeted continuous-ASL acquisition started

User explicitly requested actual data acquisition, not another source shortlist.
Free disk is approximately 12 GiB; use bounded source transfers and preserve existing data.
Downloaded RIT's public fluent-signer F13 sample (14.1 MB MOV), matching human EAF,
demographics and publisher page to data/local/continuous_asl_acquisition_20260912.
Full RIT Homework data needs institutional Databrary access; CUNY requires publisher
contact. No contact/signup or access bypass was attempted. MoLo's native publisher
video endpoint returns403; checking its openly licensed official video publication.
CLERC Epee v0.3 provides1,200 timed multi-sign pose sequences from6 publisher-described
native Deaf signers, but no raw pixels. Download underway at pinned HF revision
93bc8aa0a6526e86af80db5f59af8d133292c9b5. It is a separate MediaPipe corpus, never silently
Apple-v17-compatible training data. Next: verify all files, count exact label coverage,
repetitions and timing, and complete remaining raw-video acquisition where accessible.

## 2026-09-08 15:17 PHT — expanded Citizen-linked phrase reviewer is ready

Completed the same fail-closed audit, accepted-model auto-annotation and review-UI flow
for the expanded ASL STEM Wiki pool. A malformed source gloss with an unmatched closing
parenthesis was caught during full audit; acquisition now rejects unbalanced annotations
and a regression test covers it. The corrected pool contains 267 hash-verified and fully
decoded videos, 503 locked-string occurrences and 44/100 locked glosses. This supersedes
the provisional 268-video/506-token counts in the preceding entry.

The separate expanded queue has 206 participant/gloss pairs from 154 selected source
videos. Exact filename, video-SHA and ASL-LEX-code matching carried over all 63 completed
human reviews from the first queue, including 33 eligible spans; the original queue was
not modified. The accepted Reel landmark classifier, frozen multimodal encoder and
primary/specialist Stage-2 CTC selectors scanned all 206 rows in 2,025.1 seconds. Strict
calibration against those 63 reviews found 27 boundary-and-variant successes and set a
0.919691 high-evidence threshold. Final tiers are 8 high, 112 review, 23 abstain and 63
human-reviewed. Automatic training eligibility remains zero.

Artifacts are `artifacts/reports/asl_stem_wiki_manual_expansion_admission_v17/` and
`artifacts/reports/asl_stem_wiki_manual_expansion_auto_annotation_v17/`. The review UI
now includes a high-evidence filter in addition to review and abstention filters. All
automatic bounds remain editable, unsaved review aids; signer and exact Citizen variant
decisions remain manual. No protected evaluation split was accessed.

## 2026-09-08 12:30 PHT — ASL STEM Wiki admission audit blocks training pending expert review

The 08:43 claim that P12, P13, P21, P28, P33 and P35 were all source-verified
strong/native-like signers is incorrect and superseded. It came from an inferred
participant ordering that the source does not establish. Positive source-note evidence
maps only P13 to Subject 2 and P28 to Subject 7; the appendix places Subject 7 in its
possible-L2 group, so P28 is excluded. P13 is not in the appendix's strong-subject list,
and P12/P21/P33/P35 have no resolved subject mapping. The parser now ignores negative
notes such as `Not Subject 17`; a regression test covers this failure.

The 98 already downloaded videos are a provisional verification pool, not training
data. All 98 SHA-256 values were rechecked and every video fully decoded. The manual
gloss parser preserves parenthesized classifier descriptions, fingerspelling and
repetitions. Public pseudo annotations provide a position proposal for 33 of 71 unique
participant/raw-gloss review pairs, but model proposals do not verify boundaries or
the frozen Citizen ASL-LEX visual variant. The audited manifest has zero eligible
spans, so no feature extraction, selector training, runtime promotion, or protected
Citizen/SemLex/local/RIT evaluation access occurred.

Canonical outputs are
`artifacts/reports/asl_stem_wiki_manual_admission_v17/README.md`,
`audit.json`, `admission_manifest.json`, and `expert_review_queue.csv`. Acquisition and
audit code are `scripts/acquire_asl_stem_wiki_manual_v17.py` and
`scripts/audit_asl_stem_wiki_manual_v17.py`; eight focused tests pass. Next safe action:
an ASL-qualified reviewer completes signer quality, exact ASL-LEX variant, and exact
start/end frames in the queue. Only fully approved spans may be normalized to 30 fps,
kept at 256 frames or fewer, and considered for participant-disjoint training after
the minimum 2-train/1-validation-participant coverage gate is met.

## 2026-09-08 07:59 PHT — Cokely v2 verification complete; data remains quarantined

The provisional Cokely v1 readiness claim is superseded. Corrected the selector to
parse the main gloss tier, both supplementary hand tiers and ASL utterance boundaries;
matching simultaneous hand labels are duplicate evidence, while conflicts, OOV labels,
invalid timing, utterance changes and gaps over300ms break runs. Removed arbitrary
six-token splitting and the30-clip minimum. Full audit of2253main annotations yields
22maximal candidate runs/85tokens/18classes across4recordings and3signers. Rejections:
2019OOV,132isolated locked tokens,16supplementary-hand conflicts,1invalid timing.
David Hamilton has no surviving multi-sign run. All22 clips decode and show active,
centered motion in the full contact sheet, but66/85tokens belong to immediate repeated-
label runs, including50DIFFERENT tokens; tokenization intent remains unresolved.

Added explicit `candidate_verification` extraction support without changing default
train/validation selection. One Apple Vision v17 smoke per surviving signer passed,
then all22 produced overlapping32-frame/stride16 landmark+hand-RGB archives, all22
MobileCLIP2-S0 hand archives, and all22 frozen612-D selected Stage1 archives with zero
failures. Frozen checkpoint SHA256:
278a9933df25508aa83823cf6a8e050fbf4ba729eb39b1c790c97e55032a6558.
Segment compatibility is15/85top1 and26/85top5; excluding immediate repetition it is
4/19top1 and5/19top5. This is diagnostic and does not prove or disprove variants.
Cokely has no explicit crosswalk from its project ID-glosses to Citizen ASL-LEX codes;
the Citizen-train comparison sheet is only a review aid. Therefore all22 remain
`variant_verified:false`, `training_eligible:false`, grouped by recording/signer.
No training/runtime/model promotion and no Citizen/SemLex/local/RIT protected split
access occurred. This source is not enough for the next training step.

Canonical outputs: `artifacts/reports/continuous_reel_v17_cokely_v2/README.md`,
`manifest.json`, `extraction_full.json`, `stage1_compatibility.json`, motion and variant
contact sheets. Local features are under
`data/local/continuous_phrase_sources_v17/cokely/verified_v2_stage2_*`. Added/updated
focused tests for maximal repeats, utterance/two-hand boundaries, candidate role
selection and segment aggregation; run the focused suite plus `git diff --check`
before handoff. Next safe action: acquire another timed two-hand source with explicit
lexical provenance, or obtain expert decisions for the21 Cokely signer/class pairs and
the repeated-token convention. Do not train from v2 as currently marked.

## 2026-09-08 08:29 PHT — public ASL STEM Wiki gloss release acquired; video selection is pending

The public CVPR Findings 2026 supplemental archive from Lea et al. was located via
the author's published link and saved at
`data/local/asl_stem_wiki_bootstrap_v1/supplemental_annotations.zip`. Its ZIP
integrity check passed. The release contains the manual annotation CSV, the
fingerspelling annotations, and the larger pseudo-annotation bundles. The manual CSV
has 509 rows / 495 distinct videos, 8,709 annotated tokens, 17 raw `user` values, and
21 articles. It has 44 exact frozen Citizen raw-gloss strings with 532 occurrences;
these are lexical string matches only and do **not** establish the pinned ASL-LEX
variant, so all non-verified content must remain `OTHER`.

The official ASL STEM Wiki source ZIP is public, supports range requests, and is
187.38 GiB in full. Its central directory was acquired without downloading video and
proves that all 495 manual-video filenames exist. Their compressed bytes total 1.47
GiB (1.61 GiB decoded), which fits the available disk budget. The supplemental
review identifies subjects 3, 4, 9, 11, 12, and 14 as strong/native-or-native-like
signers and identifies 1, 6, 7, 10, 13, 15, and 16 as L2/hearing-accent signers. The
public `videos.csv` is still required to map those review subject numbers to the CSV
user IDs before video download; do not pool unfiltered signer IDs.

The next external range request for that small metadata member was automatically
rejected because the Codex usage limit was reached. No workaround was attempted. No
Citizen, SemLex, local, RIT, or other protected evaluation video was accessed. The
training-ready acquisition remains blocked only on resuming that authorized public
metadata/video transfer after usage is available.

## 2026-09-08 08:43 PHT — reviewer-filtered ASL STEM Wiki manual set is locally available

The public `videos.csv` member was range-read and CRC-verified, resolving annotation
filenames to source participants. The appendix’s subject ordering was reconciled using
the subject-10 `MODEL`/`retarded` signature, which maps to P24; the six reviewer-rated
strong/native-or-native-like signers are therefore P12, P13, P21, P28, P33, and P35.
The acquisition code and its focused test are
`scripts/acquire_asl_stem_wiki_manual_v17.py` and
`test/test_acquire_asl_stem_wiki_manual_v17.py`. The test caught and prevented a
central-directory CRC field-index error before accepting any video.

`data/local/asl_stem_wiki_bootstrap_v1/manual_candidate_manifest.json` now records
98 downloaded sentence clips from those six signers: 35.38 minutes / 400,580,618
decoded bytes, 30 locked raw-gloss strings, and 180 exact-string occurrences. Every
other manual token is explicit `OTHER`; the data is auxiliary Stage-2 sequence CTC
material only, never ASL-LEX-variant-verified Stage-1 supervision. The candidate
filter rejected non-reviewed participants, empty sequences, duplicate conflicts, and
sentences without a locked raw string.

All 98 source ZIP members passed CRC during retrieval, have a manifest SHA-256 that
was rechecked from disk, and decode successfully (640x480; 162--1,782 frames). Their
source FPS is variable and must be sampled by timestamps or normalized during the
next extraction step; do not treat frame count as a fixed-rate timing label. No
protected Citizen, SemLex, local, RIT, or other evaluation data was accessed.

## 2026-09-07 22:31 PHT — MoLo two-hand screening complete; eight provisional spans

Public media follow-up found27files (26MP4s and1document) in nodewma3e's linked
Google Drive provider; its native OSF storage and child-node lists are empty. Saved
complete provider listing and `media_audit.json`. Of8candidate spans,1has a matching
title/native download on https://ida.gallaudet.edu/molo/1/ (advertised1648.4MB),2refer
to oldS_7_8 versus currentS_4_5 (https://ida.gallaudet.edu/molo/9/), and5reference
MoLo002 videos absent from the inspected public listings. No downloadedvideo,
content hash, or time alignment claim. Video terms are publisherCCBY-NC-SA4.0.
Full findings and runnable commands: `artifacts/reports/continuous_reel_v17_web_audit_v2/README.md`.
Fresh focused4-test run and tracked/new-code whitespace checks pass. This bounded
metadata audit is complete; missing30phrases have not been replaced. Next useful
work is exact-variant and source-alignment review, or obtaining timed NCSLGR annotations
through authorized access; do not train/promote from these discovery candidates.

Implemented stdlib `scripts/audit_molo_continuous_v17.py`; all4 focused tests pass
(two-hand deduplication/repeats, other-hand blockers/variant conflicts, aliases/notes/
gaps, EAF reference and timing validation). Full16-file audit verifies source sizes
and records SHA256 provenance: 5284280bytes,6878manualannotations,551exact raw-label
occurrences before deduplication. Four files have no manual labels;11 have at least
two, representing6 filename-inferred signers. Conservative overlapping-hand merging
and maximum300ms unannotated gap yield8candidate spans,7distinct sequences,11raw
classes across3sessions;7pairs and1triple. These are fragments, not30usable phrases.
All8 remain ineligible for training: exact lexical-variant mapping, source-video
alignment, signer/session independence, and completeness remain unverified. Two
MoLo001 candidates refer to oldS_7_8 media while the public gallery lists re-edited
S_4_5; do not transfer timestamps without verifying the version. No runtime/model
change or protected split access. Outputs: web_audit_v2/audit.json,candidates.json.

## 2026-09-06 09:43 PST — code review for reducing phrase collection; context questions pending

The user requested reading recent ground truth and relevant code before asking context
questions, particularly whether improved landmark signing voices could remove the need
for the proposed new phrases. No model/source/data changes or training were performed;
this entry records the review only. Existing uncommitted work is preserved.

The two September 4 entries appended at the end of this file contain the latest local
signer correction and grounded continuous experiment; the earlier top-of-file local
phrase metrics must not be presented as signer-held-out evidence. The selected grounded
checkpoint reports 25.5% local held-out-signer exact / 37.04% WER, 82.37% isolated exact,
and 11.36% transition false emission. See
`artifacts/reports/grounded_continuous_v17_experiment_v1/README.md`. The historical
appended sentence saying the official Citizen test "remains unopened" is inconsistent
with the consumed 87.57% test gate: interpret current experiment claims as no additional
test access, not an unused official test.

Tracing `model_signing_voice_profile_v17.py` and
`scripts/render_signing_voice_phrase_v17.py` confirms the accepted voice path uses
statistical profiles, content-preserving strength selection, and duration mixing; the
neural residual generator is a rejected research alternative. The grounded renderer
uses train-only same-signer isolated clips plus learned boundary timing/inpainting in
`signing_voice_phrase_v17.py`. Its 178/200 historical machine-gate result measures
structural/recognizer acceptance, not native ASL naturalness or downstream transfer.
Generated outputs explicitly remain native-review-only and training/validation/test
ineligible. No eligibility was changed.

Working hypothesis for discussion: improve context-dependent timing and motion near
lexical boundaries, retain exact lexical/hand-participation constraints, and compare
reviewed synthetic augmentation against a matched real-only recognizer on real
signer-disjoint development data. Independent human review and downstream transfer
matter because the same Stage-1 recognizer currently filters generated content.
More style mixtures alone do not establish new-human generalization. The current
evidence cannot justify cancelling all real phrase collection; whether a smaller
targeted collection is acceptable depends on the user's scope and resources.

Next action: obtain context on the meaning/use of signing voices, whether reducing or
eliminating collection is required, capture status and native-review availability,
desired vocabulary/continuous behavior, and success/deadline/compute constraints.
No new experiment or implementation is selected before that discussion. This is a
code/report inspection, not newly measured accuracy; no protected evaluation was run.

## 2026-08-24 13:18 PST — 2M-Flores temporal replay tested; compact packaging corrected

The recommended genuine-sentence experiment is complete. The rejected 448-token
2M-Flores auxiliary CTC transfer was replaced with label-free masked temporal
reconstruction and token-contrastive alignment over bounded genuine sentence crops.
The 100-sign CTC head remained frozen. A deterministic train-only 127/28 split of the
155 acquired 2M-Flores `dev` sentences selected epoch 13 after 80.79 seconds on capped
MPS. The checkpoint is
`artifacts/models/stage2_v17_2m_flores_temporal_pretrain_v1/temporal_pretrained.pth`,
SHA-256 `ba543b1fd9fa9dd5827b5c67b5f4b0b4a52748187d45513b17a87d64274519ab`.

A controlled 0/10/20/30% synthetic replay sweep selected 10%. Seed 12701 scored
12/24 ASLLRP phrase edits, 7/259 local phrase edits, and 41/254 JONATHAN contextual
edits; seeds 12702 and 12703 reproduced 12/7/41 and 12/8/41. A final label-checked
selector-oversampling experiment scanned all 39,350 rows, found 714 selector-owned
rows but only 85 exact rows, and regressed to 12/9/41. Temporal pretraining therefore
improves the compact contextual gate from 43 to 41 edits but regresses ASLLRP phrases
from 11 to 12. It fails the no-regression contract and is not promoted. The selected
accuracy-research model remains the two-head general selector at 9/24, 6/259, and
43/254 edits.

The compact-student packaging path had a real mismatch: validation used the HOME/WHERE
context residual while `best_model.pth` stored only the bare CTC head. The trainer and
a standalone packager now save the exact evaluated `Stage2ContextAdapterV17` graph and
verify a cold reload. The retained compact artifact is
`artifacts/models/stage2_v17_compact_context_student_v1/model.pth`, SHA-256
`623f9b56141643704b3562a8d2fdcebe44269985b2f618eb8f0a471e857a2cf5`, reproducing
11/24 ASLLRP, 7/259 local, and 43/254 contextual edits. The non-promoted temporal
alternative is separately packaged at SHA-256
`a2568f6d38416a41a5f9b547224c50740874bd046cfa268c2f4a58166c88c4e6` and reproduces
12/7/41.

Full evidence is in
`artifacts/reports/stage2_v17_2m_temporal_replay_v1/EXPERIMENT.md`. Stage 3 may now be
developed as a separately evaluated reference-gloss-to-English module, but current
recognition evidence does not justify a general end-to-end translation claim. Mobile
export engineering may proceed from the exact compact artifact, but v17 Stage-2 Core
ML conversion/parity, app integration, and physical-iPhone measurement remain open.
No Citizen, SemLex, local, 2M-Flores `devtest`, or consumed RIT test split was accessed.

## 2026-08-22 13:40 PST — multi-corpus selection is now warm-started and no-regression constrained

A normalized contact sheet of the first 12 independent YouTube-ASL channels confirms
active human signing in all 12, with varied people, framing, backgrounds, signing
spaces, and both one- and two-person material.  The bounded downloader remains stable
and resumable; 52/128 channel clips were complete at this timestamp, with one failed
candidate automatically covered by the discovery reserve.  The acquisition process
uses about 40 MB RSS and does not create memory pressure.

The multi-corpus trainer no longer starts each fold blindly or selects only an
unconstrained cross-domain average.  It can now warm-start from the exact proven,
fold-matched How2Sign checkpoint, rejects mismatched held-out signers/configurations,
and pins the initial checkpoint hash.  The initial held-out How2Sign relative
improvement becomes a hard model-selection floor (zero regression tolerance by
default), while channel-held-out YouTube-ASL quality contributes to selection above
that floor.  This preserves the established signer-generalization evidence while
testing whether many new channel-level voice proxies improve web generalization.
The optional motion-distribution term is exposed but remains disabled because the
prior targeted experiment regressed held-out metrics.  Eight focused transition tests
pass, the changed entry points compile, and no sealed split was accessed.

## 2026-08-16 23:17 PST — compact 2M-Flores acquisition complete and independently verified

The selected 2M-Flores `dev` acquisition is complete: 155/155 manifest rows, 155 state
records, and 155 derived video files. An independent final pass recalculated every
derived SHA-256 and matched all 155 recorded hashes. The temporary source directory is
empty. In total, 20,327,996,229 source bytes (18.9319 GiB) were individually verified;
the aspect-preserving 720p30 H.264 corpus retains 1,355,423,637 bytes (1.2623 GiB).
The maximum recorded source-to-derived duration difference is 0.158333 seconds, below
the unchanged 0.200-second gate, and each clip passed full decode verification during
acquisition.

The final acquisition state SHA-256 is
`a7b2f363f877317c7439fd2b566945fff00df13dbd4ddeef7faa27c14c64449e`.
Completion evidence is recorded at
`artifacts/reports/stage2_v17_new_dataset_search/2m_flores_acquisition_complete.json`
(SHA-256 `ff84141641e6ee9fb9b49e79a4b4f1cd008542e963fbcdc087778739d2ee426b`).
The source plan now marks acquisition complete. The next gate is to lock the expanded
gloss vocabulary and convert all complete ordered gloss sequences into a v17 Stage 2
preprocessing manifest; the 100-label matches remain selection metadata and must not
replace the full transcripts. The 2M-Flores `devtest` split and all project evaluation
splits remain untouched.
Generated JSON validation, four focused audit/selection tests, Python compilation,
and `git diff --check` pass.

## 2026-08-16 17:02 PST — 2M-Flores acquisition resumed after correct duration-gate fix

The bounded 2M-Flores acquisition was not complete when checked: it had safely stopped
after 106/155 verified clips while processing row 682. The derived duration differed
from the source *container* duration by 0.203 seconds, three milliseconds beyond the
0.200-second threshold. Frame/timestamp inspection proved this was not truncation: the
source video stream is 15.235 seconds with 914 frames near 60 fps, while the derived
stream is 15.233333 seconds with 457 frames at 30 fps. The 15.436666-second source
container includes approximately 0.202 seconds of empty trailing container time.

The acquisition verifier now compares video-stream durations and records container
duration separately, falling back to container duration only if a stream duration is
unavailable. The 0.200-second accuracy threshold was not relaxed. Python compilation,
four focused audit/selection tests, and `git diff --check` pass. Row 682 then completed
successfully under the corrected check. At 17:02 PST, acquisition has resumed at
107/155 clips, 944,704,523 retained derived bytes, with 48 rows remaining. The live
authoritative count remains `data/local/2m_flores_asl_stage2_v17/acquisition_state.json`.
No evaluation split was accessed.

## 2026-08-16 14:13 PST — compact 2M-Flores selection locked; acquisition started safely

The full 156 GB 2M-Flores `dev` split will not be bulk-downloaded. File-level size and
LFS SHA-256 metadata were resolved from the dataset's pinned revision
`b450c1a427738e78f06362fc4619674f5d74f774`. A binary minimum-byte multicover
optimization selected 155 complete sentence videos, 18.9319 GiB of cumulative source
transfer, covering all 95 available locked labels with up to five examples per label.
For labels occurring fewer than five times, every available matching sentence is
retained. Complete gloss sequences remain unchanged for an expanded Stage 2 decoder.
The selection manifest is
`data/local/dataset_metadata/2m_flores_asl/dev_selected_v17.json`, SHA-256
`92ed45d9b52c3d34146233356563363d7a8517540a6fd9df9b32bd451f27da4c`.

The resumable acquisition worker downloads exactly one selected source at a time,
checks its pinned SHA-256, performs an aspect-preserving fit within 1280x720 at 30 fps
using the macOS hardware H.264 encoder, checks duration and dimensions, fully decodes
the result, records both hashes, and then removes only the verified temporary source.
It enforces at least 12 GiB disk headroom and never uses MPS. The first complete safety
run passed: row 3's 223.7 MiB source produced a verified 16.7 MiB derived clip, its
state was recorded under `data/local/2m_flores_asl_stage2_v17/acquisition_state.json`,
and the source temporary was removed. The remaining 154 clips are safe to resume
without repeating completed work.

At 14:23 PST, continuous acquisition has reached 10/155 verified clips and 118,947,445
derived bytes; the next selected clip is in its one-file temporary download. The live,
authoritative count is the `completed_rows` map in `acquisition_state.json`, which is
atomically updated after every verified clip. Acquisition continues in the bounded
worker; this timestamp is a progress snapshot rather than a completion claim.

The actual ASL-Homework-RGBD archive is Databrary volume 1249. The RIT supplemental
page does not hyperlink it; it only states that the dataset uses Databrary. Full access
requires an authorized Databrary account, while the RIT page publicly exposes only a
sample video/ELAN/depth triplet, prompts, annotation guide, and demographics. Four
focused selection/audit tests and Python compilation pass. No reserved dataset or
project test split was accessed.

## 2026-08-15 09:18 PST — current ASLLRP segmented-sign metadata acquired and exact clips verified

The owner obtained the current metadata for signs segmented from continuous signing,
not the separate citation-form Stage 1 datasets. The authoritative inputs are
`asllrp_sentence_signs_2025_06_28.csv` (SHA-256
`9c1641b6b95a6e6c2223c34eb101ed2e83bb1da824e18c7a21791c59b1593e86`) and
`rit_sentence_signs_2025_11_01.csv` (SHA-256
`27af833206b85ef02dbe6008173c0b1938504ba05943954f76269380c07fe6fb`). Older
2023/2024 ASLLRP snapshots remain untouched in Downloads but are superseded and were
not combined. The ASLLRP CSV has 17,519 valid rows and three malformed quoted-gesture
rows that are rejected fail-closed; RIT has 3,056 valid rows and no rejected rows.

`scripts/prepare_asllrp_continuous_citizen100_v17.py` applies the official ASL-LEX
`SignBankAnnotationID` contract rather than English-label normalization. It admits
1,719 unique segmented signs across 54/100 classes: 1,483 ASLLRP signs from four
participants across 53 classes are training candidates, while all 236 RIT signs from
two participants across 30 classes are permanently reserved for new external Stage 1
evaluation. The short clips were downloaded and independently rehashed to
`data/local/asllrp_segmented_citizen100_v17`; all 1,719 decode, all SHA-256 values are
unique, and there are zero missing, size, hash, or acquisition failures. They total
132,149,094 bytes and 476.131753 seconds. The acquisition manifest SHA-256 is
`44f50fc61c981fd774dc7eedc8c226747ca4221738c952449f44e13d7e73d622`.

These metadata expose 1,237 parent utterances containing at least one exact target
sign, but only one utterance (`WATER COLD`) has at least two tokens and no non-gesture
gloss outside the locked 100-class vocabulary. A stricter frame-level scan nevertheless
finds 70 contiguous target-only multi-sign runs across 68 parents: 56 ASLLRP training
candidates and 14 RIT external-evaluation spans, totaling 144 target tokens across 53
unique phrases. Only those 68 parent videos were downloaded; the spans were cropped at
manual first/last sign bounds plus five context frames. All 70 crops decode, have unique
SHA-256 values, and total 73.842842 seconds with zero integrity failures. The span
manifest SHA-256 is
`9e38d4ab1d289f41f402e3c9b3b5a355167c8e2783fd1ba8c91538d6932c9a4e`.

Therefore the segmented clips are high-value contextual Stage 1 data and the 70 exact
runs are valid short real-phrase Stage 2 candidates. The other parent utterances are
not falsely labeled as direct 100-class CTC training data, and no master videos were
downloaded. Stage 2 must lock either an expanded output vocabulary or a principled
partial-label objective before acquiring the other target-bearing parents. The
separate citation-form metadata is still not acquired.

The exhaustive audit is
`artifacts/reports/asllrp_continuous_citizen100_v17/audit.json` (SHA-256
`ef32b0f367f25468b88b45f2d78c21c10d2778c5d61d7c340c4740a648907147`). Four new
fail-closed parser/variant tests pass, Python compilation passes, JSON validation and
`git diff --check` pass. Citizen, SemLex, and local sealed test splits were not
accessed. The pinned Stage 2 source plan is now SHA-256
`12f13c910393a6506f2df873d7dec9c5465f3bcbdc65e422b3f038d2638d5e42`.

## 2026-08-13 22:06 PST — finalized-manifest and train-only packaging gates added

Fresh extraction remains active across four disjoint class shards; at this handoff it
had produced 2,734 of 13,382 train archives with all four Apple Vision workers healthy
and no logged extraction errors. `scripts/finalize_local_deep_clean_v17.py` now validates
every archive against the current v17 schema, emits immutable train/validation final
manifests plus a rejection ledger, and fails closed if extraction loses any of the 94
admitted classes. The training and validation loaders accept those final manifests
only when extraction completeness, schema, signer-overlap approval, and false
Citizen/SemLex test-access claims all hold. The frozen-model local evaluator now
defaults to the final validation manifest.

`scripts/package_local_deep_clean_v17_kaggle.py` stages only finalized local train and
validation features for the future private Kaggle job and explicitly forbids local
test members. Six preparation/finalization tests and the two focused finalized-loader
tests pass; affected Python files compile and focused `git diff --check` passes.
Current SHA-256 values are finalizer
`46f35d292bf0be94d8ba6c3fdac73bcccc674ccbfb30a6b3fec01a7f3fed0dce`,
packager `50d2a38ff0e5e3b2eec92ec0c484aae47bbc4959967a556cbf746e0afc3db813`,
trainer `085d6cc344aadd79dd7919b24106fa64e41bd087f4e6eb201bbce9fcf74f98cc`,
and evaluator
`e256bd2c8122e460297069ff80898756e5c1170b72afd3ee9287f801bbc0d30a`.
Neither Citizen test, SemLex test, nor the unused local test split was accessed.

The CUDA-only runner and reproducible Kaggle job builder are now implemented as
`active/v17/kaggle_stage1_local_deep_clean_runner_v17.py` and
`scripts/prepare_local_deep_clean_v17_kaggle_job.py`. The run is predeclared at
Citizen/SemLex/local source margins 0.34/0.33/0.33, seed 1701, current part-wise +
global architecture, and continuous full-circle roll augmentation. It selects only
on official Citizen validation top-1. Promotion requires all four gates: at least
361/378 Citizen validation top-1 (no more than one clip below the frozen compact
orientation model), at least 839/978 SemLex validation top-1, and strictly better
local familiar-signer validation top-1 than the frozen compact baseline, while the
eight-angle landmark-roll stress suite must retain at least the current 348/378
worst-angle score. Runner,
packager, and job-builder compile and pass `git diff --check`; their SHA-256 values are
`487d0c737ecb85e9b8c3267b3072734ee8cbedb986f05305989452635e1c384f`,
`7e1a62e9a097225a46bcf568d3d68bd7645fb53fbf528019f9d9b4602a1023a7`,
and `cf7472d35a63e71eeeaf500406edfce0ee4cd9fbbb8a44423c8131379824baa3`.
The combined preparation/training suite is 56/56 passing; the 17/17 extractor tests
also pass, including real Apple Vision rotation/mirroring and portrait/landscape
isotropy coverage.

An explicit raw-byte decontamination audit also passes before training. The 13,382
local-train and 2,896 local-validation rows have zero SHA-256-identical matches against
all 1,854 unique Citizen train/validation raw clips and all 2,448 unique SemLex
train/validation raw clips. The audit deliberately does not read either protected test
split or the unused local test split. Exact evidence is
`artifacts/reports/local_deep_clean_v17/raw_hash_overlap_audit.json`, SHA-256
`d875b76decd357ae6b630e3d7527da06eccf0f715a30e8dd3cecfd4843db873b`;
the reproducible audit script SHA-256 is
`13683961bd6178c28c4b0c34f1737b811b0a9ba5ffee341cf6b39cc022176ecc`.

## 2026-08-12 01:08 PST — SemLex-validation hand-RGB audit isolated from training

The next accuracy gate is a cross-domain component audit on the already approved
978-clip SemLex validation diagnostic before adding more architecture or local data.
`extract_hand_rgb_semlex_val_v17.py` now resolves only the exact frozen validation
selection, requires `training_eligible=false` and
`evaluation_only_never_training`, proves the 978 retained/six quarantined inventory,
and rejects partial or unplanned files. Its outputs are explicitly stamped
`semlex_val`, `val_domain_diagnostic`, and non-training-eligible. The compact
MobileCLIP2 encoder can encode this source, but the Stage-1 training loader still has
no path that accepts it. A separate evaluator validates this provenance and reports
both 100-class and 98-present-class macro F1. The one-clip real Apple Vision smoke
passed in 0.33 seconds, the focused hand suite passes 12/12, and all new scripts
compile. Neither Citizen test nor SemLex test was accessed.

Landscape orientation is not by itself a reason to reject the local corpus: the hand
extractor preserves aspect ratio, follows the v17 landmark motion interval, and
normalizes per-hand/union crops. Tiny, blurred, occluded, or uncertain-variant clips
remain quality risks. The existing 434 strongest local Tier-A clips are already in
the 80.69% hand model. After the cross-domain audit, the remaining local pool will be
re-scored with the improved multimodal teacher, exact-variant and extraction gates,
and a strict source cap so local sessions cannot dominate Citizen/SemLex.

Extraction is now complete for all 978/978 SemLex validation clips across 98 classes:
zero failures, zero empty-view clips, 356.3 MiB total, and 67.45% mean validity over
the left/right/union view grid. A full metadata/schema/provenance read found zero bad
archives. The run took 157.7 seconds. Compact frozen MobileCLIP2 encoding is the next
step; these artifacts remain evaluation-only and cannot enter training.

The previously documented `mobileclip2_env` directory was absent, while its partial
pure-Python runtime layer remained. The isolated environment was reconstructed under
the same path with host Python 3.11/torch 2.13.0 and pinned OpenCLIP 3.3.0,
torchvision 0.28.0, timm 1.0.28, and OpenCV-headless 4.13.0. It reuses the existing
official cached MobileCLIP2-S0 weights. A real SemLex-validation feature smoke ran on
MPS, wrote one finite archive with schema `c54f4edc6f62b08b`, and retained the
non-training/validation-only provenance. The first two failed launches wrote no
project dataset files: one used the Vision environment without OpenCLIP and one
exposed the partial runtime's missing compiled dependencies.

All 978 compact SemLex-validation hand embeddings are now complete after 426.0
seconds on MPS. They occupy 31.0 MiB; a full finite-value, shape, zero-mask, schema,
source, split, and eligibility audit found zero bad or empty archives. The unchanged
epoch-33 multisource compact hand checkpoint scores 73.31% top-1, 91.41% top-5, and
69.75% present-class macro F1 on this diagnostic, versus its 80.69% Citizen
validation score. The older Citizen-only compact hand checkpoint scores only 51.64%
top-1, 79.75% top-5, and 48.21% present-class macro F1 on the same clips. Thus the
reviewed SemLex/local training pool adds 21.68 cross-domain top-1 points; more clean,
variant-consistent hand data is a much stronger lever than the rejected spatial
fine-tuning. The hand branch still trails the unchanged landmark model's 85.89%
SemLex result, so it remains a complementary RGB teacher rather than a landmark
replacement. Exact outputs are under
`artifacts/reports/semlex_citizen100_val_audit/hand_mobileclip2_*`. The landmark
diagnostic was rerun only to regenerate its aligned logits and reproduced 85.89%
exactly. No test data was accessed.

The equivalent SemLex-validation visual-speech path is now implemented as a separate
non-training diagnostic. It applies the unchanged full-utterance, mouth-motion,
eye-aligned mouth/lower-face extractor, adds the proven invalid-WebM-frame-count
fallback, and stamps source/split/eligibility/audio/test provenance. A dedicated
Auto-AVSR cache builder and evaluator support the existing mouth, lower-face, and
learned mouth+lower checkpoints without creating a training-loader path. Six focused
visual-speech tests pass, all new scripts compile, and a real Apple Vision one-clip
smoke passed with schema `c44cdc314b5128c7`. Full 978-clip extraction is next.

Full SemLex-validation visual-speech extraction completed 978/978 across 98 classes
in 393.0 seconds with zero failures. The archives occupy 240.0 MiB; a complete
metadata/schema/provenance read found zero bad archives and zero clips empty in any
view. Mouth, lower-face, and full-face selected-frame validity is 99.77% for each,
and all 978 used a detected mouth-motion interval rather than fallback timing. Frozen
Auto-AVSR mouth/lower feature encoding and unchanged-head evaluation are next.

Both 978-sample frozen Auto-AVSR caches completed, strict-loaded all 120 official
frontend keys, and retained visual-only/evaluation-only provenance. On SemLex
validation, the unchanged Citizen-trained mouth head scores 15.95% top-1 / 37.01%
top-5, lower face scores 15.13% / 40.08%, and learned mouth+lower scores 19.33% /
43.66%. These are well above chance but substantially below their Citizen scores of
29.63%, 26.72%, and 31.75%, respectively. The lip/lower classifiers therefore carry
real signal but are not individually domain-robust; broader or purpose-recorded
visual-speech data is still needed before production use.

Crucially, cross-domain complementarity survives without any SemLex weight tuning.
The exact Citizen-selected four-stream per-sample-z-score weights (0.30 landmark,
0.15 mouth, 0.35 lower face, 0.20 hand) improve SemLex validation from 840/978 =
85.89% landmarks alone to 882/978 = 90.18%, with 51 corrections and nine
regressions. The fixed 75/25 landmark/hand pair reaches 87.83%, and the fixed equal
landmark/learned-mouth+lower pair reaches 88.04%. This makes the 97.88% Citizen
ensemble a credible cross-domain research teacher, though not yet a production
estimate because the original weights were selected on Citizen validation and SemLex
validation is not signer-independent. Reports are under
`artifacts/reports/semlex_citizen100_val_audit/fixed_*`. Thirteen hand/fusion tests and
six visual-speech tests pass. No Citizen or SemLex test data was accessed.

Decision: use this multimodal teacher to re-score the remaining local corpus, admit
only exact-variant/high-confidence/extraction-clean clips, and retain a strict local
source cap. Do not promote the weak standalone lip branch or tune fusion weights on
SemLex validation.

Local multimodal re-mining is now isolated under a strict resolver for the immutable
1,021-clip, 77-class cap-14 exact-text audit manifest. It requires the original
non-training eligibility, exact canonical/pinned-raw text equality, and every raw and
landmark file. New hand and visual-speech extractors stamp `local_audit` and
`train_only_review_diagnostic`; neither the landmark nor hand trainers accept this
source. The MobileCLIP encoder and visual feature cache support the diagnostic source,
and the existing local landmark evaluator now preserves aligned logits. Twenty
focused hand+visual tests pass, all affected scripts compile, and real one-clip hand
and face smokes passed. The multimodal teacher will only screen unused clips for a
small source-capped review upgrade; it will not silently relabel model disagreements.

Local hand extraction completed 1,021/1,021 in 146.0 seconds with zero failures. The
77-class crop corpus occupies 434.8 MiB, has no empty clips, and averages 74.13%
validity over left/right/union views. A complete metadata/provenance read found zero
bad archives. Frozen compact MobileCLIP2 encoding is next.

All 1,021 local compact hand features completed in 536.9 seconds on MPS and occupy
35.3 MiB. A full shape/finite/zero-mask/schema/source/split audit found zero bad or
empty archives. As label-screening agreement, not held-out accuracy, the unchanged
multisource hand model matches the exact-text folder label at 70.62% top-1 / 87.07%
top-5. The current 95.77% landmark winner was rerun on the same pool and reaches
68.76% / 86.78%. This confirms the improved hand stream is independently useful for
local re-mining rather than merely echoing the landmark model. Face extraction is the
next and only heavy process.

Local visual-speech extraction completed 1,021/1,021 across 77 classes in 284.4
seconds with zero runtime failures. The 182.3 MiB corpus has 89.79% mean view
validity; 76 clips have no usable face view, 935 use mouth-motion timing, and 86 use
the explicit full-utterance fallback. Both Auto-AVSR caches are complete. Unlike
Citizen and SemLex, the Citizen-trained visual-speech heads are unusable on this
landscape local domain: mouth is 2.15% top-1, lower face 3.23%, and learned
mouth+lower 2.94%. They are rejected as local cleaning signals.

The unchanged Citizen-fixed 75/25 landmark/hand fusion reaches 751/1,021 = 73.56%
local exact-text agreement, 91.19% top-5, with 52 corrections and only three
regressions relative to landmarks. The full Citizen four-stream weights reach the
same 73.56% top-1 but only 87.46% top-5 and cause 30 regressions because the face
domain is incompatible. Therefore local re-mining uses landmark+hand only. A strict
candidate audit found 23 previously unused Tier-B clips across 15 classes after
requiring old dual-model top-5/one-top-1, current landmark top-1, current hand top-1,
fixed-pair top-1 with standardized margin at least 0.5, at least 80% observed-hand
frames, at least 80% new hand-crop validity, and a maximum two upgrades per class.
This is a small high-confidence review upgrade, not a reason to admit the 587 unused
clips wholesale or treat correlated model agreement as lexical proof.

`select_local_multimodal_upgrades_v17.py` now reproduces that gate from immutable
logit/crop ledgers and hashes every input. The frozen output contains exactly 23 clips
across 15 classes (maximum two/class) under
`artifacts/reports/local_citizen100_quality_audit/multimodal_teacher_reaudit/upgrade_selection/`.
It remains `training_eligible:false` and requires ASL-fluent exact-variant review;
model agreement was not silently converted into ground truth. Twenty focused tests
pass, all affected scripts compile, and scoped `git diff --check` passes.

The independent confirmation protocol is now frozen in
`docs/guides/PORTRAIT_IPHONE_EVAL_V17.md` with a capture ledger template at
`active/v17/portrait_iphone_capture_template.csv`. The minimum target set is five new
people × two repetitions × 100 exact variants = 1,000 portrait-iPhone clips, plus a
recommended 100 OOV clips. Prompts disappear before capture, natural mouthing is not
forced, objective QC occurs before inference, all current model/weight hashes freeze
before evaluation, and the complete set is evaluated once with signer-clustered
intervals and paired tests. It cannot be used for tuning; after use it becomes a
consumed confirmation set.

## 2026-08-10 10:37 PST — Higher-quality external data audit identifies RIT Sign Bank lead

Official metadata and access paths were audited before acquiring another training
corpus. The evidence is consolidated in
`artifacts/reports/CITIZEN100_EXTERNAL_DATASET_AUDIT.md`. The strongest immediately
auditable video source is the newer ASLLRP/RIT isolated-sign collection: its current
official metadata has 12,197 segmented clips, 13 participant IDs, explicit main-entry
and entry/variant glosses, handshape metadata, and 60 frozen Citizen100 classes with
exact entry/variant-name matches totaling 541 clips. The segmented clips are stored in
two official range-addressable ZIP archives. One byte-range-retrieved smoke clip was
1280x720 H.264 at 30 fps and 1.7 seconds. Apple Vision v17 extracted it in 0.73 seconds
with 96.08% observed hand-frame coverage, 96.88% face/body presence, and a clean
`audit_v17.py` pass. This was a temporary audit clip, not retained project data.

RIT/ASLLRP data are research-only, noncommercial, and non-redistributable under the
Sign Bank terms. Exact text matches are only candidates: the frozen Citizen ASL-LEX
code still must be reconciled with each ASLLRP entry/variant before any clip becomes
training-eligible. A bounded three-participant-per-class audit is the next safe action;
it must remain quarantined until variant review and frozen-model triage pass.

Other official sources are weaker for the current need. ASL-100-RGBD offers 1080p RGB,
4,150 tokens, and 22 fluent/DHH signers, but only 12 direct frozen-label matches plus six
variant families and requires an authorized Databrary account. Legacy ASLLVD has 9,747
tokens from six signers and detailed variants, but its public pre-cut movies stack front
and side views in a 328x656 frame; clean citation-form downloads require login. MS-ASL
has the best secondary signer diversity: 97 name-overlap classes include 2,612 official
train clips from 110 train signer IDs, but nearly all are variable-quality YouTube
segments and exact ASL-LEX variants are not encoded. Google's Kaggle `asl-signs`
competition exposes MediaPipe landmark Parquets rather than raw RGB video, so it cannot
be re-extracted through Apple Vision v17. No dataset video beyond the two temporary
quality smokes was acquired, and PopSign/local selection remains paused.

## 2026-08-10 10:45 PST — RIT exact-name acquisition plan validated

`scripts/download_rit_citizen100_candidates.py` and its three focused unit tests were
added. The downloader reads the official dated RIT metadata, requires literal equality
on the ASLLRP `entry/variant gloss label`, indexes the two 1.7-1.9 GiB official ZIPs by
HTTP byte range, and retrieves only selected members with size and CRC verification.
Every result is quarantined with complete source provenance and
`training_eligible: false` until ASLLRP-to-ASL-LEX variant confirmation. The selection
distinguishes pinned Citizen raw-gloss equality from weaker canonical-label-only
equality and never normalizes or merges variant suffixes.

The focused tests pass. A live dry-run indexed both remote archives successfully and
found 292 unique candidate clips across 60 frozen classes: 249 clips in 51 classes are
literal pinned-raw-gloss matches, while 43 clips in nine classes are canonical-only
candidates because the Citizen raw gloss is a numbered/different variant. All 292
members exist in the official archives. At the user's direction, acquisition order is
RIT and ASL-100-RGBD first, then strictly quality-filtered local raw clips, and only
then another web source if coverage remains insufficient.

At the user's direction, PopSign and local-corpus candidate selection are now paused
while higher-quality external isolated-sign datasets are evaluated first. No PopSign
preview may enter training, and no local supplement has been selected or extracted.

## 2026-08-10 10:54 PST — Selective RIT acquisition completed; ASL-100 access audited

The selective RIT downloader completed 292/292 candidate transfers from the official
segmented archives with member-size, ZIP CRC, decode, and SHA-256 provenance checks.
The retained raw subset is 76,372,759 bytes (73 MiB), all clips decode as 1280x720,
and the candidates span 60 frozen classes and 13 participant IDs. The exact tiers are
249 pinned-Citizen-raw-gloss matches and 43 weaker canonical-label-only matches. Every
clip and the top-level manifest remain `training_eligible: false`; acquisition alone
does not establish lexical equivalence or suitability for training.

Databrary volume 1062 was also inspected through its current public API. The volume
metadata confirms 42 1080p RGB sequences from 22 fluent/DHH signers, but its 22 video
sessions have `releaseLevel: authorized_users`, `nativeAccessible: 0`, and blurred
session metadata without an authorized institutional login. No ASL-100-RGBD media was
downloaded. Keep it queued for authorized access; do not bypass the release gate.

The active order is now: extract and triage the acquired RIT candidates, acquire
ASL-100-RGBD if authorized access becomes available, audit local raw videos and retain
only genuinely strong, diversity-aware, train-only clips under a strict per-class cap,
then search further official web datasets only if coverage remains insufficient.

## 2026-08-10 11:03 PST — Local raw corpus quality shortlist completed

The local raw corpus was audited only after the RIT pass, as directed. A new
fail-closed selector and three focused tests were added in
`scripts/audit_local_citizen100_candidates.py` and
`test/test_local_citizen100_candidates.py`; all tests pass. The selector excludes all
known `msasl_`, `signasl_`, and `wlasl_` files, requires valid isolated-sign duration,
resolution, exposure, sharpness, and a conservative 0.82 composite quality floor, and
uses coarse appearance distance only to avoid near-duplicate recording sessions. It
does not infer or claim signer identity.

Of 17,179 local-style candidates inspected, 356 clips across 89 exact class folders
were shortlisted under a hard cap of four clips per class. This is intentionally much
smaller than Citizen's primary corpus so the local seven-ish recurring people/sessions
cannot dominate. Ten frozen classes have no exact local folder. The `I` folder was
visually confirmed to mix fingerspelled-I and ME/self-reference productions and is
fully quarantined, leaving 89 usable audit folders rather than 90. The shortlist has a
minimum quality score of 0.82, minimum brightness 36.83, and minimum pairwise selected
appearance distance 0.171. All clips remain `training_eligible: false`, train-only
after exact variant review, and are staged as symlinks under
`data/local/local_citizen100_quality_audit_q82/raw/`. A 356-clip visual contact sheet
is at `artifacts/reports/local_citizen100_quality_audit/shortlist_contact_sheet_q82.jpg`.

An earlier 356-symlink pre-floor audit remains under
`data/local/local_citizen100_quality_audit/` because local data must not be deleted
without explicit permission. It is superseded and must not be extracted or trained.
The q82 shortlist is now queued for Apple Vision v17 extraction and frozen-model
mismatch triage; selection itself is not approval.

## 2026-08-10 11:29 PST — Bounded MS-ASL gap acquisition completed

The official Microsoft MS-ASL annotation package was downloaded from Download Center,
SHA-256 `a8562008309eea4129e1bc0ed7f654a314fee195227222859657e307b6434c34`, and
retained under `data/local/dataset_metadata/msasl_official/`. Its C-UDA license and
official train/validation/test annotations are preserved. Only `MSASL_train.json` was
used; validation and test were not accessed for candidate acquisition.

`scripts/download_msasl_citizen100_gap_candidates.py` and five focused tests were
added; all pass. The downloader considers only canonical labels whose text exactly
equals the pinned Citizen raw gloss and that are absent from the 49-class strict local
review shortlist. It requires official annotation resolution of at least 640x360,
isolated segments of 0.4-8 seconds starting within the first 120 source seconds,
unique official train signer IDs per class, and at most three retained clips per
class. Attempts within a class are sequential and stop at the target, so the process
cannot download surplus clips. An isolated current yt-dlp 2026.07.04 environment is
under `artifacts/generated/msasl_download_env`; every retained segment is decode- and
SHA-256-verified.

The bounded pass made 192 attempts and retained 62 clips across 30 classes, totaling
24,710,250 bytes in provenance. Eight classes reached 3/3: BIG, FRIEND, GOOD, HAPPY,
LIKE, MOTHER, SICK, and SIGN; WE also reached 3/3, for nine total target-saturated
classes. COME, GIVE, GO, STOP, and YES had zero currently valid bounded sources. Other
classes retained one or two clips. The exact retained set is materialized as 62
symlinks under `data/local/msasl_citizen100_gap_audit/retained_raw/` and recorded in
`candidate_provenance.json`; all are `training_eligible: false` pending v17 extraction,
frozen-model mismatch triage, and ASL-fluent exact-variant review.

Nineteen successfully downloaded clips from the interrupted pre-120-second-filter run
remain in the broader `raw/` directory but are absent from current provenance and must
not be extracted or trained. They were not deleted because local data deletion requires
explicit permission. Only `retained_raw/` is the active MS-ASL audit input.

## 2026-08-10 11:35 PST — MS-ASL triage completed; remaining high-quality sources are gated

Apple Vision v17 extraction succeeded on all 62 provenance-linked MS-ASL clips with
zero no-hand and zero failed cases; all 62 archives passed `audit_v17.py`. Extraction
quality is strong: median observed-hand-frame coverage is 95.1%, median face presence
90.6%, and median body presence 87.5%. Frozen-model mismatch triage is 56.45% top-1
and 77.42% top-5, with 11 model-consistent, 13 ambiguous, and six high-risk classes.
The model-consistent classes are ANGRY, BIG, FRIEND, GOOD, HAVE, HOT, LIKE, SICK,
UNDERSTAND, WE, and WHY. Zero classes were automatically approved.

The conservative union now covers 63/100 classes: 49 exact-text/model-consistent local
classes, 11 new MS-ASL classes, and three additional exact-tier RIT classes not already
in that union (GIVE, LESS, WHEN). This is candidate coverage only, not approved training
coverage; all sources still require ASL-fluent exact-variant review.

Further official web research found no additional ungated, clearly higher-quality video
source that can be safely acquired now. ASL-LEX reference videos explicitly may not be
saved or used without permission. Sem-Lex is the best next corpus (91,148 videos,
3,149 signs, 41 Deaf participants, expert ASL-LEX/SignBank alignment), but the named
user must submit its Google access form and personally accept CC BY-NC-SA and
community-respect commitments. Purdue RVL-SLLL requires a signed license and issued
credentials. ASL-100-RGBD remains behind Databrary authorized-user access.

Additional ASLLRP metadata was audited without downloading gated video. The 2025
ASLLRP sentence metadata has 1,992 exact-name candidate tokens across 65 frozen classes;
DSP sentence metadata has 317/60, and DSP citation-form metadata has 142/65. The
official ASLLRP interface states that segmented-video downloads require a free login.
Do not use the exposed archive index to bypass that account gate. Metadata SHA-256s and
the current comparison are recorded in
`artifacts/reports/CITIZEN100_EXTERNAL_DATASET_AUDIT.md`.

Focused validation after all acquisition/audit changes passes 23/23 unit tests across
the RIT downloader/triage, local quality/diversity/review gates, and MS-ASL bounded
downloader/triage. All seven new scripts compile, and `git diff --check` passes. Final
on-disk active counts are 292 RIT raw/292 landmarks, 132 strict local review symlinks,
and 62 retained MS-ASL symlinks/62 landmarks. Free disk space is approximately 56 GiB.

## 2026-08-10 12:04 PST — Local 132-clip ceiling corrected; SemLex exact plan started

The earlier 132 local clips were not the total number of strong clips. They resulted
from a deliberately narrow four-candidate audit, a class-consistency screen, and a
three-per-class output cap. At the user's request, the same 0.82 visual-quality floor
and 0.08 appearance-diversity constraint were rerun with a seven-per-class cap, which
selected 623 visually strong candidates across 89 classes from the same 17,179 local
clips. Apple Vision v17 extracted all 623 with zero failures/no-hand cases and
`audit_v17.py` passed all 623 archives with zero errors.

Frozen-model mismatch triage on the expanded set is 52.33% top-1 and 80.26% top-5.
Class triage is 39 model-consistent, 40 ambiguous, and ten high-risk. Applying only
the exact-text, clip top-5, >=80% observed-hand-frame, >=50% face-presence, and
seven-per-class gates yields a 369-clip/77-class human-review pool. Adding the stricter
class-level model-consistency gate yields 209 clips across 38 classes. The selector now
writes both `clip_review_pool.json` (369) and `review_shortlist.json` (209) under
`data/local/local_citizen100_quality_audit_q82_cap7/`; neither is training-approved.
The 209 set is the conservative local supplement, while the wider 369 set must not be
silently promoted without ASL-fluent review.

The user supplied named-access SemLex metadata and six official Google Drive links.
CLI range inspection identified three video archives and three pose archives. Only the
23,673,462,199-byte official train video archive is in scope; SemLex val/test and all
provided pose archives are excluded. Official ASL-LEX 2.0 metadata was downloaded from
OSF under its CC BY-NC license (`signdata.csv` SHA-256
`080ecc3de4b307a04dd9b2c2583c22bc623a5c731ab869d6e3905d4a3540fbd3`). The supplied
SemLex metadata SHA-256 is
`f4250b2877e738f028e4b9517922952a350ec8952e59d53ceabdd778938fc3c3`.

`scripts/prepare_semlex_citizen100_candidates.py` performs an exact join from each
pinned Citizen ASL-LEX code through the official ASL-LEX EntryID to SemLex `asllex`
labels; English-name similarity and free-text labels are never accepted. The initial
five-signer cap (486 clips) was only a conservative starting point and was superseded
after the user questioned it. Acquisition and training balance are now separate: the
active acquisition plan retains all 1,624 exact matched clips (one clip per official
SemLex signer/class, 98 classes, 32 train signers), while the first balanced training
subset retains at most 12 distinct SemLex train signers per class (1,091 clips, 98
classes, 29 signers). The latter roughly matches Citizen's 11–16 train signers per
class while remaining below Citizen's 1,475 training clips overall. `THEY` is excluded
because Citizen pins
`they_2` while SemLex train provides `they_1`; `CHILD` is excluded because the SemLex
entry is `children`, not the pinned `child` entry. The full acquisition plan is under
`data/local/semlex_citizen100_train_audit/selection_plan.json`; the balanced subset is
under `data/local/semlex_citizen100_train_audit/balanced_cap12/selection_plan.json`.
Both remain `training_eligible:false` until extraction and quality triage complete.

`scripts/download_semlex_citizen100_candidates.py` and five focused SemLex planning /
download tests were added and pass. Google packages train as one gzip stream, so member
bytes cannot be remotely selected. A resume-safe parallel CLI range transfer is in
progress on the internal SSD; after the full transport passes range/length checks, the
script will extract and decode only the planned 1,624 WebM members and remove the
23.67 GB transport archive. Do not treat the transfer's sparse transport file as a
dataset or use any SemLex clip until selective extraction, v17 audit, and mismatch
triage complete.

Focused validation passes 15/15 tests across the local selector/triage/two-level
review outputs and SemLex exact mapping/range-member logic. All five involved scripts
compile and `git diff --check` passes. At 12:10 PST the throttled Google transport had
29/2,823 independently validated 8 MiB ranges complete (243,269,632 validated bytes)
with additional in-flight sparse-file bytes; the resume state is
`transport/train.tar.gz.ranges.json` and the CLI process remains active.

At 12:35 PST the first high-concurrency transfer process was stopped after Google read
timeouts; its 95 independently completed 8 MiB ranges were preserved. The same sparse
archive/state resumed with 16 workers and 20 per-range attempts. Before final resume,
the acquisition plan was expanded from the provisional five-signer cap to all 1,624
exact one-per-signer/class clips, while the separate 1,091-clip cap-12 training subset
was preserved. No validated range was redownloaded and eventual selective extraction
will use the full acquisition pool.

At 12:40 PST the user chose to download the official SemLex train video archive
manually. The active CLI range transfer was terminated (exit 143), and process checks
confirmed no SemLex downloader remained. At the user's explicit deletion request, the
sparse `transport/train.tar.gz`, its `train.tar.gz.ranges.json` checkpoint, and the
earlier `train_prefix_64m.tar.gz.partial` probe were permanently removed. No SemLex raw
video had yet been selectively extracted. The official metadata, full 1,624-clip
acquisition plan, and balanced 1,091-clip training plan were preserved. When the user
places a clean `train.tar.gz` locally, continue with selective extraction only; do not
start another Drive download.

## 2026-08-10 14:55 PST — SemLex validation/test acquisition deferred past augmented v1

The official metadata was re-counted for the 98 exact SemLex matches. Train contains
5,539 rows / 1,624 unique signer-class pairs from 32 signers; validation contains
1,861 rows / 984 unique signer-class pairs from 32 signers; and test contains 1,618
rows / 444 unique signer-class pairs across 95 matched classes from ten signers.
SemLex train and validation overlap on 31 of their 32 signer IDs, and 822/984 validation
signer-class pairs already occur in train. Test has zero signer overlap with either.

Therefore validation/test are not free additional training data. The controlled
augmented-v1 experiment remains Citizen official train plus the quality-balanced
SemLex-train supplement, selected on Citizen official validation. SemLex validation
may be downloaded after v1 for a secondary within-domain diagnostic and can only be
promoted into a later training version after its diagnostic role is finished. SemLex
test must remain protected until a final frozen candidate needs a one-time unseen-
SemLex-signer evaluation; sacrificing it for training is especially unsound because
the Citizen official test has already been consumed. No validation or test archive was
downloaded during this decision.

## 2026-08-10 16:15 PST — Exact-only local and uncapped-clean SemLex pools expanded

The earlier 623 local clips were a diversity cap, not the total mechanically valid
inventory. At the user's direction, local selection was expanded from seven to 14 per
class while making the lexical gate stricter. The selector now supports
`--exact-pinned-raw-only` and rejects canonical-folder/pinned-raw-gloss inequality
without normalization (for example, `DRINK` cannot stand in for `DRINK2`). It retained
1,021 visually strong, appearance-diverse clips across 77 exact-text classes from
14,883 inspected local-style candidates. The 2,749 files in non-exact pinned-raw
classes were quarantined before feature extraction. The selected count is below the
1,078 theoretical cap because 57 candidates were too appearance-similar to add.

The expanded local set reused 539 schema-validated archives from the cap-seven audit
and extracted 482 new videos. Extraction completed 482/482 with zero failures and zero
no-hand clips; the full 1,021-archive v17 audit passed with zero errors. All 1,021 raw
SHA-256 values are unique. Citizen-only model agreement is 56.32% top-1 / 84.13% top-5;
the balanced Citizen+SemLex model reaches 65.03% / 85.99%. Dual-model consensus plus
the 80% observed-hand and 50% face gates yields 434 Tier-A dual-top-1 clips and 154
Tier-B dual-top-5/one-top-1 clips: 588 priority clips across 76 classes, 1-14/class
with median eight. Another 85 dual-top-5-only clips remain Tier C; 203 extraction-
quality and 145 model-disagreement clips remain quarantined. Outputs are under
`artifacts/reports/local_citizen100_quality_audit/cap14_exact_consensus/`. Local A/B
clips remain train-only candidates rather than automatically approved labels because
folder equality plus correlated model agreement cannot prove the exact ASL variant or
signer identity; ASL-fluent review remains the final gate.

Exact raw-hash decontamination found zero overlap between all 1,021 expanded local
clips and the 3,102 official Citizen provenance rows, including zero Citizen validation
or test overlap. It also found zero overlap with all 1,499 retained SemLex-train raw
hashes. Thus the local shortlist contains no byte-identical copy of either primary
source; this does not convert unknown local signer identities into an evaluation set.

The SemLex cap of 12 signers/class was also removed without overwriting the immutable
1,058-clip manifest used by earlier checkpoints. The new selector accepts cap zero as
all distinct quality-passing signers, verifies every source is SemLex train with an
exact `asllex` entry/label identity, retains the existing `TAKE` mismatch quarantine,
and requires at least 70% observed-hand frames, 30% hand-node presence, and 50% face
presence. `full_clean_train_candidates.json` contains 1,388 unique-hash clips across
97 classes and all 32 SemLex train signers, with median 14 and range 2-25 clips/class.
Relative to cap-12, 37 weaker clips were dropped and 367 stronger signer/class clips
were added, a net +330. All 1,388 feature archives pass the v17 audit and the real
Stage-1 supplement loader returns exactly 1,388 while excluding `TAKE`.

The immediately safe next controlled run is 1,475 Citizen train plus 1,388 full-clean
SemLex train = 2,863 clips with online class/source-balanced sampling and the unchanged
378-clip Citizen validation. The 588 local A/B candidates would raise the pool to
3,451 only after exact-variant human approval and require a three-source sampler so the
unknown local sessions cannot dominate. Citizen test and SemLex test remain sealed.

## 2026-08-10 16:38 PST — Full-clean SemLex run reaches 95.77% Citizen validation

The controlled full-clean experiment used exactly 1,475 Citizen train plus 1,388
quality-gated exact-ASL-LEX SemLex train clips, the fixed 378-clip Citizen validation,
seed 1701, the same d=256/depth=4 architecture and augmentation, and the same online
class/source-balanced sampler as the cap-12 winner. No local candidate entered this
run. It completed 123 epochs and stopped after the declared 30 stale epochs; a late
series of genuine improvements moved the best checkpoint to epoch 93.

The retained checkpoint at
`artifacts/models/stage1_v17_citizen_semlex_full_clean_balanced/best_model.pth`
achieves 95.77% top-1 (362/378), 100.00% top-5, and 95.51% macro F1. This is +1.85
points / seven clips over the 93.92% cap-12 balanced model and +2.65 points / ten clips
over the 93.12% Citizen-only baseline. Relative to cap-12 it corrected eleven clips
and regressed four; relative to Citizen-only it corrected sixteen and regressed six,
so neither gain is an identical-prediction artifact. Sixteen Citizen-validation errors
remain, led only by `THANKYOU -> GOOD` twice; every other confusion occurs once.

The checkpoint and result provenance record 2,863 train clips, the exact full-clean
manifest SHA-256 `09f22aeff491ac16a498cbf5be02eb9867ddb0b16c139b1e302571d2bd51883a`,
50/50 expected Citizen/SemLex exposure, exactly 1% expected exposure per class, and
false Citizen-test/SemLex-test access. The immutable validation report is under
`artifacts/reports/stage1_v17_citizen_semlex_full_clean_balanced_validation/`. This
checkpoint is the new validation winner, but it must not be evaluated on the already
consumed Citizen test during development. The next independent model-quality gate is
SemLex validation after its manual archive download and selective exact-variant
extraction; SemLex test remains sealed. The checkpoint SHA-256 is
`23d5b7f1b343a6b5246e4afe03c0ab99067d3dac6e355024b5c53e5c31013f8e`.
Twenty-two focused local/SemLex/Stage-1 tests pass, the affected scripts compile, and
`git diff --check` passes.

## 2026-08-10 17:19 PST — SemLex validation and d=384 mobile ablation completed

The user's manual `/Users/frnzlo/Downloads/val.tar.gz` is the exact expected
8,076,365,890-byte SemLex validation video archive. Its SHA-256 is
`6eca70a5761f2bfeea5d4f58b1ed34431f2d1be39a20947f01a27e4a22516b90`; both gzip
identification and a complete `gzip -t` passed. The original archive was preserved.
SemLex planning/extraction now accepts an explicit split and locks validation/test as
`evaluation_only_never_training`; the Stage-1 train loader still rejects non-train
supplements.

The frozen exact-ASL-LEX validation plan selected one clip per official signer/class:
984 requested clips across 98 classes and 32 validation signers. `THEY` and `CHILD`
have no exact validation entry. Selective extraction retained 978 clips and quarantined
six VP9 files that failed complete decode validation. Apple Vision extracted 978/978
with zero failures/no-hand clips, and the full v17 schema audit passed. This diagnostic
is cross-domain but not signer-independent: 31/32 SemLex validation signer identities
also occur in SemLex train.

On the identical 978-clip diagnostic, Citizen-only d=256 scores 73.72% top-1 / 88.75%
top-5 / 70.56% present-class macro F1; cap-12 SemLex d=256 scores 82.41% / 95.81% /
79.61%; and full-clean SemLex d=256 scores 85.89% / 96.11% / 82.60%. This independently
supports the larger clean SemLex train pool. Reports are under
`artifacts/reports/semlex_citizen100_val_audit/`; SemLex test remains untouched.

The controlled d=384 challenger changed only model width from the d=256 winner. It
used the same 1,475 Citizen + 1,388 full-clean SemLex train clips, fixed Citizen
validation, class/source-balanced sampler, augmentation, seed, schedule, and patience.
It has 14,338,853 parameters versus 6,470,885 (2.22x), completed 81 epochs, and retained
epoch 51 at 95.24% Citizen-validation top-1 (360/378), 99.74% top-5, and 94.90% macro
F1. The d=256 winner remains better at 95.77% / 100% / 95.51%. Paired Citizen outcomes
are four d=384 corrections and six regressions (exact McNemar p=0.754). On SemLex
validation d=384 reaches 86.71% / 96.01% / 84.52%, only eight net clips above d=256;
it has 40 corrections and 32 regressions (p=0.410). Neither difference is statistically
persuasive, and the more trustworthy primary gate favors d=256. No test split was
accessed by either run.

Both checkpoints were successfully converted through the same fixed-shape iOS 15
FP16 ML Program path with trace/manual-attention parity and matching Core ML top-1.
The d=256 package is 12.58 MiB; d=384 is 27.59 MiB (2.19x). Warm current-Mac batch-one
latency ratios for d=384 versus d=256 are 1.79x CPU-only (1.007/0.564 ms), 1.56x
CPU+Neural Engine (0.770/0.495 ms), and 1.27x CPU+GPU (6.084/4.778 ms). These are
desktop proxy measurements, not low-end/medium iPhone latency, memory, or sustained
thermal evidence. Export outputs are under `artifacts/generated/coreml_v17_comparison/`.

Decision: retain full-clean SemLex d=256 as the Stage-1 validation/mobile winner.
d=384 is rejected because it is larger/slower without a reliable accuracy gain. The
588 local A/B clips were deliberately not used in this architecture ablation because
their exact ASL variant still lacks human confirmation. The next data action is
ASL-fluent review of those local candidates; the next deployment action is real-device
Core ML latency/memory/thermal testing on target iPhones.

## 2026-08-10 09:40 PST — Direct Citizen100 augmentation exhausted; broad pretraining opened

The Citizen acquisition, manifest, raw/extractor/quality reports, Stage 1 validation
report/history, cached official split metadata, local raw inventory, and current
Microsoft release page were re-audited before attempting an accuracy-data expansion.
ASL Citizen v1.0 remains the latest official release: 83,399 videos, 2,731 signs, and
52 signers with fixed signer-disjoint splits. The 100 pinned raw-gloss/ASL-LEX pairs
match exactly 3,102 official metadata rows; all 3,102 MP4s are already local and zero
selected exact-variant files are missing. The prior downloader already records
3,102/3,102 verified members with zero failures and zero remaining output bytes.
A fresh downloader dry-run against the official archive independently reported
1.5883 GiB existing output and exactly 0.0 GiB remaining.

Therefore there is no safe additional Citizen download for augmenting the existing
100 class labels. Fourteen remaining same-name raw-gloss/ASL-LEX pairs across 13
concept names are different lexical or numeric variants. They total 188 train, 49
validation, and 161 test clips and cannot be merged into the pinned classes; W.H.A.T is
fingerspelling and its 30 clips are already quarantined. The validation log has 26
errors on 378 clips, led by I -> WE and ANSWER -> GO twice each, but Citizen contains
no further pinned exact-variant examples for those classes. No video was downloaded:
doing so would either duplicate local data, leak the consumed official test signers,
or change class semantics. Evidence and the three safe expansion choices are recorded
in `artifacts/reports/CITIZEN100_V17_EXPANSION_AUDIT.md`.

The user correctly clarified that the 3,102 selected clips are not all videos recorded
by those official Citizen signers. All other signs can be used for broad representation
pretraining without pretending they are examples of the current 100 labels. The full
official train/validation pool is 40,154/10,304 videos covering 2,731 exact raw-gloss
plus ASL-LEX pairs from 35/6 signer-disjoint participants. Compressed transfer is
21.25/5.37 GiB and retained raw video would require 22.90/5.83 GiB, exceeding the
approximately 16 GiB free host space.

`scripts/extract_citizen2731_v17.py` temporarily provided a resume-safe,
storage-bounded route:
it accepts only train/validation, range-downloads one verified official ZIP member,
checks size and CRC, extracts Apple v17 landmarks from a temporary video, saves compact
features plus source provenance, and removes the temporary transfer copy. Four bounded
download workers overlap network fetches with the single Vision detector. The frozen
pretraining manifest is `active/v17/citizen2731_pretrain_manifest.json`, SHA-256
`7c827f8d71f4dca7266070e28f4b1ed74927ad3699ac7a4c30c103fe4ea5e203`; it has 2,731
classes and preserves exact pair identity, including separate RESEARCH1/RESEARCH2
classes despite their shared Citizen code `B_03_084`. Three focused tests cover exact
pair construction, the shared-code edge case, and fail-closed test rejection.

The first 15 streamed train clips extracted successfully at schema level with zero
failures and zero no-hand results. After enabling bounded prefetch, the 12 new clips in
the resume smoke completed at 1.68 clips/second; all three earlier smoke outputs were
recognized as existing. Raw video was not retained and Citizen test was not accessed.
Seventeen focused Citizen/Apple tests pass, and `audit_v17.py` passes all 15 smoke
archives. A full 50,458-clip train/validation stream was then started based on the
interpretation that the user's request for the signers' other videos authorized broad
2,731-class pretraining. The user questioned that expanded scope, so the process was
immediately interrupted at 299 processed items. It left 298 compact train landmark
archives, one provenance-recorded no-hand result, zero retained temporary/raw videos,
and no active extraction process under `data/local/citizen2731_v17/`. At the user's
explicit request, the entire 4.5 MiB directory (301 files including metadata/events)
was moved recoverably to
`/Users/frnzlo/.Trash/SLT_citizen2731_v17_20260810_0955`. The temporary extractor,
its focused test, and the 2,731-class manifest were removed from the worktree. The
project is again scoped strictly to the frozen 100 classes.

## 2026-08-09 18:11 PST — 72-video ASL Citizen v17 extraction

Inputs: train/validation/test, 24 videos per split, under
`data/local/ios100_audit/asl_citizen/`.

Output: `data/local/ios100_audit/landmarks_v17/`.

Result: 72/72 extracted, zero failures, zero no-hand clips. The final test split ran at
about 2.62 videos/second on the current Apple Silicon development machine.

## 2026-08-09 18:33 PST — PopSign one-hand scope clarified

The official PopSign paper states that the game has the user hold/control the phone with
one hand while the other performs the sign, that the dataset focuses on one-handed
smartphone signs, and that a general recognizer would additionally require two-handed
signs and broader viewpoints. Decision: continue using PopSign as the primary v17
accuracy corpus, but constrain the first product claim accordingly and plan a reviewed
two-handed expansion later. Source:
`https://signdata.cc.gatech.edu/res/doc/popsign_v1_0/popsign_v1_0_supplemental.pdf`.

## 2026-08-09 18:36 PST — primary dataset recommendation changed

PopSign-only training was rejected as insufficiently representative of ordinary
one- and two-handed ASL. Metadata was recalculated without combining dataset identity
counts: ASL Citizen alone has 3 signs at a strict 20 train / 5 validation / 5 test
signer-per-class floor, 45 at 15/4/5, 123 at 10/4/5, and 208 at 10/3/5. Decision:
recommend a 100-sign Citizen-only baseline chosen from the 123 signs meeting 10/4/5,
preserving the official 35-train/6-validation/11-test-signer split. Repeated takes are
not required when unavailable; cross-signer diversity is the higher priority.

This entry records the initial normalized-label metadata scan. It was superseded by
the exact raw-gloss/ASL-LEX selection below, whose locked floor is 10/3/5 and whose
actual split identity counts must come from the frozen manifest rather than the
dataset-wide identity totals.

The active PopSign `test/thankyou` transfer was stopped without deleting recoverable
partials. Stored bytes: 25,165,824-byte prefix plus four range parts of 148,417,024,
33,554,432, 25,165,824, and 125,829,120 bytes under
`data/local/popsign_v17_archives/test/`. No PopSign video was extracted.

## 2026-08-09 19:22 PST — full Citizen100 v17 extraction complete

Extraction completed for the entire corrected manifest. The feature root is
`data/local/citizen100_v17/landmarks/`. Valid archive counts are 1,476 train, 378
validation, and 1,247 test (3,101 total). One test source,
`test/HE/3500609473112364-HE.mp4`, was correctly rejected because all 14 frames contain
no visible hand; its contact sheet is
`artifacts/generated/v17_diagnostics/he_no_hands_contact.jpg`. The rejection is recorded
in `data/local/citizen100_v17/rejections.csv` and is not treated as an extractor bug.

`active/v17/audit_v17.py` reports PASS for all 3,101 archives: every file loads with the
current schema and satisfies shape, finite-value, binary-presence, missing-spatial-zero,
and missing-confidence-zero invariants. There are zero load/schema errors. Median
extraction time was 0.6607 seconds/video. Median detected hand-frame coverage increased
from 0.4667 before activity trimming to 0.875 afterward. Median hand/face/body presence
was 0.4465/0.8125/0.5312. Normalization used shoulders for 2,768 clips and palm fallback
for 333. Corrected chirality counts were 56,823 left, 81,899 right, and zero unknown.
Outputs: `artifacts/reports/CITIZEN100_V17_EXTRACTOR_AUDIT.md` and
`artifacts/reports/citizen100_v17_extractor_audit.csv`.

A train `SLEEP` clip with only four sampled hand detections was manually reviewed. It is
mostly idle and begins signing only at the end, so it is documented in the rejection
ledger while its raw video and already extracted feature remain preserved for traceable
review. A separate `BAD` clip with nine detections was visually valid and retained;
therefore no blanket hand-coverage threshold was introduced.

Focused validation passed 17 tests across extractor geometry/real Vision, Citizen
manifest/downloader, and the optional PopSign downloader. Python compilation, the v17
compatibility CLI help, and the scoped diff whitespace check also passed at this
milestone.

## 2026-08-09 22:13 PST — Hand-aware real-pixel crop corpus complete

The new branch is schema-isolated from both landmark archives and the rejected
full-frame embeddings. `extract_hand_rgb_v17.py` uses the selected Apple Vision
detector only to assign anatomical left/right hands and derive crop boxes on the same
16 raw-frame positions and frozen hand-activity interval used by the earlier RGB run.
It stores actual upright source pixels as JPEG byte blobs with explicit offsets, plus
left/right/union validity, normalized boxes, detected-joint counts, contact flags, and
source indices. Invalid views contain no JPEG, decode to exact zero, and remain
`valid=false`; no landmark, box, or RGB content is hallucinated. An overlap-aware union
view preserves two-hand contact and broader spatial context. The schema fingerprint is
`bf6508de2ea851a4`.

All allowed Citizen train/validation clips were extracted: 1,475 train in 206.9 seconds
and 378 validation in 53.9 seconds. Test is not an accepted split and was not accessed.
There are no empty clips. Aggregate valid fractions are left/right/union
42.20%/72.65%/83.49% for train and 49.92%/65.31%/84.56% for validation. Two separately
detected hands occur in 31.36%/30.67% of train/validation sampled frames; crop-box
contact occurs in 22.33%/21.30%. Similar rates across splits are a useful distribution
check. The corpus occupies approximately 669 MiB, with mean packed JPEG payloads of
378 KiB train and 398 KiB validation. A real validation contact sheet was visually
inspected: individual crops materially enlarge fingers, union crops retain body/two-hand
context, and explicitly missing boundary frames are black and masked.

New implementation files are `schema_hand_rgb_v17.py`, `extract_hand_rgb_v17.py`,
`schema_hand_mobileclip2_v17.py`, `extract_hand_mobileclip2_v17.py`,
`model_hand_mobileclip2_v17.py`, and `train_stage_1_hand_mobileclip2_v17.py`. Four crop
geometry/packing/schema tests pass. The frozen high-resolution hand-embedding extraction
and classifier smoke/full runs are the next active gate; the fine-tuned temporal visual
model must not be judged from crop availability alone.

## 2026-08-09 22:45 PST — Pre-pooling spatial corpus complete and fully audited

All 1,475 training and 378 validation clips were cached from the official
MobileCLIP2-S0 FastViT stage-3 output before global pooling. Extraction took 637.6
seconds for train and 174.5 seconds for validation on MPS, with zero skips and schema
fingerprint `530061b1c5dfcabf`. A full readback audit decompressed every archive and
found zero shape, dtype, schema, finite-value, or invalid-view-zero violations. The
valid-view fractions are 66.11% train and 66.60% validation; mean valid-map absolute
activation is 0.03233 and 0.03213 respectively. The close split statistics are a useful
distribution check. The cache occupies 3.4 GiB and leaves approximately 16 GiB free.
It is temporary training data and is not required by an eventual phone runtime. The
Citizen test split was not accessed.
