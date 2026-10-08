# capstone-paper — log

Measured results, rejected approaches, and progress snapshots. Not read start-to-end —
`rg` this file before re-running an experiment to see if it already failed.

4 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## 2026-10-08 — User-authorized main publication completed

User authorized pushing main after the read-only publication checks. Normal fast-forward
push advanced origin/main from 26884461 to af9b864c: 41 previously unpublished commits,
including the 25 dated work commits and 16 earlier commits. Live ls-remote verification
matches local main. The 25 work commits use kokoab <batiancelafrancis747@gmail.com> as
author and committer, preserving their logged dates and messages; repository-local
identity uses the same email. No force push or runtime/phone deployment.
Pre-push evidence: dry run passed, largest new blob 2.2 MiB, no new blob above 100 MiB,
no checked-in build/deployment workflow, and 125 focused tests previously passed.
Local datasets, weights, caches, incidental permissions and unrelated files remain local.
This publication record is committed and pushed separately without rewriting published
history. Next safe action: verify main/origin/main synchronization; continue development
under the existing training, evaluation and manuscript-review gates.

## 2026-10-08 — User-authorized date/topic commit series completed on main

User authorized committing the outstanding work as 25 chronological date/topic groups
from September 25 through October 8. Both Git author and committer dates follow the
logged Manila calendar date; date-only entries use ordering times, not claimed event
timestamps. Source files spanning multiple days use the available final snapshot in
the latest relevant group; missing intermediate source versions are not reconstructed.
Eligible source, tests, manifests, documentation and compact report evidence are selected
explicitly. Incidental executable-bit changes, AppleDouble files, local datasets,
weights, caches, large result files and unrelated personal files remain local.
The .gitattributes whitespace rules preserve generated SVG formatting and blank final
lines in saved baseline source snapshots, without rewriting manuscript or evidence files.
Focused validation: 125 tests pass across live Reel/app integration, temporal boundary,
segmental runtime, fingerspelling trigger, Stage 3 composition, legacy fine-tune config
and MediaPipe extraction, plus the Stage 3 corpus, hand RGB and unified-model suites;
git diff --check passes. No training or protected test access.
Completed all 25 local date/topic commits. User then requested main: its prior tip
26884461 was an ancestor, so main was fast-forwarded to the full committed history
and selected as the active branch, preserving working files, permissions and the index.
No remote push. A redundant full-worktree content scan was stopped because it rehashes
unrelated tracked local assets after the index refresh; final verification uses commit
metadata, tree paths/modes and the explicitly selected source/documentation snapshots.
Final log amendment initially failed with ENOSPC. Automatic Git maintenance had left
25 abandoned temporary packs created during this operation (9.86 GiB); only those
session-created tmp_pack files were removed. Older temporary packs and all canonical
packs were preserved. Retry disables automatic maintenance for that command only.
Final audit passes: 25 commits have matching author/committer dates in +08:00;
802 selected snapshots match the committed tree and working files. No staged changes,
mode changes, local datasets, weights or AppleDouble files entered the series, and
the complete commit-series whitespace check passes.
User corrected the Git identity to kokoab <batiancelafrancis747@gmail.com>.
All 25 commits are rewritten with that author and committer while preserving their
logged dates and messages; repository-local identity is set for future commits.
Source trees are unchanged except this required decision record in the final commit.
Next safe action: review the local main series; preserve excluded local assets and
unrelated changes.

## 2026-10-08 — Exact before/after review checklist and chart preview delivered

User requested exact proposed wording and chart changes in reviewMD with checkboxes.
Refreshed live canonical tabt.pzqwjycim1ix; revision unchanged from sweep. Updated
README_PLATFORM_NEUTRAL_REVIEW.md with20unchecked items, exactbeforeextracts, afterprose,
replacementtables, chartremovals, andverificationdependencies. Earlier checklist and
allauthornotes preserved in review_assets/20261008_proposals/README_PLATFORM_NEUTRAL_
REVIEW_20261007_archive.md. Four currentnativefigures copied asbeforepreviews; actual
newhistory generates proposed_recognizer_training.png/svg (selectedepoch4,31.42WER),
visuallychecked. Sourcegeneratorincluded; link/checklistverificationpasses.
Explained evaluation umbrella vsvalidation/tuning vstest. Use95.53validationaccuracy,
no“pooled”/datasetcount inpaperprose; one31.42developmentWER. Signer-disjointsection
preservedperuser, no newclaim aggregate is signer-disjoint. FP16complete-datasetvalues
andHELLOtrace remainexplicitlypending, notfabricated. ProposedremovalofredundantFigure15
andprecisionchart. NoGoogleDoc/manuscript/chartreplacement orproductionchanges.
Next: author checks specificitems; carryoutapproved verification beforeapplyingdependent
claims. Do not treat checkboxes as alreadychecked by prior general approval.

## 2026-10-07 — Live paper sweep: one final result, proposal only

Read canonical GoogleDoc tabt.pzqwjycim1ix ((Oct6)Revision) via connector; examined
1039text paragraphs and native current Figures11/13/14/15. Saved targeted text/revision
snapshot to canonical_recognition_comparison_v17_20261007/paper_sweep_snapshot.json.
96.31 does NOT occur in current tab. OldrecognizerTable18 is95.24FP32/94.97FP16.
User prefers one final result and no August-to-current comparison. Recommend95.53%
pooled isolated validation, accurately labeled; optional94.40equal-domain average for
balance, not choose a smaller number to simulate liveaccuracy. No blanket signer-disjoint
claim for pooledvalidation: scopeSection4.3.1 andTable10 toprimaryofficialsplit.
Propose replace4.4.1 oldpipeline scores/Table11/Figure11 with finalrecognizerreport;
Table12 learnedcombination96.30 is stale and fixedfusionnowepoch0. Tables13/14 concern
base-modelselection, cannot substituteoverall95.53 into component rows. Preserve only
needed architecturejustification with its ownscope. Table15/4.4.3 9.68WER oldcontaminated
configuration mustnotbeclaimednewheldout; use31.42tuningWER labeleddevelopment only.
Figure15 mustuse newepoch4history, not old27.43minimum. Figure14/96.83epoch100 remains
valid base-traininghistory, not finalpooledaccuracy. Section4.4.6/Table18 conversion now
97.09bothprecisions onprimaryvalidation ONLY; fullpooledFP16parity remainsunmeasured.
Recognizerpackage50.28/25.51MB different definitionfromold49.62savedweights; keepboundary
size definitionsseparate. Figure13/paragraph24.1vs112.6 areoldfullprecisionbenchmark;
new23.69vs59.15 changesrecognizeronly, cannotrelabelexistingallFP32comparison.
For singlefinalentry retain23.69ms/frame andremoveoldcomparisonchart; explain scope.
Section2.2 deployment saysMac-baseddespitedevicetests; qualify measurediPhoneprototype,
notproductionreadiness. HELLOtimelineFig2 fromoldrecordneedsnewtracebeforeattribution.
EnglishTable16/Fig16 andboundaryresults have no replacement fromrecognizertraining;
retainwithseparatescope. Participantplans/schedule/literature untouched. No manuscript,
reviewREADME, or chart asset edits. Next: discuss proposal and any missingmatchedchecks.

## 2026-10-07 — Author requests all-comparison checkpoint audit before edits

Explicit user override: CHAT FIRST; no further manuscript/asset writes. Read current
working tab and source scripts/results. Tables11/Figure11 use orientation-robust base
95.77 then unified96.30/span95.24; Table12 uses unified's source landmark branch95.50
and hand80.69, not the same95.77checkpoint. Unified checkpoint pins landmark
stage1_v17_local_deep_clean_mouth_masked_replay_ft_v1/best_promotion_gate_model.pth
(hash12a74a18...) and handstage1_v17_hand_mobileclip2_local_deep_clean_replay_ft_v1
(hash7e920d09...). Table13/Figure14 use historicalpartwise96.83 epoch100; Table14 uses
matchedfamilySqueezeformer96.30 (distinct from unified96.30) andTransformer95.50
(distinct from Table12landmark95.50). Figure11 builder hardcodes accuracy separately
from timingcheckpoint paths; should derive both from pinned result records.
Table18FP32/FP16 source checkpoint matches115286872e1e026b5352f00e53eb0365008c55dab28188437a59e26b277be450
in both conversion reports. Figure15 uses same spanmodel history but tuning WER, not
Table15streaming metric. EnglishTable16/Figure16 source same tiny run2 study; full
cross-runtime hashverification remains needed before declaring entire bundle pinned.
Table15score_test_C has no direct checkpoint metadata; resolve native replay bundle/
config hash linkage before canonical adoption. Figure13 latestprecision protocol must
remain separatefromhistorical28ms. Requirements/schedule/literaturetables anddiagrams
aren't model checkpoint comparisons. No new evaluation/training or paper updates in
this audit. Next discuss canonical per-stage checkpoint registry and whether to retain
historicalscreening; single coherent newtraininglineage needs downstreamretraining,
not arbitrary numerical replacement or revalidation ofdifferentweights.

## 2026-10-07 — Clarify historical architecture versus matched family tables

User queried Table13 96.83%/6.50ms versus Table14 96.30%/6.15ms, and confirmed
intended finding is higher Squeezeformer accuracy with similar FP16 phone classifier
speed. Source README explicitly distinguishes earlier design-selection checkpoint
from matched Mac family rerun. Working Google Doc tab t.pzqwjycim1ix captions and
list entries renamed Earlier Landmark Architecture Screening and Matched Recognition
Family Comparison. Intro now explains different trained models before tables.
Matched discussion retains Mac2.88vs6.15ms and adds phoneFP16Transformer0.95vs
Squeezeformer0.99ms, preserved validationpredictions95.50vs96.30%, four-run variability,
classifier-only scope separatefrom28ms. Accuracy gap computed fromunroundedmetrics0.79pp.
No numerical rows substituted, no new plots or training. First revision-guardedwrite
rejectedstale revision; freshtextverified exacttargets and secondwritepassed. Connector
readbackverified sixreplacements. Text-onlyedit; renderedpaginationnotrechecked.

## 2026-10-07 — Redundant charts removed before Transformer follow-up

User requested chart cleanup first and immediate return before Transformer testing.
Removed old Figures12–16 (input, architecture, family, streaming, English comparisons),
retaining all tables and result explanations. Old17–21renumbered12–16. Updated body
references, list entries and rendered page references. Latest Google Doc target remains
t.pzqwjycim1ix. Readback confirms16figure captions, no17–21references, EnglishTable16
retained. Inspected affected exported pages90,92,95; no clipped content. Targeted
visual review only. Updated local review checklist. No model/benchmark/training work.
Next: discuss and scope proper matched Transformer deployment evaluation with author.

## 2026-10-07 — Transformer alternative and chart necessity audited

Discussion only; no manuscript, visual, model or deployment changes. Rechecked saved
family result and benchmark implementation: flat Transformer95.50%Top1/2.882ms,
part-wise+global Squeezeformer96.30%/6.146ms; matched training protocol, one seed,
FP32 prepared-landmark CPU inference. Difference0.80points, about2.13x classifier
time. Supports an accuracy-first tradeoff, not proven mobile superiority or statistical
robustness. Compared model families also differ in regional input organization.
These base-model results are not directly comparable with interval-adapted recognizer
FP32/FP16conversion95.24/94.97. No saved matched complete Transformer mobile deployment
comparison was identified in the relevant records. Avoid assuming it removes FP16 need:
existing phone encoder-only FP32 substitution87.29ms versus current24.05ms demonstrates
other major precision-sensitive work; do not add/subtract component substitution effects.
The primary Squeezeformer paper concerns speech/Conformer efficiency, not proof of
faster execution than this flat landmark Transformer.
Recommendation for discussion: retain one model-choice table, remove duplicated charts,
keep concise English quality evidence because translation is in scope but omit its
redundant bar chart. English60%fully-correct/6vs8%wrong scores are automatic judgments
on generated gloss/reference sessions, not human or full camera-to-English evaluation.
Next: discuss reduced result structure and, if pursued, plan a matched Transformer
fusion/interval/mobile comparison; no training/test gate or deployment change implied.

## 2026-10-07 — Approved manuscript and visuals complete

Applied author-approved revisions in Google Doc1sC2JZ23mSpnLWeEEFDm5ce2x_As0XTS7ayMGvk5uuD4,
tab t.pzqwjycim1ix. Final read reports title (Oct 6) Revision; task did not rename it.
Slides and other tabs untouched by task. Updated review checklist, writing guide,
binding decisions and current-state record. Current deployed FP16 configuration retained.
Verified21figure captions and20inline images including both new comparison charts;
old language-screening image absent. Text checks found no validation-set count,
forbidden device addition, old timing estimates or unwanted OS-specific narrative.
Resolved74navigation entries from exported page numbers and corrected the ambiguous
Design heading lookup (activity-table cell versus actual Design section).
PDF inspection of revised figures/tables/front matter corrected table cell spacing,
one joined paragraph, and changed-paragraph font consistency. Targeted page review,
not full549-page all-tabs visual certification. No remaining requested work deferred.
Final report:artifacts/generated/paper_revision_20261007/IMPLEMENTATION_REPORT.md.
Source assets/builders underdocs/capstone_papers/review_assets; Python compile and
 git diff --check passed. Next safe action:author reviews revised manuscript; no
training, model replacement or presentation changes implied by this revision.

## 2026-10-07 — Matched Mac processing completed; comparison visuals applied

Both families completed all378identical validation clips with0failures. Apple M4
stage-sum medians (base/fusion/span)356.127/668.637/668.701ms; MediaPipe546.680/
809.295/808.632ms. MediaPipe retry used existing detector renewal between clips;
failed partial run excluded. Report records the exact workload, timing exclusions,
checkpoint paths and source hashes. No protected-test access or training.
Updated Figure11 with separate accuracy/timing panels and added concise method/results
paragraph. Figure18 now compares current FP16visual/recognition24.1ms/frame versus
FP32visual/recognition112.6ms/frame on iPhone13. English staysFP32outside timing.
Seven diagrams updated. Removed old English screening table/image and related prose;
renumbered export/storage captions. Requirements table spacing corrected after PDF
inspection. Updated writing guide and checklist with approved overrides and results.
Validation: figure source images inspected, preliminary affected PDF pages inspected,
Python compile and git diff --check passed. Navigation/final export verification next.

## 2026-10-07 — Common-Mac timing retry after known MediaPipe GPU leak

Initial MediaPipe timing stopped with MPS out-of-memory after a partial run; Apple
completed. The existing extractor and mobile-deployment log document a macOS MediaPipe
GPU pixel-buffer leak and detector recycling. Preserved initial partial JSON/log,
added detector.renew() between clips at1000calls (outside timed work), and restarted
MediaPipe from the beginning. Do not use the partial run in the paper. No high-watermark
bypass, runtime-source edits, training or protected-test access. Final report will
state timing scope and exclude model loading/recycling from steady-state stage costs.

## 2026-10-07 — Approved platform-neutral revision in progress

Author approved implementation in latest Google Doc tab t.pzqwjycim1ix, retaining
FP16 deployment and comparing current FP16 visual/recognition against FP32 only.
No validation-set size in manuscript. Slides and other tabs excluded. Applied core
platform-neutral prose, joint extractor introduction, simplified deployment/results,
requirements including Kotlin, three-stage accuracy table, conversion accuracy/storage,
and separate iPhone13 precision timing. Removed untrained-language timing table/figure
and detailed runtime discussion. Replaced seven diagrams with platform-neutral assets.
Image replaceImage sidecar failed before mutation; supported delete/insertInlineImage
replacement succeeded at the same positions/sizes. Phone figure added separately.
Matched Mac timing launched sequentially on the same validation recordings for both
pipelines; Apple completed, MediaPipe in progress. No protected test/training access.
Evidence/scripts: artifacts/reports/capstone_mac_comparison_v17_20261007/ and
artifacts/reports/phone_precision_v17_20261007/. Next: finish Mac timing, update
chart/narrative, reconcile navigation, read back and inspect export, then mark review done.

## 2026-10-07 — Second checklist feedback read; conversion concern investigated

Author checked10of12remaining items;P-04/P-06remain open. Overrides: omit Huawei
novaY70 entirely from manuscript additions; use "same validation set" in comparison
wording instead of naming ASLCitizen there. No manuscript/checklist changes applied.
Read current mobile source:LiveReelApp selects FP16hand encoder;LiveReelModels
defaults to fixed-batch FP16recognizer;LiveReelEngine loads FP16word boundary.
This verifies source configuration, not the currently installed phone binary/hash.
Sep30conversion report shows360/378correct FP32 vs359/378FP16 (one changed
prediction),Top5unchanged;recognizer+boundary storage saving28.682622MB. This
is reduced precision of the trained model, not replacement by a smaller architecture;
full-pipeline speed gain cannot be attributed solely to FP16. No deployment or
model change authorized by this discussion. Next:discuss accuracy/runtime priority
and establish whether author requests an FP32 comparison before changing deployment.

## 2026-10-07 — Remaining-proposals checklist refreshed after author review

Replaced README_PLATFORM_NEUTRAL_REVIEW.md with12unchecked remaining/new proposals;
removed all checked item blocks and all slide items as requested. Prior approvals
persist (not yet applied):G-01,G-02,C1-01..05,C2-01..07,C3-01..04,R-01..07,R-09,
M-01..04,C4-01,C4-03..12. Author comments and subsequent discussion supersede
the separate framework introduction and detailed processor/thread wording.
New review covers joint framework introduction, simpler Chapter4 without a submitted
technical-notes section, accuracy/storage conversion results, removal of graph-timing
screening, all3recognition stages, common-Mac recognition plus processing speed,
separate iPhone13 evaluation, and clearer existing signer-separation definition.
Author requests excluding additional MediaPipe dataset for fairness. Checklist
records shared ASLCitizen evaluation proposal and asks which dataset if training
additions are meant:Oct4report states matched training lists, so no MediaPipe-only
addition was assumed. Existing accuracy values are source results, not recomputed
after exclusions; matched speed values remain pending evidence. No experiments,
training, deletions, manuscript/GoogleDoc/presentation changes. Existing evidence
files remain. Next:author checks revised proposals, identifies any intended dataset
exclusion; verify live manuscript and metric provenance before implementation.

## 2026-10-07 — Author checklist feedback and signer-separation citation reviewed

Read author-edited platform-neutral checklist:41checked items;R-08 andC4-02 open;
all13slide items untouched. Author notes request joint AppleVision/MediaPipe
introduction, newer extractor results, and discussion of device/language charts.
Latest instruction requests simpler Chapter4 discussion without thread/processor
detail; supersedes those details in otherwise checked proposals. No approved
manuscript replacements applied in this discussion pass; checklist preserved.
Refreshed GoogleDoc tab t.pzqwjycim1ix: signer separation is explained/cited in
2.1.8 and4.3.1, reinforced in4.3.2 table. Verified Desai2023 support through the
primary ASLCitizen paper/dataset description; source assigns each participant to
one split. Study-specific split counts/provenance still need clear result linkage;
do not extend isolated-sign signer-disjoint claims to mixed streaming recordings.
Oct4 MediaPipe rebuild report/log gives paired Citizen validation:landmark base
Apple95.77/MP91.80;fusion96.30/94.44;interval-adapted95.24/94.44. These are distinct
stages, not drop-in replacements for old extraction-time comparison. Device timing
conditions remain unmatched for an OS speed comparison; language screening uses
untrained graph timing and differs from installed configuration. Recommend retaining
accuracy/storage/device28ms results, moving low-level timing conditions with their
component timing tables to technical notes. Next:discuss chart scope and simplify
proposals before manuscript changes. Slides remain deferred by author.

## 2026-10-07 — Platform-neutral wording review checklist prepared

Created `docs/capstone_papers/README_PLATFORM_NEUTRAL_REVIEW.md` at the user's
request: section-specific approval checkboxes, stable IDs, before/after wording,
rationale and blank author-note fields for manuscript and presentation proposals.
Incorporates the latest discussion: use mobile application in general descriptions;
retain named frameworks when introducing technologies or identifying comparisons.
Withdraws anonymous Extractor A/B labels; preserves iPhone 13 attribution for the
28 ms/frame recorded-input preparation-and-recognition result. Requirements proposals
add Kotlin and describe paired extraction/model/runtime implementations. All items
remain unchecked; this is a review artifact, not approval or application of edits.
Verified unique item IDs, one author-note field per item, local presentation link,
and whitespace. No manuscript, Google Doc, presentation, runtime or data changes.
Next: receive author feedback by item ID, then apply only the agreed revisions
after refreshing the live document.

## 2026-09-30 — MFET slides appended to Canva and verified

Completed the authorized continuation in Canva design `DAHWo5UJHoA`: original
pages1–3 preserved;13new slides occupy pages4–16 in the presentation brief's order.
Copied the original pale-blue background/grid, blue bands and university branding
through a clean temporary template export (600×337 background raster); new text,
cards and diagrams are editable. Demo/app media remain labeled placeholders.
All13presenter notes match the source brief exactly, including full scripts,
timings, visual cues and evidence notes. Reviewed all13Canva slide previews.
The import initially reversed the appended page order; corrected with12single-page
moves after Canva rejected a multi-operation request. Final readback verifies16pages,
unchanged original page IDs/thumbnail hashes, and exact notes in correct order.
Prepared files: `artifacts/generated/atlas_mfet_canva_20260930/` (PowerPoint,
builder, background, preview/contact sheet, notes and validation JSON). Installed
python-pptx in the project venv for authoring. Separate Canva staging design
`DAHWpNc3mF8` and one-page template copy `DAHWpLanbVo` are retained.
No manuscript, training, runtime or dataset changes. Next: replace media placeholders
with the actual recording/screenshots and rehearse the15-minute presentation;
progressive reveals/animations remain manual presentation preparation.

## 2026-09-30 — Canva presentation scope approved; connection required

User approved appending exactly13new MFET slides to Canva design DAHWo5UJHoA,
preserving every existing slide, copying its actual background and grid, using
placeholders for demo/screenshots, and including full scripts/timing/evidence in
presenter notes. Read the13-slide presentation brief and writing guide. Canva
metadata and page inspection both returned USER_NOT_LOGGED_IN (connector not
connected); plugin discovery confirms Canva is installed. No Canva changes made.
Next: reconnect Canva, inspect original page count/background/grid, prepare and
verify13matching slides and notes, then append within the authorized scope.

## 2026-09-30 — Slide purposes labeled beside titles

At the user's request, added `Purpose:` labels beside all13slide headings in
`ATLAS_MFET_15_MINUTE_PRESENTATION.md`, identifying each research section and the
methodology/results subtopic. Content and timing unchanged. Verified13headings;
scoped whitespace check. Next: presenter editing and rehearsal.

## 2026-09-30 — Paper-based presentation section flow implemented

User approved immediate demo reveal and early significance, with the old deck's
research progression. Reorganized presentation Markdown into13slides/900seconds:
introduction/context, gap, significance, objectives, scope/limitations, overview,
data/model development, streaming development, mobile implementation, recognition/
English results, deployment results, conclusion/next steps. Rechecked manuscript
§§1.3–1.4; objectives now include its ISO25010/software-testing objective without
claiming a completed assessment. Significance derives from §§1.1–1.2/3.2 and remains
intended benefit, not measured community impact. Actual capture/language limitations
are explicit. Removed separate biography and redundant contribution slides; retained
presenter responsibility in engineering narration,2026NCDA context and28ms/frame.
Updated scripts, visuals, section/timing map and judging alignment. Validation:
13sequential slides, complete content/visual/script sections, contiguous900seconds,
local source links and key measurement labels checked; scoped diff whitespace check.
Only presentation and log changed. Next: insert actual demo/assets and rehearse.

## 2026-09-30 — Approved deployment-first research-gap framing

Following discussion and explicit user approval, revised presentation Markdown to
make computational efficiency/local mobile deployment the main technical gap, with
connected-signing responsiveness and English output as supporting gaps. Accessibility
remains the motivation. Updated narrative, Slides3/6/9/11/12 and supporting guidance
to connect representation, model selection, precision and CoreML choices to accuracy,
storage and28ms/frame evidence. Grounded rationale in Chapter2 §§2.1.2/2.1.4/2.1.5.
Retained hybrid RGB/landmark facts; no blanket claim that RGB/Transformers are heavy,
that Squeezeformer is universally faster, or that28ms proves end-to-end latency.
Schedule remains14slides/900seconds. Manuscript, models and measurements unchanged.
Next: presenter review and timed rehearsal with the actual recorded demonstration.

## 2026-09-30 — Presentation research-gap revision from historical slides

Read all extracted text of user's27-page `~/Downloads/Capstone Presentation.pdf`
and rendered its computational-cost gap slide. Earlier deck centers manual exchange,
heavy models and recognition-only scope, with obsolete DS-GCN-TCN/RTMW-XL/CTC and
desktop assumptions. Updated `ATLAS_MFET_15_MINUTE_PRESENTATION.md` to14slides with
a dedicated cited gap, central research question, measurable objectives, and a
contribution slide answering the same challenges. Gap is a scoped engineering
investigation of stable streaming decisions, English generation and measured local
execution; no unsupported claim that prior translation/mobile systems do not exist.
Verified primary Camgoz2020/Moryossef2023/Kamikubo2025sources; links and claim limits
are in presenter notes. Replaced2020slide headline with2026NCDA registered DHH count,
retaining population/age/registration distinctions.28ms/frame emphasis preserved.
Added old-to-new comparison and visual guidance; shortened opening for demo time.
Focused checks:14sequential slides with visual/script/content sections, contiguous
900-second allocations, current statistic/speed labels, scoped diff whitespace check.
No PDF/manuscript/model edits. Next: actual demo recording and timed rehearsal.

## 2026-09-30 — Newer hearing-disability registry count located

User requires data from2023or later. Official NCDA homepage reports160,589cases in
the Deaf and Hard-of-Hearing category of DOH disability-registry data as of August3,
2026 (80,985female;79,604male). Source:https://ncda.gov.ph/.
This is a registered-disability count, not total national hearing-difficulty prevalence
and not an age5+tabulation. Present it with registry/date labels; never describe the
difference from the2020census as a decrease. Presentation headline remains unchanged
pending choice of this different measure. No comparable newer age5+estimate verified.

## 2026-09-30 — Hearing-difficulty statistic update check

User requested a newer age5+ hearing count. PSA2020published Table3 verifies the exact
count1,784,690; corrected presentation Slide2 from the rounded-percentage-derived
1.79M to1.78M and added the direct PSA PDF source.2024FLEMMS does include DIFF_HEARING,
but its catalogue warns that displayed case frequencies are not population summary
statistics. No newer directly published comparable national estimate verified in this
search; retain2020year rather than relabeling. Next: use exact census count or seek an
official newer tabulation. Only presentation and this log changed.

## 2026-09-30 — Presentation saved and optimization emphasis added

Saved the complete12-slide draft, scripts,15-minute allocations, visual suggestions,
sources, criteria mapping and preparation notes to
`docs/capstone_papers/ATLAS_MFET_15_MINUTE_PRESENTATION.md` at the user's request.
User then requested more optimization/speed emphasis. Updated Slide9 to lead with28ms
median preparation+recognition per processed frame on iPhone13 recorded-input profiling,
correcting the user's “per clip” wording. Retained CoreML/FP16, recognizer storage
49.62→25.51MB, matched validation95.24→94.97%, and incremental English-generation
explanations. Separate component measurement from end-to-end latency or sustained FPS;
no new benchmark or model change. Next: record demo, insert visuals and rehearse timing.

## 2026-09-30 — MFET narrative preferences and draft basis

User confirms ASL choice arose from available data/resources and limited initial
knowledge of alternatives. Do not assert Philippine ASL prevalence. Keep specific
campus scenarios brief. iPhone priorities: portability, efficiency and accuracy.
Development began March 2026 in Capstone1, now Capstone2. Planned recorded greeting:
“Hello, good morning. How are you?” Future priorities: expand vocabulary and collaborate
with Deaf users; closing theme is technology making lives a little better than yesterday.
User reports campus journalism since elementary, NSPC podium and current An Lantawan
creatives/multimedia director; optional brief communication connection, credentials in
biodata. These are user-reported background, not independently verified awards.

Rechecked manuscript Tables12/14/15/19 and ATLAS_MEASUREMENT_NOTES.md for presentation:
recognition comparisons are validation;9.68%WER is recorded development replay, not
live iPhone accuracy or a CTC-versus-ATLAS score. English60%fully correct is automatic
judgment on generated sessions. Phone28ms measures prepared recorded-input frame work,
not end-to-end sign-to-speech latency. Preparing a12-slide timed draft in conversation;
no manuscript, model or data edits. Next: rehearse pacing and insert actual demo output.

## 2026-09-30 — MFET presentation interview and context research

Rechecked local revised manuscript: it again contains Chapters 1–4, not the earlier
presentation brief. Presentation is 15 minutes excluding Q&A, for a mixed technical
panel. User reports sole AI/ML, pipeline, software and mobile implementation, no prior
experience, motivation from observing students signing at school, and a recorded
self-demonstration to be made later. Omit development-assistant discussion as requested.
Use manuscript technical metrics with their actual evaluation contexts; no new human
feedback or community outcome was supplied. Engineering narrative: tested CTC errors
and input/commitment delay motivated boundary-based streaming; not a universal CTC limit.

Read-only source research: PSA 2020 functional-difficulty release reports 21.1% of
8,469,426 people aged 5+ with functional difficulty had hearing difficulty (~1.79M,
derived from rounded percentage; not ASL-user count). EDCOM2 November 20, 2025 release
reports 391,089 public-school learners with disabilities in SY2024–25 and 60% without
school SNED resources; all-disability statistics, not Deaf dropout statistics.
Sources: https://psa.gov.ph/content/functional-difficulty-philippines-household-population-five-years-old-and-over-2020-census
and https://edcom2.gov.ph/5-million-filipino-children-with-disabilities-remain-underserved-edcom-2-study/.
No reliable national ASL-use percentage or hearing-specific dropout-cause count found
in this search. Manuscript specifies ASL scope but does not establish Philippine ASL
prevalence. Next: clarify resource-access rationale, intended campus interaction,
development duration and demo message, then draft timed slide content and script.
Only this log changed; manuscript and system remain untouched.

## 2026-09-30 — Google Docs transfer and section 4.3 overview

Transferred the manuscript snapshot into Google Docs tab `t.5at7inr8zi4x`,
REVISED NOT FINAL (SEPT 30), retaining the copied REC Revision footer framework.
Checked 28 embedded images and 18 native tables; applied Times New Roman, matching
black top/header/bottom table rules and white internal borders. Copied the source
sideways Gantt layout and refreshed contents/figure/table page references from PDF.
The author subsequently edited the Activity List: preserve that user version without
reconciliation or formatting changes. Added and read back a short 4.3 overview covering
cleaning, annotation, inputs, training, recognition, English generation and iPhone integration.
Inspected its exported page layout. Source tab was preserved.
The local `Capstone 2_ ATLAS_revised.md` independently changed to an MFET presentation
brief during transfer; left untouched. The cloud tab is the manuscript working copy;
do not restore the local snapshot over the user's changes. Next: author review in Docs.

## 2026-09-30 — Final manuscript review before Google Docs

Read revised Chapters 1–4; corrected contents anchors, stale pagination, headings/captions,
MediaPipe citation and missing Jiang reference, model-family context, NO-only retention metric,
Core ML component wording and timing cross-reference. User retains subset attribution for
discussion and selects Google Docs schedule as authority. Imported its Activity List from
previously downloaded source DOCX, preserved original images; documented remaining dependency
ambiguities in ATLAS_FINAL_REVIEW.md. Updated writing guide/current state. Verified four
objectives, image paths, anchors, sequential captions and original manuscript hash; scoped
whitespace check. No model changes, new evaluations or Google Docs writes. Next: resolve
subset attribution and schedule semantics, then prepare Docs transfer with embedded assets.

## 2026-09-30 — Deployment purpose established before conversion metrics

Author approved explanation of why Core ML was selected. Added to manuscript4.4.5
and technical review: native Swift/iPhone integration and local execution are the
conversion purpose; FP16 is a separate precision/storage choice. Accuracy similarity
indicates retained recognition behavior; timing is component/runtime-dependent.
No new benchmark claims or metric changes. Updated guide/current state. Verified
replacement in both documents and git diff --check. Next: author review.

## 2026-09-30 — Before/after storage size added

User additionally requested MB. Serialized full FP32 inference state only (without training
metadata), measured full source CoreML packages. Recognizer49.616359→25.506008MB;
boundary9.254398→4.682127MB. Added rows/size definition to manuscript/review conversion
tables. Added reproducible size script, sizes.json and saved reference states under
conversion report; updated report/measurement notes/guide/current state. Decimal MB;
not RAM, compiled package or full app size. Existing checkpoints unmodified.
Validated rows against measured bytes; regenerated large-file index; diff whitespace check.

## 2026-09-30 — Before/after model timing completed

Benchmark completed on Apple M4. Recognizer27.6433→11.3218ms/batch8; boundary1.3949→2.0005ms/window.
Retained slowdown honestly; CPU one-thread PyTorch vs CoreML ALL is execution-path comparison,
not precision-only or iPhone timing. Added timing rows and methods/interpretation to manuscript
and technical-review conversion tables. Updated report, measurement notes, guide/current state.
Syntax checked; both timing rows verified in both documents; git diff --check passed.
No further evaluation warranted; next author review.

## 2026-09-30 — Before/after conversion timing authorized and launched

User requested milliseconds alongside conversion accuracy. Added matched-input benchmark
scripts/benchmark_capstone_conversion_v17.py: original full recognition graph versus exact
FP16 fixed-batch package, batch8 on128validation intervals; boundary32windows from4tuning
clips. Both on same development Mac; PyTorch CPU one thread versus CoreML ALL, ten warmups,
five repeated interleaved passes. Prepared inputs exclude loading/compilation/preprocessing.
No test access or model mutation. This compares execution paths, not precision alone.
Launched log:artifacts/reports/capstone_conversion_v17_20260930/latency.log.

## 2026-09-30 — Matched FP16 recognizer accuracy and boundary agreement

User authorized boundary/recognizer comparison, validation checks and removal of phone
profile chart/table, keeping28ms as text. Saved export audit found boundary0/1944state
mismatches but deployed fixed-batch recognizer1mismatch; batch1report0mismatch is distinct.
Added scripts/check_capstone_conversion_v17.py; ran all378validation inputs with same
cached landmark/hand features, original checkpoint label mapping and deployed fixed-batch
CoreML FP16/ALL. Padded final partial batch. Top1:360→359correct(95.2381→94.9735%);
Top5:374both(98.9418%);377/378top1agreement. LIKE→MY is sole changed prediction.
No tuning/training/test access. Boundary checkpoint hash checked; reused saved100%state
agreement, explicitly not human-annotation accuracy. Full report, predictions, identities
and package hashes:artifacts/reports/capstone_conversion_v17_20260930/.

Replaced revised Table19 and technical-review profiles with accuracy/agreement table;
removed phone chart embedding and Figure19list entry; training figures20–22→19–21.
28ms retained only as one timing sentence per document. Updated measurement notes,
major changes, guide, canonical state. Source benchmarks and image assets preserved.
Full-recognizer check passed; chart links/numbering and Python syntax checked; whitespace
check passed. Existing exporter loop omits last incomplete batch despite total378 label;
new check covers all378. Initial targeted history lookup used unmatched zshglob, retried
with rg directory search. Next: author review, no model promotion or changes.

## 2026-09-30 — Established model identities before export settings

Completed author-approved edit after an interrupted read-only turn. Rewrote deployment
introduction in revised manuscript and technical review to first identify boundary,
Squeezeformer, MobileCLIP2 and T5 roles, then explicitly state their Core ML exports.
Table18 now contains these four neural components and export forms; Apple Vision and
application logic explained separately. Preserved T5 one-model/two-export distinction
and exact precision conditions. Removed repetitive conversion introduction in review.
No metrics, diagrams or original manuscript changed. Updated guide/current state;
verified four table rows, manuscript caption/list consistency and diff whitespace.

## 2026-09-30 — Replaced opaque configuration labels with explicit combinations

User found A/B/C labels unexplained. Revised manuscript/review describe shared model
composition (boundary, Squeezeformer and MobileCLIP2, supported by Vision/decoder),
then name rows by mixed versus FP16 precision and recognition processor permission.
Rebuilt phone_speed PNG/SVG with matching descriptive labels. Defined FP16 models as
those three recognition-side components, excluding FP32 T5. Timings unchanged.
Guide/current status supersede A/B/C naming. Visual check and diff whitespace check pass.

## 2026-09-30 — Corrected encoder-centered phone timing presentation

User noted prior changes left Table19 and phone chart focused on MobileCLIP2. Replaced
both manuscript/review tables with combined profiles A/B/C and separate boundary,
recognizer, recognizer compute-unit and MobileCLIP2 columns. Updated interpretation
and regenerated phone_speed PNG/SVG with matching profile labels and combined-time axis.
Preserved123/82.6/28ms and actual precision differences; no per-model timing invented.
Visually checked chart; git diff --check passes. Builder, guide and canonical state
updated. No runtime/evaluation changes.

## 2026-09-30 — Boundary/recognizer emphasis and combined workflow

Author approved centering deployment on boundary estimation and recognition, with an
overall workflow. Revised manuscript4.4.5 and technical review now lead with this pair;
reordered deployment table, added combined-operation paragraph and expanded existing
figure18 into branched visual inputs feeding boundary/recognition and stable glosses,
then T5/Finish/finalization/speech. Updated reproducible PNG/SVG/Mermaid artwork.
Clarified28ms as measured combined preparation/recognition median, without inventing
individual export gains or summing per-frame and per-utterance timings. No measurements
or numbering changed. Visually checked diagram; manuscript links and git diff --check
pass. Guide/canonical state updated. Next: author review.

## 2026-09-30 — Full deployment explanation before optimization timings

Author approved model-to-CoreML-to-iPhone explanation, allowing a separate section.
Added4.4.5 with component table18 and deployment diagram18; shifted profiles to4.4.6,
table19/figure19, training to4.4.7 and figures20–22, updating lists and cross-references.
Verified export precision against converters and Swift resource selection: FP16 visual,
boundary and recognition exports; FP32 T5 encoder/decoder. Distinguished application
logic/native frameworks from neural exports and precision from compute-unit settings.
Updated technical review, major changes, writing guide and measurement provenance.
Added reproducible deployment_flow PNG/SVG/Mermaid. Existing timing measurements unchanged.
Visual inspection and caption/link checks pass; git diff --check passes. No runtime or
training changes. Next: author review of deployment explanation.

## 2026-09-30 — Language screening reduced to tiny T5 and base candidate

Author requested only own model and T5 base. Retained tiny T5 and the actual measured
FLAN-T5-base candidate in revised Table17 and technical review; updated discussion and
language_latency PNG/SVG through a dedicated rendering helper. Kept exact candidate
identity,173/391ms estimates, untrained-weight screening context and installed FP32
configuration distinction. Full metrics/source records preserved. Visually checked
chart, verified both tables contain two rows, and git diff --check passed. No other
charts or model settings changed. Guide/current status updated.

## 2026-09-30 — Recognition-family comparison simplified to four models

Author requested BiLSTM, Temporal CNN, Flat Transformer and ATLAS only. Reduced Table14
and its discussion in the revised manuscript; synchronized technical-review family table
and prose. Regenerated only families.png/.svg with four rows via new family_chart helper
in build_review_assets.py, preserving all ten raw metric rows and source records.
ATLAS row retains96.30%/6.15ms; flat Transformer95.50%/2.88ms; no measured values changed.
Chart visually inspected; four-row tables and git diff --check verified. Guide/current
status updated. Original manuscript and architecture-variant comparison untouched.

## 2026-09-30 — Original schedule images embedded

At the author's request, replaced revised Chapter4's Gantt placeholder with all seven
original Gantt images in source order, and PERT placeholder with original pert.png.
Original Sashimi remains embedded. Preserved Activity List, schedule prose and numbering.
Verified all nine source-image references exist and occur once; git diff --check passes.
Updated guide and canonical status to record the authorized placeholder insertion.
Next: author review; no scheduling data or image content changed.

## 2026-09-30 — Original Sashimi, PERT and Gantt images extracted

User supplied Google Docs link and requested original photos. Web page tool could not
open it; public DOCX export succeeded. Extracted embedded image bytes without resizing
or editing into docs/capstone_papers/source_images: sashimi.png, pert.png and seven
Gantt JPEGs in document order. Added browsable README and source/hash manifest.
Revised manuscript now references original Sashimi image instead of the reproduction.
PERT/Gantt saved for use; existing schedule prose/placeholders remain unchanged.
Originals visually inspected for identification, byte equality verified against export,
and git diff --check passed. No cloud document changes. Next: author review/use of images.

## 2026-09-30 — Sashimi figure follows supplied diagram

Replaced generated bar-style Sashimi artwork with a code-native reproduction of the
user's supplied image: Planning → Designing → Development → Testing → Implementation,
colored rounded rectangles descending diagonally, with forward and return arrows.
Updated sashimi.png/.svg and its reproducible builder function; revised Chapter4's
phase names and image alt text to match. No other diagrams regenerated. This is a
reproduction of the supplied visual, not a byte-for-byte copy of its attachment.
Visually checked labels/arrows/layout; git diff --check passed. Original manuscript
and schedule material untouched. Next: continue author review.

## 2026-09-30 — Chapter 4 revised with methods, results and iPhone design

Author approved retaining requirements/design, adding preparation/model development
and results/discussion, including all approved comparison charts/training curves,
and filling iPhone context/data-flow/use-case/application diagrams. Activity List,
Gantt and PERT explicitly excluded from changes. Revised the separate manuscript;
original and Chapters1–3 preserved. Added 4.3/4.4, tables10–18 and figures9–21;
filled figures3/5/6/7 with four new PNG/SVG pairs generated by
review_assets/build_chapter4_diagrams.py and used existing application flow for figure8.
Finish follows incremental English → Finish control → finalization → text/local speech.
No figure footnotes or explanatory subtitles. Existing measured chart assets unchanged.

Methods distinguish annotation/manual clipping, automatic trimming/alignment, normalized
32-frame inputs, augmentation, regularization, frozen-teacher boundary distillation and
recognizer interval adaptation. Results preserve per-study metrics and timing scope,
automatic English-rating provenance and untrained-weight language execution screening.
CTC rationale included without numerical CTC results; no invented software-quality scores.
Retained existing citations and added Papineni2002 BLEU primary reference after checking
ACL Anthology. ISO page fetch failed; no new standard evidence claimed. Updated major
changes, writing guide and canonical authoring state.

Validation: original SHA256 remains42b36f7518152627d2e47f9d438fa058843ed0e3763527e26290c1559a159b25;
Chapters1–3 exact, schedule block exact, prior reference entries retained, image paths
exist, figure1–21 and table1–18 captions sequential. A broad blank-row cleanup initially
merged schedule phase rows; exact-block check detected it and original block was restored.
Numbering check adjusted for original mixed caption punctuation. All scoped checks pass;
new diagrams visually inspected and git diff --check passed. No training, evaluation or
runtime changes. Next: author review of Chapter4; schedule remains excluded.

## 2026-09-30 — Chapter 3 revised after original-first discussion

Read original Chapter 3 and asked three questions before editing. Author approved
both diagrams, iPhone13 measurement/implementation device with Mac development role,
and original Peopleware groups. Revised Chapter 3 in separate manuscript, preserving
Current/Proposed/System requirements/Peopleware organization and connected explanation.
Explained four functional components, 61-point/five-channel representation, resampled
32-frame intervals, multimodal recognition, boundary distillation/decoder roles, gloss
buffer, incremental T5, Finish and local speech. Retained Non-Sign Language Users and
Developers groups; updated their technical descriptions. Updated software/hardware tables
against Flutter project and native Swift/UIKit/AVFoundation/CoreML implementation.

Embedded existing system_flow.png and hello_how_you_timeline.png with explanations in
Markdown; no figure assets or measurements changed. Added Figures1–2; shifted six Chapter4
figure labels and list entries to3–8. No page-number claims for new figures. Updated
major changes, guide, high and canonical status. Validation: original hash unchanged;
Chapters1–2 and references exact; Chapter4 differs only in figure labels; two image links,
sequential figure labels, original Peopleware groups, third person and terminology pass.
Next: author review of Chapter 3 before Chapter 4.

## 2026-09-30 — Chapter 2 literature and related systems revision

User authorized Chapter 2. Retained original structure, connected prose and bibliography;
updated obsolete Reel/activity-controller, supplemental MediaPipe mouth, completed-buffer
translation and Mac-only descriptions. Integrated boundary/segmental roles, 32-frame
resampled landmark input, normalized coordinates, modularity, teacher/student distillation,
incremental English and iPhone execution. Added 2.1.8 on preparation, normalization,
augmentation, regularization, distillation and signer separation. Vocabulary rationale
now separates class count, foundational content, computational measurements and comparison
conditions; related-system metrics are not ranked against local results.

User steering: keep official standardized subset wording, then “just copy the original
manuscript.” Preserved original Section 2.1.1 vocabulary passage verbatim. Informed user
that attribution remains unverified; recorded distinction in measurement notes and guide.
No change to factual vocabulary research record, manifests or split gates.

Checked primary sources via web for segmentation2023/2026 versions, MobileCLIP2,
MobileCLIP author metadata, Sign2Pose, reservoir computing, Woods, distillation,
dropout, AdamW, G-Eval, introductory questions and community collaboration. Several
publisher pages blocked; used publisher-indexed text, author/institutional records
and arXiv/ACL counterparts. Removed unverified in-browser execution claim and unmatched
speed claims from related systems. Added nine bibliography entries; all prior entries
retained. Exact new MobileCLIP author list corrected after source check.

Changed revised manuscript, ATLAS_MEASUREMENT_NOTES.md, ATLAS_MAJOR_CHANGES.md,
ATLAS_WRITING_GUIDE.md and current-state pointer. Validation: exact preservation of
Chapter 1/front matter and Chapters 3–4, original manuscript hash unchanged, original
subset passage exact, all prior references retained, third-person and obsolete-term
checks pass. No runtime/data/figure changes. Next: author reviews Chapter 2 before
Chapter 3 revision.

## 2026-09-30 — Chapter 1 revised against original narrative style

Author requested applying the approved whole-paper style guide to the revised
manuscript. Continued the active Chapter 1 scope. Used original Chapter 1 as the style
and paragraph-development reference: expanded connected explanations in introduction,
purpose/description and scope, preserving the communication-to-technology-to-challenges
progression and meaningful however/despite transitions. Restored fuller descriptions
of the interaction and roles of temporal recognition and language generation while
retaining current technical facts. No vocabulary count in Introduction; no spelling
references or unsupported signer allocation. Approved four-objective block unchanged.

Validation: exact preservation of front matter, objectives and Chapter 2 onward;
original manuscript hash unchanged; retained Introduction citation set; third-person,
vocabulary placement and terminology checks pass. No models, figures, data or results
changed. Next: author review of Chapter 1 before Chapter 2 revision.

## 2026-09-30 — original voice and narrative rules confirmed for all chapters

Author approved original manuscript as primary style reference, flexible paragraph
length preserving narrative development, and connected explanatory writing throughout
without drifting from the original. Updated ATLAS_WRITING_GUIDE.md authority and existing
style bullets, plus a dedicated whole-paper editing rule. Added a canonical style
pointer in PROJECT_GROUND_TRUTH.md and promoted the decision to high.md.
Checked guide alignment and confirmed both manuscript hashes unchanged in this action.
Next: apply these rules during the next authorized manuscript revision.

## 2026-09-30 — omit fingerspelling from manuscript presentation

Author requested removal of all fingerspelling discussion and no separate topic from
glosses. Revised manuscript now uses Gloss Management and recognized gloss sequences;
removed the spelling scope sentence and related objective/component wording. Updated
ATLAS_WRITING_GUIDE.md, major changes and technical review consistently. Changed system
and runtime flow labels in the figure builder and Mermaid sources; rebuilt 16 PNG/SVG
pairs. This is presentation only, not a change to models or vocabulary capabilities.

Validation: no fingerspelling/spelling references remain in revised manuscript, review,
major changes or rendered SVG text. Original manuscript hash unchanged. Next: author
review of Chapter 1 before proceeding to Chapter 2.

## 2026-09-30 — four objectives restored to original structure

User requested exactly four objectives aligned with the original. Replaced only the
specific-objective list in Capstone 2_ ATLAS_revised.md: develop the multimodal pipeline,
evaluate model performance, implement the integrated iPhone application, validate
software quality. Retained current components and approved metrics; restored black-box
and white-box testing as objectives. Added the four-part structure to the writing guide.
Verified exactly four bullets and exact preservation of text outside that list.
Original manuscript unchanged. Next: author review of Chapter 1.

## 2026-09-30 — Chapter 1 narrative restored under approved writing rules

Author authorized manuscript update. Revised the Introduction subsection in
Capstone 2_ ATLAS_revised.md to follow communication context, technology, complementary
visual inputs, recognition-to-English needs, temporal/language modeling, deployment
challenges and ATLAS response. Restored meaningful however/despite transitions;
kept the original citation set. Framed deployment hurdles as concrete application
requirements rather than unsupported claims about all existing systems. Removed
100-sign size and question-word list from Introduction; placed the vocabulary-size
sentence in Purpose and Description. Objectives and Scope remain unchanged.

Validation: original manuscript SHA256 unchanged; front matter and Chapter 2 onward
exactly unchanged; objectives/scope unchanged; introduction has no vocabulary count
or question-word list; retained citations and third-person voice checked. Next: author
review of Chapter 1 before proceeding to Chapter 2.

## 2026-09-30 — introduction narrative rules added

Author requested updating the writing guide after finding Chapter 1 too compressed.
Added original narrative progression, meaningful however/despite transitions, preservation
of clear paragraphs/citations, supported deployment-challenge discussion and vocabulary
placement rules to ATLAS_WRITING_GUIDE.md; promoted decisions to high.md. Guide-only
update: neither manuscript changed in this action. Next: apply the agreed introduction
style and remove vocabulary details from its Introduction subsection when revising it.

## 2026-09-30 — authorized Chapter 1 revision in separate manuscript

User approved Chapter 1 and targeted rewriting, then clarified that the requested
15-signer 10/3/2 allocation is temporary drafting information to reconcile later.
Did not present that contradicted allocation as an actual measured split. Chapter 1
states signer separation without a sample allocation. Updated writing-guide status.

Revised only Chapter 1 of Capstone 2_ ATLAS_revised.md: iPhone focus, foundational
100-sign vocabulary with six question signs, Apple Vision/MobileCLIP2/Squeezeformer,
modular responsibilities, boundary/segmental Streaming Sign Recognition, integrated
spelling, incremental T5 English, one-second palms/Finish, Core ML, app activities,
and distinct evaluation tasks. Removed obsolete supplemental mouth/Reel/window settings,
Mac-only scope, claimed human ratings and Macro-F1. Preserved existing chapter citation
set and restrained unsupported guarantees of fluent/correct output or campus outcomes.
Updated ATLAS_MAJOR_CHANGES.md, ATLAS_WRITING_GUIDE.md and canonical current status.

Validation: exact equality outside Chapter 1 (front matter and Chapter 2 onward),
original SHA256 unchanged, retained citations checked, third-person/obsolete terminology
checks pass. No code, model, data or evaluation changes. Next: author reviews Chapter 1
before proceeding with Chapter 2.

## 2026-09-30 — video stills, boundary row and landmark input explanation

User requested original video frames below the simplified timeline and the boundary
detector's role. Rebuilt hello_how_you_timeline with a boundary-detector/decoder row,
recognition labels on the same time axis, and three original stills from the selected
video (zero-based frames 28,46,77 at 30 fps). Intervals remain saved model-selected
spans, not raw boundary probabilities or manual ground truth. Stored video hash,
frame indices and timestamps in metrics.json. Kept feature-extraction lines and
English-output flow out of the image. Updated technical review, measurement notes,
major chapter changes and guide to match.

User also asked about landmark input frames. Confirmed schema and current Swift
LiveReelModels recognizer array: 32×61×5 per candidate interval; explained resampling,
node groups and channels, distinct from camera FPS. No new model/input-length change.
Validation: clean 16-pair rebuild; figure visually inspected; SVG embeds exactly three
stills; frame indices, video hash, source intervals and document links verified; both
manuscript hashes unchanged. No inference or training. Next: author review.

## 2026-09-30 — simplified HELLO HOW YOU timeline

User requested emphasis on recording time and removal of feature-extraction lines.
Replaced the multi-panel timeline with one row of HELLO/HOW/YOU interval bars, their
start/end values and a recording-time axis. Removed component rows and the embedded
English-output flow; retained their explanations in ATLAS_TECHNICAL_REVIEW.md.
Updated ATLAS_MAJOR_CHANGES.md, ATLAS_MEASUREMENT_NOTES.md and ATLAS_WRITING_GUIDE.md
to match. Modified the figure builder and rebuilt PNG/SVG assets and metrics.json.
Verified plotted intervals equal the saved record, inspected the simplified figure,
and confirmed both manuscript hashes unchanged. No metrics or runtime behavior changed.
Next: author review of the simplified figure before manuscript edits.

## 2026-09-30 — preparation methods and HELLO HOW YOU worked timeline

User approved technical-review-only additions and chapter-by-chapter change updates,
selecting HELLO HOW YOU. Added concise annotation/manual clipping, cleaning, automatic
trimming, normalization, augmentation, regularization and split/model-selection discussion
with a method–purpose table to ATLAS_TECHNICAL_REVIEW.md. Checked geometry_v17,
train_stage_1_v17, recorded landmark training provenance, segmental_lab_v17 and
train_span_recognizer_v17. Manual clipping of selected recordings is author-reported;
no claim that all interval labels were manually annotated. Distinguished transcript
alignment, teacher supervision and timed annotations. Added corresponding Chapter 2–4
rows in ATLAS_MAJOR_CHANGES.md and removed its stale milestone-oriented evidence wording.

Added sequence_timeline() to review_assets/build_review_assets.py, generating the 16th
PNG/SVG pair: hello_how_you_timeline. Uses saved Swift record
local:HELLO_HOW_YOU:ff187c3f in phone_speed_v17_20260929/score_test_C.json;
reference and hypothesis HELLO HOW YOU. Selected intervals: HELLO .6667–1.1667,
HOW 1.4–1.6667, YOU 2.2667–2.8667 seconds. No new inference. Recording time is not
hardware latency; final word compute sentinel -1 prevents interpreting it as measured
live commitment. Feature-role rows and gloss/English/Finish flow are schematic.
Documented provenance in existing ATLAS_MEASUREMENT_NOTES.md, retained all explanatory
text outside the image, and updated ATLAS_WRITING_GUIDE.md. Builder records source hash
and exact interval fields in metrics.json.

Validation: clean 16-pair rebuild; new diagram visually inspected; all PNG/SVG files
parse, document links resolve, plotted intervals equal saved source values, and both
manuscript hashes match the preserved original. No model/data/runtime changes. Next:
author review of methods and timeline before any manuscript revision.

## 2026-09-30 — figure footnotes removed, document explanations preserved

User clarified that only explanatory text embedded in figure images should be removed;
titles, labels, legends, values and diagram text stay. ATLAS_MEASUREMENT_NOTES.md and
Markdown explanations are preserved. Updated review_assets/build_review_assets.py to
omit footer rendering and its reserved space; rebuilt all 15 PNG/SVG pairs. Added the
approved image-formatting rule to ATLAS_WRITING_GUIDE.md and the binding decisions.
Verification: all 15 SVG text multisets match their previous contents minus the exact
explanatory footer texts; all 15 PNGs passed image validation. Inspected the recognition
training figure visually. No metrics, runtime code or manuscripts changed. Next safe
action: author review of the figures and technical review before manuscript edits.

## 2026-09-30 — standalone modular-system review and approved source attribution

Author approved four clarified changes. Reworked ATLAS_TECHNICAL_REVIEW.md to introduce
component roles and modular architecture first, explain pretrained boundary-teacher
origin and frozen-teacher distillation, remove CTC numerical/overlap discussion, and
move hardware provenance into new ATLAS_MEASUREMENT_NOTES.md. Restored two-way streaming
comparison using complete72-recording set: boundary-guided39.78% and ATLAS9.68%WER.
Source metrics computed from preserved baseline predictions and Swift result. CTC
results remain intact in their evidence report. No new inference or runtime changes.

Verified upstream segmentation repository:2026CNN–Transformer source versus related
Moryossef2023paper/version. Added both citations and explained DGS/MediaPipe teacher,
Apple Vision boundary student and separate recognizer fine-tuning. Replaced chronological
labels throughout narrative/charts; English quality now compares only whole-sequence
and incremental configurations of the same fine-tuned model. Updated guide, high and
canonical presentation pointer. Rebuilt15PNG/SVG pairs. Checks: local links, SVG labels,
restored metrics, manuscript hashes, git diff --check. Next: author review before any
manuscript edits.

## 2026-09-30 — CTC validation, training-overlap audit, citations and complete iPhone figures

User authorized local CTC validation instead of comparing unmatched historical sets.
Added scripts/evaluate_capstone_ctc_comparison_v17.py. Saved general selector and
Core ML packages loaded; all72video hashes and Swift reference IDs/transcripts verified.
Defaults frozen before inference; native30Hz and32/30s windows with existing window,
selector and rollover functions; no training/tuning/runtime edits or protected isolated
test access. Native CTC30Hz versus current native20Hz is a whole-pipeline comparison.
Result6.45%WER(4S/6D/2I over186signs) initially reported, then corrected in conversation:
51local videos are train members in specialist's hash-matching real-training manifest.
Remaining9local+12unseen are historical validation. General selector is distinct from
later causal/repaired/Finish-time CTC checkpoints; do not mix their results.

Rescored all three systems on identical21historical-validation videos/47signs, selected
by manifest roles rather than prediction scores. CTC23.40%(4S/6D/1I), earlier boundary
48.94%(1S/21D/1I), current Swift21.28%(2S/6D/2I). Known train clips excluded from every
row. Other ancestral source exposures not exhaustively audited; not an untouched
independent test. Report, hashes, full predictions, membership audit and common subset
saved in artifacts/reports/capstone_ctc_comparison_v17_20260930/. No bad result hidden;
full-set6.45/39.78/9.68 retained and scope difference explained.

CTC rationale now concise in streaming section: saved selector's1.07s first input
window, revisable predictions/stable-prefix speech, historical camera timebase mismatch
and held-sign duplication, separate later CTC validation errors. CTC selector itself
historically3–5ms; avoid blanket claims of slow CTC decoding. Added Kamikubo etal2025
ASSETS citation:9/10surveyed Deaf/HoH ASL signers prioritize real-time translation;
co-design includes latency concerns. Supports timely feedback, no universal instant
threshold. Nielsen1993 provides separate general UI responsiveness guidance.

Traced AppShellViewController/ShellPages/LiveReelApp/ReelCamera. Added app_flow for
Home/Live/Glosses/Practice/History, sign demonstration→practice, retry/skip/results.
Single output path follows latest user instruction: T5→Finish→finalize→text/speech.
Removed visible Reel/simulation/old execution labels, sample-count clutter; phone plot
names FP32/FP16 encoder and compute choices. Rebuilt15PNG/SVG pairs and4Mermaid sources.
Updated technical review, guide and figure builder; inspected final changed figures.
Validation:5prefix+13liveCTC tests pass; references and WER arithmetic checked; all review
links resolve; obsolete SVG labels absent; both manuscript SHA256 unchanged; git diff
--check passed. Artifact index regenerated. Next: review final figures/rationale before
manuscript approval, without claiming CTC is categorically inferior.

## 2026-09-30 — readable technical review and model-choice discussion

Updated ATLAS_TECHNICAL_REVIEW.md and figure builder/assets to simplify metrics and
academic explanations. Removed pre/post-trim detection, macro-F1, McNemar, author
allocation/conversation notes; retained source records. Streaming table/chart now
compares only earlier boundary/controller39.78% and current Swift replay9.68% WER,
with Mac replay setting and familiar/unseen signer composition stated. Added concise
CTC history, Graves2006 reference, automatic-English rubric and Liu2023 G-Eval citation.
Language ratings remain automatic DeepSeek judgments, not human certification.

Squeezeformer96.30% versus flat Transformer95.50% is ~3/378 more correct; CPU6.15
versus2.88ms is2.14x, +3.27ms. Compact Transformer1.76ms/94.71%. Explain trade-off,
not overall superiority. Verified current boundary teacher is frozen pretrained DGS
via train_av_boundary_v17.py; separate boundary fine-tuning experiments did not supply
that teacher. Recognizer fine-tuning is distinct. No runtime changes.

Phone labels now specify FP32/FP16 hand-image encoder and CPU/GPU versus all Core ML
processors; all rows already Core ML. Omitted display-overlay tracking row. Finish
control moved to sentence finalization before text/speech, preserving incremental
English and explaining recognition flush. Added academic presentation rules to guide.
Rebuilt14figure pairs; visually reviewed changed figures and corrected flowchart
bottom clipping. Local links resolve; manuscripts match previous hashes. git diff
--check run. Next: discuss accuracy/speed selection and approve manuscript wording.

## 2026-09-30 — approved writing rules and verified technical review location

Added author-approved third-person voice and editing rules to ATLAS_WRITING_GUIDE.md,
with editorial source links and navigation to the three review documents. Preserve
clear original wording; the researchers is allowed, first-person prose is not.
Updated high.md. Verified ATLAS_TECHNICAL_REVIEW.md exists at
`docs/capstone_papers/ATLAS_TECHNICAL_REVIEW.md` (29,341 bytes) and every local link
resolves, including its figure assets. User reported difficulty finding it; provide
a direct absolute file link. No manuscript edits. git diff --check passed.
Next action: author review of technical material before manuscript revision.

## 2026-09-30 — author clarifies positive, factual paper framing

Author objected to making the paper sound deficient. Updated writing guide, vocabulary
research note, major-change review, technical review and binding high notes: lead with
purpose and contribution; state the 100-sign scope neutrally; place expansion in
recommendations; avoid repeated limitation language. This supersedes the earlier
interpretation that vocabulary limitations should dominate the proposed paragraph.
Evidence qualifications remain in research analysis; no claims or metrics inflated.
Manuscripts remain untouched. Documentation validation: git diff --check. Next action:
continue author discussion before approved manuscript revision.

## 2026-09-30 — foundational vocabulary research and prototype framing

Added `docs/capstone_papers/ATLAS_VOCABULARY_RESEARCH.md`; updated technical review,
major-change review and writing guide. Confirmed manifest contains WHAT, WHERE, WHEN,
WHO, WHY, HOW. Inspected authors' WLASL v0.3 metadata and official MS-ASL metadata
in memory: first100 contains respectively 3/6 and 5/6 of these labels; WLASL lacks
WHERE/WHEN/WHY, MS-ASL100 lacks WHY. No videos or training data acquired, no inference
or protected-test evaluation. Reused existing exact-code lexical-frequency audit:
selected mean5.81 versus lexicon4.13;87/100 top quartile,95/100 above median.
These are subjective ratings, not conversational coverage. Boise State Level1
Activity2 supports introductory question-sign teaching; Lifeprint distinguishes
question grammar from lexical labels. Two subsets cannot justify “most datasets.”

Author rejected everyday-communication coverage framing and requested prototype,
foundational vocabulary, limitation and expansion recommendations. Recommended wording:
“a 100-sign prototype vocabulary containing foundational signs.” Preserved manuscript
approval gate; neither manuscript edited. Next action: discuss remaining scope and
approve review content before manuscript revision.

## 2026-09-30 — major-change review, technical comparisons, and cited decision rationale

Author approved six comparison topics, numerical tables with short explanations,
comparison charts and recorded training curves, both runtime and engineering-decision
flowcharts, descriptive data labels, and retaining citations. Requested a quick review
before approving edits to the revised manuscript. Added `ATLAS_MAJOR_CHANGES.md` and
`ATLAS_TECHNICAL_REVIEW.md` beside the paper, updated `ATLAS_WRITING_GUIDE.md`, and
created `review_assets/` with a reproducible figure builder, source-hashed metric
extracts, 14 PNG/SVG figure pairs and three editable Mermaid diagrams. Eleven figures
are charts (including three training histories); three are flowcharts. Technical
review includes 14 primary references and 16 panel-question rationales.

Sources: historical capstone comparison package, numerical architecture-family result,
September 27–29 segmental/Swift replay records, physical-phone recorded-input profiles,
multi-sentence tiny-T5 report, execution screening, and actual training histories.
Used saved family timings (e.g. BiLSTM 3.39 ms rather than README 3.33 ms); kept historical
and current contexts distinct. Swift replay 9.68% WER is on Mac, not phone camera
accuracy; device 28.0 ms/frame is a 226-frame recorded-input profile. English results
are generated-session automatic judging. No new inference, training, protected-test
access, model, runtime or data changes.

Author requested 15 signers (10/3/2) and replied “just put it” to the split clarification.
Guide records the requested allocation explicitly while preserving factual provenance:
existing frozen manifest specifies minimum 10/3/5 per class and official participant
separation. No metrics were reassigned to 10/3/2 and no manifest was edited. Published
support establishes signer separation and bounded vocabulary rationale, not universal
adequacy of those exact counts.

Validation: figure generation succeeds; inspected rendered charts/diagrams; 37 local
Markdown links resolve; recorded WER arithmetic recomputes; both manuscript files
remain byte-identical; scoped `git diff --check` passes. Regenerated the large-artifact
index. Promoted authoring rules to `capstone-paper/high.md` and added a current-state
pointer. Next: author reviews the two files and approves or adjusts specific changes
before editing `Capstone 2_ ATLAS_revised.md`.

## 2026-09-30 — persistent manuscript writing guide created

User accepted “Streaming Sign Recognition,” requested simple discussion, real component
names with purposes, functional grouping of sign/spelling recognition, no individual
contribution discussion, no evaluation-status commentary in the manuscript, and no
external data-source names until terminology is discussed. Created
`docs/capstone_papers/ATLAS_WRITING_GUIDE.md` as the authoring reference. It preserves
evidence accuracy, internal provenance, and the distinction between functional grouping
and actual model topology. Comparison topics and generic data labels are suggestions,
not accepted decisions. Bibliography scope needs clarification. Neither manuscript nor
runtime changed. Next: discuss comparisons and data naming before manuscript drafting.

## 2026-09-30 — iPhone paper scope confirmed; award nomination context

User confirmed this is an implementation update: formal testing, human translation
ratings and user/software-quality evaluation remain pending. iPhone is the main
system; desktop is development support. User intends to submit for the Magsaysay
Future Engineers/Technologists Award and requested further discussion before writing.

Inspected latest live-streaming/translation logs and current sibling mobile app's
LiveReelApp, Engine, Models, Decoder and Stage3 Swift sources. Current phone uses
MobileCLIP2 FP16 image encoding, shared multimodal span recognition with word/letter
outputs, separate word and letter boundary models, algorithmic segmental decoding,
fist-letter geometry logistic regression, and incremental tiny-T5 English generation.
T5 encoder/decoder exports are parts of one language model. CTC Stage 2 is absent
from this active path; its timing/sequence role is handled by boundary models and
segmental decoding. Stage names persist in code but need not organize the paper.

Official NAST award criteria checked for submission context. Proposed emphasis:
engineering contribution and evidence, with prospective community benefits separated
from measured outcomes. Next discuss terminology, actual nominee contributions,
ASL target-user rationale, and capstone versus award document format. Manuscripts,
runtime, datasets and models remain unchanged; only this discussion log was edited.

## 2026-09-30 — Capstone revision discussion and initial source review

User requested discussion before rewriting, with a separate updated manuscript for
comparison. Read the local Capstone 2 manuscript, current ground truth, and relevant
desktop entrypoint/segmental/CTC source. The original and existing `_revised.md` are
byte-identical; neither was edited. Initial mismatches: old Reel window/controller
description, 0.4-second Finish hold (current desktop default is 1 second), Mac-only
scope, excluded fingerspelling, and Finish-only translation versus the current
ground-truth record of incremental iPhone translation. CTC remains an experimental
entrypoint rather than the app default. The contents list a performance subsection
that is absent from the body; diagrams are placeholders in this Markdown source.

Proposed terminology, not yet agreed: functional modules, streaming sign segmentation
and recognition, and gloss-to-English generation; reserve Stage labels for internal
code correspondence. Asked the user to choose manuscript status and primary platform.
Next: resolve those choices, inspect the selected platform's implementation and
evaluation evidence, discuss exclusions and human-evaluation status, then create a
new dated manuscript. No runtime/data/model changes or new experiments. Only this
discussion record changed; current project state is unchanged.

## 2026-09-03 20:30 PST — Capstone extractor/architecture claims corrected and latency benchmarked

The Capstone revision package was re-audited after the user challenged the extractor
and part-wise descriptions. The frozen local evidence confirms Apple Vision, not
MediaPipe, won the engineering decision: median active output coverage tied at 87.50%,
while Apple had higher pre-trim source detection (42.65% versus 38.54%), slightly higher
post-trim detection (85.24% versus 84.73%), 0.678 versus 1.230 seconds/clip extraction,
and 93.12% versus 89.95% matched validation top-1. MediaPipe's 82.24% versus 79.52%
mean output coverage came after trimming/interpolation and must not be presented as
superior genuine active-hand detection. The report and extractor chart now foreground
the tied median output and Apple's source-detection, speed, and classifier wins while
retaining MediaPipe's steadier bone/denser-output proxies as secondary diagnostics.
Official Apple/Google sources describe both live APIs but provide no controlled
cross-framework comparison, so the local frozen bakeoff remains decisive.

Direct inspection of the exact current Reel checkpoint
`artifacts/models/stage1_v17_unified_phrase_activity_adapt_reel_v2/best_model.pth`
confirms its landmark configuration is still `temporal_encoder=partwise_global`,
`part_depth=1`, `dim=256`, `depth=4`. The correct name for the whole Stage-1 classifier
is unified multimodal Squeezeformer: part-wise+global is its landmark submodel, joined
to the RGB hand-crop temporal Squeezeformer and learned fusion head.

A new matched batch-one PyTorch CPU single-thread benchmark used one real
`[1,32,61,5]` validation tensor, 20 warmups, three rotated rounds, and 300 timed
predictions per architecture. Median/p90 model-only latency was 11.42/11.86 ms graph
replacement, 7.35/7.80 ms wider flat d384, 4.90/5.09 ms flat d256, and 6.50/6.96 ms
part-wise+global. The raw result is
`artifacts/reports/capstone1_v17_revision_checklist_v1/architecture_latency_benchmark.json`.
The architecture chart now shows both controlled accuracy and this matched latency.
A new Stage-1 training figure reads the actual 30-epoch history from the final
phrase/activity adaptation result and plots recorded loss, validation/context-crop
top-1, learning rate, and selected epoch 26; it does not synthesize curves. The report
also separates model-only latency from live behavior: 16.31 ms Core ML landmark
proposal, 8.64 ms unified Core ML with precomputed embeddings, 359.00 ms baseline full
visual verification, 304.15 ms cached experimental verification, and 0.67 seconds
median committed candidate duration. Chart regeneration, link validation, terminology
scan, compilation, and `git diff --check` pass. No runtime model/checkpoint or protected
test data was changed or accessed.

## 2026-09-03 19:59 PST — Capstone 1 paper revision package grounded in current v17 evidence

The 69-page Capstone 1 PDF at
`/Users/frnzlo/Downloads/For Checking ATLAS (September) (1).pdf` was rendered and
audited against the current v17 implementation and existing experiment reports. The
recommended concise title is “ATLAS: A Squeezeformer-Based System for Sign Language
Recognition and English Translation.” The manuscript should describe cooperating
extraction, recognition, streaming-control, completion, and English-output modules
rather than preserve the obsolete required three/four-stage pipeline. The current Reel
default has the CTC arbiter and targeted MediaPipe mouth verifier disabled; mouth
markers are supplemental, and the Finish button plus held ten-finger gesture are the
two completion controls. The report also corrects the raw v17 tensor to
`[B, 32, 61, 5]`, identifies RGB as cropped-hand evidence, and distinguishes persistent
display landmarks from model observations.

The new private authoring package is
`artifacts/reports/capstone1_v17_revision_checklist_v1/README.md`. It contains a
paper-wide checklist, copy-ready purpose/objectives/scope text, a Mermaid architecture
flow, and matched-data result tables for recognition, hand-crop RGB versus skeletal
landmarks, Apple Vision versus MediaPipe, controlled Squeezeformer variants, contextual
adaptation, and English rephrasing. Five PNG/SVG charts were regenerated from recorded
metrics by `make_charts.py`. Dataset brands are deliberately omitted from this
user-facing package and replaced with “100-gloss corpus”; the authors must insert final
provenance/citations before academic submission. No protected test was rerun and no
new accuracy experiment was needed because matched comparisons already existed.

`STAGE3_HUMAN_EVALUATION.md` provides a 30-item blinded rating sheet: the complete 26
controlled held-out long examples plus four short examples, with semantic adequacy,
grammar, faithfulness, and overall-acceptability rubrics. It intentionally contains no
fabricated human ratings. The current automatic English results remain 94.00% exact /
99.02 chrF++ within the locked 100-gloss scope and 100% exact / 99.57 chrF++ on the 26
controlled long examples; these are prepared-reference metrics, not general
translation evidence. Chart regeneration, local image-link validation, forbidden
dataset-name scan, and `git diff --check` pass. No runtime code, model, data, or
checkpoint was changed for this documentation task.

## 2026-08-24 21:21 PST — v17 source and evidence published to GitHub

The full eligible reorganization, v17 source, iOS projects, tests, documentation,
manifests, and compact reports were committed as
`090d3149db4f0387bdcf55b6d5e1924b8413c0f2` (`Add v17 mobile SLT pipeline and
evidence`) and pushed successfully to
`https://github.com/kokoab/sign_language_translation_system.git`, branch `master`.
The push was a fast-forward from `2a6b743` and local `master` was configured to track
`origin/master`. The separate repository-default `main` branch was not merged,
rewritten, or otherwise modified.

The commit author is the user's configured
`kokoab <francis.batiancela@intechsive.com>` and contains no co-author trailer. The
GitHub publication excludes the large/local paths documented in `.gitignore`; those
assets remain present on this Mac for iPhone builds and experiments.

## 2026-08-24 21:19 PST — physical-iPhone deployment guide and GitHub hygiene

The physical installation procedure is now documented at
`mobile_benchmark/OrientationBenchmarkV17/DEPLOY_TO_IPHONE.md`. It covers exact local
model prerequisites, Apple ID/Personal Team setup, Developer Mode, automatic signing,
unique bundle identifiers, physical-device selection, first installation, file-video
inference, JSON export, physical benchmark discipline, and common signing/device/model
failures. The project remains iOS 17.0+, file-picker based, and not a live-camera app.

Before GitHub publication, the worktree was audited rather than staged blindly. Three
local files exceed GitHub's 100 MB per-file limit, and the repository contains roughly
20 GB of datasets, model assets, Core ML packages, checkpoints, generated build trees,
tool environments, archives, and mobile build products. `.gitignore` now excludes
those reproducible/local products plus generated media and mobile build output while
retaining source, documentation, manifests, and compact evidence reports. Two
reproducible Stage-2 plan JSON files above 10 MB are also excluded. The eligible
untracked publication set is approximately 69.2 MiB across 1,112 files, with an 8.6
MiB largest file. The configured Git author remains the user's
`kokoab <francis.batiancela@intechsive.com>`; no co-author trailer will be added.
