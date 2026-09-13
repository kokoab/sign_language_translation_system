# text-to-sign — log

Measured results, rejected approaches, and progress snapshots. Not read start-to-end —
`rg` this file before re-running an experiment to see if it already failed.

67 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## 2026-09-07 22:26 PHT — full public MoLo annotation audit underway

Downloaded all16 publicly listed EAF transcripts from OSF node9uevh into
`artifacts/reports/continuous_reel_v17_web_audit_v2/molo_annotations/`; all parse
as XML. Several transcripts contain no manual glosses. Both hand tiers, linked
annotation notes, source media names, and signer/session identity must be checked
before claiming usable phrases. Nodewma3e's OSF storage root currently lists zero
files; this does not establish absence of videos elsewhere. No videos or training
data admitted. Added focused auditor regression tests; first run failed as expected
because `scripts.audit_molo_continuous_v17` does not exist yet. Next: implement the
stdlib auditor, verify candidate spans and public media mapping, and record results.

## 2026-09-07 12:23 PST — HELLO symbolic candidate and external-engine feasibility audit

Added exact HELLO candidate with head rim S30007, flat hand S15a11, floor-plane
away S26500 and touch S20500. Parser now permits those exact companion symbols for
that executable pair and rejects arbitrary extras; new regression failed before the
implementation and passes after. Builder checks HELLO's locked Head/Forehead/contact
phonology instead of neutral-space defaults. signwriting_gloss_bank_v17_v5b now has
6/100 executable candidates;94remain explicitly unsupported. Rendered126-frame
YOUR HELLO GOODBYE comparison in signwriting_open_hand_v17_v2 and inspected phase
boundaries plus hand closeups. HELLO hand begins at temple and moves8cm in depth;
source also has obvious lateral displacement, so candidate still needs correction or
review rather than approval. Single-letter user message `l` treated as incidental.

Audited current public signwriting-animation repository at commit
b8c0606d4968ea418caffb7920f3c07208d70c03 (archived2026-08-17). Its documented
signwriting_to_pose CLI is an empty argument-parsing stub; no pretrained animation
checkpoint/model ID ships. Training README assumes private cluster paths, hundreds
ofGB of poses, and unavailable model artifacts. Cloned source under
artifacts/tools/signwriting-animation for evidence only; did not install its large
dependency graph. Attempt to clone public sign/data failed after75s network timeout
and left no directory. Therefore it cannot replace the local renderer now. Research
sources: https://github.com/sign-language-processing/signwriting-animation and
https://github.com/signon-project/wp5-synthesis. Continue local exact symbolic
decoder; do not claim external model inference. Goal active, no training promotion.

## 2026-09-07 12:15 PST — open-hand symbolic primitives and two review candidates

Previous goal turn made concrete progress (parser/YES/build/validation), no blocked
condition. Inspected8sourceframes each forYOUR/GOODBYE/HELLO under
signwriting_open_hand_v17_v1. YOUR source has fingers together despite lexicon5;
dictionaryflatS15a fits this observed source, discrepancy explicitly retained for
fluent review. GOODBYE source includes both lateral wave and later finger hinging.
HELLO source salutes atforehead; do not substitute ambiguouswave dictionary entry.
Official S15a/S14c photographs cached under artifacts/tools/iswa_hand_references.
Added flatS15a/spreadS14c anatomy, flat-hand forwardpush (canonicalright fromS15a28),
and S27206 triple alternating wristflex. ExistingYOU/NEED branches preserved.
23rigtests pass after open-hand regression first failed on unsupportedS15a.
Builder now creates5executable candidates in signwriting_gloss_bank_v17_v4; YOUR
andGOODBYE have explicit variant_notes. GOODBYE candidate wristwave doesnot include
source's later fingerhinging; not exact source reproduction or human-approved.
Rendering YOUR GOODBYE withhandle25671. Inspect hands/joins and validate before
claiming quality.95glosses remain unsupported. YES review pending. Fullgoal active.

## 2026-09-07 12:10 PST — final symbolic expansion checks; review question pending

Latest bank signwriting_gloss_bank_v17_v3 regenerates successfully:3/100executable,
97unsupported. All cached arrays identical to v2 rendered motion; HELP unknown
flexion preserved asnull.31focused rig/mesh/bank tests and git diff --check pass.
CLI rejects source,unknown,andmissing bank motion origins before creating output.
Connected YES YOU NEED YOU has3joins with no active-hand rest reset; checks saved
in signwriting_connected_yes_you_need_v17_v1/join_checks.json. Initial exact-equality
rest check failed on5.96e-8metre float32 rounding only; explicit1e-7tolerance passes.
No motion drift at meaningful scale. Earlier v2 bank checks record735parsed candidates
and metric anatomy. ExistingYOU/NEED cachedcores preserved; resting hand approved.
User review ofYES pending; no render processes left. Full100motion is unfinished:
next resolveYOUR spread-vs-flat notation against its pinned source, then implement
shared open-hand semantics/contact/movement as justified. Do not mark source bank
or syntax parsing as full symbolic coverage. Do not remove corpus references or
admit generated training examples without validation. Goal stays active.

## 2026-09-07 12:09 PST — YES symbolic connected candidate rendered;100 coverage remains incomplete

Reviewed all60frames of signwriting_yes_v17_v1: fist closure too loose; tightened only
S203 to80/175/230degree cumulative finger flexion. v2 symbolic bank generated with
acceptedYOU/NEED caches byte-identical and3fixed-anatomy cores.735/735dictionary
candidates parse;74distinct handshape bases. Parsing is not general motion decoding.
Rendered192frame YES YOU NEED YOU in signwriting_connected_yes_you_need_v17_v1.
Inspected phase boundaries/midpoints plus fist closeups; not all192frames visually
reviewed. Asked user whetherYES fist and repeated wrist bend look correct; pending.
31rig/mesh/banktests and git diff --check passed before subsequent metadata guards.
Builder now records locked phonology and treats missing -100 labels as unknown,
checks selected handshape/sign type/location/contact/repetition/twist explicitly.
Discovery:YOUR lexicon specifies5 while simple dictionarycandidate is flatS15a;
do not select it solely because its motion matchesYOU. SourceHELP flexion isunknown.
Default sequence renderer now also rejects unknown motion-origin labels (legacy
symbol_motion pilot retained). Verify these final guard edits and regenerate report
before handoff.97glosses still have no executable selected notation. No training or
linguistic completeness claims. Goal remains active.

## 2026-09-07 12:05 PST — symbolic parser extended; YES candidate generated, review pending

Added parse_signwriting_signbox in active/v17/avatar_rig_v17.py: preserves sorting,
spatial positions, fill/rotation and categories including fist bases through S204.
Exact executable pair checks still reject unsupported features. Added S20320/S23004
fist/double-flex with explicit thumb preset and palm-axis orientation; accepted
YOU/NEED branches preserved. Official fist photographs downloaded to
artifacts/tools/iswa_hand_references from https://www.signbank.org/iswa/203/203_bs.html.
22 rig tests pass after new regression first failed on absent parser; an initial
single-test command used the wrong test class, corrected by running the module.
New scripts/build_signwriting_gloss_bank_v17.py generates notation inventory for100,
keeps acceptedYOU/NEED caches byte-identical, and buildsYES from static avatar anatomy.
Source videos supply comparison frames only. Output signwriting_gloss_bank_v17_v1:
3executable candidates,97explicitly unsupported, complete=false. Not all100 achieved.
YES human accuracy and thumb geometry still pending visual inspection; render started.
No source fallback, training admission, classifier test, or model promotion.

## 2026-09-07 11:57 PST — user requires SignWriting-driven motion for all100; source bank is reference only

User explicitly requests SignWriting-driven motion for every gloss for accuracy and
consistency. This supersedes source-bank-as-generation plan:98source-estimated clips
must not count as completed symbolic generation. Keep100bank as comparison reference.
Sequence renderer now refuses source-world estimates by default before output creation;
only explicit --reference-motion enables diagnostic comparison. CLIguard verified.
Bankaudit1995 finished100geometry/source/hash checks; one-frame sheets are not linguistic
validation. Right-handed mixed reference render15284 completed but not promoted.

Read official ISWA category/group references at https://www.signbank.org/iswa/,
https://www.signbank.org/iswa/22a_sg.html and https://www.signbank.org/iswa/265_sg.html.
Contact, finger movement, wall/floor/diagonal curves, timing and nonmanuals need distinct
semantics. Existing parser only understands two manual symbolpairs and incorrectly
assumes every hand base beginsS1; fists extend toS204. DictionaryHELLO/GOODBYE share
some wave candidates, so exact-term availability alone cannot select pinned variants.
Next expand semantic parsing and reusable hand/motion primitives, starting with YES
(fistS20320 + repeated wristflexS23004 matching locked phonology), preserving accepted
YOU/NEED and explicit rejection for unsupported features. All100 remains unfinished.
No synthetic admission, new model training, or official test run. Full goal active.

## 2026-09-07 11:55 PST —100motion assets built; bank-wide audit in progress

Full bank build39179 completed successfully:100/100 exact frozen glosses,0failures,
98source-world estimates plus2approved symbolic cores. Output avatar_gloss_bank_v17_v1.
This establishes asset coverage only; handshape/contact/nonmanual quality unproven.
HELLO HOW YOU smoke89965 completed and source comparisons inspected at every phase
boundary/midpoint. SourceHELLO is left-handed whileYOUisright; added mirror_avatar
with side-swap/coordinate reflection regression, and sequence renderer converts
left-only source motion to right-handed avatar with explicitly mirrored source panel.
Two-handed dominance not guessed.30focused tests pass. Shared rig was extended only
with mirror helper while bank process retained earlier code; bank retarget functions
unchanged. Source-rig hash at process start remains recorded.
Right-handed mixed smoke15284 rendering. New audit_avatar_gloss_bank_v17.py verifies
100exactidentities/source/cache hashes, finite poses and metric anatomy, and renders
13source/blue-avatar coverage sheets. Running full-bank audit; inspect findings next.
No production translator promotion, synthetic training admission or official test run.

## 2026-09-07 07:28 PST — connected blue YOU NEED YOU rendered and every frame inspected

Connected render89169 completed: artifacts/reports/signwriting_avatar_connected_v17_v1/
avatar.mp4 and comparison.mp4,147frames/30fps/4.9s. Inspected all147frames on five
numbered sheets. Generated joins50:59 and92:101 keep active wrist in signing space
(minheight1.137m), maxstep3.40cm/3.22cm, no neutral reset. Inactive hand changes only
floating-point noise below1e-6m. All resampled core pose arrays equal cached pilot
samples exactly. Maxrelative hand-bone error6.24e-6; no observed palm fanning or false
face lift in reviewed frames. Recorded join_checks.json and frame_review.md.
Asked one async review question explicitly linking our blue connected avatar for
remaining unnecessary movements. No answer yet. Human acceptance and training
eligibility remain false: timing/coarticulation/ASL grammar have no paired validation.
Full goal remains active; this is a connected two-sign vocabulary pilot, not100-sign
text translation or proof synthetic training removes recording needs. Pending final
CLI guard/core preservation/test checks session71366; no rendering remains active.

## 2026-09-07 07:25 PST — coherent palm blending extended to connected signing

Prior goal turn made progress: blue-avatar boundary defects reproduced, fixed and
allframes inspected. Full objective remains unmet; new full-motion review pending.
Moved palm-local blending into shared interpolate_world_hand so source gap filling
and metric transitions use the same correction. Degenerate palm estimates retain
finite bone-arc fallback; exact endpoints/identical hands are preserved. Extended
regression failed against old shared helper, now passes. Reconstructed v10 isolated
boundaries differ by less than1e-6m after refactor; no isolated render regression.
Added connect_avatar_clips: joins cores directly with no intervening neutral pose,
marks generated transitions unobserved, and recomputes fixed-length arm IK. New
regression failed before implementation; all20rig+6mesh tests pass afterwards.
New scripts/render_signwriting_sequence_v17.py takes existing pilot cores and ordered
glosses, resamples to30fps, generates one approach/release around whole sequence,
and exports video, source comparison (joins explicitly unpaired), timeline/pose audit.
YOU NEED YOU rendering session89169 to signwriting_avatar_connected_v17_v1; check
every frame, join geometry and preserved cores next. Timing/grammar are not validated,
no training eligibility, no classifier test rerun or model change.

## 2026-09-07 07:21 PST — own blue YOU/NEED frame audit completed with revised artifacts

V10_palm_onset render39406 completed; inspected every52YOU/53NEED frame. Saved
six numbered sheets, frame_motion.csv, motion_checks.json and frame_review.md.
The previously observed upward excursions and palm fanning are removed in this
candidate. YOUcore is6cmstraight forward with no orientation changes; NEEDhas2flex
strokes around a stationary wrist. Both cores bit-identical to v6_calibrated, and
inactive hand unchanged. Approach height rises monotonically; release lowers to
exact initial neutral hand. Max release wrist step10.6cmYOU/9.7cmNEED: timing still
an assumption requiring review, not proven natural by tests. All25rig/mesh tests
passed after final code changes; git diff --check passed. User's handshape feedback
applies to own blue avatar; exact accepted prior version was not specified.
Latest review clips: artifacts/reports/signwriting_avatar_pilot_v17_v10_palm_onset/
you_avatar.mp4 and need_avatar.mp4. Matching source comparisons: you.mp4/need.mp4.
No rendering jobs remain active. Full goal remains unachieved and active: broader
notation coverage, natural connected signing, real continuous generalization and
synthetic transfer still need evidence; training_eligible and human_accepted false.
Next safe action: incorporate full-motion review and apply coherent palm blending
to connected transitions with regression/source comparison before vocabulary expansion.

## 2026-09-07 07:19 PST — palm distortion during approach reproduced and fixed

Inspected every v9_clean_motion frame (52YOU,53NEED). Core is exactly identical
to v6_calibrated (max difference0), YOUhas6cmforward travel and no core rotation,
NEEDhas stationary wrist with exactly2flex strokes, inactive hand unchanged.
Approach/release wrist height is monotonic and ends in the initial neutral pose.
However NEEDapproach still visibly fans/twists the palm: independent world-space
bone interpolation rotates each metacarpal around a different arc. New rigid-palm
regression failed with6.2cm excess finger-base separation for a rotated fixed palm.
Fixed only isolated-boundary interpolation: use installed SciPy Slerp for one palm
rotation, blend finger directions in palm-local coordinates using existing helper.
Core handshape/motion arrays remain untouched. All19rig+6mesh tests pass and
 git diff --check passes. V10_palm_onset rendering session39406 now running.
V9 remains a diagnostic candidate, not a validated final result; do not claim
natural signing from kinematic tests. Inspect every v10frame next. Full goal active.

## 2026-09-07 07:14 PST — frame-by-frame blue-avatar audit identifies false onset and release motion

User requests every-frame inspection for unnecessary movement before/after signs.
Inspected all44YOU and43NEED v7_onset frames (four numbered contact sheets saved
beside videos). YOUframes12:18 fall from face to signing space and37:43 rise again;
NEEDframes11:15 drop then39:42 rise. These are detector-gap artifacts, not intended
sign motion. v7 core was bit-identical to v3, proving that preserving it also
preserved the defect. Root retargeter mapped missing zero wrist sentinels to
shoulder-height and smoothed them into valid neighbors. All3script callers checked.
Added a regression covering leading/interior/trailing gaps and all-missing wrists;
it failed before fix. Now interpolate missing wrists from available coordinates
before smoothing, or use neutral wrist if none exist. Handshape code unchanged.
All18rig+6mesh tests and git diff --check pass. Renderingv8_onset session84702;
inspect every output frame and quantify changes before considering the fix verified.
Goal remains active; no synthetic admission, classifier training, or test-set run.

## 2026-09-07 07:10 PST — own blue avatar onset implementation and focused regression

Added prepend_isolated_approach in active/v17/avatar_rig_v17.py, reusing existing
metric hand interpolation and arm IK. Only participating hands approach from a
neutral hip pose; inactive hand and original core arrays are unchanged. Synthetic
prefix is marked unobserved and approach-assumed, never added between signs.
Regression failed with missing helper before implementation; all17rig tests pass,
including exact core preservation, fixed bone lengths, finite poses and prefix seam.
Updated scripts/render_signwriting_handshape_pilot_v17.py to show source fromframe0,
prepend at least0.4s approach, and save full-frame blue-avatar videos plus comparison.
Source lead-in does not provide reliable measured3D motion: approach remains inferred,
not a claim of source-exact timing. Render session38046 started outputv7_onset;
visual review, output/core comparison and mesh tests follow. No training changes.

## 2026-09-07 06:59 PST — user accepts YOU/NEED shapes; onset visibility is the next correction

User feedback: "the shapes are fine" for new YOU and NEED renders, but they appear
to start already in the signing position. This accepts the reviewed handshapes only;
whole signing, timing, transitions, vocabulary expansion and training eligibility
are not accepted. Keep current handshape constraints while fixing onset.

Inspected exact frame0 screenshots of you_frontal and need_you: both start at rest,
but resting fingertips are below the camera crop. YOU close-up crops out the whole
rest-to-sign approach; source comparisons also use hand-trimmed source intervals.
Next: wider framing that includes both resting hands, original source lead-in, and a
brief neutral opening hold for review only. No per-sign delay in continuous signing.

Before feedback, added signwriting_pilot_sigml in avatar_rig_v17.py and CLI
scripts/export_signwriting_sigml_v17.py. Exports selected YOU/NEED entries and ordered
combinations, rejects unsupported glosses/notational mismatches, retains provenance
and marks output unreviewed. Regression failed before implementation, all16rig tests
now pass. CLI rejects unknown gloss before creating output. Exported NEED YOU XML
is structurally identical to prior rendered draft. CWASA regenerated49frames from
exported notation (session58843 completed); numerical equivalence check follows.
No model/test partition/training action. Full goal remains active.

## 2026-09-07 06:54 PST — connected notation animation and corrected NEED finger pilot completed

CWASA now renders NEED -> YOU as one49-frame sequence (31+18), saved in
artifacts/reports/signwriting_cwasa_comparison_v17_v1/need_you/need_you.mp4.
All49 captured frame indices verified in order. Native container duration1.633008s;
CAS durations sum approximately1.633333s. No paired continuous source exists for
this join. Wrist step at the boundary4.48deg; index middle joint26.44deg per frame,
so do not claim natural transition timing from absence of a crash or a smooth mesh.

Found a semantic mismatch: HamNoSys mainbend=hooked folds both PIP and DIP,
whereas this pilot's SignWriting S106 preset bends PIP and keeps DIP straight.
Measured bend2 overrides at levels2/3/4: PIP53.86/80.79/107.72deg, DIP0.
Using Anna bend2="0 3.34 0 0" gives PIP89.94deg, DIP0. This is calibrated for
Anna and unreviewed, not a universal anatomical/linguistic equivalence. NEED uses
right shoulder location to better match training footage, and repeated wrist flex.
Preserved rejected/default hooked and chest-location versions for comparison.

Inspected connected motion sheet and new NEED source comparison. Source and avatar
phase still differ; comparison explicitly retimes avatar rather than proving timing.
need_you/need_comparison.mp4 contains35 source frames12:47 (1.166822s container).
YOU close-up comparison remains you_frontal/you_detail.mp4 (19frames,1.266160s).
Reports pin source, notation, capture script, page and vendor hashes. All8vendor
assets retain upstream hashes. git diff --check passed; no production code or model
changed and no test-set evaluation performed. No synthetic training admission.

Full goal remains active and unachieved: arbitrary text/SignWriting, all100 exact
variants, natural connected signing/nonmanuals, synthetic transfer and general live
recognition remain unproven. Latest YOU human-review question is still pending.
Next useful action is apply that review and validate/automate notation conversion
before expanding beyond these2signs. All rendering jobs completed. Local review
server47782 remains available at127.0.0.1:8769 (verified HTTP200 at handoff).

## 2026-09-07 06:43 PST — frontal YOU comparison rendered; human handshape review requested

Previous goal turn made progress: an external notation engine now generates actual
avatar motion. Full objective remains unmet; no training/runtime promotion.
CWASA comparison directory now contains you_frontal/you_comparison.mp4 and
YOU hand close-up you_detail.mp4, inspected against actual P51 train footage.
Frontal camera improves visibility of the forward-pointing index. Original tilted
camera made it look downward. Merged handconfig, split handconfig and equivalent
HamNoSys tokens all generate similar wrist rotations; encoding style is not the
cause. Probe produced54frames across3signs, saved under orientation_probe/.

Fixed capture harness: player frame buttons wrap and schedule steps asynchronously.
Earlier root step images have cyclic offset; do not use those for timing assessment.
New capture waits for exact reported frame0..17 and verifies each before screenshot.
Native animation600.00005ms; source frames7:26 at15.00577fps last1.26618s. Comparison
retimes avatar to source duration and explicitly labels phase alignment. No matching
timing claim. Comparison report pins source hash, vendor hash and capture indices.
Latest async question asks whether rightmost YOU finger/palm is better or acceptable;
no answer yet. Earlier v3 question is superseded for YOU by this engine comparison.
No source classifier test, training data admission, or production code changed.

## 2026-09-07 12:39 PHT — SignWriting procedural bank reached structural coverage

`scripts/build_signwriting_procedural_bank_v17.py` completed
`artifacts/reports/signwriting_procedural_bank_v17_v1/` with 100/100 frozen Citizen
gloss entries and no generation failures. The six hand-reviewed symbolic profiles are
preserved; the other 94 select dictionary notation by locked-phonology score and use
the official-photo ISWA hand templates plus the procedural movement interpreter. This
is structural generation coverage only: it is not human validation of all 100 signs,
and nonmanual head/body/dynamics symbols remain recorded but unanimated. A full rig
audit and source-comparison render review are the next gate before any training use.

The first full audit passed 38 focused tests and verified finite fixed anatomy for all
100 assets (worst relative bone-length error 9.97e-6; largest wrist step 3.58 cm).
Thirteen source/avatar coverage sheets expose that the bank is not yet semantically
acceptable. The shared motion interpreter does not execute S216/S218 squeeze or S21d
flick, and the automatic candidate score does not compare ISWA handshape names against
the locked selected-finger/flexion fields. UNDERSTAND demonstrates the root cause: its
locked baby-O/bent onset selected a fully extended S100 index candidate. Location and
palm orientation also visibly disagree for several early pronouns. Keep the report's
`training_eligible=false` and do not call 100/100 file coverage signing accuracy.

## 2026-09-06 22:09 PST — CWASA produces first YOU animation; stepped capture underway

Client-side CWASA now loads Anna and produces one YOU sign with18 CAS frames;
bone arrays differ across frames (18 changed bones between first and midpoint).
Saved you_animation.json. Avatar screenshot confirms textured humanoid. This is
engine execution evidence, not correct ASL. Event capture has no window errors;
engine emits animgenAllocate diagnostic warnings, retained in detailed logs.

Copied unchanged official assets locally with SHA256 vendor_manifest.json. Two
missing qskin shader files caused blank rendering after successful local avatar
load; retrieved those exact official files. No vendor code modified. Native headless
requestAnimationFrame probe varies across fresh processes, so the offline capture
harness now explicitly uses a timer frame scheduler; no real-time/mobile claim.
Direct canvas export in avatarframe hook was black (before compositor rendering).
Use native Chrome screenshot after GUI Suspend/Next frame stepping, preserving
all18frames for review. Session96524 rendering; local server47782 remains live.

## 2026-09-06 22:05 PST — external renderer smoke has not generated animation yet

Created artifacts/reports/signwriting_cwasa_comparison_v17_v1/index.html,
you.sigml and capture.mjs. Experimental YOU uses gestural SiGML finger2,
forward extended-finger axis, inward palm, small forward motion. Conversion is
manually bounded and unverified; not a general SignWriting converter.
CWASA engine downloaded unchanged for source inspection, SHA256
294699dacfa460ae8402cb4fbc27a1da985a310d9d5d81153d942991b8822c2c.
No runtime or training pipeline changed.

Chrome --dump-dom timed out even for a trivial data page. Native Chrome debugging
connection works with an isolated temporary profile; screenshots and page state
captured. Current screenshot has empty avatar panel. Assets and avatar JAR requests
complete, but avatar startup has not resolved. Native requestAnimationFrame probe
passes; do not blame missing display clock. Capturing detailed official engine logs
now. Server sigmlserver.pl returns HTTP500 for both our YOU and UEA's own iTakeMug
reference, so that endpoint is unavailable independently of our input.

Local server session47782 (127.0.0.1:8769) still live for this experiment. No usable
avatar comparison or acceptance claim. Performs (UPF successor of SignON-realizer)
is an alternative MIT engine with SiGML/BML and nonmanual controls; only docs read,
no install/integration. Source https://github.com/upf-gti/performs .

## 2026-09-06 21:56 PST — CWASA comparison interface and usage terms checked

UEA's current standard release resolves to vhg2026. Official conditions permit
example page adaptation under CC BY-SA; underlying engine must remain unchanged
(on terms equivalent to CC BY-ND). Standard avatars include Anna, Marc, Francoise,
Luna and Siggi; do not assume extra evaluation avatars can ship. No engine modified,
no authors contacted. Official event hooks expose per-sign animation frame JSON;
client-side SiGML generation is available. This does not accept SignWriting directly.

Next experiment: a local comparison page using the unmodified hosted engine and a
bounded YOU/NEED notation bridge. Validate actual generated hand pose before scaling.
In-app browser execution tool is absent after tool discovery; installed Chrome's
headless CLI can test an isolated local page without accessing the user's profile.
No new dependency needed. No production integration or fluency claim.
Sources: https://vh.cmp.uea.ac.uk/index.php/CWASA_Conditions_of_Use
https://vhg.cmp.uea.ac.uk/tech/jas/vhg2026/index.html
https://vh.cmp.uea.ac.uk/index.php/Logging_and_Event_Hooks

## 2026-09-06 21:51 PST — calibrated symbolmotion inspected; renderer comparison warranted

V6calibrated completed, no liveprocess. Inspected YOU/NEEDmidframes; NEED nowuses
class-specifichigherplacement, but orientation/phase still visibly differs. Video
containerdurations verify YOU1.199805s, NEED1.100034s. Per-signerpose and corephase
remain unresolved; corpusmedians are not sufficient anatomy/motion validation.
Duration-report fields were added to source while rendering; v6reportcodehash was
corrected to the executedversion by removing exactly those two laterlines, with an
explicit provenance note. Avoid editing active renderer during future runs.

Currenthandwritten engine supports3pairs versus210symbolbases in dictionarycandidates.
Next useful comparison is existing notation-to-avatar machinery (UEACWASA/SiGML),
subject to actual availableinterface/terms; SignWriting would still require a semantic
bridge. No claim of working integration or arbitrarySignWriting support. This is
an engineering comparison to avoid proliferating unvalidatedper-symbolmotionrules.

## 2026-09-06 21:50 PST — symbolic placement/duration calibrated from13 train signers per sign

Previous turn progress: wrongNEEDrepetition fixed,100dictionarycandidatecoverage.
Measured officialtrain archives with≥10observed rightwrist/indexMCPframes:13distinct
signers each YOU/NEED. Mediantrimduration YOU1.1998s, NEED1.1s; median body-relative
wrist YOU[-.30127,.44080], NEED[-.71094,.11011]. Map isotropically with avatarshoulder
width. These are noisy detector/corpus medians, not measured3Dmotioncapture.
Saved train_motion_calibration.json with26archivehashes, per-samplevalues and explicit
selection. Sourcevideos/testpartitions unmodified; no extraction/training restart.

animate_signwriting_pilot accepts finite wrist_position; translation regression
verifies calibratedoffset without changedfingergeometry. Pilot --calibration requires
symbolmotion and pinsunchangedtrainarchives; consumes per-sign placement and median
duration. Sourcepanel phases are retimed to same duration and report explicitly says
this is not originalsource-speed comparison. Depth.28m/travel.06m/flex65deg remain
assumptions.21rig/mesh tests pass;diffcheck passes. V6calibrated rendering under
session10982, logrender_v6.log. No humanapproval or full100generationclaim.

## 2026-09-06 21:26 PST — dictionary covers100 exact terms; repeated NEED renders but remains visually weak

Read the cached official dictionary against exact casefolded canonical labels or
pinned ASL-LEX entry IDs (no substring alias guessing). All100classes have candidates;
none is yet variant-verified. Across candidateentries there are210distinct spatial
symbolbases, including alternate forms/nonmanuals. Saved allcandidates, hashes and
coverage in signwriting_avatar_pilot_v17_v1/dictionary_100_coverage.json. This rules
out missing lexical terms as the immediate blocker; it does not establish usable
100sign animation coverage. Current pilot parser executes onlythree exactpairs.

V5repeated render completed (session94873). Inspected NEED frame21: repeatedmotion
is implemented, but source/generated phase, wristplacement and orientation still
visibly disagree. Numeric wristangle has firstmaximum atframe10, returnsnearstart,
then ends atsecondmaximum; maxangle1.13446rad. Correct repetition alone is not enough
for acceptance.29focusedrig/mesh/context tests pass; no nativeapproval, synthetic
trainingadmission or runtimepromotion. Humanv3handshapequestion remainspending.
Fullgoal remainsactive and unachieved. No liveprocess fromthisturn.

## 2026-09-06 21:25 PST — wrong NEED dictionary variant identified and corrected before tuning

Previous turn progress: optionalproposalruntime and symbolmotionrender evidence.
Read lockedphonology: NEED(C_02_034) repeated_movement=1, YOU=0. Initial NEED
candidate S10620+S22e04 denotes singlewristflex, contradicting pinnedvariant.
Official ISWA https://www.signbank.org/iswa/22a_sg.html confirms S230 doublewristflex.
Selected existing exactterm dictionarycandidate AS10620S23004M514x524S10620486x476S23004489x506
instead; preserved oldpilotentries as pilot_entries_single_flex_rejected.json.
Newstatus verifies repetitiononly, not whole lexical equivalence/nativeacceptance.

Added signwriting_pilot_symbols: parses one signbox separately from sortprefix,
refuses unsupported extra symbols/pairs, validates repetition against lockedlexicon.
Renderer invokes before extraction and pinslexiconhash. Added doubleflex execution
flex→return→flex; tests verify repeatedposeextrema and refusal of single-vs-repeat
mismatch. Both RED→GREEN,15rigtests pass. Rendering v5_repeated, logrender_v5.log;
no claimedvisual/native success. Existing v3/v4reportcopies remain historical.

## 2026-09-06 21:22 PST — symbol-motion v4 renders; visual fidelity still insufficient for promotion

V4motion process24067 completed. Inspected YOU midpoint and NEED early/late frames.
Symbol pipeline executes the smallforwardstroke and changingwristorientation with
metric boneerror≤6.24e-6, but NEED's stylized startingorientation, wristheight/timing
and renderedhand still visibly differ from source. No claim of superior signing or
nativeacceptance. Outputs: signwriting_avatar_pilot_v17_v4_motion/you.mp4,need.mp4,
source/detector/symbolpanels plus numericalaudits andreport. Handshape-onlyv3 remains
the pending humanreview, avoid conflating it with approval of symbolicmotionv4.

Extended unknown/empty guard regression to exercise proposal-mode CTC argument;
scorer with no proposal method is nevercalled for guardedleadinghypotheses.27focused
context/avatar/mesh tests pass and diffcheck passes. Fullgoal still active: optional
proposalruntime works but omitsSCHOOL, symbolmotion coversonly2patterns, all100 and
naturaljoins/nonmanuals/synthetictransfer remain unproven. No livejobs fromthisturn.
Next: fit/validate symbolic motion timing and start/end hand orientation against
multiple training source performances; use humanv3feedback whenavailable. Keep
phonologicalclarity distinct from surface meshrealism and do not remove data gates.

## 2026-09-06 21:14 PST — corrected symbolic handshape clips inspected and sent for review

V3 SignWriting handshape comparison completed for both YOU/NEED. Full-view frames
and rendered closeups were inspected: index is foreshortened forward for YOU and
palmward bent for NEED after the anatomical-side fix. This is model visual assessment,
not fluent acceptance. Sent async request with you_detail.mp4 and need_detail.mp4 in
artifacts/reports/signwriting_avatar_pilot_v17_v3. Detail clips deliberately show
middle16 normalized phases at15Hz, explicitly recorded as non-original-speed; full
clips restore source duration.19 rig/mesh/source-match tests and diffcheck pass.
Still source-derived palm orientation/thumb/wrist; no full FSW motion implementation,
no signed-human review, no continuous synthetic training admission.

While review is pending, launched actual-video smoke on same6 development clips as
video_nine, using causal_v2 recognizer and matching Stage3_causal_v2 (existing runtime
beam-only context). Output continuous_rebuild_v17_v1/video_causal_v2 with per-cliplogs.
No sealed test or altered selection. Compare runtime results rather than treating
cached development metrics as real-camera evidence. Full goal remains incomplete.

## 2026-09-06 21:12 PST — symbolic pilot v2 rendered; inspection catches palm-side sign error

SignWriting handshape v2 rendered YOU and NEED with source/detector/symbol panels.
Both have13/32 imputed world frames, explicitly counted; metric bone error below
5e-6. Visual inspection contradicts readiness: NEED bent dorsally instead of toward
palm. A canonical right-palm-facing-camera test reproduced wrongZcurl (RED); corrected
anatomical side sign (right+cross(index-pinky,middle-wrist),left-negative).18 rig/mesh
tests now pass. V3 rendering under session50517, log render_v3.log in v1pilotfolder.
This is actual progress and a falsified initial rendering, not human-approved signing.
Full FSW orientations/movement/nonmanuals and generated continuous joins remain absent.

## 2026-09-06 21:11 PST — first SignWriting handshape rendering experiment implemented

Previous goal turn classified as progress (official movement-symbol evidence), not
verified wait; no background job was assumed running. Answered user's status query:
renderer fixes tested, SignWriting rendering not yet established, broader system
experimental. Added constrain_signwriting_handshape in shared avatar rig for S100
Index and S106 IndexBent only. It preserves metric lengths and source palm/knuckle
positions, closes nonselected fingers, uses explicit fixed-angle presets, and refuses
unsupported symbols. Test RED(import missing)→GREEN;17 rig/mesh tests now pass,
git diff --check passes. This is a controlled handshape pilot, not full notation
execution: fill/rotation, movement, wrist, thumb and timing are still source-derived.

First YOU preview was visually inspected and exposed an elevated pointing direction
from using the metacarpal axis; selected index now starts from source proximal axis
projected into the palm plane. This has not yet been proven visually correct.
New scripts/render_signwriting_handshape_pilot_v17.py compares exact P51 train
YOU/NEED footage, detector retarget and symbol-controlled handshape. It parses only
spatial signbox hand symbols, not the optional FSW sorting prefix, and saves NPZ
geometry, hashes and imputation counts. Render running under session83448,
log signwriting_avatar_pilot_v17_v1/render_v2.log, output signwriting_avatar_pilot_v17_v2.
No new dataset split, training admission, runtime promotion or fluent acceptance.

## 2026-09-06 16:44 PST — dictionary-grounded SignWriting candidates acquired for video review

Fetched the official SignMaker ASL dictionary into non-split local cache
`data/local/signwriting_symbolic_pilot_v1/dictionary-ase.js` (1,380,817bytes).
Exact term lookup found45 candidate rows across YOU/NEED/HELLO/HOW, including
multiple lexical variants and aliases. Initial parser failed on literal tabs in JS
strings; stdlib JSON strict=False reads them without executing JavaScript.
Saved source URL, full-file SHA256 and exact candidates in
`artifacts/reports/signwriting_avatar_pilot_v17_v1/dictionary_candidates.json`.
`pilot_entries.json` pins provisional YOU(S10040 plusS26500) and
NEED(S10620 plusS22e04) candidates, explicitly not variant/native verified.
The previous public model's YOU handshape/movement matches a dictionary candidate;
that is useful provenance, not proof of its whole-text translator's accuracy.
Official Index and IndexBent reference photos were downloaded and inspected.
P51 Citizen train NEED video/contact sheet was inspected; no test data touched.
The large27.8MB hand-symbol PDF was rejected by web-tool content-size limit;
no claim of reading its full contents. No new avatar generator was implemented in
this step. Next: verify candidate orientation/movement against exact source video,
then make the bounded symbol-to-handshape renderer comparison. User reviewers assess
video; none reads SignWriting. Keep this branch review-only and unpromoted.

## 2026-09-06 16:40 PST — no reviewers read SignWriting; evaluate rendered ASL instead

User confirms none of the available fluent ASL reviewers reads SignWriting. This
is not a blocker or permission to treat machine-generated notation as verified.
Use traceable dictionary/reference symbols and compare resulting video to exact
source signs. Reviewers assess intended meaning, handshape, palm orientation,
location, movement and transitions in video; notation correctness remains indirectly
validated and should be labeled accordingly. No new recording request is required
for this bounded pilot. YOU/NEED are the first controlled contrast.
Official ISWA Index group confirms S100=Index and S106=Index Bent; avoid guessing
symbol identities or conflating the handshape symbol with the full sign YOU/NEED.
Reference: https://signbank.org/iswa/100_sg.html . Dictionary lookup is in progress.

## 2026-09-06 16:39 PST — user reopens SignWriting → avatar → video as a generation route

User explicitly requests investigating text→SignWriting→avatar→video for accuracy.
This steers the existing reverse-generation objective; it does not cancel continuous
recognition or authorize claiming synthetic transfer success. Read the existing
scripts/evaluate_signwriting_symbolic_pilot_v17.py and its saved report. The Sep2
pilot only tested a public text↔SignWriting model and notation rendering; it never
implemented/evaluated SignWriting→3D motion. All5 natural phrases parsed;0/5 strictly
round-tripped, versus7/12 isolated probes. Reverse translation is a fallible diagnostic,
not native semantic ground truth. HELLO/HOW/YOU isolated probes did round-trip but
remain unverified model outputs. Do not rerun the same experiment as new evidence.

Primary-source feasibility review: SignWriting ISWA explicitly covers hands,
movement, dynamics/timing, head/face, body and locations. tuniSigner describes an
SWML parser→explicit linguistic representation→joint animation/IK system, including
handshape/orientation, movement and nonmanuals. This supports feasibility, not a
measured advantage on this project's ASL. Its intermediate rules and timing choices
show why notation does not directly supply a complete natural 3D performance.
Sutton core processes FSW/SWU text; it is not an avatar engine. UEA CWASA animates
SiGML derived from HamNoSys, a distinct notation; it can be a renderer comparison,
not a drop-in SignWriting parser or an assumed automatic conversion.
Sources checked:
- https://www.signwriting.org/lessons/iswa/
- https://www.signwriting.org/symposium/archive/sws0023_Paper_tuniSigner_Avatar_Programming_Yosra_Bouzid.pdf
- https://github.com/sutton-signwriting/core
- https://github.com/sign-language-processing/signwriting-translation
- https://vh.cmp.uea.ac.uk/index.php/CWA_Signing_Avatars_Demos
- https://arxiv.org/abs/1502.02961

Recommended bounded experiment: exact locked gloss/ASL variant→verified FSW→explicit
handshape/orientation/contact/movement constraints→existing metric avatar→video.
Start YOU versus NEED (extended versus bent index), HELLO and HOW plus their joins.
Retain source videos for style/timing and use native comparison for semantic and
motion acceptance. Free-text ASL planning remains a separate translation gate; do not
substitute word-by-word English lookup for ASL grammar. Generated motion can also
export landmarks but must remain review-only until genuine held-out transfer improves.
Asked whether the existing fluent reviewers read SignWriting; answer pending. No
new SignWriting implementation, dependency, training data, API call, or model promotion.

Other work preserved: v10 YOU closeup exists at
artifacts/reports/avatar_source_comparison_v17_v10/you_detail.mp4 and was inspected.
The index/renderer improved, but curled fingers and motion are still not human-approved.
V10 full comparison's middle panel accidentally contains the previous three-panel
comparison; use the focused YOU clip for review, not the cluttered full comparison.
Shared context reranker now accepts optional exact-CTC-scored motion proposals;
new missing-candidate regression passes,7 context tests pass. Runtime has not yet
been wired to supply those CTC rows, so live behavior still uses beam-only reranking.

## 2026-09-06 16:35 PST — source YOU closeup reveals incorrect camera origin

V9 isolated the mesh roll fix and produced comparison.mp4 plus you.mp4/source closeup.
Visual review still showed incorrect pointing projection. `project` was dividing
absolute world height by camera depth, equivalent to a ground-level camera shifted
on screen. Thus advancing a finger below chest height appeared to move it upward.
A new regression reproduced this. Projection now subtracts cameraheight1.2m before
perspective division, principalrow40% ofimageheight. Both camera and mesh regressions
pass;16 focused rig/mesh tests and git diff --check pass. V10 sourcecomparison is
rendering with both fixes and the earlier lexical extension constraint. These remain
review-only anatomical/camera assumptions; no native acceptance or training admission.

## 2026-09-06 16:13 PST — avatar body scale corrected; matched-camera extraction reaches220train clips

Source-to-avatar XY was still independently scaled by.43/.48 despite v17 already
using isotropic shoulder-width units. New regression failed on .43 vs avatar shoulder
span.335425; both axes now use .335425 and shoulder center1.285. All nine rig tests
pass. V7 source comparison completed under avatar_source_comparison_v17_v7; its visual
review is still pending. This must not be confused with new user acceptance of v6.

Causal local cache process is confirmed live and reached220train clips; do not restart.
Training cache now also hashes original source archives after substitution, preserving
split/annotation provenance across cached retraining. Added automatic utterance-end
handling after1.2s of no detected hands (configurable/disableable), independent of
per-sign recognition; Enter still finishes explicitly. This does not require a pause
between signs. The new end-of-utterance UI branch needs an actual video smoke before
handoff. No model promotion or synthetic admission is authorized by these changes.

## 2026-09-06 16:10 PST — retimed generation passes machine gate; causal local retraining preparation started

Retimed generation v2 completed successfully for HELLO HOW YOU, selecting P33.
Source/composed Stage1 labels are exact and existing boundary gates pass. Output:
`artifacts/reports/grounded_source_timing_v17_v2`. This is machine-gated review-only
content; no fluent acceptance or synthetic transfer claim. The original P51 comparison
v6 remains the pending user review artifact.

Read-only duration audit: all487 local development videos are30Hz;287train clips
contain18,714frames (mean65.21),200validation clips24,852frames (mean124.26). That
~1.9x duration difference and causal-vs-whole-window normalization justify a bounded
matched-camera experiment rather than another decoder threshold tweak.

Added `scripts/cache_causal_local_v17.py` to re-extract those exact development videos
using current live normalization, all-frame Apple body/face/hand detections, source
hashes and role/identity/target provenance. It is running (confirmed40train clips) at
`artifacts/generated/causal_local_v17_v1`, log causal_local_cache.log. No test videos.
`train_continuous_evidence_v17.py` now optionally loads pinned causal local archives
and adds explicitly named training-only duration variants while validation remains
unchanged. Cache records and verifies data_policy. Python compile passed; full load,
training, runtime measurement and promotion remain pending until extraction finishes.

## 2026-09-06 16:07 PST — source-timing fix reaches generator; six-template camera check remains weak

`load_isolated` in generate_grounded_text_to_sign_v17 now restores source duration
from processed frames, decoded/sample fraction andfps before activity trimming,
default15Hz; generator passes its logical_fps and records timing in artifact contract4.
The first actual generation run failed because `recognize` forwarded >32frames to
Stage1. Added a failing real-model60-frame regression, then resample only the classifier
input to32; both timing tests now pass. Animation duration remains independent.
The retimed HELLO HOW YOU generation v2 is running, log grounded_timing_v2.log.
Renderer now respects artifactlogical_fps rather than unconditional duplicate frames.

Actual-video evaluation finished on the first clip per distinct local validation
phrase. There are six distinct phrases in this signer split, not nine (report folder
video_nine is a stale descriptive name). Raw2/6 exact, guarded-context3/6; WER46.67%
->33.33%. TOMORROW SCHOOL GO emitted nothing. Its raw footage was inspected (upright,
lowlight, hands visible); no rotation bug established. These small development results
are not a continuous success gate or proof of replacement-data sufficiency.

V6 source-comparison motion sheet was inspected: the earlier looping YOU hand and
raised elbows are corrected in checked frames, and joins have smoother wrists. A new
concrete side-by-side review question is pending. No acceptance inferred. Removed the
three unused procedural-mesh helpers from the new renderer (capsule/ellipsoid/draw_mesh)
after confirming no callers.19 focused post-cleanup tests and gitdiffcheck passed.

## 2026-09-06 16:02 PST — full motion review rejects raised elbows; fixed-pole IK and wrist paths added

V5 comparison video was inspected as a12-frame motion sheet. Corrected source hand
assignment fixes YOU's loop, but joins expose raised/flipping elbows and implausible
wrist excursions. A new low-wrist regression measured an elbow at1.3425m above its
1.285m shoulder. Replaced discrete planar elbow-solution selection with a continuous
downward/outward pole; all12 avatar tests pass. Source-world animation transitions
now use quintic endpoint wrist interpolation and arc-interpolated hand shapes. This
is explicit procedural animation, not a learned or validated natural-signing claim.
V6 comparison is rendering under avatar_source_comparison_v17_v6.

Fixed camera tick semantics: a late camera frame no longer populates earlier30Hz
ticks; missing elapsed ticks hold the preceding observed feature. No new model/data
experiment or test access. Full visual/native signing acceptance and synthetic transfer
remain open. Source timing is restored only in the comparison tool; the original
landmark generator's normalized timebase remains a separate unresolved issue.

## 2026-09-06 15:59 PST — source comparison catches 3D handedness dropout; timing/shape joins corrected

Source comparison v2 improved HELLO/HOW but YOU still looped. Inspection found the
world estimate missing on the expected right slot at source frame16: MediaPipe
misclassified chirality on frames11/15/16. Added minimum-distance wrist matching to
Apple's observed hand tracks in `compare_avatar_source_v17.py`; a targeted test passes.
V3 source comparison confirms the YOU frame now uses3D, and its planar loop is gone.
Source footage/rejected/current preview remains under avatar_source_comparison_v17_v3.

V4 restores each gloss's duration from actual source frame mappings/fps. Missing3D
samples interpolate or extend same-sign estimates and are marked in world_imputed;
they must not count as detector observations. No replacement data enters recognition.
Added `interpolate_world_hand` with arc directions and length preservation, tested
on opposite endpoints after an initial failure. V5 rendering is now running with
whole transition handshape interpolation between actual retargeted endpoint poses,
including inactive neutral hands, preserving the separate wrist trajectory/arm IK.
All41 focused rig/mesh/signing-voice/transition/continuous/context tests passed.

This is still an animation audit, not successful synthetic corpus generation:
review of every transition, nonmanual grammar, unseen combinations, real-data transfer,
and robust continuous-camera recognition remain incomplete. The previous goal turn
made material progress (source evidence, implementation, rendered artifacts/tests).
No genuine external blocker is established. Keep the full goal active.

## 2026-09-06 10:41 PST — user rejects avatar fingers/joins; actual source videos expose root causes

User reviewed v5: fingers loop unnaturally, hands are wrong, transitions remain poor;
explicitly requested comparison with actual videos. Root inspected the three exact
P51 Citizen train raw videos (HELLO 081380748015633, HOW 1334132333742275, YOU
17987836792170242). Five-frame source contact sheets and source frame/fps metadata
are saved under `artifacts/reports/avatar_source_comparison_v17_v1/`.

Confirmed: avatar X/hand-side mapping mirrors the source; fixed-length reconstruction
normalizes every foreshortened 2D finger edge into a full-length planar edge, creating
loops; source inactive arm is down/out of frame while heuristic same-signer rest
retrieval inserts a curled stomach pose. Source trim durations differ (HELLO 28 frames
at30fps, HOW40 at30fps, YOU19 at15fps), while the phrase reuses normalized32-frame
samples on a15fps animation clock. These are code/data defects, not approval issues.
The mirrored-hand regression now fails as expected. A separate transition rotation
regression also found a 180-degree midpoint snap when endpoint directions oppose.

Next action: correct coordinate handedness and use genuine detector-estimated 3D hand
shape from the exact train videos for a source-aligned retargeting audit, with its own
animation-only contract. Existing MediaPipe hand model/package are available locally;
Apple trained model inputs remain unchanged. Native rejection means synthetic admission
stays false. Do not describe v5 as realistic signing or ask for another review before
checking the replacement against these sources.

The first end-to-end CPU video smoke (validation PLEASE_HELP_ME) completed: raw
PLEASE PLEASE HELP I -> guarded-context PLEASE HELP I -> reviewed English Please help
me.123 camera observations,7.51s wall for4.1s video; this is successful integration on
one clip, not real-time/general accuracy. Report: continuous_rebuild_v17_v1/video_smoke.

## 2026-09-06 10:31 PST — motion-context results are mixed; avatar clearance regression passes

Stage-3 scorer v1 selected epoch 12 and development weight 1.0. Against CTC beam,
local phrase exact rose 57/200 -> 112/200 and WER fell 33.15% -> 20.37%; 68 clips
improved, six worsened, one correct phrase was corrupted. Citizen rolling exact rose
336/378 -> 356/378; SemLex 745/978 -> 769/978, with nine correct clips corrupted.
ASLLRP remained 0/12 exact. NCSLGR full-token WER (including OTHER) worsened 59.50%
-> 66.12%. Shuffled motion gives worse local WER (23.52%) but better isolated scores
than matched motion, exposing a large language/length-prior contribution. The result
is NOT promoted. Corrected the trainer/report's misleading automatic `promoted` flag.
This does not prove a reliable landmark-informed fallback or unseen combinations.

Eight-clip CPU incremental smoke: 0/8 exact, 52.38% WER; median inference 57.39 ms,
p95 106.30 ms at an observation every 133.33 ms. Desktop inference only, concurrent
CPU training, excludes camera extraction; no iPhone or end-to-end latency claim.

Avatar v4 adds an explicit .235 m signing plane with bounded .035 m scale cue; mesh
front measured below .16 m. This is an animation assumption, never source 3D recovery.
The new regression failed before the change and all seven avatar tests now pass.
A closer fixed camera and v4 render are running; visual/native acceptance remains open.

## 2026-09-06 10:22 PST — interrupted avatar work audited; surface and palm defects corrected

The avatar worker reached a MakeHuman mesh prototype before its usage-limit failure.
The v2 artifacts exist under `artifacts/reports/rigged_avatar_v17_v2/`; same-signer P51
training observations supply 136/73 low-motion left/right rest candidates. This is
heuristic rest retrieval, not human-verified rest annotation. Source assets and hashes
are recorded in its report and kept under `artifacts/tools/makehuman_cc0`.

Visual inspection rejected the v2 contact sheet: dropped alternate mesh triangles
caused holes, and unmapped metacarpals stretched the hands. The root thread corrected
`scripts/render_rigged_avatar_v17.py` to retain all 26,756 body triangles (formerly
11,002 after trimming/decimation), map the four metacarpals on each hand, and replace
an antiparallel -I reflection with a proper 180-degree rotation. Three new regression
tests first failed on the relevant defects and now pass; all six avatar mesh/rig tests
pass. A replacement render and visual acceptance remain pending.

The continuous prefix decoder/runtime scaffolding now exists in
`continuous_decode_v17.py`, `continuous_runtime_v17.py`, and
`scripts/evaluate_continuous_runtime_v17.py`. The decoder's two focused tests pass:
beam probabilities agree with exhaustive CTC paths including repeated words, and new
words arrive before earlier words stabilize while recent text can revise. This proves
decoder mechanics only; real runtime recognition has not yet been evaluated. The clean
Stage-1 v2 job remains live and has reached epoch 18; do not restart it from a stale log.

## 2026-09-04 11:56 PST — annotation audit identifies suppression and domain errors

Read-only inspection does not support the hypothesis that the twelve ASLLRP validation
sequences are simply mislabeled or cropped incorrectly. All 24 manually annotated
sign intervals fall inside their extracted clips; every cached landmark window is
valid and has full hand-frame presence. When the phrase-adapted Stage-1 checkpoint is
given each manual sign core separately, it reaches 16/24 top-1 and 21/24 top-3. The
labels and signs therefore mostly agree, while the weak second-position performance
and very short signs remain real difficulties: validation sign cores have a median
seven frames, the second signs a median 5.5 frames, and the minimum is two frames.

Two supervision problems are now the primary explanation. First, the selected Stage-1
phrase adaptation was trained only on local phrases using equal ordered partitions;
it never used the manually aligned ASLLRP sign cores. Its 95.77% Citizen result does
not imply continuous ASLLRP signer/domain robustness. Second, the unified-streaming
run hard-labeled 3,802 isolated prefixes at 50--72% completion as CTC blank. Many are
already recognizable at that point, so this teaches delayed emission and can directly
cause missing final signs. Only genuine background and inter-sign transition regions
should be blank; uncertain prefixes should be ignored, softly aligned, or handled by
CTC without hard negative labels.

The current multimodal phrase cache also contains only six of the nine raw local
templates. `I WANT FOOD`, `SORRY I LATE`, and `YESTERDAY TEACHER MEET` were excluded
because `FOOD`, `LATE`, `TEACHER`, and `MEET` are outside the locked 100 glosses.
Their 624 training and 156 validation full-trajectory v17 archives already exist and
can be reused safely by mapping the out-of-vocabulary tokens to explicit `OTHER`, not
blank. This expands local temporal variety without pretending those tokens are known.

The next bounded experiment should therefore precede any architectural expansion:
(1) adapt the Stage-1 encoder on the 92 manually aligned ASLLRP training sign cores
with isolated replay and validate on the 24 manual validation cores; (2) remove
isolated prefixes from blank supervision; (3) add the three omitted local templates
as known-plus-`OTHER` sequences; and (4) retrain the same rolling CTC head. Required
gates are improved ASLLRP continuous WER/exact match, Citizen isolated retention near
the accepted checkpoint, and low false emission on genuine transitions. Ordinary
non-sign activity remains a separate missing negative set and must not be substituted
with unlabeled signing.

## 2026-09-03 08:48 PST — ten-finger finish control added to stable Reel path

The default `scripts/live_reel_stage1_v17.py` path now treats two separated, upright
open palms with all ten fingers extended as the Finish control. The pose must remain
detected for 0.4 seconds, tolerates only a 0.15-second landmark dropout, latches after
one activation, and rearms only after release. While the pose is active, its frames are
excluded from both Stage-1 candidates and the optional Stage-2 buffer so a UI gesture
cannot become a learned gloss. The HUD shows hold progress. The existing Finish button
and `F` key remain available; `--no-finish-gesture` disables the gesture and
`--finish-gesture-hold-seconds` adjusts its hold duration. The JSON session config and
finish event preserve whether/how it was used. Nineteen focused Reel tests, Python
compilation, CLI-help validation, and `git diff --check` pass. This is deterministic
landmark geometry, not a newly trained sign class; it still needs a live user check for
Apple Vision hand-landmark tolerance and false triggers.

## 2026-09-02 07:17 PST — standalone SignWriting pilot runs locally but fails the first semantic gate

The public English-to-ASL SignWriting checkpoint was evaluated by itself before any
pose/video work. `scripts/evaluate_signwriting_symbolic_pilot_v17.py` runs the official
`signwriting-translation` code in an isolated Python 3.11 environment, renders Formal
SignWriting locally, and compares three symbolic strategies: natural-English input,
project-gloss-order input, and concatenation of separately generated isolated forms.
It does not generate landmarks, pose, or video.

The local non-split cache is
`data/local/signwriting_symbolic_pilot_v1/hf_cache/` (about 463 MB). The forward model
is `sign/sockeye-text-to-factored-signwriting` revision
`3a45c3bedc0c6ee08fcb3e87b1aaee602dfa06d9`, licensed CC BY-NC 4.0. The separate
diagnostic reverse model is `sign/sockeye-signwriting-to-text` revision
`8f314871ee42ad68f9e01ab23e818d8aaa695665`, whose model card says MIT. The existing
project venv was not modified; the official package needs Python 3.10+ whereas the
project venv is Python 3.9.6.

The review artifact is `artifacts/reports/signwriting_symbolic_pilot_v1/index.html`,
with raw FSW, local renders, and machine-readable results in `report.json`. All five
natural phrase outputs and all twelve isolated lexical outputs are structurally
parseable FSW. Syntax alone is insufficient: none of the five natural phrase outputs
strictly round-trips to its source text through the separate reverse checkpoint. The
diagnostics are: `Good morning -> daughter`, `Hello, how are you? -> you`, `My name ->
synagogues`, `Thank you friend -> face the punctuation`, and `I will go to school
tomorrow -> day before yesterday`. Only 7/12 isolated probes strictly round-trip.
Gloss-order prompting and concatenated isolated outputs remain parseable but do not
repair phrase semantics under the same diagnostic. The reverse model is not ground
truth and these results still require a fluent ASL signer who reads SignWriting, but
the severe omissions and unrelated returns are enough to fail an unattended
automation gate.

Decision: keep SignWriting as a potentially useful representation, but hold this
public checkpoint before connecting either Rylo or a local pose renderer. The next
safe comparison is native review of the report followed by a dictionary-grounded or
locally fine-tuned symbolic planner using only verified entries for the 100-sign
vocabulary. No pilot output was added to Stage-2 training/validation/test, no split or
sealed data was accessed, and no pose generation was resumed.

## 2026-09-02 06:59 PST — evaluate SignWriting as the reverse symbolic layer, not Stage-2 motion data

The reverse-generation branch is paused for architecture discussion. Rylo's public
design should be evaluated for reuse instead of rebuilding its language-production
stack: spoken text -> normalized text/SignWriting -> pose -> skeleton/avatar. Public
references are `https://github.com/sign/translate`,
`https://github.com/sign-language-processing/signwriting-translation`, and
`https://github.com/sign-language-processing/spoken-to-signed-translation`.

SignWriting and SignBank+ are valid candidate datasets for a future reverse symbolic
planner. They can supervise English/text -> ASL notation, expand lexical and
phonological coverage, and provide explicit handshape, orientation, location,
movement, and nonmanual constraints for a SignWriting-conditioned pose generator.
They are not continuous-landmark datasets: notation rows do not contain measured
frame timing, signer identity/style, natural inter-sign coarticulation, camera/view
variation, detector missingness, or genuine phrase video. Therefore they must not be
counted as Stage-2 CTC training examples or used by themselves as proof that generated
motion is human-correct.

A motion generator would still require paired SignWriting + genuine pose/video (and
preferably temporal alignment) or independently reviewed motion. Rylo's public
gloss-to-pose baseline primarily performs lexicon lookup, cropping, concatenation,
and smoothing; its output can be an external renderer/baseline but not recognition
ground truth. Keep three data roles separate: genuine continuous phrase landmarks for
Stage 2, SignWriting/text pairs for reverse language planning, and generated poses for
native review only unless separately validated.

No SignWriting data or model was downloaded or admitted to a split in this decision.
The newly staged transition-safe coverage audit remains unrun, and no further reverse
generation was started while this architecture is being discussed.

## 2026-09-01 22:41 PST — genuine-entry fallback removes sampled jerk; full reverse demo passes

All four v6 high-jerk failures originated at the join into the next genuine clip, not
inside the learned transition. `synthesize_join` now tries the original boundary and,
only if its local speed/acceleration/jerk gate fails, may skip at most the first one or
two complete genuine entry frames. It never fabricates a frame, never changes hand
participation, and does not use a longer crop merely to make a numerical gate pass.
The composed right-hand lexical segment is reclassified after this preparation and
must retain its exact Stage-1 gloss.

The balanced 48-pair fast report is now
`artifacts/reports/stage2_v17_grounded_transition_scalability_fast_v7/report.json`.
All 48 boundaries pass motion, presence, hand-side, and bone gates. Overall 46/48
(95.83%) pass after adding the exact post-preparation lexical-label gate; `HOT LESS`
and `LEARN DAY` are rejected because P37's only complete-edge source is no longer
classified as the intended right gloss. The prior 44/48 result therefore improved
without hiding lexical damage.

The reserved unbiased confirmation audit is
`artifacts/reports/stage2_v17_grounded_transition_scalability_200_v2/report.json`:
178/200 (89.0%) pass all required gates, versus 159/200 (79.5%) in the previous
200-pair report. Motion alone passes 189/200 (94.5%); thirteen rows fail the new
post-preparation Stage-1 label safeguard, with two overlapping motion failures. Only
five of 200 accepted/rejected samples needed a one- or two-frame entry fallback. The
one-to-two-hand stratum remains weakest at 46/56 (82.14%). This is evidence of a
remaining source-clip/data-quality gap, not permission to trim 3–4 lexical frames.

The grounded text renderer now retains every Citizen candidate clip per signer/gloss
instead of silently overwriting all but the last. For each phrase it evaluates the
primary exact candidate plus at most two one-clip substitutions per gloss, and
`--allow-different-signers` now actually selects a same signer independently for each
phrase rather than requiring every signer to cover the union of all requested words.
Both source clips and post-join gloss segments must classify exactly. Local boundary
motion, anatomy, bone, complete-hand, and hand-participation checks remain hard gates.
The corpus-wide normalized-motion ratio is now a ranking diagnostic, matching the
scalability audit, because isolated Citizen clips upsampled into a phrase have much
lower high-order motion than the nine directly extracted 128-frame local phrases; it
must not veto an otherwise exact and locally plausible boundary.

The complete current reverse-path run is
`artifacts/reports/stage2_v17_grounded_text_input_demo_v2/`. All five supported English
inputs render successfully with exact source and composed gloss predictions and all
hard machine gates: `GOOD MORNING` (P40), `HELLO HOW YOU` (P51), `MY NAME` (P52),
`THANKYOU FRIEND` (P52), and `TOMORROW SCHOOL GO` (P31). Three pass the global motion
diagnostic; `MY NAME` and `THANKYOU FRIEND` do not, which is explicitly recorded rather
than hidden. The contact sheet was visually inspected and shows complete hands plus
stable face/body rigs with correct one-/two-hand changes. Repaired outlier boundary
frames are separately shown in
`artifacts/reports/stage2_v17_grounded_transition_repaired_outlier_review_v1/contact_sheet.png`.

`venv/bin/python -m unittest test.test_signing_voice_v17 -v` passes all 18 tests and
`git diff --check` passes. Generated artifacts remain native-review-only and are not
training, validation, or test data. No Citizen test or other sealed/test split was
accessed.

## 2026-09-01 22:19 PST — grounded joins now use complete real hand edges; fast gate reaches 44/48

The shared grounded phrase path now separates Stage-1 recognition trimming from
transition-only edge trimming. Stage 1 still receives the original observed isolated
span. Immediately before joining, `trim_transition_span` removes partial-hand edge
frames and retains the actual one-/two-hand participation. Transition stabilization
then reconstructs a complete 21-node hand only for a side genuinely present in at
least one neighboring sign; a side absent from both signs remains absent. This fixes
the floating-point/partial-hand rendering defect without forcing every sign to use two
hands. The face/body anatomy rig remains complete on every rendered frame.

The timing path retains its learned duration except for a narrow kinematic guard: if
the generated transition speed is below 0.25x the genuine neighboring-gloss p95, it
tries shorter allowed spans and accepts a replacement only when speed, acceleration,
and jerk all fall inside the existing 0.25x–4x gate. This changed `IMPORTANT WATER`
from 12 to 11 transition frames without regressing an accepted pair.

The same seeded 48-pair/four-stratum audit now passes 44/48 (91.67%), up from 40/48
(83.33%):
`artifacts/reports/stage2_v17_grounded_transition_scalability_fast_v6/report.json`.
All presence, no-invented-hand, handless-frame, transition-only-node, and hand-bone
gates pass on all 48. The four rejected high-jerk boundaries are `LISTEN EAT`,
`SICK MY`, `SICK COLD`, and `TRY COLD`; testing every learned allowable duration
(4–12 frames) found no duration that passed, so these must remain rejected rather
than admitted as generated data. The earlier direct hand-completion trial failed the
transition-only-node gate (6/48), and spherical/interior-time smoothing trials
regressed the motion screen; neither trial is retained.

Two native-review-only renders visually confirm the repaired structural behavior:
`artifacts/reports/stage2_v17_grounded_text_input_visual_fix_v1/` renders
two-handed `IMPORTANT` into one-handed `WATER`, while
`artifacts/reports/stage2_v17_grounded_text_input_visual_one_hand_v1/` keeps
`FIND THEY` one-handed throughout. Both reports pass boundary gates, preserve source
and rig hand participation exactly, keep every active rig hand at 21 nodes, and keep
face/body nodes complete. Contact sheets were visually inspected; native ASL motion
review is still required. These artifacts remain ineligible for recognition training,
validation, or testing.

`venv/bin/python -m unittest test.test_signing_voice_v17 -v` passes all 17 tests, and
`git diff --check` passes. No Citizen test split or other sealed/test split was
accessed.

## 2026-09-01 22:03 PST — 48-pair fast transition gate replaces blind batch rendering

The scalability audit now uses seeded random sampling within each hand-participation
stratum instead of lexicographically spaced rows. Its focused regression passes. A
fast gate of 12 samples from each of the four observed one/two-hand transition
patterns completed in approximately 12 seconds and passed 40/48 pairs (83.33%). The
report is
`artifacts/reports/stage2_v17_grounded_transition_scalability_fast_v1/report.json`.

All eight failures were `transition_motion_orders_in_gloss_range`; two also failed the
non-blocking global motion diagnostic. There were no missing-body, missing-hand,
forced-second-hand, presence, or hand-bone failures. Pattern pass rates were 75.0%
one-to-one hand, 83.33% one-to-two, 83.33% two-to-one, and 91.67% two-to-two. The
failed ordered pairs were `FIND THEY`, `LIKE MY`, `LISTEN EAT`, `FATHER BAD`,
`HE HUNGRY`, `ANGRY HOME`, `WHAT WATER`, and `MAKE READ`.

Use this 48-pair numerical screen before rendering phrase batches. Render only
accepted phrases; retain the unbiased 200-pair audit as a later confirmation gate.
This shortens iteration but does not relax native-signer review or make generated
artifacts eligible for training, validation, or testing. No sealed/test split was
accessed.

## 2026-09-01 21:25 PST — natural-text input reaches the gated grounded renderer

`active/v17/text_to_sign_phrase_catalog_v17.json` is the first explicit reverse-path
language contract. It maps five curated conversational English inputs and bounded
aliases to the already gated gloss sequences. The generator now accepts repeated
`--text` inputs, records the original text and catalog hash, renders both the text and
resolved glosses in each video, and preserves the full lexical/transition/anatomy gate
chain. Unsupported English fails before model or artifact work and lists the supported
normalized inputs; this prototype does not pretend that a five-entry catalog is a
general English-to-ASL translator.

The complete natural-text run is
`artifacts/reports/stage2_v17_grounded_text_input_demo_v1/`: `Good morning`,
`Hello, how are you?`, `My name`, `Thank you friend`, and
`I will go to school tomorrow` all resolve to the intended gloss sequences and pass
the same exact-content, one-hand, motion, bone, transition, face/body, and artifact
gates. `index.html` is the canonical native-review entry point. The source catalog
and all generated artifacts remain training/validation/test-ineligible, and native ASL
review is still incomplete. No sealed or test split was accessed.

## 2026-09-01 21:11 PST — grounded text-to-sign pilot passes machine gates; native review required

The first reviewable unseen-combination path now grounds lexical motion in exact
train-only ASL Citizen isolated clips from one real signer and generates only the
short boundary between adjacent signs. `scripts/generate_grounded_text_to_sign_v17.py`
discovers train signers covering every requested gloss, verifies each isolated clip
with the frozen landmark Stage-1 checkpoint, preserves each source gloss's own
left/right-hand participation, and uses the all-real transition inpainter plus the
multi-corpus timing model for the boundary. Signer selection uses exact component
recognition, source anatomy coverage, and distance to the aggregate genuine local
train motion distribution. It does not inspect either held-out phrase trajectory.

The selected signer is P40. Stage 1 correctly recognizes all five requested source
clips: `GOOD` 0.7847, `MORNING` 0.9150, `TOMORROW` 0.9049, `SCHOOL` 0.9715, and `GO`
0.9116. The independent `SORRY` audit is exactly one-handed (`[true,false]`) and is
recognized as `SORRY`, proving the path does not add a second hand. Per-gloss source,
generated-observation, and render-rig participation match exactly: `GOOD` and
`TOMORROW` remain one-handed; `MORNING`, `SCHOOL`, and `GO` use both hands.

Both unseen compositions pass the aggregate genuine-motion gate. `GOOD MORNING` p95
speed/acceleration/jerk are 1.446x/1.051x/1.092x genuine local train;
`TOMORROW SCHOOL GO` is 1.183x/1.083x/0.957x. Every transition contains a hand, has
zero presence changes at its joins, and introduces zero nodes absent at both boundary
endpoints. Transition/gloss p95 speed ratios are 0.524x and 0.731x. The v3 animation
contract makes face and body complete on every rendered frame and every active hand a
complete 21-node hand, while leaving an unused hand absent. Contact-sheet inspection
confirmed readable bodies and hands without between-sign disappearance.

The review artifacts and videos are under
`artifacts/reports/stage2_v17_grounded_text_to_sign_pilot_v1/`; the machine-readable
record is `report.json`. Every artifact explicitly sets training, validation, and test
eligibility to false. Passing these gates permits native-signer review only; it does
not establish fluent coarticulation and must not be used as recognition ground truth.
No Citizen test, local test, SemLex test, How2Sign validation/test, or other sealed
split was accessed.

## 2026-09-01 18:12 PST — 64-step temporal tokenizer passes every unseen reconstruction gate

The accepted tokenizer doubles temporal resolution to 64 discrete code steps for 128
frames, increases coordinate weight only inside the reconstruction objective, and
retains EMA codebook updates plus class-balanced absence supervision. Its checkpoint
is `artifacts/models/temporal_motion_tokenizer_v17_v2/model.pth`, SHA-256
`efc6e138a397bb48352a4686b814de2fec902913379880329964d67b4857fc09`, selected at
epoch 38. Neither `GOOD MORNING` nor `TOMORROW SCHOOL GO` appeared in tokenizer
training.

All held-out reconstruction gates pass: coordinate loss 0.0320 (<0.04), presence F1
0.9827, exact phrase-level hand-participation accuracy 1.000, 51 active codes with
23.53 aggregate perplexity, and reconstructed p95 speed/acceleration/jerk at
1.030x/0.957x/1.075x genuine motion. Ordinary validation has coordinate 0.0275,
presence F1 0.9900, hand participation 0.9973, and 81 active codes. This proves the
discrete temporal representation can reproduce unseen genuine phrase trajectories
without pose averaging or adding an unused hand. It authorizes training a
gloss-conditioned temporal code prior, but does not yet authorize an unseen generated
phrase render.

## 2026-09-01 18:00 PST — temporal tokenizer smoke restores real motion range; full gate pending

`model_temporal_motion_tokenizer_v17.py` replaces one utterance-wide latent with 32
time-varying discrete codes across each 128-frame trajectory. Its convolutional
encoder/decoder reconstructs observation XYZ, detector presence, and confidence; it
does not use the animation rig. `train_temporal_motion_tokenizer_v17.py` excludes both
compositional holdouts from self-supervised fitting, balances genuine source families,
and blocks temporal-prior training unless reconstruction anatomy, coordinate, motion,
and code-utilization gates all pass.

The first three-epoch smoke was rejected because its learned-gradient codebook
collapsed to three active codes and its reconstruction marked every hand/body node
present. EMA codebook initialization and class-balanced absence supervision fixed the
root mechanisms. After five smoke epochs, reconstructed holdout p95 motion reached
0.675x genuine speed, 0.786x acceleration, and 0.801x jerk, passing all three motion
range gates and substantially exceeding the rejected global-latent generators. It no
longer forces all nodes present. The smoke still fails overall: holdout presence F1 is
0.847, hand participation is 0.917, coordinate loss is 0.066, and only 10/128 codes
are active. Therefore it remains non-renderable and cannot yet authorize prior
training. The next action is a full tokenizer run, not a gloss generator.

## 2026-09-01 17:50 PST — 2,095 genuine whole-utterance trajectories complete and audited

The whole-utterance Apple Vision extraction completed in 3,353 seconds: 2,094 new
archives plus one previously verified archive, zero failures, and exact coverage of
all 2,095 manifest rows. A full cold-read integrity audit found no missing archives,
no non-finite values, and exact `[128,61,5]` observation shapes and manifest metadata.
Counts are 780 local phrases, 1,104 ASLLRP `OTHER`, 155 2M-Flores, and 56 exact
ASLLRP spans. Split counts remain 1,702 train and 393 validation with the ASLLRP signer
boundary intact. Observed source participation is 60 right-only, 81 left-only, and
1,954 two-hand trajectories; these are detector observations, not rig-forced hands.
Mean observation presence is 84.27% (range 26.95--100%), mean duration is 4.53 s
(range 0.3--42.13 s), and the corpus occupies 89 MB. No generated motion or sealed
test source is present. The canonical extraction report is
`artifacts/reports/stage2_v17_full_trajectory_generation_v1/extraction.json`.

## 2026-09-01 17:24 PST — signer-profile transfer is presence-invariant; jerk is a render gate

`apply_voice_profile_to_trajectory` now transfers a learned isolated-signer spatial
profile onto an arbitrary-length continuous trajectory without modifying detector
presence or confidence. Coordinates are transformed only where the source trajectory
is present; absent nodes remain exactly zero. Therefore this style/accent operation
cannot invent a second hand for a one-handed sign. Its resampled temporal curve is
disabled by default until continuous-motion evaluation justifies using it. A focused
128-frame one-handed regression test proves the unused hand remains absent and all
auxiliary channels are bit-identical; all 17 signing-voice and full-trajectory tests
pass at that step. The unseen-generation evaluator now selects candidate latents using speed,
acceleration, and jerk together and independently requires every generated holdout to
remain within 0.25--4x the genuine held-out p95 for each measure. This prevents a
right-speed but snapping trajectory from passing. All 18 focused tests pass after the
new three-order motion gate. This only establishes the representation safety invariant, not that generated
signer identity or motion is perceptually genuine. The whole-utterance extraction is
still live at 1,375/2,095 rows with one existing skipped archive and zero failures.
Re-evaluating the strongest local-only CVAE smoke under the stricter gate rejected it:
`GOOD MORNING` reached 0.239x genuine speed, 0.305x acceleration, and 0.279x jerk;
`TOMORROW SCHOOL GO` reached only 0.112x, 0.104x, and 0.107x respectively. It also
lost to the source-family mean on validation/holdout coordinates and holdout velocity.
The machine-readable rejection is
`artifacts/models/full_trajectory_generator_v17_local_cvae_kl_smoke/evaluation_motion_gate.json`;
rendering remains disabled.

Checkpoint selection for future full-trajectory runs now uses
`validation_prior.loss`, not target-conditioned posterior reconstruction loss. This
aligns early stopping with the actual text-to-sign inference path; posterior metrics
remain diagnostic only. The focused full-trajectory tests pass. Extraction is at
1,425/2,095 with zero failures.

## 2026-09-01 17:19 PST — whole-utterance conditional generation corpus/model underway; unsafe smokes rejected

The reverse path no longer uses independently normalized 32-frame pieces.
`scripts/prepare_full_trajectory_manifest_v17.py` combines every available labeled
real clip from all nine local phrase families, ASLLRP exact spans, ASLLRP `OTHER`
spans, and 2M-Flores into
`active/v17/full_trajectory_generation_manifest_v17.json`: 2,095 rows, 454 literal
gloss/special tokens, 1,702 train and 393 validation. Source roles remain unchanged;
ASLLRP validation is signer-held-out, local validation is familiar-source, and 2M dev
remains train-role only. Local `FOOD`, `LATE`, `TEACHER`, and `MEET` stay literal and
are not mapped into Stage 1. The extractor writes one 128-step trajectory per complete
video under one coordinate normalization. Its active run has reached 1,125/2,095 with
zero failures; all 780 local clips are complete.

`model_full_trajectory_v17.py` and `train_full_trajectory_v17.py` implement a
gloss-conditioned frame decoder with coordinate, detector-presence/confidence,
velocity, acceleration, jerk/scale, hand-participation, and duration supervision.
`GOOD MORNING` and `TOMORROW SCHOOL GO` are excluded entirely as compositional
holdouts because all five component glosses occur independently outside those exact
sequences. `HELLO HOW YOU` was not used as the holdout because `HELLO` has only one
training occurrence outside that phrase, which would confound composition with a
missing-token problem.

The first deterministic local-only smoke was rejected: although ordinary validation
coordinates beat a source-family mean, unseen coordinates did not, participation was
91.7%, and every hand node appeared in every generated frame. Stronger negative
presence and temporal-change loss removed the all-present collapse, but generated
speed remained only 10--11% of genuine holdouts. A conditional motion-latent encoder
was then added because point regression averages distinct performances into slow
motion. With posterior motion evidence, the local latent smoke reached 90.9% held-out
presence F1 and 97.9% participation, proving representation capacity; honest prior
samples reached 25.2% of genuine speed for `GOOD MORNING` but only 11.2% for
`TOMORROW SCHOOL GO`, and coordinate gates still failed. All these smokes remain
non-renderable. `evaluate_full_trajectory_generator_v17.py` now rejects all-present,
static-presence, slow-motion, participation, coordinate, and velocity failures before
any unseen-combination video can be written. No synthetic result entered any dataset
or review video.

## 2026-09-01 16:46 PST — v3 preserves one-handed signs; safe exact text-to-sign path added

The user correctly rejected the v2 rig assumption that every displayed sign should
contain two hands. `complete_landmark_anatomy` now preserves linguistic hand-side
participation: a hand never observed in the source remains exactly absent, a
participating hand bridges at most three missing detector frames, and longer inactive
spans remain absent. Face/body completion remains render-only. Artifact contract v3
stores explicit `animation_rig_presence` and `animation_rig_confidence` alongside XYZ;
loaders reject v2/legacy artifacts, require rig and observation hand participation to
match, and retain the hard training/validation/test-ineligible flags.

The rebuilt train-only anatomy package is
`artifacts/models/signing_landmark_anatomy_v17_v3/anatomy.npz`, SHA-256
`fdd286f73e49a69d686b7b014b9d736192b29b818f22d575cc1e8efca29e0f06`.
It preserves exact participation for all 100 gloss prototypes: 48 selected sources are
one-handed and remain one-handed; 52 are two-handed and remain two-handed. Both sparse
observation and v3 rig content accuracy are 100% on this train-only content gate. V2 is
explicitly marked obsolete and must not be used.

The all-nine genuine reference was regenerated under v3 after auditing every train
recording and choosing, within each phrase's top detection-quality quartile, the take
closest to median duration. `SORRY_I_LATE` is one-handed in this selected performance
and remains `[left=false, right=true]` in the rig; every other phrase's observed/rig
side participation also matches exactly. `scripts/render_text_to_sign_retrieval_v17.py`
adds an honest reverse baseline: plain text such as `hello how you` resolves to an
exact known local phrase and renders its genuine full trajectory as a landmark-only
skeleton. Unknown combinations fail closed instead of invoking the rejected isolated
stitcher. The example is
`artifacts/reports/stage2_v17_text_to_sign_retrieval_v1/phrase.mp4`. Twenty-five
focused tests pass and `git diff --check` is clean. This is retrieval, not yet unseen
phrase generation.

## 2026-09-01 14:47 PST — genuine all-nine phrase render establishes the visual baseline

`scripts/render_genuine_local_phrase_reference_v17.py` now selects the longest
train-role recording in each of the nine local phrase families and extracts each whole
performance under one coordinate normalization. It does not stitch independently
normalized windows, concatenate isolated signs, or generate transitions. The 53.9 s
side-by-side reference video is
`artifacts/reports/stage2_v17_genuine_local_phrase_reference_v1/genuine_local_phrase_reference.mp4`
(SHA-256 `22bf2ae1c159eabb4afcc4ae2dfc199ece6ed077f98f9283ec95f71da9093040`).
The left panel shows sparse detector observations; the right shows a render-only
completed rig driven by the exact same genuine motion. The report folder also contains
a nine-phrase contact sheet, machine-readable report, and one separated raw artifact
per phrase.

Across these full phrase references, real observation presence ranges from 60.3% to
79.5%. Every artifact preserves observed XYZ exactly after float16 serialization
(maximum measured error 0), retains observation masks separately, contains no
ambiguous `landmarks` tensor, and is explicitly reference-only/ineligible for dataset
splits. Twenty-two focused tests pass and `git diff --check` is clean. This proves the
render path can keep the avatar visible without altering recognizer evidence. It does
not prove novel-combination generation, learned full-utterance coarticulation, or
native linguistic naturalness; rejected synthetic generation remains disabled pending
review of this genuine-motion baseline.

## 2026-09-01 14:44 PST — render rigs can no longer masquerade as recognizer observations

The landmark artifact architecture now separates two representations. Detector-style
`recognition_prototypes` retain real sparse presence/confidence, while
`animation_rig_prototypes` are completed only for drawing. The anatomy builder writes
the v2 contract without the old ambiguous `content_prototypes` key. The phrase renderer
requires v2 and saves `animation_rig_xyz`, `observation_xyz`,
`observation_presence`, and `observation_confidence` separately; it never writes a
recognition-shaped `landmarks` tensor. Every synthetic file is explicitly ineligible
for training, validation, and testing. The review builder fails closed on legacy
ambiguous files and treats presence changes/missing nodes as detector evidence rather
than structural failures. This implements an architectural contamination barrier; it
does not make the current isolated-medoid phrase composer human-natural.

The train-only v2 anatomy package is
`artifacts/models/signing_landmark_anatomy_v17_v2/anatomy.npz`, SHA-256
`0098c0e2b2fd8d66edf73344b3e5bc612c4e6d33b9f112f988b6a44a1feb79ee`.
All 100 selected prototypes retain their correct class in both observation and rig
form. Genuine observation presence is 70.44%; only the render rig is 100% complete.
No held-out/test source was accessed. Eighteen focused anatomy, artifact-contract,
transition, and signing-voice tests pass. Novel phrase generation remains disabled;
the next gate is rendering a genuine local full-phrase trajectory through the
separated observation/rig path.

## 2026-09-01 14:25 PST — all nine local phrase families now have v17 motion-only trajectories

The user authorized using all genuine downloaded sources for experimentation while
phrase generation remains stopped. `scripts/prepare_local_phrase_motion_manifest_v17.py`
now pins all 780 local videos as motion-only continuous rows. It preserves the existing
20-recording capture batches and every-fifth-repetition familiar-source validation
contract, yielding 624 train and 156 validation recordings across all nine phrase
families. The folder phrase is provenance only; target sequences are empty, so
`FOOD`, `LATE`, `TEACHER`, and `MEET` are not added to Stage 1 and no `FOOD -> EAT`
mapping is implied.

The standard v17 continuous extractor completed all 780 videos in 506.9 seconds with
zero failures and schema fingerprint `b872fa3dcc16aab5`. The new tree is
`data/local/local_phrase_motion_landmarks_v17`; its manifest is
`active/v17/local_phrase_motion_manifest_v17.json` (SHA-256
`a2b5eab75f61cb0dffae7ce7ac89bb56c6ad7150cf85ab4b84184d0a58c9388f`) and its
extraction report is
`artifacts/reports/stage2_v17_real_motion_reference_audit/local_phrase_motion_extraction.json`.
There are 2,226 valid 32-frame trajectories out of 2,349 attempted windows: 1,781
train-side and 445 validation. Every phrase has valid motion on both roles, including
the previously uncached I_WANT_FOOD (144/35 train/validation windows), SORRY_I_LATE
(298/75), and YESTERDAY_TEACHER_MEET (169/39). The validation rows are familiar-source
repetitions because signer identity is unavailable; they are useful engineering gates
but not signer-generalization evidence. Three focused manifest/audit tests pass. No
sealed or public test split was accessed.

## 2026-09-01 14:13 PST — generation stopped; all genuine sources expose anatomy/motion design failure

At the user's direction, phrase generation was stopped before any further training or
rendering. No generation or Stage-2 training process remains active. Both generated
review pilots are now explicitly rejected and quarantined by `REJECTED.md` files. V1
loses or fragments detected bodies/hands; v2's attempted anatomy completion forces all
61 detector-presence values to one while its motion remains visibly nonhuman. Neither
pilot may enter recognition training, validation, or testing.

The new reproducible audit is
`artifacts/reports/stage2_v17_real_motion_reference_audit/README.md`, backed by
`audit.json`, `reconstruction_transfer.json`, and
`scripts/audit_real_motion_reference_v17.py`. All nine local phrase folders were
visually sampled in `local_phrases_contact_sheet.png`. Their 780 raw clips comprise
GOOD_MORNING 60, HELLO_HOW_YOU 140, I_WANT_FOOD 60, MY_NAME 60, PLEASE_HELP_ME 140,
SORRY_I_LATE 140, THANKYOU_FRIEND 60, TOMORROW_SCHOOL_GO 60, and
YESTERDAY_TEACHER_MEET 60. The existing v16-era archives cover all 780 for
reference-only presence analysis; only the six vocabulary-compatible phrases have
current v17 Stage-2 archives (390 train clips / 1,144 valid windows). The v16 arrays
remain prohibited from v17 training.

The compatible real-data experiment used every currently extracted train-side source,
without accessing held-out or sealed splits: 390 local clips, 44 older exact-target
ASLLRP clips, 879 ASLLRP `OTHER`-CTC spans, 155 2M-Flores `dev` clips, 1,026 How2Sign
train clips, 166 NCSLGR clips, and 103 train-side YouTube-ASL channel proxies. The
three retained OpenASL clips were included in the inventory as raw visual/domain
references but could not be scored because no compatible v17 extraction exists.
How2Sign/YouTube English metadata were not promoted to ordered gloss truth.

Detector-presence distributions prove that v2 would poison recognition training. All
61 nodes are present in 100% of v2 frames, versus 54.6% across pooled genuine v17
train-side frames and 0% across local v17 phrase frames. V2 also suppresses genuine
dynamics: its hand speed/acceleration/jerk p95 values are 0.1843/0.2291/0.3523, while
the genuine-source p95 ranges are 0.3269--0.5497 / 0.4652--0.8063 /
0.8080--1.3894. Low jerk here is under-articulation, not evidence of human smoothness.
The local videos visibly contain preparation, overlapping articulation, continuous arm
travel, retraction, and rest that isolated medoids plus a short inpainted gap cannot
represent.

`scripts/evaluate_transition_real_sources_v17.py` then froze the existing
How2Sign+YouTube transition checkpoint and masked a deterministic 4--12 frame interval
in every compatible genuine window. Relative reconstruction improvements over endpoint
interpolation were positive on all sources: local 14.0%, older ASLLRP 29.0%, ASLLRP
`OTHER` 21.7%, 2M-Flores 21.0%, How2Sign 26.0%, NCSLGR 27.7%, and YouTube train 7.7%.
However only 50.1% of local windows improved, the weakest practical transfer result;
the current inpainter is therefore not ready to drive local phrase generation. These
are train-side masked-reconstruction diagnostics, not independent naturalness or
text-to-sign evidence.

The architectural correction is now locked for consultation before more generation:
use all genuine corpora with source-balanced sampling, learn/retrieve full continuous
phrase trajectories rather than joining isolated medoids, keep semantic/gloss
supervision separate from motion-only self-supervision, and split the product into a
persistent kinematic animation rig plus a separate learned detector-observation mask.
An always-present internal rig is appropriate for rendering but must never be written
into recognition features. Two focused audit tests pass. No Citizen, SemLex, local
sealed, RIT, How2Sign validation/test, YouTube internal validation, or 2M-Flores
`devtest` split was accessed.

## 2026-09-01 13:43 PST — native visual review rejects pilot anatomy continuity despite prior structural gate

The user reviewed the ten generated landmark-avatar videos and reported disappearing
bodies, disappearing/fragmentary hands, and a frequent single-hand appearance. The
pilot is therefore rejected for training and remains quarantined as synthetic review
evidence. Its previous machine pass was necessary but insufficient: it checked that at
least one hand node existed on every transition frame and that presence did not change
at the exact splice, but did not require complete 21-node hands, a persistent torso,
or stable handedness/anatomy throughout each gloss and transition.

Targeted tracing localizes the root cause upstream of the avatar renderer. The final
profile package stores one observed train-only medoid clip as the content prototype for
each gloss. `build_prototypes` minimizes XYZ error only over nodes present in a
candidate and has no coverage/completeness penalty, so a sparsely detected clip can be
selected as the medoid. `apply_voice_profile` modifies only XYZ and deliberately
preserves the prototype's presence/visibility channels. `trim_observed_span` trims on
any observed hand node, not on a complete hand/body criterion. The transition
composer's endpoint-anchored presence prevents new pop-in at the splice but necessarily
propagates missing source anatomy. Finally, the abstract renderer draws the torso only
when both shoulder nodes are present and draws each hand bone only when both endpoint
nodes are present; sparse masks therefore become visibly missing bodies and fragmented
wireframe hands.

The pinned prototypes confirm this directly. `HELLO` has no complete left-hand frame,
only 26/32 complete right-hand frames, and 0/32 complete-body frames; `I` has no
complete right-hand or complete-body frame; `NEED` has no complete left-hand or
complete-body frame; `READY` has a complete four-node body in only 1/32 frames. Some
active prototype frames contain only 5--10 of the 42 possible hand nodes. Across the
rendered pilot, the fraction of frames with a complete body is only 11--45% for nine
of ten phrases (the ASLLRP-heavy `GOOD NIGHT FAMILY` is 87%). `I NEED WATER` contains
both hands concurrently in 0% of its rendered frames because all three chosen medoids
carry only one observed/active hand. One-handed lexical signs may legitimately use one
active hand, but a human avatar should still retain the inactive hand and body rather
than treating nondetection as anatomical absence.

No generator fix or rerender was made during this diagnosis. The correct next fix is
to rebuild content prototypes with coverage-aware source selection and a persistent
avatar anatomy contract, distinguishing an unobserved/inactive landmark from an absent
body part. The revised native-review gate must measure full-hand and body continuity
over every frame, not merely any-hand presence at transition boundaries. No test split
was accessed.

## 2026-09-01 13:28 PST — first 10 generated phrase videos ready for native review

The first ten rows of `stage2_generated_phrase_review_plan_v17.json` were rendered as
30 complete synthetic landmark trajectories: ten phrase videos, each showing the same
sequence in the three novel profile voices Aster, Cobalt, and Juniper. The review
directory is `artifacts/reports/stage2_v17_generated_phrase_review_pilot_v1/`; its
`index.html` is a playable local gallery, `README.md` links every MP4, `review.csv`
contains one blank native-review row per phrase/voice, `contact_sheet.png` provides a
visual overview, and `manifest.json` pins all generation reports, raw landmarks,
videos, previews, source-voice mixtures, transition spans, hashes, and diagnostics.
Every output is labeled `synthetic_native_review_only`, validation/test-ineligible,
and human-review-required.

The first render audit found a genuine composition defect: isolated prototypes retained
their extractor padding, so several generated transitions began and ended on empty
frames, and the learned auxiliary presence output could make hands or body landmarks
appear only inside a generated interval. Those renders were overwritten and were not
admitted to the review bundle. `signing_voice_phrase_v17.py` now trims every isolated
prototype to its observed hand-motion span before duration resampling. Generated XYZ
coarticulation still comes from the frozen How2Sign/web transition inpainter, while
presence and visibility are anchored by interpolation between the two observed
endpoints. This prevents a landmark from flashing into or out of existence solely
inside the synthesized span without replacing learned spatial motion with a direct
join.

The corrected 30 voice/phrase trajectories all pass the bounded machine gates: every
isolated content prediction matches its requested gloss; every learned transition has
at least one observed hand on every frame; no landmark presence changes at either join;
and no node appears only inside a transition when absent at both endpoints. The ten
H.264 MP4 files independently decode at 1920x900 and 30 fps. Thirteen signing-voice
tests and the new review-auditor test pass. The profile and transition packages also
cold-reload and retain their pinned hashes. These structural results do not prove ASL
naturalness, grammatical acceptability, or absence of perceptual jerk; native review
is now the required gate before any generated item may be copied into training. No
Citizen, SemLex, local sealed, RIT, 2M-Flores `devtest`, How2Sign validation, or
How2Sign test split was accessed.

## 2026-09-01 12:48 PST — online overlap survey completed before further generation work

At the user's request, phrase generation and additional Stage-2 training are paused
while public datasets overlapping the locked Citizen-100 vocabulary are ranked. The
full evidence table is
`artifacts/reports/stage2_v17_online_overlap_survey/ONLINE_DATASET_OVERLAP.md`.

The best immediately usable ordered-gloss source remains the already acquired
2M-Flores-ASL `dev` material: 95/100 normalized lexical labels across 811/999
target-bearing sentences, with only `GOODBYE`, `PLEASE`, `SAD`, `SORRY`, and
`TOMORROW` absent. Its signer field is not adequate for a new signer-disjoint claim
(997 rows use local ID 0 and two use ID 1), and lexical strings are not proof of the
pinned ASL-LEX variants. The best exact-variant source remains ASLLRP continuous:
53/100 pinned variants and 1,483 target tokens in the 1,104 spans already acquired.
The already acquired NCSLGR static subset contributes only 17/100 lexical labels and
198 target tokens from two native signers.

DSP Sentences is the strongest newly quantified lead if permission can be secured. Its
official metadata contains 3,172 continuous sign tokens from 15 signers. Exact joining
through the pinned ASL-LEX `SignBankAnnotationID` finds 50/100 variants, 281 target
tokens, 218 target-bearing utterance files, and at least one target token from every
one of the 15 signer codes. Current BU documentation explicitly excludes
DawnSignPress data from downloadable video, so this is a permission/contact candidate,
not an immediately usable video corpus. RIT sentence metadata shows 30/100 exact
variants and 236 target tokens, but RIT remains permanently excluded from development
because its external evaluation was already consumed. This survey read only the
already retained RIT CSV to report corpus-level overlap; it did not access RIT video,
features, predictions, or metrics.

Apple and Gallaudet's April 2026 paper is the highest-upside future source. It reports
nearly 500 manually glossed ASL STEM Wiki videos, 8,655 sign annotations, 16 signers,
411 unique glosses in the ASL Citizen dictionary, and over 300 hours of
pseudo-annotations. The promised annotation data were not found on the Apple page, in
the arXiv source bundle, or in the current Microsoft ASL STEM Wiki repository, so
locked-100 overlap cannot yet be computed. The paper's own review distinguishes
native/native-like signers from multiple L2 signers; future ingestion must preserve
that split rather than pool all 16. How2Sign still exposes video/keypoints plus English
translations rather than downloadable ordered gloss targets and remains suitable only
for transition/self-supervised objectives under the current contract.

For isolated data, online lexical coverage is high but does not solve Stage 2: WLASL
has 99/100 lexical strings (`HE` absent), MS-ASL has 95/100 exact lexical strings or
96/100 with `BYE` treated only as a candidate alias, Sem-Lex official train metadata
maps 98/100 ASL-LEX-linked classes, and the bounded ASLLVD selection already covers
52/100 exact variants. These sources can reinforce Stage 1 or signer style, but they
contain no genuine phrase transitions. No project test split or test video was
accessed, and no new corpus video was downloaded.

## 2026-08-22 16:51 PST — generated-style verification corrected before promotion

Fold 1 was interrupted after epoch 7 because a deeper audit found that the prior style
AUC compared two **real** same-signer clips.  That proved the encoder could identify
signer manner but did not prove the generator emitted it.  Consequently, the 0.8184
style-AUC statement in the 16:46 entry below is invalidated, and its fold-0 checkpoint
is not promotion-eligible despite its still-valid reconstruction and content results.

The corrected metric now embeds the fully generated gloss, compares it with a real
same-signer target gloss, and ranks those positives against exhaustive generated-to-
real cross-signer pairs.  It separately records generated-to-conditioning-style
cosine.  Eight focused signing-voice tests pass.  All three folds will be rerun from
scratch under this stricter output-style metric before any final artifact or video is
promoted.  No sealed or project validation/test split was accessed.

## 2026-08-22 15:54 PST — transition synthesis rendered for direct inspection

A reproducible six-panel visual demonstration now shows the original human RGB,
genuine extracted landmarks, linear interpolation, learned deterministic transition,
and stochastic temperatures 0.10/0.20.  It contains three high-motion examples from
How2Sign signer 3, How2Sign signer 5, and one public channel voice proxy.  The timing
model exactly recovers the deliberately hidden spans in all three demonstrations
(8/8, 8/8, and 11/11 frames); this is a visualization sanity check using the final
train-all artifacts, not new independent accuracy evidence.

The 16.1-second 1920x1080 H.264/yuv420p MP4 is
`artifacts/reports/transition_multivoice_visual_demo_v17.mp4`, SHA-256
`5ac68b3e652ed2660efcee30a4c3130dbc877dda95991baefcc4d9f6e9e47617`.
All 483 frames decode successfully.  The preview PNG is
`artifacts/reports/transition_multivoice_visual_demo_v17_preview.png`, SHA-256
`1e3a4a77ce55a6491037c07d92dbd679a38cbd5d470935f1c3a6192e9754757f`,
and the provenance report is
`artifacts/reports/transition_multivoice_visual_demo_v17.json`, SHA-256
`74ef0f637c20703f265bebe04ee3ac1c83f09aeabb10f53603acaa07ce6730c9`.
The yellow interval in the video is the only region synthesized; visible context is
identical.  This is an abstract landmark rendering, not synthesized RGB or a human-
naturalness rating.  No sealed split was accessed.

## 2026-08-10 14:52 PST — Local A-Z fingerspelling audit passed as a separate track

The planned Citizen-train plus balanced-SemLex first run contains 2,533 training clips
(1,475 Citizen and 1,058 SemLex) for the fixed 100 lexical classes. This is enough for
a meaningful controlled augmentation experiment, but not evidence of production
robustness. None of the fixed 100 labels are alphabet classes: canonical `I` is pinned
to the lexical ASL-LEX entry `ME`, not the fingerspelled letter I. Alphabet clips must
remain a separate 26-class fingerspelling model/head; J and Z retain their motion.

The local `data/raw_videos/ASL VIDEOS/{A..Z}` corpus was visually and mechanically
audited before considering another web download. It contains genuine A-Z fingerspelling
from multiple visible people/environments but also repeated sessions, scraped-source
copies, exact `__from_MARIAH_` duplicates, and weaker clips. The conservative selector
`scripts/audit_local_alphabet_candidates.py` inspected 6,307 MP4 candidates, excludes
known duplicate/scraped/unknown sources and quality/duration failures, caps named
single-session contributions, and retained exactly 312 clips (12 per letter). Selection
provenance is `data/local/local_alphabet_quality_audit/candidate_selection.json`; raw
materialization uses symlinks and all candidates remain `training_eligible:false`.

Apple Vision v17 extracted 312/312 shortlisted alphabet clips with zero no-hand and
zero failed cases. `audit_v17.py` passed all 312 archives with zero errors. Median
observed-hand-frame coverage is 100% and median hand-node presence is 50%; the videos
are uniformly 640x480 and are often tight hand crops, so median face presence is 87.5%
while body and shoulder coverage are 0%, using the extractor's wrist normalization
fallback. The local clips are technically good enough for train-only fingerspelling
experiments, so no web alphabet dataset was downloaded. They are not a credible
validation/test set because signer identities and independence are not established;
future alphabet accuracy claims require a signer-disjoint labeled evaluation source.

## 2026-08-09 19:03 PST — fingerspelling guard and raw corpus audit

The first metadata rank selected raw gloss `W.H.A.T` because it had slightly more
coverage than the lexical sign. This was caught before extraction reached that class.
The 30 downloaded fingerspelling clips were moved intact—not deleted—to
`data/local/citizen100_v17/quarantine/w_h_a_t/`. The manifest builder now rejects dotted
fingerspelling whenever an eligible lexical sign exists. WHAT is pinned to raw gloss
`WHAT1`, ASL-LEX `D_02_094`.

The corrected manifest contains 3,102 videos: 1,476 train, 378 validation, and 1,248
test. Selective acquisition completed with 3,102/3,102 official ZIP size/CRC checks and
SHA-256 provenance. Three apparent finalization failures during an overlapping resumed
transfer were traced to shared `.part` filenames; all final files were valid, and temp
names now include the worker thread identity. A clean resume verified 3,102/3,102 with
zero failures.

`active/v17/audit_citizen100_raw.py` reports PASS: all 3,102 videos decode, there are 100
classes, the selected corpus contains 32 train / 5 validation / 11 test participants,
and participant overlap between every split pair is empty. All inputs are landscape:
2,982 at 640x480 and 120 at 960x540. Report:
`artifacts/reports/CITIZEN100_RAW_AUDIT.md`.

Full v17 extraction is now active for train, validation, and test. The train run resumed
from 493 schema-valid archives; no extraction/no-hand failures had appeared at this
timestamp.
