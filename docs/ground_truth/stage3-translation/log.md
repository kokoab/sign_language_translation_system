# stage3-translation — log

Measured results, rejected approaches, and progress snapshots. Not read start-to-end —
`rg` this file before re-running an experiment to see if it already failed.

12 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## 2026-09-29 — multi-sentence tiny model + incremental translation installed on iPhone 13

User approved incremental integration with run 2. Swift: counted-output parsing,
`LiveIncrementalTranslator` (port of Python `incremental(verify=True)`), coalesced language-queue
drains, tail-only Finish render with word-count fallback, warm-up after load, per-render and per-step
autorelease pools (a native 323-session replay exhausted IOSurface memory without them), new
`stage3_lock` history events. Bundled Stage 3 packages/token table replaced (hashes verified); exporter
now writes `output` into the token table. Native macOS Swift replay: whole-buffer, incremental text and
lock/tail counts 323/323 equal to Python. iPhone 13: 10/10 RunnerTests pass incl. new
`testStage3MultiSentenceModel`; Release rebuilt, codesigned, installed; sessions intact. Python Stage 3
suite 67/68 (pre-existing mobile-naturalizer hash failure). Desktop still v2. Not verified with live
camera signing or live latency. Rollback copies:
`artifacts/reports/stage3_multisentence_bakeoff_v17_20260929/app_backup_before/`.

## 2026-09-29 — new corpus + retrained tiny T5 passes the multi-sentence bar; not deployed

User asked for smallest-first (tiny, then flan-t5-small, then base only if needed), each run ≤1 h,
bar ≥60% judged fully right, <10% wrong, NO never dropped. Corpus: 54,061 rows from 17,560 distinct
DeepSeek sentences, held-out sentences/sessions removed. **Spend $2.37, over the $2 cap**: the rerun's
guard counted only its own spend; now a persistent ledger. Trainer
`scripts/train_stage3_multisentence_v17.py` (wall-clock budget, re-measured speed, LR decay).
Tiny run 1 hit the wall-clock stop at 68% of plan; run 2 (pre-designated candidate) completed its
schedule: 4,539 steps, 16.4 min. Held-out (300): run 2 whole buffer 60% right / 6% wrong / NO 0/42
(v2: 6% / 31% / 21/42); incremental with verified locking 60% / 8% / 0/42; count-only locking
58% / 8% / 1/42 (counts right in only 202/323). Run 1 64% / 7%, within noise. 4-sentence sessions
44% / 8%. No larger model trained. Core ML export: 0/1,100 tokenization mismatches, 200/200 greedy
parity. Estimated idle-phone Finish p90: 233 ms incremental, 284 ms whole buffer (max 437).
Model drops a high-score gloss in noisy probes (MY NAME FS0 TAKE → "My name is FS0."). Nothing
installed; app unchanged. Report: `artifacts/reports/stage3_multisentence_bakeoff_v17_20260929/REPORT.md`.
Next: user decides on app integration (incremental locking in Swift, counted-output contract, warm-up).

## 2026-09-29 — step 3 paused by user (corpus build stopped mid-run)

User capped corpus spend at $2. `scripts/build_stage3_multisentence_corpus_v17.py` (sentence-level
admission, eval sentences/sessions held out, "N: sentence" counted targets, 8% low-score spurious
gloss rows, 5K previous-corpus rows) passed a 20-call trial, then the full run was stopped on user
request before writing a corpus; completed API calls are cached in
`data/local/stage3_multisentence_eval_v17/api_cache.jsonl`, so a rerun resumes at no cost. The
`corpus.jsonl` present is the 663-row trial only. `scripts/train_stage3_flan_t5_base_v17.py` is written
(3 length buckets, optional bf16 autocast); smoke tests on the trial corpus: 0.74 steps/s fp32 vs
0.66 bf16 on MPS, so fp32 stays. No trained weights exist yet. Finding: real flan-t5-base encoder
residual stream peaks at 332,589 (FFN out 120,950) on eval inputs, beyond FP16; HF hides this by
clamping. Core ML FP16 export needs the exact 1/32 encoder residual rescale (RMSNorm is scale-
invariant); re-measure after fine-tuning. Resume: rerun the corpus build with the same arguments,
then train, score on the held-out set, then FP16/Core ML parity.

## 2026-09-29 — multi-sentence held-out evaluation and iPhone 13 candidate latency

User direction: Stage 3 must handle multiple sentences per Finish, answer in under one second,
and be evaluated with DeepSeek via OpenRouter (no human reviewer). Steps 1–2 only; no training,
no renderer/app behaviour change.

Step 1: `scripts/eval_stage3_multisentence_v17.py` built 300 English-first sessions (2–4
sentences, locked 100 + FS0), rule-checked both directions with the corpus lemma table,
DeepSeek-vetted for grammar only, and removed any session in either composition training corpus
(2,385 parsed → 610 rule → 582 vetted → 300 sampled; 237 have no sentence seen in training), plus
23 phone Finish inputs. About $0.17 total. The v2 renderer on the whole buffer:
31% faithful, drops ≥1 sign in 68%, DeepSeek judge 6% fully right / 31% wrong
(mean 0.74 vs 0.77 for word-by-word literal; reference control 2.00). Four-sentence sessions:
0% right / 47% wrong. Oracle sentence boundaries only reach 10% right, so composition fails
inside sentences too. NO is dropped in 21/42 sessions. Phone slice looks good (83%) but half
of it is training data. References and judge are both DeepSeek; not ASL accuracy.
Report: `artifacts/reports/stage3_multisentence_eval_v17_20260929/REPORT.md`.

Step 2: `scripts/bench_stage3_latency_export_v17.py` exported random-weight T5 (tiny, t5-small,
flan-t5-small/base; no-cache and stateful KV; FP32/FP16/int8) and decoder-only LMs (SmolLM2
135M/360M, Gemma-3-270M, Qwen2.5-0.5B; FP16/int4) with verified KV-cache equivalence;
`RunnerTests.testStage3CandidateLatency` timed them on the physical iPhone 13. Idle p90-session
(28 tokens) estimates: tiny KV 173 ms, deployed v2 241, flan-t5-small KV FP16 249, flan-t5-base
KV FP16 391, Qwen int4 502, SmolLM2-360M 925, Gemma-3-270M ≥1210; Qwen FP16 killed the app on
load; SmolLM2 fails Neural Engine compile. Int8 T5 is slower than FP16. Per-sentence (11 tokens):
flan-t5-base 159, Qwen int4 232. Live v2 Finish logs are 1–3× idle, so incremental translation
is needed for the larger models. Phone: bench models removed, Release rebuilt/codesigned/reinstalled,
10 saved sessions intact. Report: `artifacts/reports/stage3_latency_bench_v17_20260929/REPORT.md`.
Next: user chooses model/translation scheme before step 3 corpus generation and training.

## 2026-09-29 — first composition weights fix both user phrases; request-role continuation

Fixed five-epoch training completed in1326.80s; final loss.03108. Both excluded real user
inputs now generate their intended English directly: HELLO MY FRIEND HOW YOU ->
“Hello, my friend. How are you?”; HELLO GOOD MORNING HOW YOU FRIEND ->
“Hello, good morning. How are you, friend?”. GOOD DAY and doctor/tomorrow/morning also
improve.44direct-neural probes saved; no held-out accuracy claim. Historical noisy buffers
still lose content or confuse roles. Diagnostic PLEASE GIVE MY CHILD WATER gives
“Please give. My child is a water.” (old also failed), exposing missing recipient/object
composition training. No deployment yet. A bounded fixed3epoch continuation adds4194
recipient/time/message/address training rows plus4000balanced prior TRAIN replay, lr1e-4.
Old reserved sequence overlap0; both user phrase training overlaps0. Generated rows remain
training-only; no synthetic validation/test selection. This extends model training, not
runtime rules. Original and first-candidate weights preserved. First-candidate CoreML
export passes200/200greedy outputs and33,295tokenization rows with0mismatches; native
Swift matches44/44direct neural probes, including whole-utterance delivery across long
pauses and over-context preservation. Continuation detached worker79496 launched.
Prepared one physical-device Stage3 test (greetings, requests, time, name, negation) and
added checkpoint identification to future saved phone history. Mobile edits are scoped;
concurrent user's UI changes retained. Next: completion probes,
CoreML/Swift parity, then install only the reviewed replacement. Continuation artifacts:
artifacts/reports/stage3_composition_v17_20260929/request_continuation/.
Continuation completed96.12s training (112.22s with probes), final loss.01485. Both real
user phrases still correct; PLEASE GIVE MY CHILD WATER now “Please give my child water.”
HELLO FRIEND I HELP now preserves direct address. Simple previously correct phone cases
retain their meanings. Remaining malformed/noisy inputs still fail: I HELLO... identifies
“I am Hello”; USE FEEL... emits malformed English; MY NAME FS0 TAKE misassigns roles.
Two-slot MY NAME FS0 FS1 still needs the existing slot-preserving literal fallback.
No claim of general conversational accuracy. Final v2 CoreML export pending; no installation yet.
First v2 export invocation failed before conversion: relative CLI checkpoint path was
passed to relative_to(absolute ROOT). Exporter now resolves input/output paths at parse
time; retry uses the same unchanged weights. Failure log preserved as export.log.
Final v2 export:8194tokenization rows0mismatch,200/200CoreML/PyTorch greedy parity.
Native Swift44/44parity, neural-only mode/full-utterance/context-preservation all pass.
54Python tests pass after switching Torch/CoreML defaults and current segmental Stage3
paths to v2; five actual desktop rendering smoke cases pass, including slot restoration.
Prior mobile model files backed up and new package copies hash-verified. Signed Release
build and codesign verification pass. Initial device test failed on disconnected iPhone;
user reconnected/unlocked it, and the physical test is now running. No final installation
claim yet at that point. Physical iPhone13 retry subsequently passed the Stage3 model test
in7.162s: both user phrases, request/time/greeting/name/negation and direct neural mode.
Final non-testable signed Release rebuild and codesign verification pass; devicectl
installation and launch succeed on iPhone13 (bundle com.kokoab.sltMobileApp). Existing app data
retained; original weights/resources preserved. Final report cross-checks all counts;
scoped git diff --check passes. Artifact index refreshed. Next safe action: user live-camera
check of both greetings and requests; use the recorded checkpoint ID in subsequent history
reviews. Remaining noisy-input failures documented, no general translation-accuracy claim. Full result/limitations: artifacts/reports/stage3_composition_v17_20260929/REVIEW.md.

## 2026-09-29 — model-only composition repair authorized and prepared

User explicitly requests fixing the model without triggers. New text-only recipe retains
16,070 existing training rows plus17,225 generated compositions; no new phrase overrides.
Old non-train sequence overlap0; both user phrase training overlaps0. Four focused tests
pass for exclusion, role/time targets, determinism and overlap rejection. Generated data
is training-only; fixed five-epoch final checkpoint, no synthetic validation selection.
Stage1/2 and visual phrase archives untouched; approved phrase verifier passes494archives
and retains training_ready=false. Existing generic trainer would reopen the old test split,
so a new bounded text-only trainer avoids that route. Outputs/versioned hashes and plan:
artifacts/reports/stage3_composition_v17_20260929/. Changes:
scripts/repair_stage3_composition_v17.py, test/test_stage3_composition_v17.py.
Detached MPS worker71474 launched with completion notification; no progress polling.
Prepared model-contract flag to disable existing templates in Python/Torch, CoreML and
Swift. Export accepts explicit training-only numerical-parity rows and fails on token/output
mismatch.14focused tests pass, including both backends invoking the model for formerly
reviewed phrases. Existing checkpoints retain their behavior until replacement is admitted.
Integration refinement: new contract also delegates sentence boundaries to the model by
passing the full Finish buffer. Added context-capacity rejection into existing literal
fallback; no phrase triggers or semantic output rewrites. Python and Swift paths honor
this contract; legacy checkpoints retain pause splitting.16focused tests pass.
Swift Core/Models/Decoder/Stage3 type-check and optimized native parity harness build pass.
The harness requires direct neural mode, full-utterance delivery despite long pauses,
PyTorch/CoreML exact output parity, and preservation of over-context literal input.
Broader54test live-renderer/corpus/composition suite passes; artifact index refreshed.
Results are infrastructure checks only while weights are still pending.
Next: completion-driven direct-neural history/user probes, then CoreML parity before
integration. Old models preserved; no deployment yet.

## 2026-09-29 — complete saved mobile Live history; exact greeting fails after correct recognition

Copied all12current iPhone Live session JSON files (September28–29 PHT):26translations,
286word events/292saved glosses,1094previews,72resets;7incomplete autosaves. Inspected all
saved sentence/Finish pairs; exact flattened-clause/input and saved-sentence consistency
assertions pass. Report includes every translation, compact non-preview event transcript,
source hashes and per-session counts. No recognition accuracy inferred without video.
Latest115725second translation: HELLO MY FRIEND HOW YOU (scores.8686–.9833), oneclause,
gaps.267–.367s -> “Hello, how are my friend?” in175.01ms. Short HELLO MY FRIEND is correct.
User also requested HELLO GOOD MORNING HOW YOU FRIEND: macOS CoreML with current mobile
source packages/manifest and hi-bucket scores returns “Hello, how are you a friend?”
Six text-only probes completed; historical greeting error reproduced exactly. Short HELLO
GOOD MORNING works; HOW YOU FRIEND drops FRIEND. Hashes/results in model_probes.json.
These are development probes, not physical-iPhone/signing accuracy. Initial duplicate
model-loading process terminated after restarting with optional heavy imports disabled.
This isolates Stage3, not the pause splitter or missing recognized words. Other history
adds HE/FRIEND, drops DAY, and turns MORNING into a destination; pause fragmentation and
spelling errors also persist separately. Current Swift source still accepts nonempty
<=300character generation; no semantic acceptance check. No app/model change or deployment.
Changed reports:artifacts/reports/phone_all_history_review_20260929/; current-state summary
updated. Next: use these as development regressions; validate meaning-preserving fallback
and phrase-aware segmentation on additional reviewed compositions, not one greeting alone.

## 2026-09-29 — latest iPhone history confirms translation hallucination and bad clause splits

Reviewed phone session20260929_004215 twice; latest autosave448.57s, complete=false,
6translations/7two-hand Finish events/26resets. HELLO HOW / MORNING / GOOD DAY becomes
“Hello, how is he? In the morning, Friend. Good.” Unsupported HE/FRIEND and missing DAY
are downstream translation errors. HUNGRY MY NAME fs-GELO becomes “hungry is Gelo.”
HOW YOU splits at2.00s; HELP detaches at1.533s; HUNGRY MY stays joined at1.467s. Current
Swift splitter uses an unconditional1.5s cutoff. LiveStage3 accepts any nonempty <=300char
model output; slot validation protects occurrence, not the surrounding meaning. Spelling
also varies(GEC/GELO/RGELCO); the log cannot recover physical ground truth without video.
Report/data:artifacts/reports/phone_translation_review_20260929/. No source/model edits
or new inference replay. Next: exact-history translation regression cases, output content
validation/fallback, and phrase-aware segmentation; simply raising the gap is insufficient.

## 2026-09-22 — ASL-order renderer promoted to live default; two live-only defects fixed

User instructed "make the new model the default so i dont have to opt in".
`DEFAULT_STAGE3_TINY` in `scripts/live_isolated_v17.py` now points at
`artifacts/models/stage3_v17_asl_order_v1`; the legacy checkpoint is kept as
`LEGACY_STAGE3_TINY`. Rather than flip a flag default, the input format is declared by
`stage3_input_contract.json` written beside the weights and resolved by
`stage3_encoding_for()`, because `scripts/live_continuous_v17.py` builds the naturalizer
without the flag and would otherwise have fed the new model plain gloss text. Rollback is
`--stage3-checkpoint <legacy path>`, which auto-resolves to plain.

Running it as default immediately exposed two defects that 1,261 held-out rows had not.

First, a confidently recognized gloss can still need dropping. Session
`20260922_122517_239185` produced `I SICK MY HUNGRY` at confidences 0.97/0.89/0.91/0.52
and rendered "I am sick, and my family is hungry." MY scored 0.91, so no confidence
threshold could have caught it. The gloss was grammatically stranded — a possessive with
no noun — and the model supplied the commonest possessed noun. This falsified the design
assumption that omission is always evidence-driven. `Utterance` gained
`noise_is_confident`; stranded rows draw confidence from the GENUINE band (median 0.65) so
the renderer cannot learn "drop iff low score". Restricted to possessive-before-adjective:
"less time" and "more water" are ordinary English and treating those as artifacts would
teach deletion of real modifiers. `MY MOTHER SICK` is unchanged. Two bugs in the first
attempt were caught before training: `inject_noise` layered a second artifact over
stranded rows and discarded the first, and determiner-before-noun was wrongly marked.

Second, verbless wh-questions were absent entirely: `WHO YOU` became "Who do you see?"
The locked vocabulary has no copula and the real sessions are full of wh plus a noun
phrase — WHO YOU, WHERE YOUR FAMILY, WHAT TIME TOMORROW, HOW YOUR DAY — so the model
invented a verb instead of supplying "is". Added `build_wh_copula`/`build_wh_time`.

Corpus 13,029 of 13,096 admitted (1,263 stranded, 188 wh-copula); all 67 rejections were
invented content. Retrained fresh from base, epoch 7 of 8 on validation BLEU 93.32.
Test opened once: BLEU 92.74 vs deployed 43.58, chrF2++ 96.60 vs 69.91, exact 0.905 vs
0.260, noise suppressed 0.992 vs 0.589, negation 1.000 vs 0.990. Genuine glosses dropped
0.21% vs 3.09%; noise kept 1.21% vs 51.21%. Words invented from no gloss at all: 1/1,261
vs 57/1,261 — the deployed model's include a spurious "not" that inverts meaning.
Slices: reordering 89.43 vs 7.53, noisy 92.95 vs 36.63, time-fronted 82.87 vs 52.02,
negation 83.42 vs 68.05, SVO regression guard 96.42 vs 55.38.
Both live failures verified fixed: `I SICK MY HUNGRY` -> "I am sick and hungry.",
`WHO YOU` -> "Who are you?", with `MY MOTHER SICK` unchanged. 19 of 20 structure probes
and 9 of 11 noise probes correct; `WHAT WORK TIME` drops TIME and two long garbled
multi-clause buffers still fail.

Deliberately unchanged: `active/v17/stage3_mobile_naturalizer_manifest_v17.json` still
states "never delete, replace, reorder, or invent a recognized gloss". It governs the
separate mobile bounded renderer (Swift, audit and orientation-benchmark scripts), which
did not change, so editing it would misdescribe that path. The desktop live default now
diverges from that contract; this is recorded rather than papered over.

Still not validated by a fluent signer. BLEU measures agreement with model-written English
on rule-generated sequences. 109 focused tests pass, `git diff --check` clean. Total API
spend across all corpus builds $0.31.

Next: user judges live signing on the default; if it holds, consider whether the 36
reviewed templates still earn their override, and whether a fluent reviewer should validate
a sample of the generated English.

## 2026-09-22 — ASL-order evidence-conditioned Stage 3 trained; large gain, no promotion

User reported that live translations are wrong in SVO and that noise glosses are
rendered, citing `I TIRED`. Root cause measured, not assumed: the deployed
`stage3_v17_t5_efficient_tiny_locked100_v1` is a monotone function-word inserter. Of
11,840 rows of its training CSV whose glosses can all be located in the target, 11,745
preserve gloss order and only 95 reorder; just 742 of 15,843 rows are fully inside the
locked 100 and those are template `TIME SUBJ VERB OBJ` already in English order. Its
contract also forbade omission. On held-out rows it drops 3.03% of genuine glosses,
keeps 57.50% of noise glosses, and 231/958 outputs fail a gloss-to-word content check
(211 invented content, 20 lost subject — the reported `I TIRED` class). The exact string
"is it tired?" was never reproduced; `I TIRED` yields "I'm tired." and the closest
observed garbling is `I TIRED TIME` -> "I am tired of the time."

User authorized four changes this session: Stage 3 may reorder and drop low-confidence
glosses (breaking the locked "never delete, replace, reorder" rule); training pairs come
from rule-generated ASL syntax with model-written English; Stage 3 receives per-gloss
confidence; the gate is BLEU on a held-out generated split.

New: `active/v17/asl_corpus_v17.py` (ASL grammar generator, noise injection, English
validation guard), `active/v17/stage3_asl_encoding_v17.py` (hi/mid/lo bucket per gloss),
`scripts/build_stage3_asl_corpus_v17.py`, `active/v17/train_stage3_asl_v17.py`,
`scripts/probe_stage3_asl_v17.py`, `test/test_stage3_asl_corpus_v17.py`.
Corpus `data/local/stage3_asl_corpus_v17/`: 12,951 of 13,000 admitted, 10,281/1,310/1,360
split by clean gloss core, zero cross-split sequence overlap; all 49 rejections were
invented content. English from DeepSeek V4 Flash via OpenRouter (key read from
`~/.continue/config.yaml`), batched and cached, $0.107 total.

Confidence bands overlap deliberately. Live accepted predictions reach 0.250 (2,006
accepted, median 0.615, p10 0.315 across 21 sessions), so a disjoint noise/genuine split
would teach "low score means drop" as a perfect rule and delete real content on any
uncertain sign. A sub-0.40 gloss is genuine ~60% of the time in the corpus.

Result, fresh from the grammar-correction base, epoch 6 of 8 selected on validation BLEU
92.13, test opened once: test BLEU 92.26 vs deployed 41.10, chrF2++ 96.34 vs 69.05,
exact 0.893 vs 0.221, noise suppressed 0.992 vs 0.550, negation preserved 1.000 vs 0.986.
Genuine glosses dropped 0.28% vs 3.03%; noise kept 0.97% vs 57.50%; content-check
failures 5/1,367 vs 364/1,367. By slice: reordering 89.89 vs 5.72, noisy 92.48 vs 32.30,
time-fronted 85.92 vs 50.19, negation 84.90 vs 68.75, and the SVO regression guard 99.07
vs 53.98, so previously-working cases did not regress. 26.1ms per sentence on CPU.
This also repairs the lost-subject defect recorded in PROJECT_GROUND_TRUTH.md on
2026-09-16: deployed `I`->"Is that?", `I I`->"Is it?", `I SICK`->"Is I sick?",
`YOU`->"Thank you."; retrained gives "I.", "I am.", "I am sick.", "You." A build without
single-gloss rows answered `I` with "I am happy.", inventing a predicate, so 96
single-gloss rows were appended after the random draw to keep the cache valid. Report
`artifacts/reports/stage3_v17_asl_order_v1/README.md`.

A first 10k corpus scored 90.35 and its probe exposed four generator gaps — GO toward a
person (DOCTOR rewritten to hospital), coordinated states, `YOU`+state carrying both
question and statement targets, and dropped greetings. All four were closed in the
generator and the 13k rebuild fixed all four probes. A 300-row pilot scored BLEU 81.58
while still turning `I NO LIKE WATER` into "I like the water"; a dedicated
negation-preservation metric was added because BLEU hid that.

Live path is opt-in and the default is unchanged: `--stage3-encoding {plain,evidence}`
on `scripts/live_isolated_v17.py`, inherited by the reel and app-shell parsers.
`scripts/live_reel_stage1_v17.py` now keeps `gloss_scores` in lockstep with `glosses`,
cleared at both reset sites and passed only when it aligns with the selected sequence;
Stage-2 CTC and visual-redecode paths pass none, which buckets to `hi`. Deferred
ambiguous-pair glosses carry their own score rather than the following sign's. The plain
path keeps the 48-token window because widening it changes deployed output past ~41
tokens (verified: 0/47 real session buffers differ, but 4/4 synthetic long buffers do).

NOT PROMOTED. BLEU measures agreement with model-written English on rule-generated
sequences; it is not human-judged translation quality, the corpus is not genuine ASL, and
no fluent signer reviewed the targets. Two long multi-clause noisy probes remain weak and
one drops a legitimate MORNING. No recognition model, checkpoint or threshold changed; no
protected split accessed. Pre-existing unrelated failure: `test_stage3_mobile_naturalizer_v17`
errors in setUpClass on a recognizer-hash mismatch between the naturalizer manifest and
the Stage-2 contract; those files were not touched. 82 focused tests pass,
`git diff --check` clean.

Next: user runs the live opt-in and judges real signing; then decide promotion, whether
the reviewed-template override still earns its place, and whether a fluent reviewer should
validate a sample of the generated English before any promotion.

## 2026-09-06 09:56 PST — user context resolved: synthetic voices, continuous input, evidence-aware Stage 3

The user clarified that signing voices serve both recognition training and animated
output. The desired outcome is to eliminate the planned 30-phrase/13-signer collection
if generation succeeds, recognize connected combinations of the locked 100 glosses,
and let the signer continue without waiting for individual locks. Revisable partial
text is explicitly acceptable. Use existing local corpora for real evaluation;
new recording is not a prerequisite for the next experiment. Preserve protected tests,
signer-disjoint evaluation, exact label variants, and synthetic provenance.

Human feedback on the Aster/Cobalt/Juniper and grounded animations identifies
unnatural transitions, apparent hand zooming, and a uniform stomach resting posture.
Fluent reviewers are available for revised short animations, and the user prefers a
realistic avatar and authorizes use of an API or a local implementation. No external
API upload, spending, or asset acquisition occurred in this review.

The user proposes giving Stage 3 landmark evidence and more contextual training so it
can recover recognition errors. Code inspection confirms `TinyStage3Naturalizer` in
`scripts/live_isolated_v17.py` consumes only lowercased chosen gloss text. Its existing
checkpoint has no landmark-conditioning path. A proposed evidence-aware extension
should retain scored sequence alternatives/timing and use an explicitly trained
projection of temporal motion embeddings if multimodal conditioning is added. Simply
passing coordinate numbers to the existing text checkpoint does not confer motion
understanding. Contextual corrections must be evaluated against the recognizer alone,
including harmful changes to previously correct words and unknown signing.

Visual inspection of the grounded contact sheet and generated_01 voice preview
confirms abstract sparse rigs rather than realistic avatars. Both renderers use fixed
coordinate bounds, so framewise camera auto-zoom is not the mechanism in these paths.
A read-only diagnostic on the five grounded NPZ files measured total projected XY
hand-tree bone length across each transition plus one neighboring frame per side,
using only complete observed hand frames. Max/min reaches 2.19 for GOOD MORNING's left
hand, 1.57 for the final HELLO HOW YOU right-hand boundary, and 1.58 for TOMORROW SCHOOL
GO's left hand at the second boundary. This is a scale-change diagnostic, not proof
that every change is wrong: real projection/foreshortening also changes 2D bone lengths.

`stabilize_transition_hands` interpolates endpoint-derived bone lengths/directions;
it does not impose fixed physical avatar anatomy. `complete_landmark_anatomy` retains
the observation mask and currently omits never-observed hands, with no literal stomach
rest constant found. Thus the reported resting posture must be traced to the specific
source/prototype and animation policy, not assumed to be a hardcoded current constant.
Observation presence, lexical participation, and persistent avatar anatomy are distinct.
`geometry_v17.py` stores a hand-scale log proxy for depth, shared across each hand;
this is not per-joint metric 3D suitable for direct physical rig coordinates.

Next implementation priorities are (1) a constrained avatar/motion representation with
fixed physical bone lengths, learned/contextual rest behavior and explicit imputation;
(2) reviewable transition improvements preserving lexical evidence; (3) measured
synthetic-to-real recognition transfer on existing signer-held-out development data;
and (4) continuous revisable decoding plus evidence-aware contextual correction.
MakeHuman core assets are a researched local rig candidate, not an installed renderer.
No generation, training, deployment, or promotion was performed during this context
review. Only this canonical handoff was edited; no further context questions are needed
to start the bounded implementation. Success at replacing recording is still unproven.

## 2026-09-02 09:54 PST — 15.6M T5 is the lightweight naturalizer candidate, but needs in-domain tuning

Three Hugging Face encoder-decoder candidates were pinned, cached outside the repo, and
measured on the Mac using only existing reviewed Stage-3 templates and handcrafted
in-vocabulary combinations. No Citizen validation/test data or new video data was
accessed. `visheratin/t5-efficient-tiny-grammar-correction` at revision
`a98f126664317cf2e68d33234c7072fdd6b289f3` is the clear deployment candidate: 15.58M
parameters, 62.3 MB FP32 weights, and 32.16 ms median greedy inference over the first
14-phrase probe. On all 36 reviewed Stage-3 templates it reproduced 23/36 references
after case/punctuation normalization at 43.73 ms median. It handled `I NEED WATER`,
`I FEEL SICK`, `HELLO HOW YOU`, and `WHERE HOSPITAL`, but also hallucinated content
(`HELLO` -> `Hello everyone!`), changed intent (`GOODBYE` -> `Good night!`), and missed
important ASL order/negation/question cases. It is therefore not safe to replace the
current naturalizer unchanged.

`HamdanXI/t5_small_aslg_pc12` revision
`b35e3323732c7244236189674bdc0728f37b31e8` is directly trained gloss-to-English and
has 60.5M parameters/242 MB weights, but measured 115.78 ms median and failed badly on
unseen combinations (`FATHER SICK` and `MOTHER FEEL GOOD`). Its reported high in-domain
BLEU is not persuasive for this project because ASLG-PC12 was created by rule-transforming
Project Gutenberg English rather than from genuine ASL, and published work explicitly
warns that it is unreliable for SLT. The model card says Apache-2.0 while ASLG-PC12 is
reported as CC BY-NC 4.0, so downstream licensing also needs care.

`jbochi/coedit-small` revision `6ce9822b4ff6e4af86b70f979c890e9e41f04366`
has 77M parameters/approximately 296 MB cached and measured 63.48 ms median after its
first warm call. It fixes ordinary English but frequently leaves gloss order unnatural
or duplicates meaning, so it is a weaker starting point than T5-efficient-tiny for this
bounded task. The three research downloads currently occupy approximately 789 MB in the
user Hugging Face cache; they were not added to git or the repository and have not been
deleted.

Recommended next experiment: retain exact reviewed templates as the instantaneous
fail-closed path, then fine-tune the 15.58M T5-efficient-tiny checkpoint on genuinely
reviewed, locked-100 gloss-to-English pairs and evaluate on held-out phrase combinations.
Do not train on its own generated sentences or claim that generic grammar correction is
ASL translation. A successful checkpoint can be converted to Core ML or quantized ONNX;
the base architecture is about 31 MB FP16, 15 MB INT8, or 7.5 MB INT4 before packaging.
This is substantially smaller and faster than the 1B Ollama model, but accuracy—not raw
latency—is the gate for replacing it.

## 2026-08-24 21:00 PST — bounded Stage 3 and file-video iPhone app are deployable

The locked-100 mobile pipeline is ready for signing and interactive file-video testing
on an iPhone. The selected compact Stage-2 model and strict downstream contract remain
unchanged. The iOS app now reads arbitrary native aspect ratios/orientations, samples
at most 256 frames, retains only one 32-frame window at a time, runs Apple Vision,
creates real-pixel left/right/union hand crops, executes all three neural components in
Core ML, collapses the CTC sequence, validates the Stage-2 hashes/label mapping, and
renders bounded English. This replaces the previous interactive-path block and keeps
the video preprocessor explicitly memory bounded.

No neural Stage-3 translator was promoted. The fail-closed audit found 1,165 genuine
gloss/English pairs (999 2M-Flores `dev`, 166 NCSLGR), but zero complete sequences are
fully expressible with the locked 100-gloss vocabulary. Training by deleting OOV signs
would corrupt the target meaning. The promoted Stage 3 therefore contains 35 exact,
meaning-conservative templates and a deterministic literal fallback that never
deletes, replaces, reorders, or invents glosses. Both literal and naturalized text,
rendering mode, and fallback status are exposed. Template coverage is 85/97 local
validation rows and 0/12 ASLLRP contiguous rows; all others fall back literally. This
is a bounded naturalizer, not a general/open-domain translator.

The final iPhone 13 simulator run passed all eight expanded-canvas rotations at
0/17/37/73/90/123/180/270 degrees, each with exactly 200 timed inferences. All eight
predicted HELLO and Stage 3 rendered `Hello.`. Its final result is
`artifacts/reports/orientation_v17_simulator_benchmark/latest_result.json`, SHA-256
`2f2c513b2f50b8f3c7a587782fe825ff727f6017738c92ef3c0f4764202f6806`.
Reports explicitly set `videoFileToGlossEndToEnd=false`,
`cameraToGlossEndToEnd=false`, `hardwarePerformanceClaim=false`, and
`thermalsInterpretable=false` because Apple Vision preprocessing occurred on the Mac
host and Core ML executed on simulator hardware.

The separate Swift-source video-to-English gate passed 8/8 HELLO rotations through
Apple Vision, all three Core ML models, CTC, and Stage 3. Maximum process RSS was
279,773,184 bytes, which is Mac-host evidence only. The unsigned generic iPhoneOS
Release build passes and produces a 110,350,170-byte arm64 app containing exactly the
three selected Core ML models. The bundled Stage-3 manifest matches source SHA-256
`68c7ce67632f66ee70fa3b3d36eb8df33ad72dc674edbf3b720e93c1240f84a6`.

One hundred fifty-seven focused tests pass. The machine-readable deployment audit
passes with zero acceptance errors at
`artifacts/reports/stage3_mobile_v17/deployment_audit.json`; its SHA-256 is
`065f42985a5285d143338520709a8211007358b353bd346954a6caf060d5fbf8`.
The complete evidence summary is
`artifacts/reports/stage3_mobile_v17/README.md`. No Citizen, SemLex, local, or
2M-Flores `devtest` split was accessed.

Claim boundary: the app is deployable for videos selected from the iPhone file picker.
A live-camera capture UI is not implemented. Physical-iPhone accuracy, latency,
memory, thermals, and ANE behavior remain unmeasured and must not be inferred from the
simulator or Mac-host results.
