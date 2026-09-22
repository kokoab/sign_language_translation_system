# stage3-translation — log

Measured results, rejected approaches, and progress snapshots. Not read start-to-end —
`rg` this file before re-running an experiment to see if it already failed.

5 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

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
