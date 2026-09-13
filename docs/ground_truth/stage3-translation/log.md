# stage3-translation — log

Measured results, rejected approaches, and progress snapshots. Not read start-to-end —
`rg` this file before re-running an experiment to see if it already failed.

3 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

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
