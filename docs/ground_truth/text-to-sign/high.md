# text-to-sign — high

Dated evidence for constraints that still bind. The constraints themselves are
stated in `PROJECT_GROUND_TRUTH.md`; these are the receipts behind them.

6 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## 2026-09-07 11:50 PST — resting hand accepted;100-gloss implementation plan

User answered yes to revised rest close-up/connectedv2. Recorded acceptance scoped
to this pilot. All active core and join frames14:137 match acceptedv1 exactly;
resting wrists unchanged, no new resting-hand motion.27rig/mesh tests and diffcheck
passed. No render jobs pending. Single-letter messages are treated as incidental
steering; the authorized100-gloss expansion and original full objective persist.

Implementation plan (canonical handoff location per project instructions):
1. Add scripts/build_avatar_gloss_bank_v17.py. Select only train rows matching each
   frozen class index/raw gloss/ASL-LEX code, prefer complete hand observations and
   a consistent signer on ties. Retain exact accepted YOU/NEED symbolic cores.
   Decode both hands from existing raw source via current Apple/MP matching helpers,
   fill missing world estimates with shared palm interpolation, use fixed metric rig.
   Cache one audit/item per gloss with source hashes, detection coverage and exact
   dictionary candidates. Source-driven classes remain explicitly nonsymbolic.
2. Add focused test for strict source identity/split filtering and hand participation;
   run red then green. Build small one-/two-handed smoke before all100.
3. Extend existing sequence renderer to consume either pilot or bank report, preserve
   per-hand states, accept all exact covered glosses and save gloss timeline/video.
   Reject unavailable glosses before output creation; no silent alias translation.
4. Generate100bank, audit every class and representative connected sequences, inspect
   source comparisons/contact sheets, record incomplete/poor motions individually.
   All100coverage must be demonstrated rather than inferred from available files.
5. Full SignWriting decoding, human sign quality and synthetic transfer remain separate
   evidence requirements; source-bank coverage alone does not complete the full goal.
No new dependencies, no official Citizen test rerun, no training promotion. Continue
in current authorized shared workspace without disturbing pre-existing changes.

## 2026-09-07 07:34 PST — connected motion accepted; rest pose rejected;100-gloss expansion authorized

User accepted blue YOU NEED YOU: "its already good. complete the connected combination
signing and glosses". This accepts that clip's connected motion, not100-sign coverage
or training transfer. Then user supplied a close-up and rejected the resting hand as
"eerie and non human". Fix shared rest before carrying it into100-gloss generation.
Old fallback had8.43cm index-to-pinky knuckle span and almost straight fingers.
New fallback uses measured directions from existing MakeHuman bind joints, narrower
knuckle spacing, increasing relaxed finger curl, adducted thumb and palms60degrees
inward. These are animation assumptions; learned signer rest modes remain preferred.
New regression failed on old fan width; all21rig+6mesh tests pass after correction.
Sequence renderer refreshes only uncertain rest in cached cores; active poses preserved.
Saved rest_comparison.png under signwriting_avatar_rest_v17_v1. Connectedv2_relaxed_rest
render64220 finished; verify active/transition preservation and full frame review next.
Earlier CLIguard/core checks71366 completed successfully;26tests/diffcheck passed then.

100-gloss inventory: all100 dictionary terms have candidates, but only YOU/NEED symbolic
motion is executable. Existing train provenance has full100 coverage for P37,P52,P33
(117/100/103 archives including repeats); no new recordings are needed to assemble a
source-grounded review bank. Mean dominant wrist presence about0.80–0.81, so missing
frames need explicit handling. Use pinned Citizen raw gloss/ASL-LEX codes and train only;
do not call source-derived98-sign motion a full SignWriting decoder. Preserve approved
YOU/NEED symbolic cores. Remaining scope includes100-gloss bank, general connected
composition, exact-gloss lookup, coverage/geometry/source review, then transfer evidence.
Full goal remains active; no classifier test-set run or synthetic training admission.

## 2026-09-07 07:08 PST — correction: user reviewed our bald blue avatar, not CWASA

User explicitly clarified that YOU/NEED handshape acceptance and missing onset
refer to our own bald blue MakeHuman avatar. The preceding06:59 entry misapplied
that feedback to CWASA. CWASA handshape review remains pending; corrected those
artifact review fields. Own pilot trims source to the hand-active interval, and v3
close-ups further use only frames8:24. Preserve core handshapes and restore a
visible approach in our own renderer. No approval of full signing or training.
Prior CWASA onset work is retained but does not address this clarified feedback.
Previous turn made technical progress on the wrong render target; next work is
our own onset, not further CWASA camera changes. Full objective remains active.

## 2026-09-01 20:58 PST — four whole-utterance code priors rejected before rendering

Four free-running temporal-code priors were evaluated with `GOOD MORNING` and
`TOMORROW SCHOOL GO` excluded from training. The plain prior ranked the intended
phrases 7th/3rd; pooled-text/history-dropout v2 ranked them 3rd/8th; monotonic-gloss
timeline v3 ranked them 4th/6th and incorrectly made `TOMORROW SCHOOL GO` one-handed.
The v4 text-only code/side anchor corrected that hand-participation error and retained
the exactly one-handed `SORRY I LATE` audit, but semantics still ranked 5th/5th and
unseen motion fell to 0.34--0.49x genuine on at least one derivative order. All four
remain rejected and unrendered. The v4 checkpoint is
`artifacts/models/temporal_code_prior_v17_text_anchor_v4/model.pth`, SHA-256
`4f26aaad3bd42beba187bbcac8e26f0d0efb9ad1d202679d806452da1f475a3c`;
its rejection report is
`artifacts/reports/stage2_v17_temporal_text_to_sign_v4/evaluation.json`.

This falsifies whole-utterance autoregression with only weak/equal gloss alignment as
the immediate reverse-generation route. The safer architecture is exact per-gloss
content grounding plus a separately learned, short coarticulation model. The temporal
tokenizer remains accepted as a representation, but none of its rejected priors may
produce training data.

## 2026-09-01 17:55 PST — multi-source global-latent generator rejected; temporal latent needed

The source-balanced full-trajectory CVAE trained on 1,606 non-holdout train rows and
369 validation rows, reserving all 24 validation performances of `GOOD MORNING` and
`TOMORROW SCHOOL GO`. With KL weight 0.01 and prior-loss checkpoint selection, early
stopping retained epoch 3 of 10. The 1,544,946-parameter checkpoint is
`artifacts/models/full_trajectory_generator_v17_multisource_holdout_v1/model.pth`,
SHA-256 `a6a5ee3e1931e0a342cfdea3a0bca47de3dc225075f319bb2ee7de22823ca941`.
Ordinary prior validation presence F1 is 0.919 at its calibrated 0.30 threshold and
hand-participation accuracy is 0.966, but coordinates lose to the source-family mean.
On the compositional holdout, prior presence F1 is 0.802 and hand participation is
0.917; coordinate loss also loses to the family mean.

The strict 32-prior-sample motion gate rejected both unseen combinations. The
`GOOD MORNING` output produced only 0.028x genuine p95 speed, 0.035x acceleration,
and 0.038x jerk;
`TOMORROW SCHOOL GO` produced 0.026x, 0.034x, and 0.036x. This is substantially worse
than the local-only CVAE's already-rejected motion. No review render or training sample
was written. The evaluator result is
`artifacts/reports/stage2_v17_full_trajectory_generation_v1/evaluation.json`.

The diagnosed architecture bottleneck is the single mean-pooled 32-D motion latent,
which is broadcast identically to all 128 frame queries. It can encode a target for
posterior reconstruction but cannot supply time-varying motion to the text-only prior;
coordinate regression therefore averages performances into nearly static poses.
Mixing more genuine sources does not fix that bottleneck. The next justified model is
a temporal/discrete motion representation (motion tokenizer plus conditional temporal
prior, or a temporal-latent diffusion model), trained on all genuine trajectories as
an unlabeled motion prior while keeping gloss-conditioned local content and hand-side
participation explicit. This is a meaningful architecture replacement and awaits user
consultation. Rejected synthetic motion remains barred from every dataset and renderer.
All 18 focused generation/signing-voice tests pass, Python compilation succeeds, and
`git diff --check` is clean.

## 2026-09-01 12:32 PST — Stage 2 data, collection, and generation plan locked

The user approved the following execution order: finish and audit the newly prepared
1,104-span ASLLRP `OTHER`-CTC expansion; train it together with the existing real
phrase replay and train-only synthetic replay on MPS; then use the already acquired
How2Sign train-only material to learn transition motion and timing; then compose new
locked-vocabulary phrase combinations; finally retrain Stage 2 and evaluate only on
genuine signer-disjoint recordings. Long MPS runs are authorized. The generated
review artifacts must be returned as a directory for native-signer inspection.

The vocabulary remains the exact pinned Citizen/ASL-LEX 100-class inventory. A single
explicit `OTHER` class is added only to the Stage-2 CTC head (101 nonblank classes,
blank index 0, `OTHER` CTC index 101), and `OTHER` is removed after CTC collapse. No
new lexical class or numeric/lexical variant is silently merged. The 1,104 natural
ASLLRP spans are combined with the old real local/ASLLRP phrase replay and the selected
train-only multivoice pool. Synthetic phrases are training-only: they must never enter
validation, sealed testing, or model-selection truth.

The bounded genuine collection remains 30 conversational gloss sequences, 13 native
ASL signers, three independent normal-speed performances per signer and phrase, and
three synchronized phone views per performance. Thus the planned capture is 1,170
independent performances and 3,510 videos; the three camera recordings of one
performance are correlated views, not three independent takes. Capture should include
the intended varied front/oblique angles at comparable quality. The preferred
signer-disjoint split is 9 training, 2 validation, and 2 sealed test signers. Every
signer identity and synchronized performance group must stay in exactly one split.
All 13 may record all 30 prompts, but the six composition-test prompts below remain
quarantined from training even when recorded by a training signer.

The approved 30-prompt collection inventory is:

| Role | ID | Exact gloss sequence |
| --- | ---: | --- |
| real base/train | 01 | `HELLO HOW YOU` |
| real base/train | 02 | `GOOD MORNING` |
| real base/train | 03 | `GOOD NIGHT` |
| real base/train | 04 | `THANKYOU FRIEND` |
| real base/train | 05 | `PLEASE HELP I` |
| real base/train | 06 | `I NEED HELP NOW` |
| real base/train | 07 | `YOU NEED HELP PLEASE` |
| real base/train | 08 | `WHAT YOUR NAME` |
| real base/train | 09 | `MY NAME` |
| real base/train | 10 | `I UNDERSTAND YOU` |
| real base/train | 11 | `I NO UNDERSTAND` |
| real base/train | 12 | `I KNOW MY FAMILY` |
| real base/train | 13 | `I NO KNOW WHERE HOME` |
| real base/train | 14 | `WHERE HOSPITAL` |
| real base/train | 15 | `WHERE DOCTOR` |
| real base/train | 16 | `I SICK` |
| real base/train | 17 | `I HUNGRY WANT EAT` |
| real base/train | 18 | `I WANT WATER PLEASE` |
| real base/train | 19 | `I WANT EAT NOW` |
| real base/train | 20 | `YOU READY GO` |
| real base/train | 21 | `WAIT PLEASE` |
| real base/train | 22 | `STOP PLEASE` |
| real base/train | 23 | `I GO HOME NOW` |
| real base/train | 24 | `SEE YOU TOMORROW` |
| held-out recombination | 25 | `HELLO HOW YOU READY` |
| held-out recombination | 26 | `SEE YOU NEED HELP` |
| held-out recombination | 27 | `HELP I WANT WATER` |
| held-out recombination | 28 | `I UNDERSTAND YOU NEED HELP` |
| hard unseen transition | 29 | `I NEED DOCTOR` |
| hard unseen transition | 30 | `I GO HOSPITAL` |

These are exact model gloss prompts, not claims about English word order. Native
signers must use the pinned lexical variants and may flag a prompt as linguistically
unnatural before capture; any approved replacement must remain inside the locked 100
and be recorded in this handoff before data collection. Prompts 01--24 provide the
real transition base. Prompts 25--28 test whole-sequence recombination from learned
parts, while 29--30 deliberately test transitions absent from the base inventory.

How2Sign is not a replacement for this phone collection. The local bounded subset has
1,027 train-only rows from six How2Sign signers and deliberately empty gloss targets,
so it is suitable for self-supervised transition inpainting, duration, rhythm, and
nonmanual/body-context learning but cannot directly supervise the locked-100 CTC
sequence. The already completed train-all transition package combines 4,938 How2Sign
windows with 994 train-only web windows. Its held-out How2Sign reconstruction gain is
20.2012%, its generated-vs-real discriminator remains above chance (61.1289% balanced
accuracy, AUC 0.648197), and its timing model reaches 92.3887% exact duration accuracy
with 0.1721-frame MAE. These are useful machine gates, not proof of human naturalness.

Phrase generation must preserve complete recognizable gloss cores and synthesize only
the missing coarticulation interval, conditioned on genuine left/right context and the
predicted 4--12-frame span. Direct concatenation, linear interpolation, unexplained
hand appearance/disappearance, and out-of-distribution position, velocity,
acceleration, jerk, bone geometry, or presence changes are rejection conditions.
Generated samples must retain the requested Stage-1/Stage-2 content before they may be
used for training. The first expansion target is 20 generated combinations, producing
a 50-phrase training inventory in conjunction with the 30 real prompts; this is a
training inventory only, and support for a generated combination may be claimed only
after it is recognized in a genuine held-out recording. Native review follows the
first training/generation run rather than blocking it.

The ASLLRP expansion's remaining preprocessing step is now complete. The frozen
selected Stage-1 encoder cached all 1,104/1,104 archives on MPS in 52.78 seconds with
zero failures, a 12% process cap, and 121,847,808 peak MPS driver bytes. The independent
frozen-input audit reports 1,104 expected/actual archives, all 4,505 windows valid,
879 train plus 225 signer-held-out validation rows, no unexpected archives, and a
valid 3,978-item/100-class train-only multivoice pool with 18,000 planned sequences.
Reports are `artifacts/reports/stage2_v17_asllrp_other_ctc/frozen_cache.json` and
`artifacts/reports/stage2_v17_asllrp_other_ctc/frozen_audit.json`. No test split was
accessed. The combined Stage-2 run is now unblocked.

The first full combined `OTHER`-CTC adaptation then ran on MPS and is rejected for
promotion. It completed ten epochs plus the epoch-zero baseline in 58.95 seconds under
the 12% cap; the selected result is deliberately epoch zero because no trained epoch
preserved both legacy guards (local at most 7 edits and sparse held-out ASLLRP at most
11 edits). The trained trajectory proves the new spans are learnable: natural-ASLLRP
full WER fell from 107.1848% at epoch zero to 53.3724% at epoch 10 and target-only WER
fell from 223.2394% to 79.5775%. However, sparse held-out-ASLLRP target edits worsened
from 11 to 12--14 through epoch 5 and 13--14 afterward; local edits ranged 7--11. The
hash-pinned epoch-zero safeguard is
`artifacts/models/stage2_v17_asllrp_other_ctc_v1/best_model.pth` (SHA-256
`8113f5b96d8b2e2f17bcc3e491dda66511b678bb50cf54070b7763efe6924a0b`), and the full
history is `artifacts/models/stage2_v17_asllrp_other_ctc_v1/result.json`. This is a
negative single-head result, not evidence against the acquired data; it shows that the
current objective trades legacy phrase competence for natural `OTHER` competence.
No test split or consumed external evaluation was accessed.
