# Revisable continuous transcription and transition supervision

Completed: revisable transcription runtime, two 12-epoch training experiments, 45 raw-video replays and 72 focused tests. **Neither new checkpoint qualifies for promotion.** The existing default runtime/checkpoint is unchanged.

The strongest new connected WER is 73.94% versus 168.31% baseline, but it deletes 53.87% of reference signs and fails familiar-phrase/isolated retention. The additional gap losses do not improve the annotated-gap proxy beyond the previous adaptation. The runtime behavior is implemented; reliable coarticulation recognition is not claimed fixed.

## Authorized outcome and implementation

The user wants continuous transcription with revisable words, without a hold or permanent lock after each sign. `--revisable-transcript` implements that behavior in the existing continuous Reel entry point. It implies sequence preview, bypasses Stage1 proposals/verification and irreversible prefix confirmation, replaces the displayed gloss sequence when new evidence changes it, and records the revised portion. The HUD labels the live transcript as changeable. Speech/naturalization consumes the finalized phrase, not interim tokens. The original command defaults remain unchanged. While signing, the most recent eight-window context is re-evaluated; older displayed words are carried forward provisionally until Finish can revisit them. They are not irreversible final output.

At Finish, the runtime re-runs the temporal model over every retained accepted visual-feature window. Eight-window chunks overlap by two windows; their uncollapsed logits are averaged at absolute time positions, then CTC is collapsed once. This allows earlier provisional words to change and preserves genuine repeated labels separated by blanks. It is not a new global-context model: the temporal receptive field is still eight windows. Feature storage is temporary disk-backed storage during capture, then materialized for Finish; memory at Finish therefore grows with utterance length. Reset clears retained features and stale asynchronous results. No automatic deduplication or text-only grammar rule is allowed to erase legitimate repeated signs.

This visual finalization cannot reconstruct frames rejected for insufficient observations. For phrases fitting eight windows, rerunning the same model may reproduce its last live hypothesis exactly; another pass is not inherently an accuracy gain. The long-recording replay verifies that earlier windows are revisited.

## Training and data integrity

Reused all 1,647 existing matched-input videos: 1,313 training and 334 development clips. The fixed observer/extractor fingerprint still matches the existing cache. Stage1 visual and image encoders remain frozen; accepted Stage2 primary/context/specialist weights and existing OTHER projection are trained. These remain the existing model architecture, not a new ensemble or transducer.

Supervision comes from original ASLLRP annotation intervals, including annotations outside the known vocabulary. Every unfamiliar annotated sign excludes blank supervision and remains OTHER=101. Blank=0 means no token emission; it is not a claim that all coarticulation is meaningless. Only complete CTC output bins inside annotation gaps, guarded by 0.10 seconds on both sides, receive auxiliary blank loss. Bin-to-time correspondence is approximate because the model has temporal receptive fields; this is not a frame-perfect physical transition classifier.

There are 1,160 annotated ASLLRP clips (923 train, 237 development), with 1,505 training and 104 development blank bins. The 487 local phrase clips lack equivalent boundary annotations and receive no fabricated boundary targets. Prefix CTC targets are generated only when the observed endpoint is outside a sign and the complete annotated sequence agrees with the existing target. Consecutive OTHER annotations collapse consistently; repeated known signs remain distinct; all emitted targets are CTC-feasible. Annotation timestamps use each cropped video's declared frame rate.

Two bounded 12-epoch runs use seed17101, 2,048 source-balanced samples per epoch, the previous optimizer/sampling and original training replay. Both add 0.25 annotated-gap blank cross entropy and 0.25 eligible observed-prefix CTC to the existing gold CTC plus 0.5 teacher-sequence replay loss. The second adds 0.25 known-core cross entropy to oppose excessive silence: 1,299 train and 296 development bins entirely inside guarded known-sign cores without overlapping annotations. Its supervision is in a separate artifact, leaving the first run's input hash unchanged. These are paired loss variants on one seed, not independent replications.

No synthetic still-frame holds were added. Natural transitions remain in complete training sequences. We did not visually review all annotation gaps, establish accurate temporal labels for local recordings, or obtain a verified transcript of the user's recording. Explicit varied holds, true repetitions and independently checked transition-only footage remain limitations; the result must not be described as fully validated coarticulation robustness.

Checkpoint selection retains the previous declared rule: first eligibility on both original retention and matched-input gates, then the sum of the three matched source WERs. All epoch results are retained. Comparison uses the repaired deployed baseline, not the weaker differentiable branch initialization. Original and matched evaluations share underlying development clips and are not independent tests. No protected Citizen, SemLex, RIT or local test clips were used for evaluation/training.

## Measurement

Report final WER with substitutions, deletions and insertions separately; exact phrase rate; incorrect known emissions within conservative annotated gap bins; first nonempty partial output; and how often existing words are revised. A revision event changes or removes a prior word; merely appending a word does not count. Gap-bin emission rate is an annotation proxy, not a complete false-transition rate. Small denominators and exclusions are disclosed.

Cached first-output timing is the endpoint of the available source window and excludes image encoding, scheduling and UI costs. Raw paced video histories separately provide wall-clock timing. Desktop work overlapped training, so latency is observed load-dependent behavior, not a dedicated performance benchmark or iPhone claim. The input window remains about 1.067 seconds; this experiment does not claim speech-like latency.

The previous adaptation already exposed the suppression tradeoff: on OTHER-containing development spans, insertions fell from 323 to 32, but deletions rose from 11 to 138. Thus lower WER alone cannot demonstrate reliable sign coverage. Known-core supervision was added as an explicit comparison in response to this measured failure.

## Sources behind the approach

AWS documents partial transcripts that can change and optional stabilization with an accuracy/latency tradeoff: [streaming partial results](https://docs.aws.amazon.com/transcribe/latest/dg/streaming-partial-results.html). Google defines interim stability separately from finality: [streaming recognition results](https://docs.cloud.google.com/speech-to-text/docs/reference/rest/v2/StreamingRecognitionResult).

The EMNLP 2024 sign-recognition paper trains sliding-window classification with background examples and surrounding-sign augmentation: [Towards Online Continuous Sign Language Recognition and Translation](https://aclanthology.org/2024.emnlp-main.619.pdf). It motivates a controlled experiment; its benchmark performance is not evidence for our ASL model. Speech deliberation uses both original signal and preliminary hypotheses: [Google research](https://research.google/pubs/deliberation-model-based-two-pass-end-to-end-speech-recognition/). We implement visual re-decoding, not that paper's trained deliberation architecture.

## Decoder control and raw baseline

An exploratory final beam search reused the existing v17 CTC prefix decoder (beam 8, top 12 symbols; vetoed negative infinity floored to-10000 for its finite-input requirement). It changed OTHER-domain errors478->477, kept exact errors11, and worsened local errors19->20. It is not enabled in the runtime. Results: `baseline_beam_evaluation.json`.

The raw baseline completed all 15 sessions:12 exact-variant development clips,2 local development examples, and the user's saved recording. The long recording retained 22 accepted windows and re-decoded four overlapping chunks at Finish; its live output revised 12 times. Its final visual decode changed the last provisional hypothesis. There is no verified reference transcript, so its output is diagnostic only and has no WER. The saved low-resolution constant-rate video cannot recover the original capture's timing.

PLEASE HELP I produced a first partial at2.54s and three updates before Finish, but final recognition repeated I. HELLO HOW YOU produced a first partial at1.40s and two updates before Finish; the wrong KNOW substitution persisted. These examples prove that provisional revision and final visual re-evaluation work, while also showing that the same model can retain mistakes. They do not justify relying on grammar to repair recognition.

`hud_preview.png` is a rendered and inspected preview of the changeable-transcript display. No camera was opened for this rendering. The existing test suite was expanded to 72 focused checks; all passed. Tests establish implementation behavior, not accuracy on natural signing.

## Reproduce the implemented mode

From the project directory, use the required Apple Vision virtual environment:

```sh
venv/bin/python scripts/live_reel_continuous_v17.py --revisable-transcript --no-speech --naturalizer literal
```

This explicitly enables revisable transcription with the existing repaired checkpoint. Press F/Finish once the phrase is complete; R resets. The live chips are provisional. The literal naturalizer option keeps the measured visual output separate from generative grammar; existing reviewed phrase templates may still render ordinary English. It does not claim to fix lexical mistakes.

To inspect the transition-trained experimental checkpoint, append:

```sh
--stage2-live-checkpoint artifacts/models/stage2_v17_revisable_v1/seed_17101/best.pth
```

The raw-video replay command is `venv/bin/python scripts/replay_revisable_transcription_v17.py --output artifacts/reports/stage2_v17_revisable_v1/replay_again`, optionally with the checkpoint above. It runs the frozen 12-clip exact-variant subset, two named local development clips and the diagnostic recording. Output must be reviewed as development evidence, not independent generalization.

Training commands:

```sh
venv/bin/python active/v17/train_stage2_live_adapt_v17.py --transition-supervision artifacts/reports/stage2_v17_revisable_v1/supervision.json --output-dir artifacts/models/stage2_v17_revisable_v1/seed_17101
venv/bin/python active/v17/train_stage2_live_adapt_v17.py --transition-supervision artifacts/reports/stage2_v17_revisable_core_v1/supervision.json --core-weight .25 --output-dir artifacts/models/stage2_v17_revisable_core_v1/seed_17101
```

Use a different output directory for a new run to preserve these artifacts. Both supervision artifacts pin source/cache hashes and roles. Training rejects development-role auxiliary targets. The protected official test gate remains closed.

## Completed transition-loss result

The gap+prefix run completed all 12 epochs in 1,738s (28.97min under concurrent desktop load). The declared selection chose epoch 12; no epoch passed either complete gate group. Matched connected WER is78.52% (223/284), local6.18% (16/259), exact50.00% (12/24). This improves connected WER relative to the repaired baseline168.31%, but is not a promotion: original local errors6->12, exact9->14 and contextual43->57. Citizen validation331/378->328/378 (86.77%) stays at the tolerance edge; STEM16/21 (76.19%) is retained.

On the connected OTHER-containing set, the new loss produces22 insertions,43 substitutions and158 deletions. Relative to the preceding matched adaptation (32/56/138), it removes10 insertions and13 substitutions but adds20 deletions. Known emissions in annotated development gap bins remain3/104 across ASLLRP, unchanged from that preceding adaptation (baseline13/104). Thus this particular gap/prefix objective has not demonstrated an additional transition-suppression benefit over the previous model, and it worsens sign coverage.

The completed positive-core comparison is evaluated separately below. A lower revision count is not automatically better: blank/empty outputs can be stable too.

## Failures found during implementation

Raw replay of an adapted checkpoint initially rejected `--revisable-transcript` because the provenance validator recognized only the older `--sequence-preview` flag. The shared validator now accepts either matched mode, still checks the identical input contract and teacher hash, and has a regression check. The failed attempt is retained under `raw_transition`; the complete retry is recorded separately under `raw_transition_retry`. A HUD mock also required its new keyword argument; the actual rendered HUD and expanded tests were checked after correction. These failures are not counted as successful replays.

## Final model comparison

All rows are matched development evaluations. Lower WER is better; WER can exceed 100% because insertions count. Each adapted row uses its declared selected checkpoint, with no eligible checkpoint substituted or gate relaxed.

| Model | Connected WER | Local WER | Exact-variant WER | Connected insertions | Connected deletions | Known emissions / annotated gap bins |
|---|---:|---:|---:|---:|---:|---:|
| Repaired baseline | 168.31% | 7.34% | 45.83% | 323 | 11 | 13/104 (12.50%) |
| Previous adaptation | 79.58% | 4.25% | 50.00% | 32 | 138 | 3/104 (2.88%) |
| Gap + prefix | 78.52% | 6.18% | 50.00% | 22 | 158 | 3/104 (2.88%) |
| Gap + prefix + known cores | 73.94% | 6.56% | 50.00% | 16 | 153 | 4/104 (3.85%) |

The known-core run completed 12 epochs in 1,821s (30.35min), selecting epoch 7. Its connected WER is56.07% lower relative to the repaired baseline, but153/284 reference signs are deleted (53.87%). It has144/225 clips with any known output (64.00%), versus223/225 (99.11%) for the repaired baseline. Empty output and fewer revisions cannot be counted as successful recognition.

Original retention for selected known-core epoch 7: local16/259 (6.18%) versus6/259 (2.32%); exact13/24 (54.17%) versus9/24 (37.50%); contextual56/254 (22.05%) versus43/254 (16.93%); Citizen316/378 (83.60%) versus331/378 (87.57%); STEM16/21 (76.19%) unchanged. None of the 24 trained epochs across the two runs passes promotion.

The direct gap-bin result is modest and sparse: the previous adaptation already reached3/104 known emissions; the new gap loss stays3/104 and the known-core variant reaches4/104. This does not establish additional coarticulation robustness from the new auxiliary losses. There are only5 eligible bins in exact-variant clips and99 in OTHER spans; no equivalent annotated local denominator exists.

| Model | Local exact phrases | Exact-variant exact phrases | Connected exact phrases | Local clips with revisions | Connected clips with revisions |
|---|---:|---:|---:|---:|---:|
| Repaired baseline | 79/97 (81.44%) | 3/12 (25.00%) | 17/225 (7.56%) | 21/97 | 118/225 |
| Previous adaptation | 86/97 (88.66%) | 2/12 (16.67%) | 55/225 (24.44%) | 24/97 | 62/225 |
| Gap + prefix | 83/97 (85.57%) | 2/12 (16.67%) | 54/225 (24.00%) | 23/97 | 41/225 |
| Gap + prefix + known cores | 81/97 (83.51%) | 2/12 (16.67%) | 59/225 (26.22%) | 21/97 | 30/225 |

## Final raw verification and decision

All 45 raw paced replays succeeded (15 per model). Every model matches its corresponding cached final hypothesis on all 12 exact-variant clips. Each model replays all 22 accepted windows from the saved user recording through four overlapping Finish chunks. See `raw_verification.json`; the original failed adapted-mode invocation is retained separately and excluded from these 45 successes.

| Raw model | Exact-variant WER | Complete exact phrases | Clips with nonempty output before Finish |
|---|---:|---:|---:|
| Repaired baseline |45.83% |3/12 (25.00%) |1/12 (8.33%) |
| Gap + prefix |50.00% |2/12 (16.67%) |0/12 (0%) |
| Gap + prefix + known cores |50.00% |2/12 (16.67%) |0/12 (0%) |

This table counts nonempty output, not merely an accepted empty hypothesis. Short external clips still frequently end before the input window, image computation and inference can provide useful output. The UI no longer waits for irreversible agreement, but the input/compute latency remains. Longer local clips produced nonempty output before Finish for all three models. The gap and core variants both recognized HELLO HOW YOU correctly in the named raw example, and both emitted an extra HELP in PLEASE HELP I. No examples were used to override the full development gates.

**Decision:** keep the opt-in revisable experience, retain the existing default checkpoint, and do not promote either new temporal adaptation. Adding a blank penalty did not resolve the suppression-versus-recall tradeoff. The known-core comparison offers lower aggregate connected WER but harms isolated/familiar retention. Another ensemble or a grammar-only correction stage is not supported by these results.

The next research issue is temporal alignment/recall under partial context, including explicit observed holds and repetitions. Existing gap labels are sparse and the auxiliary CTC-bin supervision is approximate. A subsequent experiment should use independently checked boundary/hold labels and measure recall alongside transition insertions; it should not label all transition-like movement blank or optimize on the protected test. This is a recommendation, not uncompleted work hidden inside this experiment.

Verification: `verification.json`, `training_verification.json`, `raw_verification.json`, `tests.log`. Best and last artifacts from both runs reload with finite parameters and matching supervision hashes. The fixed live input contract and baseline checkpoint hash are unchanged. `git diff --check` passes. Unrelated pre-existing worktree changes are preserved. Natural-language translation accuracy and performance on real iPhones were not evaluated.
