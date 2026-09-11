# Restore reliable continuous transcription using Stage 1

## 1. Goal and approach

Keep the existing Squeezeformer and adapt it to recognize signs within continuous motion. Preserve the transcription experience: provisional words appear while signing, recent words can change, and Finish triggers final visual evaluation before grammar correction.

**Chosen priority:** accuracy first, targeting a median first-correct-word delay of at most one second after the annotated sign ends.

Implement one bounded Stage 1 candidate and compare it against older Reel and current revisable CTC. Do not introduce a new backbone, ensemble, or transducer, or repeat the rejected Stage 2 loss experiments.

## 2. Diagnose the regression and freeze the comparison

- Extend the existing replay tool to accept a recording manifest and original frame timestamps. Use the recovered HUNGRY recording as a diagnostic example; disclose its reduced resolution and exclude its damaged final frame.
- Run identical frames through the older phrase-adapted Stage 1 proposal, its visual verifier, and current Stage 2.
- Record HUNGRY’s rank and score, competing labels, hand-detection coverage, window boundaries, and Stage 2 outputs before and after OTHER suppression. Record rejection reasons rather than only accepted/rejected status.
- Repeat the diagnostic with window origins shifted by one-quarter, one-half, and three-quarters of the current window period. Keep checkpoints and frames unchanged.
- If a reproducible timestamp, extraction, or suppression defect explains the regression, fix that defect first and rerun the comparison. Skip training if the repaired existing path meets all acceptance gates below.

Freeze checkpoint hashes, evaluation identities, decoding settings, and baselines before candidate training. The user recording remains diagnostic-only; do not train on it or report whole-session WER without a verified transcript.

## 3. Build one candidate trained for continuous windows

**Starting point:** the phrase-adapted landmark Stage 1 used by older Reel. Reuse the core-adaptation trainer and existing emission wrapper; train the encoder as well as the emission output.

Relevant starting points:

- `active/v17/train_stage1_asllrp_core_adapt_v17.py`
- `active/v17/model_reel_emission_v17.py`
- `scripts/live_reel_stage1_v17.py`

**Training data**

- Use permitted Citizen/SemLex isolated training replay and continuous recordings with verified labels and sign boundaries.
- Do not reuse equal-duration phrase partitions as ground-truth boundaries. Admit foreground and transition supervision only from checked intervals.
- Construct windows around annotated sign centers, retaining actual preceding/following motion. Use durations of approximately 0.27, 0.53, and 1.07 seconds; preserve timestamps and resample to the existing 32-frame input.
- Supervise the center sign when the window contains verified foreground evidence. Supervise NO_EMIT only for verified transition/background windows.
- Exclude ambiguous partial signs and unreviewed regions from background supervision. Keep out-of-vocabulary signing identified separately in the manifest and evaluation.
- Preserve source and signer splits. Record distinct signs, sign pairs, signers, durations, and annotation coverage—not merely clip counts.

**Bounded training recipe**

- Retain the existing 100 gloss outputs and append the existing NO_EMIT mechanism. Use a separately versioned checkpoint; do not overwrite accepted artifacts.
- Update the landmark encoder, classifier, and emission head. Train on window classification plus foreground-only classification using the annotated foreground mask.
- Sample 50% isolated replay, 30% contextual positive windows, and 20% verified background windows, balanced within sources/classes. Fail preparation if a required category lacks valid supervision.
- Use loss weights of 1.0 for window classification, 0.5 for foreground classification, and 1.0 for replay consistency against the frozen starting model. Apply replay consistency only to known-sign replay.
- Run 12 epochs, 3,000 samples per epoch, batch size 64, AdamW at `2e-5`, weight decay `1e-4`, cosine decay, and gradient clipping at 1.0.
- Run seed 17111 first. Run confirmation seed 17112 only if the first produces an eligible candidate. Do not launch an open-ended hyperparameter search.

This differs from the rejected experiments by updating the recognizer on contextual windows with verified foreground supervision, rather than training another decoder over frozen evidence.

## 4. Integrate revisable Stage 1 transcription

- Add an opt-in `--transcript-backend stage1-window` option alongside the existing CTC backend, with an explicit candidate-checkpoint argument.
- Evaluate a trailing 0.53-second window every 0.13 seconds, using actual timestamps. Training must include this exact window schedule.
- Use the existing emission output to suppress background. Require two consecutive agreeing predictions for a provisional sign; merge repeated observations of the same sign until an intervening NO_EMIT run or different sign occurs.
- Recompute the latest two seconds of provisional text from retained predictions. Do not require the signer to pause or hold until a permanent lock.
- At Finish, re-evaluate retained input with the same classifier and window schedule, including the final partial window. Then run the existing grammar stage.
- Keep the expensive hand-image verifier out of this candidate’s routine loop. Evaluate it as a diagnostic comparator; do not silently add a second recognition pipeline if landmark-only accuracy fails.
- Keep existing runtime defaults unchanged. Export the selected candidate to Core ML only after its development gates pass.

Explicitly test repeated identical signs: the proposed duplicate-removal rule is a baseline whose limitations must be measured, not assumed solved.

## 5. Acceptance, verification, and handoff

**Select checkpoints using complete streaming results, not cropped-sign accuracy alone.** Among eligible checkpoints, prefer lower connected WER, then fewer deletions, then lower latency, then the earlier epoch.

A candidate is eligible only when:

- Connected WER improves by at least **10% relative** against current revisable CTC on the frozen matched development pool.
- Connected insertions and deletions each do not increase.
- Familiar-phrase WER does not worsen against older Reel under matched replay.
- Citizen and SemLex isolated validation accuracy each drop by no more than **1 percentage point** from the candidate’s own starting model.
- Transition false emissions do not increase on the verified transition subset.
- Median first-correct-word delay is at most **1 second** after sign completion. Report p95 and missed-sign frequency separately; never omit misses to make latency look better.

Verify:

- HUNGRY attempts, short signs, boundary shifts, long holds, repeated signs, rapid sign pairs, hand loss, idle motion, and out-of-vocabulary signing.
- Reset, Finish, long utterances, partial final windows, timestamp gaps, and preservation of recent-word revisions.
- PyTorch/Core ML agreement and paced raw-video replay through the actual runtime.
- Focused affected tests first, followed by relevant integration checks and `git diff --check`.

Use historical development sets for engineering comparisons only. Never reopen protected tests. Missing independent repetition, background, or signer coverage must be reported explicitly; broader reliability claims require a fresh independently annotated evaluation.

Deliver a comparison report, timestamped error examples, checkpoint provenance, measured latency, and one runnable experimental Mac command. Record results in project history during implementation. If no candidate passes, retain the existing defaults and report the specific failed gates—without another speculative training run or a claim that the model is fixed.
