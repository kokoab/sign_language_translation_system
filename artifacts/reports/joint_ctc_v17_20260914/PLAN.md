# Frozen versus joint CTC experiment

Approved by the user's instruction to run the experiment and return with results.
Implements `../stage2_data_learnability_audit_v17/NEXT_STEP_REVIEW.md`.

- [ ] Add tested timestamped chunk/CTC data routing and a differentiable wrapper around
  the existing Stage-1 encoder and UnifiedStreamingCTCHeadV17. Files:
  `active/v17/joint_ctc_v17.py`, `test/test_joint_ctc_v17.py`.
- [ ] Implement bounded preparation, preflight, training and evaluation in report-local
  `run_experiment.py`; freeze provenance before optimizing. Use the existing frozen
  1,901/1,356 isolated replay identities and 334-recording development manifest.
- [ ] Verify gradients, finite feasible CTC, tiny-subset fit, raw-core/contextual-core
  diagnostics and CPU/MPS agreement before the full comparison.
- [ ] Train frozen and joint arms for 12 epochs, seed 17111, identical examples and
  update schedules. Evaluate every epoch; retain checkpoints and report failed gates.
- [ ] Verify report/checkpoint provenance, focused tests and diff; update current state
  and history. Confirmation only if a candidate passes every existing gate.

Fixed recipe: reuse three causal dilated blocks (128 hidden units). Stage-1 uses
timestamped non-overlapping chunks of at most 0.53s, 32 resampled frames per chunk,
with duplicate endpoint ownership removed. Final partial chunk is flushed. Frame tokens
become available only when their chunk closes; this is bounded-lookahead recognition.
The isolated head uses its original 32-frame input. Sequence CTC has 102 outputs.
No reduction of the temporal token stream and no language model/phrase grammar.

Fully annotated ASLLRP spans use all known+OTHER targets. O5S5 supplies exact positive
cores only. Verified ASLLRP gaps supply an explicit blank loss. Each training sequence,
O5S5 core, admitted background window and isolated replay clip is visited each epoch;
sources contribute separately normalized losses. Background IDs include timestamps.
Joint uses encoder LR 1e-5; both heads use LR 3e-4. AdamW, weight decay 1e-4,
gradient clipping 5.0; sequence batch 8, isolated batch 32, core/background batch 16.
Loss: source-balanced sequence CTC + isolated CE + replay KL (T=2) + O5S5 core CE
+ 0.2 verified-gap blank CE. No mixed whole-window positive CE.

Budget comparisons use identical optimizer updates and training schedules in both
arms; historical experiments are descriptive references, not compute-matched controls.
Both encoders stay in eval mode (gradients enabled only for joint), so encoder dropout
is identical; the CTC head trains normally. Freeze the Stage-1 teacher in eval mode.
Single-observation exact cores repeat only that observed pose across the fixed input;
report them separately as static evidence. Preserve Citizen test sealing and defaults.
Tiny-fit preflight is train-only, disposable, at most 200 updates; reset from the same
original initialization before each real arm. Record any implementation/preflight
failure and its correction before continuing.

Promotion reuses `failed_gates()` from `scripts/evaluate_stage1_window_v17.py`:
connected WER at least 10% better than accepted baseline, no extra insertions/deletions,
no worse familiar WER or verified-gap emissions, <=1 point isolated retention drop,
matched full pool, complete streaming, runtime-inclusive median delay <=1s. Cached
timing cannot satisfy the runtime gate. Accuracy rejection stops promotion/confirmation;
cached delay, misses and CPU execution cost will still be reported separately.
