# Stage 2 held-sign diagnostics implementation plan

**Goal:** Produce pinned emission/replay reports and correct the independently
reproduced CTC boundary-counting defect while preserving true repetitions.

**Spec:** `artifacts/reports/stage2_research_review_20260915/README.md`.
**Architecture:** Reuse the existing encoder, CTC selector, observation extractor,
and window grouping. Carry the last raw CTC token across a rolled boundary; save
optional compressed logits and observed source timestamps. No new model or loss.

## Constraints

- Work on `codex/stage2-held-sign-diagnostics`; preserve pre-existing user changes.
- Diagnostic webcam observations are not training labels or expert truth.
- No Citizen test access, promotion, automatic training, or extractor replacement.
- Run long work detached with one completion/failure notification; never poll it.
- New human-labelled, signer-disjoint hold/repetition recordings remain unavailable.

## Tasks

1. [ ] Add regression cases in `test/test_live_stage2_ctc_v17.py` for a continued
   boundary run, a blank-separated repeat, and an OTHER-separated repeat. Run
   the focused tests before changing production code.
2. [ ] Extend `collapse_ctc_path` with `previous_token=0`. Pass it through
   `LiveStage2CTC.decode_frozen_windows` and `classify_window`. Both live callers
   retain the raw context path and use its exiting-window last token at rollover.
   Clear that state on reset/Finish. Keep full raw argmax in results.
3. [ ] Add opt-in `--ctc-trace-dir` compressed emission files containing logits,
   exact observed source timestamps, frozen visual window, and boundary token.
   Existing runs need not enable this diagnostic output.
4. [ ] Add a report-local replay runner using `recording_observations`,
   `phase_windows`, and the actual selector. Freeze existing viewer examples as
   diagnostic/validation input; compare four window origins, legacy versus
   corrected prefix handling, and final stitched collapse. Store checkpoints'
   and inputs' hashes. Report per-clip results without invented webcam WER.
5. [ ] Verify affected CTC/Reel tests and a minimal real-model smoke, update ground
   truth, then launch the bounded replay job with `Popen(start_new_session=True)`.
   The job writes results, Markdown report, completion JSON and one macOS
   notification on success/failure. End the turn after launch; no status polling.

Training is conditional on evidence and suitable training data. The initial run
is diagnosis/inference, not another optimization experiment. A gloss-free model
comparison follows this report and uses its own input schema and task metrics.
