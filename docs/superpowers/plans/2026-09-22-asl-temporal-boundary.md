# ASL temporal boundary implementation plan

**Authorization:** User approved the preceding design and explicitly requested implementation,
2026-09-22. Continue this plan without unrelated architecture, UI or acquisition work.
Use superpowers:executing-plans inline; preserve all unrelated working-tree changes.

**Goal:** Recognize locked100 signs individually or naturally connected without required
pauses/wrist movement, retaining correct words while reducing transition insertions.
Stable output target is approximately0.5–1s after sign end; actual timing must be measured.

**Design/spec:** Continuous Apple Vision finger/hand/body observations feed a small
native-rate temporal convolutional boundary model. Separate start/end/inside outputs
allow adjacent signs without blank frames and repeated identities. Reel remains the
identity baseline. Bounded future context is explicit; no full-video normalization or
attention may leak future evidence into online boundaries. Matched whole-video evaluation
compares unchanged Reel, reviewed intervals with explicit trimming policy, and learned
intervals. Train identity adaptation only if the oracle comparison supports it; preserve
the current default unless whole-stream accuracy and retention justify promotion.

**Stack:** Existing Python3.9 venv, NumPy, PyTorch/MPS, OpenCV/Apple Vision, existing CoreML
Reel recognizers. No new dependencies, foreign blank veto, downloads or protected test.

## Constraints and review focus

- Existing annotations have separate identity, timing and completeness contracts. Mix
  compatible admitted sources; do not label incomplete gaps, OOV or unresolved labels blank.
- Preserve current roles, parent-event deduplication, source hashes and canonical manifest
  verification. New boundary recipe authorizes only its own dedicated runner.
- Repeat/hold decoder must emit once per event, not once per class and not on every frame.
- Missing detections, timestamp gaps, EOF and reset/page changes must not fabricate events.
- Training/inference preprocessing, sampling, context and decoding must match; tests cover
  bounded causality, chunk equivalence, touching intervals and ignored supervision.
- Record real metrics with denominators; no claim of robust accuracy from compatibility tests.

## Tasks and persistent progress

- [x] 1. Pin boundary source eligibility and whole-video evaluation membership; verify the
  canonical approved manifest. Write new recipe only after source review/preflight.
- [x] 2. Implement/test native-rate temporal model, causal features, masked targets and
  repeat-safe interval decoder in `active/v17/temporal_boundary_v17.py`; focused test
  `test/test_temporal_boundary_v17.py` must fail before implementation and then pass.
- [x] 3. Implement `scripts/train_temporal_boundary_v17.py`: mixed eligible timelines,
  hashed recipe, MPS preflight, fixed two seeds/epochs, detached completion notification.
  Compare skeleton features with added within-hand geometry under the same contract.
- [x] 4. Implement `scripts/evaluate_temporal_boundary_v17.py`: actual Reel video replay,
  reviewed interval identity/commit diagnostics with trim on/off, learned streaming
  boundaries, edit counts and correct-event retention, boundary accuracy and latency.
- [x] 5. Implement opt-in shared live/evaluation boundary recognizer and app routing,
  preserving default Reel. Verify reset/EOF, adjacent repeats and training/live parity.
- [x] 6. Run bounded experiment/evaluation, inspect completion results, make the conditional
  identity-adaptation decision. Do not auto-extend a failed recipe or promote failed weights.
- [x] 7. Review, focused integration checks, `git diff --check`, artifact index, and canonical
  current-state/log update with measurements and exact next safe action.

## Decisions / execution ledger

- Start: current branch `codex/stage2-held-sign-diagnostics` includes required uncommitted
  upstream work. Work in place without checkout/stash/commit; new isolated worktree would
  omit the approved current pipeline. User implementation instruction supplies authorization;
  no repeated design-approval request.
- Historical boundary-phase and segment-first failures reviewed. Native-rate continuous
  sequence features, bounded causality and shared streaming replay distinguish this run
  from normalized32-frame window classification and pooled interval vetoes.
- One independent read-only data contract review requested; no other parallel implementation.
- Source review ruling: gaps are not certified physical background even in the complete
  transcript subset. Use two independent START/END outputs, no untrainable inside gate.
  Within accepted sign interiors only, non-edge values are zero; all uncertain regions
  remain ignored. 4,310 events after54 ASLLRP duplicate removals/current O5S5 crosslink,
  3,518train/792validation across1,121raw sequences. This is20Hz live-matched observation
  rate, not native camera FPS. ASLLRP OOV timing can teach boundaries without lexical labels.
- Preparation and MPS preflight pass:909training/146parent-held calibration/260validation
  chunks;78,402parameters with hand geometry; finite loss/gradients; zero optimizer steps.
  Generic phrase manifest remains false and its494archives still verify.
- Baseline evaluator initially rejected transcript mismatch because historical annotation
  timelines include short OTHER edge fragments absent from the current known transcript.
  Fixed at reference construction: retain all events, score only approved known identities.
  No data relabeling. First failure preserved in /tmp/slt_boundary_baseline_20260922.log.
- Paired12video/24reference replay: actual Reel4correct,1substitution,19deletions,
  zero insertions (83.33%WER); reviewed intervals+100ms12correct,1substitution,
  11deletions (50%WER), retains3/4baseline correct reference positions. Verifier16/24.
  No-trim and trim arms identical on this sample. Timing is useful but not sufficient;
  contextual identity/commit errors remain, no new low-motion evidence. Continue boundary
  comparison first; do not launch unrelated backbone/identity training before its result.
- Final reviews corrected two-edge decoding (a new START cannot fabricate an END),
  independently masked start/end targets (two dataset cells previously depended on event
  ordering), raw-edge coverage assertions, full cached observer-contract verification,
  inference code hash checks and incompatible live-option rejection. Source reviewer
  verified all1121metadata identities/contracts and4310edge pairs. Earlier preflight/recipe
  snapshots preserved; no training used those preliminary recipes.
- Final validation:87focused tests pass; full `git diff --check` passes; MPS preflight
  finite loss/gradients with zero optimizer steps. Final recipe SHA256:
  `25b24499633df10cebf69bbe99f49f571c020986a8145f017a65b6917dc057ad`.
- Task6 launched detached under caffeinate PID50241 at2026-09-22T03:47:13Z. Four bounded
  runs:seeds17621/17622×skeleton/hand-geometry,8epochs each, then automatic whole-video
  shared-runtime comparison and REPORT.md generation. One completion/failure notification.
  Do not poll. `artifacts/reports/asl_temporal_boundary_v17_20260922/completion.json`
  records outcome; read it after notification/next authorized session. Do not assume
  success from launch. Runtime code is pinned during execution; default Reel unchanged.
- Task7 code/source reviews and prelaunch tests are complete. Final result review and
  conditional contextual-identity recipe remain pending task6 completion. No silent
  promotion, threshold sweep, extra epochs or new data acquisition.

- Result review on user “check”: all four runs completed (131.44s total); eight code
  hashes, dedicated recipe and four checkpoint hashes match. Skeleton 5–6/24 correct,
  geometry 7–8/24 versus Reel 4/24; geometry retains only 3/4 baseline correct and
  misses 16/24, with zero exact videos. No promotion. Median successful-word delay
  0.58–0.65s; boundary CPU p95 61–80ms exceeds a 50ms frame budget. This reused
  replay has no baseline pure-gap commits, so no false-transition reduction established.
  Conditional identity decision: diagnose saved interval/proposal/verifier/commit evidence
  first; any contextual identity training needs a bounded evidence-based recipe, not an
  automatic extension. The experiment checklist is complete; the recognition goal is not.

- User continuation now follows pretrained transfer: shipped DGS pose model verified,
  frozen offline comparison11/24correct/retains4of4. Feature preparation for existing1121
  records launched detached; no training/promotion yet. Follow
  `artifacts/reports/pose_boundary_transfer_v17_20260922/PLAN.md` and REPORT.md.
