# Held-sign diagnostic prechecks

2026-09-16 PHT. Branch: `codex/stage2-held-sign-diagnostics`.

The independently demonstrated CTC rollover defect is corrected in both the
standalone Stage-2 demo and the Reel runner. The last raw token before the cut is
carried into collapse of the new context. An uninterrupted run is counted once;
blank-separated and OTHER-separated repetitions remain possible. State is cleared
on Finish/reset. This does not fix or explain duplicates originating inside a
short model context.

Verification: the new boundary regression failed before implementation (actual
two emissions versus expected one continuation); 51 focused tests now pass.
The actual Reel event-loop fixture spans several context rollovers. Both CLI
parsers accept `--ctc-trace-dir`; syntax and whitespace checks pass. Real-model
replay results are pending the background job, not claimed by these checks.

Optional tracing writes compressed logits, the frozen visual window, original
observed source timestamps, and the previous boundary token. The Reel history
records the raw path, trace path and context offset. Trace files use exclusive
creation to avoid overwriting previous evidence; use a new trace directory for
each session. Reported total inference time includes trace-writing overhead.

The frozen manifest contains 10 recordings × four window offsets = 40 replay runs:
the nine reviewed video examples plus a 20-second unsegmented webcam diagnostic
for context rollover. Source hashes, readable streams and timestamp coverage were
checked. Four corpus examples have existing validation references; webcam and
partial O5S5 examples remain unscored diagnostics. No protected Citizen test access.

The worker uses the session's pinned original selector, without OTHER-preservation
or experimental learned adapters. It compares legacy and corrected collapse of
the same logits, plus a final whole-context or overlap-stitched decode. It records
the distinction between exact single-context and stitched comparisons. Low-resolution
re-extraction cannot reproduce the original camera features exactly; original live
output equivalence and real-time latency are not assumed.

Outputs after execution: `REPORT.md`, `results.json`, `summary.json`,
`provenance.json`, per-phase emission archives, and `completion.json`.
Failure produces `FAILURE.md` and a failed `completion.json`.
The detached worker sends one macOS notification on exit; it does not poll or
automatically reopen this chat. A hard OS kill/power loss cannot run an exit handler.

No model training is launched. The next decision depends on the replay report.
Expert-labelled, independent hold-versus-repeat recordings are still missing;
saved webcam observations are not converted into training labels. The broader
gloss-free challenger evaluation remains a subsequent, separate comparison.
