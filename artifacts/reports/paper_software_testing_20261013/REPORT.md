# ATLAS software testing — October 13, 2026

Authorized by the user to execute tests, correct stale expectations, and apply verified results to Google Docs Manuscript, section4.5.5.

## Results

- Full focused Python discovery:245passed,0failed,0skipped;21.550s. Selected29current-processing/interface cases for the paper.
- Flutter:7passed,0failed,0skipped (5widget,2data-unit cases).
- Selected native simulator run:8passed,1failed. The failure expected a removed manual Letters button. Current development log explicitly records removal. Corrected the obsolete test to check absence of removed controls and presence/visibility of Back, landmarks and Help. Retest of corrected toolbar and portrait/landscape case:2passed. Nine distinct native cases have passing final results; the orientation retest is not counted twice.
- Before execution, corrected older portrait rejection assertion to match current portrait-first behavior while retaining landscape checks.
- JVM:2passed,0failed,0skipped in forced re-execution. Initial broad Gradle task inadvertently selected third-party plugin tests and failed on missing offline Mockito dependencies. Restricted to the intended app test target. Its first success was UP-TO-DATE, so forced the app test task to execute again and retained fresh XML.
- Paper subset:47distinct cases =38white-box/internal +9interface/integration. No final failures or skips. This is the latest recorded result per case, not one simultaneous combined run.

## Scope

Tables20–21 summarize the47case subset, with AppendixAregister. Existing participant provisions and model measurements unchanged. Main narrative records initial test-expectation failure and correction. Native test execution used the iOS26.3iPhone17Pro simulator, not physical-device tests. Python interface tests can use controlled model substitutes. No model retraining, protected accuracy evaluation or physical app installation.

The245Python pass count is broader regression evidence, not the main application total. Profiling/replay harnesses, unexecuted native model-dependent checks, experimental-only tests and empty macOS scaffold excluded from47. Repeated inputs,100playbackassets and retests not inflated as distinct methods.

## Evidence

- `results.json`: selected cases, groups, source hashes, base revision and corrections.
- `CASE_REGISTER.md`: exact selected test methods and final evidence location.
- `python.log`, `flutter.jsonl`, `swift.log`, `swift_retest.log`, `kotlin_fresh.log`, `kotlin_results.xml`.
- Native result bundles: `/Users/frnzlo/slt_paper_whitebox_20261013.xcresult` and `/Users/frnzlo/slt_paper_whitebox_retest_20261013.xcresult`.
- `manuscript_tail_before.json`: backup of changed tail before insertion.

Only test expectations changed in application sources. Existing unrelated native test/UI edits preserved. No shared-code behavior altered to force passing tests. `git diff --check` passes for the affected test source.

Next: review inserted manuscript tables and AppendixA; independent participant and device evaluations retain their separate scope.

Final verification: prior manuscript paragraphs and other tab bodies preserved; all3native tables match intended cells exactly (9,7,48rows); no insertion markers remain. PDF layout inspected and corrected; contents/list entries updated to printed pages108,109,111and118. Scope whitespace check passes.
