# ATLAS — White-Box and Automated Application Testing

**Prepared:** October 13, 2026  
**Status:** Executed results applied to the Manuscript tab, Section 4.5.5. Original proposal retained below for review history.  
**Destination:** [Manuscript tab](https://docs.google.com/document/d/1sC2JZ23mSpnLWeEEFDm5ce2x_As0XTS7ayMGvk5uuD4/edit?tab=t.d1buvx4fb1j1)

The manuscript passages below use the proposed final wording. Placement instructions, review notes and the evidence register are editorial material and should not be pasted into the paper. Unverified result cells remain marked rather than being presented as completed findings. Nothing is hidden in dropdowns.

## Applied results — October 13, 2026

### Software Testing Results and Discussion

Automated software testing examined the internal processing logic and application behavior of ATLAS. The reported application subset comprised 47 distinct cases: 38 internal-logic cases and nine interface and integration cases. All 47 passed in their final recorded executions, with no failed or skipped cases remaining in the reported subset. Results were recorded on October 13, 2026, and grouped by component rather than programming language.

The subset was drawn from checks of the documented processing and application functions. The broader Python regression run also passed all 245 tests, but its additional experimental-model, data-preparation and development checks are not included in the 47-case application summary. Each reported case has one primary group; shared coverage and retests do not increase the case count. Appendix A provides the case register and execution conditions.

### White-Box Testing

White-box testing exercised internal functions, decision conditions and state changes using controlled inputs and defined expected outcomes. The 38 cases checked input preparation, frame handling, recognition interfaces, boundary processing, sign acceptance, the Finish gesture, English-generation integration and application data handling.

*Table 20. White-Box Testing of Internal Processing*

| ID | Component and checks | Cases | Pass | Fail | Skip |
|---|---|---|---|---|---|
| WB-01 | Input preparation
Coordinate rotation, missing landmarks, sampling and input-format checks | 8 | 8 | 0 | 0 |
| WB-02 | Frame processing
Replacement of stale frames and invalidation of stopped work | 1 | 1 | 0 | 0 |
| WB-03 | Recognition integration
Model input structure, anatomical processing and hand-image slot handling | 4 | 4 | 0 | 0 |
| WB-04 | Boundary detection and decoding
Candidate intervals, confidence rejection, repetition handling and temporal ordering | 16 | 16 | 0 | 0 |
| WB-05 | Finish and state management
Control-gesture conditions, activation timing and state reset | 1 | 1 | 0 | 0 |
| WB-06 | English-generation integration
Gloss delivery, model routing, excessive-input rejection and name restoration | 6 | 6 | 0 | 0 |
| WB-07 | Application data handling
Recognition-result parsing and saved-session formatting | 2 | 2 | 0 | 0 |
| Total |  | 38 | 38 | 0 | 0 |

All 38 internal-logic cases passed. Input checks verified the expected treatment of coordinates, missing observations and input structures. Frame-processing checks verified that queued stale frames were replaced and stopped work was invalidated. Boundary and decoder checks exercised candidate construction, low-confidence rejection, repeated-sign handling and ordering. The Finish checks covered valid and invalid control conditions, activation timing and reset behavior.

English-integration checks examined gloss delivery, routing through the model interface, rejection of excessive input and restoration of name placeholders. Some cases used controlled substitutes for model outputs to isolate application logic. These results establish the behavior of the tested interfaces and processing rules; they are not additional measurements of language quality or sign-recognition accuracy.

### Automated Interface and Integration Testing

Nine automated cases examined navigation, layouts, controls, preferences, reference playback and session presentation. Five cases exercised the application interface through widget tests, while four exercised native layouts, controls or media integration. The combined preference/reference case also checked layouts; it is counted once under its primary responsibility.

*Table 21. Automated Interface and Integration Testing*

| ID | Component and checks | Cases | Pass | Fail | Skip |
|---|---|---|---|---|---|
| AI-01 | Navigation
Application entry, page navigation and user-manual navigation | 3 | 3 | 0 | 0 |
| AI-02 | Layout and controls
Manual layout, orientation support and the current camera toolbar | 3 | 3 | 0 | 0 |
| AI-03 | Preferences and reference display
Landmark preference persistence and reference enlargement/dismissal | 1 | 1 | 0 | 0 |
| AI-04 | Practice media
Playback and looping of all 100 reference videos | 1 | 1 | 0 | 0 |
| AI-05 | Session presentation
Recognized signs, timeline and unfinished-session display | 1 | 1 | 0 | 0 |
| Total |  | 9 | 9 | 0 | 0 |

All nine interface and integration cases passed in their final recorded executions. The playback case checked all 100 bundled reference videos for playback and looping. These assets are multiple inputs within one test method, rather than 100 independent test cases. The reported interface checks used controlled test environments, including a mobile simulator, and do not establish physical-camera or sustained-device performance.

### Test Corrections and Interpretation

Review of the native tests identified an obsolete landscape-only expectation, which was updated to reflect the application's portrait and landscape support while retaining the landscape layout checks. The initial native execution also found a test that still expected a manual Letters control removed from the current interface. The test was aligned with the documented interface requirements to verify the absence of removed controls and the visibility of the remaining controls. The corrected toolbar and orientation cases passed on re-execution. These were corrections to outdated test expectations; the application behavior was not changed to obtain a passing result.

Passing cases support the functions and conditions exercised by those tests. They do not establish complete code-path coverage or guarantee correct recognition of every sign. Recognition accuracy, word error rate, English-output quality and on-device processing time remain separate measurements. Software test execution times were not used as substitutes for inference or application-processing times.

These technical testing results do not constitute participant-based software-quality findings. Participant evaluation remains subject to the approved instruments and research procedures.

[Full 47-case execution register](../../artifacts/reports/paper_software_testing_20261013/CASE_REGISTER.md). The Google Docs appendix contains the case summary. Earlier unresolved fields below belong to the historical proposal and are superseded by these results.

---

## Historical review checklist

- [ ] WBT-01 — Opening testing discussion within Results and Discussion, Section 4.5.5.
- [ ] WBT-02 — White-box discussion and component summary within Section 4.5.5.
- [ ] WBT-03 — Automated interface and integration discussion within Section 4.5.5.
- [ ] WBT-04 — Interpretation of results and separation from model accuracy.
- [ ] WBT-05 — Appendix format and test traceability.
- [ ] Resolve the outdated native orientation assertion before reporting a complete-suite pass.
- [ ] Confirm the selected test scope and fill the result tables from a fresh, recorded execution.

## Placement and preservation notes — editorial only

Place WBT-01 through WBT-04 entirely within the results-and-discussion material under **4.5.5 Functional Testing and Software Quality Evaluation**, within **4.5 Implementation Results**. Use WBT-01 as the opening discussion, followed by the white-box results, interface and integration results, and interpretation. Use the unnumbered subheadings shown below. Do not insert a separate testing-methodology passage in Section 4.2. Retain the existing SDLC, Activity List, GANTT, PERT, design diagrams and other development discussions unchanged. This does not introduce another main Chapter 4 section. Preserve the existing distinction between technical system testing and human participant evaluation, together with its participant safeguards. The new technical testing results do not replace participant-based software-quality findings.

Use two component summaries in the main paper and an individual-case register in the appendix. Assign final table and appendix numbers during insertion, updating the lists and cross-references together. Group by system responsibility rather than programming language. Framework names and execution environments belong in the appendix evidence record.

## WBT-01 — Opening manuscript wording for Results and Discussion, Section 4.5.5

### Software Testing Results and Discussion

The software-testing results are organized into internal processing checks and automated interface and integration checks. The internal checks examine input preparation, frame processing, recognition integration, temporal boundary handling, sign acceptance, finalization and application data handling. Their outcomes are interpreted against the expected outputs and state changes defined for each test case.

Test inputs include valid observations, missing observations, incomplete sign intervals, repeated inputs and invalid data structures. Depending on the function under examination, the expected outcome is a defined output, a preserved state, rejection of an invalid input or completion of a required state change. Comparisons between processing implementations check whether the same prepared inputs produce the expected corresponding outputs.

Additional automated interface and integration tests examine navigation, screen layouts, preference handling, reference-video playback and session presentation. These tests complement the internal logic checks by exercising interactions between application components. User-based functional evaluation and participant-based software-quality assessment follow their respective procedures.

The component summaries distinguish individual test cases from repeated executions and multiple inputs checked within a case. The accompanying test records identify the source version, environment, expected outcome, actual outcome and execution status. Recorded outcomes are discussed below, while the individual-case register provides the supporting detail.

**Review note:** This is the opening explanation of the software-testing results, not a separate methodology insertion. It does not claim that every repository test belongs to the selected application or that statement or branch coverage has been measured.

## WBT-02 — Manuscript wording for Section 4.5.5

### White-Box Testing

White-box testing examines the internal operations that connect visual input, sign recognition, sequence processing and English generation in ATLAS. The checks are organized by processing responsibility so that related tests can be interpreted together. Each group may contain several individual cases covering normal operation, boundary conditions and rejection behavior.

*Table. White-Box Testing of Internal Processing*

| Test group | Component | Internal behavior examined | Executed | Passed | Failed | Skipped |
|---|---|---|---:|---:|---:|---:|
| WB-01 | Input preparation | Coordinate transformations, missing-landmark handling, temporal sampling and conformance to the expected model-input dimensions | — | — | — | — |
| WB-02 | Frame processing | Replacement of queued stale frames, controlled processing of new frames and invalidation of work after processing stops | — | — | — | — |
| WB-03 | Recognition integration | Agreement between individual and batched image encoding, alignment of recognition inputs and preservation of model features during display-only updates | — | — | — | — |
| WB-04 | Boundary detection and sequence decoding | Candidate-interval construction, confidence-based sign acceptance, repeated-sign handling and preservation of temporal ordering | — | — | — | — |
| WB-05 | Finish and state management | Recognition of the defined control gesture, rejection of incomplete control conditions, prevention of repeated activation and reset of the relevant state | — | — | — | — |
| WB-06 | English-generation integration | Delivery of gloss sequences to the language model, retention of completed sentences, processing of the remaining text at finalization and handling of excessive input | — | — | — | — |
| WB-07 | Application data handling | Parsing of recognition results and conversion of saved-session records into the expected application representation | — | — | — | — |

The groups describe distinct internal responsibilities rather than individual model-accuracy measurements. A case is counted once in the summary under its primary responsibility, even when it also provides evidence for another component. The individual-case register identifies the functions and conditions exercised by each case.

**Review note:** The dashes are unresolved editorial fields, not zero results, skipped tests or passing outcomes. Fill them only after mapping and executing the selected cases. Do not publish the table with unresolved cells. The final report should distinguish attempted cases from completed and skipped cases; use a consistent counting convention and state it in the appendix.

## WBT-03 — Manuscript wording for Section 4.5.5

### Automated Interface and Integration Testing

Automated interface and integration testing examines how application components work together during navigation, display, playback and session review. These checks complement white-box testing by assessing observable application behavior under controlled conditions.

*Table. Automated Interface and Integration Testing*

| Test group | Application area | Behavior examined | Executed | Passed | Failed | Skipped |
|---|---|---|---:|---:|---:|---:|
| AI-01 | Navigation | Opening the application, entering Practice, Glosses, History and the user manual, and returning through the intended navigation flow | — | — | — | — |
| AI-02 | Layout and controls | Visibility and placement of controls and manual content across the tested screen sizes and orientations | — | — | — | — |
| AI-03 | Preferences and reference display | Persistence of landmark-visibility preferences and operation of reference-video enlargement and dismissal | — | — | — | — |
| AI-04 | Practice media | Availability, playback and looping of the supported reference videos | — | — | — | — |
| AI-05 | Session presentation | Display of recognized signs, timeline information and unfinished-session status | — | — | — | — |

The recorded application checks include a seven-case suite covering navigation, manual layout, session presentation, history formatting and recognition-result parsing. All seven cases passed in the documented execution. Five cases exercised interface behavior, while two checked data formatting and parsing. The two data-handling cases contribute to the white-box summary rather than being counted again as interface tests.

A separate focused native execution passed three test methods covering camera-coordinate transformations, reference-video playback and application controls. The playback method checked all 100 bundled reference videos for playback and looping. The controls method examined the tested screen layouts, landmark-visibility preference and reference-display interactions. A subsequent execution repeated the updated layout method after a placement adjustment; this retest is part of the same case history rather than an additional independent test.

These recorded executions establish the outcomes of the named checks in their documented environments. They do not establish that every test method in the repository was executed together or that all device functions were evaluated on a physical mobile device.

**Review note:** The seven-case result is recorded in `artifacts/reports/mobile_camera_controls_20261013/REPORT.md`, together with the three-method simulator result and the later one-method layout retest. The three native methods span both tables: coordinate transformation belongs primarily to WB-01, playback to AI-04, and the combined controls/preference method needs one primary assignment with secondary links. Do not duplicate its count across AI-02 and AI-03. A component group may be covered by a shared case without receiving an additional independently counted case.

**Review note:** These are recorded executions, not a fresh consolidated run. The main tables remain unresolved until a current, consistently scoped run supplies their values. Retain the distinction between test methods, assertions, tested assets and repeated executions. The 100-video playback check must not be presented as 100 independent test methods or evidence of recognition accuracy.

## WBT-04 — Manuscript wording for Section 4.5.5

### Interpretation of Software Testing Results

Software testing and model evaluation address different aspects of ATLAS. The automated tests examine whether implemented functions and component interactions produce their defined outcomes under the tested conditions. Recognition accuracy and word error rate measure the performance of trained recognition components and sequence processing on the stated evaluation inputs. English-output assessment examines the quality of generated language.

A passing software test therefore supports the behavior exercised by that case; it does not establish that every sign will be recognized correctly or that every possible execution path has been tested. Likewise, successful reference-video playback establishes media compatibility under the tested conditions and does not establish recognition performance on those videos.

Test outcomes are interpreted alongside their execution environment and scope. Simulator-based interface checks support the reported layout, control and playback behavior, while device measurements support the separately described on-device workload. Software test execution times are not used as substitutes for model-inference or application-processing measurements.

**Review note:** Keep this short interpretation after the tables. No additional comparison chart is proposed. The existing participant-evaluation provisions remain intact.

## WBT-05 — Proposed appendix wording and format

### Automated Test Case Register

The test case register links the component summaries to their individual automated checks. Each record identifies the relevant function or behavior, test conditions, expected outcome, observed outcome and execution status. The source revision and execution environment identify the implementation to which the result applies.

A test case has one primary reporting group. Secondary component links may be recorded where a case exercises more than one responsibility, but those links do not increase the total number of independently counted cases. Parameterized inputs and repeated assertions are described within the case record. Retests retain the same case identifier and include their execution date and result.

| Record field | Required entry |
|---|---|
| Case ID | Unique identifier retained across retests |
| Primary group | WB-01–WB-07 or AI-01–AI-05 |
| Secondary links | Other component groups supported by the same case, where applicable |
| Implementation reference | Test file, test method and function or component examined |
| Test condition | Prepared inputs, starting state and relevant boundary or invalid-input condition |
| Expected outcome | Output, state change, rejection behavior or comparison criterion |
| Actual outcome | Recorded observation or assertion result |
| Status | Passed, failed or skipped; include the reason for a skip |
| Execution record | Run identifier, date, source revision and evidence location |
| Environment | Test framework, runtime, simulator or device, and required model or asset version |

**Review note:** Populate the register from the selected test execution. Do not manufacture detailed actual outcomes from method names alone. Add a short summary of failures and corrections when supported by the recorded run history.

## Evidence register — editorial only

### Current inventory and counting boundaries

| Source | Observed inventory | Reporting treatment |
|---|---|---|
| `ATLAS/python/tests/` | 245 test methods across 24 test files | Select cases relevant to the documented implementation. The inventory also contains data-preparation safeguards, alternative models and development-only checks. |
| `ATLAS/mobile_app/slt_mobile_app/ios/RunnerTests/RunnerTests.swift` | 22 test methods | Classify each method before counting. The file includes internal assertions, application integration checks, replay routines and profiling methods. |
| `ATLAS/mobile_app/slt_mobile_app/test/widget_test.dart` | 7 cases: 5 widget tests and 2 unit tests | Map interface cases to AI groups and data-handling cases to WB-07; retain exact IDs in the register. |
| `ATLAS/mobile_app/slt_mobile_app/android/app/src/test/kotlin/com/example/slt_mobile_app/live/LiveAlignmentTest.kt` | 2 tests | Candidate evidence for WB-04; do not imply that these establish complete platform equivalence. |
| `ATLAS/mobile_app/slt_mobile_app/macos/RunnerTests/RunnerTests.swift` | Empty `testExample` scaffold | Exclude from substantive testing totals. |
| `ATLAS/mobile_app/slt_mobile_app/ios/LiveReelHarness/speed_checks.swift` | Deterministic checks and repeated microbenchmarks | Treat correctness comparisons separately from timing repetitions; do not count loop iterations as independent tests. |
| `scripts/validate_ios_stage2_preprocessor_v17.swift` | Standalone preparation/inference diagnostic | Verify relevance to the selected pipeline before using it; it is not an additional XCTest method. |

Do not publish a combined “276 tests passed” claim from this inventory. A source count is not an execution result, and the inventories include different kinds of checks.

### Representative implementation mapping

These are starting points for the register, not a complete selection or a claim that every historical implementation remains selected.

| Group | Representative evidence |
|---|---|
| WB-01 | Python `test_normalization_keeps_missing_nodes_exactly_zero`, `test_resampling_preserves_binary_presence`; Swift `testLandmarkCaptureRotation` |
| WB-02 | Swift `testLatestFrameDropsBacklogAndInvalidatesStoppedWork` |
| WB-03 | Swift `testLiveHandEncoderBatchMatchesSerial`, `testFreshOverlayPreservesModelFeatures`; Python multimodal input checks |
| WB-04 | Python `test_low_confidence_is_not_emitted`, `test_stream_plus_flush_equals_clip`; Swift `testWordModeBridgesRestDipInsideSign`; Kotlin `delayedClosureDoesNotRewindWordEndpoints`, `configuredBoundaryDipKeepsWholeGestureCandidate` |
| WB-05 | Swift `testFinishPoseRightOpenLeftFistAboveShoulders`; finalization/reset assertions within `testStage3MultiSentenceModel` |
| WB-06 | Swift `testStage3MultiSentenceModel`; Python `test_model_receives_how_you_together_across_long_pause`, `test_coreml_overlong_input_falls_back_without_silent_truncation` |
| WB-07 | Flutter `history rows are described like the desktop shell`, `native inference result contract parses single and phrase output` |
| AI-01 | Flutter application-entry, page-navigation and manual-navigation widget tests |
| AI-02 | Flutter `every manual section lays out on a phone`; Swift `testCameraLayoutsLandmarkPreferenceAndReferenceOverlay` |
| AI-03 | Preference persistence and reference enlargement/dismissal assertions in the same Swift controls method; secondary mapping, not another test count |
| AI-04 | Swift `testAllPracticeReferencesPlayAndLoop` |
| AI-05 | Flutter `session detail lists recognized signs and the timeline` |

### Issue to resolve before consolidated execution

`testLiveAndPracticeKeepLandscapeLayout` currently asserts that portrait orientation is unsupported. The current controller returns `.allButUpsideDown` and prefers portrait, while `testCameraLayoutsLandmarkPreferenceAndReferenceOverlay` checks the newer behavior. The older assertion conflicts with the implemented requirement. Update the obsolete expectation while retaining its useful landscape checks, then record the resulting execution. This draft does not modify test code.

Native replay and profiling methods require separate treatment. In particular, `testStageProfile` measures processing; `testReplaySavedTakesOnDevice` and `testSimulatedLiveReplayOnDevice` depend on saved recordings and can skip when inputs are absent. Their presence does not establish a passed correctness test or supply a recognition-accuracy result for the paper.

### Reference-paper comparison

The two accessible DIRTX versions summarize white-box testing in nine application-module rows, with columns for test ID, module, functions, time and remarks. ATLAS adopts the compact component-summary approach while separating internal correctness checks from interface/integration checks. Per-case timing is omitted from the proposed main tables because processing speed already has its own evaluation.

- [DIRTX manuscript](https://drive.google.com/file/d/133LxZ2C8tMTXYDaepDg2usDbT1uBDnYW/view), printed pages 39–40.
- [DIRTX main paper](https://drive.google.com/file/d/1FRBKXcaEEM_CVew63w0IbZWeFGceJhu_/view), printed pages 39–40.

Denver's shortcut returned an access error; Paula's folder did not expose an accessible paper. Neither was used as a reviewed reference. This comparison is editorial background, not a proposed addition to the manuscript's literature review.
