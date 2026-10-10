# MY/HAVE iPhone investigation — 2026-10-11

## Status

Physical iPhone 13 (`angelo`, iPhone14,5), iOS26.7.1. User authorized testing on the connected device. Testing is complete. **The two-model MY/HAVE candidate failed normal-app recognition checks and was rejected. The requested guarded pre-contact-HAVE rollback is installed and launched. MY is still unresolved.** XCTest completion by itself does not establish recognition correctness.

## What was tested

The October10 installed-source decoder, the earlier joint V2 candidate, and that candidate with a complete feature-sampling reset. All phone tests explicitly select the existing FP16 image encoder and recognizer with `.all` compute units. No weights, datasets, training gates, or protected tests were changed.

The extra LocalA recognizer is now an explicit, eagerly constructed dependency shared by all three word decoders. Its graph is warmed before processing. It is absent from the letters-only decoder. This removes per-decoder lazy loading and silent initialization failure from the prototype. Candidate files are preserved in this directory; diagnostic source edits are backed up in `source_backup/`.

Seven cached traces were replayed twice, in reverse case order. These are six phone-related MY/HAVE inputs and the PLEASE HELP I regression control, totaling5,624 processed frames per decoder. The full57.3-second take03 was replayed twice through Vision and the complete mode-switching engine, totaling2,294 scheduled frames. Cached traces exclude extraction/image encoding; full-video timing includes these, but excludes file decoding, resizing, UI, camera capture, and speech.

The phone also replayed59 labeled local validation videos (22MY,37LIKE), with body/face sampling explicitly starting at phase zero in both arms to isolate decoder changes. These are development controls, not an independent test set.

## Recognition results

| Input | Installed decoder | Joint candidate | Joint + complete reset |
|---|---|---|---|
| Exact saved first MY failure, both repeats | HELLO LIKE NAME | HELLO MY NAME | HELLO MY NAME |
| take11 HAVE count |10|10|10|
|09:55 HAVE count|9|9|9|
|Exact10:06 HAVE count|9|9|9|
|Full take03, MY around9.2s, first replay|LIKE|MY|MY|
|Full take03, MY around9.2s, second replay|LIKE|LIKE|MY|
|PLEASE HELP I control|PLEASE HELP I|PLEASE HELP I|PLEASE HELP I|

All other cached-trace hypotheses remained unchanged. First take11 HAVE→ANGRY remains an existing error. The user's deliberate LIKE switch is not precisely labeled; other LIKE outputs in that recording are not automatically counted as mistakes.

All59 labeled phone-control hypotheses are identical between the installed decoder and joint/reset candidate: MY18/22exact; LIKE21/37exact,23/37containingLIKE. Four MY and12LIKE clips are silent; two LIKE clips contain MY in both arms. This is a no-regression result on these controls, not perfect recognition.

## Confirmed reset defect

`LiveVision.resetTracking()` cleared hand assignment and signer state but retained `processed`. Body/face features are sampled every eighth processed frame. A1,147-frame replay advances the next starting phase by3.

Holding the phone video and models fixed, forcing starting phases0–7 gave:

- Phase0: MY at9.167s.
- Phases1,2,3,6,7: LIKE in that window.
- Phases4,5: no word in that window.

The Mac with explicitly selected FP16 encoding reproduced this pattern. Restarting `processed=0` with the other tracking state then gave eight identical phone hypotheses and MY at9.167s for all eight prior states. Two full-video repeats also became identical. The assertion-based reset test passed.

This establishes carry-over between sessions as a real source of repeat instability. It does **not** establish robustness to different frames selected during an active session.

## Normal app load exposes a remaining failure

An additional paced test uses the real `LiveReelShared.warm()` path, one shared engine, the resident warmed English model, the existing latest-frame queue, and concurrent overlay Vision. It feeds the saved video at its actual presentation-time pace.

The reset candidate still produced LIKE at9.2s in this test. It selected a frame at3.233s where the successful paced replay selected3.200s, plus differences at3.3,5.3,and9.3s. The3.2s difference is an auxiliary-feature frame near NAME. This motivates a controlled capture-side20Hz selection experiment; it is not yet evidence that that change fixes recognition.

The first Vision call took387ms, but the single omitted scheduled frame occurred after27.17s. Therefore that omitted frame cannot explain MY9.2s. An initial suspicion that the startup stall caused the earlier MY failure was corrected after inspecting the timestamps.

## Timing, residency, and thermals

| Workload | Installed-source baseline | Shared/reset candidate |
|---|---:|---:|
|Paced full app, frames|1,146|1,146|
|Median processing|26.55ms|25.90ms|
|90th percentile|45.94ms|46.90ms|
|99th percentile|63.01ms|63.29ms|
|Maximum|399.49ms|387.02ms|
|Ready process physical footprint|227.36MiB|238.24MiB|
|Peak sampled process physical footprint|261.50MiB|270.52MiB|
|Thermal state during run|fair|serious|

Different thermal/startup conditions prohibit attributing the small speed difference to the fix. Memory is sampled process physical footprint, not all system/Neural Engine allocations. The English model is loaded and warmed but does not render/speak during this test. This is a bounded recorded-video concurrency test, not an extended live-camera battery/thermal certification.

The initial decoder-only model load measured7.98s baseline and12.36s candidate, followed by17.6/29.4ms graph warm-up. A later already-cached candidate construction took270ms. Thus12.36s is a cold/cache-dependent observation, not a stable startup prediction. Normal shared-app candidate warm-up including English took13.49s. CoreML warm-up does not cover the observed cold Vision spike.

Offline full-video medians were10.64ms baseline,11.01ms joint,10.68ms joint/reset. These exclude concurrent overlay work and must not replace the paced numbers. Back-to-back unpaced tests raised thermals from fair to serious.

## Evidence and limitations

- `baseline_device.json`, `candidate_device.json`, `reset_candidate_device.json`: paired trace/video records, per-frame times, publication observations.
- `phase_device.json`, `phase_reset_device.json`: phone phase intervention and assertion-checked reset results.
- `controls_baseline_device.json`, `controls_reset_candidate_device.json`: all59 paired phone control outcomes.
- `shared_baseline_paced.json`, `shared_candidate_paced.json`: actual selected timestamps, words, processing costs and shared-app residency.
- The older Mac whole-video harness inherited the FP32 image-encoder default. New Mac phase tests and all phone tests explicitly use FP16. Do not describe the older whole-video results as FP16 phone parity. Cached exact-phone-input comparisons remain valid because they preserve the phone embeddings.
- Harness publication records and existing app `event.seconds` record when words leave the engine on the input timeline. `word.commit_seconds` remains the earlier identity-lock time; neither is a measured screen-presentation timestamp.
- Initial build required the existing development team and `ENABLE_TESTABILITY=YES`; one phase build used the wrong working directory; the first generated capture-grid test missed a closing brace. Corrected retries are separately logged. These failures occurred before their corresponding device experiments.

## Capture-side scheduling experiment (rejected)

Four paced phone passes, same shared engine, with capture-side20Hz selection off/on/on/off. The candidate includes the complete reset and prewarmed shared fallback. All four runs reported fair thermals.

|Pass|Capture selection|MY around6.37s|MY around9.2s|Frames|
|---|---|---|---|---:|
|0|off|MY|LIKE|1,147|
|1|on|silent|MY|1,143|
|2|on|MY|MY|1,142|
|3|off|MY|MY|1,146|

Capture-side selection did not establish a safe fix: one run loses an earlier MY, and the off control itself varies with scheduling. It remains a harness-only experiment; the capture callback was not modified.

A separate Mac replay explicitly using the actual selected phone timestamps reproduced the outcome change: the successful schedule gives MY9.167 with NAME entry/exit5.267/9.267; the failed shared-app schedule gives LIKE9.2 with NAME entry/exit4.767/8.767. Model and decoder are fixed. These regenerate Vision/crops on Mac, so this is controlled schedule evidence, not bit-exact phone feature parity.

The user-authorized pre-HAVE contact-closure rollback, retaining the existing one-way bent-finger HAVE/ANGRY guard and TIRED roll requirement, is now being checked on the phone. No new capture/clock heuristic is being promoted.


## Final rollback and deployment

The exact pre-contact rollback reintroduced old duplicate-HELP and word-order defects. It was tested, not released. The installed bounded rollback keeps the independent V2 bookkeeping repairs: identity locks at its original time; only publication waits for earlier words; earlier closures cannot replace the latest word; frame/time endpoints advance together. It removes the HAVE contact-closure rule and global identity-order delay, while retaining the original one-way hand-shape HAVE/ANGRY check and TIRED roll rule.

The deployed decoder contains no second recognizer. Engine, Core/Vision, Models, Stage3, AppDelegate, and RunnerTests match their pre-test sources. All diagnostic hooks and startup suppression were removed. The new capture-grid and auxiliary-clock/reset experiments are **not deployed**. Python runtime and model weights are unchanged; this deployment changes the phone decoder only.

Final native cached regression: all570case hypotheses are identical to the installed-source baseline, including103pair controls,378Citizen validation clips spanning100classes, and89development sentences. These pools overlap and are not570independent examples. Development sentence errors remain29/226 (12.83%WER). Protected tests were not used. The guarded rollback passes the phone one-HELP/order assertions and hand-shape/roll checks. HAVE counts return to5/8/4 on the three phone controls, as expected when discarding the contact rule.

The guarded rollback still produces LIKE for the exact first MY case and the MY9.2 case in the normal shared-app paced run. This is documented as an unresolved recognition error, **not** a successful MY fix. Its normal shared-app paced processing median is25.10ms, p90 45.76ms;1,147frames; process footprint230.30MiBready/260.45MiBsampledpeak; fair thermals. Two repeated full-video replays remain sampling-phase sensitive because the feature-clock experiment was not deployed.

Signed normal Release built at `/Users/frnzlo/slt_my_have_guarded_rollback_build/Build/Products/Release-iphoneos/Runner.app`, installed as `com.kokoab.sltMobileApp`, and launched successfully. It contains no XCTest bundle. Existing session/video data are preserved. `installation.json`, `launch.json`, `installed_app.json`, `guarded_rollback_source_hashes.json`, `guarded_rollback.patch`, and `FINAL_RESULTS.json` record provenance. Only the external iOS decoder source intentionally differs from the pre-test source hashes.

`verify_results.py` checks all570paired hypotheses, all59phone controls, the eight reset experiments, guarded rollback trace assertions, source restoration, and explicitly retained MY failures. `git diff --check` reports only the pre-existing whitespace at `scripts/app_shell_v17.py:632`.

Next safe action: deliberately labeled MY/LIKE recordings with capture timestamps and a trace covering NAME mode, then evaluate temporal/frame-selection robustness. Do not infer that all LIKE outputs in the mixed recording are errors or train on guessed labels. This result does not authorize a protected-test rerun or bypass any training-data gate.
