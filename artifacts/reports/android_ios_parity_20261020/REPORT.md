# Android current-iOS behavior port — 2026-10-20 PHT

User requested Android match the current iOS app. Android keeps its separately
trained MediaPipe/TFLite recognizers, boundary model, performance mode selection
and vocabulary. No training, dataset edits or protected sign-test evaluation.

## Changes

- Ported the five-network Live English caption engine and its timed caption policy
  to Kotlin, using byte-identical iOS caption assets. T5 no longer loads in normal
  Android app startup; legacy code/model assets remain preserved.
- Live captions update from committed signs; Finish flushes pending recognition,
  finalizes English, saves the sentence and speaks once. Speak again repeats it.
- Matched the current right-open-palm/left-fist Finish pose, above fresh shoulder
  landmarks for one second, with progress/latching. Practice never uses Finish.
- Live/Practice opens portrait and permits rotation; Live can flip cameras.
  Preview/analysis share a CameraX viewport; the overlay uses the actual lens mirror.
- Added persistent Show landmarks / Show signs, guide and first-Live tips,
  framing outline and delayed lighting/person/shoulder/hand-edge advice.
- Practice loads the current shared reference clips asynchronously, loops at 0.75×,
  supports tap-to-expand and cancellation when targets/screens change.
- Added Android native settings, History Speak again/Delete/Clear; shared Flutter
  shell now exposes these supported Android controls. Finish-flushed signs remain
  in stored history arrays. Session writes/deletes are serialized and stamps validated.
- Optional silent Live video saving defaults off. CameraX records takes to app Movies/
  live_reel_videos; deletion follows session stamps. Unsupported simultaneous camera
  stream combinations fall back to recognition and show a recording-unavailable notice.
- Added `ATLAS/mobile_app/build_android_mirror.sh` for internal-disk builds. Existing
  iOS helper and compatibility symlinks are preserved.

## Deliberate platform differences

NAME spelling remains unavailable: there is no compatible Android letter model.
Android settings do not present that switch. Recognition models and decoding settings
remain Android-specific; this feature port establishes no new signing accuracy or
recognition parity with Apple Vision. Native UI follows current iOS behavior through
Android controls, rather than claiming pixel-identical rendering.

## Validation

Receipts are adjacent to this report. Final build/test summary is in validation.json.
All 352 desktop/iOS timed caption fixture streams (91,445 events) are compared for
exact output text, including provisional updates, pauses, finalization, names and resets.
Focused JVM tests also cover shoulder freshness, Finish hold/latch/release and framing
priority/delay. Flutter analyzer and the existing 10 widget/contract tests cover the
shared shell. Model asset hashes are in model_assets.json. Source/mirror/APK hashes
are in validation.json. No physical Android phone was connected (`adb devices` empty).
Camera alignment, recording compatibility, reference playback, device speech and
sustained caption/recognition latency still require real-device verification.

Initial unqualified Gradle `testDebugUnitTest` also started unrelated Flutter plugin
Robolectric tests; stopped that run and used `:app:testDebugUnitTest`. No passing
claim is made for unrelated plugin suites. An initial preservation check assumed
TFLite assets were in Git HEAD; they are ignored, so the corrected check uses original
2026-10-04 export bytes (task assets compare with HEAD). All nine payloads match.

Camera recording implementation follows the official CameraX API and handles the
hardware-dependent Analysis + Preview + Video combination:
https://developer.android.com/media/camera/camerax/video-capture

## Next safe action

Install the debug APK on the target Android phone and check Live/Practice/History,
front/back cameras in both orientations, Finish pose/speech, both recognition modes,
optional recording and session deletion. Measure physical-phone latency/thermals;
retain the default Android model selection until those measurements support a change.
