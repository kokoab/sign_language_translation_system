# Mobile camera controls and practice reference playback — 2026-10-13

Implemented in the canonical iPhone application source. No physical phone installation has
been performed for these changes.

## Behavior

- Live and Practice open in portrait, allow either landscape orientation, and show
  “Landscape recommended” in portrait. Controls adapt to the available width.
- Live starts with the front camera and has a front/back switch. Practice remains front-only.
  Model input stays unmirrored; front preview is mirrored and rear preview is not.
- Landmarks are transformed from the exact captured frame rotation back to sensor coordinates,
  then mapped by the preview layer, which accounts for preview rotation, mirroring and cropping.
  Stale overlay work is invalidated when camera geometry changes.
- The landmark button appears in both modes. Its shared preference defaults on, persists,
  and only hides drawing; recognition continues.
- Tapping the Practice reference moves the same playing video layer into an enlarged overlay.
  Only video and X appear there. X or the backdrop closes it. No playback controls were added.
  Practice does not judge signs or automatically advance while that overlay is open.
- Reference loading awaits asset playability/duration, cancels superseded loads, and cleans up
  old looping queues. Failures are surfaced instead of silently hiding the reference.
- Live recordings split at rotation/camera changes so changed-size frames are not silently lost.

## Playback investigation and media repair

The user reported certain unknown glosses failing on a real iPhone. All 100 original clips
exist, decode completely with macOS AVFoundation, and play/loop in the iOS 26.3 simulator.
The exact real-device failure was therefore **not reproduced** here.

All originals used MPEG-4 Part 2 Simple Profile. Twenty exceed 30 fps or 2.5 Mbps, outside
Apple's older published MPEG-4 envelope. This is a compatibility hypothesis, not proof of
which limit the affected modern iPhone hit. References were converted to H.264 Main level 3.1,
yuv420p, CRF 8, with fast-start metadata; the preview builder now emits that encoding too.
Apple's older format specification: [iPhone 6 technical specifications](https://support.apple.com/en-us/111954).
The asynchronous loader follows [AVPlayerLooper asset-loading guidance](https://developer.apple.com/documentation/avfoundation/avplayerlooper/init(player:templateitem:timerange:)).
Landmark mapping uses [AVCaptureVideoPreviewLayer coordinate conversion](https://developer.apple.com/documentation/avfoundation/avcapturevideopreviewlayer/layerpointconverted(fromcapturedevicepoint:)).

All 100 references preserve the selected source, source hash, label, frame count, dimensions,
and duration (within 1 ms). Original MPEG-4 bytes/catalog are retained in `before/media/`.
No raw dataset, source selection, model, decoder, training gate, or protected test changed.
Updated media and matching catalogs are synchronized across Flutter, native and desktop
preview directories. The bundled simulator app's 100 media hashes were checked as well.

Decoded-frame SSIM: minimum **0.993861**, mean **0.997013**, all 100 above 0.99.
CRF 14 initially missed that quality gate on ANGRY (0.987780), so quality was raised.
A subsequent apparent GIVE failure was a comparison-timebase bug; normalizing both streams
to the same timebase before comparing corresponding frames corrected it (0.997520).
The checker now applies that normalization to every clip.

The prior 100/100 recognition check applies to the original MPEG-4 preview bytes and original
source videos. The new H.264 pixels have **not** been re-scored for recognition; catalog
metadata explicitly preserves that boundary. Playback testing is not recognition validation.

## Validation

- Preview builder tests: **16/16 pass**, including actual H.264 synthetic video export,
  unchanged frame count and aspect ratio.
- Flutter widget/navigation/manual tests: **7/7 pass**. The old forced-rotation text expectation
  initially failed and was updated to the new recommendation.
- Final iOS simulator XCTest methods: **3/3 pass** in 34.835 s:
  all **100 H.264 references play and loop**, inverse capture rotation at all four right angles,
  portrait/landscape controls, shared landmark persistence, Practice without camera flip,
  enlarged reference dimensions, X dismissal, and outside dismissal.
- Native layouts were rendered and inspected at 320×568, 390×844 and 844×390 points.
  The small Practice caption was changed to wrap rather than truncate. Stored layer snapshots
  omit camera/video compositing and glass effects; their black video areas are not playback evidence.
- Simulator Core ML warmup logs unsupported GPU backend errors; these UI/media tests do not
  assert model inference success or camera performance in the simulator.
- Two initial `/tmp` builds failed Xcode/Core ML output sandbox permissions. A fresh build under
  `/Users/frnzlo/slt_mobile_controls_build` succeeded without changing source/model permissions.
- Xcode reported `TEST SUCCEEDED` and exited0. Its optional simulator diagnostic collection
  stalled after test completion; only that diagnostic subprocess was stopped. Structured
  test results are saved in `ios_test_summary.json`.
- Repository-wide `git diff --check` passed. Final scoped check is recorded separately.
- Unsigned iPhone Release build: **succeeded**. All100 bundled media hashes match; no XCTest bundle shipped.
  Build: `/Users/frnzlo/slt_mobile_controls_device/Build/Products/Release-iphoneos/Runner.app`.

## Changed source

- `ATLAS/mobile_app/slt_mobile_app/ios/Runner/LiveReel/LiveReelViewController.swift`
- `ATLAS/mobile_app/slt_mobile_app/ios/RunnerTests/RunnerTests.swift`
- Flutter `lib/shell/{app_shell,practice_page,manual_page}.dart` and `test/widget_test.dart`
- `scripts/build_gloss_examples_v17.py`, `test/test_build_gloss_examples_v17.py`
- 100 preview MP4s plus catalog in each existing Flutter/native/desktop preview directory.
- Current-state and mobile-deployment decision/history records; artifact size index.

## Next physical-device check

Open Live/Practice on the affected iPhone, confirm portrait-first presentation and rotation,
follow hands/face through portrait and both landscape directions, switch Live front/back,
check mirroring and recording segments, toggle landmarks and relaunch, and play several
Practice references including ANGRY, GIVE, READY, KNOW and WHEN. Enlarge/close references
both ways. Confirm actual video playback and camera alignment before treating the original
real-device symptom as closed. Installed releases remain unchanged.

## 2026-10-13 — Portrait placement verified; not installed

Practice reference is top-center; round/sign card is bottom-center 12pt above centered
Skip, with retry to its left. Live portrait toolbar and bottom controls are centered as
groups. Landscape keeps its previous positions. Canonical controller and existing
RunnerTests layout check changed. Focused native test passes (1/1, 1.957s) across
320×568, 390×844 and 844×390; no Auto Layout conflict logged. Reviewed narrow Practice
and standard Live screenshots. Unsigned iPhone Release rebuild succeeds. Repository-wide
`git diff --check` passes. Evidence: mobile_camera_controls_20261013/portrait_layout_test.log,
portrait_iphone_build.log and portrait_followup_screenshots/. Layer snapshots exclude
composited camera/video and glass; no physical-iPhone installation or verification occurred.
Next safe action: check this layout and existing camera/playback changes on the real iPhone.


## 2026-10-13 — Updated camera controls and portrait layout installed on angelo iPhone 13

User explicitly requested installation. Signed canonical Release build succeeded with the
existing development team. First install failed with CoreDevice 3002 / IXRemoteErrorDomain 6
(connection invalid); retry succeeded in place on 00008110-00111D1A0130A01E. devicectl then
successfully launched com.kokoab.sltMobileApp. No app removal/data reset was performed.
Evidence: artifacts/reports/mobile_camera_controls_20261013/iphone_signed_build.log,
iphone_install.json (failed attempt), iphone_install_retry.json and iphone_launch.json.
Current camera/portrait/reference changes are now installed on angelo iPhone 13; installation
and launch do not establish camera alignment or resolution of intermittent reference playback.
Next safe action: user checks portrait positions, rotation/landmark alignment, front/back
switching, remembered visibility, reference playback and enlargement on this physical phone.


## 2026-10-13 — Single-row Live toolbar checks passed

Focused native layout and capture-rotation checks pass 2/2 (1.874s total), covering portrait
320×568/390×844 and landscape844×390. Assertions verify Live buttons share Back row and both
screens lack Diagnostics/Letters controls. Narrow portrait screenshot reviewed; all visible
buttons retain44pt touch targets. Flutter tests pass7/7 and repository diff check passes.
Updated remaining obsolete Gear help text after initial signed build; final signed build is
in progress before installation. Evidence: topbar_test_retry.log, topbar_flutter_test.log,
topbar_diff_check.log and topbar_screenshots in mobile_camera_controls_20261013 report directory.


## 2026-10-13 — Single-row toolbar and removed toggles installed

Final signed Release build passed and installed in place on angelo iPhone13
(00008110-00111D1A0130A01E); devicectl launch succeeded. Live has one top row for Back,
LIVE/status, landmarks, flip and Help. Live/Practice have no Settings or Letters button.
The manual Letters override is false in both modes and no saved preference can re-enable it;
existing sign-triggered spelling behavior remains. App data was not reset. Updated help
text/diagrams shipped. Evidence: topbar_signed_build_final.log, topbar_install.json and
topbar_launch.json in artifacts/reports/mobile_camera_controls_20261013/.
Next safe action: verify updated toolbar and Practice behavior on the connected real phone.

