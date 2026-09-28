# Live corrections and landmark tracking — 2026-09-29

## Current changes

Live and Practice retain the original landscape-only native layout. Home/history/glosses
keep their prior orientation. Camera output remains upright/unmirrored1280×720; the
full-screen preview mirrors once and uses aspect fill, with identical coordinate mapping
for landmarks. No portrait page redesign remains.

The old controller delivered camera frames and ran recognition on the same serial queue.
Even publishing the pose before recognition could not start the next hand detection until
recognition returned. The replacement uses an independent capture/display Vision queue.
It keeps one newest pending recognition frame, replaces stale pending frames, and
reschedules work so Stop/Reset/Finish can interleave. Start remains required in Live.
Display tracking targets20Hz, with body/face every4displayframes. Model Vision keeps its
original every8processedframe body/face contract. Rotation/stop/reset invalidate obsolete
frames and display results. Two-hand one-second Finish remains enabled in Live.

The synchronous every-frame body/face candidate was rejected: physical iPhone median
Vision5.62→17.83ms, despite exact feature parity. It is not the production display path.

## Validation

- Physical iPhone13: landscape Live/Practice controls, latest-frame queue lifecycle, and
  geometry/provisional-spelling tests all pass (`landscape_queue_test.log`).
- Physical iPhone13 short recorded-input concurrency test:120displayframes at a nominal
  20Hz input schedule versus48recognitionframes; display processing median12.30ms,
  p95 29.61ms (`independent_tracking_test.log`). This is not sustained live-camera FPS,
  preview alignment, power or thermal validation. The printed21.61Hz interval statistic
  excludes the first cold detection and must not be presented as sustained20FPS proof.
- Native Mac harness compiles. Native frame-gate checks cover latest-frame replacement,
  stop/restart and rejection of a frame from before rotation (`frame_gate_check.swift`).
-31Python runtime/integration/boundary tests pass; after removing unused rejected
  experimental APIs,15focused runtime tests pass again.

## Recognition changes retained

Conservative pair-only visible geometry resolves G/Q direction and U/R finger crossing.
The pair probability mass and all other logits remain unchanged. Missing/ambiguous
geometry abstains; at least3visible middle-span frames and80%agreement are required.
Existing1271session-separated letter validation clips:1099→1129correct,30fixes,
0regressions (`full_letter_check.json:summary`). This is full-span validation, not an
independent signer/iPhone accuracy claim. Prior78streaming comparisons fixed one G/Q
and four R/U errors; U→K and N extras remain.

Pending spelling stays open while hands remain visible and locks at the next lexical
word or Finish. With hands absent, the idle timer measures from the last sign end rather
than adding decoder delay. A lexical word may replace provisional overlapping handshape
letters only when it starts at the beginning of the same run; a late partial overlap must
not erase an established name. This handles CB then HELLO with the same onset, but does
not guarantee HELLO when the word decoder never recognizes it.

Both histories were checked. The latest available desktop session was230119_650297;
phone session224024. Newly reported HOME/YESTERDAY/YOUR attempts were not attributable
to saved intervals. Desktop now logs preview changes. Phone logs preview changes and
atomically autosaves every5seconds, rather than only on exit.

Rejected confidence rescue, letter-segment cost, deferred word-decoder letters and
whole-span rescue candidates are preserved in this report folder; none is enabled.
Several recover a word example while corrupting ANGELO or other spelling. Rejected
experimental APIs were removed from the production runtime. Model weights are unchanged;
no training or official Citizen test access occurred. HE/HOME/YESTERDAY/YOUR and
HUNGRY/PLEASE word-versus-letter errors still require further model/decoder work.

## Delivery

Final signed Release (`ENABLE_TESTABILITY=NO`) build succeeded; installed and launched
on physical iPhone13 with devicectl (`final_release_build.log`, `install.log`, `launch.log`).
Final paired replay completed: 93 clips / 4,528 frames. Of the 91 clips with explicit
sequence targets, exact sequences improved from 70 to 75: five fixes and zero previously
correct sequences regressed. Two user ANGELO excerpts are qualitative, not fully annotated
sequence tests. One improves QILO→GILO; the other retains the same output. An incorrect
HOME→OM result disappears through the absent-hand timer, but HOME is not recovered.
See `final_replay_summary.json`. Three Flutter tests pass; final scoped diff and whitespace
checks pass. Original landscape layout functions match the baseline byte for byte.

Files changed in the main repository: `active/v17/segmental_runtime_v17.py`,
`scripts/live_segmental_v17.py`, `test/test_segmental_runtime_v17.py`, current-state/history
documents and this report directory. Mobile changes are under
`/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios/`: `LiveReelViewController.swift`,
`LiveReelCore.swift`, `LiveReelEngine.swift`, `LiveReelDecoder.swift`, `LiveReelApp.swift`
in `Runner/LiveReel/`, plus `RunnerTests/RunnerTests.swift`.

Remaining verification: live-camera visual alignment while moving/rotating, sustained
thermal/frame-rate behavior, independent iPhone recognition evaluation, and the remaining
word-versus-letter/N failures. Do not represent this bounded regression set as a guarantee
that no unseen input can regress.
