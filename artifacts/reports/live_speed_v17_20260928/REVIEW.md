# Exact-computation live speed update — 2026-09-28

User authorized faster recognition with regression checks; reported repeated ANGELO,
N/other letters, G/Q and HELLO/C in addition to U/R, HE and HUNGRY/PLEASE.

## Changes

- Python and Swift submit the existing hand crops together to the same FP32 Core ML
  encoder. Separate Swift pixel buffers preserve each crop until synchronous completion.
- Boundary features retain the full shoulder-reference history but calculate normalized
  nodes and velocity only for the final two rows consumed by streaming.
- Swift reads contiguous model outputs directly, retaining the padded-stride fallback.
- Phone publishes detected landmarks before crop encoding/decoding; skeleton path
  updates disable implicit animation. Unchanged HUD chips reuse their existing views.
- Sampling target remains 20 Hz. No weights, thresholds, image crops, precision,
  word/letter arbitration or spelling/commit delays changed.
- Concurrent one-second Finish changes in shared desktop/mobile files were preserved.

## Paired validation

`replay_speed.py`, `membership.json`, `paired_replay.json` reproduce the comparison.
78 clips: 12 each U/R/N/G/Q validation, all 12 Citizen validation clips for
HE/HUNGRY/HELLO, four existing phrase tuning clips, and two excerpts of the user's
ANGELO session. No protected Citizen test access or training. The letter validation
split is not certified signer independent. User excerpts are not frame-annotated.

Both arms consume identical precomputed Vision observations at the 20 Hz target;
before/after execution order alternates by clip. The comparison checks complete
output records including gloss, scores, sign boundaries and commit timestamps.

| Recognition processing on Mac | Before | After |
| --- | ---: | ---: |
| Frames | 3,652 | 3,652 |
| Median | 45.77 ms | 39.27 ms |
| p90 | 90.68 ms | 83.51 ms |
| Sum | 179.67 s | 156.41 s |

Zero changed output records across 78 clips; maximum embedding difference 0.
Median reduction is 14.2%. Timing excludes Vision, camera and UI, and some runs
overlapped build activity; this is not sustained full-camera FPS evidence.

Physical iPhone 13 XCTest passed: batch sizes 1/2/3 exactly equal serial embeddings,
and the empty batch is empty. Twelve alternating three-crop timing rounds measured
32.90 ms serial versus 30.55 ms batched (7.2% reduction). This deterministic-input
microbenchmark measures encoder work, not sustained live latency or thermals.
Evidence: `iphone_test_enabled.log`.

26 focused Python tests passed (temporal boundary, segmental runtime, app navigation).
Three Flutter tests passed. Native synthetic checks compare 54 unchanged results:
44 crop cases, five boundary histories, three contiguous array types and two strided
layouts. A 742-frame saved Swift fixture retained the expected words; cross-language
FP16 boundary differences are not exact equality claims. Release iOS build passed.

## History findings and limits

The desktop snapshot includes fs-QELO, fs-AN, fs-ASGEO, fs-GELO, fs-GJN,
fs-ASQELO and other outputs; it contains no exact ANGELO. At 64.74 s fs-CB overlaps
HELLO's span. This supports investigating word/letter arbitration and letter identity;
it does not justify hardcoded correction of names. Existing recognition errors remain
in both comparison arms. Replay equality is regression evidence for these fixtures,
not a guarantee that all real-camera behavior is unchanged.

This change does not shorten the spelling pause, fix portrait handling, or establish
accurate landmarks during body/face detection gaps. The earlier diagnosis found
that higher sampling rates could worsen letters and an incomplete HUNGRY prefix
could resemble PLEASE, so earlier commits need separate accuracy evaluation.

## Failed attempts and next checks

A cubic-resize SIMD experiment showed no useful gain and was removed. Initial
native compilation needed the batch API's options argument; fixed. Initial phone
test invocation lacked a team and then failed because Release disabled testability;
rerun with the signing team and ENABLE_TESTABILITY=YES passed. Production build
uses ENABLE_TESTABILITY=NO. No project-wide signing/testability defaults changed.

Signed production Release installed and launched successfully on the connected
iPhone 13 (`install.log`, `launch.log`); application data was not erased. The build
also includes the concurrent one-second Finish implementation. Real-camera gesture,
portrait and sustained-load behavior have not been validated by this install.
Large-artifact index regenerated; affected tracked files pass `git diff --check`.

Next: check the installed app under sustained live signing, including START/STOP,
Practice, overlay alignment and one-second Finish. Investigate word/letter competition
and segment boundaries separately before lowering commit delays. Desktop must restart
to load the changed Python modules.
