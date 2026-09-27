# Recognition diagnosis — 2026-09-28

Discussion and diagnostic testing only. User reports U->R, desktop HE misrecognition
(replacement forgotten), HUNGRY->PLEASE, and requests faster recognition/commit without
accuracy loss. Existing HELLO->CB and phone overlay/portrait/Finish issues remain in scope.
No app changes, training, checkpoint selection, deployment or official Citizen test access.

## Protocol

`run_probe.py` runs the frozen v3 Core ML desktop runtime with fresh Apple Vision on
the same 12 U and 12 R validation clips at 20, 10 and 30 sampled frames/s. Membership
matches the first usable clips selected by the existing letter evaluator; filenames,
session labels, raw outputs and rates are retained in `rate_probe.json`.
The diagnostic permits single-letter output (minimum=1); normal app spelling requires
at least two letters. Exact success requires only the correct single-letter token.
This is an offline rate-sensitivity check, not measured iPhone throughput or a test
of connected natural fingerspelling. Changing rate also changes auxiliary detection
cadence and the physical duration of frame-count-based decoder rules.

HE and HUNGRY are checked with the same Core ML recognizer on complete cached isolated
validation spans and their first 35%, 50% and 65%. This separates identity evidence
from live boundary selection, but does not reproduce the user's unlabelled attempts.
Only Citizen validation is accessed; the official test stays sealed. Other caches are
existing SemLex/local validation. Local alphabet roles are session-separated, with
signer identities unestablished; they are not independent phone accuracy evidence.

Nine existing segmental runtime tests pass (`python -m unittest
test.test_segmental_runtime_v17 -v`). These cover decoder mechanics, spelling flush,
slot preservation and boundary parity, not camera recognition accuracy. Existing
Flutter tests check UI/contracts; they do not validate live letter or word recognition.

## Findings from source and existing evidence

Fresh 72 video replays (same 24 clips, three rates):

| Sampled frames/s | U exact | R exact | Combined exact |
| --- | --- | --- | --- |
| 10 | 9/12 | 8/12 | 17/24 |
| 20 | 11/12 | 8/12 | 19/24 |
| 30 | 7/12 | 7/12 | 14/24 |

At 10 Hz, U_12.mp4 and U_14.mp4 become R; both were correctly U at 20 Hz.
Four named-session R clips become U at 20 Hz (also at 10 Hz). At 30 Hz, extra
letters/duplicates appear (UK, UU, RU, RR, UR). This reproduces rate sensitivity,
including U->R, without proving the exact cause of the user's phone attempt.
Raising the sampling cap alone failed this bounded check. The remaining U_13.mp4
becomes K at all three rates. No rates or policies were changed in the application.

The first runner completed all 72 rate trials, then its separate cached-word stage
hit a diagnostic input-container error (dictionary instead of the recognizer's
tuple contract, KeyError 0). Corrected runner resumes with `--he-only`, preserving
the completed rate trials. This error did not come from the app or model.

Corrected full-span word check completed (57 spans, 171 prefix predictions):

- HE word-only top-1: Citizen 4/4, SemLex 1/1, local 34/38 = 39/43.
  Four complete local spans predict ASK, I, WHERE and THEY. On 23/43 HE spans,
  the letter head exceeds .5; 21 of those still have HE as the word head's top-1.
  This demonstrates conflicting word/letter evidence on the identical input, not
  merely a missing vocabulary class. Most false letter candidates are Z or D.
  Stream-level arbitration can choose different spans, so this is not a 23/43
  measured live failure rate.
- HUNGRY word-only top-1: Citizen 4/4, SemLex 10/10 = 14/14 (no local cache rows).
  No complete HUNGRY span passes the letter threshold. SemLex
  HUNGRY/W0uAhfTl6UZQkkgoVb0n predicts PLEASE at 50% and 65% prefixes (.542/.589),
  then HUNGRY on the full span (.806). Prefix PLEASE is below the .9 early-commit
  threshold but above the .5 ordinary decoder identity threshold; premature segment
  closure is a plausible failure route. It is not proof of this user's live cause.
  Other HUNGRY prefixes include MY/I/EAT; normal full-span success does not establish
  continuous recognition or independent phone generalization.

Completed data checks: 72 distinct (rate, clip) rows, identical 24-clip membership
at each rate, 43 HE + 14 HUNGRY rows. No new recognition policy has been tried or
promoted. Actual on-device confirmation and labelled user failures remain open.

- HE is a supported class (index 4, Citizen raw HE, ASL-LEX F_02_044) in both runtimes.
- Existing prefix audit: HE prefixes become ASK in 33/129 fraction-level predictions;
  HUNGRY prefixes become MY in 9/42. These are correlated prefixes, not independent
  full clips. ASK and MY already have an early-commit guard. This does not identify
  the user's unknown HE output or explain HUNGRY->PLEASE.
- Boundary lookahead is six processed frames: nominally .30 s at 20 fps, .60 s at
  10 fps, .20 s at 30 fps. Additional decoder/compute/display time follows. The
  phone's 20 fps value is a sampling target; historical local spans ran mostly 6–11.
- Current replay latency uses decoder-estimated end times, not independently labelled
  human sign endings. Its offline loop also does not reproduce phone frame drops.
- Word/letter competition can suppress words using letter NONE probability. A high
  letter confidence alone does not establish correct identity or true fingerspelling.
- FP16 image encoding previously changed outputs (203 vs 205 correct tune words);
  that speed shortcut does not satisfy an unchanged-accuracy requirement.

## Recommended fixes and acceptance gates

1. Profile capture, Vision, hand encoding, span scoring, display and spoken output
   with shared source timestamps, queue ages and gap-reset counters on the phone.
   Move visual tracking updates ahead of long recognition work; make cached points
   expire; check scheduling contention. Preserve recognition feature contracts.
2. First optimize exact computation/scheduling while keeping the trained 20 Hz
   semantics, weights and acceptance rules. Prove tensor/logit and output parity for
   implementation-only optimizations. Do not simply raise FPS, shorten lookahead,
   lower thresholds or duplicate stale frames to claim throughput.
3. Inspect U/R (and U/V/H/K), HELLO/letters, and HUNGRY/PLEASE/HE confusion evidence
   before changing classification. Test calibrated word-versus-letter arbitration
   and ambiguous-letter deferral as candidates; measure deletions and extra delays
   as well as fewer wrong commits. Never hard-map R to U or PLEASE to HUNGRY.
4. Preserve provisional spelling display and next-confirmed-word lock; distinguish
   letter recognition latency from the >2.5 s idle run-flush delay. Test genuine
   doubled letters, slow holds, fast runs, words interleaved with names, and J/Z.
5. Restore one-second two-open-palm Finish with a latch/progress and gesture exclusion;
   verify no false finish during signs. Support portrait and landscape with matched
   mirroring, rotation and preview transforms. Save attempt boundaries in history.

For a candidate, compare the same recordings before/after on desktop and the real
iPhone: isolated targets, connected phrases/names, rest/transitions, portrait/landscape,
both hands, realistic distance/light, and a sustained thermal run. Keep calibration
recordings separate from confirmation. Report substitutions/deletions/insertions,
per-class errors, whole-spelling exact accuracy, actual processed FPS, frame age,
and median/p90 sign-end-to-visible-commit latency using reviewed sign boundaries.
Require retained correct baseline cases and no increased false commits on the fixed
confirmation set before promotion; small-set parity cannot guarantee universal accuracy.
