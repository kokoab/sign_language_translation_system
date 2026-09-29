# Signer lock on iPhone — 2026-09-29/30

User request: when several people are in frame, the hands, face and body should stay on one person.
Lightweight, no performance cost, automatic signer choice, automatic re-pick when the signer leaves,
and no regressions.

## Behaviour (`Runner/LiveReel/LiveReelCore.swift`, `LiveVision` + `LiveSigner`)

- The signer is picked automatically as the largest face (nearest the camera) and followed at every
  face/body frame by position (next face within two face widths/heights, size within 2x). If their face
  is not seen for 2 seconds, the signer is re-picked. Reset also re-picks.
- **With one real person in view, selection is exactly the previous rule** (first two hands in Vision's
  order, largest face, most confident body), including its quirks, because the recognizer was trained
  on it.
- **Lock mode** starts when another person's face is seen (beside the signer, at least 1.5 face widths
  away horizontally, and not covering a detected hand, since raised hands are sometimes found as faces). It stays on
  for 2 seconds after that face was last seen. In lock mode:
  - the face used is the signer's (none while it is hidden, instead of switching to the bystander);
  - the body used is the one whose nose/neck belongs to the signer's face;
  - each hand goes to the nearest person (head, shoulders, elbows and wrists from Vision's body pose,
    or face and torso estimates for a person seen only by face), and only the signer's hands are kept.
- Vision's hand request allows 4 hands instead of 2, so a bystander's hands cannot crowd out the
  signer's before filtering.
- `LiveVision.signerLockEnabled` (and `LiveVision(signerLock:)`) switches the lock off, restoring the
  earlier code path exactly; used for all regression comparisons below.
- The engine's recognition path and the display overlay both use `LiveVision`, so both are locked.

No new Vision requests or models: the lock uses face and body detections the app already runs (every
8th recognition frame, every 4th overlay frame) plus arithmetic.

## Regression checks (macOS, the app's own sources)

**Detection level** (`replay/main.swift`): the same frames at the app's 20 Hz go through the lock-off and
lock-on paths, and every frame's left/right hand, body and face are compared bit for bit.

| Set | Clips | Frames | Frames differing |
|---|---:|---:|---:|
| Previous live-correctness replay set (78 + 15 Citizen) | 93 | 18,977 | 2 (one frame of one recording, listed twice) |
| Additional Citizen validation clips | 150 | 8,088 | 0 |

The remaining frame (252.1 s of the user's recording, while standing up and out of view): a face in the
mirror on the right counts as another person, and the lock swaps a false hand at the top edge of the
frame for the real hand at the waist. Earlier versions of the lock also changed frames where a raised hand
was detected as a face, and frames with a spurious body at the frame edge. Both were corrections, but
they changed recognizer inputs, so the lock now leaves single-person selection untouched.

**Recognition level** (`ios/LiveReelHarness`, `videos` mode, full Swift engine with the app's FP16
encoder and `.all` units; `SIGNER_LOCK=0/1`): 238 clips (the recording listed twice counted once).
**All 238 produce identical word sequences**; exact-match 129/237 either way. One word differs in timing
only: GIVE at 253.4 s in the user's recording ends at 254.2 s instead of 253.8 s (the standing-up moment
above).

## Two-person effectiveness

`replay/main.swift pair`: 24 pairs of Citizen validation clips side by side (one at full size, the other
at 80%), each placement both ways (48 runs, 2,474 frames). Scored against the person the lock acquired.

| | Wrong-person hands | Wrong-person faces | Wrong-person bodies |
|---|---:|---:|---:|
| Lock off (previous behaviour) | 1,177 / 2,339 (50%) | 25 / 324 | 249 / 318 |
| **Lock on** | **69 / 1,307 (5%)** | **0 / 322** | **3 / 281** |

The lock keeps more of the signer's own hands (1,238 vs 1,162) and drops the other person's. The
remaining leaks are mostly moments when Vision returns neither a body nor a face for the bystander.

## iPhone 13

`RunnerTests.testSignerLockVisionCost` (bundled sign clips; 3 alternating rounds, 168 frames each):

| Detection step median | Lock off | Lock on |
|---|---:|---:|
| One person | 12.41 ms | 12.24 ms |
| Two people | 16.32 ms | 17.36 ms |

One person: no cost. Two people: about +1 ms (Vision processes the extra hands that the 4-hand limit
now allows). In the two-person composite all 54 locked hands came from one person (lock off: 66 and 40
split between the two). All 11 RunnerTests pass on the physical iPhone 13. The Release build was rebuilt,
codesign-verified and installed; saved sessions are intact.

## Changed files

- `ios/Runner/LiveReel/LiveReelCore.swift` (lock); pre-edit copy in `app_backup_before/`.
- `ios/RunnerTests/RunnerTests.swift`: `testSignerLockVisionCost`.
- `ios/LiveReelHarness/main.swift`: `SIGNER_LOCK=0` environment switch (pre-edit copy in `app_backup_before/`).

## Limits

The two-person checks use composited clips, not two real people filmed together; not verified with live
camera use. Only the iPhone app is changed; the desktop Python app still selects people as before.
A person standing directly behind the signer (face not beside them) does not start lock mode.
