# Similar-sign failures on the phone: local data audit and finish gesture — 2026-10-08

User report: similar pairs (ASK/NEED, BIG/LANGUAGE, LESS/SCHOOL, GO/ANSWER, FAMILY/IMPORTANT,
WAIT/MAYBE, I/WE, ANGRY→NOW, CHILD→HEAR/GOODBYE; EASY, GOODBYE, HEAR, HOME, NEED, SAD, SMALL,
MAYBE hard to trigger), suspected local-data poisoning; two-handed signs trip the finish gesture.
Phone session `data/local/phone_session_pull_v17_20261008/` (26 min, mostly alphabetical walk
through the vocabulary; phone runs the August recognizer `SpanRecognizerV17LocalALettersB8FP16`).

## 1. Isolated validation: local adaptation fixes, not causes, most pair confusions

Pair confusions (A→B + B→A, Citizen+SemLex+local validation) before → after local adaptation:
BIG/LANGUAGE 21→2, I/WE 14→0, GO/ANSWER 6→5, LESS/SCHOOL 3→0, CHILD/HEAR 3→0, ASK/NEED 1→0,
GOOD/THANKYOU 22→13; ANGRY/NOW, FAMILY/IMPORTANT, WAIT/MAYBE 0 in both. Citizen/SemLex recall for
these classes is unchanged or better. Data: `confusions` script output in this conversation;
models 96.83 isolated, Variant C local branch, 96.83-chain recognizer, a7490409, August local
branch, phone recognizer.

## 2. But several local classes are a different sign or variant from the pinned Citizen class

Pre-local model (never saw local data) on local validation, with visual check (`sheets/`):

| Class | Pre-local local val | Read as | Visual finding (local vs Citizen) |
|---|---:|---|---|
| HOME | 0/29 | HELLO 25 | **Wrong sign**: all 158 train clips come from v16 `HOUSE` (two-hand roof); Citizen HOME = flat-O mouth→cheek |
| CHILD | 8/22 | YES, HEAR | **Different sign**: raised open hand near shoulder (wave-like); Citizen = flat hand patting down at waist |
| I | 8/70 | WE 13, YESTERDAY 12 | **Mixed**: fingerspelled letter I (pinky) plus ME pointing; Citizen I = ME |
| GOODBYE | 0/18 | HOT, DRINK | Compound GOOD+BYE starting at the chin; Citizen = plain wave |
| HEAR | 0/35 | LISTEN 24 | Bent index at the ear; Citizen HEAR2 = open hand at the ear |
| WHAT | 0/31 | NOW, SMALL | Both palms up shaking; Citizen WHAT2 = index across palm |
| BIG | 3/29 | LANGUAGE 19 | Flat hands facing, low; Citizen = clawed/L hands wide |
| SIGN | 4/34 | ANSWER, DIFFERENT | Mixed forms |
| ASK | — | — | Static raised index (no flick/bend); close to NEED's X hand |
| COME, WHY, WOMAN, TELL | 4–29% | | not visually reviewed |

Manifest flags: 2,088 local train clips are `current_v16_canonical_lineage_variant_unverified`
(DOCTOR, DRINK, EAT, GOODBYE, HEAR, HOSPITAL, HOW, I, NIGHT, SAME, THEY, WANT, WHAT, WOMAN); label
lineage differs for HOME←HOUSE, SAME←ALSO (visually mostly the Y-hand SAME), EAT←EAT_FOOD,
MAKE←MAKE_CREATE. Local validation has the same variants, so local validation accuracy cannot
reveal these mismatches. Training on them teaches two different forms per class (or the wrong
sign), which matches the live symptoms for HOME, CHILD, GOODBYE, HEAR, I/WE, BIG/LANGUAGE.

## 3. Live-only causes

- WE: the decoder commits I on WE's first point, then WE (I I WE I WE …) — segmentation.
- BIG→LANGUAGE committed at 0.87–0.95; CHILD previewed after GOODBYE/HEAR; FAMILY preceded by
  IMPORTANT×2 and a WAIT insertion; MAN and MAYBE → WAIT; NEED alternates with ASK.
- HUNGRY and 5 others have no local clips; the phrase recognizer covers only 15 signs.

## 4. Finish gesture

`LiveReelEngine` skips classification on every frame where the finish pose is seen (before the
1 s hold). The old pose (two separated upright open palms) is entered by two-handed signs.
Replay of take 17 (241 s, WANT…YOUR): old pose 40 frames in 11 runs (none ≥1 s), each frame
dropped from recognition; new pose 0 frames. Full validation scan: `gesture_false_positive_scan.json`.

New pose (user request): signer's right hand open and upright, left hand a fist, both wrists above
the shoulder line (last body detection ≤1.5 s old), held 1 s. Implemented identically in
`scripts/live_reel_stage1_v17.py` (+ `live_segmental_v17.py` call site) and the iOS app
(`LiveReelCore.swift`, `LiveReelEngine.swift`); Python tests 20/20 + 83 related pass; Swift
`testFinishPoseRightOpenLeftFistAboveShoulders` passes on the iPhone 13. Source backups and
pre-change hashes: `app_source_backup/`.

### Validation scan (Citizen val 378 + local val 2,896 clips, 20 Hz Apple Vision)

Old pose seen in 326 clips (2,722 frames), would fire Finish in 14 clips; concentrated in the
user's hard-to-trigger signs: local HOME 29/29 clips (3 would finish), SMALL 28/30, BIG 22/29,
SCHOOL 22/41, FAMILY 21/29, EASY 20/24, SAD 19/19, WAIT 18/37 (6 finish), ANGRY 13/19, MAYBE 7/22.
New pose: 0 frames in all 3,274 clips. Fist detector fires on fist signs (WHO .81, YES .77 of
frames) and 0 on open-hand Citizen classes. Release build with the new pose installed on the
iPhone 13 (2026-10-08); models unchanged (still August recognizer). Previous sources in
app_source_backup/ (hashes_before.txt).
