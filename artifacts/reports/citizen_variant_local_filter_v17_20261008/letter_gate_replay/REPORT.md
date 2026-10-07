# Gated letter sink for word mode — 2026-10-08

Problem: with letters off, the phone's word decoder lets the 26 letters compete for every span and then
hides letter wins. Confident words (WHERE 0.96–0.97, HOME, HEAR) were lost to hidden letters D/X/O/C.

Change: letters may compete for a word-mode span only when the recognizer's top word probability is
below a gate. Python `SegmentalRuntime(letter_gate=…)` (active/v17/segmental_runtime_v17.py) and Swift
`LiveSegmentalRuntime(letterGate:)` (LiveReelDecoder.swift); `LiveReelEngine.wordLetterGate = 0.95`
drives a separate word-mode decoder. Spelling/name modes keep the ungated dual runtime.

Citizen validation, 378 clips, phone view (letters hidden), Python reference, shared spans:

| Setting | Correct | Nothing | Wrong | Extra words |
|---|---:|---:|---:|---:|
| sink (before) | 311 (82.3%) | 30 | 37 | 68 |
| gate 0.95 (installed) | 315 (83.3%) | 26 | 37 | 68 |
| gate 0.9 | 315 | 24 | 39 | 70 |
| gate 0.8 | 316 | 22 | 40 | 73 |
| gate 0.7 | 317 | 18 | 43 | 76 |
| words only | 321 (84.9%) | 13 | 44 | 92 |

User take 20261008_174458_05: sink `HE WHERE YOU GO YESTERDAY YOU` (4 hidden letters) → gate 0.95
`HE WHERE WHERE YOU GO YESTERDAY WHERE YOU` (2). Peak RSS 2.18 GB flat (no leak). Run-to-run MPS
nondeterminism: sink equals the earlier replay's total but differs on 61 clips individually.
Device: testStageProfile (≈24.5–25 ms/frame) and finish-pose test pass; Release installed.
Backups: app_source_backup/ (hashes_before.txt).
