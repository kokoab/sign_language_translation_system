# Citizen-variant local data experiment — 2026-10-08

User decision: for every confused class, the Citizen variant wins. Local clips of a different sign or
variant were removed from local training **and** local validation; Citizen/SemLex data, recipes, seeds
and selection rules are identical to the audited 96.83 chain (Variant C + approved phrase split).

Removed local train clips: {'ASK': 147, 'BIG': 119, 'CHILD': 113, 'COME': 159, 'GOODBYE': 108, 'HEAR': 157, 'HOME': 158, 'I': 208, 'SIGN': 143, 'WHAT': 113} (kept 11956); local val removed 293 (kept 2603). I keeps only ME clips (index to chest).
All rows below are scored on the same inputs: Citizen val 378, SemLex val 978, filtered local val 2603. Validation only; the lab held-out test and protected tests were not used.

## Accuracy by stage

| Stage | Chain | Citizen | SemLex | Local (filtered) | Mean of 3 |
|---|---|---:|---:|---:|---:|
| Local adaptation | current | 368/378 (97.35%) | 854/978 (87.32%) | 2558/2603 (98.27%) | 94.32% |
| Local adaptation | Citizen-variant | 363/378 (96.03%) | 856/978 (87.53%) | 2557/2603 (98.23%) | 93.93% |
| Multimodal fusion | current | 369/378 (97.62%) | 861/978 (88.04%) | 2558/2603 (98.27%) | 94.64% |
| Multimodal fusion | Citizen-variant | 363/378 (96.03%) | 866/978 (88.55%) | 2555/2603 (98.16%) | 94.25% |
| Phrase adaptation | current | 369/378 (97.62%) | 861/978 (88.04%) | 2553/2603 (98.08%) | 94.58% |
| Phrase adaptation | Citizen-variant | 362/378 (95.77%) | 863/978 (88.24%) | 2545/2603 (97.77%) | 93.93% |
| Interval recognizer | current | 367/378 (97.09%) | 864/978 (88.34%) | 2546/2603 (97.81%) | 94.41% |
| Interval recognizer | Citizen-variant | 362/378 (95.77%) | 865/978 (88.45%) | 2548/2603 (97.89%) | 94.03% |
| (phone today) | August local_a | 360/378 (95.24%) | 877/978 (89.67%) | 2506/2603 (96.27%) | 93.73% |

## Affected classes, recognizer stage (Citizen + SemLex validation, correct/total)

| Class | Phone today | Current chain | Citizen-variant chain |
|---|---:|---:|---:|
| HOME | 14/14 | 14/14 | 14/14 |
| CHILD | 4/5 | 5/5 | 4/5 |
| GOODBYE | 4/7 | 5/7 | 5/7 |
| HEAR | 9/9 | 7/9 | 9/9 |
| WHAT | 4/4 | 4/4 | 4/4 |
| BIG | 17/21 | 17/21 | 16/21 |
| SIGN | 8/9 | 8/9 | 8/9 |
| ASK | 9/13 | 11/13 | 10/13 |
| COME | 6/9 | 6/9 | 6/9 |
| I | 16/17 | 16/17 | 16/17 |

## Pair confusions, recognizer stage (A→B + B→A, all three validation sets)

| Pair | Phone today | Current chain | Citizen-variant chain |
|---|---:|---:|---:|
| BIG/LANGUAGE | 2 | 2 | 2 |
| ASK/NEED | 0 | 0 | 1 |
| CHILD/HEAR | 0 | 0 | 0 |
| CHILD/GOODBYE | 1 | 0 | 1 |
| I/WE | 1 | 0 | 1 |
| WAIT/MAYBE | 0 | 0 | 0 |
| FAMILY/IMPORTANT | 0 | 0 | 0 |
| LESS/SCHOOL | 2 | 0 | 0 |
| GO/ANSWER | 4 | 4 | 6 |
| ANGRY/NOW | 0 | 0 | 0 |
| HOME/HELLO | 0 | 0 | 0 |
| HEAR/LISTEN | 0 | 0 | 0 |
| GOOD/THANKYOU | 17 | 16 | 15 |
| COME/NEED | 0 | 0 | 0 |

## Continuous signing (tune pool, 89 clips, selection metric — not held-out)

Current chain: epoch 4, WER 37.61% → 31.42% (P 86.6, R 77.0).
Citizen-variant chain: epoch 4, WER 35.84% → 30.09% (P 87.4, R 77.0).

## Selections

Local branch: epoch 50 (Citizen 363/378, floor 361; SemLex gate eval 856/978).
Fusion: seed 1701 epoch 0. Phrase adaptation: epoch 21, phrase segments 75.47%, activity 74.67%.

## Limits

- Local validation is familiar-signer and now omits the removed classes; it measures retention, not the phone.
- Removed classes now rely on Citizen + SemLex only (≈20–35 clips each) — fewer examples in our capture conditions.
- GOOD keeps the current (one-handed) data; the two-handed GOOD option is a separate decision.
- Not exported or installed; the phone still runs the August recognizer with the new finish gesture.

## Deployed to the iPhone 13 for live testing (2026-10-08)

- Letter head retrained with the head-A recipe on the new recognizer, NONE replay from the filtered
  cache (so fingerspelled-I local clips are not taught as NONE): 26-way 85.92% (head A 86.62%),
  recall 84.42% at τ 0.6 (head A 86.15% at 0.5), tune false letters 0.97%.
- FP16 fixed-batch-8 package `SpanRecognizerV17CitizenVariantLettersB8FP16` (25.51 MB): exporter parity
  0/378 mismatches; full 378-clip check 362/378 = PyTorch, 378/378 agreement (deploy/export_check.json).
- App: package added to `Runner/LiveReel/Models`, default recognizer name and letter threshold (0.6)
  changed in `LiveReelModels.swift`; the August package stays bundled. Device tests
  `testStageProfile` (deployed config ≈22–23 ms/frame median) and the finish-pose test pass. Release
  build installed. Rollback: restore deploy/app_source_backup/ (hashes_before.txt) and rebuild.
