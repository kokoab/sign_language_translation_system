# Continuous fingerspelling reader (letter_ctc_v17) — 2026-10-02

Goal (user): fingerspelling without a mode trigger, at natural speed, generalizable. This report covers
the letter reader only (no spelling-vs-signing detector yet). Nothing is deployed.

## Model and data

- `active/v17/letter_ctc_v17.py`: per-frame CTC (blank + A-Z + `#` for signed non-letters) over the live
  20 Hz Apple Vision inputs. Camera-invariant landmarks (joints relative to wrist in palm lengths, wrist
  relative to shoulders in shoulder widths, sizes, deltas). 6 symmetric depthwise-conv blocks, d=192,
  look-ahead 18 frames (0.9 s) -> streamable. 1.3 M params (landmarks + hand-crop embeddings) or less
  without embeddings.
- `scripts/train_letter_ctc_v17.py`: FSboard batch1 train + pilot = 3,000 clips / 117 signers (3 clips
  with impossible spans dropped); selection on FSboard validation (750 clips / 15 other signers).
  Augmentation: speed 0.8-2.6x, rotation/scale/position jitter, landmark noise, frame repeats,
  embedding dropout. 60 epochs, AdamW, ~10-25 min per run on the M4.
- `scripts/evaluate_letter_ctc_v17.py`: FSboard validation; ASLLRP in-sentence fingerspelling
  (dev = signer Cory, 98 words; test = all other signers, 774 words), word windows +-0.5 s after
  frame-matched time correction; false letters during non-spelling time; current phone letter decoder
  as baseline; lexicon rescoring (`data/local/name_lists_v17/lexicon_v1.txt`).

## Results

| Run | Inputs | Speed aug | Data | FSboard val CER | names CER | ASLLRP test CER (raw / lexicon) | false letters / min |
|---|---|---|---|---|---|---|---|
| A | landmarks + embeddings | yes | 100% | 10.9% | 10.6% | 94.3% / 93.8% | 20.0 |
| E | landmarks + embeddings (seed 1) | yes | 100% | 11.2% | 11.2% | 93.2% / 92.3% | 18.4 |
| B | landmarks + embeddings | no | 100% | 11.6% | 11.3% | 93.8% / 92.8% | 19.4 |
| C | landmarks + embeddings | yes | 25% | 17.5% | 16.3% | 96.5% / 95.1% | 22.2 |
| D | landmarks + embeddings | yes | 50% | 12.9% | 11.7% | 95.6% / 95.6% | 16.3 |
| **F** | **landmarks only** | yes | 100% | **8.6%** | **8.1%** | 88.5% / 86.9% | 36.2 |
| G | landmarks only (seed 1) | yes | 100% | 8.8% | 8.6% | 87.3% / 86.7% | 37.3 |
| H | landmarks only | no | 100% | 9.6% | — | not run | — |
| current phone decoder | — | — | — | — | — | 99.97% | 0.75 |

ASLLRP exact words: <= 0.7% for every model; the current phone decoder emits almost nothing (0%).

User's own deliberate spelling (5 desktop GELO sessions, 640x360 ~13 fps, user never in training; runs
split at 1 s gaps): old live pipeline 5 exact GELO/ANGELO runs (22 within 1 edit) in 127 runs;
F 16 exact (29 within 1) in 80 runs; G 10 (25) in 78; A 7 (18) in 74. Most common F error: ELO.

Hand size (60 FSboard validation clips re-extracted with the frame shrunk; F): original 117 px palm
6.8% CER, 0.5x (57 px, like the user's phone recordings: 49-62 px) 7.5%, 0.33x (37 px, like ASLLRP)
10.7%. Vision at 1280 instead of 640 px on ASLLRP dev: F 90.1% -> 87.5%, G unchanged.

## Conclusions

1. Deliberate fingerspelling generalizes: unseen FSboard signers 8.6% CER (names 8.1%), small hands
   <= 11%, and the user's own webcam spelling read exactly ~3x as often as the current model.
2. Landmarks-only beats landmarks + hand-crop embeddings on every generalization measure (FSboard val,
   ASLLRP, user sessions); the embeddings fit training signers' appearance. Speed augmentation helps
   (9.6% -> 8.6%). The learning curve is still falling (17.5 -> 12.9 -> 10.9% at 25/50/100%): more
   FSboard clips should help further.
3. Natural in-sentence fingerspelling is NOT solved: ~87% CER on ASLLRP test (10 letters/s median,
   coarticulated). Not explained by hand size or Vision resolution. FSboard is deliberate typing-style
   spelling (2.4-3.9 letters/s); training data with natural spelling (ChicagoFSWild / FSWild+) is the
   identified next step.
4. Not usable always-on by itself: F emits ~36 false letters per minute during ordinary signing. A
   spelling-vs-signing detector (or gating) is required before removing the trigger.
5. The pre-registered dev rule (pick configuration by ASLLRP dev CER) is uninformative here: all
   configurations are 85-91% on 98 dev words. F is preferred on FSboard validation and the user sessions.
   The lexicon gives <= 1.6 points on ASLLRP because the raw output is mostly wrong.

Artifacts: checkpoints `artifacts/generated/letter_ctc_v17/<run>/best.pt` (+ history.json, logs);
`eval_main.json` and per-word records in this folder.

## Update 2026-10-03 — ChicagoFSWild added (natural, in-the-wild spelling)

Data: ChicagoFSWild (dl.ttic.edu, 14.3 GB; 7,304 sequences, official signer-disjoint partitions train 87 /
dev 37 / test 36 signers; image folders treated as 30 fps; `scripts/extract_fswild_v17.py`, 7,304/7,304
extracted, 0 failures; median palm ~50 px, like the user's phone recordings). Training = FSboard 3,000 +
FSWild train 5,429 usable (204 signers); checkpoint selection on FSWild dev (rule fixed before results).
One FSWild clip with no body/face and collapsed hands made the body scale 0 -> NaN loss; fixed in
`body_reference` (minimum scale) and `base_features` (non-finite -> 0).

| Run | Inputs | Speed aug | FSWild dev | FSWild test (exact) | ASLLRP test raw / lexicon (exact) | FSboard val | false letters/min | user GELO exact (within 1) |
|---|---|---|---|---|---|---|---|---|
| F (FSboard only, reference) | landmarks | yes | 58.9% | 62.4% (10.1%) | 88.5% / 86.9% (0.6%) | 8.6% | 36.2 | 16 (29) |
| **I (selected)** | landmarks | yes | **35.3%** | **42.2% (26.5%)** | **73.2% / 72.0% (3.1%)** | 10.0% | 62.2 | **20 (28)** |
| J (seed 1) | landmarks | yes | 35.9% | 41.0% (26.2%) | 72.9% / 71.6% (3.5%) | 10.7% | 54.9 | 15 (27) |
| K | landmarks | no | 37.2% | 42.1% (25.8%) | 78.0% / 74.9% (2.7%) | 10.2% | 94.5 | 17 (30) |
| L | landmarks + embeddings | yes | 38.3% | 43.7% (25.7%) | 75.8% / 75.4% (2.5%) | 11.4% | 59.0 | 14 (28) |

Reference points (not the same splits): the FSboard paper's 300M ByT5 model scored 37.7% CER on
ChicagoFSWild+; human annotators 13.9-17.3%.

Confidence gating (run I, letters kept when their peak posterior >= tau; ASLLRP): dev/test CER and false
letters per minute: tau 0 72.0/73.2%, 53/64 per min; 0.5 71.6/72.9%, 43/49; 0.7 75.8/75.1%, 28/29;
0.9 81.9/82.2%, 13/14; 0.95 85.9/85.7%, 10/9. A posterior cutoff of 0.5 is free; stricter cutoffs trade
accuracy for fewer false letters and never reach a usable always-on rate.

Conclusions: natural-spelling data is what moved natural-spelling accuracy (FSWild test 62 -> 42%,
ASLLRP 88 -> 73%), at a 1.4-point cost on deliberate FSboard spelling. Landmarks-only and speed
augmentation remain best. Natural in-sentence spelling (ASLLRP, ~10 letters/s) is still mostly wrong
at the letter level (3% exact words with the lexicon). Always-on spelling without a trigger needs a
separate spelling-vs-signing gate: the reader alone emits 49-64 false letters per minute of ordinary
signing. Not deployed.

## Update 2026-10-03 — spelling-vs-signing gate (no trigger)

`active/v17/spell_gate_v17.py`, `scripts/train_spell_gate_v17.py`, `scripts/evaluate_spell_gate_v17.py`.
Per-frame probability of fingerspelling, same landmark features / streaming conv blocks as the reader
(0.9 s look-ahead). Spelling: FSboard spans + ChicagoFSWild train. Not spelling: ASL Citizen train-signer
signs (1,476 clips), FSboard margins, O5S5 sign glosses of 5 signers (O5S5 'FS' glosses = spelling;
unannotated frames ignored). O5S5 annotation timing verified: the reader's letter activity is 2.8-6.3x
higher in FS spans than in signs at zero shift and ~1x at +-1-2 s. Training sequences concatenate 1-4 random
clips plus 8 s O5S5 windows. Selection (fixed before results): mean of FSWild dev spelling recall, Citizen
val-signer rejection and O5S5 LG balanced accuracy. Gate seed 0: 96.0% / 98.9% / LG spelling 86.0%, signs
rejected 90.2% (seed 1: 95.6 / 98.9 / 83.2 / 89.8).

End to end (reader run I, letters with peak >= 0.5; theta chosen on ASLLRP dev by: fewest false letters/min
with dev CER <= ungated + 0.03). Seed 0 gate:

| Gating | theta | ASLLRP test CER | false letters/min ASLLRP test | Citizen val letters/min | O5S5 LG letters/min in signs (FS spans hit) | FSWild test CER | user GELO exact (within 1) | fake spelled words (>=2 letters)/min ASLLRP test / LG |
|---|---|---|---|---|---|---|---|---|
| none | - | 72.9% | 48.7 | 12.3 | 39.6 (32/35) | 41.6% | 20 (28) | 10.0 / 8.1 |
| per letter | 0.5 | 75.5% | 20.7 | 0.6 | 9.1 (30/35) | 42.6% | 9 (26) | 4.2 / 2.5 |
| per run, mean | 0.4 | 75.4% | 27.9 | 0.8 | 14.7 (30/35) | 42.2% | 20 (28) | 5.0 / 1.8 |
| per run, max | 0.8 | 75.4% | 28.0 | 0.6 | 16.1 (29/35) | 42.4% | 20 (28) | 5.0 / 2.1 |

Per-letter gating clips the first letter of a word (gate switch-on delay: in the user's exact GELO runs the
gate mean is G 0.64, E 1.00, L 1.00, O 0.95); a 13 fps -> 20 Hz degradation of FSWild test does not reproduce
it (gate keeps 98% of letters either way). Run-level gating (letters grouped at 1 s gaps, whole run kept or
dropped) keeps every GELO and matches per-letter gating on fake spelled words. The dev rule ranks per-letter
gating first on letters/min (dev 25.3 vs 29.2); the recommendation of per-run mean gating at 0.4 uses the
user's sessions and the fake-word metric and is labelled as such. Seed 1 gives the same picture.

Remaining limits: ~2-5 fake spelled words per minute of continuous signing (one every ~12-30 s);
in-sentence natural spelling still ~75% CER (ASLLRP), natural in-the-wild 42% (FSWild). Not deployed.
