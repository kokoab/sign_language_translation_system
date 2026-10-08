# Mid-sign splits (BIG/LANGUAGE → GO) and HAVE → ANGRY — 2026-10-09

## Root cause of the splits
1. The AV boundary student marks the fast internal movement of a sign as rest (dips on 64% of isolated Citizen
   train clips; on the user's BIG/LANGUAGE the dip reaches act ≈ 0.1 for 2–8 frames).
2. The word decoder could not span more than 3 rest frames and committed the first piece 1 frame after it ended.
3. The recognizer never saw partial signs; the raised bent-L end hold of BIG resembles the start of Citizen GO.

## Installed (iPhone 13)
- Word decoders: `max_inner_o 8`, `inner_floor 0.5`, `lag 6` (Python options, Swift `LiveDecoderConfig.v3Words`).
- Recognizer: `SpanRecognizerV17FragmentLettersB8FP16` (v2 recipe + fragment augmentation 0.3).

## HAVE → ANGRY
HAVE alone is read correctly (0.97); in "I HAVE BIG" it is ANGRY 0.84 on Mac and phone, landmarks-only too.
HAVE has 0 local clips (22 total) vs ANGRY ~140; the in-sentence HAVE exits upward (ANGRY's movement).
14 classes have 0 local clips: WHAT GOODBYE HAVE COME FIND HEAR LISTEN TALK SICK HUNGRY BIG HOME CHILD SIGN.

## Results
See `table.txt` (tune = 89-phrase tuning pool WER; citizen = 378 Citizen val in phone word view; takes =
user takes 20261009_082836_01–03). No test split was read.
