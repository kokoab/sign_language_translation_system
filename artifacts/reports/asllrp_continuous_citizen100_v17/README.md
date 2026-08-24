# ASLLRP segmented continuous-sign audit for Citizen100

- ASLLRP exact-variant Stage 1 train candidates: 1483 signs across 53 classes.
- RIT exact-variant external evaluation reserve: 236 signs across 30 classes.
- Combined exact-variant Stage 1 coverage: 1719 signs across 54/100 classes.
- Target-bearing full utterances: 1237.
- Directly eligible locked-100-class multi-token CTC utterances: 1.
- Exact contiguous target-only spans: 70 spans across 68 parent utterances and 144 target tokens.

The Stage 1 mapping is official ASL-LEX/Sign Bank exact-variant matching, not English-label normalization. RIT remains a held-out external evaluation source. Most full utterances contain glosses outside the locked 100 classes and therefore must not be presented as fully supervised CTC data. Contiguous target-only spans are cropped at manual sign boundaries plus five context frames and may be evaluated as short real phrases without assigning labels to intervening out-of-vocabulary signs.
