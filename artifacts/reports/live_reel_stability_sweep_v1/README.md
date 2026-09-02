# Reel stability timing sweep

This development-only sweep uses five existing local phrase recordings and the Reel
Stage-1 path with Stage 2 and the lip verifier disabled. Confidence and margin gates
remain unchanged. No Citizen, SemLex, local test, or sealed split was accessed.

| Timing preset | Exact | Token edits | Median committed candidate | Outputs |
| --- | ---: | ---: | ---: | --- |
| 0.62 s candidate / 0.14 s probe | 1/5 | 5 | 1.07 s | `GOOD EASY`; `HELLO HOW YOU`; `NAME`; `I`; `THANKYOU READ` |
| 0.50 s candidate / 0.12 s probe | 3/5 | 4 | 0.67 s | `GOOD MORNING`; `HELLO HOW YOU`; `MY NAME`; `I`; `GOOD READ` |
| 0.45 s candidate / 0.10 s probe | 3/5 | 4 | 0.60 s | `GOOD MORNING`; `HELLO HOW YOU`; `MY NAME`; `STOP I`; `GOOD READ` |

The intermediate preset is selected. It provides the same exact and edit result as
the fastest preset without adding its false `STOP`, and reduces median candidate
duration by 37.5% from the previous default. This is fitted development evidence, not
an independent accuracy estimate. Remaining PLEASE/HELP and THANKYOU/FRIEND errors are
classification/domain errors rather than justification for weakening global gates.

