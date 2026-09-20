# Finish-time bounded CTC experiment

**Result: the simple Finish-time head is better balanced than the repaired causal CTC,
but it is still dominated by deletions and must not replace Reel or the accepted Stage 2.**

The experiment froze the Stage-1 Squeezeformer and trained one 400,174-parameter
bidirectional GRU with one CTC objective. Its internal alphabet is blank, the locked 100
glosses, and UNKNOWN. Decoding proved that only the 100 locked labels reached the visible
transcript. The official Citizen test remained sealed.

## Results

| Measure | Finish CTC | Repaired causal CTC |
| --- | ---: | ---: |
| Connected WER | **83.80%** | 98.24% |
| Connected substitutions / deletions / insertions | 18 / **215** / 5 | 65 / 90 / 124 |
| Connected exact recordings | 32 / 225 | not used as a selection gate |
| Connected empty recordings | 160 / 225 | — |
| Familiar WER | **88.03%** | 107.72% |
| Familiar substitutions / deletions / insertions | 37 / **191** / 0 | 112 / 74 / 93 |
| Familiar exact recordings | 0 / 97 | — |
| Contiguous WER | 83.33% | 50.00% |
| Citizen isolated CTC exact | 344 / 378 (91.01%) | 339 / 378 (89.68%) |
| SemLex isolated CTC exact | 812 / 978 (83.03%) | 762 / 978 (77.91%) |
| LG core CTC exact | 2 / 57 (3.51%) | 4 / 57 (7.02%) |
| Verified blank false emissions | 0 / 16 | 1 / 16 |
| Adjacent duplicate outputs / references | 6 / 3 | not recomputed |

The lower aggregate WER is real, but it is not a usable recognition improvement. The
model emitted only 74 tokens for 284 connected reference tokens and 68 for 259 familiar
tokens. It therefore replaced the causal model's insertion problem with a more severe
blank/deletion problem. Complete-utterance future context did not fix held/repeated-sign
behavior: adjacent duplicates remained, and these development recordings are not the
expert-labelled held-once versus twice set needed to measure that distinction.

Training also underfit the complete sequences. Final continuous training WER was 74.05%
and the best training epoch was 71.34%, despite covering all 923 continuous, 1,901
isolated, 199 positive-core and 494 verified-blank samples every epoch. This isolates the
failure to the frozen-feature/simple-head recipe and available continuous supervision;
it does not reject Finish-time continuous recognition as the product direction.

## Decision

Do not promote this checkpoint and do not add suppression rules. Keep the current
runtime while treating bounded continuous recognition as the intended replacement once
it can fit training sequences and pass signer-disjoint phrase, hold/repeat and OOV gates.
The next experiment must address continuous evidence/coverage rather than add another
decoder or language model.

An initial reporter incorrectly included alignment matches in WER. The model and saved
predictions were unaffected; `REPORTING_CORRECTION.json` records every corrected value.
`verification.json` independently recomputes all final edit metrics and the locked-
vocabulary contract.
