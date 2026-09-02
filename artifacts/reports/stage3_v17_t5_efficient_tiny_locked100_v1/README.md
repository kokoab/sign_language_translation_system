# Stage 3 tiny naturalizer v1

This report evaluates the 15.58M-parameter T5-efficient-tiny checkpoint after one
validation-selected MPS training epoch. It is a bounded English renderer for recognized
locked-vocabulary gloss buffers, not evidence of general ASL translation quality.

## Data decision

- `slt_stage3_dataset_final.csv`: 15,843 unique synthetic pairs; useful as broad
  grammar supervision, including 1,537 sequences with at least five glosses.
- `slt_dialogue_dataset.csv`: 11,245 rows but only 35 unique pairs and no 5+ gloss
  rows; audited but excluded.
- 36 reviewed templates: training-only and authoritative on target conflicts.
- 143 controlled locked-100 compositions of 5–12 glosses: 82 train, 35 validation,
  and 26 test, split by complete gloss sequence.

There is no exact gloss-sequence overlap across train, validation, and test. Citizen
test and 2M-Flores devtest were not accessed.

## Results

| Slice | Validation exact | Test exact |
| --- | ---: | ---: |
| Overall | 93.98% (1,483/1,578) | 93.36% (1,490/1,596) |
| Locked 100 | 91.67% (110/120) | 94.00% (94/100) |
| Five or more glosses | 95.76% (158/165) | 91.90% (193/210) |
| Controlled locked-100, 5+ | 100% (35/35) | 100% (26/26) |

The predefined promotion gate passed. The two non-exact locked-100 long test outputs
were natural article variants (`to school` versus the synthetic CSV target `to the
school`), not dropped gloss content. Some out-of-vocabulary synthetic test failures are
genuine semantic or fingerspelling errors, so outputs outside the locked scope require
review.

The saved model is `../../models/stage3_v17_t5_efficient_tiny_locked100_v1/` and its
weight SHA-256 is
`c0875815d47d9242bae9d19bd1040f85647550a5b2f24c482bdb9b2f4525a5b0`.
Detailed provenance and metrics are in `result.json`; per-example evidence is in the
validation/test JSONL files.

For the live prototype, a warm nine-gloss generation measured 137.96 ms median on CPU
and 1,678.02 ms on MPS. CPU is the live default because MPS autoregressive decoding is
slower and would contend with Stage 1.
