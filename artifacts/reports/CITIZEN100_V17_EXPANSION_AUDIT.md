# Citizen100 v17 expansion audit

**Status: DIRECT 100-CLASS AUGMENTATION EXHAUSTED; BROAD PRETRAINING AVAILABLE**

This audit asks whether more ASL Citizen videos can safely be added to the current
100-class Apple Vision Stage 1 training corpus without changing class semantics,
merging ASL-LEX variants, crossing official signer splits, or reopening the consumed
test gate.

## Evidence reviewed

- `PROJECT_GROUND_TRUTH.md`
- `artifacts/reports/CITIZEN100_V17_MANIFEST.md`
- `artifacts/reports/CITIZEN100_RAW_AUDIT.md`
- `artifacts/reports/CITIZEN100_V17_EXTRACTOR_AUDIT.md`
- `artifacts/reports/CITIZEN100_V17_LANDMARK_QUALITY.md`
- `artifacts/reports/stage1_v17_validation/REPORT.md`
- `artifacts/models/stage1_v17_baseline/history.json`
- `scripts/build_citizen100_v17.py`
- `scripts/download_citizen100_v17.py`
- the three cached official ASL Citizen split CSVs

The official Microsoft dataset page still identifies ASL Citizen version 1.0 as the
latest released version. It contains 83,399 videos, 2,731 signs, and 52 signers with
official signer-disjoint train/validation/test assignments.

## Exact-variant inventory result

- Official metadata rows across all signs and splits: 83,399
- Rows matching the 100 pinned raw-gloss plus ASL-LEX pairs: 3,102
- Rows expected by the frozen manifest: 3,102
- Matching local raw MP4 files: 3,102
- Missing pinned exact-variant videos: **0**
- Existing selective-download result: 3,102/3,102 size/CRC/SHA-provenance checks,
  zero failures, and zero remaining output bytes
- Fresh official-archive downloader dry-run: 1.5883 GiB existing output and exactly
  0.0 GiB remaining output

The selected corpus already contains every official Citizen video for each chosen
exact variant. Re-running the downloader can only verify the same files; it cannot add
training examples.

The current effective Apple landmark counts are 1,475 train, 378 validation, and 1,247
test. One train clip is excluded by the explicit low-quality rejection ledger, and one
test source contains no visible hand. Neither is recoverable by downloading it again.
The official test split has already been consumed once and must not be moved into
training or validation.

## Remaining same-name rows are different variants

The metadata contains 14 unselected raw-gloss/ASL-LEX pairs across 13 current concept
names. They are not additional examples of the pinned class and must remain separate
unless an ASL-fluent review deliberately defines a new experiment and label space.

| Current concept | Unselected raw gloss | ASL-LEX | Train videos | Val videos | Test videos |
| --- | --- | --- | ---: | ---: | ---: |
| DOCTOR | DOCTOR2 | `K_03_015` | 15 | 4 | 12 |
| DRINK | DRINK2 | `K_03_092` | 14 | 5 | 11 |
| EAT | EAT2 | `B_03_082` | 12 | 3 | 9 |
| HEAR | HEAR1 | `E_01_097` | 14 | 4 | 11 |
| HOSPITAL | HOSPITAL2 | `J_02_022` | 11 | 3 | 10 |
| HOW | HOW2 | `F_02_086` | 14 | 3 | 14 |
| NIGHT | NIGHT2 | `D_02_047` | 10 | 3 | 10 |
| SAME | SAME2 | `B_02_013` | 14 | 3 | 15 |
| TALK | TALK2 | `H_01_023` | 14 | 3 | 14 |
| THEY | THEY2 | `E_01_044` | 11 | 3 | 8 |
| WANT | WANT2 | `C_03_013` | 15 | 3 | 12 |
| WHAT | W.H.A.T | `G_02_089` | 15 | 3 | 12 |
| WHAT | WHAT2 | `D_02_084` | 15 | 4 | 12 |
| WOMAN | WOMAN2 | `K_01_101` | 14 | 5 | 11 |

These pairs total 188 train, 49 validation, and 161 test videos. The 30 W.H.A.T clips
were previously downloaded and quarantined after being identified as fingerspelling.
Downloading the others would be useful only for a separate-variant vocabulary or an
explicit OOV/rejection dataset; they cannot safely supplement the current 100 labels.

## Training-log implication

The frozen validation run made 26 top-1 errors on 378 clips. Its two repeated
confusions were I -> WE and ANSWER -> GO (two clips each); the remaining confusion
pairs occurred once. Citizen contains no additional pinned exact-variant examples for
those four classes. More epochs over the same videos or adding alternate ASL-LEX codes
does not provide the missing signer/capture diversity.

## Safe next data choices

1. Record or obtain new examples of the same 100 exact variants from new signers,
   preferably portrait iPhones, with separately frozen train/validation/evaluation
   assignments.
2. Download selected unchosen Citizen variants as separate classes after ASL-fluent
   review, which expands or changes the vocabulary rather than augmenting current
   classes.
3. Download unselected Citizen train/validation signs as an explicit OOV dataset for
   UNKNOWN-rejection work, never relabeling them as one of the 100 known signs and
   never reopening Citizen test.

## Broad-vocabulary pretraining clarification

The 3,102-video result is a complete download only for the selected 100 exact variants,
not for every video recorded by those Citizen signers. The full official train and
validation pools contain 40,154 and 10,304 videos respectively across 2,731 exact
raw-gloss/ASL-LEX variant pairs. Those other signs cannot be relabeled as any of the
current 100 classes, but they can support a separate 2,731-class representation-
pretraining stage followed by leakage-safe 100-class fine-tuning.

Retaining all train/validation raw videos would require 28.73 GiB while the host had
only about 16 GiB free. A temporary extractor therefore streamed one
official ZIP member at a time, verified size and CRC, extracted the Apple v17 archive,
and deleted only the temporary transfer copy. It rejected Citizen test by construction.
Its exact-pair manifest preserved
`RESEARCH1` and `RESEARCH2` as distinct classes even though Citizen assigns both
ASL-LEX code `B_03_084`.

A 15-item acquisition smoke completed with 15 schema-valid features, zero no-hand
clips, zero failures, no retained raw video, and 1.68 clips/second after bounded
four-worker download prefetch was enabled. This is pretraining data; it does not alter
the frozen 100-class baseline or reopen Citizen test.

A subsequent full pretraining stream was stopped immediately when the user questioned
the expanded scope. It ended with 298 compact train features and one no-hand event,
zero retained temporary/raw videos, and no running process. At the user's explicit
request, the 4.5 MiB directory was moved recoverably to
`/Users/frnzlo/.Trash/SLT_citizen2731_v17_20260810_0955`, and the temporary broad-
pretraining script/test/manifest were removed. Citizen100 remains the sole scope.
