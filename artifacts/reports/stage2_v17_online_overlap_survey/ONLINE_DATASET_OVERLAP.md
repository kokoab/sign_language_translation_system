# Online ASL dataset overlap with the locked Citizen-100 vocabulary

Checked 2026-09-01. Counts below come from official metadata already retained in the
repository and are compared with `active/v17/citizen100_manifest.json`. Exact-variant
counts require the pinned ASL-LEX `SignBankAnnotationID`; lexical-name counts do not
prove that the recorded sign is the pinned variant.

## Continuous signing with ordered gloss supervision

| Dataset | Locked-100 overlap | Signers relevant to the audited data | Availability | Stage 2 decision |
| --- | ---: | ---: | --- | --- |
| 2M-Flores-ASL `dev` | 95/100 normalized lexical labels; 811/999 sentences contain a target | local signer ID 0 on 997 rows and ID 1 on 2 rows; not a reliable cross-person split | Public, CC-BY-SA-4.0; selected 155-row subset already acquired | Best immediately usable sentence-gloss source, but weak signer diversity and gloss strings are not pinned ASL-LEX variants |
| ASLLRP continuous | 53/100 exact variants; 1,483 target tokens in 1,104 target-bearing spans | 4 native signers; current split is 3 train / 1 validation | Research/education use; targeted material already acquired | Best exact-variant source; the first single-head `OTHER`-CTC adaptation learned the new data but regressed the old phrase guards |
| NCSLGR public static subset | 17/100 lexical labels; 198 target tokens in 132/166 utterances | 2 native signers in the acquired subset | Public research/education download; already acquired | Genuine transitions, but low-resolution, narrow coverage, and old gloss conventions limit direct CTC value |
| DSP Sentences | 50/100 exact variants; 281 target tokens in 218 utterances | all 15 dataset signers represented | Metadata is visible, but the current BU documentation excludes DawnSignPress video from download | Strong candidate only if DawnSignPress/BU grants video-use access; do not treat metadata as an available video corpus |
| RIT Sentences | 30/100 exact variants; 236 target tokens | 2 signers | Research-only | Ineligible here: the project already consumed this external evaluation source, so it must not be moved into training or model selection |

The five locked labels absent from 2M-Flores `dev` are `GOODBYE`, `PLEASE`, `SAD`,
`SORRY`, and `TOMORROW`.

## Promising releases that are not usable supervised CTC data yet

- Apple's 2026 ASL STEM Wiki work reports nearly 500 manually glossed videos from 16
  signers, 8,655 annotations, and 411 unique glosses that occur in the ASL Citizen
  dictionary. It also reports more than 300 hours of pseudo-annotations. Exact overlap
  with the locked 100 cannot be measured because no annotation CSV/JSON was found on
  the Apple page, in the arXiv source bundle, or in the current Microsoft ASL STEM Wiki
  repository. The paper's own signer review identifies a mix of native/native-like and
  L2 signers, so any future ingestion must filter signers rather than pooling all 16.
- ASL-RGBD-Homework contains 935 continuous videos from 45 signers with ELAN gloss
  annotations, but full access is through an authorized Databrary account. Its fluent
  and learner signers must be kept as separate domains; locked-100 overlap cannot be
  measured without the annotation package.
- How2Sign advertises more than 80 hours of multiview continuous ASL, but the current
  public download exposes videos, keypoints, and English translations rather than the
  ordered gloss targets needed by this CTC trainer. It remains transition/pretraining
  material, not gloss ground truth.
- YouTube-ASL, OpenASL, ASL STEM Wiki's base release, and FLEURS-ASL are captioned or
  partially annotated corpora. They can support representation/transition learning or
  separately validated pseudo-labeling, but they do not currently supply verified
  locked-100 gloss sequences.

## Isolated-sign overlap (Stage 1 only)

| Dataset | Locked-100 overlap | Important boundary |
| --- | ---: | --- |
| ASL Citizen | 100/100 by construction | Existing primary isolated source; no transition evidence |
| Sem-Lex | 98/100 exact ASL-LEX-linked classes in official train metadata | 41 Deaf participants overall; already used as an exact-variant Stage 1 supplement |
| ASLLVD | 52/100 exact variants in the bounded acquired selection | Six consultants; already evaluated/used, citation form only |
| WLASL | 99/100 lexical strings (`HE` absent) | Web video, isolated, variant identity is not guaranteed, C-UDA/noncommercial |
| MS-ASL | 95/100 exact lexical strings; 96/100 if `BYE` is treated only as a candidate alias for `GOODBYE` | 222 signers overall, isolated web video, no ASL-LEX variant IDs |

## Recommendation

Do not download another large corpus yet. The immediate Stage 2 order is:

1. Treat the already acquired 2M-Flores subset and ASLLRP exact spans as the real
   supervised sources; fix the objective/forgetting problem rather than adding more
   unlabeled video.
2. Request DSP Sentences video permission because its 15 signers and 50 exact variants
   would add the most useful new signer diversity if approved.
3. Monitor or contact the Apple/Gallaudet authors for the promised ASL STEM Wiki
   annotation bundle; audit exact locked-100 overlap and native/native-like signer IDs
   before downloading the much larger video collection.
4. Keep How2Sign/NCSLGR/YouTube-ASL-style material in a transition or self-supervised
   lane. Never label generated or pseudo-labeled sequences as genuine ground truth.

This survey did not access any project test video or rerun any evaluation. During the
metadata comparison, the already retained RIT sentence CSV was read only to report its
dataset-level overlap; no RIT video, feature, prediction, or metric was accessed, and
RIT remains permanently excluded from development.
