# Citizen100 external isolated-ASL dataset audit

**Audit date:** 2026-08-10  
**Purpose:** find higher-quality RGB video that can supplement the frozen Citizen100
training split without changing its exact raw-gloss/ASL-LEX class definitions.

## Decision

Three bounded audits are complete. The strongest immediately usable human-review pool
is the quality-filtered local subset (132 clips/49 exact-text classes), followed by
exact-tier RIT candidates and a 62-clip MS-ASL gap audit. None is automatically approved:
all require ASL-fluent exact-variant review and may enter training only.

The next materially better corpus is Sem-Lex because it has 41 Deaf participants and
expert ASL-LEX alignment, but the user must personally submit its access form and agree
to the CC BY-NC-SA and research-respect terms. ASL-100-RGBD remains blocked by
Databrary's authorized-user gate. Do not bypass either access gate.

## Comparison

| Source | Useful evidence for Citizen100 | Main blocker | Current decision |
| --- | --- | --- | --- |
| Quality-filtered local raw | 17,179 local-style clips inspected; 132 strongest clips across 49 exact-text, model-consistent classes after v17 gates | No trustworthy signer or lexical-variant IDs | Small train-only review pool; never validation/test |
| ASLLRP/RIT isolated signs | 12,197 segmented clips; 13 participant IDs; 292 unique exact-name candidates across 60 classes; explicit variants and handshapes | Research-only/noncommercial; name match does not prove the frozen ASL-LEX variant | Acquired/extracted; 15 exact-tier classes model-consistent |
| ASL-100-RGBD | 4,150 tokens; 22 fluent/DHH signers; 1920x1080 RGB; deliberately checked target productions | Only 12 direct label matches plus six variant families; authorized Databrary account required | Pursue only if access already exists |
| MS-ASL | Official train-only gap pass retained 62 verified clips/30 classes; 11 classes model-consistent | YouTube attrition; no ASL-LEX code | Bounded fallback acquired; manual review required |
| Sem-Lex | 91,148 isolated videos, 3,149 signs, 41 Deaf participants, expert ASL-LEX/SignBank alignment | Named user must submit access form and personally accept CC BY-NC-SA/research-respect terms | Best next source after user obtains access |
| Additional ASLLRP segmented corpora | Exact-name metadata has 1,992 clips/65 classes in ASLLRP sentences, 317/60 in DSP sentences, and 142/65 in DSP citation signs | Official UI requires a free login for segmented downloads | Metadata audited; do not bypass login |
| Legacy ASLLVD | 9,747 citation-form tokens; six signers; detailed lexical variants | Public pre-cut files stack two views; clean downloads require login; research-only | Reference/variant audit, not first training source |
| Kaggle ASL Signs | 250 labels and participant metadata | MediaPipe landmark Parquets only; no original RGB video for Apple Vision v17 | Incompatible with current extractor |
| WLASL | Broad vocabulary and many sources | Variable web quality, link attrition, computational-use terms; already represented in the mixed local corpus | Do not prioritize |
| ASL-LEX 2.0 | Exact lexical/phonological reference for 2,723 signs | Reference videos explicitly may not be saved or used without permission | Metadata/variant reference only; no video acquisition |
| PopSign previews | 43 overlaps and 129 successful Apple extractions | Website previews are downsampled/speed-normalized and never training-eligible | Paused |

## RIT quality smoke

The official 2024-06-27 RIT metadata CSV contains 12,197 rows and 13 distinct
participant IDs. Restricting on exact `entry/variant gloss label` equality and resolving
unique archive members yields 292 clips across 60 classes: 249 pinned-raw-gloss matches
and 43 weaker canonical-only matches. This is a candidate-name match only; ASLLRP and
ASL-LEX use different identifiers.

One member was retrieved from the official segmented-sign ZIP by HTTP byte range rather
than downloading either 1.7-1.9 GiB archive:

- H.264, 1280x720, 30 fps, 1.7 seconds
- Apple Vision v17 elapsed time: 0.73 seconds
- observed hand-frame coverage: 96.08%
- face/body presence: 96.88% / 96.88%
- schema audit: passed with zero errors

The full selective acquisition is now complete. All 292 clips are 1280x720, all 292
extracted successfully, and every v17 archive passed schema audit. Frozen-model triage
found 16 model-consistent classes overall, including 15 pinned-raw exact-tier classes;
zero were automatically approved.

## ASL-100-RGBD vocabulary overlap

The paper lists 100 glosses, concentrated heavily on clock times, weekdays, and
time-related expressions. Twelve labels directly match the frozen manifest:
`HOW1`, `MORNING`, `NIGHT`, `NO`, `NOW`, `TIME`, `TOMORROW`, `WEEK`, `WHAT1`,
`WHERE`, `YESTERDAY`, and `YOU`.

Six more concept families require visual/variant confirmation: `I_ME`,
`IX_HE_SHE_IT`, `IX_THEY_THEM`, `WHEN1/WHEN2`, `WHO1/WHO2/WHO3`, and
`WHY1/WHY2`. The dataset is therefore high quality but small for the present frozen
vocabulary.

## MS-ASL result

The official Microsoft annotation ZIP contains 95 canonical-name overlap classes in
the train split; 81 also equal the pinned Citizen raw text. The bounded gap downloader
used train only, excluded the strict local review classes, required >=640x360 metadata,
unique signer IDs, short early source segments, and capped retention at three/class.
It retained 62 verified clips across 30 classes from 192 attempts. All 62 extracted and
passed v17 audit. Frozen-model triage found 11 model-consistent, 13 ambiguous, and six
high-risk classes; zero were automatically approved.

## Access-gated leads

- Sem-Lex is the best remaining lead: 91,148 videos from 41 Deaf participants with
  expert ASL-LEX/SignBank alignment. Its Google access form requires the named user to
  accept noncommercial/share-alike and community-respect commitments.
- Databrary volume 1062 confirms ASL-100-RGBD's 42 sequences and 22 fluent/DHH signers,
  but every video session is `authorized_users` and inaccessible without authorization.
- Purdue RVL-SLLL offers professional-studio recordings from 14 fluent Deaf signers,
  but requires a signed license sent to Purdue before credentials are issued.
- ASL-LEX reference videos are not a training source: the official license prohibits
  saving, displaying, or other use without explicit permission.

## Sources

- [ASLLRP Sign Bank](https://dai.cs.rutgers.edu/dai/s/signbank)
- [ASLLRP dataset overview](https://www.bu.edu/asllrp/about-datasets.pdf)
- [ASLLRP/RIT metadata directory](https://dai.cs.rutgers.edu/asllvd/signbank/)
- [ASLLRP Sign Bank terms](https://www.bu.edu/asllrp/signbank-terms.pdf)
- [ASL-100-RGBD paper and vocabulary](https://aclanthology.org/2020.signlang-1.14.pdf)
- [ASL-100-RGBD Databrary record](https://nyu.databrary.org/volume/1062)
- [MS-ASL official project](https://www.microsoft.com/en-us/research/project/ms-asl/)
- [MS-ASL official annotations](https://www.microsoft.com/en-us/download/details.aspx?id=100121)
- [ASL-LEX 2.0](https://asl-lex.org/about.html)
- [ASL-LEX video-use restrictions](https://asl-lex.org/download.html)
- [Sem-Lex repository and access instructions](https://github.com/leekezar/SemLex)
- [Purdue RVL-SLLL access page](https://engineering.purdue.edu/RVL/Database/ASL/asl-database-front.htm~)
- [WLASL project](https://github.com/dxli94/WLASL)
