# New Stage 2 dataset shortlist

Checked 2026-08-16. This search targets real continuous ASL with ordered gloss
supervision. It does not change the selected Stage 2 v2 checkpoint and does not use
Citizen, SemLex, local, 2M-Flores `devtest`, or any other sealed test split.

## Recommended acquisition order

| Priority | Dataset | What is usable | Current constraint | Decision |
| --- | --- | --- | --- | --- |
| 1 | [2M-Flores-ASL](https://huggingface.co/datasets/facebook/2M-Flores-ASL) | Human-created sentence glosses with expert harmonization; 1080p60 RGB; CC-BY-SA-4.0 | 326 GB total; signer IDs are local rather than globally identifying people | Acquire only `dev`, stream each source file through a bounded 720p30 transcode, record source and derived hashes, and keep `devtest` sealed |
| 2 | [ASL-Homework-RGBD](https://latlab.ist.rit.edu/lrec2022/) | 935 continuous videos, 45 signers, ELAN gloss and nonmanual annotations; CC BY 4.0 | Full volume is restricted to authorized Databrary users | Request institutional Databrary authorization; prioritize the 24 fluent signers and treat 21 learners as a separate robustness domain |
| 3 | [Apple ASL STEM Wiki annotations](https://machinelearning.apple.com/research/sign-language-annotations) | Nearly 500 professionally glossed videos with 8,655 sign annotations, plus larger pseudo-annotation sets described by the paper | No downloadable annotation bundle or repository was located | Monitor the official page or contact the authors; ingest only after the files and license are public and verifiable |

## Measured 2M-Flores `dev` result

The metadata-only audit read all 999 `dev` rows from the official Hugging Face dataset
server. It found:

- 811 sentences containing at least one locked Citizen-100 lexical label.
- Coverage of 95/100 labels; the absent labels are `GOODBYE`, `PLEASE`, `SORRY`,
  `SAD`, and `TOMORROW`.
- 4,388 normalized gloss tokens across the split.
- Local signer values of `0` for 997 rows and `1` for two rows. The dataset card warns
  that this field is not a global signer identity, so it cannot support a new
  signer-disjoint claim.
- No access to the reserved `devtest` split.

The complete row-level manifest is
`data/local/dataset_metadata/2m_flores_asl/dev_locked100_audit.json`, SHA-256
`4dcb426bab947fdd455a364ede8c7039c10518316bd031f709712f2ee18d7130`.

### Training contract

Use all usable sentence glosses with an expanded Stage 2 vocabulary. Do not remove
out-of-vocabulary tokens and pretend the remaining target signs are adjacent: that
would corrupt CTC order and timing. The 811-row overlap is an acquisition priority,
not permission to rewrite its transcripts as Citizen-100-only phrases. Keep the
existing local and ASLLRP validation gates unchanged and reserve 2M-Flores `devtest`
for a later, one-time external evaluation after the training design is frozen.

## Useful later, but not direct gloss-CTC training data today

- [How2Sign](https://how2sign.github.io/index.html): over 80 hours and 11 signers, but
  the current public download exposes video/keypoints and English translations rather
  than its gloss annotations.
- [ASL STEM Wiki](https://www.microsoft.com/en-us/research/project/asl-stem-wiki/):
  64,266 sentence videos and 316 hours, but the base release does not provide full
  sentence gloss targets.
- [FLEURS-ASL](https://www.kaggle.com/datasets/googleai/fleurs-asl): 1,749 sentence
  videos and 7.49 hours with English and fingerspelling annotations, not complete
  gloss sequences.
- [OpenASL](https://github.com/chevalierNoir/OpenASL), YouTube-SL-25, and the ASL
  portion of [SignNet-1M](https://signnet.chatsign.ai/): useful for translation,
  representation learning, or future weak supervision, but no verified ASL sentence
  gloss target suitable for the present CTC trainer.
- [ASL 1000](https://registry.opendata.aws/asl_1000/): controlled-access high-quality
  video and landmarks; its public schema does not establish the ordered continuous
  gloss supervision needed here.

## Storage-safe acquisition design

The official 2M-Flores card lists about 156 GB for `dev` and 326 GB total. The compact
selection manifest chooses 155 rows (18.93 GiB total source transfer), covers all 95
available locked labels, and requests up to five examples per label. A binary optimizer
minimizes total source bytes while preserving all rows needed for rare labels. Because
each source is removed immediately after verified transcoding, 18.93 GiB is not held
on disk at once. Acquisition is resumable and bounded:

1. Download one manifest-pinned MOV at a time.
2. Verify and record its source identifier and hash.
3. Transcode without aspect-ratio distortion to at most 720p, 30 fps, H.264.
4. Verify duration and decode, hash the derived file, then release only the temporary
   source copy.
5. Extract v17 features in bounded workers and retain the compressed video needed for
   RGB crops.
6. Stop automatically on memory pressure, insufficient disk headroom, a hash/schema
   mismatch, or an annotation parsing failure.

This is dataset discovery evidence, not a new accuracy or mobile-readiness claim.
