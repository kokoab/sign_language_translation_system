# Free continuous-ASL replacement search

## Decision

Use the **YouTube-ASL Clip Keypoint Dataset** from LINDAT as the immediate public
source for connected-sign motion and transition pretraining. It is freely downloadable,
requires no account, and its repository record declares CC BY 4.0. Keep the public
How2Sign landmark release as a smaller controlled secondary source.

This does **not** replace human gloss boundaries. YouTube-ASL supplies sentence-level
English translations and 2D keypoints, not timed gloss sequences. It must therefore
not create new output labels or supervise the locked 100-gloss decoder directly. The
deployed output remains blank plus the existing 100 glosses.

## What was verified

- The LINDAT record contains 10 ZIP archives with **390,547 unique JSON sequences**.
- The archives total **373,639,866,013 bytes** (347.98 GiB), but the server supports
  byte ranges. Python can inspect and selectively extract individual members without
  downloading the full corpus.
- The provided annotations contain 353,195 train and 38,299 development clips.
  Keypoints exist for 351,307 train and 38,167 development annotations: **389,474 of
  391,494 (99.48%)** overall.
- Train and development contain 8,479 and 943 source video IDs, with zero overlap.
  These are source-video IDs, not verified signer identities.
- Three real sequences were range-extracted and parsed. They contain 85, 178 and 97
  frames. Each frame exposes 33 pose, 21 right-hand, 21 left-hand and 478 face slots
  when present. This is 553 possible 2D points, despite the repository description
  saying 208, so conversion must follow the files rather than the prose.
- All locked vocabulary words occur as English tokens in train captions and 99 occur
  in development captions. This is only caption-token coverage; it is not evidence
  that each token corresponds to a visible, individually aligned gloss.

The complete central-directory index is saved locally, so a pilot can select clips
before transferring data. Metadata, hashes, samples and the index are under
`data/local/youtube_asl_keypoints_20260921/`.

## Recommended experiment

1. Use the frozen `pilot_2000_manifest.csv`: 2,000 train sequences from 2,000 unique
   source video IDs, with one non-empty-caption clip nearest 120 frames per selected
   video and a 50–250-frame limit.
2. Map pose 33 + hands 42 + the existing 53-face subset into the shared 128-point
   layout. Preserve masks; do not invent a depth coordinate.
3. Pretrain only the temporal encoder with masked-landmark reconstruction and temporal
   order prediction. This teaches holds, movement and transitions without pretending
   the captions are gloss timing.
4. Fine-tune the closed blank+100 head on the existing trusted ASLLRP, O5S5 and local
   labeled material. Keep the supplied YouTube-ASL development split out of pretraining.
5. Promote only if held-out local/ASLLRP boundary, repeat and WER gates improve.

This is the smallest defensible use of the new corpus. Direct caption-to-gloss training
would reopen the vocabulary and recreate the weak-alignment problem already measured
in the failed translation experiment.

The pilot is estimated at **1.91 GB compressed (1.78 GiB)** from the corpus-wide ZIP
mean. Allow approximately **6 GB** for extracted JSON plus working space. The full
347.98 GiB corpus is not needed.

## Other sources checked

| Source | Result |
|---|---|
| How2Sign landmarks | Public CC BY-NC 4.0; 35,176 sentence sequences and 66.94 hours. Useful secondary data, but only nine filename signer IDs and no gloss boundaries. |
| 3D Continuous ASL | Rejected: only 2,000 tensor files exist for 98,196 metadata references (2.04%). |
| SignNet-1M | Rejected for this gap: very large synthetic/noncommercial resource; ASL portion lacks the required timed gloss supervision. |
| SignAvatars | Rejected for immediate use because download access requires a request form. |
| ASL-Homework-RGBD | Still the best exact human-boundary source, but the current Databrary account is not authorized. |

Machine-readable evidence is in `verification.json`; the source comparison is in
`sources.csv`.

## Primary links

- YouTube-ASL keypoints: <https://hdl.handle.net/11234/1-5898>
- Dataset code: <https://github.com/zeleznyt/T5_for_SLT>
- How2Sign: <https://how2sign.github.io/>
- Public How2Sign landmarks: <https://huggingface.co/datasets/martinctl/how2sign-asl-landmarks>
