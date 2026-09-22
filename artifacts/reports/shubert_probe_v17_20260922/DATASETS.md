# SHuBERT data register — 2026-09-22

Status: record for later experiments; no new dataset videos acquired or admitted. Model checkpoints and MediaPipe assets were acquired for the authorized comparison.

## What supplied the pretrained encoder

The [official SHuBERT data instructions](https://github.com/ShesterG/SHuBERT/blob/main/DATASETS.md) name **YouTube-ASL**, excluding material overlapping OpenASL validation and test. The repository releases the encoder and separate fine-tuned DINO face/hand weights; downstream fine-tuning instructions remain TODO at pinned commit cc1929326075bfbad7ad73159b2acf84356059bb.

[YouTube-ASL's official release](https://github.com/google-research/google-research/tree/master/youtube_asl) describes 11,093 videos, 984 hours, and 610,193 English captions, quality-filtered by native Deaf annotators. It distributes video IDs ([official listing](https://console.cloud.google.com/storage/browser/gresearch/youtube-asl)), not a guaranteed complete downloadable video archive. Availability and source permissions need checking before future acquisition. English caption segments are not our exact 100-gloss sequences or framewise sign/transition labels.

The [SHuBERT paper](https://arxiv.org/html/2411.16765v3) uses masked cluster prediction over face, both hands, and body streams. Its downstream datasets include How2Sign/OpenASL for translation, FLEURS-ASL for zero-shot evaluation, ASL Citizen/Sem-Lex/WLASL2000 for isolated recognition, and ASL-Stem-Wiki for fingerspelling detection. It avoids reporting MSASL because of pretraining overlap. These are separate task-specific adaptations, not evidence that the released encoder has a ready 100-sign or transition head.

## Relation to our existing YouTube work

Our existing plan `artifacts/reports/youtube_motion_pretrain_v17_20260921/PLAN.md` froze 1,411 acquired keypoint clips: 1,191 count-consistent and 220 quarantined for manifest/frame-count mismatch. Its pilot targeted three 128-wide causal CTC blocks, not SHuBERT's RGB feature models and contextual encoder. The source inventory separately lists 128 local YouTube videos. These representations must not be added together as independent examples without matching their source IDs.

Our old keypoints cannot reconstruct the appearance information needed by the released face/hand DINO models. Existing raw video can supply those inputs after crop/extraction checks. Therefore, trying the pretrained encoder does **not** require downloading the entire pretraining corpus; recreating its pretraining would be a different, much larger experiment.

## Later experiment admission checklist

- Preserve video IDs, time spans, signer/source groups, and official split assignments; deduplicate overlapping raw/keypoint/clip representations.
- Explicitly record upstream pretraining overlap with evaluation sources. Do not re-open Citizen's consumed test gate.
- Keep unlabeled motion, caption-supervised translation, isolated labels, and reviewed sign intervals as different supervision contracts.
- For a transition decision task, fit and evaluate on reviewed local sign/transition intervals. Captions or fingerspelling intervals cannot silently become transition labels.
- Keep Renz's BSL/PHOENIX boundary-feature data in the candidate register in `continuous_recovery_v17_20260922/PLAN.md`; do not mix its gloss vocabulary into ASL labels.
- Decide any combined training recipe only after SHuBERT and the other shortlisted candidates are reviewed. No bulk download or training started here.
