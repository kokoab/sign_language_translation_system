# Continuous ASL data acquired — 2026-09-12

Actual data files are in `data/local/continuous_asl_acquisition_20260912/`.
This acquisition adds timed continuous-signing material; it does not establish
recognition accuracy or make every file compatible with the existing Apple Vision model.
No training, checkpoint changes, protected-test access, or source-data deletion occurred.

| Acquired source | What is on disk | Measured coverage | Current use |
|---|---|---|---|
| CLERC Épée v0.3 | 1,200 NPY motion sequences and 1,200 timed JSON annotations; 196.6 MB | Six source signer IDs; 70.86 minutes; 3,788 gloss tokens; 68/100 exact Citizen raw-gloss strings | Separate MediaPipe motion/timing research; not Apple Vision input |
| Gallaudet MoLo003 systems | Original 926.5 MB MP4, two original EAFs, and two per-signer video crops | 17.29-minute recording; two annotated signers; 1,517 timed hand-gloss annotations | Raw-video extraction/alignment inspection; partial annotations and variant checks remain |
| RIT ASL-Homework public sample | 14.1 MB MOV, matching EAF, demographics and source page | One publisher-coded fluent signer, F13; 6.5 seconds; nine timed glosses | Small raw-video/annotation sanity sample |

These counts describe different modalities and are not one pooled training dataset.
Source signer IDs are retained separately; no attempt was made to identify or link
pseudonymized Épée signers to other datasets. No random video splits were created.

## Épée: the most relevant new temporal corpus, with a format limitation

Source: [publisher dataset](https://huggingface.co/datasets/CLERC-DATA/epee).
Pinned revision: `93bc8aa0a6526e86af80db5f59af8d133292c9b5`.

Each clip has a `[frames, 128, 3]` float32 MediaPipe array and a JSON file with signer
ID, phrase ID, frame rate, frame count, English text, and timed ASL gloss segments.
All six source signer IDs have 200 clips. The publisher describes the signers as native
Deaf ASL signers and the annotations as linguistically validated. Those are publisher
statements, not independent assessments made by this acquisition.

Measured from the downloaded files:

- 127,545 frames, 724 distinct raw gloss strings, 3,788 annotated tokens.
- 68 of the frozen 100 **raw gloss strings** occur, totaling 1,146 tokens.
- HUNGRY appears six times across five source signer IDs.
- 83 clips repeat a raw label; 19 adjacent pairs have identical raw labels.
- 244 annotated segments exceed 1.07 seconds. This is duration coverage, not proof
  that all 244 are deliberate long holds.
- All 1,200 arrays match their annotation frame counts and contain finite values.
  No invalid/out-of-video intervals, overlapping sequential segments, or duplicate
  pose-array hashes were found.

[Per-class coverage CSV](epee_locked100_coverage.csv) preserves each frozen ASL-LEX code,
exact raw string, count, and source signer IDs. Numeric variants and aliases are not
merged. A text match does not verify the exact visual variant.

**The release contains no raw video.** Its 128-point MediaPipe arrays lack our Apple
Vision observation contract and cannot be fed into the existing v17 checkpoint by
renaming channels or selecting joints. The publisher reserves original pixels for
separate licensing. This corpus can support a separately specified temporal experiment;
it is not a replacement training manifest for the current model.

The downloader hit an anonymous Hugging Face API rate limit. It respected the reported
reset window and resumed through the library's ordinary HTTP transport, retaining
completed files. No login, token collection, alternate identity, or rate-limit bypass
was used. All 2,400 annotation/array files were checked against their publisher commit
and content hashes. See [audit](epee_audit.json) and [integrity checks](integrity.json).

## MoLo: actual raw video and human timing

The [publisher](https://sites.google.com/gallaudet.edu/card/data/molo/molo-primary-data)
links the recording and transcript collection. The native video was acquired from
the [official public OSF file](https://osf.io/download/67e6bc94cdaca359ce17d7bd/).
The matching original EAFs had been located in the earlier project audit; the newly
downloaded video now supplies the missing media for this recording.

- Original: `molo/201102_FrankGriffin_JonHenner_MoLo003_S_4_5_Logi.mp4`.
- SHA-256: `0821b76c49e18c5970fdc650ca51bda6e980cd4f363a99cf82c9f4987c97c252`.
- 1280×720, 30000/1001 fps, 31,086 decoded frames, 1,037.2362 seconds.
- Frank Griffin EAF: 780 timed hand annotations; Jon Henner EAF: 737.
- Their EAF media filenames match exactly, both time origins are zero, and all manual
  intervals fall within the downloaded video. Original EAF hashes are retained.
- These annotations contain 94 exact locked raw-label occurrences across 19 labels,
  before simultaneous-hand deduplication. They are not 94 approved training examples.

`FrankGriffin_MoLo003_S_crop.mp4` and `JonHenner_MoLo003_S_crop.mp4` isolate the two
top-row participants at 640×360. The speaker layout is explicitly documented in the
[publisher's video description](https://www.youtube.com/watch?v=T3hy9E88Ko0).
Cropping uses the original pixels with no resizing or aspect-ratio distortion;
video is re-encoded at CRF 18 and audio omitted. Both crops retain all 31,086 frames,
and their presentation timestamps match the source within one microsecond.
`molo/signer_crops.json` records the crop rectangles and hashes.

The transcripts are a work in progress. Matching filename/time range is a structural
check, not independent proof of every semantic boundary. Preserve unsupported labels
and unannotated intervals; do not label those intervals NO_EMIT. This is Zoom imagery,
not independent portrait-iPhone capture. Exact Citizen visual variants remain unapproved.

## RIT: paired sample, not the full corpus

[Official public sample and metadata](https://latlab.ist.rit.edu/lrec2022/).
The MOV fully decodes to 195 frames at 1920×1080/30 fps. All nine human-timed glosses
fall inside the 6.5-second video. The source filename identifies F13 and the publisher
defines F-prefixed IDs as fluent signers. The EAF participant field says `Kas`; that
field is preserved rather than silently substituted for source identity. Background
text in the preview appears mirrored, so orientation must be resolved before extraction.

The full 45-participant Homework corpus requires institutional Databrary authorization.
CUNY's corpus requires a publisher access request. Neither full corpus was downloaded,
and no email, application, purchase, or access-control bypass was attempted.

## What remains missing

This acquisition materially adds repetitions, longer timed segments, and multiple
signers, but it is **not a complete, directly trainable Apple Vision 100-sign corpus**.
The largest new corpus is pose-only; the new raw material has limited signer/vocabulary
coverage. Exact variant mapping, background supervision, and independent phone
coverage remain unresolved. A new training plan would have to respect those limits.

## Files and checks

- [Épée audit and per-file SHA-256](epee_audit.json)
- [Raw video, EAF timing and provenance audit](raw_video_audit.json)
- [Publisher hashes and crop timestamp checks](integrity.json)
- `verify_acquisition.py`: runnable acquisition checks from the repository root.
- `acquisition_status.json`: completed acquisitions, failed transfers and access limits.

Épée is published under CC BY-NC-SA 4.0. MoLo's collection states CC BY-NC-SA 4.0;
its video description also contains a CC BY-NC 2.0 notice. Both source statements
are retained; this acquisition makes no commercial-license claim. The RIT source
page and access restrictions are preserved with the sample.

## 2026-09-13 follow-up: broader goal remains incomplete

Acquired an additional Daily Moth research clip and EAF from the author's Figshare
release: 75.1 MB, 6.21 minutes, 1920x1080, all 11,166 frames decoded. Publisher MD5
and local SHA256 are recorded. Its 32 first-person-reference annotations are selective,
not complete word/gloss targets. See `daily_moth_audit.json` and `daily_moth_source.json`.
A matching MoLo interview has only one hand gloss across four EAFs; its 2.14 GB
transfer was stopped and the partial file is excluded from acquisition counts.
See [full remaining requirements](DATA_NEEDED.md). No complete-data claim or training.
