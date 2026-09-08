# Cokely locked100 continuous-video acquisition

> **Superseded by the [v2 verification audit](../continuous_reel_v17_cokely_v2/README.md).**
> The 31-clip count below ignored utterance and supplementary-hand boundaries and
> split maximal runs to satisfy an arbitrary minimum. The corrected result is 22
> candidate runs, 85 tokens, three signers, and zero training-eligible clips.

Acquired 2026-09-08. The dataset now has **31 non-overlapping continuous clips**
from five public recordings and four identified signers, enough to begin the next
compatibility and feature-extraction step. These clips are candidates, not yet admitted
training samples.

## Acquired data

| Item | Result |
| --- | --- |
| Timed ELAN annotation files | 6; 1,032,898 bytes |
| Source videos needed by strict candidates | 5; 218,097,458 bytes |
| Source video format | 640×360, 29.97 fps, H.264 |
| Extracted continuous clips | 31; 3,829,618 bytes |
| Identified signers | 4 |
| Recordings | 5 |
| Target tokens | 97 |
| Distinct target sequences | 23 |
| Locked100 classes represented | 23 |

The source is the [Dennis Cokely Parallel Corpus](https://encompass.eku.edu/cokely_videos/),
which publishes ASL video, timed ID-gloss ELAN annotations, and signer attribution.
It is licensed CC BY-NC-SA 4.0, so this material is restricted to non-commercial use
with attribution and share-alike terms.

The acquired signers are David Hamilton, Patrick Graybill, Mark Morales, and
MJ Bienvenu. Source videos and clips live under
`data/local/continuous_phrase_sources_v17/cokely/`. Original EAFs, the generated
manifest, provenance, hashes, and review frames live in this report directory.

## Selection rule

The selector uses exact, case-sensitive Citizen raw labels from the frozen100
manifest. It does not normalize spelling, accept aliases, or collapse numeric
variants. Any unknown annotation or missing timestamp breaks a run. A run must contain
at least two consecutive locked labels, with no more than 300 ms between annotated
signs. Runs longer than six signs are split only at an annotation boundary; this turns
27 maximal runs into 31 non-overlapping clips.

The set covers 23 classes and 23 distinct sequences. It is imbalanced: 50 of 97 target
tokens are `DIFFERENT`, reflecting repeated annotations in one source recording.
That repetition must be weighted or capped during training.

## Verification

All source files have recorded SHA-256 hashes. Each source is 640×360 at 30000/1001
fps. Source durations exceed every referenced EAF endpoint. Every extracted clip was
decoded and probed; its duration agrees with the annotation interval within 80 ms.
Representative frames from all four signers were visually inspected and show active,
centered signing at the selected times. The five reviewed frames are in `previews/`.

The failed acquisition attempts are also material: the publisher initially returned
HTTP 403 to stateless requests. Its normal public page-cookie flow retrieved all EAFs.
The public 640×360 HLS renditions supplied the five videos. No account, protected
split, or authentication bypass was used.

## Remaining admission gate

Cokely uses a project-specific ID-gloss lexicon. An identical English label does not
by itself prove the same visual variant as the Citizen ASL-LEX class. Therefore every
row in `manifest.json` deliberately has `variant_verified: false` and
`training_eligible: false`.

The next step is to extract v17 features and run the frozen Stage-1 model over the
annotated signs as a compatibility screen, followed by visual review of disagreements.
Only compatible clips should enter a Stage-2 train-only supplement. Existing held-out
ASLLRP and local evaluation sets must remain untouched.

## Reproduce

```sh
venv/bin/python scripts/prepare_cokely_continuous_v17.py \
  --annotation-root artifacts/reports/continuous_reel_v17_cokely_v1/annotations \
  --video-root data/local/continuous_phrase_sources_v17/cokely/source_videos \
  --clip-root data/local/continuous_phrase_sources_v17/cokely/locked100_clips \
  --output artifacts/reports/continuous_reel_v17_cokely_v1/manifest.json

venv/bin/python -m unittest test.test_prepare_cokely_continuous_v17 -v
```

`manifest.json` contains the source pages and stream URLs, source and clip hashes,
signer IDs, annotation IDs, exact time ranges, raw and canonical sequences, and video
probe results for all 31 clips.
