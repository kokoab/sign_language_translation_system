# Cokely locked100 verification audit

Verified 2026-09-08. **Twenty-two runs passed structural and motion screening;
zero are admitted for training.** This report supersedes the provisional v1 count.
The locked 100-gloss vocabulary and all protected evaluation sets remain unchanged.

## Corrected selection

The parser now reads the main `ASL-individual-cp` tier, both supplementary hand
tiers, and `ASL-TT` utterance boundaries. It keeps maximal runs inside one utterance.
OOV labels, missing timing, conflicting simultaneous hand annotations, gaps over
300 ms, and utterance boundaries break a run. Matching simultaneous hand labels are
duplicate evidence. Sequential repetitions remain separate for review.

| Result | Count |
| --- | ---: |
| Main-tier annotations audited | 2,253 |
| Candidate maximal runs | 22 |
| Candidate tokens | 85 |
| Identified signers | 3 |
| Recordings | 4 |
| Distinct locked classes | 18 |
| Distinct raw sequences | 18 |
| OOV main annotations rejected | 2,019 |
| Isolated locked tokens rejected | 132 |
| Conflicting supplementary-hand annotations rejected | 16 |
| Missing/invalid timing rejected | 1 |
| Training-eligible runs | 0 |

The v1 script produced 31 clips by splitting long runs at six tokens and enforcing a
30-clip minimum. Those controls are removed. David Hamilton no longer contributes a
multi-sign run once utterance boundaries are honored.

## Visual and lexical review

All 22 candidate clips decode and show active, centered signing in the five-frame
[motion contact sheet](motion_contact_sheet.jpg). The set remains unsuitable as a
training supplement: 66/85 tokens are inside immediate repeated-label runs, including
50 `DIFFERENT` tokens. These may encode discourse repetition or aspectual movement;
the EAF alone does not establish the intended CTC token count.

The [variant reference sheet](variant_reference_contact_sheet.jpg) compares one
representative Cokely segment per represented class with a frozen Citizen **train**
example for the exact raw gloss and ASL-LEX code. It is a review aid, not proof of
lexical identity. Cokely uses its own ID-gloss lexicon and provides no crosswalk to
the Citizen ASL-LEX codes, so `variant_verified` remains false for every row.

## Feature and frozen-model compatibility

One smoke run per signer succeeded before full extraction. All 22 runs then produced:

- overlapping 32-frame Apple Vision v17 landmark and hand-RGB windows, stride 16;
- MobileCLIP2-S0 hand embeddings; and
- frozen 612-dimensional evidence from the selected Reel Stage 1 checkpoint.

All stages completed with zero failures. The frozen checkpoint SHA-256 is
`278a9933df25508aa83823cf6a8e050fbf4ba729eb39b1c790c97e55032a6558`.
The feature archives are under
`data/local/continuous_phrase_sources_v17/cokely/verified_v2_stage2_*`.

Segment-level Stage 1 agreement is weak: **15/85 top-1 and 26/85 top-5**. On the 19
tokens outside immediate repeated-label runs it is **4/19 top-1 and 5/19 top-5**.
This screen is diagnostic only: disagreement can reflect rapid contextual signing,
domain shift, imperfect annotation boundaries, or a visual variant mismatch. Model
agreement never grants training admission. Detailed rows are in
`stage1_compatibility.json`; extraction hashes and schema are in
`extraction_full.json` and `manifest.json`.

## Decision

This source does **not** provide enough verified data for the next training step.
The 22 runs stay in `candidate_verification`, grouped by entire recording/signer, with
`training_eligible: false`. The useful next data action is another source with timed
two-hand annotations and an explicit lexical crosswalk, or expert review of these
21 signer/class pairs plus the repeated-token interpretation. Training or runtime
changes from this set would overstate the evidence.

The source is the [Dennis Cokely Parallel Corpus](https://encompass.eku.edu/cokely_videos/),
licensed CC BY-NC-SA 4.0. Source files, EAFs, hashes, signer IDs, annotation IDs, exact
times, source groups, raw labels, canonical labels, and ASL-LEX target codes are
preserved in the manifest.

## Reproduce

```sh
venv/bin/python scripts/prepare_cokely_continuous_v17.py \
  --annotation-root artifacts/reports/continuous_reel_v17_cokely_v1/annotations \
  --video-root data/local/continuous_phrase_sources_v17/cokely/source_videos \
  --clip-root data/local/continuous_phrase_sources_v17/cokely/verified_v2_clips \
  --output artifacts/reports/continuous_reel_v17_cokely_v2/manifest.json

venv/bin/python scripts/extract_stage2_multimodal_v17.py \
  --manifest artifacts/reports/continuous_reel_v17_cokely_v2/manifest.json \
  --output-root data/local/continuous_phrase_sources_v17/cokely/verified_v2_stage2_rgb \
  --report artifacts/reports/continuous_reel_v17_cokely_v2/extraction_full.json \
  --role candidate_verification --source cokely_verified --window-stride 16

venv/bin/python scripts/audit_cokely_stage1_compatibility_v17.py \
  --manifest artifacts/reports/continuous_reel_v17_cokely_v2/manifest.json \
  --crop-root data/local/continuous_phrase_sources_v17/cokely/verified_v2_stage2_rgb \
  --frozen-root data/local/continuous_phrase_sources_v17/cokely/verified_v2_stage2_frozen \
  --output artifacts/reports/continuous_reel_v17_cokely_v2/stage1_compatibility.json
```
