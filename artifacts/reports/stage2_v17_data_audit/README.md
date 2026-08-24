# Stage 2 v17 phrase-source audit

Generated: `2026-08-14T06:07:25.617982+00:00`

## Local corpus

- 780/780 videos probed; 780 unique SHA-256 values.
- Total duration: 0.62 hours across 9 fixed phrases.
- Strict 100-class coverage: 520 videos.
- Coverage if the `FOOD -> EAT` variant is manually approved: 580 videos.
- No signer IDs are present, so a signer-disjoint split cannot be reconstructed from filenames alone.
- Existing phrase/synthetic arrays use legacy v16 schemas and temporal preprocessing and must not feed v17.

## External sources

- How2Sign metadata: 32906 train/validation sentences; 100/100 labels appear as English words.
- How2Sign public files provide English translations, not released CTC gloss sequences; use only for weak/self-supervised work unless labels are created.
- ASLLRP exact query coverage: 76/100 labels and 2313 matching sign occurrences inside real utterances.
- ASLLRP is the preferred supervised source because it has real continuous utterances plus linguistic XML; bulk download requires an ASLLRP account.
- NCSLGR public subset acquired: 166 utterances, 166 verified frontal videos, 198 target-vocabulary gloss occurrences across 17 labels.
- NCSLGR is real frame-aligned supervision but low-resolution and narrow-vocabulary; retain it as supplemental training data.

## Decision

Use the local raw videos after full-length v17 re-extraction, add the acquired NCSLGR subset as supplemental real data, regenerate synthetic sequences from current v17 isolated archives, and acquire modern ASLLRP utterance videos/XML as the primary broad supervised source. Do not spend disk on How2Sign RGB until a weak-label or self-supervised experiment is predeclared.
