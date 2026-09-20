# Strict whole-sign recognition and boundary audit

This report uses complete, non-overlapping sign intervals with at least four raw target observations covering both annotated edges. ASLLRP rows previously marked as crop-incomplete are excluded. O5S5 contributes exact sign cores only; its surrounding context is not treated as fully annotated.

## Curated evidence

Eligible sign cores: **8,367** of 11,936 annotated events. Known: **1,536**; unknown: **6,831**. Boundary windows: **42,111** from complete ASLLRP crops.

## Whole-sign recognition

| Model/input | Split/source | Correct | Accuracy |
| --- | --- | ---: | ---: |
| starting_encoder|core | train|asllrp_contiguous | 25/59 | 42.37% |
| starting_encoder|core | train|asllrp_other_ctc | 324/1040 | 31.15% |
| starting_encoder|core | train|o5s5 | 57/133 | 42.86% |
| starting_encoder|core | validation|asllrp_contiguous | 9/17 | 52.94% |
| starting_encoder|core | validation|asllrp_other_ctc | 125/244 | 51.23% |
| starting_encoder|core | validation|o5s5 | 11/43 | 25.58% |
| starting_encoder|context | train|asllrp_contiguous | 27/59 | 45.76% |
| starting_encoder|context | train|asllrp_other_ctc | 371/1040 | 35.67% |
| starting_encoder|context | validation|asllrp_contiguous | 8/17 | 47.06% |
| starting_encoder|context | validation|asllrp_other_ctc | 116/244 | 47.54% |
| adapted_encoder|core | train|asllrp_contiguous | 52/59 | 88.14% |
| adapted_encoder|core | train|asllrp_other_ctc | 901/1040 | 86.63% |
| adapted_encoder|core | train|o5s5 | 101/133 | 75.94% |
| adapted_encoder|core | validation|asllrp_contiguous | 14/17 | 82.35% |
| adapted_encoder|core | validation|asllrp_other_ctc | 178/244 | 72.95% |
| adapted_encoder|core | validation|o5s5 | 10/43 | 23.26% |
| adapted_encoder|context | train|asllrp_contiguous | 51/59 | 86.44% |
| adapted_encoder|context | train|asllrp_other_ctc | 907/1040 | 87.21% |
| adapted_encoder|context | validation|asllrp_contiguous | 14/17 | 82.35% |
| adapted_encoder|context | validation|asllrp_other_ctc | 164/244 | 67.21% |

## Boundary and rejection evidence

A threshold selected only on training windows gives **61.12%** held-out balanced accuracy for the phase head's known/not-known decision (known recall 43.14%, non-known recall 79.10%). Gloss confidence alone gives **59.57%**.

The original three-way endpoint phase accuracy is **49.79%**. On endpoints inside eligible known cores, gloss top-1 is **521/846 = 61.58%**. Correct gloss plus the train-selected phase threshold localizes **152/261** held-out ASLLRP known events.

Annotation gaps remain named gaps. This audit does not certify them as physical transitions. See `metrics.json`, `curated_manifest.json`, and `videos.html` for full evidence.

No model was trained or promoted. Citizen test remained sealed.

## Interpretation

The strict ASLLRP sign cores are learnable. After continuous adaptation, known-gloss accuracy reaches 901/1,040 = 86.63% on training cores and 178/244 = 72.95% on the held-out signer. Supplying complete sign evidence plus limited context does not improve held-out ASLLRP-other accuracy (67.21%), so missing surrounding context is not the remaining gloss bottleneck.

O5S5 does not transfer: the adapted encoder reaches 101/133 = 75.94% on strict training cores but only 10/43 = 23.26% on LG. That is a signer/source generalization failure even after incomplete and overlapping examples are removed.

Localization remains inadequate. The train-selected phase threshold retains only 43.14% of held-out known endpoints, and correct gloss plus phase localizes 152/261 = 58.24% of eligible held-out ASLLRP signs. Selected video failures also show visible signing inside windows called gaps, which is expected when a trailing window overlaps the preceding sign; these clips cannot serve as pure transition examples.

Decision: keep the 100-gloss vocabulary and do not promote this checkpoint. The next model work should use the strict whole-sign manifest for identity training and a separately reviewed boundary set for localization. Do not derive a physical TRANSITION class from annotation gaps alone.
