# Expanded evaluation audit and native TCN comparison

The augmented fine-tune completed in 2,230.48 seconds (37.17 minutes): cache 677.56s,
training plus validation 1,497.21s, remaining evaluation/check overhead about55.71s.
16 epochs, selected8; calibrationBCE0.652261, validationBCE0.818841. Original12video
recognition7/24 versus9/24 for the first fine-tune. No promotion.

## Expanded set audit

Verified72 unique video IDs,186 reference signs,12ASLLRP/24signs plus60local/162signs.
All local clips are local_signer_02. Every reference matches the current approved source
manifest; per-video edit counts and all supplied WER/correct/exact-video summaries
recompute exactly from saved hypotheses. Actual video hashes checked by row loaders and
again after comparison. No video-hash overlap with either boundary model's training or
calibration; the60local hashes are also disjoint from the familiar-signer split's train
hashes. This is clip-disjoint reused development, not pipeline-wide unseen-signer evidence.
This is a provenance/scoring audit, not a new manual review of every video/transcript.

The supplied membership JSON lists only12videos although evaluation contains72.
A complete72video membership with source hashes is saved beside this report.
The saved ASLLRP boundary_candidates are246/261/287 (all72videos) but must be19/21/18.
This does not change matched-boundary recall or WER. The evaluator source now restricts
candidate counting to annotated rows, so the saved JSON and current code differ here.
Original artifacts were preserved; audit.json records the discrepancies.

Local clips have no curated interval labels: their reported zero gap counts are NOT
measured zero false transitions. Interval/gap metrics are unavailable there. Local
word insertions ARE measurable: frozen15, first fine-tune11, augmented14. Both adapted
models preserve75/121 of frozen correct local signs, despite differing total recall.
There is no standalone default-Reel local60 arm in the supplied comparison; do not
claim an improvement over the default live pipeline from these figures.

## Native TCN comparison protocol

All four existing checkpoints (two seeds, with/without hand geometry) are evaluated,
not selected using this set. Use the shared BoundaryRecognizer.observe streaming path
on the same Apple Vision observations and classify_interval with frozen identical Reel,
no wrist trim, same100ms classifier context and commit rules. Reel provenance must exactly
match the supplied expanded evaluation. Every checkpoint hash and embedded recipe/model
code contract is checked. All72videos are processed for each model; no fitting.

This is conditional full-video composition, not the asynchronous live app scheduler.
TCN retains its native200mslookahead and unscored tail; pretrained models use500mslookahead
and explicit partialEOFpadding. This compares the existing candidates, not a controlled
architecture-only experiment. Unknown local interval/gap metrics remain unavailable.

Comparison completed: all four checkpoints on all72videos in420.37s. Source hashes rechecked at completion; no product defaults changed.

## Results

| Arm | ASLLRP WER | Local WER | Combined WER | Correct /186 | Exact /72 | Insertions |
|---|---:|---:|---:|---:|---:|---:|
| frozen_pretrained_bio | 75.00% | 34.57% | 39.78% | 127 | 17 | 15 |
| finetune_epoch7 | 62.50% | 53.09% | 54.30% | 96 | 5 | 11 |
| augmented_epoch8 | 70.83% | 51.23% | 53.76% | 100 | 2 | 14 |
| TCN seed17621 geometry=False | 79.17% | 59.26% | 61.83% | 89 | 6 | 18 |
| TCN seed17622 geometry=False | 75.00% | 64.81% | 66.13% | 78 | 6 | 15 |
| TCN seed17621 geometry=True | 70.83% | 61.73% | 62.90% | 84 | 3 | 15 |
| TCN seed17622 geometry=True | 66.67% | 58.64% | 59.68% | 99 | 7 | 24 |

The strongest TCN observed here is seed17622/geometry:99correct,59.68%WER,7exact,
24insertions; this is a descriptive ranking, not a promotion/checkpoint selection.
Augmented100correct/53.76%WER/2exact/14insertions; frozen127correct/39.78%WER/17exact/
15insertions. Thus the TCN is close to the augmented model on correct-sign count but
emits more extra words. All adapted candidates trail the untouched pretrained pipeline.
Hand geometry helps one seed and hurts the other on local60; no consistent capacity or
feature superiority is established. TCN ASLLRP results exactly reproduce the earlier
5/6/7/8 correct counts, supporting protocol consistency.

Decision: no promotion, distillation or new training. More epochs did not solve this;
augmentation lowered BCE without improving the old12video recognition count. Preserve
the frozen model as reference. A next diagnostic can apply the unchanged BIO head to
adapted backbone outputs to separate representation drift from the new START/END readout
and decoder before choosing another training objective. This is a proposed diagnostic,
not a result or a launched experiment. NativeTCN can improve but larger architecture
alone is not supported as the next fix. Unknown-transition supervision and incomplete
local interval annotation remain unresolved; insertions cannot all be called gap errors.

Validation: all saved reference/hypothesis edit counts and aggregates recomputed; model,
video, source and Reel provenance checks passed;72 unique video hashes; no boundary
training/calibration overlap. Diagnostic initially compared Python tuple alignments to
JSON lists and stopped; serialization-normalized comparison corrected that audit-only
assertion before inference. Partial summaries skip empty subsets to avoid division by
zero. Existing production/model code and supplied report artifacts were not edited.
