# Verified-frame refinement: completed training result

All12epochs25–36completed with full coverage, continuing the preceding model/optimizer.
The sole added objective is direct CE on150,876verified sequence-frame labels:
23,454known,103,251OTHER and24,171blank tokens inside0.10s guarded interior gaps.
The remaining86,112tokens are ignored by this CE. Unknown O5S5 context, clip edges,
overlaps, padding and unguarded transitions receive no invented target.

Final TRAIN matches1,263/1,291known signs (97.83%), with3deletions,25substitutions,
1,370insertions and108.29%WER. IsolatedCTC1,897/1,901 (99.79%). Frame supervision
reduces insertions from2,336but leaves substantial sequence errors. All33focused tests
pass including incomplete-annotation refusal, guards, overlap/padding gradients and
batch-partition invariance. Independent review was attempted but hit an agent usage
limit; the root agent directly reviewed the helper, training flow and provenance.

The training-only `curriculum_diagnosis.json` shows original target-normalized CTC
preserves a strong nonblank warm start in the controlled example, although it gets
stuck blank from a weak start. The final six-epoch alignment pass in
`../joint_ctc_aligned_v17_20260915/` therefore restores original CTC strength while
keeping every verified-frame/anchor/pooled/background objective. No held-out outputs
informed this decision. No new data, decoder tuning, Citizen test access or promotion.
