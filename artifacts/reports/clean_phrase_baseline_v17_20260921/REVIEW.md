# Clean phrase baseline review — 2026-09-21

Both MPS seeds completed18epochs in12.40seconds combined after12.72seconds of frozen
feature preparation. This establishes the requested baseline; recognition remains weak.
No model is promoted. Results are reused development validation, not test accuracy.

| Metric on211 validation clips | Seed17321 | Seed17322 |
|---|---:|---:|
| Selected epoch |8|11|
| Overall known WER |50.27%|49.38%|
| Exact phrases |34/211 (16.11%)|31/211 (14.69%)|
| Substitutions / deletions / insertions |92 /119 /71|70 /114 /93|
| Blank-only clips |15/211|8/211|
| Clips emitting OTHER |0|1|
| Local known WER (199clips) |49.91%|48.98%|
| Local exact |31/199|30/199|
| ASLLRP-contiguous known WER (12clips) |58.33%|58.33%|
| ASLLRP-contiguous exact |3/12|1/12|

Selection follows the preregistered mean of local/ASLLRP source WER, earliest epoch on
ties. Overall WER instead pools561reference tokens. These are different aggregations.
Known WER omitsOTHER; exact phrases and OTHER-emission counts retain it. Validation has
no OOV targets, so these results provide no UNKNOWN recall/generalization measurement.

Missing signs remain substantial:119/114deleted tokens out of561references. Blank-only
outputs do not account for all failures; substitutions and insertions remain as well.
Both seeds exhibit similarly poor held-out recognition, but two seeds do not establish
a statistical confidence interval. The12ASLLRPclips are especially small support.

The untrained-head diagnostics had487.52%overall WER, dominated by repeated insertions.
Their improvement is a sanity check that the head learned, not a comparison against a
trained prior baseline. Historical v16/v17 scores used different splits, data or recipes
and are not controlled before/after comparisons. Weak results alone do not prove that
remaining annotations are wrong, the model cannot learn, or more data alone will fix it.

Verification: recomputed all211predictions per seed; independently checked total token
edit distance; checked18history entries, selected epochs, checkpoint and unchanged frozen
base hashes, MPS recipe, manifest digest and no-test flags. Review used saved predictions
and artifacts; no new training or protected-test evaluation. Evidence:
`review_verification.json`, `results.json`, `status.json`.

Recommendation: preserve both checkpoints as the clean-data reference. Next compare
training-set versus validation-set performance for these saved checkpoints and inspect
errors by phrase/class. Poor training performance would direct attention to fitting and
representation; a large train/validation gap would direct attention to generalization
and coverage. Neither diagnosis is established by the current held-out metrics. Do not
increase epochs, reacquire data or change the architecture based on this result alone.
