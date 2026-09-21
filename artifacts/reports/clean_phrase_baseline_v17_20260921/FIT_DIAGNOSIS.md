# Saved-checkpoint fit diagnosis — 2026-09-21

Both heads fit the training phrases strongly and generalize poorly to the held-out
signers. This is strong evidence of overfitting/generalization failure, rather than
failure to optimize the training objective. It does not isolate whether signer style,
capture conditions, frozen features or the head's reliance on training patterns causes
the gap. No retraining or test evaluation was performed.

| Metric | Seed17321 | Seed17322 |
|---|---:|---:|
| Training known WER (283clips) |2.60%|0.27%|
| Validation known WER (211clips) |50.27%|49.38%|
| Training exact |264/283 (93.29%)|281/283 (99.29%)|
| Validation exact |34/211 (16.11%)|31/211 (14.69%)|
| Local training exact |226/232 (97.41%)|232/232 (100%)|
| Local validation exact |31/199 (15.58%)|30/199 (15.08%)|
| ASLLRP-contiguous training WER |13.04%|2.17%|
| ASLLRP-contiguous validation WER |58.33%|58.33%|

Training and validation source mixtures differ; the within-source comparisons above
confirm that the gap is not merely an aggregate mixture artifact. Training had no
blank-only outputs. Both heads emittedOTHER on all7OTHER-containing training clips,
with full-sequence exact6/7 and7/7; this is fitting evidence only, not UNKNOWN validation.

## Coverage versus generalization

Stage2training contains44known classes; validation contains26. GOOD and MORNING each
occur20times in validation but never as Stage2training targets; BAD occurs once and
also has no Stage2training target. They remain part of the frozen Stage1 vocabulary;
these are not claims of zero exposure anywhere in the complete model's history.

All20GOOD MORNING validation phrases fail in both seeds. However, restricting the
analysis to190validation clips containing only Stage2-seen labels still yields46.24% /
45.47%knownWER, exact34/190 and31/190. This is a diagnostic slice only: no original
validation clip is removed or reassigned, and the baseline selection score stays unchanged.
182validation clips have a target sequence also present in training; only31/182 and
30/182are exact. The problem therefore extends to familiar phrases from held-out signers.

## Phrase and sign errors

- TOMORROW SCHOOL GO:0/19exact in both seeds despite Stage2training examples.
- THANKYOU FRIEND:2/20exact in both seeds.
- HELLO HOW YOU:9/60 and5/60exact.
- PLEASE HELP I:12/60 and13/60exact, with53/65insertions over180reference tokens.
- MY NAME:8/20 and10/20exact.
- SCHOOL:1/19matched reference tokens in both seeds;25training occurrences exist.
- FRIEND:7/27 and8/27matched tokens;31training occurrences exist.
- HOW:35/60 and22/60matched tokens;74training occurrences exist.

Sign-level matching uses the existing deterministic minimum-edit alignment on known
tokens. A missed reference may be a substitution or deletion; unmatched predictions
may be substitutions or insertions. These are sequence-conditioned token recalls,
not isolated-sign classification accuracies. OTHER is omitted from these token counts
and retained in exact-sequence reporting. Small supports limit per-sign conclusions.

## Recommendation

Preserve this baseline and split. More epochs are not supported by the near-perfect
training fit. The next controlled experiment should target generalization, with a
predefined regularization or realistic feature-augmentation change and the same two
seeds/data/metrics. Check available history before choosing the intervention; this
review does not establish that a particular intervention will work. Filling documented
Stage2coverage holes with already-local, established training examples is useful but
cannot by itself explain or fix the large seen-label gap. Do not move validation clips
into training, weaken identity gates or claim that additional data alone is the answer.

## Verification and artifacts

MPS inference on the two immutable saved checkpoints; no optimizer. Data/base/code/cache
manifest verified, both checkpoint hashes verified. Rerun validation predictions and
metrics match every saved prediction exactly. All283training and211validation examples
per seed evaluated. Repeated-token/OTHER counting regression test passes. Compile and
diff checks pass. No protected-test access or new acquisition.

- `fit_diagnostic.json`: complete metrics, predictions, phrase and sign breakdowns.
- `fit_slices.json`: seen-label and shared-target-sequence slices, ranked errors.
- Reproduce primary inference: `venv/bin/python scripts/diagnose_phrase_fit_v17.py`.
