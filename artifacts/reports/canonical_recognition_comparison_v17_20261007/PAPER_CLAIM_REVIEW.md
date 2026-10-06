# Paper claim review — 2026-10-07

Fresh inference independently reproduced every domain count for all five new phases
and the August recognizer rebuilt on the scoped split. Evidence:
paper_claim_verification.json and verify_paper_claims.py. No training or protected test
inference occurred. Core ML FP32 and FP16 were also rerun: each retains 367 correct
predictions and all 378 PyTorch top-1 decisions. Evidence: paper_export_verification.json.

## Verified figures and interpretation

- All five phase rows and both aggregate columns recompute correctly. The pooled
  measure weights local examples at 68.11%; prefer three domain columns in the paper.
- Final isolated-recognition scores: 97.09 / 88.34 / 97.76 percent. These are not
  continuous-sequence accuracy, English translation accuracy or new-signer phone accuracy.
- Fusion selected epoch 0 for every seed: fixed 75/25 normalized score fusion. Its
  gains cannot be attributed to the fusion-stage distillation training. Later phrase
  and interval adaptation remain trained stages.
- Phrase segment accuracy 67.20 to 75.20 percent is valid on automatically partitioned
  transcript segments (equal-width assignment), not manually timed sign boundaries.
- Recomputed selected tuning WER: new 71/226 = 31.4159%; clean August 68/226 = 30.0885%.
  New within-chain baseline is 37.6106%. These are selection-pool gloss WERs.
  The new chain is 1.33 percentage points worse than the clean August control on WER.
- The original August recognizer is not the matched clean comparator. Its phase
  training inherited lab tune/test overlap; do not use its streaming score as held out.
- The new isolated parent has HIGHER SemLex accuracy than the August isolated parent
  (87.22 vs 85.79), contradicting “1–2 points lower throughout.” Later stages have
  lower SemLex accuracy; benefits vary by domain.
- Conversion preserves top-1 decisions, not necessarily identical numerical logits.
  Package sizes: 50.280376 MB FP32 / 25.506008 MB FP16, decimal MB.
- Phone medians recomputed from per-frame samples using the benchmark's upper-middle
  sample convention: new FP16 23.69 ms, old FP16 24.25 ms, new recognizer-FP32 59.15 ms.
  Both other visual components remain FP16. This is NOT an all-FP32 pipeline comparison.
  New FP32-recognizer configuration takes about 2.50 times the new FP16 time. The small
  new/old FP16 difference is within observed pass spread; no demonstrated speedup or
  formal statistical equivalence. Scope: recorded-frame preparation and recognition,
  excluding camera capture/loading/UI/English generation.

## Scoped manifest review

Canonical verifier passes and remains training_ready=false. Scoped manifest hash and
canonical hash match recorded recipe; 179 train and 139 validation identities have the
correct approved roles, no cross-split overlap, no lab-test membership, and no lab-tune
membership in training. All 636 RGB/hand file hashes and symlink-view membership verify.
Both recognizer training-video lists are subsets of approved recipe training; selected
epochs and WER arithmetic match recorded histories.

The valid wording is “excluded from the new scoped stages' training and selection.”
Do not say the lab test was never touched in all experiments: historical-recipe
reproduction deliberately reused the older contaminated split. This review used only
test identity metadata to verify exclusion, not held-out inference.

One provenance claim needs correction: the manifest says span labels and captured span
inputs are hash-pinned by recipe.json. That file pins isolated embedding caches, base
weights, code and span configuration, but not those auxiliary span files; span history
records training video identities, not their derived file hashes. Fresh metric
reproduction succeeds, but this is a reproducibility gap, not a fully immutable input
record. Do not modify the historical manifest silently; use an additive audit/version
if completing that provenance. No new training authorized by this review.

## Proposed presentation (not applied)

Use one five-phase table with three validation domains; omit redundant pooled and mean
columns. Explain domain adaptation, fixed fusion and the isolated/sequence tradeoff.
Use one compact development-sequence table with before/after WER and the clean August
control. Keep conversion/storage and phone timing together, stating precision scope.
Do not add another chart duplicating each table. Keep debugging/reproduction details
in repository evidence. The evaluated candidate has not replaced the production default.
