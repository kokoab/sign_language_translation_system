# Historical Transformer run audit — 2026-10-07

Verdict: the six saved classifiers reproduce their published validation Top-1,
Top-5 and macro-F1 exactly. They are usable single-seed, common-recipe baselines.
They do not establish that Transformer or part-wise Transformer architectures have
been adequately optimized, or that Squeezeformer is generally superior.

| Saved model | Reproduced Top-1 | Best epoch | Completed epochs |
| --- | ---: | ---: | ---: |
| Flat Transformer | 95.50% | 97 | 127 |
| Part-wise Transformer | 93.92% | 105 | 135 |
| Conv-augmented Transformer | 94.18% | 59 | 89 |
| Anatomical-token Transformer | 93.12% | 92 | 122 |
| Compact Transformer | 94.71% | 52 | 82 |
| Part-wise/global Squeezeformer | 96.30% | 57 | 87 |

## What was verified

- Loaded each saved state dictionary strictly into its corresponding architecture and
  independently evaluated it on CPU. All reported classification metrics reproduce;
  parameter counts and aggregate/per-model results agree. Outputs are finite, including
  an empty-input probe. Trained Transformer temporal layers differ from each other.
- All histories use seed 1701 and the same recorded data/sampling summary. Epochs are
  contiguous; each checkpoint corresponds to the first highest Top-1 epoch. Each run
  ended after exactly 30 epochs without a new best, within the 160-epoch ceiling.
  Every recorded learning rate matches the declared warm-up/cosine schedule.
- Current source uses shared AdamW settings, label smoothing, augmentation defaults,
  balanced sampling, EMA selection and CPU validation. The same schema-checked loaders
  resolve the recorded data counts. No protected test features were loaded.
- Citizen train and validation participants are disjoint in the saved provenance.
  No train/validation path overlap or exact raw-video hash overlap was found, including
  SemLex training versus Citizen validation. This is not an identity audit across corpora
  or a near-duplicate video search.
- The original SemLex candidate manifest still has false eligibility flags. An initial
  audit assertion caught this. The September 22 admission explicitly documents and
  supersedes those historical flags without modifying the old manifest. Every SemLex
  training path and feature hash exactly matches that later approved manifest. This is
  a resolved provenance issue, not evidence that these runs used an unapproved supplement.

## What the runs cannot prove

1. Each design has one training seed. Squeezeformer gets 12 validation examples right
   that the Transformer misses; Transformer gets 9 right that Squeezeformer misses.
   The net accuracy difference is small and does not establish a robust architecture win.
2. The same optimizer recipe is a useful baseline, not architecture-specific tuning.
   The part-wise design was actually trained, but its lower score does not rule out
   a better fusion, initialization, regularization or training recipe.
3. This is a whole-design comparison: eight Transformer global layers versus four
   Squeezeformer-style blocks, different internal operations and stochastic-depth behavior.
   Similar parameter counts do not isolate the effect of convolution or regional streams.
4. Shared augmentation settings do not mean identical augmented tensors. The sampler
   has its own seeded generator, but augmentation and dropout consume shared device RNG
   state; different architectures can receive different random transformations.
5. Historical family checkpoints contain family, class count and weights, but do not
   embed full optimizer/RNG state, label maps, immutable data manifests or source hashes.
   Present-day re-evaluation and history consistency are verified; bit-for-bit reconstruction
   of the original training is not established. Current hashes are archived by this audit.
6. These variants were selected on the same repeatedly used validation split. Reaching
   96% there would be a development result, not new independent test accuracy.

The earlier log wording that the part-wise result “rejects” anatomical splitting as
an explanation, or establishes the convolution/attention combination as the cause,
overstates a single-seed whole-design comparison. Preserve the measured scores and
qualify that interpretation in future work.

## Next experiment

No new training occurred during this audit. Existing authorization remains: improve
Transformer, report all attempts, and perform a matched downstream comparison.
The next bounded experiment should retain flat and Squeezeformer controls, explicitly
pin inputs/configuration, and test a residual part-wise branch that preserves the
trained flat path. Matched fine-tuning is a fast screening step; multiple fine-tuning
seeds from one checkpoint must not be called independent from-scratch replications.
Any selected candidate still needs the matched multimodal/distillation/interval stages
and real-device comparison before a production or manuscript replacement.

Evidence: `audit.py`, `audit.json`, `current_input_manifest.json`, `audit.log`, and
`audit_initial_eligibility_flag.log`. Admission resolution:
`artifacts/reports/supplement_finalization_v17_20260922/REPORT.md` and `semlex.json`.
