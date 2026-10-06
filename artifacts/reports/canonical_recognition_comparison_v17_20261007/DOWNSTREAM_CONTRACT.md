# Downstream comparison contract

This records the next stages of the authorized comparison. It does not claim they
have been retrained or authorize a deployment or manuscript replacement.

1. **Isolated landmark classifier.** Preserve the historical 96.83% checkpoint.
   Compare the paired mild-roll and full-roll fine-tunes using upright accuracy,
   eight-angle landmark diagnostics and actual selected weight identity. Keep
   the flat Transformer and flat Squeezeformer results separately identified.
2. **Local replay.** Initialize explicitly from the selected landmark checkpoint;
   retain the existing geometry-safe augmentation and class/source balancing.
   Mask only the four mouth landmarks on local examples, not the Citizen/SemLex
   examples or all face landmarks. Reuse the approved local manifests and record
   their hashes. Compare Citizen, SemLex, local validation and orientation retention.
   Local familiar-signer validation does not establish signer generalization.
3. **Multimodal fusion and distillation.** Keep hand features, hand checkpoint,
   teachers, training inputs, seed and selection rule fixed across landmark
   candidates. Rebuild embeddings whenever landmark weights change; old cached
   embeddings cannot represent a new encoder. Pin all source checkpoint and cache
   hashes. Distillation is part of this stage, not proof that the base classifier
   has improved. Report all measured outcomes, including regressions.
4. **Phrase and interval adaptation.** The canonical phrase verifier currently
   reports training_ready=false. Do not reuse historical mixed-data trainer defaults.
   Prepare a recipe-scoped manifest with approved whole-phrase, positive-interval
   and isolated-replay roles; preserve existing ASLLRP/O5S5/reviewed STEM admissions.
   Check split compatibility, overlapping source events, exact targets and auxiliary
   supervision before training. The 494-clip phrase view is not the entire available
   dataset. Do not label unverified gaps as blank or use excluded sources.
5. **Deployment evaluation.** Re-export the actual selected checkpoints and measure
   both precisions on identical inputs. Keep classifier-only latency separate from
   feature extraction, combined recognition and end-to-end processing. The historical
   28 ms measurement cannot be reassigned to newly trained weights.

The same encoder weights cannot simultaneously describe the base, locally adapted,
multimodal and interval-adapted stages. Consistency means a pinned checkpoint for
each named stage and explicit parentage, used identically everywhere that stage appears.
Existing selected checkpoints establish a matched evaluation, not matched training.
No protected Citizen test access is permitted.

Evidence: TRAINING_PLAN.md; current trainer and unified fusion trainer;
docs/ground_truth/stage1-architecture/high.md (local replay retention rules);
artifacts/reports/dataset_reconciliation_v17_20260922/REPORT.md;
active/v17/approved_phrase_manifest_20260921_v2.json and its verifier.
