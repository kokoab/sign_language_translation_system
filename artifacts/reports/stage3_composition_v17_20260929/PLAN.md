# Stage 3 model composition repair — 2026-09-29

User authorized fixing the translation model, without triggers or hardcoded phrase fixes.
This is text-only fine-tuning of the current 15.6M-parameter slot T5. Stage 1/2 and visual
phrase data are outside this recipe. Approved phrase verifier passes 494 hashes but remains
training_ready=false; no visual phrase trainer or manifest is changed or bypassed.

Inputs: 16,070 existing training rows plus 17,225 generated training-only compositions.
All old non-training gloss sequences remain reserved. Neither user-reported phrase is in
training. New material covers greetings, addresses, questions, names, requests, time
attachment and concatenated clauses. No external API, acquisition, or protected test run.
Generated phrases are not validation/model-selection truth. Existing held-out/test labels
are not used for targets. Exact user regressions and saved real-session outputs are development
diagnostics; additional generated probes carry no held-out accuracy claim.

Fixed recipe: initialize stage3_v17_asl_order_fs_v1, seed 17929, five epochs, batch 32,
Adafactor learning rate 0.0002, gradient norm 1, MPS, final epoch only. Refuse truncation,
nonfinite loss and cross-split input overlap. Save new weights separately; preserve original.
Run detached, save completion status, and deliver macOS completion/failure notification.

After completion: compare direct neural outputs against the old checkpoint, inspect all
26 phone histories plus user phrases, and check additional compositions, names, negation,
pronouns and time attachment. Repair must reproduce both requested meanings without a
phrase override. Export to Core ML and verify output parity before mobile integration.
New model contract disables existing reviewed-phrase overrides. No semantic trigger,
output-rewriting rule, greeting lookup or new confidence gate is introduced. Existing
spelling-slot restoration and technical failure handling remain model-independent plumbing.

Integration refinement: the composition model receives the full finished utterance and learns
punctuation from its training targets. Its input contract bypasses legacy pause splitting;
there is no phrase-specific segmentation rule. Over-capacity input uses existing literal
fallback rather than silently discarding words. Signing recognition remains separate. No independent real-signer translation accuracy can be inferred from text probes.
