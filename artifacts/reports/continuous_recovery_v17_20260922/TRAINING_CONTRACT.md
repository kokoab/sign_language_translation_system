# Bounded combined-data frozen versus adapted comparison

This implements steps2–4of the user-approved recovery plan. The user instructed us to
continue after the frozen/adapted comparison was described. This is research training;
it does not replace the live model, change admission of old recipes, or finish step5.

## Question and material difference

Does allowing the current Reel landmark encoder to adapt improve recognition on the
cleaned combined supervision, compared with the same encoder held fixed?

Earlier experiments already compared frozen/joint training, used unpooled tokens, tried
positive-core supervision and repaired CTC blank collapse. Those ideas are not new.
The new condition is the pinned20260922combined input contract: all finalized isolated/
segmented/reviewed positive sources plus the494approved phrase/subspan records, without
the excluded whole-sequence targets used by older recipes. Both arms share every choice
except whether the base parameters receive optimizer updates. Results compare these
arms; they do not attribute improvement over older reports to a single data source.
The prior joint experiment already used the same proposal initialization chosen below;
new initialization is not claimed as a difference.

## Initialization and model scope

Checkpoint: artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth
(SHA25625a4e0551b190890712c79be549ed2381386d72d7c6c0611565e8b538125db3c).
Use its model_config and model_state_dict with strict loading and exact locked100 mapping.
This is the current Reel proposal (Stage1PhraseAdaptReelV17 export lineage), not the
different landmark branch inside its multimodal verifier. This comparison does not
retrain the RGB hand branch or fusion verifier. Preserve the full live system unchanged.
Preflight must verify correspondence to current Reel initialization and record hashes.

Reuse JointCTC and UnifiedStreamingCTCHeadV17. Input is ordered unpooled256dim landmark
tokens plus100class logits; head hidden128/3causal blocks, output102(blank0, known1–100,
existing OTHER101). Do not give target text to inference or add a phrase-language model.

## Inputs and time contract

Consume only combined_dataset_v17_20260922/manifest.json and its pinned membership.
Verify feature/raw/source-manifest hashes, roles and vocabulary. Keep4547train/1874val.
All1975windowed records have contiguous nonoverlapping ranges; maximum8windows/phrase
and2windows/single-sign record. Reuse original normalized32frame windows in source order;
keep32ordered tokens/window. Never feed them to normalize_time_window, which expects
unnormalized Vision observations. Do not invent source timestamps for isolated tensors.
No concatenation of separate isolated examples into purported natural phrases.

This preserves the cached window input contract and tests recognition. Within-window
encoding sees the complete window; it is not a frame-causal encoder. No latency claim,
0.5s guarantee or direct live export follows. Short tails contain resampled evidence,
not32independent observations. Shorter live windows require a separate matched test.

## Fixed training comparison

- Two seeds17421/17422, frozen and adapted arms per seed;12epochs each.
- Identical initial base/head state and deterministic minibatch order between arms.
  Reset Python/NumPy/Torch RNGs after model construction in each arm, matching head
  dropout draws as well as sampling; this does not promise bitwise MPS determinism.
- Base eval mode in both arms: same dropout policy; adapted weights still receive gradients.
  Head trains normally in both arms. Frozen base requires_grad=false.
- Every epoch visits all4264single-sign training records once in shuffled batches32.
  Pair each update with4phrase records, cycling reshuffled283phrase records as needed.
  Uniform sampling within each group; log actual visits, including repeats. This balances
  the two supervision groups, not each source or signer individually.
- Loss: phrase target-normalized CTC + single-sign target-normalized CTC +0.25single-sign
  identity CE. Single-sign CE averages pooled window logits within the record before loss,
  so a two-window sign remains one supervised identity, not two independent signs.
- Shared-parent supplemental records get fixed0.5weight in single-sign losses, others1.
  This is a conservative correlation discount, not statistical independence or full parent
  deduplication. Retain all records and report their139cross-representation relationships.
- AdamW: head lr0.002, adapted base lr0.00003, weight decay0.0001, gradient norm cap5.
  No hyperparameter search, extra background clips or inferred OOV supervision.
- MPS model, CPU CTC only (differentiable transfer), CPU threads2, MPS fraction0.35.

## Measurements and selection

At initialization and every epoch: per-source validation knownWER with S/D/I/reference
counts, exact transcript including OTHER, single-sign CTC exact and base pooled top1.
Known WER removes OTHER only after CTC collapse. No aggregate claim of signer-disjoint
accuracy across SemLex/shared-signer rows. Measure initial isolated retention with these
same inputs; do not substitute an archived differently processed score.

Select each arm by mean(local_phrase knownWER, ASLLRP_contiguous knownWER), earliest tie.
Keep actual source results visible because smallASLLRPvalidation cannot support strong
conclusions. Report both seeds. Do not auto-promote or silently trade isolated accuracy
for sequence gains; defer such a deployment decision until measured comparison exists.
This dataset does not yet certify hold/repeat behavior or unseen-OOV rejection.

After each arm, reload its selected best checkpoint and evaluate every training record
once with the same metrics. Pair these results with that checkpoint's validation results
to distinguish fitting failure from generalization failure; training metrics never select
the checkpoint. Save per-epoch history and actual source exposure counts as the run proceeds.

## Preparation, provenance and execution

Dedicated runner: scripts/train_combined_frozen_joint_v17.py; focused tests:
test/test_combined_frozen_joint_v17.py. Dedicated recipe:
active/v17/combined_frozen_joint_manifest_20260922.json. Old generic gates stay unchanged.

Before training: check all inputs, finite forward over all7367windows, training-only
representative gradients for both arms, frozen-base gradients absent, adapted-base/head
gradients finite/nonzero, no optimizer steps. Verify the complete phrase loss receives
whole targets and single-sign CE never consumes phrase labels. Pin runner/dependency,
checkpoint, recipe and prepared-cache hashes; training refuses stale preflight.
The gradient batch uses the32longest training single-sign records and4longest training
phrases. Every existing gradient must be finite, with nonzero gradients in both trainable
groups. Every record uses its unique feature path, never a potentially repeated signer ID.

Run focused tests, review the runner and preflight, then launch detached with caffeinate
and completion/failure notification. No polling training. Save immutable best checkpoints,
per-epoch metrics, coverage, status and results under combined_frozen_joint_v17_20260922.
Original datasets/models and concurrent UI work are untouched. Record any preflight
failure and resolve it before launching; fail-closed is not a reason to reuse old defaults.
