# Stage 1 direct translation experiment

**Goal:** Test the user's approved combination: retain our isolated encoder and head,
train continuous-video-to-English without CTC, and report measured results.

**Architecture:** Apple v17 windows -> existing Stage 1 unpooled frame tokens ->
linear projection -> pretrained mT5 encoder-decoder -> English. The original
isolated head supplies an auxiliary classification loss. Preserve all window order;
never collapse an utterance to one 32-frame sign. No repetition suppression.

**Scope:** Completed utterances at Finish, research only. Existing experiment branch
and workspace; preserve unrelated edits. No production replacement or mobile claims.
The user approved implementation and detached execution, with no assistant polling.

**Initialization:** Existing runtime Stage 1
`stage1_v17_phrase_adapt_reel_v2/best_model.pth`; complete mT5 component from the pinned
released Uni-Sign How2Sign checkpoint. Reuse its ASL translation pretraining, not its
pose encoder. This is a hybrid adaptation experiment, not a Uni-Sign reproduction.

**Data:** Reuse acquired, realigned How2Sign TRAIN sentence pairs and existing
Apple v17 continuous archives. Exclude filename signer IDs 1 and 2 (the evaluation
signers). Verify each archive against the source hash and schema, report exclusions.
Keep all valid continuous windows in chronological order. Invalid windows remain
masked; no synthetic blank labels or invented translations. Isolated replay uses
the existing verified 1,901 TRAIN / 1,356 validation manifests only; Citizen test
stays sealed. The existing 12 realigned paired validation videos and nine qualitative
diagnostics get Apple features under the same continuous extraction contract.

**Qualification:** Adaptation train/evaluation signers are disjoint. The inherited
Uni-Sign mT5 was pretrained/fine-tuned externally; whole-system signer disjointness
cannot be claimed. This is an in-domain How2Sign pilot with webcam diagnostics, not
evidence of conversational/iPhone reliability.

**Fixed recipe:** Seed 17111; 20 epochs; batch two utterances; first epoch trains
the projection with the encoders frozen, remaining 19 jointly train Stage 1,
projection and mT5. Translation token cross-entropy plus 0.5 isolated CE. Adafactor
with explicit learning rates: Stage 1 1e-5, projection 3e-4, mT5 1e-4. Gradient norm
clip 1. Deterministic complete training coverage each epoch; spread isolated replay
over the epoch without dropping any replay sample. English targets are never used
by generation. No dev-based early stopping or checkpoint selection: evaluate final
epoch 20, plus the initialized hybrid and unchanged Uni-Sign baseline.

**Checks and results:** Unit checks for variable-length masks, differentiability,
generation without targets and isolated-head gradients. Real-model preflight checks
gradient flow and train-only tiny-fit loss reduction, then resets initialization.
Report BLEU/chrF, per-video predictions/references, isolated retention, training
coverage/losses, and a zero-visual-input generation control to expose language-only
guessing. Repetition is reported but is not an automatic rejection criterion.
Save latest and final checkpoints, hashes, optimizer state and provenance. One
detached worker produces REPORT or FAILURE and a macOS exit notification. No polling.

## Tasks

- [x] Write failing core checks and implement the minimal direct-translation module.
- [x] Build/freeze paired data and reuse the verified extraction/replay helpers.
- [x] Implement fixed training, evaluation, provenance and detached exit reporting.
- [ ] Run focused checks, record precheck, then launch as the final action.
