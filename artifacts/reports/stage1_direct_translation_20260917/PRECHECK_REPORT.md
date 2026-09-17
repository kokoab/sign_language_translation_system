# Stage 1 direct-translation precheck

Prepared the authorized training experiment. **No trained-model result yet.**

- Our Stage-1 encoder's unpooled sequence -> linear projection -> ASL-pretrained
  mT5 encoder-decoder. Existing isolated classifier retained as an auxiliary task.
- No CTC, gloss commitment, or repeat suppression; repetition is not a rejection gate.
- 994 continuous English-paired training clips; 1,901 isolated training examples.
- 12 paired validation clips, nine qualitative diagnostics, 1,356 isolated validation
  clips. All 4,272 input hashes verified; official Citizen test stays sealed.
- Adaptation excludes evaluation signer IDs 1/2. The externally pretrained text
  component prevents a claim of fully signer-disjoint pretraining.
- Maximum 256 source tokens and 70 target tokens; no English target truncation.
- Two focused tests pass: actual encoder/decoder gradients, valid-window masks,
  isolated-head gradient, target-free generation and complete deterministic coverage.
- Initial SemLex split-path check was corrected to its existing dataset-specific
  roots; preparation then passed. No model trained during that failed preparation.

Fixed run: seed17111, 20 epochs, projection-only first epoch followed by joint
training; translation CE + 0.5 isolated CE; Adafactor. Full-model gradient/tiny-fit
preflight precedes training, and its weights are discarded. Compare final epoch 20
with initialization, unchanged Uni-Sign, and a zero-visual control. Save all outputs,
losses, coverage, retention, checkpoint/optimizer and provenance.

The detached supervisor waits for worker exit without polling and sends one macOS
notification. Success writes REPORT.md/summary.json; failure writes FAILURE.md.
Both write completion.json. No automatic runtime promotion.
