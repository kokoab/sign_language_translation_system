# Bounded pretrained boundary probe

User requested continued work and specifically pretrained boundary weights to fine-tune.
Preserve Reel and the locked vocabulary. No dataset acquisition or protected test access.

1. Pin author code/weights and verify strict CPU/MPS synthetic forward. Complete:
   commit22ca3a6f63b6f031bfb1c0d717fcb259143ba7db,147tensors,5.734Mparameters.
2. Use the existing12development videos, real MediaPipe Holistic preprocessing and
   upstream normalization/decoding, then frozen Reel on resulting intervals. Preserve
   input aspect ratio; measure actual outputs and mark this full-sequence/offline only.
3. Audit local saved proposal/verifier decisions separately. No threshold sweep or
   automatic promotion. Verifier-only counterfactual changes the score authority as
   well as the proposal veto; measure both added and removed outputs.
4. Record outcome and feasibility of ASL fine-tuning. A new training recipe must pin
   valid MediaPipe features and interval roles; Apple61x5 cannot be fed directly into
   this50x6 checkpoint. Existing generic phrase gate stays disabled.

No claim that checkpoint availability establishes an accurate boundary model for ASL.

- User-authorized speed update: resume850checkpointed records,three independent video
  processes for271remaining. Three-video smoke and resume corruption checks pass.
  DetachedPID38560 replaces serialPID69981; completion notification retained.

- Preparation and structural validation COMPLETE:1121records,889train/232validation,
  4310intervals,190797frames; no missing files or validation errors. See validation.json.
  User requested next-step discussion. No training launch; dedicated bounded-context
  start/end adaptation recipe remains the next decision.
