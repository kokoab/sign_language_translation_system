# Authorized pretrained ASL boundary adaptation

User approved implementation and requested sufficient epochs,then an under30minute target
if possible. Keep frozen Reel and all admitted training sources/signer splits.

- Two seeds17621/17622. Maximum120epochs; minimum40 before convergence stopping;
  patience20epochs,minimum meaningful calibration improvement0.0001.
- First5epochs train new start/end outputs; then unfreeze all4pretrained attention blocks
  at low LR. Freeze CNN and its normalization; cache their outputs.
  This adaptation preserves the existing architecture; speed does not restrict temporal learning. Final batch128 benchmark:0.28446s/step,about66.85s/trainingepoch. Two seeds near40epochs
  roughly1.5–2hours total; near120 roughly5hours including caching/calibration/evaluation.
  No hard30minute deadline; user subsequently prioritized accuracy.
- Input20Hz past-observation sampling;64frame windows,targetindex53,10future frames
  (0.5s). Normalize only that window. Prefix/explicitEOF padding has zero confidence;
  EOF uses available evidence,never forces an END. All windows share this contract.
- Source4310events/1121poses already validated. Start/end positives and interior negatives
  reuse boundary_targets; unknown regions ignored. Train-parent-held calibration only
  selects checkpoints. Validation is never used for thresholds/checkpoint selection.
- Train every supervised20Hz window,including edges nearEOF. Unknown-window inference
  still runs over entire evaluation videos. Keep batchnorm/frozenblock dropout in eval.
- Cache projected features float32; no quantization/distillation. Batch128 training;
 16window preprocessing batches. AdamWheadLR0.001,lastblockLR0.00005,weightdecay0.0001,
  gradientclip1.0. Fixed threshold0.5,existing boundary decoder and Reel commit rules.
- Validate future isolation,feature/cache forward parity,sourcehashes,finitegradients,
  heldoutrole separation. Independent focused review before launch.
- Detached training with completion/failure notification; no polling. Evaluate selected
  checkpoints on same12developmentvideos using cached MediaPipe poses and actual frozen
  Reel interval classification. Report fullvideo WER,retention,intervalmatches,false
  gapcommits,EOF dependence and timings. Offline scheduling is not a live latency claim.
- No default appchange,protectedtest access,dataset acquisition or generic gate changes.

Ruling: User subsequently prioritized accuracy over speed. Restore all4attentionblocks for adaptation;
only the frozenCNN is cached. The25–35minute estimate applied to final-block-only training
and is superseded. No hard30minute cutoff; benchmark the final configuration.


Execution:17focused tests passed; independent bounded review completed and cached-pose/video
identity guard strengthened. Source/canonical verification passes (genericphrase gatefalse).
Final MPS preflight finite gradients on4,727,042trainable parameters,zerooptimizersteps.
121code/weight/dependency pins verified. Dedicated recipeSHA256
6e80b22a8be08d9abd3d0e5229d990d916cdb6f786c4a1e18e3c743abf76b0ed.
Detached caffeinatePID94956 launched2026-09-22T06:25:19Z. Cache construction first;
training then automatic paired evaluation and completion/failure notification. No polling.
Check completion.json,training_results.json,history files,evaluation.json andREPORT.md
on next authorized check. Launch is not a completion or accuracy claim.
