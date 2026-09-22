# Bounded boundary-adaptation diagnostic

Authorized 2026-09-22: preserve Reel, weights, source/signer splits, old artifacts and defaults.
First reproduce frozen BIO on all 72 videos and swap each adapted backbone under the
unchanged original BIO head/decoder. Assert CNN, normalization and BIO weights are identical.
Reconstruct all cached targets; check unknown gradients, pose/cache parity and clocks.

Bounded follow-up: expose explicit BIO versus START/END readout in the offline evaluator;
compare the fixed three existing backbones with original BIO and existing adapted edge
paths on training-held calibration. No gradient steps or epoch/threshold sweep. Reusing
pretrained BIO is the minimal preservation intervention if the swaps recover recognition.
Only two existing calibration clips have approved complete locked-vocabulary transcripts
(four signs). Report this limitation, do not invent OOV/background transcripts or change splits.
Calibration selection: require strictly lower whole-video WER than untouched BIO, no fewer
retained reference positions and no more insertions; otherwise retain untouched BIO.
Ties prefer untouched weights. Expanded72 is descriptive comparison only and never selection.
No new fitting unless the evidence justifies it; no long training, deployment or distillation.
Benchmark batch-one normalized window + model with synchronized MPS, and report whole-video
frontend, boundary and Reel time separately. Cached MediaPipe excludes its extraction cost;
500ms future and EOF partial context remain explicit. No live/iPhone readiness claim.
