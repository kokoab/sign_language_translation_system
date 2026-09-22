# Pretrained Renz segmentation → Reel: completed 2026-09-22

**The pretrained model runs, but this transfer does not yet solve Reel's transition problem.**
Official BSLCorpus I3D and matching MS-TCN checkpoints were loaded strictly with no missing
or unexpected keys. No fine-tuning, new source videos, protected test or live model changes.

## Scope and results

Twelve previously used approved ASLLRP validation videos, 24 reference signs. The same raw
video identities/hashes used in the matched Reel comparison. Full approved membership check
passed. The pretrained segmenter predicts boundaries; current Reel proposal/verifier and
conditional commit rules then classify each resulting segment. This bypasses live candidate
activation/stability and uses offline future context. WER below is for this bounded offline
composition, not overall dataset WER, live-app WER or a published Renz benchmark score.

| Diagnostic variant | Emitted tokens | Edit distance / 24 references | WER | Eligible cores with best segment IoU≥.5 |
|---|---:|---:|---:|---:|
| Valid 16-frame windows; raw sign regions | 0 | 24 | 100% | 2/17 |
| Edge-padded windows; raw sign regions | 6 | 19 | 79.17% | 9/17 |
| Edge-padded + published midpoint region extension | 10 | 15 | 62.50% | 13/17 |

Final variant gets 2/12 complete sequences exactly right. Examples: FRIEND NOW and TIME
FRIEND succeed on individual clips; WORK WHERE produces WORK WORK. Other clips omit one
or both signs. The repeated WORK cannot be assigned to a pure transition solely from the
transcript; its predicted region can overlap a real sign and transition. This experiment
has not established reliable rejection of the user's HELLO→MY→falseGOODBYE case.

The reference has no OTHER tokens on these12clips. All predicted candidate outputs and
threshold outcomes are retained. Best IoU is a per-core coverage diagnostic, not one-to-one
segmentation precision/recall. There were31candidate regions in padded runs; midpoint
extension reduces too-short (<4Reelobservations) candidates from11to4. Existing gates also
reject some correct top predictions. No same-input full default-Reel scheduler comparison
was run, so improvement over the live default has NOT been established.

## Execution and compatibility

Source: https://github.com/RenzKa/sign-segmentation
Pinned commit29cc10963b41179c09e6fab4e0585c263f4917c9.
Downloaded model archive via the official download/download_model.sh Drive identifier;
initial download timed out97%, resume completed, ZIP member extraction succeeded.
Files under artifacts/models/renz_pretrained_v17; SHA256s in provenance.json.
Upstream architecture imported unchanged; no package installation or environment replacement.

Model input: RGB25fps,16frames, stride1, mean.5/std1,256resize/224center crop,1024D I3D
embeddings, four-stage MS-TCN,100feature chunks, boundary threshold.5. Class1 is boundary;
class0 runs form sign regions. Midpoint extension follows upstream generate_vtt_file's
integer half-gap expansion. Midpoint clocks map back to original video seconds.

Explicit departures/limits:
- Aspect-preserving padding before resize/crop, because project forbids aspect distortion;
  upstream demo stretches non-square video. This is NOT exact published frontend parity.
- Safe ffmpeg pipe resampling rather than upstream destructive source rewrite.
- CPU rather than CUDA. Strict state-dictionary loading and finite-output checks passed.
- Correct per-batch feature assignment, no empty final MS-TCN chunk.
- Edge variant adds8copies of first frame and7oflast before16frame extraction: one feature
  per25Hzsourceframe, rather than leaving first/last~.3seconds without output. This matters
  for these short two-sign clips. It is a disclosed adaptation, not an ASL-tuned checkpoint.
- Frozen Reel consumes native20HzApple observations, independent of RGB25Hz extraction.

Initial unpadded extraction/inference+Reel took50.18seconds total; padded pass100.39seconds
on this Mac CPU. Final segment postprocessing replay reused padded cached features. These
are offline runtimes, not camera latency. MS-TCN and I3D use future frames.

## Verification and interpretation

check.py passed span/edit selfchecks, pinnedinput/checkpoint hashes,12clipcompletion,
finitecachedfeatures/probabilities,exactframecoverageandeditcounts,allthree preservedrunner
hashes. OriginalvideoSHA256 was checked by each runner. git diff --check passed.
initial_runner.py and edge_runner.py preserve earlier script versions; results are kept
separately, not overwritten or presented as independent experiments.

This demonstrates a usable pretrained boundary reference and a significant short-clip
coverage issue. It does NOT show that the ASL dataset is defective, that all boundary
models fail, or that a new threshold would solve the problem. Model language/domain,
frontend differences, segment duration and Reel gates are confounded in this transfer.
Do not promote. Next discussion should choose between adapting this segmenter on approved
ASL boundary supervision and evaluating stronger ASL-pretrained temporal representations
(SHuBERT); neither has been launched. Preserve current Reel live model.
