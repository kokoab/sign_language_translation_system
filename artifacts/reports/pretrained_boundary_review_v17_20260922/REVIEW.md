# Pretrained boundary audit — 2026-09-22

Seed 1 completed 40 epochs (78.15 minutes), selected epoch 7. Seed 2 was interrupted by the user; completion failure is KeyboardInterrupt, not evidence of numerical failure. No restart or new fine-tune was launched during this review. Default Reel remains unchanged.

## Whole-video evidence

Same reused 12 development videos, 24 reference signs; not independent generalization evidence:

| Arm | Correct | S/D/I | WER | Retained original correct | Exact videos | Guarded-gap commits | EOF-partial commits |
|---|---:|---|---:|---:|---:|---:|---:|
| Frozen bounded BIO | 6 | 1/17/0 | 75% | 3/4 | 0/12 | 0 | 5 |
| Adapted seed 17621 | 9 | 2/13/0 | 62.5% | 2/4 | 1/12 | 1 | 8 |

Adapted loses original WORK (842935 span00) and FRIEND (841415 span00); the guarded-gap case is 7345233 span00. Candidate is not ready for promotion or distillation. Zero WER insertions does not imply zero gap emissions: substitutions can contain false emitted words. Offline unbounded teacher previously reached 11 correct, but differs in normalization, sampling, context and decoder; not a clean context ablation.

## Hypothesis verdicts

- Cached project() inputs are fixed, with no pose augmentation in the adaptation trainer. Upstream fps/frame dropout are dataset transforms, not activated by merely loading config. Attention/feedforward dropout and weight decay remain. Missing augmentation is plausible, not proven as sole cause.
- 30,014 train windows overlap heavily. Actual optimization subset is 758 poses / 727 parents / 8 signers / 3,019 events. Other prepared poses/events belong to calibration or validation. 4,310 is not an effective independent training sample count; events themselves correlate.
- Upstream 1024 versus adapted 64 frames is a real context shift. Adapting all four blocks may overfit; last one/two is a controlled hypothesis, not demonstrated superiority.
- Best head-only calibration BCE .725667 (epoch 4), best adapted .666061 (epoch 7): 8.21% relative loss reduction, not recognition accuracy. Epoch 40 train .065435 versus calibration 2.135350 is strong overfitting evidence.
- Current LR is NOT flat: ReduceLROnPlateau factor .5, patience 8; by epoch 40 head/attention LR .000125/.00000625. One-cycle is upstream practice, not proven better here.
- Replaying patience 8 with no floor stops epoch 15 and retains epoch 7, saving 50.59 minutes. All 33 epochs after 7 are hindsight non-improvements; they cannot all be known unnecessary at epoch 7. Existing patience 20 alone would stop at 27, so the floor specifically adds 13 epochs. Best checkpoint is retained, so later overfit does not overwrite it. Native TCN experiments were capped at 8 epochs; their selected epochs do not establish natural convergence at 7–8.
- Multiple genuinely pose-augmented cached variants are valid if timestamps, labels, masks and bounded lookahead remain consistent. Feature dropout is cheaper but not equivalent; arbitrary mixup risks invalid masked boundary targets. A third cache-preserving option is periodically refreshing a subset from already extracted raw poses, without video re-extraction. Keep held-out inputs fixed.
- 30,014 * 64 = 1,920,896 frame slots for 30,014 target predictions is real redundant window work. It is not proof of a 64x achievable gain. Whole-video causal attention alone would not fix the temporal CNN or window normalization contract.
- frame_cnn is NOT per-frame spatial-only: it reshapes to [batch*joints,channels,time] for temporal Conv1d/U-Net. Changing a neighboring input frame changed target projection by max .563895. Exact per-frame projection reuse is invalid even with fixed normalization.
- Window normalization depends on the window. Streaming normalization changes trained inputs and needs retraining/evaluation; it does not alone enable CNN caching.

## Measured desktop timing

Synchronized MPS, selected checkpoint, real pose window, 4 warmups, 30 samples (60 under contention). See measurements.json and reproducible measure.py.

| Component, batch 1 | Median ms | p95 ms |
|---|---:|---:|
| CPU window normalization | 1.11 | 1.35 |
| CNN projection | 10.21 | 12.49 |
| Attention/head | 3.85 | 5.55 |
| Full model | 15.82 | 29.28 |
| Normalization, transfer, model | 16.50 | 21.41 |
| Same path with concurrent frozen Reel classification | 13.70 | 202.05 |

Separate timings are noisy and non-additive. Lower concurrent median is not a speedup claim. Reel interval classification median 252.53 ms (13 calls). No live MediaPipe frontend, camera, iPhone or sustained thermals measured; intrinsic .5s future context is additional latency. Combined real-time performance is not established.

Training batch-128 microbenchmarks: original memmap median 366.87 ms, fewer-sync warm memmap 296.96 ms, resident FP32 322.09 ms, resident FP16 roundtrip 297.10 ms. Variants ran sequentially over identical indices: original gather 50.00 ms versus subsequent warm memmap .98 ms means cache warming confounds attribution to synchronization. Do not claim a measured sync-only gain. No measured 2x FP16 benefit. FP16 roundtrip changed probabilities by max .0001413 with zero .5-threshold flips on 128 training windows; not accuracy equivalence. Microbenchmarks omit full calibration/checkpoint overhead and do not fully explain sustained 125.65s epochs versus the earlier synthetic estimate. Benchmark updates were disposable, no weights/caches overwritten.

## Recommended next experiment (proposal, not launched)

Keep Reel frozen. First compare current adaptation against pose-augmented cached adaptation, same splits and architecture, with no arbitrary epoch floor and patience 8 after attention adaptation has had a fair chance. Permit up to 80 epochs if improvement continues; avoid ending exactly when a scheduled LR reduction first occurs. If augmented all-block adaptation still diverges, compare last-two-block adaptation under the same augmented inputs. Treat lower attention LR as another controlled ablation; one-cycle and architecture rewrites are lower priority. Start with one seed, repeat a promising result rather than paying for two failed runs.

Use train-parent-held calibration for checkpoint/decoder selection and paired whole-video evaluation for the decision: retained correct signs, false emissions, deletions, exact videos and EOF dependence. Preserve unknown masks; resolving inadequate transition-negative supervision needs reviewed labels, not invented blank targets. Existing 12-video development replay is reused and insufficient for final promotion.

Our Apple-landmark TCN can improve and need not remain 78k parameters. First establish a reliable boundary teacher or supervised baseline. Then compare direct use against an improved native TCN, and only then consider training-only timestamp-aligned teacher probabilities plus real labels. MediaPipe versus Apple inputs and .5s versus .2s future contracts must be handled explicitly. Distilling this candidate now risks transferring its missed signs and gap errors.

Source: pinned author repository https://github.com/sign-language-processing/segmentation/tree/22ca3a6f63b6f031bfb1c0d717fcb259143ba7db ; local upstream source, active/v17/pretrained_boundary_v17.py, scripts/train_pretrained_boundary_v17.py and saved training/evaluation artifacts. No protected test or new dataset accessed.
