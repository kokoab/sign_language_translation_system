# Independent next-step review — 2026-09-14

## Recommendation

Run one bounded comparison of **joint versus frozen Stage-1 training**, reusing the
existing shallow causal CTC machinery. Correct supervision and coverage in both arms.
Start independent portrait-iPhone data collection alongside that work. The evidence
supports this experiment; it does not establish that a new architecture will solve LG.
This review did not launch training, access the Citizen test, or send access requests.

## What was independently verified

- The current combined supervision file matches the audit and verification SHA256:
  `6b506552f7d35014e539e5df9f5e9d8d40a8e227544974ddb407f8d52cae4178`.
- Streaming reductions of the saved prediction CSV reproduce 8,978 rows: 6,819/1,649
  train/validation context windows and 494/16 background windows. Exposure reductions
  reproduce 3,516/5,515 unseen ASLLRP-other and 154/1,005 unseen O5S5 train windows.
- Event CSV reductions reproduce 199 O5S5 training events, 57 LG events, median
  durations 0.259/0.340 seconds, and 23 of 49 O5S5 train classes having one signer.
  The long-window O5S5 foreground mean is 29.0891%.
- The stored core-probe results contain 61.09% O5S5-train and 27.67% LG accuracy;
  the audit has 15 isolated-model result entries. These are report checks, not freshly
  rerun checkpoint inference or raw-video extraction.
- All 14 tests in `test_stage1_window_training_v17`,
  `test_stage1_window_evaluation_v17`, and `test_prepare_o5s5_citizen100_v17` pass
  in a fresh `venv/bin/python -m unittest` run.

Two initial verification assumptions failed and were investigated: window identity
strings are not globally unique, and `model_summaries` is a list of grouped metric
rows rather than one entry per model. Model count was verified using `isolated`.
There are 52 repeated identity groups (57 extra rows), all background. The loader
resets `ordinal` for each interior gap, so different features receive the same ID.
All colliding groups retain the same source/target; no identical-feature conflicting
targets were found. Context identities are unique, so this does not invalidate the
quoted positive-window exposure totals. Before using IDs for coverage or deduplication,
include gap identity or timestamps; do not discard these distinct background examples.

## Corrections to the earlier interpretation

1. **The core probe is contextual.** `foreground_core_probe.py:31` encodes the full
   window before masking its pooled tokens. `model_v17.py:419` uses unmasked temporal
   attention and symmetric convolution padding. Therefore this is boundary-informed
   pooling, not independent recognition of raw cropped sign cores. The remaining LG
   gap is consistent with domain/variant/signer difficulty, but does not isolate those
   causes or rule out local extraction and annotation errors.
2. **A causal head does not make this encoder causal.** The learned position table
   also has 32 slots. Define timestamped 32-frame chunk assembly, overlap ownership,
   and the frame availability time before training. With the existing encoder, treat
   recognition as bounded-chunk/revisable and measure its lookahead; do not concatenate
   overlapping frame tokens as though they were new observations or call them
   zero-lookahead streaming features.
3. **The proposed head largely exists.** `model_unified_streaming_ctc_v17.py` already
   provides causal dilated temporal blocks and blank/known/OTHER outputs. Its trainer's
   `encode()` uses `torch.inference_mode()` and caches NumPy features, preventing CTC
   gradients from reaching Stage 1. Reusing these blocks in a differentiable path is
   the first experiment. Multi-level losses and a second shared classifier can wait.
4. **Core and retention losses also exist.** `window_objective()` already includes
   foreground CE and replay teacher KL. Adding their names to a recipe is insufficient:
   remove mixed whole-window positive CE, preserve isolated supervision, and establish
   that sequence gradients actually update the encoder.
5. **The CTC output requires 102 logits:** blank 0, known glosses 1–100, OTHER 101.
   The earlier diagram's “101-way” CTC classifier would omit a required category.

## What the research supports

[Squeezeformer](https://arxiv.org/abs/2206.00888) supports compatibility with CTC,
not a prediction of ASL or streaming accuracy. [VAC](https://arxiv.org/abs/2104.02330)
supports improving visual-feature learning with auxiliary alignment supervision.
[Cross-temporal contexts](https://openaccess.thecvf.com/content/CVPR2023/papers/Guo_Distilling_Cross-Temporal_Contexts_for_Continuous_Sign_Language_Recognition_CVPR_2023_paper.pdf)
reports that a shallow temporal module can improve training of the spatial module.

The [2025 region-aware pose/Conv1D paper](https://openaccess.thecvf.com/content/ICCV2025W/MSLR/papers/Tran_Generalizable_Sign_Language_Recognition_via_Local_Temporal_Convolutions_and_Region-Aware_ICCVW_2025_paper.pdf)
is relevant architectural evidence, but its challenge concerns **unseen sentences**
on an Isharah subset, approximately 14,000 clips and 18 signers. Its reported 49.4%
development WER does not establish unseen-signer ASL transfer or causal iPhone latency.
Direct CVF fetches returned 403; the official CVF indexed paper text supplied the
abstract, dataset description and ablation results used here.

[Online CSLR, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.619/) demonstrates
dictionary training and sliding-window recognition on its benchmarks. Local failed
window experiments justify changing this project's next experiment, not declaring that
published approach inherently unable to work.

My inference from these sources and the code: end-to-end temporal supervision is a
reasonable next hypothesis. Causality, multi-scale decoding and signer generalization
are separate properties that must be tested rather than inferred from a paper title.

## Concrete next experiment

1. **Preflight before a full run.** Freeze the data, starting checkpoint, compute budget
   and existing promotion gates. Check a small train-only set of timestamped sequences
   for CTC alignment feasibility, including repeated adjacent labels. Confirm a nonzero
   sequence-loss encoder gradient and that a tiny training subset can be fitted.
   Compare raw-core crops with the existing contextual-core probe, with one result
   per original event. A failing preflight calls for fixing the contract before scale-up.
2. **Use the same supervision in both arms.** Traverse every admitted training
   sequence/core before repeats, preserving explicit source and isolated-loss weights.
   Count original events as well as overlapping windows. Fully annotated ASLLRP spans
   receive complete known+OTHER CTC targets. O5S5 receives positive-core supervision
   only; unknown context receives no fabricated negative or complete-sequence target.
   Explicit background loss remains restricted to verified gaps. Retain isolated CE
   and the existing teacher retention mechanism.
3. **Run frozen versus joint with matched data and updates.** Reuse the shallow head;
   make encoder gradient flow the principal difference. Use one seed and a fixed cap,
   with no sweep of architectures, distillation modules or emission thresholds. Compare
   both arms with the unchanged accepted runtime on the same development recordings.
4. **Require all existing gates.** Connected and familiar WER, insertions, deletions,
   Citizen/SemLex development retention, verified-gap emissions and runtime-inclusive
   delay all matter. Report LG event-level accuracy and class support separately;
   318 overlapping windows from 57 events and one signer are not 318 independent
   generalization trials. No full-narrative LG WER. A training-only improvement is not
   promotion evidence; confirmation runs follow only an eligible first seed.

## Data work that matters now

Target the 27 classes without combined continuous training coverage and the O5S5
classes with one training signer. Obtain ASL-fluent review of exact visual variants,
then collect natural connected examples with signer IDs, sign boundaries, repetitions,
holds, OOV signing and genuine nonsign intervals. Existing access-request drafts can
support acquisition, but sending them requires explicit authorization.

Create separate development and sealed evaluation signer groups for portrait-iPhone
recordings before model iteration. LG and the current ASLLRP development signer have
already supported many selections. Further improvement on them is development evidence;
it cannot replace fresh independent evaluation. If a correctly trained pilot fits train
cores but still fails held-out signers, prioritize signer/variant coverage and targeted
visual review before another architecture sweep. Real-device profiling follows useful
accuracy; compression and translation expansion remain deferred.
