# stage3-translation — high

Dated evidence for constraints that still bind. The constraints themselves are
stated in `PROJECT_GROUND_TRUTH.md`; these are the receipts behind them.

4 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## 2026-09-29 — user decisions: multi-sentence, <1 s, flan-t5-base, incremental, DeepSeek data

Requirements: a Finish may hold multiple sentences; translation must complete in under one second
on the iPhone 13. No human reviewer: DeepSeek via OpenRouter generates data, references and judge.
After the held-out multi-sentence evaluation (v2 6% fully right) and on-device timing, the user chose:
**flan-t5-base** (248M) as the next Stage 3 model; **incremental translation** (finished sentences
rendered while signing, Finish renders only the open tail); a new English-first DeepSeek training
corpus of ~30–50K sessions with the 300-session evaluation set kept strictly out of it; and deletion of
the random-weight latency packages (done). Evidence:
`artifacts/reports/stage3_multisentence_eval_v17_20260929/REPORT.md`,
`artifacts/reports/stage3_latency_bench_v17_20260929/REPORT.md`.

## 2026-09-29 — user requires model-based translation repair, not phrase triggers

After phone history confirmed correct HELLO MY FRIEND HOW YOU recognition but corrupted
English, and the mobile model also dropped GOOD MORNING from the second requested phrase,
the user explicitly authorized fixing the model and rejected triggers. The replacement
must use learned generation for these compositions. Do not disguise a missing learned
capability with exact-phrase overrides or semantic output rewriting. New checkpoint
contracts disable reviewed-template lookup and let the model punctuate the full Finish
buffer. Technical context-limit fallback and spelling-slot restoration preserve raw input;
they do not insert guessed content. Existing recognition remains outside this repair.
Generated examples are training-only; the fixed final epoch is not chosen from synthetic
validation/test scores. No independent signer/general translation accuracy claim follows
from text regression probes. Plan: artifacts/reports/stage3_composition_v17_20260929/PLAN.md.

## 2026-08-24 18:31 PST — locked-100 Stage 2 handoff is ready for Stage 3

The requested Option A boundary is complete: the selected compact Stage-2 recognizer
emits only the frozen 100 glosses, and a future Stage 3 consumer now has a strict,
hash-pinned interface. The authoritative contract is
`active/v17/stage2_to_stage3_contract_v17.json`, SHA-256
`8be66a44d337dd99484d3ee3140f3124c2e121abe20e93ce7f09b94d96ecc30d`.
It fixes blank `0`, tokens `1...100`, label order, greedy CTC collapse, maximum eight
windows, exact required output keys, checkpoint/vocabulary hashes, empty-sequence
behavior, and fail-closed Stage-3 consumer rules. All 363 full-pipeline validation
outputs passed this contract.

The final iPhone 13 simulator suite passed at 0, 17, 37, 73, 90, 123, 180, and 270
degrees. Each condition completed exactly 200 timed inferences; all 8 conditions
predicted HELLO, and all 1,600 individual decoded iteration votes were unanimously
CTC token 15. Exact quadrant corrections were 0->0, 90->270, 180->180, and 270->90;
intermediate residual roll remained within 45 degrees. The final result is
`artifacts/reports/orientation_v17_simulator_benchmark/latest_result.json`, SHA-256
`84b90290dff1e6e1a39aae3eab9bd074fb2bf2cd18e07e4f7267b58b13ce5cea`.

Two simulator integration faults were found and fixed before promotion. First, the
old harness corrected landmarks but cropped RGB from uncorrected rotated pixels; the
new harness creates both together after one shared v17 orientation correction.
Second, Swift `MLMultiArray` padding was allocator garbage because the padded windows
were not explicitly initialized; landmarks, validity, boxes, embeddings, and window
masks are now zero-filled exactly like the Python training/evaluation tensors. The
pre-fix run-to-run blank predictions disappeared, and both a 20-iteration regression
and the final 200-iteration suite are unanimous across all angles.

Both final unsigned Release builds pass: generic iPhoneOS and arm64/x86_64 Simulator.
Each app bundle contains exactly the three selected models—MobileCLIP2 hand image
encoder, frozen multimodal encoder, and compact context/CTC head—and no obsolete
Stage-1 package. The full raw-crop Core ML report remains exact over 363 samples, 574
windows, and 22,046 crops with zero cached-Core-ML or PyTorch decode mismatches; its
SHA-256 is `99f5ba02d3fbe624044dcf12048ccb1224f2a59dceedefb3d935c3c6463c8c5b`.

The 1,100-row independent capture pack setup audit passes with its ledger unchanged:
1,000 target and 100 OOV plans, zero errors/warnings, no model inference, and no test
access. Its provenance was repaired to pin the earlier frame-API refactor of
`model_hand_mobileclip2_v17.py`; the existing equivalence test proves that refactor
does not change pooled Stage-1 output. Ninety-six unique focused tests pass, affected
Python entry points compile, all generated JSON parses, and `git diff --check` passes.
The complete evidence summary is
`artifacts/reports/mobile_100gloss_v17/README.md`.

This readiness statement is specifically the Stage-2-to-Stage-3 gloss interface. The
existing Stage-3 translator remains a weak research model and is not promoted. The
simulator runtime also lacks Apple Vision pose weights, so simulator preprocessing was
performed by macOS Apple Vision; `cameraToGlossEndToEnd=false`, and there is no claim
about physical-iPhone latency, memory, thermals, ANE behavior, or independent capture
accuracy. No Citizen, SemLex, local, or 2M-Flores test split was accessed.

## 2026-08-24 13:58 PST — Stage 3 genuine-reference gate established; legacy translator rejected

The old Stage-3 evidence is invalid for promotion. `models/stage3_history_simulated.json`
is explicitly simulated, while the actual 60.5M-parameter T5-small checkpoint scored
only 0.2832 BLEU and 6.9142 chrF++ across 321 genuine reference-gloss/English pairs.
On the fixed 155-row 2M-Flores `dev` validation subset it scored 0.0390 BLEU and
4.0782 chrF++; it frequently collapsed to short synthetic-template responses.

Only text metadata was acquired from the official 2M-Flores dataset-server endpoint:
999 current `dev` rows, no video bytes, SHA-256
`6acfaeba2ef680c2b952b412fb28751045e578330e00e9e0f4afd0268ca20626`.
Every one of the 155 previously selected rows matches exactly. The other 844 rows plus
166 public NCSLGR pairs formed 1,010 genuine train rows. A deterministic 1,010-row
sample of the legacy synthetic CSV supplied equal-mass replay; all fixed validation
`(id, signer)` keys were excluded from training.

A bounded warm-start run selected epoch 2 at 0.8544 BLEU and 12.5181 chrF++ on the
unchanged 155-row genuine gate, improving from the 0.0390/4.0782 baseline. Epoch 3
regressed slightly and was rejected. The cold-reloaded package reproduces the selected
metrics exactly. Its checkpoint is
`artifacts/models/stage3_v17_reference_replay_v1/model.safetensors`, SHA-256
`25a0deb4599da88de613d70fad1ad94ca138d0c0ef6ba50efba96650e593cb82`.

Two earlier MPS attempts stopped safely before producing a checkpoint when the 40%
allocator cap detected growing variable-shape graph caches. The final run fixed both
input and target shapes, used batch size 4 and Adafactor, kept `num_workers=0`, and ran
autoregressive validation on CPU. It completed in 699.47 seconds with zero nonfinite
batches and no system memory-pressure failure.

This is a real improvement but not deployable translation quality: exact match is
0/155, and the current Stage-2 recognizer emits only 100 glosses while the genuine
sentences contain substantial out-of-vocabulary content. Stage 3 therefore remains a
research module, and no end-to-end conversational translation claim is supported.
Full evidence is in
`artifacts/reports/stage3_v17_reference_replay_v1/EXPERIMENT.md`. No Citizen, SemLex,
local, 2M-Flores `devtest`, or other test split was accessed.

Sixty-three focused Stage-2, Stage-3, signing-voice, transition, and data-contract tests
pass. All changed Python entry points compile, generated JSON parses, and
`git diff --check` passes. After training and cold reload, system-wide free memory was
63%.
