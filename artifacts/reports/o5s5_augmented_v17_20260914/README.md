# O5S5-augmented development experiment — 2026-09-14

**Result: all 12 checkpoints fail promotion. Better context reduced connected WER, but did not fix the model. No confirmation seed was run because the first run failed the required gates.**

Fresh training started from the original phrase-adapted Stage-1 base, never from the earlier trained candidate. Seed 17111 was deliberately reused for a paired data-only comparison; seed 17112 was reserved for conditional confirmation. The original manifest was copied byte-for-byte, and the 12-epoch recipe, decoder and thresholds were unchanged. This is one new development run, not an additional independent seed confirmation.

## Frozen inputs and execution

- Manifest SHA-256: `6b506552f7d35014e539e5df9f5e9d8d40a8e227544974ddb407f8d52cae4178`.
- [Combined supervision](combined_supervision.json), [development freeze](development_freeze.json), [checkpoint provenance](checkpoint_provenance.json). The original combined manifest is unchanged.
- 1,005 O5S5 training windows across 49 classes, from five signers, supplement 5,814 ASLLRP context windows. All 494 background windows come from verified ASLLRP gaps; no O5S5 gap is background.
- Isolated replay identities exactly match the prior freeze: 1,901 training clips and 1,356 validation clips (378 Citizen, 978 SemLex). The freeze verifies 4,520 raw/replay inputs, including six new O5S5 archives.
- Same bounded sampler: 1,500 isolated / 900 contextual / 600 verified-background draws per epoch, with source/class balancing. Adding O5S5 splits contextual sampling mass between sources; it does not add optimizer updates. These are sampling-pool counts, not a promise that every context window is sampled each epoch.
- All 12 epochs completed in 108.43s on MPS, with 564 optimizer updates. Each checkpoint is separately saved; no model/default was promoted.
- All checkpoints evaluated on the same 334 development recordings: 225 connected ASLLRP, 97 familiar phrases, 12 exact ASLLRP. Each covers 7,212 scheduled windows including Finish tails; 86,544 window predictions in total.
- Citizen test was not accessed. No acquisition, MediaPipe conversion, access inquiry, Core ML export or phone deployment occurred.

## Lowest connected-WER checkpoint: epoch 3

| Metric | Frozen comparator / requirement | Epoch 3 | Gate |
|---|---:|---:|---|
| Connected WER | 168.31%; require ≤151.48% | 123.59% | Pass |
| Connected insertions | ≤323 | 196 | Pass |
| Connected deletions | ≤11 | 42 | **Fail** |
| Familiar WER | ≤56.37% | 111.97% | **Fail** |
| Citizen validation | own start 95.24%; floor 94.2381% | 93.3862% | **Fail** |
| SemLex validation | own start 85.28%; floor 84.2761% | 84.2536% | **Fail** |
| Verified-gap false emissions | ≤15/16 | 6/16 | Pass |
| Runtime-inclusive median word delay | ≤1s, complete annotated pool | Not measured | Unverified |

Connected WER improves by 26.57% relative to CTC. Against the prior no-O5S5 run’s best epoch, it improves from 146.13% to 123.59% (415→351 edits). However, deletions increase from 33 to 42 against that prior best too. The new epoch 3 has 113 substitutions, 42 deletions and 196 insertions over 284 reference signs. WER can exceed 100% because insertions are unbounded.

Only epoch 1 retains both isolated validation accuracies, but its connected WER is 311.62%, with 768 insertions, familiar WER 128.57%, and 16/16 gap emissions. Epochs 2–12 all fail Citizen retention. Thus no checkpoint can pass the conjunction of gates, independently of missing runtime timing.

## All epochs

| Epoch | Connected WER | D | I | Familiar WER | Citizen | SemLex | LG positive accuracy | Gap emissions /16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 311.62% | 2 | 768 | 128.57% | 94.97% | 85.48% | 25.79% | 16 |
| 2 | 178.17% | 13 | 372 | 118.92% | 94.18% | 84.87% | 22.64% | 10 |
| 3 | 123.59% | 42 | 196 | 111.97% | 93.39% | 84.25% | 21.07% | 6 |
| 4 | 130.28% | 46 | 207 | 115.44% | 93.65% | 84.25% | 23.27% | 5 |
| 5 | 147.89% | 38 | 264 | 120.08% | 92.86% | 84.36% | 24.84% | 5 |
| 6 | 146.48% | 44 | 254 | 117.76% | 93.12% | 84.15% | 24.21% | 5 |
| 7 | 140.14% | 40 | 248 | 112.74% | 93.65% | 84.25% | 22.64% | 4 |
| 8 | 145.07% | 35 | 266 | 116.60% | 93.65% | 84.66% | 23.27% | 3 |
| 9 | 147.18% | 44 | 261 | 114.67% | 93.39% | 84.66% | 23.27% | 2 |
| 10 | 147.54% | 43 | 262 | 115.06% | 93.12% | 84.97% | 22.96% | 2 |
| 11 | 149.30% | 39 | 272 | 115.06% | 93.12% | 84.97% | 22.96% | 2 |
| 12 | 148.24% | 39 | 268 | 115.83% | 93.12% | 84.97% | 22.96% | 2 |

Machine-readable [gate selection](selection.json), [epoch comparison](epoch_comparison.csv), and [timestamped error examples](timestamped_error_examples.csv) retain the measured failures. Full per-recording transcripts and source-time updates are in `evaluations/`.

## Held-out O5S5 signer LG

LG is validation-only: 318 positive windows across 27 classes (162 centered, 156 trailing windows). All 12 checkpoints were evaluated. Windows overlap and are correlated; these numbers do not represent 318 independent signs or multi-signer test accuracy. Annotation completeness is unproven, so full-narrative LG WER, insertion and background rates would be misleading and are not reported.

| Model | Correct /318 | Positive-window accuracy |
|---|---:|---:|
| Original isolated base | 76 | 23.90% |
| Prior no-O5S5 epoch 1 | 73 | 22.96% |
| Prior no-O5S5 best connected epoch 4 | 68 | 21.38% |
| New O5S5 epoch 1 | 82 | 25.79% |
| New O5S5 best connected epoch 3 | 67 | 21.07% |

The best LG window score is epoch 1 at 25.79%, a 1.89-point increase over the original base. That checkpoint fails the streaming gates. The lowest-WER checkpoint scores 21.07% on LG, below the original base. No consistent held-out signer improvement is established. [All LG predictions and schedule/macro metrics](lg_evaluation.json).

## Verification, latency and decision

- 14 focused training/selector/O5S5 tests passed. Evaluator assertions verify 334 recordings and 284 connected reference signs per epoch; all 12 checkpoint hashes, encoder/classifier changes, 564 updates, raw-input hashes and exact isolated replay identities were checked.
- CPU/MPS labels agree on all 318 LG windows for epochs 1 and 3 (636 comparisons). [Parity scope](cpu_mps_lg_parity.json) is explicit: this is not complete-stream or runtime parity.
- The first sandbox launch failed before training because MPS was unavailable inside the sandbox. The authorized GPU run completed. Both logs are preserved.
- Batched cached evaluation excludes extraction and scheduling latency. No fresh paced raw-runtime or phone timing was run because every checkpoint already fails measured accuracy/error gates. Source timestamps must not be presented as runtime-inclusive latency. The latency gate remains unverified, never passed by assumption.
- Confirmation seed 17112 was not run: the user’s “continue only if” condition failed. No Core ML export or real-iPhone test followed. Default runtime and accepted checkpoints remain unchanged.

The bounded result supports a narrower conclusion: O5S5 context helps reduce insertion-heavy connected WER under this recipe, but deletion and retention failures remain, and LG generalization is weak. This does not establish that more context alone fixes the recognizer. Preserve this failure result; another training recipe requires a separately bounded decision.

## Reproduction

The exact commands are below; the development freeze records the recipe. Training refuses to overwrite its checkpoint directory. The evaluation driver also refuses to overwrite existing epoch reports.

```sh
venv/bin/python active/v17/train_stage1_window_v17.py --train --seed 17111 \
  --supervision artifacts/reports/o5s5_augmented_v17_20260914/combined_supervision.json \
  --development-freeze artifacts/reports/o5s5_augmented_v17_20260914/development_freeze.json \
  --audit artifacts/reports/o5s5_augmented_v17_20260914/training_data_audit.json \
  --output-dir artifacts/models/stage1_window_o5s5_v17_seed17111 --device mps
venv/bin/python artifacts/reports/o5s5_augmented_v17_20260914/evaluate_experiment.py
```
