# Stage 3 v17 genuine-reference replay experiment

## Decision

The legacy 60.5M-parameter T5-small translator is rejected. Its retained “85.6 BLEU”
history is explicitly simulated, and its actual cold evaluation on 321 genuine
reference-gloss/English pairs scored only 0.28 BLEU overall. On the fixed 155-row
2M-Flores validation subset it scored 0.04 BLEU and 4.08 chrF++.

A bounded repair experiment warm-started that checkpoint using:

- 844 current 2M-Flores `dev` rows excluding every fixed validation `(id, signer)`;
- 166 public NCSLGR reference-gloss/English rows;
- 1,010 deterministically selected legacy synthetic replay rows, giving equal genuine
  and synthetic mass.

The fixed validation gate is the unchanged 155-row 2M-Flores `dev` subset previously
acquired for locked-vocabulary research. The reserved `devtest` split was not queried.

## Result

Training used fixed 192-token input and 96-token target tensors, batch size 4,
Adafactor, `num_workers=0`, and a 40% MPS allocator cap. Autoregressive validation ran
on CPU. This eliminated the variable-shape MPS cache growth that safely stopped two
earlier pre-checkpoint attempts.

| Epoch | Mean train loss | BLEU | chrF++ |
| ---: | ---: | ---: | ---: |
| Baseline | — | 0.0375 | 4.0799 |
| 1 | 3.7114 | 0.8232 | 10.5368 |
| **2** | **2.9093** | **0.8544** | **12.5181** |
| 3 | 2.8245 | 0.8498 | 12.3458 |

Epoch 2 is selected. Cold reload reproduces exactly 0.8543718622 BLEU and
12.5181007028 chrF++ on all 155 validation rows. The checkpoint is
`artifacts/models/stage3_v17_reference_replay_v1/model.safetensors`, SHA-256
`25a0deb4599da88de613d70fad1ad94ca138d0c0ef6ba50efba96650e593cb82`.

This is a measurable improvement but is **not deployable translation quality**. It has
zero exact matches on the 155 genuine validation sentences, and the Stage 2 recognizer
can emit only its locked 100 glosses while these genuine sentences contain substantial
out-of-vocabulary content. No Stage 3 or end-to-end conversational translation claim
is supported yet.

Evidence:

- baseline: `artifacts/reports/stage3_v17_reference_gloss_baseline/metrics.json`
- selected training: `artifacts/models/stage3_v17_reference_replay_v1/result.json`
- cold reload: `artifacts/reports/stage3_v17_reference_replay_v1_cold_reload/metrics.json`
- acquisition snapshot: `data/local/dataset_metadata/2m_flores_asl/dev_all_metadata_v17.json`

No Citizen, SemLex, local, 2M-Flores `devtest`, or other test split was accessed.
