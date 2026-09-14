# Direct-positive CTC repair: completed training result

This first correction adds single-token CTC to all isolated replay clips and verified
O5S5 positive cores, alongside the original objectives. It repairs the missing
positive gradient route into the CTC head, but continuous alignment remains weak.

| Full training metric | Original joint epoch 12 | Positive CTC epoch 12 |
| --- | ---: | ---: |
| Known continuous matches | 2 / 1,291 | 183 / 1,291 |
| Known continuous deletions | — | 974 / 1,291 |
| Known continuous WER | — | 87.84% |
| Single-clip isolated CTC correct | not measured on train | 1780 / 1,901 (93.63%) |

All 12 epochs completed: 1,320 optimizer updates, with every admitted 923 sequence,
199 core, 1,901 replay and 494 background samples covered each epoch. Optimizer states
are saved for recovery. Training code and recipe hashes still match the freeze.
`training_summary.json` contains every epoch; `gradient_diagnosis.json` demonstrates
the original missing head gradient. `DIAGNOSIS.md` records the blank-dominated
marginal-likelihood basin measured using training clips only.

No development evaluation was used to select the subsequent anchor correction.
The next run is `../joint_ctc_anchor_v17_20260915/`: existing verified event boundaries
provide sparse positive emission anchors. No additional data, decoder bias change,
Citizen test access, deployment, or promotion occurred here.
