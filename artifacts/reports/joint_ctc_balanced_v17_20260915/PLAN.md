# Correct CTC / positive-CE loss scaling

The verified-anchor run completed all 12 epochs: 330/1,291 known training matches,
817 deletions, 79.47% WER and 1,858/1,901 isolated CTC correct. It does not yet fit
continuous training adequately. No held-out evaluation has informed this correction.
A 64-training-sequence probe shows blank wins 87/106 known anchors; excluding blank
would identify 62/106, separating blank competition from remaining label confusion.

A controlled 32-frame one-sign problem demonstrates the objective scale failure.
Target-normalized CTC plus one positive CE converges to target probability 0.0627
and empty greedy output. Dividing CTC likelihood by valid output frames instead
reaches 0.9954 with nonempty greedy output under the same initialization/optimizer.
See the preceding report's loss_scale_diagnosis.json. This is a training-only
counterexample, not a claim that the real corpus is solved.

Continue from the fixed anchor epoch 12 including its optimizer state. Change only
CTC normalization: per valid output frame for complete sequences and /32 for positive
single-clip CTC. Keep anchor CE, pooled CE/KL, verified-background loss, population
weights, architecture, learning rates and greedy decoder. Train 12 further epochs
(13–24), 110 updates each, using original full-coverage schedules and seed 17111+epoch.
Save full optimizer/checkpoint state and measure all training outputs every epoch.

Evaluate the final candidate on all334 frozen development recordings,1,356isolated
validation clips and57LG positive cores; verify provenance, counts, focused tests and
CPU/MPS output consistency. Do not access the Citizen test or promote a failed model.
Retain prior correction artifacts. No data acquisition, decoder bias sweep or
held-out tuning is part of this correction.
