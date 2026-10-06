# No-full-roll Transformer result

Fresh CPU validation matches the saved selection. Training stopped at epoch 77; first best was epoch 47. Explicit full-roll probability 0, mild roll ±12 degrees.

| Checkpoint | Upright Top-1 | Mean rotated Top-1 | Worst rotated Top-1 |
|---|---:|---:|---:|
| no_full_roll | 94.71% | 32.65% | 1.59% |
| previous_transformer | 95.50% | 93.58% | 92.06% |

Eight angles: 0, 17, 37, 73, 90, 123, 180, 270 degrees. Rotated mean excludes zero. Software landmark diagnostics, not raw-camera or phone accuracy.

Single-seed development comparison. Historical checkpoint augmentation implementation is not immutably recorded, so do not claim an isolated causal augmentation effect. No protected test access, model promotion or manuscript edits.
