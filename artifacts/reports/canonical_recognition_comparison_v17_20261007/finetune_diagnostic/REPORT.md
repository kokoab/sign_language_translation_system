# Diagnostic rerun: trained-epoch weights at eight landmark roll angles

Identical paired recipe rerun with diagnostic checkpoints. best_model.pth selection unchanged; diagnostic weights are not promoted. Validation development experiment; no protected test access.

| Arm | Trained epochs | Selected epoch | Selected = original | History identical to first run | Max |ΔTop-1| vs first run |
|---|---:|---:|---|---|---:|
| mild_control | 20 | 0 | True | True | 0.00 |
| full_roll | 20 | 0 | True | True | 0.00 |

| Weights | Epoch | 0° | 17° | 37° | 73° | 90° | 123° | 180° | 270° | Mean rotated | Worst rotated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original_96.83 |  | 96.83 | 96.03 | 91.80 | 39.95 | 10.58 | 1.06 | 0.53 | 10.05 | 35.71 | 0.53 |
| mild_control_final | 20 | 96.03 | 96.30 | 91.80 | 42.59 | 10.85 | 1.06 | 0.26 | 9.26 | 36.02 | 0.26 |
| mild_control_best_trained | 1 | 96.56 | 95.77 | 91.80 | 39.42 | 10.58 | 1.32 | 0.26 | 8.73 | 35.41 | 0.26 |
| full_roll_final | 20 | 94.71 | 94.97 | 91.53 | 77.78 | 65.61 | 55.03 | 52.38 | 67.20 | 72.07 | 52.38 |
| full_roll_best_trained | 2 | 95.77 | 95.50 | 91.80 | 53.44 | 27.25 | 6.08 | 3.97 | 25.13 | 43.31 | 3.97 |
| reference_a7490409_from_scratch |  | 95.77 | 94.71 | 94.71 | 94.44 | 93.92 | 92.59 | 92.06 | 93.65 | 93.73 | 92.06 |

best_trained is chosen on upright validation among trained epochs, so its upright score is optimistically selected. final is the last trained epoch. The a7490409 reference is a separate 138-epoch from-scratch run, not a matched budget.

Rotation diagnostics transform existing landmarks. They do not measure raw-camera extraction or independent phone generalization.
