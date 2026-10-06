# Paired fine-tuning verification

Same starting checkpoint, data, seed and 20-epoch budget. Validation development experiment; no protected test access.

| Selection | Epoch | Upright Top-1 | Mean rotated Top-1 | Worst rotated Top-1 | Original weights? | Best / final trained epoch Top-1 |
|---|---:|---:|---:|---:|---|---|
| original | 100 | 96.83% | 35.71% | 0.53% | True | n/a / n/a |
| mild_control | 0 | 96.83% | 35.71% | 0.53% | True | 96.56% / 96.03% |
| full_roll | 0 | 96.83% | 35.71% | 0.53% | True | 95.77% / 94.71% |

Selection maximizes upright validation accuracy and retains epoch 0 when there is no strict improvement. A retained original is not evidence that orientation training improved the model.

Rotation diagnostics transform existing landmarks. They do not measure raw-camera extraction or independent phone generalization.

Full per-angle scores, actual selected weight identity and best trained-epoch accuracy are recorded in audit.json. No manuscript or deployment change.
