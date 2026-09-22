# Reel context adaptation

No candidate passed retention gates; frozen models retained.

Runtime-matched reviewed sign crops; proposal classifier and verifier fusion adapted. Frozen encoders, frozen BIO windows, unchanged acceptance. Citizen/SemLex train replay and validation retention; protected tests untouched.

Completed 14 epochs in 0.04 minutes after 20.73 minutes preparation. Selected epoch 0.

| Readout | WER | Correct | Extra words |
|---|---:|---:|---:|
| Baseline | 49.33% | 82 | 6 |
| Best trained (not necessarily eligible) | 51.33% | 81 | 8 |

See history.json for retention and isolated-domain results; completion.json for confirmation. This is reused familiar-signer development, not independent generalization evidence. No distillation or deployment. Temporal backbone adaptation remains untested in this bounded head phase.
