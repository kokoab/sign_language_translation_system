# Pretrained ASL boundary adaptation

No automatic promotion. Same12reused development videos/24known signs; frozen Reel.
Cached-pose bounded-context evaluation is not a live latency or phone result.

| Arm | Epochs / selected | Correct | WER | Retained baseline |
| --- | --- | --- | --- | --- |
| frozen_pretrained_bio None | — / — | 6/24 | 75.00% | 3/4 |
| adapted_start_end 17621 | 16 / 8 | 7/24 | 70.83% | 2/4 |

All approved supervised20Hz windows used.5head warm-up epochs then all4attentionblocks adapted; CNN cached/frozen.
Maximum80epochs,minimum0,patience8. Checkpoints selected only on train-parent calibration.
Training augmentation: {'variants': 2, 'seed': 176210, 'sensor_fps': [15, 20], 'observation_dropout': [0, 0.15], 'kind': 'past-only observation hold before normalization; source clock/targets unchanged'}. Held-out windows remain unaugmented.
Unknown gaps masked; explicit EOF partial-context predictions tracked. Original frozen BIO and adapted START/END decoders differ.
See training_results.json,history files,evaluation.json and completion.json. Do not infer unseen-signer or real low-motion/repeat accuracy from this small replay.
