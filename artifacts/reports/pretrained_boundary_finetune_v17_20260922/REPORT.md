# Pretrained ASL boundary adaptation

No automatic promotion. Same12reused development videos/24known signs; frozen Reel.
Cached-pose bounded-context evaluation is not a live latency or phone result.

| Arm | Epochs / selected | Correct | WER | Retained baseline |
| --- | --- | --- | --- | --- |
| frozen_pretrained_bio None | — / — | 6/24 | 75.00% | 3/4 |
| adapted_start_end 17621 | 40 / 7 | 9/24 | 62.50% | 2/4 |

All approved supervised20Hz windows used.5head warm-up epochs then all4attentionblocks adapted; CNN cached/frozen.
Maximum120epochs,minimum40 before convergence stopping,patience20. Checkpoints selected only on train-parent calibration.
Unknown gaps masked; explicit EOF partial-context predictions tracked. Original frozen BIO and adapted START/END decoders differ.
See training_results.json,history files,evaluation.json and completion.json. Do not infer unseen-signer or real low-motion/repeat accuracy from this small replay.
