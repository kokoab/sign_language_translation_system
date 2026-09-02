# All-real transition experiment

Phrase generation remained stopped. This experiment used only genuine v17 trajectories and retained the original detector presence/confidence channels.

## Data

The trainer samples six corpus families with equal probability:

| Family | Train windows |
| --- | ---: |
| All nine local phrase families | 1,781 |
| ASLLRP continuous + `OTHER` | 3,761 |
| 2M-Flores | 2,686 |
| How2Sign | 4,938 |
| NCSLGR training signer | 424 |
| YouTube-ASL + OpenASL | 810 |

Generated v1/v2 trajectories were not loaded.

## Selected result

Epoch 4 passed every fixed guard and was selected:

| Domain | Frozen initialization | Selected model | Change |
| --- | ---: | ---: | ---: |
| Local all-nine validation | 16.07% | 20.74% | +4.67 points |
| ASLLRP `OTHER` signer-held-out | 23.08% | 23.50% | +0.42 points |
| NCSLGR held-out signer | 20.41% | 21.27% | +0.86 points |
| How2Sign replay guard | 25.87% | 25.41% | -0.45 points, within the 0.50-point gate |
| YouTube-ASL replay guard | 10.15% | 10.74% | +0.59 points |
| OpenASL reference-only | 0.43% | 1.79% | +1.36 points |

Cold reload on every compatible train-side source is positive. With all nine local families included, the model improves 24.63% over interpolation and improves 62.38% of 1,781 local windows. The detailed cold-reload record is `all_real_checkpoint_transfer.json`.

## Decision

Using all genuine sources with explicit source balance is better than the How2Sign+YouTube-only initialization. The checkpoint is retained for motion pretraining, but it is not yet a text-to-sign generator: it reconstructs masked 4–12-frame intervals and has no full-phrase gloss conditioning or native-signer naturalness pass.

The next architecture must predict a continuous animation rig separately from the recognizer observation mask. An always-present rig may be rendered, but the original/generated observation mask—not an all-ones mask—must remain the only recognition-format signal.
