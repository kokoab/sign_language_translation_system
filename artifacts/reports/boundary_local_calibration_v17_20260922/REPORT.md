# Local BIO calibration and confirmation

No training or deployment. Selection frozen before confirmation.

| Phase | Candidate | WER | Correct | Insertions | Retained baseline |
|---|---|---:|---:|---:|---:|
| calibration | frozen_min3 | 49.33% | 82/150 | 6 | 82/82 |
| calibration | frozen_min5 | 56.00% | 70/150 | 4 | 70/82 |
| calibration | adapted_min3 | 53.33% | 81/150 | 11 | 78/82 |
| calibration | adapted_min5 | 57.33% | 73/150 | 9 | 71/82 |
| confirmation | frozen_min3 | 43.42% | 46/76 | 3 | 46/46 |

Selected: **frozen_min3**. Confirmation improvement gate passed: **False**.
Elapsed: 11.2 minutes. Calibration59 clips; confirmation30; unused282.

Familiar-signer repeated-phrase development. No session IDs/near-duplicate certification; exact hashes unique. Transcript provenance validated, not new human relabeling. 6templates/15glosses. No interval/gap labels.

Both backbones use original BIO, same500mslookahead/EOF handling, frozen Reel and100ms classifier context. Only minimum segment length3 versus5frames changes. Filtering reuses identical per-interval predictions; no repeated extraction or classifier work for the duration variants.
Original371/60roles unchanged. New experiment only reserves the selected local clips from future boundary fitting; does not authorize training unused clips.72video expanded set untouched. No interval-level transition claim or unseen-signer/100sign accuracy claim.
Hashes and transcript membership verified against approved manifests; natural-recording near duplicates/session linkage and prior Reel-training exposure remain limitations. Evidence is relative fixed-Reel calibration, not a fresh whole-pipeline test.

No challenger qualified on calibration. Confirmation evaluated the selected frozen baseline only; no head-to-head improvement is claimed.

| Phase | Signer | Correct | WER |
|---|---|---:|---:|
| calibration | local_signer_01 | 13/52 | 75.00% |
| calibration | local_signer_02 | 42/60 | 35.00% |
| calibration | local_signer_03 | 27/38 | 36.84% |
| confirmation | local_signer_01 | 10/26 | 61.54% |
| confirmation | local_signer_02 | 22/30 | 33.33% |
| confirmation | local_signer_03 | 14/20 | 35.00% |
