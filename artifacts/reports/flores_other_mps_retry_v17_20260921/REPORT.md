# Flores OTHER matched comparison

Same baseline initialization per seed; only added Flores supervision differs. No YouTube pretraining.

| Seed / arm | Local WER | ASLLRP WER | NCSLGR WER | Isolated exact | OTHER without expected isolated sign |
|---|---:|---:|---:|---:|---:|
| without_flores_17321 | 48.52% | 54.17% | 92.00% | 83.92% | 3/1356 |
| with_flores_17321 | 55.56% | 50.00% | 84.00% | 83.41% | 3/1356 |
| without_flores_17322 | 50.19% | 62.50% | 98.00% | 83.41% | 0/1356 |
| with_flores_17322 | 48.33% | 41.67% | 86.00% | 81.12% | 8/1356 |

Paired WER/retention/rejection gates passed: 0/2. No automatic promotion.

Full error counts, duplicate outputs, synthetic hold/repeat accuracy and conditional ASLLRP emission delay: comparison.json and behavior_*.json.
No real held/repeat test or global Flores signer identities are established. Existing development sets are reused; this is not a new unbiased test result.
