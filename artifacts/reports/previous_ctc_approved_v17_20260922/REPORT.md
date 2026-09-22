# Earlier causal CTC checkpoint on current approved validation

Evaluated 211 current approved phrase clips: 199 local and 12 ASLLRP.

| Source | Clips | Exact | WER | S / D / I |
|---|---:|---:|---:|---:|
| asllrp_contiguous | 12 | 33.33% | 45.83% | 4 / 5 / 2 |
| local_phrases | 199 | 25.63% | 38.73% | 56 / 71 / 81 |
| total | 211 | 26.07% | 39.04% | 60 / 76 / 83 |

This uses the old checkpoint's source-frame restoration and 8-frame, stride-4 causal windows. It is not the newer cached-32-window joint-CTC pipeline. No protected test data was accessed.
