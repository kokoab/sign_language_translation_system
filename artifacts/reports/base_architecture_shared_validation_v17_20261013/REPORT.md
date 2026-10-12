# Base architecture evaluation on shared validation — 2026-10-13

Pinned base checkpoints, identical3957inputs/targets to current-lineage component evaluation. Existing25SemLex exclusions applied before inference. No training, test access, or timing benchmark.

| Architecture | Parameters | Correct | Accuracy |
|---|---:|---:|---:|
| Flat Transformer | 6.72 M | 2929/3957 | 74.02% |
| Flat Squeezeformer | 6.47 M | 2966/3957 | 74.96% |
| Part-wise and global Squeezeformer | 6.79 M | 3035/3957 | 76.70% |

Citizen-only results still reproduce361/362/366 correct (95.50/95.77/96.83%). Broader data include local signing before these base models were adapted to it; local correct1756/1789/1841 outof2626. SemLex812/815/828 outof953. These results do not replace final-primary95.50% with a base-model number. Same evaluation membership does not establish matched training or causal superiority.

Inference loop: 19.7 seconds; not a latency benchmark.
