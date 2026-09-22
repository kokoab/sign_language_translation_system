# Completed frozen/adapted comparison review

4 checkpoint hashes; 12 epochs each; earliest-best selection and saved metrics; per-epoch coverage.

| Run | Selected epoch | Local train WER | Local validation WER | Local exact /199 | ASLLRP validation WER | ASLLRP exact /12 |
|---|---:|---:|---:|---:|---:|---:|
| 17421:frozen | 9 | 9.19% | 60.89% | 13 | 62.50% | 0 |
| 17421:adapted | 8 | 0.48% | 50.84% | 29 | 54.17% | 0 |
| 17422:frozen | 11 | 11.25% | 66.85% | 5 | 58.33% | 1 |
| 17422:adapted | 3 | 1.74% | 53.26% | 22 | 62.50% | 0 |

Adaptation helps local phrase recognition relative to the matched frozen arms, but both adapted seeds retain a large training/validation gap. The 12 ASLLRP validation clips remain weak and too small for strong conclusions. Citizen pooled Stage1 validation accuracy changes from95.24% to94.71%/94.18%; these are validation figures under this input contract, not protected-test accuracy.

The prior clean phrase-only baseline had roughly49–50% overall validation WER; this is not a controlled comparison to that run because its architecture/input/training recipe differ. No demonstrated continuous-recognition breakthrough or live promotion. Shared-signer sources remain labeled as such; no global signer-disjoint claim.

Next: recover per-record predictions from selected checkpoints and compare known sign identity against missing/extra sequence events, using existing annotated cores. Do not extend training merely because training loss falls. All five recovery steps remain open where live/event evidence is missing.

## Aggregate clarification — 2026-09-22

Previously reported60.89/66.85% frozen and50.84/53.26% adapted WER referred only to199local phrase validation clips. Token-weighted combined phrase validation (211clips,561reference signs): frozen60.96/66.49%, adapted50.98/53.65%. All admitted validation sources combined (1874records,2224reference signs): frozen38.67/39.03%, adapted27.56/32.69%. Single-sign validation alone (1663records): frozen31.15/29.77%, adapted19.66/25.62% CTC WER. Aggregate computed as total substitutions+deletions+insertions divided by total reference signs, not mean source WER. It describes only this recipe's validation membership, not all datasets in the repository. The aggregate is dominated by single-sign examples and cannot stand in for continuous performance.

Next-step recommendation is a bounded matched identity-versus-sequence diagnostic, not an established unique best intervention. Existing gap proves weak generalization under this evaluation, not its cause. Compare whole-sequence versus annotated-core outputs with the same selected weights, documenting any frontend/resampling differences; review whether failures follow source/input conditions before choosing new training or extraction. No additional training authorized by this clarification alone beyond the existing recovery scope.
