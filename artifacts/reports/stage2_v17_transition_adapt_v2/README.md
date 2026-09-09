# Stage-2 transition adaptation: matched failure result

Input SHA-256: `0468e3f491f1872f7be8a915eeb994581801f99006b6bee6e630e2d6f134cda7`. Selection is **null**; runtime action is `retain_current_selector`.

The per-example evidence bundle covers six snapshots over all seven domains: baseline selector, the epoch-0 initialized candidate (identical across all four histories), and all four best seed checkpoints (7,272 rows).
No arm qualified: every seed improved target-only ASLLRP OTHER WER, but each violated one or more frozen retention gates.

| Arm / seed | target WER | OTHER-inclusive WER | local familiar-domain edits | exact edits | contextual edits | Citizen correct | STEM held participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline selector | 2.1232 | 1.0279 | 6/259 | 9/24 | 43/254 | 331/378 | 16/21 |
| initialized candidate / epoch 0 | 2.2324 | 1.0718 | 7/259 | 11/24 | 56/254 | 334/378 | 16/21 |
| no_stem / 1701 | 0.7535 | 0.5191 | 13/259 | 16/24 | 54/254 | 332/378 | 13/21 |
| no_stem / 1702 | 0.7641 | 0.5425 | 13/259 | 14/24 | 52/254 | 328/378 | 14/21 |
| with_stem / 1701 | 0.7641 | 0.5220 | 15/259 | 16/24 | 57/254 | 333/378 | 16/21 |
| with_stem / 1702 | 0.7852 | 0.5249 | 14/259 | 14/24 | 51/254 | 331/378 | 15/21 |

## OTHER-inclusive validation details

The accepted selector has no OTHER output; its row is descriptive only and is not a comparative baseline or selection gate.

| Run | edits/tokens | WER | exact/samples | sequence accuracy | substitutions | deletions | insertions | repeated-sign errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline selector | 701/682 | 1.0279 | 0/225 | 0.0000 | 474 | 80 | 147 | 44 |
| initialized candidate / epoch 0 | 731/682 | 1.0718 | 0/225 | 0.0000 | 480 | 74 | 177 | 44 |
| no_stem / 1701 | 354/682 | 0.5191 | 16/225 | 0.0711 | 96 | 209 | 49 | 176 |
| no_stem / 1702 | 370/682 | 0.5425 | 21/225 | 0.0933 | 112 | 204 | 54 | 197 |
| with_stem / 1701 | 356/682 | 0.5220 | 22/225 | 0.0978 | 114 | 181 | 61 | 213 |
| with_stem / 1702 | 358/682 | 0.5249 | 13/225 | 0.0578 | 124 | 168 | 66 | 228 |

## STEM held-participant details

| Run | edits/tokens | WER | exact/samples | sequence accuracy | substitutions | deletions | insertions | repeated-sign errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline selector | 5/21 | 0.2381 | 16/21 | 0.7619 | 4 | 0 | 1 | 1 |
| initialized candidate / epoch 0 | 5/21 | 0.2381 | 16/21 | 0.7619 | 4 | 0 | 1 | 1 |
| no_stem / 1701 | 9/21 | 0.4286 | 13/21 | 0.6190 | 4 | 1 | 4 | 1 |
| no_stem / 1702 | 7/21 | 0.3333 | 14/21 | 0.6667 | 4 | 1 | 2 | 1 |
| with_stem / 1701 | 5/21 | 0.2381 | 16/21 | 0.7619 | 4 | 0 | 1 | 1 |
| with_stem / 1702 | 6/21 | 0.2857 | 15/21 | 0.7143 | 4 | 0 | 2 | 1 |

Best unqualified checkpoint for comparison: `artifacts/models/stage2_v17_transition_adapt_v2/no_stem/seed_1701/best_model.pth` (1919f49a7097434700c6460f6696d554705c3a13287d39a716d7286cb109c287).
Target-only by-position aggregates, baseline: `{"first": 258, "last": 52, "middle": 3}`; best candidate: `{"first": 150, "last": 50, "middle": 3}`.
Target-only by-duration aggregates, baseline: `{"2_windows": 44, "3_windows": 76, "4_windows": 53, "5_windows": 33, "6_windows": 10, "7_windows": 2, "8_windows": 3}`; best candidate: `{"2_windows": 30, "3_windows": 55, "4_windows": 45, "5_windows": 29, "6_windows": 8, "7_windows": 2, "8_windows": 3}`.

Frozen gates: target <=542 edits (10% relative from 603), local <=6, exact <=9, contextual <=43, Citizen >=328/378. Local phrase is familiar-domain retention, not a generalization claim.

Provenance: manifest `d120d9747ed01bc7d7cc0d68d2609a25114c75826484536ee942c278820d4af0`, frozen inputs `063cdb1314e62e0c20784687a4cfb112778da9dadc9e9f8251a6e3b4a611e274`, warm start `6a72ca836247fe8717e0b4a7f930b11b291aa2889b49cc839a9fbc6d86d6cf8e`, teacher `0782d052f0500164a2433ebfee86dcce7413c6bcffca03fae379871ece86dc3d`.

Reproducible rollback launch (accepted selector, unchanged):
```sh
venv/bin/python scripts/live_reel_continuous_v17.py --camera 0
```

Protected-test flags are all false. This result does not establish iPhone performance or general continuous-ASL accuracy; no further sweep was launched.
