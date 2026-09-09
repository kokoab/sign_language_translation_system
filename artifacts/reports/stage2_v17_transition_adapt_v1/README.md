# Stage-2 transition adaptation: matched failure result

Input SHA-256: `badd541dbb156a4f57b8cbffc67fb8fdca36ac9a7c9c21e73954ff494b125e1e`. Selection is **null**; runtime action is `retain_current_selector`.

The per-example evidence bundle covers six snapshots over all seven domains: baseline selector, the epoch-0 initialized candidate (identical across all four histories), and all four best seed checkpoints (7,272 rows).
No arm qualified: every seed improved target-only ASLLRP OTHER WER, but each violated one or more frozen retention gates.

| Arm / seed | target WER | OTHER-inclusive WER | local familiar-domain edits | exact edits | contextual edits | Citizen correct | STEM held participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline selector | 2.1232 | 1.0279 | 6/259 | 9/24 | 43/254 | 331/378 | 0/21 |
| initialized candidate / epoch 0 | 2.2324 | 1.0718 | 7/259 | 11/24 | 56/254 | 334/378 | 0/21 |
| no_stem / 1701 | 0.7500 | 0.5220 | 12/259 | 16/24 | 57/254 | 331/378 | 0/21 |
| no_stem / 1702 | 0.7641 | 0.5440 | 14/259 | 14/24 | 51/254 | 327/378 | 0/21 |
| with_stem / 1701 | 0.7782 | 0.5337 | 15/259 | 15/24 | 50/254 | 322/378 | 4/21 |
| with_stem / 1702 | 0.7887 | 0.5513 | 14/259 | 15/24 | 53/254 | 324/378 | 3/21 |

## OTHER-inclusive validation details

The accepted selector has no OTHER output; its row is descriptive only and is not a comparative baseline or selection gate.

| Run | edits/tokens | WER | exact/samples | sequence accuracy | substitutions | deletions | insertions | repeated-sign errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline selector | 701/682 | 1.0279 | 0/225 | 0.0000 | 474 | 80 | 147 | 44 |
| initialized candidate / epoch 0 | 731/682 | 1.0718 | 0/225 | 0.0000 | 480 | 74 | 177 | 44 |
| no_stem / 1701 | 356/682 | 0.5220 | 19/225 | 0.0844 | 92 | 229 | 35 | 151 |
| no_stem / 1702 | 371/682 | 0.5440 | 19/225 | 0.0844 | 113 | 214 | 44 | 184 |
| with_stem / 1701 | 364/682 | 0.5337 | 14/225 | 0.0622 | 115 | 191 | 58 | 201 |
| with_stem / 1702 | 376/682 | 0.5513 | 19/225 | 0.0844 | 118 | 213 | 45 | 184 |

## STEM held-participant details

| Run | edits/tokens | WER | exact/samples | sequence accuracy | substitutions | deletions | insertions | repeated-sign errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline selector | 22/21 | 1.0476 | 0/21 | 0.0000 | 21 | 0 | 1 | 1 |
| initialized candidate / epoch 0 | 22/21 | 1.0476 | 0/21 | 0.0000 | 21 | 0 | 1 | 1 |
| no_stem / 1701 | 25/21 | 1.1905 | 0/21 | 0.0000 | 19 | 2 | 4 | 1 |
| no_stem / 1702 | 23/21 | 1.0952 | 0/21 | 0.0000 | 20 | 1 | 2 | 1 |
| with_stem / 1701 | 17/21 | 0.8095 | 4/21 | 0.1905 | 14 | 1 | 2 | 1 |
| with_stem / 1702 | 18/21 | 0.8571 | 3/21 | 0.1429 | 13 | 2 | 3 | 1 |

Best unqualified checkpoint for comparison: `artifacts/models/stage2_v17_transition_adapt_v1/no_stem/seed_1701/best_model.pth` (1d2fa04269faf766a1d40ae4d186d7698cb20b0c590253ad8f8e55838b33d748).
Target-only by-position aggregates, baseline: `{"first": 258, "last": 52, "middle": 3}`; best candidate: `{"first": 151, "last": 50, "middle": 3}`.
Target-only by-duration aggregates, baseline: `{"2_windows": 44, "3_windows": 76, "4_windows": 53, "5_windows": 33, "6_windows": 10, "7_windows": 2, "8_windows": 3}`; best candidate: `{"2_windows": 29, "3_windows": 55, "4_windows": 45, "5_windows": 29, "6_windows": 8, "7_windows": 2, "8_windows": 3}`.

Frozen gates: target <=542 edits (10% relative from 603), local <=6, exact <=9, contextual <=43, Citizen >=328/378. Local phrase is familiar-domain retention, not a generalization claim.

Provenance: manifest `2404aa96c03f5949d6cd3af914f1e97381c279abd980b2e4c5a37b7ce516e2de`, frozen inputs `d12a94d85b1c59175771f7fcfbcc7fde722717f3ad4c2b31f041af954186a8ff`, warm start `6a72ca836247fe8717e0b4a7f930b11b291aa2889b49cc839a9fbc6d86d6cf8e`, teacher `0782d052f0500164a2433ebfee86dcce7413c6bcffca03fae379871ece86dc3d`.

Reproducible rollback launch (accepted selector, unchanged):
```sh
venv/bin/python scripts/live_reel_continuous_v17.py --camera 0
```

Protected-test flags are all false. This result does not establish iPhone performance or general continuous-ASL accuracy; no further sweep was launched.
