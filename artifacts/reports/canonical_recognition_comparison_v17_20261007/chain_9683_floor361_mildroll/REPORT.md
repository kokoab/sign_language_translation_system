# 96.83% chain (chain_9683_floor361_mildroll): verified results

## Step 2 — rotated video with automatic orientation correction

Same 100 Citizen validation clips (one per class) for every row; pixels rotated on an expanded canvas, Vision re-run with auto-orientation. Correct out of 100.

| Model | 0° | 17° | 37° | 73° | 90° | 123° | 180° | 270° | 8-angle mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96.83 (5c40b133) | 96 | 94 | 80 | 96 | 96 | 88 | 96 | 96 | 92.75% |
| a7490409 rerun | 93 | 96 | 85 | 94 | 93 | 91 | 93 | 93 | 92.25% |
| a7490409 August record | 93 | 95 | 82 | 94 | 93 | 89 | 93 | 93 | 91.50% |
| new local branch | 96 | 97 | 86 | 96 | 96 | 95 | 96 | 96 | 94.75% |

a7490409 rerun reproduces the August per-angle record exactly: **False**.

## Step 3a — local replay from 96.83

Epochs completed: 80 (patience 80, full-roll probability 0). Parent initial masked local Top-1: 64.12%. Gates: Citizen ≥361/378, SemLex ≥853/978, local > parent, worst landmark-roll angle ≥2/378.

| Candidate | Epoch | Citizen | SemLex | Local masked | Worst rotated angle | Mean rotated | Gates passed |
|---|---:|---:|---:|---:|---:|---:|---|
| promotion_gate | 55 | 368/378 (97.35%) | 854/978 (87.32%) | 98.17% | 7/378 | 39.46% | citizen, semlex, local, orientation |
| citizen_best | 55 | 368/378 (97.35%) | 854/978 (87.32%) | 98.17% | 7/378 | 39.46% | citizen, semlex, local, orientation |
| August branch (a7490409 parent) | 21 | 361/378 (95.50%) | 860/978 (87.93%) | 96.34% | 356/378 | — | reference |

Selected for fusion: **promotion_gate** (gate-eligible: True).

## Step 3b — multimodal fusion and distillation

Hand branch, four-stream teacher, data, seeds and selection rule unchanged; caches rebuilt and their landmark-encoder hashes verified.

| Run | Seed | Epoch | Citizen | SemLex | Local |
|---|---:|---:|---:|---:|---:|
| August record | 5101 | 15 | 364/378 (96.30%) | 871/978 (89.06%) | 2812/2896 (97.10%) |
| August rerun (fresh cache) | 5101 | 15 | 364/378 (96.30%) | 871/978 (89.06%) | 2812/2896 (97.10%) |
| New (96.83 chain) | 1701 | 0 | 369/378 (97.62%) | 861/978 (88.04%) | 2844/2896 (98.20%) |

August fusion rerun reproduces the recorded selection exactly: **True**.

Per-seed (new chain): seed 1701 epoch 0: 369/861/2844; seed 3407 epoch 0: 369/861/2844; seed 5101 epoch 0: 369/861/2844

## Not run

Phrase-activity adaptation and the interval recognizer: canonical phrase verifier reports training_ready=false.

Validation development evidence only. SemLex validation reuses SemLex train signers; local validation is familiar-signer. Landmark/pixel rotation is not phone evidence. No test access, promotion or manuscript change.

## Interpretation notes

- This is the corrected chain (Variant C in CHAIN_PLAN.md): full-roll augmentation off to match
  the 96.83 parent's training. The earlier chain_9683/ local and fusion results are confounded
  by mismatched full-roll augmentation and must not be used.
- Variant C's floor (361) and patience (80) were chosen after the first result (user-approved).
  The selected epoch 55 also satisfies the stricter predeclared 365 floor (368/378).
- Epoch 55 is the single Citizen maximum (368); epochs 45–58 sit at 367, so 97.35% is
  validation-selected and slightly optimistic; the plateau is about 97.09%.
- Fusion: every seed selected epoch 0, the fixed untrained 75/25 z-scored landmark/hand
  score fusion. Trained residual/distillation epochs peaked at 368/861/2843, slightly below
  it. The multimodal gain here comes from the fixed score fusion, not learned distillation.
- Landmark-space roll robustness is low (worst 7/378) because full roll was off; with Vision
  auto-orientation on rotated video the branch scores 94.75% (eight-angle mean, 100 clips).
