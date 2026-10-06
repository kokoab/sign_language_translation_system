# 96.83% chain (chain_9683): verified results

## Step 2 — rotated video with automatic orientation correction

Same 100 Citizen validation clips (one per class) for every row; pixels rotated on an expanded canvas, Vision re-run with auto-orientation. Correct out of 100.

| Model | 0° | 17° | 37° | 73° | 90° | 123° | 180° | 270° | 8-angle mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96.83 (5c40b133) | 96 | 94 | 80 | 96 | 96 | 88 | 96 | 96 | 92.75% |
| a7490409 rerun | 93 | 96 | 85 | 94 | 93 | 91 | 93 | 93 | 92.25% |
| a7490409 August record | 93 | 95 | 82 | 94 | 93 | 89 | 93 | 93 | 91.50% |
| new local branch | 96 | 94 | 80 | 96 | 96 | 88 | 96 | 96 | 92.75% |

a7490409 rerun reproduces the August per-angle record exactly: **False**.

## Step 3a — local replay from 96.83

Epochs completed: 20 (patience 20, full-roll probability 0.35). Parent initial masked local Top-1: 64.12%. Gates: Citizen ≥365/378, SemLex ≥853/978, local > parent, worst landmark-roll angle ≥2/378.

| Candidate | Epoch | Citizen | SemLex | Local masked | Worst rotated angle | Mean rotated | Gates passed |
|---|---:|---:|---:|---:|---:|---:|---|
| promotion_gate | 0 | 366/378 (96.83%) | 853/978 (87.22%) | 64.12% | 2/378 | 35.71% | citizen, semlex, orientation |
| citizen_best | 0 | 366/378 (96.83%) | 853/978 (87.22%) | 64.12% | 2/378 | 35.71% | citizen, semlex, orientation |
| August branch (a7490409 parent) | 21 | 361/378 (95.50%) | 860/978 (87.93%) | 96.34% | 356/378 | — | reference |

Selected for fusion: **promotion_gate** (gate-eligible: False).

## Step 3b — multimodal fusion and distillation

Hand branch, four-stream teacher, data, seeds and selection rule unchanged; caches rebuilt and their landmark-encoder hashes verified.

| Run | Seed | Epoch | Citizen | SemLex | Local |
|---|---:|---:|---:|---:|---:|
| August record | 5101 | 15 | 364/378 (96.30%) | 871/978 (89.06%) | 2812/2896 (97.10%) |
| August rerun (fresh cache) | 5101 | 15 | 364/378 (96.30%) | 871/978 (89.06%) | 2812/2896 (97.10%) |
| New (96.83 chain) | 3407 | 4 | 366/378 (96.83%) | 862/978 (88.14%) | 2447/2896 (84.50%) |

August fusion rerun reproduces the recorded selection exactly: **True**.

Per-seed (new chain): seed 1701 epoch 2: 366/864/2225; seed 3407 epoch 4: 366/862/2447; seed 5101 epoch 1: 367/865/2008

## Not run

Phrase-activity adaptation and the interval recognizer: canonical phrase verifier reports training_ready=false.

Validation development evidence only. SemLex validation reuses SemLex train signers; local validation is familiar-signer. Landmark/pixel rotation is not phone evidence. No test access, promotion or manuscript change.
