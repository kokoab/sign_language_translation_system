# Locked-100 mobile Stage 2 and Stage 3 handoff

## Status

The selected compact Stage 2 recognizer is ready to provide a frozen 100-gloss
sequence boundary to Stage 3. This does **not** mean that the existing Stage 3
translator is production-ready; its translation quality remains a separately failed
research gate.

The mobile recognizer consists of exactly three Core ML packages:

| Component | Tree SHA-256 |
| --- | --- |
| MobileCLIP2-S0 hand image encoder | `9309548dd69a5c8e899ea00ee4f0bbe88505ed803d12520a60e5d954ff370974` |
| Frozen multimodal temporal encoder | `1146b539800e6f09a743f4a8ee882c9b2cd2b01503ff3f362a11e97e8c827bb9` |
| Compact context/CTC head | `e92ba7d8b7c61c52bc776840e953c73abb6b012637991d01582d4fd64067760a` |

The selected Stage 2 checkpoint is SHA-256
`623f9b56141643704b3562a8d2fdcebe44269985b2f618eb8f0a471e857a2cf5`.
The vocabulary manifest is SHA-256
`3a665bda8d2b916c504406be815e601eeb55badfe62afcec42c7869885eab7cf`.

## Validation evidence

`full_coreml_validation.json` regenerates every RGB hand embedding through the Core ML
image tower and then runs both Stage 2 packages. It covers 363 validation samples,
574 windows, and 22,046 valid hand crops with zero decoded-sequence differences from
the cached Core ML and PyTorch paths. All 363 outputs also pass the frozen Stage 3
handoff validator.

| Validation source | Edits / reference tokens | Exact sequences |
| --- | ---: | ---: |
| ASLLRP contiguous phrases | 11 / 24 | 2 / 12 |
| Local phrases | 7 / 259 | 90 / 97 |
| ASLLRP segmented contextual | 43 / 254 | 213 / 254 |

The MobileCLIP2 export separately passes full class-spanning parity on all 378 Citizen
validation clips. Maximum absolute error versus PyTorch is `1.49012e-06`; minimum
cosine is `0.999999881`.

## Stage 2 to Stage 3 interface

The authoritative contract is
`active/v17/stage2_to_stage3_contract_v17.json`, SHA-256
`8be66a44d337dd99484d3ee3140f3124c2e121abe20e93ce7f09b94d96ecc30d`.
Stage 2 emits `slt_stage2_gloss_sequence_v17` objects with ordered one-based CTC token
indices and their exact frozen labels. Blank is `0`; valid gloss tokens are `1...100`.
There is no unknown token. Stage 3 must reject checkpoint/vocabulary hash mismatches,
must not merge synonyms, and must treat an empty recognized sequence as valid.

## iPhone 13 simulator evidence

The final result is
`artifacts/reports/orientation_v17_simulator_benchmark/latest_result.json`. On the
dedicated iPhone 13 simulator (`iPhone14,5`), all eight expanded-canvas rotations
0/17/37/73/90/123/180/270 degrees predicted `HELLO`. Each angle completed exactly 200
timed Stage 2 inferences, and every one of the 1,600 decoded iteration votes was token
15 (`HELLO`). Exact quadrant corrections and the <=45-degree residual-roll gate pass.

This is functional simulator evidence only. Apple Vision ran on the macOS host because
the simulator runtime omits its pose weights; the simulator itself decoded the real
hand crops and ran all three Core ML models plus CTC. It is not physical-iPhone
latency, memory, thermals, ANE, or camera-to-gloss accuracy evidence.

## Integrity

- Unsigned Release builds pass for both the simulator and generic iPhoneOS, and each
  final app bundle contains exactly the three selected Core ML models.
- The untouched independent capture ledger still contains 1,000 target and 100 OOV
  plans; its setup audit passes without model inference or test access.
- Ninety-six unique focused extractor, RGB, Stage 1/2, Stage 3 contract, simulator,
  and capture-pack tests pass.
- Changed Python entry points compile, generated JSON parses, and `git diff --check`
  passes.
- No Citizen, SemLex, local, or 2M-Flores test split was accessed.
