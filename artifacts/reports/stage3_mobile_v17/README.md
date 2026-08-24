# Locked-100 mobile Stage 2 + bounded Stage 3 evidence

**Status:** pass — ready for signing and interactive file-video testing on an iPhone.

## Promoted design

The retained Stage-2 recognizer is unchanged and remains pinned by
`active/v17/stage2_to_stage3_contract_v17.json` (SHA-256
`8be66a44d337dd99484d3ee3140f3124c2e121abe20e93ce7f09b94d96ecc30d`).
The mobile Stage-3 consumer validates that contract and renders either:

- one of 35 exact, meaning-conservative English templates; or
- a deterministic literal rendering that preserves every recognized gloss in order.

Both the literal and naturalized strings, rendering mode, and fallback flag are always
visible in the UI and JSON. The component never deletes, replaces, reorders, or invents
a recognized gloss. It is deliberately described as a bounded naturalizer, not a
general or open-domain neural translator.

## Why no neural Stage 3 was promoted

The fail-closed data audit examined 999 genuine 2M-Flores `dev` pairs and 166 genuine
NCSLGR pairs. None of the 1,165 complete gloss sequences is fully expressible using the
locked 100-gloss Stage-2 vocabulary. Training a translator on truncated inputs would
silently discard out-of-vocabulary meaning. The 15,843-row legacy set is synthetic and
is retained only as development reference, not genuine validation evidence.

On the existing Stage-2 ground-truth validation sequences, exact templates cover 85/97
local phrase rows. The remaining 12 local rows and all 12 ASLLRP contiguous rows use the
literal fallback. This measures deterministic rendering coverage, not translation
quality or independent signer accuracy.

## Mobile implementation

The iOS app now performs the complete interactive file-video path:

1. read an arbitrary native aspect ratio/orientation without anisotropic stretching;
2. sample at most 256 source frames;
3. process at most eight non-overlapping 32-frame windows, one window at a time;
4. run Apple Vision landmarks and generate 16-frame real-pixel left/right/union crops;
5. run MobileCLIP2, the frozen multimodal encoder, and compact CTC head in Core ML;
6. validate the Stage-2 output contract and render bounded Stage-3 English.

The unsigned generic iPhoneOS Release app is
`artifacts/generated/stage3_mobile_release_device/OrientationBenchmarkV17.app`.
It is a 110,350,170-byte arm64 bundle containing exactly these three compiled models:

- `MobileCLIP2S0ImageEncoderV17FP32.mlmodelc`
- `Stage2FrozenEncoderV17FP32.mlmodelc`
- `Stage2CompactContextV17FP32.mlmodelc`

The bundled Stage-3 manifest exactly matches the source SHA-256
`68c7ce67632f66ee70fa3b3d36eb8df33ad72dc674edbf3b720e93c1240f84a6`.

## Validation

- Swift video-to-English host gate: 8/8 expanded-canvas HELLO rotations passed;
  maximum process RSS was 279,773,184 bytes. This is Mac-host memory evidence only.
- Final iPhone 13 simulator gate: 8/8 angles passed at
  0/17/37/73/90/123/180/270 degrees, with exactly 200 timed inferences per angle and
  Stage 3 output `Hello.`.
- Final generic iPhoneOS Release build: pass.
- Focused Python regressions: 157/157 pass.
- Deployment audit: pass with zero acceptance errors.
- Citizen, SemLex, local, and 2M-Flores `devtest` splits accessed: none.

Machine-readable evidence:

- `artifacts/reports/stage3_mobile_v17/deployment_audit.json`
- `artifacts/reports/stage3_mobile_v17/data_and_coverage_audit.json`
- `artifacts/reports/stage3_mobile_v17/swift_video_to_english_validation.json`
- `artifacts/reports/orientation_v17_simulator_benchmark/latest_result.json`

## Claim boundary

The app is ready to sign, install, and test using videos selected from the iPhone file
picker. A live-camera capture UI is not implemented. No physical-iPhone latency,
memory, thermals, ANE behavior, or independent-capture accuracy has been measured, and
simulator or Mac timings must not be presented as such.
