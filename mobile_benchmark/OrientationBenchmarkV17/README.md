# Locked-100 Stage 2 mobile benchmark

For physical-phone signing, installation, first-run, report export, and troubleshooting
instructions, see [`DEPLOY_TO_IPHONE.md`](DEPLOY_TO_IPHONE.md).

This project is the iOS integration and simulator gate for the selected v17 compact
Stage 2 recognizer. It emits an ordered sequence drawn only from the frozen 100-gloss
vocabulary. The app bundles and runs all three neural components in Core ML:

1. `MobileCLIP2S0ImageEncoderV17FP32` converts each real 256x256 left-hand,
   right-hand, and union crop into the normalized 512-D training embedding.
2. `Stage2FrozenEncoderV17FP32` fuses the Apple Vision landmarks, hand embeddings,
   validity masks, and normalized crop boxes over at most eight windows.
3. `Stage2CompactContextV17FP32` emits 101-way CTC logits; the app greedily collapses
   blank `0` and maps tokens `1...100` to the pinned Citizen-100 label order.

The checkpoint, vocabulary, MobileCLIP2 source checkpoint, and all three Core ML
package tree hashes are pinned in `Stage2MobileV17_manifest.json`. The machine-readable
downstream boundary is `active/v17/stage2_to_stage3_contract_v17.json`.

## Physical-iPhone status

All mobile neural models are implemented and an unsigned generic iPhoneOS Release
build passes. The interactive file picker now decodes at most 256 evenly sampled
source frames, processes one 32-frame window at a time through Apple Vision, creates
the 16-frame left/right/union RGB crops from the same upright pixels, runs the three
Core ML models, and emits both the locked gloss sequence and bounded Stage-3 English.
It never retains all eight decoded windows at once. Physical-device accuracy,
latency, memory, and thermal evidence still requires an actual iPhone and is not
inferred from simulator or Mac-host execution.

Physical interactive reports set `videoFileToGlossEndToEnd: true` and
`cameraToGlossEndToEnd: false`: the implemented input is the document-picker video
path, not a live camera path.

## Bounded Stage 3

The bundled Stage 3 is a fail-closed naturalizer, not an open-domain neural
translator. It validates the frozen Stage-2 hashes and exact token-to-gloss mapping.
An exact reviewed sequence can use a meaning-conservative English template; every
other sequence returns the literal gloss-preserving rendering. The UI and JSON report
always expose the recognized glosses, literal English, rendering mode, and whether
the safe fallback was used. This design is deliberate: the available genuine
gloss/English corpora contain no complete sentence restricted to the locked 100
labels, so training a general translator by dropping unsupported signs would corrupt
the references.

## Automated iPhone 13 Simulator benchmark

Run the fixed eight-angle suite with:

```bash
venv/bin/python scripts/run_orientation_simulator_benchmark_v17.py
```

The runner requests iOS 26.2 and otherwise uses the nearest compatible iOS 26.x
runtime. It creates or reuses only the dedicated
`SLT Orientation Benchmark iPhone 13`, generates expanded-canvas
0/17/37/73/90/123/180/270-degree inputs without anisotropic stretching or cropping,
builds and installs the Release app, executes 200 Stage 2 inferences per angle, and
retrieves JSON evidence beneath
`artifacts/reports/orientation_v17_simulator_benchmark/`.

The available iOS 26.3.1 simulator runtime omits Apple Vision pose Espresso weights.
The host therefore creates landmarks and real-pixel JPEG hand crops together after
one shared v17 coarse-orientation correction. It does not create hand embeddings.
The iPhone 13 simulator decodes those crops and runs MobileCLIP2, the multimodal
encoder, the compact CTC head, and CTC collapse itself. Reports consequently set:

- `endToEndPipeline: true` for the complete mobile neural path;
- `allMobileNeuralModelsInCoreML: true`;
- `videoFileToGlossEndToEnd: false` because Vision ran on the macOS host;
- `cameraToGlossEndToEnd: false` because Vision ran on the macOS host;
- `hardwarePerformanceClaim: false` and `thermalsInterpretable: false`.

This is valid functional simulator evidence for orientation, crop serialization,
model loading, inference, and decoding. It is not physical-iPhone latency, memory,
thermal, ANE, or independent-capture accuracy evidence.
