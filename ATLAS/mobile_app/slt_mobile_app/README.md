# ASL Translator

iOS-first Flutter app for the locked v17 mobile pipeline:

1. Record a complete sign or short phrase with the front camera.
2. Apple Vision extracts orientation-corrected landmarks and RGB hand crops.
3. A one-sign CTC result is reclassified by the stronger full-activity Stage 1
   hybrid. Multi-sign results use the pinned Stage 2 CTC phrase head.
4. Stage 3 renders an exact reviewed English template when available or shows the
   gloss-preserving literal fallback.

Every attempt is retained under **Files → On My iPhone → ASL Translator →
Diagnostics**. Each unique capture directory contains the original MP4, JSON report,
raw and model-ready landmarks, masks, hand boxes, and exact hand embeddings as NumPy
files. Failed extraction attempts retain the video and an `error.json` file.

The Settings sheet contains an off-by-default **Device benchmark** switch. When
enabled, one extraction and 20 timed model runs produce median/p90 inference,
resident memory, thermal state, model/checkpoint hashes, and a shareable JSON
report. Simulator reports explicitly deny hardware-performance and thermal claims.
Core ML uses all compute units selected by iOS. Models are cached for the app session,
and Apple Vision hand observations are reused for both landmarks and RGB crops.

## Run on an iPhone

Requirements: iOS 17+, Xcode signing for team `2BJ6GHCUJ2`, and the five local
Core ML packages under `ios/Runner/Models/`.

```sh
flutter pub get
flutter devices
flutter run --release -d <iphone-device-id>
```

The model packages are intentionally ignored by Git. Android native inference is
not implemented in this version.
