# ATLAS Mobile Deployment Plan

Complete deployment guide for shipping the ATLAS sign language translation system to iOS and Android.

**Current state (2026-04-16):**
- Stage 1 (Squeezeformer, d=384, 5ch): **96.00% test accuracy**
- Stage 2 (CTC, shared encoder): **5.07% WER**
- Stage 3 (Flan-T5-Small): retrain pending
- All models: PyTorch, trained on Apple Vision extracted landmarks

**DEPLOYMENT STATUS UPDATE:** Conversion pipeline has already been built and validated.
See `mobile_export/reports/CONVERSION_BRIEFING.md` and `mobile_export/reports/BRIEFING_UPDATE_500.md`.

### What's Already Done (Phase 1 + Phase 2 of original plan)

All Stage 1 and Stage 2 conversions complete with **zero accuracy loss on 500-sample validation**:

| Format | Stage 1 | Stage 2 | Location |
|---|---|---|---|
| **ONNX fp32** | 59 MB, 10ms, 100% match | 114 MB dynamic, 12.7ms, 100% match | `mobile_export/artifacts/*.onnx` |
| **CoreML FP16** | 29 MB, 1ms | Split A+B = 56.5 MB, 1.9ms | `mobile_export/artifacts/coreml/*.mlpackage` |
| **TFLite fp32** | 58 MB, 2.9ms | 5×53 MB (S4/8/12/16/20) = 265 MB, 5.3ms | `mobile_export/artifacts/tflite/*.tflite` |

Stage 3 (T5): encoder + 1-step decoder exported to ONNX. **Still needs:** KV-cache wiring + native generation loop.

### Key Findings from Conversion Work

1. **Stage 2 must be split into A (ClipEncoder) + B (SeqCTC)** — both CoreML and TFLite can't express the `T//32` dynamic reshape in Stage 2's forward. Split is mathematically identical (max|Δ| = 0.00).

2. **ManualMHA swap required** — `torch.nn.MultiheadAttention` fused op (`aten::_native_multi_head_attention`) isn't in CoreML or TFLite op sets. Unfused equivalent has same math and weights. See `mobile_export/scripts/coreml_convert_v2.py::swap_mha_inplace`.

3. **TFLite needs one .tflite per sequence length** — CoreML handles `EnumeratedShapes` in one file, TFLite can't. Produced S ∈ {4, 8, 12, 16, 20} = 1-5 seconds of signing. For MVP, ship S4 + S20 only (~80 MB FP16).

4. **No accuracy loss** — all 500 test samples, all 4 formats, 100% agreement with PyTorch on top-1 and decode sequence.

### What's Next

The deployment work is now a **build-and-integrate** task, not a "can we convert?" task:

1. Framework decision (Flutter recommended)
2. Build iOS/Android app skeletons that load the pre-converted models
3. Port `src_v16/extract_v16.py` to Swift (iOS) and Kotlin+MediaPipe (Android)
4. Integrate extraction → model pipeline
5. Decide T5 strategy (on-device with KV-cache vs cloud API)
6. UI/UX build
7. Testing

---

## 1. Core Strategy

**Keep PyTorch for training. Convert at deployment.**

Do not rewrite in TensorFlow. The conversion path (PyTorch → ONNX → CoreML/TFLite) is well-established. Rewriting introduces weeks of risk to code that currently achieves 96% accuracy.

**Deployment flow:**
```
PyTorch (.pth)
    ↓ torch.onnx.export
ONNX (.onnx) ← intermediate, validate accuracy here
    ↓
iOS: CoreML (.mlpackage) via coremltools
Android: TFLite (.tflite) via onnx-tf + tf.lite.TFLiteConverter
```

---

## 2. Platform-Specific Architecture

### iOS
```
Swift/SwiftUI app
    ↓ AVFoundation camera capture
    ↓ Apple Vision framework
        - VNDetectHumanHandPoseRequest (21 kpts × 2 hands)
        - VNDetectHumanBodyPoseRequest (shoulders, elbows)
    ↓ [32, 61, 5] landmark tensor
    ↓ CoreML model (Stage 1 + Stage 2)
    ↓ Gloss sequence
    ↓ CoreML T5 OR cloud API (Stage 3)
    ↓ English text
```

**Target:** iPhone 12+ (Neural Engine for 3-5x speedup)
**Extraction:** Apple Vision native (matches training extraction exactly)

### Android
```
Kotlin/Jetpack Compose app
    ↓ CameraX camera capture
    ↓ MediaPipe Hands + MediaPipe Pose
        - 21 kpts × 2 hands
        - body pose for shoulders/elbows
    ↓ [32, 61, 5] landmark tensor (transformed to match AV format)
    ↓ TFLite model (Stage 1 + Stage 2)
    ↓ Gloss sequence
    ↓ TFLite T5 OR cloud API (Stage 3)
    ↓ English text
```

**Target:** Android 10+, 4GB+ RAM, GPU with Vulkan support (for TFLite GPU delegate)
**Extraction:** MediaPipe (84.9% agreement with Apple Vision — close enough)

### Framework Decision: Native or Cross-Platform?

| Option | Pros | Cons |
|---|---|---|
| **Native Swift + Kotlin** | Best performance, direct platform API access, smallest app size | Two codebases, longer dev time |
| **React Native + native modules** | Single UI codebase, large community | Bridge overhead, some ML libs not mature |
| **Flutter** | Single codebase, better perf than RN | Smaller ML ecosystem, Dart learning curve |

**Recommendation:** **Native Swift + Kotlin**. The performance-critical parts (camera, pose estimation, model inference) need native anyway. UI duplication is worth it for best UX and smallest app size.

---

## 3. Deployment Phases

### Phase 1: ONNX Export + Validation (3-5 days)

**Goal:** Convert all 3 stages to ONNX, confirm accuracy preservation.

**Tasks:**
1. Write export script for Stage 1 (`src_v16/model_v16.py::SLTStage1V16`)
2. Write export script for Stage 2 (`src_v16/model_v16.py::SLTStage2V16CTC`)
3. Export Flan-T5-Small encoder and decoder separately (Hugging Face has `optimum.onnxruntime`)
4. Validate ONNX output against PyTorch on test set
   - Stage 1: accuracy within 0.01% of PyTorch
   - Stage 2: WER within 0.1% of PyTorch
   - Stage 3: BLEU within 0.5 of PyTorch

**Risks:**
- Some PyTorch ops may not have direct ONNX equivalents (e.g., `F.scaled_dot_product_attention` in newer PyTorch versions)
- Dynamic shapes for Stage 2 CTC may require extra care
- T5 decoder with past_key_values is tricky — use `optimum.onnxruntime` helpers

**Deliverable:** 3 ONNX files that pass accuracy validation.

### Phase 2: Platform Conversion (1-2 weeks)

#### iOS (CoreML)
```python
import coremltools as ct
import onnx

onnx_model = onnx.load('stage1.onnx')
mlmodel = ct.convert(
    onnx_model,
    inputs=[ct.TensorType(name='landmarks', shape=(1, 32, 61, 5))],
    compute_units=ct.ComputeUnit.ALL,  # CPU + GPU + Neural Engine
    minimum_deployment_target=ct.target.iOS15,
)
mlmodel.save('Stage1.mlpackage')
```

Expected inference on iPhone 15 Neural Engine: **~8ms** (vs 25ms PyTorch CPU).

#### Android (TFLite)
```python
# Via onnx-tf
from onnx_tf.backend import prepare
import onnx
import tensorflow as tf

onnx_model = onnx.load('stage1.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('stage1_tf')

converter = tf.lite.TFLiteConverter.from_saved_model('stage1_tf')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # FP16 quantization
tflite_model = converter.convert()
```

Expected inference on Snapdragon 8 Gen 2 with GPU delegate: **~15ms**.

**Quantization options:**
| Precision | Size | Accuracy Loss | Speed |
|---|---|---|---|
| FP32 | 56 MB | <0.01% | baseline |
| FP16 | 28 MB | ~0.1-0.3% | 1.5-2x faster |
| INT8 | 14 MB | ~1-3% | 2-4x faster |

**Recommendation:** Ship FP16 by default. Offer INT8 as a "low storage" option.

### Phase 3: Native Extraction Pipelines (1-2 weeks)

**iOS:** Port `src_v16/extract_v16.py` logic to Swift.
- Apple Vision calls → same API, different language
- Normalize on wrist, scale by palm length → pure math, straight port
- Kalman filter for missing frames → straight port
- EMA smoothing → one-liner

**Android:** Implement extraction with MediaPipe.
- MediaPipe Hands gives 21 landmarks per hand (same as Apple Vision)
- MediaPipe Pose gives 33 body landmarks — extract shoulders (11, 12) and elbows (13, 14)
- Apply same normalization as iOS
- Same Kalman + EMA + palm scale

**Critical:** Validate extraction parity.
- Record the same 50 test videos on both iOS and Android
- Compare landmark outputs — should be within 5-10% (MediaPipe vs Apple Vision difference)
- Run Stage 1 on both — accuracy should be within 2% between platforms

### Phase 4: Model Integration (1 week)

**iOS CoreML integration:**
```swift
import CoreML

let config = MLModelConfiguration()
config.computeUnits = .all  // Neural Engine when available

let model = try Stage1(configuration: config)
let input = try MLMultiArray(shape: [1, 32, 61, 5], dataType: .float32)
// Fill input from landmark extraction...
let output = try model.prediction(landmarks: input)
let logits = output.logits  // [1, 310]
```

**Android TFLite integration:**
```kotlin
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate

val options = Interpreter.Options().apply {
    addDelegate(GpuDelegate())
}
val interpreter = Interpreter(loadModelFile("stage1.tflite"), options)

val input = Array(1) { Array(32) { Array(61) { FloatArray(5) } } }
val output = Array(1) { FloatArray(310) }
interpreter.run(input, output)
```

### Phase 5: Stage 3 Translation Strategy (1-2 weeks)

T5 is the hardest part of mobile deployment because of autoregressive generation.

**Option A — Cloud API (recommended for MVP):**
- Deploy Flan-T5-Small as a FastAPI service on a cheap VPS ($5-10/month)
- Device sends gloss sequence, receives English text
- Latency: ~200ms (including network)
- Pros: no mobile T5 complexity, easy to update/improve T5 model
- Cons: requires internet

**Option B — On-device with manual generation loop:**
- Convert T5 encoder and decoder separately to CoreML/TFLite
- Write native generation loop (call decoder N times, track KV cache)
- Greedy decoding only (beam search on mobile is painful)
- Pros: fully offline
- Cons: significant native code, slower than server (~500-800ms)

**Option C — Hybrid:**
- Try on-device first, fall back to cloud if it fails or times out
- Caches common phrases locally
- Pros: best of both worlds
- Cons: most code to maintain

**Recommendation for demo/defense:** **Option A (cloud API)**. Ship working product first. Move to on-device later if privacy or offline use becomes a requirement.

### Phase 6: App UI and Integration (2-3 weeks)

**Core screens:**
1. Camera view with skeleton overlay
2. Recording controls (start/stop)
3. Live gloss predictions (Stage 1 output)
4. Final English translation (Stage 3 output)
5. History/saved translations
6. Settings (model precision, cloud/local translation)

**UX considerations:**
- Show confidence scores (dim low-confidence predictions)
- Let user correct predictions manually (for data collection)
- Offline mode indicator
- Onboarding that teaches framing (signer fills frame, good lighting, etc.)

### Phase 7: Testing and Polish (2 weeks)

**Test matrix:**
- 5 iOS devices (iPhone 12, 13, 14, 15, 16)
- 5 Android devices (Pixel 6/7/8, Samsung S22/S23, mid-range Xiaomi)
- 3 lighting conditions (bright, dim, harsh)
- 2 signers (user + one other person)
- 50 signs + 9 phrases each

**Metrics to track:**
- Extraction quality (detection rate per device)
- Stage 1 top-1 accuracy per device
- Stage 2 WER per device
- End-to-end latency (camera to English text)
- App size, memory usage, battery drain
- Crash-free rate

---

## 4. File Reference Guide

For new deployment engineer/session, read these in order:

### Essential understanding (read first)
1. **`CLAUDE.md`** (project root) — Architecture overview, inviolable constraints, file map
2. **`md files/PAPER_REVIEW.md`** — Detailed technical pipeline breakdown, all channels, all stages
3. **`md files/PIPELINE_TECHNICAL_DISSECTION.md`** — Deep dive on how the model processes data

### Model architecture (to export correctly)
4. **`src_v16/model_v16.py`** — Squeezeformer, Stage 1, Stage 2 CTC definitions
   - `SLTStage1V16` class (line 379)
   - `SLTStage2V16CTC` class (line 494)
   - `SqueezeformerBlock` (line 41)
   - `MultiScaleTCNV16` (line 441)
5. **`src_v16/train_stage_1_v16.py`** — Training config, EMA, data loading format
6. **`src_v16/train_stage_2_v16.py`** — CTC training, sequence handling

### Extraction pipeline (to replicate on mobile)
7. **`src_v16/extract_v16.py`** — Apple Vision extraction, Kalman, normalization, palm scale
   - `extract_frames_v16()` (line 501) — main extraction function
   - `normalize_sequence()` — landmark normalization
   - `compute_palm_scale()` — palm scale feature
8. **`scripts/extract_apple_vision.py`** — Reference for Apple Vision API usage patterns

### Existing inference examples
9. **`src/camera_inference.py`** — Live webcam inference (Python/Mac) — shows full pipeline logic
10. **`docker/run_inference.py`** — .npy → Stage 1/2/3 inference — shows model loading and decoding
11. **`src/demo_classify.py`** — Desktop demo app — UI flow reference

### Configuration and data
12. **`models/manifest_v16.json`** — 310 class names → index mapping
13. **`output_v16_d384/best_model.pth`** — Stage 1 checkpoint (96.00% test)
14. **`output_stage2_v16/stage2_best_model.pth`** — Stage 2 checkpoint (5.07% WER)
15. **`weights/slt_final_t5_model/`** — Stage 3 T5 checkpoint

### Reviews worth reading
16. **`md files/Senior_ML_Lead_System_Review.md`** — Architecture critique, things to watch for
17. **`md files/STAGE1_TRAINING_REVIEW.md`** — Training internals
18. **`md files/STAGE2_CTC_REVIEW_R2.md`** — Stage 2 specifics
19. **`md files/OPTIMIZATION_AND_MOBILE_PLAN.md`** — Earlier mobile planning (some outdated)

### Memory system (cross-session context)
20. **`.claude/projects/-Users-frnzlo-Documents-machine-learning-SLT/memory/`**
    - `project_v16_handoff.md` — Complete project state, all 9 runs
    - `project_mobile_deployment.md` — Earlier mobile notes
    - `feedback_extraction_consistency.md` — Critical: extraction must match between training and inference
    - `feedback_encoder_mismatch.md` — Critical: encoder must match Stage 1 exactly

---

## 5. Known Risks and Mitigations

### 5.1 Extraction Format Mismatch
**Risk:** Mobile extraction (especially MediaPipe on Android) produces slightly different landmark values than Apple Vision. Model trained on AV may not generalize to MediaPipe input.

**Mitigation:**
- Validate landmark parity on same test videos before full integration
- If gap is significant, retrain Stage 1 with **combined** AV + MediaPipe extracted data
- Document agreement rate (~84.9% based on prior testing)

### 5.2 Stage 2 Variable-Length Input
**Risk:** TFLite prefers fixed shapes. Variable-length CTC sequences are hard.

**Mitigation:**
- Fix max sequence length to 320 frames (10 clips × 32 frames = ~10 seconds)
- Pad shorter inputs with zeros + mask channel
- For longer inputs, split into overlapping 10-second windows, merge predictions

### 5.3 T5 Generation Loop
**Risk:** Autoregressive decoding can't be exported as a single graph.

**Mitigation:**
- Use cloud API for MVP (simpler)
- Or: export encoder + decoder separately, write native generation loop
- `optimum.onnxruntime` has helpers for this

### 5.4 Model Size
**Risk:** App size bloats past 200MB, App Store rejects or users uninstall.

**Mitigation:**
- FP16 quantization for all models (halves size)
- Download models on first launch (keeps initial app size small)
- Offer INT8 as optional "lite" mode

### 5.5 Inference Latency
**Risk:** End-to-end latency >1s makes app feel slow.

**Budget (iOS Neural Engine):**
- Camera frame: 33ms (30fps)
- Extraction: 10ms
- Stage 1: 8ms
- Stage 2: 30ms (on sequence)
- Stage 3: 50ms (local) or 200ms (cloud)
- Total: ~100-300ms (acceptable)

**If over budget:**
- Profile each stage, find bottleneck
- Consider skipping Stage 2 for single-sign interactions (use Stage 1 directly)
- Reduce Stage 3 beam search width

### 5.6 Platform Parity
**Risk:** iOS works great, Android lags behind due to MediaPipe vs Apple Vision.

**Mitigation:**
- Test Android first on mid-range device (not just flagships)
- If Android quality is consistently worse, ship iOS first
- Document known limitations per platform

---

## 6. Success Criteria

**Minimum viable product:**
- Works on iPhone 13+ and Pixel 7+
- End-to-end latency <500ms
- Stage 1 accuracy on device within 3% of PyTorch test accuracy
- App size <250MB
- Crash-free rate >99% in testing

**Stretch goals:**
- Works on all iPhones from iPhone 12 onwards
- Works on Android 11+ with 4GB+ RAM
- Full offline mode (no cloud dependency)
- Real-time continuous signing (not just turn-based)
- Multi-language output (not just English)

---

## 7. Timeline Summary

| Phase | Duration | Blocking for next phase? |
|---|---|---|
| 1. ONNX export + validation | 3-5 days | Yes |
| 2. Platform conversion | 1-2 weeks | Yes |
| 3. Native extraction pipelines | 1-2 weeks | No (parallel with 2) |
| 4. Model integration | 1 week | Requires 2 and 3 |
| 5. Stage 3 strategy (cloud API) | 1-2 weeks | No (parallel with 4) |
| 6. App UI + integration | 2-3 weeks | Requires 4 and 5 |
| 7. Testing and polish | 2 weeks | Final step |
| **Total** | **8-12 weeks** | |

Parallelize where possible. Realistic timeline with one dev: 10-12 weeks. With two devs (iOS and Android split): 6-8 weeks.

---

## 8. What NOT To Do

1. **Don't rewrite in TensorFlow** — ONNX conversion works. Rewriting is weeks of risk.
2. **Don't retrain with different extraction** — the Apple Vision models are the best we have. MediaPipe for Android only.
3. **Don't try to optimize before measuring** — profile first, optimize bottlenecks.
4. **Don't ship without on-device testing** — simulator performance is not real-device performance.
5. **Don't skip extraction parity validation** — this is the #1 silent failure mode.
6. **Don't use React Native if you want best performance** — native Swift/Kotlin is worth it for ML apps.
7. **Don't quantize without validating** — FP16 is usually safe, INT8 needs per-model validation.
8. **Don't forget the v16 architecture details** — read `src_v16/model_v16.py` carefully. Wrong d_model or wrong channels = silent garbage output.

---

## 9. Open Questions To Resolve Before Starting

1. **Framework:** Native vs cross-platform? (Recommendation: native Swift + Kotlin)
2. **T5 deployment:** Cloud vs on-device? (Recommendation: cloud for MVP)
3. **Privacy stance:** Is cloud API acceptable, or must everything be on-device?
4. **Model updates:** How will new model versions be distributed? (Bundled with app update vs downloaded at runtime)
5. **Data collection:** Does the app phone home with user-recorded videos for future training? (Consider privacy implications)
6. **Monetization:** Free, paid, or freemium? (Affects hosting budget for cloud API)
7. **Target users:** General public, specific deaf communities, university students? (Affects which signs to prioritize)

These should be answered before Phase 1 starts.
