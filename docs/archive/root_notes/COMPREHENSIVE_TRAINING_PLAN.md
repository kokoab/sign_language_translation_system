# SLT Comprehensive Training Plan: Stage 1 + 2 + 3

## Research Basis: 80+ Papers (2020-2026)

---

## Executive Summary

| Stage | Current | Proposed | Expected Result |
|-------|---------|----------|-----------------|
| Stage 1 | Single-stream DS-GCN+Transformer, 84.99% | 6-stream ensemble + Balanced Softmax + ArcFace | 90-93% Top-1 |
| Stage 2 | Frozen encoder + BiLSTM + CTC, synthetic data | Transformer encoder + Joint CTC/Attention + CR-CTC + InterCTC | 25-35% WER (skeleton-only realistic range) |
| Stage 3 | Flan-T5-Base, 17K synthetic pairs, no BLEU metric | T5-small (mobile) or LoRA on larger LLM + noisy gloss augmentation | BLEU-4 25-30 |
| Deploy | Desktop only | Mobile (iOS/Android) via ExecuTorch/CoreML, <1s end-to-end | <500MB total |

**Total training time: ~4-6 hours on 2x RTX 4080 (~$1.50-2.00)**

---

## STAGE 1: Multi-Stream Ensemble (Already Implemented)

### Architecture
6 streams trained in parallel, ensemble at inference:

| Stream | Input | Model | d_model | Epochs | GPU |
|--------|-------|-------|---------|--------|-----|
| S1 Joint | all 16ch | DS-GCN+Transformer | 384 | 200 | 0 |
| S2 Bone | bone+mask (4ch) | DS-GCN+Transformer + ArcFace | 256 | 150 | 1 |
| S3 Velocity | vel+mask (4ch) | DS-GCN+Transformer + ArcFace | 256 | 120 | 1 |
| S4 Bone Motion | bone_motion+mask (4ch) | DS-GCN+Transformer | 192 | 100 | 0 |
| S5 Angle | 42 geo angles + mask (43) | Transformer-only | 192 | 100 | 1 |
| S6 Angle Motion | temporal diff of S5 | Transformer-only | 192 | 80 | 1 |

### Key Improvements (already coded)
- Balanced Softmax Loss (Ren et al. 2020) — fixes class balance without degrading frequent classes
- ArcFace Head (Deng et al. CVPR 2019) — forces angular separation of confused pairs
- OneCycleLR for auxiliary streams — 30-50% fewer epochs to converge (Smith & Topin 2018)
- TF32 Tensor Cores enabled — 5-15% free speedup
- Fused AdamW — 5-10% faster optimizer step
- torch.compile(mode="reduce-overhead") — CUDA graphs for static shapes, 10-30% speedup
- GPU dataset preloading — eliminates CPU-to-GPU transfer
- All .item() GPU sync bottlenecks eliminated

### Training: ~2 hours on 2x RTX 4080

### Expected: 90-93% Top-1 (ensemble), up from 84.99% (single stream)

---

## STAGE 2: Continuous Sign Recognition

### Current Architecture
```
Frozen DS-GCN Encoder --> MultiScaleTCN (32-->4 tokens) --> BiLSTM (2-layer, 512) --> CTC
```
- Synthetic data (concatenated isolated signs, 35% transition injection)
- Greedy CTC decoding during training
- 100 epochs, batch=32, lr=1e-3

### Problems Identified

| Problem | Impact | Source |
|---------|--------|--------|
| BiLSTM is weaker than Transformer for temporal modeling | Limits sequence accuracy | CorrNet+ 2024, Uni-Sign 2025 |
| CTC spike problem: model over-predicts blanks | Under-utilizes visual features | CR-CTC (ICLR 2025) |
| No intermediate supervision | Feature extractor saturates early | InterCTC (ICASSP 2021) |
| Only greedy decoding during training | Trains on suboptimal assumptions | MWER (Interspeech 2021) |
| Encoder frozen forever | Can't adapt to continuous signing patterns | Transfer learning best practice |
| Transitions only 35% of the time | 65% of sequences unrealistic | Domain gap analysis |
| Fixed 4 tokens per clip | Inflexible compression | Swin-MSTP 2024 |
| No CTC blank penalty | Blank dominates | Focal CTC (Feng 2019) |
| Stage 3 never sees noisy gloss input | Fragile to Stage 2 errors | Pipeline analysis |

### Proposed Architecture
```
DS-GCN Encoder (from Stage 1 joint stream)
    |
    |--> [Frozen epochs 1-30, fine-tune epochs 31+ at 0.1x LR]
    |
MultiScaleTCN (32 --> 4 tokens, existing)
    |
Transformer Encoder (4-6 layers, d_model=256, nhead=8, Pre-LayerNorm)
    |
    |--> CTC Head (Linear --> vocab)           [monotonic alignment]
    |--> Attention Decoder (cross-attn to enc)  [output dependencies]
    |
Loss = 0.7 * CTC + 0.3 * Attention + 0.1 * InterCTC (at layer 2)
```

### Changes (ranked by impact)

#### 2.1 Replace BiLSTM with Transformer Encoder
- **Why:** Bidirectional by design, better long-range dependencies, Flash Attention, compatible with torch.compile
- **Config:** 4 layers, d_model=256, nhead=8, Pre-LayerNorm (Xiong et al. 2020), dim_ff=1024
- **Research:** CorrNet+ (2024), Swin-MSTP (Neurocomputing 2024)
- **Expected:** Better temporal modeling, ~5-10% WER improvement

#### 2.2 CR-CTC: Consistency Regularization
- **What:** Feed two differently-augmented views through the model, enforce consistency via KL-divergence
- **Why:** Addresses CTC spike problem — model over-predicts blanks, under-utilizes features
- **Research:** Yao, Kang et al. ICLR 2025 — SOTA on LibriSpeech, Aishell-1, GigaSpeech
- **Implementation:** Second augmented forward pass + KL loss term
- **Expected:** 10-20% relative WER reduction

#### 2.3 Intermediate CTC Loss (InterCTC)
- **What:** Attach auxiliary CTC loss at intermediate Transformer layer (e.g., layer 2 of 4)
- **Why:** Forces intermediate representations to be directly predictive, prevents feature saturation
- **Research:** Lee & Watanabe, ICASSP 2021 — 10-20% relative WER reduction, <5% training overhead
- **Implementation:** Linear projection head at layer 2, one extra loss term
- **Expected:** 10-20% relative WER reduction, compounds with CR-CTC

#### 2.4 Joint CTC + Attention Loss
- **What:** Add a small attention decoder alongside CTC head
- **Why:** CTC handles monotonic alignment (sign order), attention handles output dependencies (verb agreement)
- **Research:** "Improvement in SLT Using Text CTC Alignment" (COLING 2025), Hybrid CTC/Attention (Watanabe et al.)
- **Loss:** alpha * CTC + (1-alpha) * Attention, alpha=0.7
- **Expected:** Better handling of non-monotonic cases

#### 2.5 Focal CTC Loss
- **What:** Apply focal weighting (1-p)^gamma to CTC loss per frame
- **Why:** Down-weights easy/blank-dominated frames, focuses on hard information-carrying frames
- **Research:** Feng et al. 2019, adapted from focal loss (Lin et al. 2017)
- **Expected:** Reduces blank dominance, improves label frame utilization

#### 2.6 Encoder Unfreezing Schedule
- **What:** Epochs 1-30: frozen. Epochs 31+: unfreeze with 10x lower LR
- **Why:** Lets new layers learn first, then adapts encoder to continuous signing patterns
- **Research:** Standard transfer learning practice, validated in TwoStream-SLR (2022)
- **Expected:** +1-3% WER improvement

#### 2.7 Synthetic Data Improvements

| Change | Why | Research |
|--------|-----|---------|
| Always inject transitions (100%, not 35%) | Real signing always has transitions | Domain gap analysis |
| Variable transition length (2-8 frames) | Real transitions vary | Sign-D2C (2024) |
| Add anticipatory motion: last 3 frames of sign N blend toward sign N+1 | Models coarticulation | Phonology of continuous signing |
| Add random hold frames (2-5 frames, 20% prob) | Signers pause within signing | Observational |
| Add idle segments (hands resting, 10% prob) | Real conversations have pauses | Observational |
| Curriculum: start with 2-sign, gradually increase to 8 | Prevents early overfitting to noise | SignAvatar (2024) |
| Speed perturbation (0.9x, 1.0x, 1.1x) | Signer speed variation | ASR standard practice |
| Temporal drop (15% of frames randomly) | Simulates tracking failures | Min et al. ICCV 2025 Workshop (1st place SignEval) |
| Jittering (noise=0.01) | Simulates extraction noise | Min et al. ICCV 2025 Workshop |

#### 2.8 Training Recipe

| Parameter | Current | Proposed | Why |
|-----------|---------|----------|-----|
| Sequence model | BiLSTM (2-layer, 512) | Transformer (4-layer, 256) | Better temporal modeling |
| Loss | CTC only | Joint CTC + Attention + InterCTC + CR-CTC | Multi-objective, addresses spike problem |
| Epochs | 100 | 150 | More time for unfreezing phase |
| Batch size | 32 | Dynamic (bucket batching, max_tokens=4096) | 20-40% throughput gain (SpeechBrain) |
| LR | 1e-3 | 5e-4 (encoder 5e-5 after epoch 30) | Lower for Transformer stability |
| Scheduler | Cosine warmup | Noam (standard for Transformer+CTC) | Proven in ASR |
| Patience | 25 | 35 | Need more time with unfreezing |
| Transition prob | 0.35 | 1.0 (always) | Realistic data |
| Speed perturbation | None | 0.9x, 1.0x, 1.1x (3x data) | Standard in ASR |
| Regularization | Dropout 0.3 | + R-Drop + stochastic depth | R-Drop (NeurIPS 2021) |
| Decoding | Greedy only | + beam search eval (beam=10) | Better WER measurement |

### Post-Training
- Checkpoint averaging: top-5 by WER (standard in ASR competitions)
- Test-time augmentation: speed perturbation + flip, average CTC log-probs

### Training Time: ~1.5-2 hours on 1x RTX 4080

---

## STAGE 3: Sign Language Translation

### Current Architecture
```
Flan-T5-Base (250M params) fine-tuned on 17K synthetic gloss-->English pairs
- 10 epochs, patience=3
- No BLEU/ROUGE metrics
- ~5.3% question coverage
```

### Problems Identified

| Problem | Impact | Source |
|---------|--------|--------|
| Flan-T5 underperforms LLMs for SLT | Lower translation quality | SignLLM (CVPR 2024): T5 << LLaMA |
| Only 17K training pairs | Limited diversity | Gloss2Text (EMNLP 2024): 22K with paraphrases |
| No BLEU/ROUGE evaluation | Can't measure quality | Standard SLT practice |
| 5.3% question coverage (should be 15-20%) | Poor question generation | Dataset analysis |
| Stage 3 never sees noisy input | Fragile to Stage 2 errors | Pipeline analysis |
| 10 epochs with patience=3 | Likely underfitting | Too aggressive early stopping |
| No back-translation augmentation | Missing standard technique | SignBT (CVPR 2021) |
| 250M params too large for mobile | Deployment blocker | Mobile research |

### Two-Track Strategy

#### Track A: Mobile-Optimized (T5-small, 60M params)
For deployment on phones:
- T5-small fine-tuned specifically for gloss-to-English
- INT8 quantized: ~30-60 MB on disk
- Inference: ~100-500ms for short sequences on mobile
- Fits alongside entire pipeline in <500 MB RAM

#### Track B: Maximum Quality (LoRA on NLLB-200 or LLaMA)
For server-side or powerful devices:
- NLLB-200 (3.3B) with LoRA (rank=16, alpha=32) — only 26M trainable params
- BLEU-4 28.20 on PHOENIX-2014T (Gloss2Text, EMNLP 2024)
- Semantically Aware Label Smoothing (SALS)
- Can distill into T5-small for mobile

### Changes (ranked by impact)

#### 3.1 Noisy Gloss Augmentation (CRITICAL)
- **What:** Augment training data with simulated Stage 2 errors
- **How:** For each gloss sequence, with 30% probability:
  - Delete a random gloss (simulates missed sign)
  - Insert a random gloss (simulates hallucinated sign)
  - Swap a gloss with one from the confusion matrix (simulates misrecognition)
  - Keep the target English unchanged
- **Why:** Stage 3 trains on perfect glosses but receives noisy CTC output at inference. This gap causes catastrophic quality drops.
- **Expected:** Major robustness improvement, prevents pipeline error cascade

#### 3.2 Expand Dataset with LLM-Generated Paraphrases
- **What:** Use Claude/GPT to generate 5-10 paraphrases per existing gloss-text pair
- **How:** API call: "Given ASL gloss 'HELLO HOW YOU' and translation 'Hello, how are you?', generate 5 natural alternative translations"
- **Cost:** ~$1-2 for 17K pairs x 5 paraphrases = 85K API calls
- **Expected:** 17K --> 85K+ training pairs, much more linguistic diversity
- **Research:** Gloss2Text (EMNLP 2024) used paraphrases + back-translation to reach 22K pairs

#### 3.3 Fix Question Coverage
- **What:** Regenerate dataset with 15-20% questions (currently 5.3%)
- **How:** Add more WH-question templates to generate_stage3_data_v2.py
- **Templates:** WHO, WHAT, WHERE, WHEN, WHY, HOW, HOW-MANY, YES/NO questions
- **Expected:** Better question generation at inference

#### 3.4 Add BLEU/ROUGE/BERTScore Evaluation
- **What:** Add proper translation metrics to training
- **How:** Use sacrebleu for BLEU, rouge-score for ROUGE, bert-score for semantic similarity
- **Best metric for SLT:** BLEU-1/4 for benchmark comparison, COMET or BERTScore for development
- **Research:** Metrics comparison in gloss-free SLT survey (2026)

#### 3.5 Longer Training
- **What:** Increase from 10 to 20-30 epochs, patience 3 to 7
- **Why:** Early stopping at 3 epochs is too aggressive for fine-tuning
- **Expected:** Better convergence, especially with augmented data

#### 3.6 Semantically Aware Label Smoothing (SALS)
- **What:** Instead of uniform label smoothing, distribute smoothed probability to semantically similar tokens
- **Research:** Gloss2Text (EMNLP 2024) — cosine similarity threshold 0.6, beta 0.1
- **Expected:** +3-7% BLEU improvement

### Training Recipe

| Parameter | Current | Track A (Mobile) | Track B (Quality) |
|-----------|---------|-------------------|-------------------|
| Model | Flan-T5-Base (250M) | T5-small (60M) | NLLB-200 (3.3B) + LoRA |
| Trainable params | 250M | 60M | 26M (LoRA) |
| Dataset | 17K pairs | 85K+ (augmented) | 85K+ (augmented) |
| Epochs | 10 | 30 | 60 |
| Batch size | 32 (eff 64) | 64 | 32 |
| LR | 2e-4 | 3e-4 | 2e-4 |
| Noisy gloss aug | No | 30% | 30% |
| Label smoothing | 0.1 (uniform) | 0.1 (SALS) | 0.1 (SALS) |
| Evaluation | Loss only | BLEU + ROUGE + BERTScore | BLEU + ROUGE + BERTScore |
| Quantization | None | INT8 PTQ | None (server) |

### Training Time:
- Track A (T5-small): ~30-45 min on 1x RTX 4080
- Track B (LoRA): ~1-2 hours on 1x RTX 4080

---

## MOBILE DEPLOYMENT PLAN

### Target: iOS and Android, <1 second end-to-end latency

### Pipeline on Device
```
Camera (30fps)
    |
MediaPipe Hands + Face Mesh (native, ~8-15ms/frame)
    |
Landmark Buffer (ring buffer, last 32-96 frames)
    |
Stage 1: Distilled single-model classifier (from 6-stream ensemble)
    - Knowledge distillation: 6-stream teacher --> single student
    - CRD (Contrastive Representation Distillation, ICLR 2020)
    - INT8 quantized via Degree-Quant (QAT)
    - ~20-50 MB, ~10-30ms per window
    |
Stage 2: Pruned Transformer + CTC
    - Structured pruning (2x FLOPs reduction)
    - INT8 quantized
    - ~10-30 MB, ~20-50ms per sequence
    |
Stage 3: T5-small (INT8)
    - Fine-tuned for gloss-to-English
    - INT8 post-training quantization
    - ~30-60 MB, ~100-500ms per translation
    |
Display translation
```

### Memory Budget (targeting 6-8 GB phone)

| Component | RAM | Disk |
|-----------|-----|------|
| MediaPipe models | ~50-100 MB | ~30 MB |
| Stage 1 (distilled, INT8) | ~30-50 MB | ~20-50 MB |
| Stage 2 (pruned, INT8) | ~20-40 MB | ~10-30 MB |
| Stage 3 T5-small (INT8) | ~100-200 MB | ~30-60 MB |
| Frame buffers + landmarks | ~50-100 MB | — |
| **Total** | **~300-500 MB** | **~100-200 MB** |

### Inference Framework Options

| Framework | Best For | Recommendation |
|-----------|----------|---------------|
| ExecuTorch | Cross-platform, LLM support | Primary choice |
| CoreML | iOS only, Neural Engine | iOS-specific optimization |
| ONNX Runtime Mobile | Cross-platform fallback | If ExecuTorch has op gaps |
| MNN (Alibaba) | Fastest CNN inference | Alternative for GCN layers |

### Knowledge Distillation: 6-Stream Ensemble --> Single Model

The 6-stream ensemble is too expensive for mobile (6 models). Distill into one:

1. **Teacher:** 6-stream ensemble (weighted softmax averaging)
2. **Student:** Single SLTStage1 with d_model=256, in_channels=16
3. **Distillation method:** CRD (Contrastive Representation Distillation)
   - 57% relative improvement over other distillation methods (ICLR 2020)
   - Transfers structural relationships, not just soft labels
4. **Expected:** Student reaches ~88-90% of teacher accuracy (DKE-GCN 2024: student can even outperform teacher)
5. **Training:** ~1 hour additional

### Quantization Pipeline
```
Training (FP32) --> QAT Fine-tuning (INT8 simulation) --> Export ONNX -->
Quantize (Degree-Quant for GCN, dynamic for Transformer) -->
Convert to ExecuTorch .pte / CoreML .mlmodel
```

### Extraction on Mobile
- **Primary:** MediaPipe Hands + Face Mesh (native, 30fps)
- **RTMW is NOT feasible** on mobile for real-time whole-body
- **iOS alternative:** Apple Vision framework (21 hand landmarks, similar to MediaPipe)
- **Key difference from training:** Training uses RTMW (better quality), mobile uses MediaPipe (faster). This creates a domain gap.
- **Fix:** Fine-tune Stage 1 on MediaPipe-extracted landmarks (or distill RTMW knowledge into MediaPipe-compatible model)

### Battery and Thermal Considerations
- Continuous CNN inference triggers thermal throttling after ~2.5 minutes
- **Mitigation:** Only run recognition when hand motion detected (motion gating)
- Extraction is cheap (~5ms); recognition is expensive (~30-50ms)
- Run recognition every ~1 second, not every frame

---

## TRAINING SPEED OPTIMIZATIONS (All Stages)

### Already Implemented (Stage 1)

| Optimization | Speedup | Status |
|---|---|---|
| TF32 Tensor Cores | 5-15% | Done |
| Fused AdamW | 5-10% | Done |
| torch.compile(mode="reduce-overhead") | 10-30% | Done |
| GPU dataset preloading | 15-30% | Done |
| Batch size 256, accum 2 | 10-20% | Done |
| Eliminated .item() GPU syncs | 2-5% | Done |
| GPU-side loss accumulation | 2-5% | Done |
| OneCycleLR for aux streams | 30-50% fewer epochs | Done |

### To Apply to Stage 2

| Optimization | Expected Speedup | Effort |
|---|---|---|
| TF32 + fused AdamW | 10-20% | 2 lines |
| torch.compile on Transformer (not CTC loss) | 10-30% | Low |
| Bucket batching by sequence length | 20-40% throughput | Medium (SpeechBrain approach) |
| Mixed precision with FP32 CTC loss | 2x throughput | Low (CTC must stay FP32) |
| GPU preload (dataset fits in VRAM) | 15-30% | Low |
| Speed perturbation as data augmentation | 3x effective data, same compute | Low |

### To Apply to Stage 3

| Optimization | Expected Speedup | Effort |
|---|---|---|
| Use HuggingFace Trainer's built-in bf16/fp16 | Already enabled | — |
| Gradient checkpointing for T5 | 3x memory savings | 1 line |
| LoRA (if using large model) | 10x fewer trainable params | Low |
| torch.compile on T5 encoder | 10-20% | Low |

---

## COMPLETE TRAINING SCHEDULE

### Single Script: train_all_stages.sh

```
Phase 1: Stage 1 Multi-Stream (2h)
    GPU 0: S1 Joint (200ep) --> S4 BoneMotion (100ep)
    GPU 1: S2 Bone (150ep) --> S3 Velocity (120ep) --> S5 Angle (100ep) --> S6 AngleMotion (80ep)
    Then: Ensemble evaluation + weight optimization

Phase 2: Stage 2 CSLR (1.5-2h)
    GPU 0: Stage 2 training (Transformer+CTC, 150 epochs)
    GPU 1: Stage 3 Track A training (T5-small, 30 epochs) [PARALLEL - independent]

Phase 3: Post-Training (30 min)
    Checkpoint averaging (Stage 2)
    Knowledge distillation: ensemble --> single model (Stage 1 for mobile)
    INT8 quantization calibration

Phase 4: Evaluation + Package
    Full pipeline test (Stage 1 --> 2 --> 3)
    Per-class accuracy, WER, BLEU reports
    Package all checkpoints into tar.gz
    Auto-pause instance
```

### Wall Time: ~4-5 hours total
### Cost: ~$1.50 at $0.30/hr

---

## EXPECTED RESULTS

| Metric | Current | After Plan | Mobile |
|--------|---------|-----------|--------|
| Stage 1 Top-1 | 84.99% | 90-93% (ensemble) | 87-90% (distilled) |
| Stage 2 WER | Unknown | 25-35% | 28-38% |
| Stage 3 BLEU-4 | Unknown | 20-28 | 18-25 |
| Pipeline latency | Desktop only | <1s on phone | <1s |
| Total model size | ~500MB+ | ~500MB (server) | ~150-250 MB (mobile) |
| RAM usage | 4-6 GB | 4-6 GB | ~300-500 MB |

---

## FALLBACK PLAN

### If Stage 1 ensemble doesn't reach 90%
- Check confusion matrix -- if fingerspelling dominates, need 3D hand mesh extraction (HaMeR)
- Try SupCon pre-training on encoder
- Minimum viable: Joint + Bone (2-stream) = ~87-88%

### If Stage 2 WER is too high
- Increase transition injection quality (add more coarticulation modeling)
- Try CTC + Attention hybrid before full Transformer replacement
- Fall back to BiLSTM + CR-CTC (simpler, still beneficial)

### If Stage 3 BLEU is low
- Generate more training data with LLM API ($1-2)
- Try prompted GPT-4o/Claude API at inference (no training needed, best quality)
- Fall back to rule-based gloss-to-English as baseline

### If mobile deployment is too slow
- Reduce d_model further (192 or 128)
- Use fewer Transformer layers (2 instead of 4)
- Offload Stage 3 to server (API call)
- Use T5-small INT4 instead of INT8

---

## RESEARCH REFERENCES (80+ papers)

### Stage 1 (Sign Classification)
1. CTR-GCN (ICCV 2021) - Channel-wise topology refinement
2. InfoGCN (CVPR 2022) - Information-theoretic GCN
3. HD-GCN (ICCV 2023) - Hierarchically decomposed GCN
4. BlockGCN (CVPR 2024) - Topology awareness
5. DSTA-SLR (COLING 2024) - Dynamic spatial-temporal for SLR
6. TMS-Net (Neurocomputing 2023) - 6-stream skeleton SLR
7. SkateFormer (ECCV 2024) - Skeletal-temporal Transformer
8. SkelMamba (2024) - Mamba for skeleton recognition
9. ProtoGCN (2024) - Prototypical perspective
10. ArcFace (CVPR 2019) - Angular margin loss
11. Supervised Contrastive Learning (NeurIPS 2020)
12. Balanced Softmax (Ren et al. 2020)
13. Decoupled Training (ICLR 2020) - Representation vs classifier
14. JMDA (ACM TOMM 2024) - Joint mixing augmentation
15. Skelbumentations (WACV 2024) - Realistic skeleton augmentation
16. SkeletonMAE (ICCV 2023) - Self-supervised pre-training

### Stage 2 (Continuous Recognition)
17. CorrNet+ (IEEE TIP 2024) - SOTA CSLR
18. Uni-Sign (ICLR 2025) - Unified pre-training
19. Swin-MSTP (Neurocomputing 2024) - Multi-scale temporal
20. CR-CTC (ICLR 2025) - Consistency regularization
21. InterCTC (ICASSP 2021) - Intermediate CTC loss
22. Focal CTC (2019) - Focal loss for CTC
23. AdaMER-CTC (2024) - Adaptive entropy regularization
24. BRCTC (ICLR 2023) - Bayes risk CTC
25. MWER Training (Interspeech 2021) - Minimum WER
26. Pre-LayerNorm (2020) - Transformer training stability
27. R-Drop (NeurIPS 2021) - Regularized dropout
28. VAC (ICCV 2021) - Visual alignment constraint
29. SMKD (ICCV 2021) - Self-mutual distillation for CSLR
30. CoSign (ICCV 2023) - Co-occurrence signals
31. SignBT (CVPR 2021) - Sign back-translation
32. CTC Blank Collapse (2022) - Compressing CTC emissions
33. CTC Layer-skipping (2024) - Dynamic layer skipping
34. Hybrid CTC/Attention (Watanabe et al.) - Joint decoding
35. Text CTC Alignment (COLING 2025) - Joint CTC/Attention for SLT
36. Sign-D2C (2024) - Diffusion-based transitions
37. Min et al. (ICCV 2025 Workshop) - 1st place SignEval, skeleton CSLR
38. Flash Attention (NeurIPS 2022)
39. Scheduled DropHead (EMNLP 2020)
40. Dynamic Dropout (2024)
41. Stochastic Depth for Transformers (2019)
42. GradNorm (ICML 2018) - Multi-task loss balancing

### Stage 3 (Translation)
43. SignLLM (CVPR 2024) - LLMs for SLT
44. Sign2GPT (ICLR 2024) - Pseudo-gloss + frozen LLM
45. Spotter+GPT (2024) - Gloss detection + GPT prompting
46. Gloss2Text (EMNLP 2024) - LoRA on NLLB-200 + SALS
47. AulSign (ECAI 2025) - Few-shot LLM for SLT
48. Gloss-Free SLT Survey (2026) - Unbiased evaluation
49. SignFormer-GCN (PLOS ONE 2025) - Skeleton-based SLT
50. Cross-Modality Augmentation (EMNLP 2023)
51. Weight Tying in Transformers
52. SpeechBrain Dynamic Batching

### Mobile Deployment
53. SignNet-Nano (2025) - <20K params edge SLR
54. TinyMSLR (2026) - CNN-Transformer hybrid edge
55. MobileNetV2 Self-Attention SLR (MDPI 2025)
56. Degree-Quant (ICLR 2021) - GNN quantization
57. DKE-GCN (IEEE 2024) - Decoupled knowledge distillation
58. MK-SGN (2024) - Spiking GCN + distillation
59. LG-AKD (2024) - Lightweight GCN distillation
60. CRD (ICLR 2020) - Contrastive representation distillation
61. Part-Level KD (2024) - Skeleton-specific distillation
62. ExecuTorch (Meta) - Mobile inference framework
63. CoreML Transformers (Apple ML Research)
64. MNN (Alibaba) - Mobile neural network
65. ONNX Runtime Mobile
66. MediaPipe Hands (Google Research)
67. RTMPose (2023) - Real-time pose estimation
68. Apple Intelligence Foundation Models (2024)
69. MobiLoRA (ACL 2025) - Mobile LoRA inference
70. Picovoice Translation Models Survey (2025)

### Training Optimization
71. OneCycleLR (Smith & Topin 2018)
72. Fused AdamW (PyTorch)
73. torch.compile modes (PyTorch 2.x)
74. TF32 Tensor Cores (NVIDIA)
75. FlashAttention v2 (Dao et al.)
76. GPU Dataset Preloading
77. Gradient Checkpointing
78. Mixed Precision Training (PyTorch AMP)
79. Bucket Batching (SpeechBrain)
80. Checkpoint Averaging (ASR standard)
81. Knowledge Distillation Survey (2024)
82. INT8 Quantization Best Practices (NVIDIA)
83. Thermal Throttling on Mobile (MDPI 2020)
