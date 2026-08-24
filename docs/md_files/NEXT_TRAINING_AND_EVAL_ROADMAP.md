# Next training cycle: experiments, metrics, and borrowable methods

This document proposes **what to test next**, **what to log and decide from each experiment**, and **which external ideas apply** to the SLT pipeline. It assumes Stage 1 (or v16) has reached roughly **~96% Top-1** on a held split while **generalization**, **efficiency**, and **fingerspelling** remain weak.

---

## 1. Diagnose the gap: accuracy vs. the three pain points

High accuracy on a stratified random split can coexist with:

1. **Generalization** — models exploit signer, studio, camera geometry, or phrase templates rather than sign content.
2. **Efficiency** — latency, memory, and power on device; training cost; export size (TFLite/Core ML).
3. **Fingerspelling** — your pipeline is optimized for **lexical glosses** (Stage 1/2) and **English** (Stage 3), not **letter-by-letter** spelling. Fingerspelling needs fine hand shape and fast temporal dynamics; a gloss classifier may “know” the word *HELLO* without ever modeling rapid letter transitions.

**Rule:** treat **96%** as a **reference metric on one protocol**, not proof that the system works for new signers, new environments, or spelled names.

---

## 2. What to measure (before changing architecture)

Define a **fixed evaluation protocol** so experiments are comparable.

### 2.1 Splits that stress generalization

| Protocol | Construction | What it tests |
|----------|--------------|----------------|
| **Signer-held-out** | Group samples by person/source ID; leave one or more signers entirely out of train | Signer independence |
| **Environment-held-out** | Split by background, lighting bucket, or device (phone vs webcam) | Robustness |
| **Rare-class tail** | Per-class counts; hold out or overweight classes with &lt; N samples | Tail behavior |
| **Temporal compositional** | Phrases not seen as wholes in training | Template memorization vs. composition |

**Deliverable from each training run:** JSON or CSV with **per-split** Top-1, Top-5, per-class recall, and **confusion slices** (e.g. same handshape family).

### 2.2 Bucketed error analysis (mandatory)

For every eval, tag failures into buckets:

- **Coarticulation / boundary** — errors at sign boundaries in continuous video.
- **Minimal pair** — confusions between signs that differ by one phonological feature.
- **Fingerspelling** — spelled sequences vs fingerspelling-capable baseline (see §6).
- **Short / long sequence** — performance vs. clip length or phrase length.
- **Low landmark quality** — samples where mask confidence or detection rate is low.

**Goal:** move from “we need better generalization” to “40% of errors are signer X” or “most errors are minimal pairs among location variants.”

### 2.3 Efficiency metrics (same checkpoint, multiple configs)

Log **end-to-end** numbers, not model FLOPs alone:

| Metric | Notes |
|--------|--------|
| **Latency p50 / p95** | Full path: capture → pose → Stage 1/2 → (Stage 3) on target hardware |
| **Memory peak** | Especially mobile |
| **Model size** | FP16 vs INT8; number of separate graphs if split |
| **Wake frames / sustained FPS** | Thermal throttling on phones |

### 2.4 Statistical hygiene

- **Multiple seeds** (at least 2–3) for any claim “+X%” on generalization splits.
- **Confidence intervals** on signer-held-out metrics when fold count is small.

---

## 3. Experiments: generalization

### 3.1 Data and sampling

- **Signer-balanced batches** or **stratified sampling** so each mini-batch is not dominated by a few signers.
- **Hard-negative mining** — oversample minimal pairs (same handshape, different movement).
- **Cross-environment augmentation** already partially present (mirror, scale, rotation, temporal jitter in `src_v16/train_stage_1_v16.py`); extend with **CutMix along time** (Kaggle fingerspelling winners mixed **within signer**), **structured finger dropout** over time windows, and **temporal masking** (zero contiguous time spans).
- **Explicit geometry** — optional `use_pairwise` / `use_angles` in `src_v16/model_v16.py` for signer-invariant signals; ablate on signer-held-out only.

### 3.2 Training objectives

- **Strong regularization** — weight decay, dropout, DropPath (`drop_path` in Squeezeformer blocks), **EMA** (already used in v16 training patterns).
- **Self-distillation or auxiliary consistency** — temporal jitter should predict same label (if not already).
- **Domain randomization** — stronger noise on XY/Z, occasional drop of **face** or **body** nodes (analog to “face/pose dropout” in competition writeups).

### 3.3 Stage 2-specific (continuous signing)

- Continue **`train_stage_2_v16_fixed.py` ideas**: partial phrases, isolated clips as 1-sign sequences, heavier synthetic mix — all aimed at **breaking phrase templates** and teaching short outputs.
- Measure **WER** on **real continuous** data separately from synthetic val.

### 3.4 What to extract from each generalization run

- **Signer-held-out accuracy** (primary).
- **Gap** vs. random split (train same architecture).
- **Calibration:** reliability diagram or expected calibration error (optional but useful for deployment gating).
- **Saved confusion matrices** per held-out signer.

---

## 4. Experiments: efficiency

### 4.1 Model-side

- **Depth vs width** — your v16 notes favor **d_model=256** in quick benchmarks; re-verify on **target export** (INT8, Core ML, TFLite).
- **Attention variant** — Kaggle 1st place replaced relative positional encodings with **RoPE** and **cached** rotary tables for large speedups in training and TFLite. Your stack uses **learnable absolute position embeddings** (`pos_enc` in `SqueezeformerEncoder`); worth an ablation if ONNX/TFLite latency dominates.
- **Encoder sharing** — Stage 2 already shares/freeze-then-unfreeze encoder weights; log **delta** in WER when unfreezing.

### 4.2 Graph and runtime

- **Operator fusion** (Core ML / TFLite delegates).
- **Split models** — clip encoder vs sequence head (you already have multi-stage exports under `mobile_export/`).
- **Variable-length inference** — competition teams masked padding during training and ran **no pad** at inference; audit whether padding teaches spurious patterns in Stage 2.

### 4.3 What to extract from each efficiency run

- **Latency p50/p95** on device for the **exported** artifact.
- **Accuracy drop** vs FP32 baseline on the **same** eval protocol.
- **Size** (MB) and **whether batch=1** matches training assumptions.

---

## 5. Fingerspelling: treat as a sub-problem (recommended)

Gloss accuracy **does not** imply fingerspelling capability. Plan a **parallel track**.

### 5.1 Why gloss models fail at spelling

- **Class vocabulary** — letters and spellings are not in the 310-class gloss set as fine-grained temporal classes.
- **Temporal resolution** — letters transition faster than many lexical signs; **32-frame clips** and heavy temporal pooling may erase detail.
- **CTC vs autoregressive** — ASR-style **encoder–decoder with CE** often beats CTC on spelling benchmarks (as in the Google Kaggle writeup); your Stage 2 is **CTC-oriented** for gloss sequences.

### 5.2 Practical options (incremental → heavy)

| Option | Effort | Idea |
|--------|--------|------|
| **Auxiliary head** | Low | Small classifier for **fingerspelling vs lexical sign** + optional **letter** classes on segments labeled as fingerspelling |
| **Finer temporal encoder** | Medium | Shorter clips, less pooling, or parallel high-resolution branch for hand ROI features |
| **Dedicated spelling module** | High | Separate small model (pose → characters) only activated when gate fires |
| **Data** | Critical | Curated **fingerspelling-only** clips (names, rare words); consider public resources (e.g. large-scale fingerspelling corpora — see §7) |

### 5.3 Metrics for spelling

- **Character Error Rate (CER)** or **normalized edit distance** on spelled content.
- **Segmentation** — start/end of spelling vs lexical signs in continuous video (even manual labels on a small set help).

---

## 6. Borrowable ideas (mapped to your repo)

### 6.1 From Kaggle ASL Fingerspelling 1st place (Henkel / team)

| Idea | Fit |
|------|-----|
| **CutMix along time** (within signer) | Stage 1 / Stage 2 augmentation |
| **FingerDropout** (zeros on finger joints over windows) | Your joint dropout is random; **structured** finger masking may help spelling-like discrimination |
| **Decoder-input masking** | Only if you add an autoregressive decoder path |
| **RoPE + cached embeddings** | Possible swap in `SqueezeformerBlock` attention path for speed |
| **Confidence head** + low-confidence fallback | Postprocessing for “garbage frame” detection in camera pipeline |
| **Padding masks through encoder** | Stage 2 training when T is padded |

### 6.2 From your existing v16 design

- **Apple Vision extraction** (`src_v16/extract_v16.py`) — keep improving **Z**, **Kalman**, **palm_scale**; log failure cases when hands are lost.
- **Squeezeformer encoder** — strong baseline; improvements are mostly **data**, **augmentation**, **eval protocol**, and **spelling side-channel**.

---

## 7. Recent technologies and datasets (2024–2026)

Use these as **bibliography and inspiration**, not as drop-in code.

### 7.1 Architectures and papers

- **Stack Transformer / spatial-temporal attention for fingerspelling** — hierarchical attention over joints and time (e.g. SSTAN-style ideas). arXiv: [2503.16855](https://arxiv.org/abs/2503.16855).
- **Efficient Squeezeformer variants** — e.g. Fastformer-style attention reported to reduce training time and parameters while keeping competitive NLD on fingerspelling tasks (Springer / Institution of Engineers India, 2025 — search title for exact citation).
- **Pose-based transformers for fingerspelling** — “Fingerspelling PoseNet” line (WACV workshop): pose-centric encoder–decoder for **spelling** benchmarks (e.g. ChicagoFSWild family).
- **FSboard** — very large **smartphone-collected** fingerspelling corpus (millions of characters, many signers); useful for **pretraining** or benchmarking letter models. arXiv: [2407.15806](https://arxiv.org/abs/2407.15806).
- **Surveys** — “Comprehensive survey on recent advances and challenges in sign language recognition” (*Discover Artificial Intelligence*, 2025) for **continuous SLR** and deployment gaps.

### 7.2 Themes across recent SLR

- **Signer-independent** evaluation is standard in papers but often missing in internal dashboards — align your splits with paper protocols where possible.
- **Multi-modal** (pose + RGB) wins many benchmarks but conflicts with **efficiency** goals; pose-only remains attractive for mobile if robust enough.

---

## 8. Suggested phased test matrix

### Phase A — Measurement only (1–2 weeks)

- Build **signer-held-out** (or proxy) eval.
- Run **bucketed error analysis** on current best checkpoint.
- Record **latency/size** for current mobile exports.

**Decision gate:** quantify generalization gap and spelling gap before large retrains.

### Phase B — Low-risk training changes

- Augmentation: **CutMix-time**, **finger-structured dropout**, stronger **temporal masking**.
- Stage 1: toggle **`use_angles` / `use_pairwise`** ablation on signer-held-out.
- Stage 2: confirm **`train_stage_2_v16_fixed.py`** strategies are enabled for real-phrase memorization issues.

### Phase C — Efficiency

- RoPE or lighter attention ablation **if** profiling shows attention as bottleneck.
- INT8 / quantization sensitivity table vs signer-held-out metric.

### Phase D — Fingerspelling track

- Small labeled set + **CER** metric.
- Optional auxiliary head or small parallel model; consider **FSboard** or similar for methodology comparison only (license / domain match required).

---

## 9. Artifacts to save per experiment

For reproducibility and decisions:

- Config JSON (all CLI flags, seeds, manifest paths).
- Checkpoint path + **eval JSON** (random split + signer-held-out + spelling subset if any).
- Confusion matrix + **per-class recall** CSV.
- Export bundle version (TFLite/Core ML) + **latency log**.
- Short **run summary** (5 bullets: what moved, what did not).

---

## 10. Summary priorities

1. **Generalization:** signer-grouped CV + bucketed errors + augmentations that mimic competition winners (CutMix-time, structured finger dropout).
2. **Efficiency:** device latency first; consider RoPE/caching and padding masks if exports are slow or pads leak signal.
3. **Fingerspelling:** separate metric (CER), data, and possibly a **small dedicated pathway**; do not expect Stage 1 gloss accuracy to transfer without explicit modeling.

This roadmap is intentionally broad so you can trim phases based on the first **Phase A** numbers.
