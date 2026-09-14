# Stage 1 / Stage 2 architecture research

Date searched: 2026-09-14 PHT  
Question: which architecture lets isolated recognition remain strong while continuous
sequence supervision actually adapts the visual encoder?

## Answer

Use one shared frame-level visual encoder with two training heads:

1. the existing pooled 100-class isolated head for Citizen/SemLex retention; and
2. a shallow, local, multi-scale temporal CTC head over unpooled frame tokens for
   continuous recognition.

Train both heads jointly. Add encoder-level alignment/core supervision, and keep the
continuous decoder deliberately shallow so the sequence loss must improve the visual
encoder. In this design, “Stage 1” is the shared visual encoder plus its isolated
auxiliary head; “Stage 2” is an alignment/decoding head, not a second model consuming
frozen Stage-1 decisions.

CTC should remain the first alignment objective. The problem is not that CTC is
incompatible with Squeezeformer. The original CTC work was designed for unsegmented
sequence labels, and Squeezeformer itself was evaluated with CTC. The repository's
problem is the interface and training flow: Stage 1 globally pools a window into one
label, while the accepted Stage 2 freezes Stage 1 and learns a separate Transformer.
No Stage-2 gradient can repair Stage-1 visual features.

## Why this matches the measured failure

- The exact-core probe demonstrates that the encoder can learn continuous ASLLRP visual
  evidence: 85.51% top-1 on held-out ASLLRP-contiguous cores at epoch 12.
- Whole 1.07-second windows are much worse than 0.27/0.53-second windows because the
  target occupies only about one quarter of the input. A local multi-scale head can
  align short signs without declaring the entire wide window to be one class.
- `NO_EMIT` training currently trades insertions for deletions. CTC provides blank and
  sequence alignment in one objective rather than an independent pooled-window rejector.
- The existing encoder already exposes 32 frame tokens. Reusing them is the shortest
  implementation path; replacing the Apple Vision input contract or isolated classifier
  is unnecessary.
- LG remains weak with exact boundaries. Joint training is necessary but cannot invent
  missing signer/class coverage; O5S5 should be an auxiliary positive-core source, not
  the sole proof of generalization.

## Recommended architecture

```text
Apple Vision frames [T, 61, 5]
          │
          ▼
shared region-aware landmark encoder / Squeezeformer
          │ frame tokens [T, 256]
          ├──────── pooled isolated head ─────── CE on Citizen/SemLex
          │
          ▼
causal local Conv1D stage (fine scale)
          │──────── shared 101-way classifier ─ CTC auxiliary loss
          ▼
causal dilated Conv1D stage (coarse scale)
          │──────── same classifier ─────────── primary CTC loss
          ├──────── exact-core alignment loss on admitted boundaries
          └──────── blank loss only on fully annotated ASLLRP gaps
```

Keep the temporal reduction at no more than 4× initially. The median target is roughly
0.26 seconds; aggressive or fixed one-second pooling erases the useful alignment scale.
Use causal convolutions for the runtime candidate and train with the same left-context
contract used at inference. The existing revisable transcript can consume CTC prefix
hypotheses without changing the UX.

The shared classifier at fine and coarse temporal levels is useful here because it
forces both resolutions to describe the same gloss space. Preserve the isolated head as
an auxiliary loss, with its current checkpoint as the initialization and teacher. That
directly makes continuous training update Stage 1 while the isolated loss penalizes the
observed Citizen regression.

### Data/loss routing

| Source | Primary sequence loss | Auxiliary loss | Forbidden use |
|---|---|---|---|
| Fully annotated ASLLRP | CTC over complete known + `OTHER` sequence | exact-core and verified blank alignment | none of its labeled signing becomes blank |
| O5S5 positive-only | no full-narrative CTC | exact-core CE/CTC on short local spans | unannotated gaps as blank or complete transcript truth |
| Citizen/SemLex isolated replay | one-token CTC if helpful | existing pooled isolated CE + teacher retention | sequence/generalization claim |

Every admitted context window should be covered deterministically before replacement
sampling repeats it. Remove 1.07-second whole-window classification; retain long context
only as encoder context while supervising the known core/alignment. Keep 0.27 and 0.53
seconds as local auxiliary crops because they are materially more learnable.

## Evidence from primary sources

### Strongest match: pose regions + local multi-scale CTC

The 2025 ICCV Workshop paper *Generalizable Sign Language Recognition via Local Temporal
Convolutions and Region-Aware Pose Encoding* uses pose-only input split into left hand,
right hand, face, and body; a two-stage Conv1D temporal backbone; a shared classifier;
and multi-level CTC supervision. Its stated motivation is robustness to unseen sentence
composition and signing-speed variation. This maps closely to the existing 61-node
Apple Vision schema and supports the recommended shallow local decoder.

Paper: https://openaccess.thecvf.com/content/ICCV2025W/MSLR/html/Tran_Generalizable_Sign_Language_Recognition_via_Local_Temporal_Convolutions_and_Region-Aware_ICCVW_2025_paper.html

### Train the visual encoder, not only the decoder

VAC identifies insufficient visual-feature training as a central CSLR problem and adds
visual and visual-to-alignment auxiliary constraints to make the network end-to-end
trainable. The local audit shows the same interface failure: the current accepted Stage
2 freezes Stage 1, while the window model learns a pooled label rather than sequence
alignment.

Paper and official implementation:
https://arxiv.org/abs/2104.02330 and https://github.com/ycmin95/VAC_CSLR

The CVPR 2023 *Distilling Cross-Temporal Contexts* paper reports that a shallow temporal
aggregation module permits more thorough training of the spatial perception module,
then transfers local/global context through distillation. This directly argues against
putting another powerful independent Transformer after a weak or frozen visual module.

Paper: https://openaccess.thecvf.com/content/CVPR2023/html/Guo_Distilling_Cross-Temporal_Contexts_for_Continuous_Sign_Language_Recognition_CVPR_2023_paper.html

### CTC and Squeezeformer are compatible

CTC directly trains unsegmented input-to-label sequences using a blank symbol. It does
not require a separate frozen Stage 2. Squeezeformer was developed as an efficient
sequence encoder and reports Conformer-CTC comparisons; keeping the encoder while
changing the pooled-window objective is technically coherent.

Papers and implementation:
https://www.cs.toronto.edu/~graves/icml_2006.pdf,
https://arxiv.org/abs/2206.00888, and
https://github.com/kssteven418/squeezeformer

### Optional later additions

Temporal Lift Pooling explicitly targets preservation of discriminative temporal
movement and gloss borders during downsampling. It is a reasonable ablation only after
the simple 4× Conv1D baseline, because it adds machinery to a problem that first needs a
working joint objective.

Paper and code:
https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950506.pdf and
https://github.com/hulianyuyy/Temporal-Lift-Pooling

TwoStream-SLR jointly models RGB and keypoints with lateral interaction, auxiliary
supervision, and frame-level self-distillation. If the landmark-only joint model still
fails LG exact cores, the repository already has hand-crop embeddings that could supply
a smaller second stream. Full RGB TwoStream-SLR is not the first move because it raises
memory and iPhone cost before the landmark objective is fixed.

Paper and official code:
https://papers.nips.cc/paper_files/paper/2022/hash/6cd3ac24cdb789beeaa9f7145670fcae-Abstract-Conference.html and
https://github.com/FangyunWei/SLRT

The EMNLP 2024 online CSLR work shows that an isolated dictionary plus contextual crop
training and sliding inference can work on its benchmarks. This repository has now run
that family twice and measured poor boundary/emission behavior, so it remains useful as
a comparison rather than the next primary architecture.

Paper and code:
https://aclanthology.org/2024.emnlp-main.619/ and
https://github.com/FangyunWei/SLRT

## Options ranked for this repository

| Rank | Architecture | Fit to evidence | Cost/risk |
|---:|---|---|---|
| 1 | Shared Squeezeformer encoder + shallow causal multi-scale Conv1D + shared CTC heads + isolated CE | Directly fixes frozen/pooling interface; reuses current encoder and features | Smallest credible change |
| 2 | Same joint model plus aligned hand-crop stream | Can address handshape/domain errors if landmark exact-core LG remains weak | Higher memory, extraction, and device cost |
| 3 | Full RGB/keypoint TwoStream or CorrNet family | Strong published CSLR direction | Poor first fit for offline low-end iPhone and current cached data |
| 4 | RNN-T or autoregressive gloss decoder | Native streaming/history modeling | More data, decoder complexity, and alignment risk; no local evidence it fixes visual learning |
| 5 | Another frozen Stage-2 Transformer or pooled sliding-window classifier | Already available | Repeats the measured failure mode |

The cited results come from other sign languages, datasets, and compute settings. They
support the design choice but do not predict ASL performance in this repository; only a
matched local experiment can establish that.

## Bounded next experiment

One architecture comparison is enough:

1. initialize the shared encoder and isolated head from the accepted Stage 1;
2. add only the two-stage causal Conv1D and shared CTC classifier;
3. train on every fully annotated ASLLRP sequence each epoch, every O5S5 exact core at
   least once per epoch, and the existing isolated replay;
4. compare joint training against an otherwise identical frozen-encoder control;
5. select on the existing development gates, adding full-pool exact-core accuracy and
   source-level deletion/blank calibration; and
6. stop after one seed unless the joint model beats the frozen control and every
   retention gate. The Citizen test remains sealed.

This comparison answers the architecture question directly: whether gradients from
sequence alignment improve the shared visual encoder. A new backbone, RNN-T, language
model, or hand-image stream would confound that test.
