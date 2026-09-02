# Lightweight streaming recognizer research

Date: 2026-09-03

## Decision

There is no existing checkpoint that can replace v17 Stage 2 directly. Published
models use different sign languages, vocabularies, keypoint schemas, and usually
non-mobile RGB backbones. The best first experiment for this repository is a small
**causal landmark TCN with a 101-way CTC head**: the 100 frozen glosses plus CTC
blank. It should consume v17 frames incrementally, cache temporal state, and emit a
gloss only after a stable non-blank run followed by blank evidence.

This is a streaming sequence recognizer, not the current whole-phrase Stage-2 live
policy and not another sliding-window isolated classifier.

## Primary evidence

1. [Towards Online Continuous Sign Language Recognition and Translation (EMNLP
   2024)](https://aclanthology.org/2024.emnlp-main.619/) is the closest published
   analogue to Reel. It trains an isolated model on CTC-derived sign crops and an
   explicit background category, then applies 16-frame sliding windows and voting.
   Its paper reports 320 ms algorithmic latency, 29 ms window processing latency,
   and 5.1 GB memory on an NVIDIA V100. The official
   [SLRT implementation](https://github.com/FangyunWei/SLRT/tree/main/Online/CSLR)
   uses S3D video/keypoint networks. Its background and crop-augmentation method is
   relevant; the model is not a lightweight mobile drop-in.

2. [Fully Convolutional Networks for Continuous Sign Language Recognition (ECCV
   2020)](https://arxiv.org/abs/2007.12402) uses temporal convolutions and CTC and
   demonstrates intermediate recognition on partial and newly combined sequences.
   It supports a convolutional sequence head over an RNN that memorizes sentence
   order, but its published frontend is RGB and the paper does not establish causal
   mobile inference.

3. [A Closer Look at Skeleton-based Continuous Sign Language Recognition (ICCVW
   2025)](https://openaccess.thecvf.com/content/ICCV2025W/MSLR/html/Min_A_Closer_Look_at_Skeleton-based_Continuous_Sign_Language_Recognition_ICCVW_2025_paper.html)
   is the closest input match. It finds coordinate keypoints more effective and
   compact than skeleton heatmaps, groups body/hands/face/mouth, and supervises both
   a short GCN/1D-CNN output and a longer BiLSTM output with CTC. Its short-term
   prediction generalizes better to unseen sentences in its experiment. The
   [official code and weights](https://github.com/VIPL-SLP/MSLR_ICCV2025) target a
   different language and 86-keypoint schema, so architecture can transfer but
   weights cannot.

4. [CoSign (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Jiao_CoSign_Exploring_Co-occurrence_Signals_in_Skeleton-based_Continuous_Sign_Language_Recognition_ICCV_2023_paper.html)
   shows why hands, body, face, and mouth should remain separate landmark groups.
   Its single-stream model is still 21.4M parameters and 5.8 GFLOPs for 100 frames,
   excluding pose extraction, so copying the full model would miss the mobile goal.

5. [Pose-Based Temporal Convolutional Networks for Isolated Indian Sign Language
   Word Recognition (WSLP 2025)](https://aclanthology.org/2025.wslp-main.8/)
   demonstrates the smallest relevant building block: MediaPipe landmarks and four
   residual dilated causal Conv1D blocks. It is isolated recognition, not CSLR, but
   its causal TCN is directly reusable before replacing its global classifier with
   a framewise CTC head.

6. The original [CTC paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
   provides the required training objective for unsegmented sequences and the blank
   symbol that separates signs. If a small causal TCN lacks sufficient long history,
   [Emformer](https://arxiv.org/abs/2010.10759) is a proven cached streaming-attention
   upgrade, but it is unnecessary complexity for the first 100-gloss experiment.

## Proposed v17 pilot

- Input: each new 61-by-5 v17 landmark frame, including existing mouth points.
- Encoder: separate lightweight projections for left hand, right hand, body, and
  face/mouth, fused to 128 dimensions.
- Temporal head: four to six depthwise-separable causal Conv1D residual blocks with
  dilations. A 31-frame receptive field is about 1.55 seconds at 20 FPS.
- Output: framewise logits for CTC blank plus the frozen 100-gloss vocabulary.
- Runtime: cached state, one new-frame update at a time; greedy prefix decoding first.
- Training: genuine continuous phrase clips with exact gloss sequences, augmented by
  isolated clips with leading/trailing blank and stitched development sequences.
  Signer-disjoint validation remains mandatory.
- Gate: compare against Reel on exact sequence, token error rate, insertion rate,
  time-to-first-correct-gloss, correction rate, sustained FPS, and Core ML latency.

A 128-dimensional implementation should be well under one million parameters. That
is a design estimate, not a measured result. Do not add Emformer, RNN-T, a language
model, or beam search unless the causal-TCN development results show a specific need.

## Data implication

Phrase data is still required. Isolated clips teach lexical identity, but genuine
phrases teach the model what blank/coarticulation regions look like and prevent it
from treating every moving window as a completed sign. Stitched isolated clips are
useful augmentation, not a substitute for signer-disjoint continuous validation.
