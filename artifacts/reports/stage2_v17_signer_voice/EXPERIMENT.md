# Stage 2 signer-voice/coarticulation pilot

**Run date:** 2026-08-22 (Asia/Manila)  
**Status:** development experiment; not independent evidence and not a claim of
perceptually natural sign synthesis

## Question

Can train-only human signing trajectories be recomposed with a consistent signer
"voice"—timing, boundary motion, and neutral pose—so that the Stage 2 recognizer
generalizes better to genuine contiguous signing from a signer excluded from
training?

The held-out validation signer, JONATHAN, was never used to build the synthetic
sequences. Reproducing that signer's personal style directly would require using
held-out examples and would invalidate the experiment. The tested hypothesis is
therefore **unseen-style generalization from multiple training voices**, not cloning
JONATHAN.

## Evidence behind the design

- Saunders et al., *Signing at Scale* (CVPR 2022) learns sparse monotonic temporal
  alignment between dictionary concatenations and continuous signing, explicitly
  targeting coarticulation:
  <https://openaccess.thecvf.com/content/CVPR2022/papers/Saunders_Signing_at_Scale_Learning_to_Co-Articulate_Signs_for_Large-Scale_Photo-Realistic_CVPR_2022_paper.pdf>
- Joshi et al., *PoseStitch-SLT* (EMNLP 2025) reports that pose stitching and
  linguistic templates can improve low-resource sign-language modeling:
  <https://aclanthology.org/2025.emnlp-main.698/>
- Yang et al., *Combinational Sign Language Recognition* (CVIU 2024) treats
  feature-level composition and context consistency as a recognition problem:
  <https://www.sciencedirect.com/science/article/pii/S1077314224000535>
- ASLLRP provides native-signer continuous signing with frame-level sign bounds and
  nonmanual annotations, which is the appropriate real-data source for learning
  transitions rather than inventing them from isolated clips:
  <https://www.bu.edu/asllrp/rpt18/asllrp18.pdf>

## Defect found in the previous synthetic data

The previous ASLLRP synthetic phrases independently sampled every token. A single
phrase could therefore switch among BENJAMIN_JAMES_BAHAN, CORY, and RACHEL at token
boundaries. It also assigned every source sign the same 32-frame duration even when
the decoded source interval was shorter. That does not represent a coherent human
voice or natural timing.

## Pilot implementation

`active/v17/stage2_signer_voice_plan_v17.json` contains 12,000 train-only sequences:
6,000 Citizen replay sequences and 6,000 ASLLRP signer-voice sequences, balanced at
2,000 per training signer. The ASLLRP subset covers all 53 available classes and uses
90% two-sign and 10% three-sign sequences. Plan SHA-256:
`f0048f83047a4a969af491eab9bd9fbb2e12b59cab316ab5fcd3f370591fe75f`.

For each ASLLRP sequence, the composer:

1. samples every token from one training signer;
2. restores the sign's observed decoded duration from authoritative source ranges;
3. uses a monotonic boundary search to trim at most three peripheral frames while
   retaining at least four frames per sign;
4. inserts a two-frame linear bridge at each boundary;
5. moves through five frames of that signer's train-only neutral endpoint estimate at
   the beginning and end; and
6. repacks the raw-time stream into the extractor's 32-frame window contract.

This is a recognition-oriented feature compositor. Linear bridges are a controlled
baseline, not proof that the synthesized motion is perceptually or linguistically
natural.

## Results

All results below are development validation. Citizen, SemLex, and local test splits,
2M-Flores `devtest`, and the already-consumed RIT test were not accessed.

| Candidate | ASLLRP phrases (12 clips / 24 tokens) | Local phrases (97 clips / 259 tokens) | JONATHAN contextual (254 tokens) |
| --- | ---: | ---: | ---: |
| Existing v2 base | 12 edits, 50.0000% WER | 11 edits, 4.2471% WER | 54 edits, 21.2598% WER |
| Scratch signer-voice pilot | 12 edits, 50.0000% WER | 36 edits, 13.8996% WER | not promoted |
| Conservative v2 adaptation | **11 edits, 45.8333% WER** | **8 edits, 3.0888% WER** | 58 edits, 22.8346% WER |
| Adaptation + fixed context residual, weight 1.5 | **11 edits, 45.8333% WER** | **8 edits, 3.0888% WER** | **46 edits, 18.1102% WER** |

The conservative adaptation started from the exact v2 checkpoint, retained a frozen
v2 teacher, and optimized CTC plus temperature-scaled KL preservation at learning rate
`1e-5`. Three seeds were run. Seed 4702, epoch 4 won in 97.47 seconds. Its checkpoint
SHA-256 is
`26f7f1d63ee0c9aa106ae5b7fa38230f28005b82e1f32feae50c0ff3ffb88a75`.

The combined loadable artifact is
`artifacts/models/stage2_v17_signer_voice_context_adapted_w1p5_pilot_v1/model.pth`,
SHA-256 `62677053984e675f6e6d3d792d0551bfc36c5192aabc431a112df99ed7fd2cce`.
The context residual is the existing adapter fitted only on the three training
signers. Weights 0.5, 0.75, 1.0, 1.25, and 1.5 were inspected on development
validation; 1.5 was selected because it restored the existing 18.1102% contextual
WER while leaving both phrase metrics unchanged. This selection is validation-tuned.

The scratch pilot showed a useful but unstable specialization: it recognized three of
five `FRIEND NOW` clips exactly, versus zero for v2, but lost substantially elsewhere.
The conservative candidate did not retain that exact-sequence gain.

## Decision

The combined artifact is retained as an **experimental Stage 2 development
candidate**, not a new proven final model. It improves both phrase validation domains
and matches the existing contextual WER, but the ASLLRP gain is only one token on a
tiny set and was selected on that same set.

The next scientifically valid step is to learn boundary selection/transition dynamics
from more genuine train-only continuous utterances, then evaluate once on a new signer
and capture set. A visual naturalness claim additionally needs generated/rendered
motion reviewed by fluent signers; recognition WER alone cannot establish natural
coarticulation or preservation of nonmanual grammar.
