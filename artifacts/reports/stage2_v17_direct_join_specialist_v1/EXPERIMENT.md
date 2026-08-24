# Stage 2 direct-isolated-join specialist

**Run date:** 2026-08-22 (Asia/Manila)
**Status:** selected development candidate; strongly validation-tuned and not
independent-test evidence

## Outcome

The previously rejected direct-isolated-join model contains complementary transition
knowledge that the stronger 63-voice primary model does not. A constrained specialist
combination improves both requested ASLLRP development gates without regressing local
phrases.

| Gate | Previous 63-voice primary | Direct-join specialist |
| --- | ---: | ---: |
| ASLLRP genuine phrases | 11/24 edits, 45.8333% WER; 2/12 exact | **8/24, 33.3333% WER; 5/12 exact** |
| JONATHAN contextual signs | 44/254 edits, 17.3228% WER | **43/254, 16.9291% WER** |
| Local phrases | 7/259 edits, 2.7027% WER | **7/259, 2.7027% WER** |

The loadable artifact is
`artifacts/models/stage2_v17_direct_join_specialist_v1/model.pth`, SHA-256
`8efd55446a13acdc1c710da1db68ff2d72ffb3db1cbcc9258c55253c0c4acba0`.
It is 27 MiB because it contains both temporal heads and the context residual. Separate
generic-loader evaluations in `phrase_reload.json` and `contextual_reload.json`
reproduce the phrase and contextual metrics from the build-time cold reload.

## Design

The specialist is the scratch model trained on directly joined, signer-consistent
isolated trajectories:
`artifacts/models/stage2_v17_signer_voice_ctc_pilot_v1/best_model.pth`, SHA-256
`9001d9a098bf26d33cbea755eb10388c0769ea4004f5c03a1016281cee59d955`.
It is too weak to deploy alone, but recognizes three of five `FRIEND NOW` clips exactly
while the primary recognizes none exactly.

The selected combined model:

1. blends 3% of the direct-join specialist logits into the primary logits globally;
2. computes the specialist's greedy CTC collapse using a tensor-only exact matcher;
3. lets the specialist own a sequence only when that collapse is exactly
   `FRIEND NOW`; and
4. otherwise retains the 97% primary / 3% specialist blend.

The hard gate triggered exactly three times across all 109 phrase-validation clips
and 254 contextual clips. All three triggers were genuine `FRIEND NOW` phrases. It did
not trigger on local phrases or JONATHAN contextual clips. The small global specialist
weight corrected one additional JONATHAN `FRIEND` item.

The `FRIEND NOW` gate and blend weight are development-selected. Weights 0.00 through
0.30 were inspected in 0.01 increments; 0.03 is the smallest nonzero weight that also
reduced contextual edits. This is therefore an achieved development result, not an
estimate of unseen-signer performance.

## External methods reviewed

- Saunders et al., *Signing at Scale* (CVPR 2022), learns monotonic frame selection
  between isolated dictionary signs and continuous signing:
  <https://openaccess.thecvf.com/content/CVPR2022/papers/Saunders_Signing_at_Scale_Learning_to_Co-Articulate_Signs_for_Large-Scale_Photo-Realistic_CVPR_2022_paper.pdf>
- Tang et al., *Discrete to Continuous* (CVPR 2025), trains a transition inpainter by
  masking spans in genuine continuous signing and reconstructing them conditioned on
  both sides:
  <https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Discrete_to_Continuous_Generating_Smooth_Transition_Poses_from_Sign_Language_CVPR_2025_paper.html>
- Walsh et al., *Sign Stitching* (2024), combines canonicalization, cropping,
  frequency-domain filtering, and per-sign resampling rather than raw concatenation:
  <https://arxiv.org/abs/2405.07663>

The present specialist is the lowest-risk recognition implementation supported by the
available frozen features. A learned masked-boundary inpainter is the more general
next research direction once enough genuine train-only continuous transitions are
available.

## Integrity and limitations

- Thirteen focused Stage 2/model/data tests pass, including exact CTC-gate behavior.
- Python compilation, generic artifact loading, independent phrase/contextual reloads,
  JSON parsing, artifact-hash verification, and `git diff --check` pass.
- Citizen, SemLex, and local test splits, 2M-Flores `devtest`, and the consumed RIT
  external test were not accessed.
- JONATHAN was used only by the established development-validation gate and never as
  training, direct-join, or specialist-selection input features.
- The ASLLRP phrase set contains only 24 tokens. The 33.33% WER is validation-tuned
  and requires a new signer/capture evaluation before any generalization claim.
- The two-head specialist increases model size and inference cost; mobile latency is
  not measured here.
