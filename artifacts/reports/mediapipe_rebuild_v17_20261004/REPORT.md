# MediaPipe (Android) family — exact rebuild of the words-only live chain (2026-10-04)

Every stage of the shipped Apple Vision words-only chain was retrained on MediaPipe-only inputs
with the Apple recipe (same code paths in `active/v17` and `scripts/`, same data lists, seeds,
hyperparameters and selection rules), then scored on the same validation sets. **All stages pass
the user's gate (MediaPipe within 5 points of its Apple counterpart).** No sealed test split was
read. Nothing existing was deleted or overwritten; every output is a new `mp_*_20261004` path.

MediaPipe validation numbers below count the clips MediaPipe could not extract (too few hands) as
errors: Citizen 0, SemLex 4/978, local 16/2896.

## Stage gates (validation, top-1 %)

| Stage | Domain | Apple | MediaPipe | Δ | Gate |
|---|---|---:|---:|---:|:-:|
| Landmark base (part-wise, roll aug) | Citizen | 95.77 (362) | 91.80 (347) | −3.97 | pass |
| Landmark branch (local replay FT, promotion ckpt) | Citizen | 95.50 (361) | 93.65 (354) | −1.85 | pass |
| | Local | 96.34 | 95.99 (2780/2896) | −0.35 | pass |
| Hand base (MobileCLIP2 crops) | Citizen | 80.69 (305) | 79.63 (301) | −1.06 | pass |
| Hand branch (local replay FT; both keep epoch 0) | Citizen | 80.69 | 79.63 | −1.06 | pass |
| | Local | 58.15 | 58.08 (1682/2896) | −0.07 | pass |
| Unified fusion (3 seeds; both select seed 5101) | Citizen | 96.30 (364) | 94.44 (357) | −1.85 | pass |
| | SemLex | 89.06 (871) | 87.73 (858) | −1.33 | pass |
| | Local | 97.10 (2812) | 97.62 (2827) | +0.52 | pass |
| reel_v2 phrase/activity adaptation | Citizen | 96.03 | 94.44 | −1.59 | pass |
| | SemLex | 89.16 | 87.73 | −1.43 | pass |
| | Local | 97.03 | 97.51 (2824) | +0.48 | pass |
| | Phrase segments (259) | 66.41 | **77.61** | +11.20 | pass |
| | Activity crops (1036) | 69.79 | **79.05** | +9.27 | pass |
| Span recognizer local_a (both select epoch 4) | Citizen | 95.24 (360) | 94.44 (357) | −0.79 | pass |
| | SemLex | 89.67 (877) | 87.73 (858) | −1.94 | pass |
| | Local | 96.31 (2789) | 97.20 (2815) | +0.89 | pass |
| Boundary student (same 43 val videos, 5,391 frames) | KL to DGS teacher ↓ | 0.2698 | 0.2687 | better | pass |
| | Frame agreement | 76.3% | 77.8% | +1.5 | pass |

## End to end (live replay, words only, PyTorch backend, `stream_config_v2` values)

| Set | Apple WER | MediaPipe WER | Notes |
|---|---:|---:|---|
| Tuning pool (89 local) | 10.18% | 13.27% | MP 207 vs 206 correct; precision 93.2 vs 95.8 (more insertions) |
| **Held-out 72 videos / 186 signs** | **11.83%** | **8.60%** | **gate ≤ 16.83%: pass** |
| — local60 (familiar signer, training templates) | 8.64% | 1.85% | 162/162 correct, 3 insertions |
| — asllrp12 (unseen fluent signer) | 33.33% | **54.17%** | 11/24 vs 17/24 correct |

Held-out: precision 97.2 / recall 93.0 (Apple 92.6 / 94.6); words shown within 0.5 s of sign end
96.6% (Apple 98.1%; Mac timing incl. MediaPipe, not phone timing). The held-out set was replayed once,
after every choice was frozen.

**Read with care.** The overall improvement comes from the familiar local signer. On the only
unseen-signer subset (ASLLRP, 24 signs) MediaPipe is clearly worse, consistent with the measured
continuous hand-detection gap (ASLLRP hand-slot frames 80% vs Apple 93%; not fixable by threshold
or tracking mode). 24 signs is a small sample; it is the clearest risk for real users.

## Deviations from a bit-exact Apple replay (all recorded in provenance)

1. Stage 1 base ran on MPS instead of a Kaggle T4 (CUDA + AMP); the user chose MPS.
2. Current `active/v17` code (user instruction) instead of the historical overlay copies; the hand
   trainer now requires `--no-cache` with local data, which was followed.
3. Span-recognizer tuning selection uses `alpha=1.0` (verifier only, as the live runtime ships);
   Apple's selection used `alpha=0.7` with an Apple-only proposal model that has no MediaPipe twin.
4. The fusion distils Citizen clips from Apple's cached four-stream teacher scores (by clip ID),
   as the amended locked decision allows; Apple models never receive MediaPipe features.
5. Recipe values that were hard-coded Apple numbers were re-derived by the same rule: fusion Citizen
   floor = landmark-branch correct (Apple 361 → MP 354); landmark FT floor = base − 1 (346);
   span floors = reel_v2 − 1, SemLex kept at incumbent (93.44 / 88.09 / 97.06).
6. Training sets matched to what Apple actually used when its models were trained: boundary
   student on Apple's 538-video key set (reconstructed; reproduces its 5,391 validation frames);
   span recognizer on the 6,254 non-prefix spans (prefix spans were added later for recognizer B).
7. `early_unsafe` recomputed for the MediaPipe recognizer on isolated validation (15 glosses, 11
   shared with Apple). Decoder and stream values are unchanged from `stream_config_v2`.

## Artifacts

Config: `stream_config_mediapipe_v2.json`; replays: `live_replay_{tune,test}_mediapipe.json`;
early-unsafe: `prefix_confusion.json`. Checkpoints (SHA-256 prefix):

| Model | Path | SHA-256 |
|---|---|---|
| Landmark base | `artifacts/models/mp_stage1_v17_orientation_robust_v1_20261004/best_model.pth` | 248095ad0f74fe85 |
| Landmark branch | `artifacts/models/mp_stage1_v17_local_deep_clean_mouth_masked_replay_ft_v1_20261004/best_promotion_gate_model.pth` | 061b7d078bf21f32 |
| Hand base | `artifacts/models/mp_stage1_v17_hand_mobileclip2_multisource_balanced_20261004/best_model.pth` | 3a5d608255fba005 |
| Hand branch | `artifacts/models/mp_stage1_v17_hand_mobileclip2_local_deep_clean_replay_ft_v1_20261004/best_model.pth` | 7ec884b7e53ed578 |
| Unified fusion | `artifacts/models/mp_stage1_v17_unified_multimodal_student_v1_20261004/best_model.pth` | c389928b68efbb5a |
| reel_v2 | `artifacts/models/mp_stage1_v17_unified_phrase_activity_adapt_reel_v2_20261004/best_model.pth` | d8d5341df9aff2e9 |
| Span recognizer | `artifacts/models/mp_span_recognizer_v17_local_a_20261004/best_model.pth` | a9340d6ef7359604 |
| Boundary student | `artifacts/models/mp_av_boundary_student_v17_l6_a_20261004/model.pth` | 9a2603d292144d23 |

Inputs: `data/local/mediapipe_full_v17_20261003/` (fingerprint d17b7cd2ecc5614f). Logs:
`artifacts/generated/mp_*_20261004.log`. Code changes are additive MediaPipe options; Apple defaults
are unchanged; 98 focused tests pass.

## Next

TFLite conversion with parity checks (boundary student, span recognizer, MobileCLIP2-S0 image tower,
Stage 3 T5), then the Kotlin port and 4 GB-phone measurements. The ASLLRP unseen-signer gap is the
main accuracy risk to address (e.g. a second hand pass on a body-centred crop) before broad use.
