# SHuBERT paired decision probe — 2026-09-22

## Outcome

**Mixed result; no live promotion.** The frozen SHuBERT feature readout rejected the known false WORK gap that the prior landmark gate and DINO/body control accepted. It also suppressed three correctly recognized FRIEND cores. It therefore does not meet the practical requirement of removing transition errors while retaining successful recognition. This is a screening result for one fixed readout, not a verdict that SHuBERT cannot learn the task.

## Paired held-out results

These are conditional commits on annotated windows. They bypass the live candidate scheduler and are not full-stream recognition or WER. The three context columns reuse the same 17 cores; they must not be summed as independent signs.

| Gate | Gaps rejected / 3 | False gap commits | Correct tight-core commits / original 5 | Correct +100ms commits / original 9 | Correct +250ms commits / original 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| unchanged | 0 | 1 | 5 | 9 | 5 |
| confidence | 1 | 1 | 5 | 9 | 5 |
| landmarks | 0 | 1 | 5 | 8 | 4 |
| dino_body | 0 | 1 | 5 | 8 | 4 |
| shubert | 2 | 0 | 2 | 6 | 2 |

SHuBERT rejects the WORK and NOW gap windows, but accepts WRITE. Only WORK would conditionally commit in the unchanged baseline, so false gap commits fall from 1 to 0. On tight cores SHuBERT passes 12/17 windows overall, but removes three successful FRIEND commits and one existing wrong commit. This is why raw gate-pass counts alone would make the result look better than it is.

No held-out core has the earlier wrist-start-failure flag. **Low-wrist-motion improvement was not tested** by this sample. No 100-sign accuracy or overall WER was measured.

## Fixed experiment contract

- Same 247 interval records and original roles as the earlier decision probe; 45 videos with qualifying records from the 56-video approved annotation view. No new or dropped interval records.
- Parent videos: 26 fit, 7 calibration, 12 validation. Fit:48cores ×3contexts plus14gaps; calibration:11cores ×3contexts plus2gaps; validation:17cores ×3contexts plus3gaps.
- The existing gaps are guarded internal known-to-known annotation gaps, not newly collected or manually confirmed live confusions. Validation is reused development data.
- All neural weights frozen. Compare four temporal bins of landmarks, DINO face/hands plus body, or SHuBERT contextual output, each with the same two recognizer scores.
- Same balanced ridge100, standardization and train-parent calibration split. Threshold is minimum calibration-positive score, with no held-out threshold sweep. Feature dimensionality differs, so this single regularization setting is a limited readout comparison, not proof of the best achievable model.
- SHuBERT sees full-clip future context. No causal/live behavior is claimed. Only a small closed-form readout was fit; generic phrase-training gate remains false.

## Extraction and verification

All45videos/1,585frames completed on MPS for YOLO/DINO/SHuBERT; MediaPipe used its own runtime. Total per-video extraction wall time272.02seconds (median5.685seconds), including repeated model loads and intermediate crop writes. This is offline extraction timing, not streaming latency.

| Role | Frames | Face detections | Left hand | Right hand | Pose |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 1205 | 1196 | 1053 | 1073 | 1205 |
| validation | 380 | 380 | 328 | 329 | 380 |

No video had zero face detections; all frames had pose detections. Missing face/hand crops use the published carry-forward behavior, so successful tensor extraction is not equivalent to every frame having a fresh detection. No source was skipped due to detection quality.

The first scoring attempt aborted before fitting because intermediate MP4 encoding rounded30000/1001FPS to29.97. Exact annotation endpoints then lost frames. Corrected the comparison clock to original source FPS, reusing unchanged cached features. Independent checks decoded all original sources, verified source/crop frame counts and every corrected interval frame count against prior evidence. The initial runner, recipe and failure log remain preserved; the revised recipe explicitly admits those unchanged extraction caches. This was a harness bug, not a discovered fault in the existing Reel model.

Checks passed: interval pooling, parent split isolation, original-clock regression, source/crop frame counts, all247prior interval frame counts, exact reproduction of prior landmark/unchanged/confidence results, positive calibration retention, and no invented positive or gap commits. Full coverage/timing audit is in `audit.json`; source/weight/script hashes are pinned by the versioned recipe and cache completion records.

## Decision and next action

Keep SHuBERT as a candidate representation: its contextual features separated one difficult transition that the noncontextual control did not. Do **not** deploy this readout or tune it against these three reused gaps. It loses too many correct commits and has no low-motion evidence.

Continue the requested pretrained comparison with Zuo/TwoStream before choosing broader training. If SHuBERT is retained afterward, the next step is a trained temporal decision head with broader already-available reviewed interval supervision and independent evaluation, rather than another threshold tuned on these same cases. That is a separate bounded recipe; no combined training or dataset acquisition was started here.

Data provenance and future dataset options remain in [the SHuBERT data register](../shubert_probe_v17_20260922/DATASETS.md).

## Reproduce

```sh
venv/bin/python artifacts/reports/shubert_decision_probe_v17_20260922/run.py
venv/bin/python artifacts/reports/shubert_decision_probe_v17_20260922/check.py
```

The runner validates hashes and reuses completed extraction caches. Default live Reel and app code were not changed in this comparison.
