# Zuo online blank-head transfer — 2026-09-22

## Result: no live promotion

The actual released PHOENIX-trained online model rejected **3/3 gap-centred windows**, including the false WORK transition. It also rejected **all5 of the baseline's correct tight-core commits**. Only4/17 core-centred windows were nonblank, and none of those corresponded to the five previously successful commits.

| Conditional outcome | Existing Reel evidence | Released Zuo blank veto |
| --- | ---: | ---: |
| False gap commits | 1 | 0 |
| Correct tight-core commits | 5 | 0 |
| Core windows allowed through | 17 | 4 |

The suppressed successful signs were four FRIEND cores and one COLD core. This transferred gate is too destructive for Reel. It does not show that Zuo's method cannot work after ASL adaptation; its classifier and blank decision were trained in a different language/domain. Language, framing, pose extraction and the explicitly changed resize geometry are possible contributors, not individually established causes.

## What was actually run

- Recovered the author's **online PHOENIX-2014T checkpoint**, not a generic Kinetics S3D substitute. Strictly loaded all938model-state tensors, including both streams, lateral fusion and three classification heads. Vocabulary has1115glosses plus blank at0.
- Ran published HRNet-W48 DARK COCO-WholeBody pose extraction on MPS. Full133points, selecting63in official order:42hand,10mouth,11pose. The exact pose checkpoint used by the author for supplied dataset keypoints is unconfirmed.
- Used the same17reviewed ASLLRP validation cores and3guarded gaps as the prior diagnostic, across12videos. For each interval centre, selected16frames at25Hz using original source timestamps, nearest-frame selection and edge clamping. This is **not the same input window protocol as the SHuBERT interval-pooling readout**, so it is not an encoder ranking.
- Reused the existing signer-crop videos after checking frame counts against original sources. Aspect-preserving224square letterbox replaces the upstream direct square resize; corresponding keypoints move with it. Heatmaps use224square coordinates, sigma8, then112square output. This preserves geometry but is a declared transfer deviation, not faithful PHOENIX benchmark reproduction.
- BGR pixels normalized to[-1,1]. Three published-head softmax vectors averaged; blank is argmax0. No ASL head fitting, confidence-threshold sweep or validation-selected parameter. Foreign gloss predictions are not interpreted as ASL labels.

The annotation centre selects a candidate window for this diagnostic. The gate then vetoes the earlier conditional Reel decision. It does not run the actual candidate scheduler, de-duplication, or full transcript decoder. No WER,100-sign accuracy, sustained live latency or low-wrist-motion improvement is claimed. Three gaps and reused development data cannot establish general transition accuracy.

## Measured execution and checks

The20-window transfer run took88.86seconds after model loading:73.69seconds in HRNet calls and11.03seconds in recognition forwards. Models useMPS; unsupported3Dpooling runs onCPU. These are diagnostic timings with cached per-video frame poses, not end-to-end live throughput.

Earlier synthetic16-frame compatibility smoke:3.314seconds for one cold forward, all938tensors loaded strictly; maximum CPU/MPS output probability difference1.788e-7. Synthetic compatibility is not an accuracy result. Real-frame HRNet smoke loaded strictly, returned133finitekeypoints, and its overlay was inspected before the transfer run.

Focused checks passed: aspect-preserving geometry, all20exact held-out identities, original baseline decisions,16ordered indices per window, finite probability vectors, blank/veto decisions and independently recomputed summary counts. Independent read-only review confirmed keypoint order, BGR normalization, three-head fusion and conditional-metric interpretation. `git diff --check` passed. No app/live code changed.

## Recovered weights and documented data

The old official SharePoint links returned404. The [author's replacement notice](https://github.com/FangyunWei/SLRT/issues/106#issuecomment-4915936329) led to a public Google Drive archive. HTTP-range ZIP reads fetched only the456MBonlinecheckpoint and12.6KBvocabulary, verifying ZIP CRC; hashes are saved in `weights.json`. No signing dataset payload was downloaded. The older PHOENIX-2014T TwoStream S2G teacher is represented by an error-file placeholder in its recovered archive, but the online checkpoint is intact.

[DATASETS.md](DATASETS.md) records PHOENIX/CSL data, pseudo-boundary versus manual-label distinctions, HRNet/Kinetics/WLASL initialization, access restrictions, original and replacement links, and what was actually acquired. Source commit, HRNet asset hash and isolated-runtime notes are preserved. Generic phrase training remains blocked; no training gate was changed.

Dependency repairs were confined to `artifacts/vendor/zuo_runtime`: MMPose0.29.0, compatible MMCV1.7.0, and legacy build dependencies. Initial import/build failure logs remain. HRNet's CUDA-only scatter was locally adapted to move prepared image tensors toMPS while retaining metadata onCPU. No model math was replaced.

## Decision

Do not attach this zero-shot blank head to Reel. Preserve the recovered weights and frontend for later ASL adaptation; do not tune against these three gaps. Next in the user's requested comparison is Zhao/MHB release availability. Only after that review should we choose a broader in-domain training experiment. Renz, SHuBERT and this online model have been tested under different scoped protocols; none is demonstrated as a ready-to-deploy solution to the current live transition errors.

## Reproduce

```sh
venv/bin/python artifacts/reports/zuo_twostream_probe_v17_20260922/model_smoke.py
venv/bin/python artifacts/reports/zuo_twostream_probe_v17_20260922/pose_smoke.py
venv/bin/python artifacts/reports/zuo_twostream_probe_v17_20260922/transfer.py
venv/bin/python artifacts/reports/zuo_twostream_probe_v17_20260922/check.py
```

The transfer recipe pins source evidence, manifests, runner and model assets. `transfer_results.json` contains per-window outputs and limitations. `list_remote_zip.py` and `download_online.py` reproduce selective archive discovery/acquisition; do not run the vendor's bulk dataset download script.
