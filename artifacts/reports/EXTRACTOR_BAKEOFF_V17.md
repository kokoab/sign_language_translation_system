# v17 Extractor Bakeoff

- Frozen manifest: `active/v17/extractor_bakeoff_manifest.json`
- Manifest entry SHA-256: `f27bd7be8bc904c36fa3d57d1c429765ec185e529e94e3297233400e396b36eb`
- Clips: 300 across 100 classes
- Splits used: train and validation only; official test remains sealed.
- Subset: per class, low and median Apple-coverage train clips plus lowest Apple-coverage validation clip.
- Bone-length CV is a tracking-stability proxy, not landmark ground truth.
- MediaPipe confidence is whole-hand confidence; Apple confidence is per-joint, so raw confidence is intentionally not compared.

## Aggregate measurements

| Candidate | Available | Median active output | Median source detection (pre-trim) | Median hand-node presence | Median two-hand activity | Median bone CV | Median seconds/clip | Median genuine-depth coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| apple | 300/300 | 0.8750 | 0.4265 | 0.4338 | 0.0156 | 0.2539 | 0.6780 | 0.0000 |
| mediapipe_t30 | 300/300 | 0.8750 | 0.3962 | 0.4531 | 0.0625 | 0.1991 | 1.3647 | 0.4555 |
| mediapipe_t50 | 300/300 | 0.8750 | 0.3854 | 0.4688 | 0.0938 | 0.1818 | 1.2302 | 0.4661 |

## Decision gate

This automated table does not select a winner. Review overlays on the largest coverage gains/losses and two-handed clips for false detections and anatomical placement. If MediaPipe is visually sound and competitive, extract the complete train/validation corpus and compare identical Stage 1 training runs. Validation accuracy is the final extractor-selection metric; latency, missingness, stability, package size, and genuine depth break close ties.

Per-clip measurements: `artifacts/reports/extractor_bakeoff_v17.csv`

## Final full-corpus Stage 1 selection

Both extractors used the same official signer-disjoint splits, model architecture, seed, augmentations, optimizer, schedule, EMA, and early stopping. The test split remained sealed.

| Extractor | Top-1 | Top-5 | Macro F1 |
| --- | ---: | ---: | ---: |
| Apple Vision | 93.12% | 99.47% | 92.53% |
| MediaPipe 0.50 | 89.95% | 97.35% | 89.81% |

Apple alone classified 26 clips correctly; MediaPipe alone classified 14. The exact paired two-sided p-value is 0.0807. The five validation signers make this uncertainty worth stating, but Apple wins the engineering decision on higher top-1/top-5, faster extraction, and better visual recovery of overlapping hands.

**Selected v17 extractor: Apple Vision.**
