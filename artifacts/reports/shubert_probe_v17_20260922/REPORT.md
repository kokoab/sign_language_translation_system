# SHuBERT published-weight MPS probe — 2026-09-22

## Outcome

The released SHuBERT encoder and its fine-tuned face/hand DINO models load **strictly** and run on MPS. A real approved ASLLRP validation clip produced finite 41 × 768 contextual features. This is a working extraction/inference probe, **not** a recognition, WER, or transition-rejection result. No head was trained, dataset combined, default live mode changed, or protected test used.

A meaningful extraction finding: skipping the published signer crop gave **0/41 face detections**. Applying the published YOLO signer-crop routine gave **41/41** on the same video. Do not dismiss or compare this model using incomplete frontend inputs. This finding is specific to this clip, not proof that our Apple Vision pipeline has the same failure.

## Measured on WATER COLD

Approved manifest role: validation; path contains historical `train_candidate`, so role was taken from the current manifest. Source hash was checked before reading. Duration 1.368 seconds, 41 frames.

| Measurement | Published signer crop enabled |
| --- | ---: |
| Face detections | 41/41 |
| Body pose detections | 41/41 |
| Left/right hand detections | 34/41 and 37/41 |
| YOLO crop, including load and intermediate write | 4.727 s |
| MediaPipe + face/hand/body extraction | 2.715 s |
| DINO feature extraction, including model loading | 2.283 s |
| SHuBERT encoder forward | 0.456 s |
| Output | 1 × 41 × 768, finite |

These are one-clip timings, not sustained live throughput. Missing hand crops carry the previous crop as upstream does. YOLO, DINO, and SHuBERT use MPS; MediaPipe uses its own runtime. The encoder sees the whole clip, including future context. Loading costs, frontend work, and noncausality prevent calling the 0.456-second encoder figure end-to-end live latency.

## Reproduce

```sh
venv/bin/python artifacts/reports/shubert_probe_v17_20260922/smoke.py --signer-crop
venv/bin/python artifacts/reports/shubert_probe_v17_20260922/check.py
```

The smoke assertions check the approved source hash, strict checkpoint compatibility, finite output, and frame alignment. `check.py` checks stream dimensions/finite values and CPU/MPS encoder agreement on the saved real-video streams. Measured maximum absolute CPU/MPS encoder difference: **1.824e-5**; check passed. It does not check CPU parity for the complete frontend or DINO.

Official SHuBERT source pinned to `cc1929326075bfbad7ad73159b2acf84356059bb`. DINO source, all weight hashes and download IDs are saved in `provenance.json` and `downloads.json`. Published weights total approximately 1.28 GB, excluding small detector assets. Runtime dependencies are isolated in `artifacts/vendor/shubert_runtime`; the existing live environment's dependencies were not replaced.

Upstream helper functions are executed directly without their unused decord/Slurm driver imports. OpenCV decodes RGB; face/hand crops stay in memory instead of intermediate lossy MP4 encoding. Signer cropping uses the published routine, including its intermediate MP4. Two DINO source files need postponed type annotations for Python 3.9; numerical operations are unchanged. Initial failed import logs and uncropped results are retained. No automatic xFormers/CUDA dependency was installed.

## Data and decision

See [DATASETS.md](DATASETS.md) for the source links, pretraining versus downstream data, overlap risks, and relation to our existing YouTube keypoints. SHuBERT supplies a richer pretrained representation; it does not supply a ready transition classifier or our 100-sign head. Its data is documented for later combined experiments, not downloaded or admitted here.

Next useful SHuBERT accuracy experiment: freeze the encoder, cache features with this complete frontend, and compare a small sign/transition readout against the existing features on the **same reviewed training and held-out intervals**. Keep transition rejection, retained correct signs, low-motion recall, and extraction failures separate. Do not score an unfitted encoder as WER, select thresholds on held-out gaps, or treat the tiny existing gap set as conclusive. That probe requires a reviewed bounded training contract; the generic phrase-training gate remains unchanged. Keep the other pretrained candidates in the comparison queue before deciding on broader retraining.
