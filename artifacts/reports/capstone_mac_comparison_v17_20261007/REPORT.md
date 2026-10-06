# Matched Mac pipeline processing — 2026-10-07

## Outcome

Both pipelines completed all 378 identical canonical validation recordings with zero extraction failures. No protected test, training, model selection, dataset changes or acquisition. These timings are development-computer measurements, not mobile operating-system rankings.

| Stage | Apple Vision (ms/clip) | MediaPipe (ms/clip) |
| --- | ---: | ---: |
| Landmark-only | 356.127 | 546.680 |
| Combined inputs | 668.637 | 809.295 |
| Interval-adapted | 668.701 | 808.632 |

## Measurement contract

Apple M4 development Mac. Each family ran separately to avoid competing benchmark processes. Both used the same ordered input videos and recorded orientation decisions, production sampling contracts, a shared FP32 MobileCLIP2 image encoder, and FP32 PyTorch recognition on MPS. Models were warmed before measurement. Recognition cost is the median of two synchronized calls per graph per clip.

Per-clip costs sum measured stages: landmark extraction and feature preparation, tensor transfer, applicable hand-crop preparation/hand-image encoding, and recognition. The reported aggregate is the median of these per-clip stage sums, not the sum of population medians or a live streaming response time. Initial video decoding/resizing, model loading, detector renewal, camera capture, English generation and speech/UI work are excluded. Hand-crop preparation includes its selected-frame decoding. Landmark-only totals include the same small shared tensor-transfer measurement; the separate hand-image extraction cost is excluded from that stage.

The three recognition accuracy values are sourced from mediapipe_rebuild_v17_20261004/REPORT.md, not recomputed by this timing script. Same validation membership; no additional MediaPipe-only evaluation dataset. The prior rebuild reports matched training lists; this benchmark does not modify them. Timed features are freshly extracted and encoded; recognition accuracy evidence uses the saved evaluation contract. Accuracy and timing are separate measurements of the named stages.

## Failed first attempt and recovery

Initial MediaPipe execution exceeded MPS memory capacity because of the documented macOS GPU pixel-buffer leak. Its partial JSON and failure log are preserved and excluded. The complete retry renews the detector between clips once its call counter reaches 1000, outside timed work. This follows the existing detector renewal mechanism and per-sequence tracking contract. No allocator limit was disabled. Apple completed without this failure. These are steady-state staged costs, not sustained deployment/resource readiness evidence.

## Provenance

benchmark.py contains exact checkpoint and source paths. apple.json and mediapipe.json retain every per-clip component timing. summary.json records hashes and checks. chart_values.json drives the manuscript figure. Device comparison remains in ../phone_precision_v17_20261007/REPORT.md.
