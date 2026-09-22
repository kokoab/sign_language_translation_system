# SHuBERT frozen feature comparison

User authorized continuation on 2026-09-22.

1. Verify approved manifest and pin evidence, extraction code and weights in recipe.json.
2. Reuse complete signer-crop/MediaPipe/DINO/SHuBERT extraction on only the 45 existing evidence videos, MPS, source hashes checked. Cache features; no labels used in extraction.
3. Pool the identical 247 intervals on the original 20 Hz schedule. Keep all parent-video contexts in their existing role.
4. Fit the existing fixed balanced ridge100 readout and train-only calibration rule. Compare unchanged/confidence, landmarks, DINO/body and SHuBERT. No tuning on held-out outcomes.
5. Verify exact reproduction of old controls, report false transition commits and retained correct signs separately, preserve extraction coverage and failures. No default live promotion.

Ruling: reuse the already admitted interval evidence and negative contract; do not expand the negative pool or treat this as permission for generic mixed phrase training. The small gap sample and noncausal context limit any conclusion.

Ruling: use the same probe extractor with per-video output parameters instead of implementing a second frontend. Repeated model loading is acceptable for this bounded 45-video diagnostic; cache output for later reuse.

Review follow-up: an independent pass flagged source/crop clock checks and final coverage reporting. Added an independent post-run audit that decodes each original source, verifies equal frame counts, near-identical FPS and exactly identical sampled interval indices, and summarizes all detector coverage. Negative commits must never increase relative to unchanged, but zero remaining false commits is an outcome to measure, not a prerequisite for recording a completed experiment. No model/threshold settings changed.

Completed: all five steps. First scoring aborted before fit due to cropped FPS rounding; original-clock correction and regression test added, unchanged feature caches retained under explicit recipe compatibility. Final checks passed; SHuBERT rejects falseWORK but removes3correctFRIEND tight-core commits. No promotion. See REPORT.md/results.json/audit.json.
