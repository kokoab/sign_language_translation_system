# YouTube motion pilot compatibility audit

Validated **1411/1411 clips**, 168,531 frames; 0 structural failures. Full frozen subset: True.

Quarantined for manifest/actual frame-count disagreement: **220**. These are provenance discrepancies, not evidence of invalid XY. No padding or invented timestamps.

At least one hand observed: 157,158 frames; both hands: 109,782. Zero coordinate pairs: 0.

**Direct Apple feature substitution is not admitted. Separate-adapter temporal transfer remains conditional, not disproven.**

- Detector confidence is absent from raw XY point pairs; Apple confidence cannot be reconstructed.
- Hand slot names do not verify anatomical chirality or source mirroring without paired visual evidence.
- No validated MediaPipe input adapter into the selected Apple temporal encoder exists in this pilot.
- Exact face semantics and the Apple relative log-scale channel require explicit mapping, not zero padding.
- No frame-rate/timestamp fields found; frame order exists but elapsed-time motion is unverified.

Raw XY geometry, frame ordering and masks support a potential 2D reconstruction experiment. A learned source adapter could isolate detector differences while sharing only temporal weights. Its transfer benefit still needs the matched downstream comparison. No confidence, timing, boundaries or signer IDs were invented.

Per-clip counts, gaps and content hashes: `compatibility_clips.jsonl`. Aggregate evidence and required bridge checks: `compatibility.json`.
