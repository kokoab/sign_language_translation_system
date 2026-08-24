# v17 Extractor Audit

**Status: PASS**

- Input: `data/local/local_deep_clean_v17/landmarks/val`
- Loaded archives: 2896
- Archives passing all invariants: 2896/2896
- Load/schema errors: 0
- Orientation counts: `{"landscape": 2893, "portrait": 2, "square": 1}`
- Normalization sources: `{"palm_length": 425, "shoulder_width": 2471}`
- Chirality observations: `{"left": 78983, "right": 72493, "unknown": 0}`

## Metrics

Values are min / median / max across videos.

| Metric | Min / median / max |
| --- | --- |
| `elapsed_seconds` | 0.1273 / 0.9523 / 3.2673 |
| `hand_frame_fraction_before_trim` | 0.2237 / 0.8983 / 1.0000 |
| `hand_frame_fraction_after_trim` | 0.2807 / 0.9524 / 1.0000 |
| `hand_presence_fraction` | 0.0997 / 0.5713 / 1.0000 |
| `face_presence_fraction` | 0.0000 / 0.8438 / 1.0000 |
| `body_presence_fraction` | 0.0000 / 0.5938 / 1.0000 |
| `shoulder_coverage` | 0.0000 / 0.7391 / 1.0000 |

## Enforced invariants

Every archive must load with the current schema fingerprint, have shape `[32, 61, 5]`, contain only finite values, use binary presence values, and keep missing spatial/depth/confidence channels exactly zero.

Per-video measurements: `artifacts/reports/local_deep_clean_v17/val_v17_audit.csv`
