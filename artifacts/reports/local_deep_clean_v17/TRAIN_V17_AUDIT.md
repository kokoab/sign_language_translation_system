# v17 Extractor Audit

**Status: PASS**

- Input: `data/local/local_deep_clean_v17/landmarks/train`
- Loaded archives: 13381
- Archives passing all invariants: 13381/13381
- Load/schema errors: 0
- Orientation counts: `{"landscape": 13365, "portrait": 10, "square": 6}`
- Normalization sources: `{"palm_length": 1958, "shoulder_width": 11423}`
- Chirality observations: `{"left": 363335, "right": 338949, "unknown": 0}`

## Metrics

Values are min / median / max across videos.

| Metric | Min / median / max |
| --- | --- |
| `elapsed_seconds` | 0.0390 / 0.5267 / 15.4968 |
| `hand_frame_fraction_before_trim` | 0.1579 / 0.8939 / 1.0000 |
| `hand_frame_fraction_after_trim` | 0.2353 / 0.9524 / 1.0000 |
| `hand_presence_fraction` | 0.1138 / 0.5142 / 1.0000 |
| `face_presence_fraction` | 0.0000 / 0.8438 / 1.0000 |
| `body_presence_fraction` | 0.0000 / 0.6016 / 1.0000 |
| `shoulder_coverage` | 0.0000 / 0.7391 / 1.0000 |

## Enforced invariants

Every archive must load with the current schema fingerprint, have shape `[32, 61, 5]`, contain only finite values, use binary presence values, and keep missing spatial/depth/confidence channels exactly zero.

Per-video measurements: `artifacts/reports/local_deep_clean_v17/train_v17_audit.csv`
