# v17 Extractor Audit

**Status: PASS**

- Input: `data/local/citizen100_v17/landmarks`
- Loaded archives: 3101
- Archives passing all invariants: 3101/3101
- Load/schema errors: 0
- Orientation counts: `{"landscape": 3101}`
- Normalization sources: `{"palm_length": 333, "shoulder_width": 2768}`
- Chirality observations: `{"left": 56823, "right": 81899, "unknown": 0}`

## Metrics

Values are min / median / max across videos.

| Metric | Min / median / max |
| --- | --- |
| `elapsed_seconds` | 0.1373 / 0.6607 / 2.9038 |
| `hand_frame_fraction_before_trim` | 0.0417 / 0.4667 / 1.0000 |
| `hand_frame_fraction_after_trim` | 0.2976 / 0.8750 / 1.0000 |
| `hand_presence_fraction` | 0.1295 / 0.4465 / 0.9136 |
| `face_presence_fraction` | 0.0312 / 0.8125 / 1.0000 |
| `body_presence_fraction` | 0.0000 / 0.5312 / 1.0000 |
| `shoulder_coverage` | 0.0000 / 0.6944 / 1.0000 |

## Enforced invariants

Every archive must load with the current schema fingerprint, have shape `[32, 61, 5]`, contain only finite values, use binary presence values, and keep missing spatial/depth/confidence channels exactly zero.

Per-video measurements: `artifacts/reports/citizen100_v17_extractor_audit.csv`
