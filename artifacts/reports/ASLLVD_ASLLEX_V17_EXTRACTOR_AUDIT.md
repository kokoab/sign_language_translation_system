# v17 Extractor Audit

**Status: PASS**

- Input: `data/local/asllvd_asllex_v17/landmarks`
- Loaded archives: 175
- Archives passing all invariants: 175/175
- Load/schema errors: 0
- Orientation counts: `{"landscape": 1, "portrait": 173, "square": 1}`
- Normalization sources: `{"palm_length": 15, "shoulder_width": 160}`
- Chirality observations: `{"left": 5786, "right": 6218, "unknown": 0}`

## Metrics

Values are min / median / max across videos.

| Metric | Min / median / max |
| --- | --- |
| `elapsed_seconds` | 0.1900 / 0.6397 / 1.6159 |
| `hand_frame_fraction_before_trim` | 0.9365 / 1.0000 / 1.0000 |
| `hand_frame_fraction_after_trim` | 0.9365 / 1.0000 / 1.0000 |
| `hand_presence_fraction` | 0.5000 / 1.0000 / 1.0000 |
| `face_presence_fraction` | 0.0312 / 0.8750 / 1.0000 |
| `body_presence_fraction` | 0.0000 / 0.8125 / 1.0000 |
| `shoulder_coverage` | 0.0000 / 0.8462 / 1.0000 |

## Enforced invariants

Every archive must load with the current schema fingerprint, have shape `[32, 61, 5]`, contain only finite values, use binary presence values, and keep missing spatial/depth/confidence channels exactly zero.

Per-video measurements: `artifacts/reports/asllvd_asllex_v17_extractor_audit.csv`
