# v17 Extractor Audit

**Status: PASS**

- Input: `data/local/semlex_citizen100_val_audit/landmarks_v17`
- Loaded archives: 978
- Archives passing all invariants: 978/978
- Load/schema errors: 0
- Orientation counts: `{"landscape": 978}`
- Normalization sources: `{"palm_length": 133, "shoulder_width": 845}`
- Chirality observations: `{"left": 14693, "right": 23481, "unknown": 0}`

## Metrics

Values are min / median / max across videos.

| Metric | Min / median / max |
| --- | --- |
| `elapsed_seconds` | 0.1094 / 0.3488 / 0.8900 |
| `hand_frame_fraction_before_trim` | 0.1042 / 0.5510 / 1.0000 |
| `hand_frame_fraction_after_trim` | 0.3600 / 0.8621 / 1.0000 |
| `hand_presence_fraction` | 0.1428 / 0.5696 / 0.8979 |
| `face_presence_fraction` | 0.0312 / 0.7812 / 1.0000 |
| `body_presence_fraction` | 0.0000 / 0.4453 / 1.0000 |
| `shoulder_coverage` | 0.0000 / 0.6296 / 1.0000 |

## Enforced invariants

Every archive must load with the current schema fingerprint, have shape `[32, 61, 5]`, contain only finite values, use binary presence values, and keep missing spatial/depth/confidence channels exactly zero.

Per-video measurements: `artifacts/reports/semlex_citizen100_val_audit/v17_audit.csv`
