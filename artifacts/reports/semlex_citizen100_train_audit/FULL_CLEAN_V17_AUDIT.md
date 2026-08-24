# v17 Extractor Audit

**Status: PASS**

- Input: `data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17`
- Loaded archives: 1388
- Archives passing all invariants: 1388/1388
- Load/schema errors: 0
- Orientation counts: `{"landscape": 1388}`
- Normalization sources: `{"palm_length": 188, "shoulder_width": 1200}`
- Chirality observations: `{"left": 21028, "right": 34407, "unknown": 0}`

## Metrics

Values are min / median / max across videos.

| Metric | Min / median / max |
| --- | --- |
| `elapsed_seconds` | 0.0884 / 0.3326 / 0.9290 |
| `hand_frame_fraction_before_trim` | 0.1562 / 0.5556 / 1.0000 |
| `hand_frame_fraction_after_trim` | 0.7000 / 0.8621 / 1.0000 |
| `hand_presence_fraction` | 0.3052 / 0.5667 / 0.9644 |
| `face_presence_fraction` | 0.5000 / 0.7812 / 1.0000 |
| `body_presence_fraction` | 0.0000 / 0.4453 / 1.0000 |
| `shoulder_coverage` | 0.0000 / 0.6429 / 1.0000 |

## Enforced invariants

Every archive must load with the current schema fingerprint, have shape `[32, 61, 5]`, contain only finite values, use binary presence values, and keep missing spatial/depth/confidence channels exactly zero.

Per-video measurements: `artifacts/reports/semlex_citizen100_train_audit/full_clean_v17_audit.csv`
