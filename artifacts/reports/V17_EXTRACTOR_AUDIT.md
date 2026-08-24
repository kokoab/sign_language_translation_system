# v17 Extractor Audit

**Status: PASS**

- Input: `data/local/local_citizen100_quality_audit_q82_cap14_exact/landmarks`
- Loaded archives: 1021
- Archives passing all invariants: 1021/1021
- Load/schema errors: 0
- Orientation counts: `{"landscape": 1021}`
- Normalization sources: `{"palm_length": 173, "shoulder_width": 848}`
- Chirality observations: `{"left": 24886, "right": 21608, "unknown": 0}`

## Metrics

Values are min / median / max across videos.

| Metric | Min / median / max |
| --- | --- |
| `elapsed_seconds` | 0.0738 / 0.2322 / 0.9487 |
| `hand_frame_fraction_before_trim` | 0.2800 / 0.8966 / 1.0000 |
| `hand_frame_fraction_after_trim` | 0.4231 / 0.9535 / 1.0000 |
| `hand_presence_fraction` | 0.2188 / 0.6709 / 1.0000 |
| `face_presence_fraction` | 0.0000 / 0.8125 / 1.0000 |
| `body_presence_fraction` | 0.0000 / 0.5156 / 1.0000 |
| `shoulder_coverage` | 0.0000 / 0.6721 / 1.0000 |

## Enforced invariants

Every archive must load with the current schema fingerprint, have shape `[32, 61, 5]`, contain only finite values, use binary presence values, and keep missing spatial/depth/confidence channels exactly zero.

Per-video measurements: `artifacts/reports/v17_extractor_audit.csv`
