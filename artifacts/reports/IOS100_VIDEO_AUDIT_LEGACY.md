# iOS-100 video audit

**Dataset:** ASL Citizen selective audit subset  
**Videos:** 72 across 12 signs and 29 unique public participant IDs

## Media findings

- Orientation counts: `{"landscape": 72}`
- Resolution counts: `{"640x480": 72}`
- Decode failures: **0**
- Duration seconds, min/median/max: **1.03 / 2.45 / 7.17**
- Decoded frames, min/median/max: **23 / 70.0 / 215**

## Apple Vision extraction

- Requested: **True**
- Python bridge available: **True**
- Body detection interval: **8**
- Coordinate schema: **legacy_anisotropic**
- Status counts: `{"ok": 72}`

The detailed per-video measurements are in `artifacts/reports/ios100_video_audit.csv`.
