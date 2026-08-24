# Citizen100 raw dataset audit

**Status: PASS**

- Provenance rows: 3102
- First-frame decode success: 3102/3102
- Classes: 100
- Videos by split: `{"test": 1248, "train": 1476, "val": 378}`
- Unique participants by split: `{"train": 32, "val": 5, "test": 11}`
- Cross-split participant overlap: `{"train_test": [], "train_val": [], "val_test": []}`
- Decoded orientation: `{"landscape": 3102}`
- Decoded dimensions: `{"640x480": 2982, "960x540": 120}`
- Errors: 0

Files were already checked against the official ZIP member size and CRC during
download. This audit independently enforces manifest counts, first-frame decode,
and signer-disjoint split membership.
