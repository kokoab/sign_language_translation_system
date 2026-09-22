# Combined dataset — 2026-09-22

Created `data/local/combined_dataset_v17_20260922/manifest.json`.

**6,421 records: 4,547 training / 1,874 validation.** Includes all five finalized
supplement lists and all 494 approved baseline records, including 431 local phrases.
The manifest references existing files; it does not copy or alter source datasets.

| Source | Training | Validation |
|---|---:|---:|
| asllrp_contiguous | 44 | 12 |
| asllrp_other_ctc | 6 | 0 |
| asllrp_segmented | 1116 | 254 |
| asllrp_verified_span | 1 | 0 |
| citizen | 1475 | 378 |
| local_phrases | 232 | 199 |
| o5s5 | 195 | 57 |
| semlex | 1388 | 953 |
| stem | 90 | 21 |

Each record has its original split, feature/raw-video hashes, explicit representation,
label sequence and nonblank CTC target indices. Blank remains 0; approved OTHER is 101.
The manifest pins all six source manifests and the locked vocabulary. The 139 known
ASLLRP records sharing baseline parent utterances remain marked as correlated inputs.
Signer overlap does not exclude records. Existing exclusions remain excluded.

Verification reopened the saved manifest and loaded all 6,421 records / 7,367 windows;
feature/raw hashes, source/evidence hashes, target mappings and array contracts passed.
The positive-sign hand floor does not apply to every window of a full phrase: approved
natural rest/transition windows are preserved while structural checks still apply.
The initial combined check caught that distinction; the loader was corrected, then the
full build/verification passed. The focused feature-contract test also passed.

Builder and representation-aware loader: `scripts/build_combined_dataset_v17.py`.
`load_features(row, label_to_index)` returns windows plus CTC targets. Windowed records
retain source-frame ranges in their archives for the eventual temporal training recipe;
32-frame isolated features and timestamp-normalized O5S5 cores have explicit distinct
representations. These are mixed types of supervision, not 6,421 full phrases.

Data assembly is complete. No training was started and no old trainer's gate was bypassed.
The next training runner must consume this manifest, preserve validation roles, mix only
training records, and define temporal resampling, source weights and evaluation metrics.
Reusing a prior trainer unchanged would not automatically use this new manifest.

Manifest SHA-256: `089da8b13943b65663691cc7e23c8bd1dcb279fceca56ab1675fa768d19c95f2`.
