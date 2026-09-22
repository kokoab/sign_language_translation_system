# Finalized existing supplements — 2026-09-22

**5,927 accepted supplemental records: 4,264 training / 1,663 validation.**
These are supervised isolated signs, segmented signs, and verified positive intervals;
they are not 5,927 multi-sign phrases. The existing 494 phrase/subspan baseline is separate.
Together the input lists contain **6,421 records: 4,547 train / 1,874 validation**.
This is a record count, not a count of independent sign events, signers, or full narratives.

| Dataset | Train | Validation | Manifest |
|---|---:|---:|---|
| Citizen | 1,475 | 378 | [citizen.json](citizen.json) |
| SemLex | 1,388 | 953 | [semlex.json](semlex.json) |
| ASLLRP segmented | 1,116 | 254 | [asllrp_segmented.json](asllrp_segmented.json) |
| Human-reviewed STEM | 90 | 21 | [stem.json](stem.json) |
| O5S5 exact positive cores | 195 | 57 | [o5s5.json](o5s5.json) |

## What is finalized

Each manifest lists exact feature paths, feature and raw-video SHA-256 hashes, original
split, exact locked ASL-LEX label/code, zero-based class index, CTC target (class+1),
representation, signer information when available, and pinned annotation evidence.
Existing Citizen/SemLex/ASLLRP/STEM files stay in their original locations. Three repaired
landmark-only caches and 252 O5S5 core files are under
`data/local/finalized_supplements_v17_20260922/`. Originals were not modified or deleted.

The three recovered tails are one ASLLRP training clip and STEM P35:077/P6:103.
Repairs use hash-verified source video and the original frame timeline; STEM uses only
the human-verified interval. Final windows cover every source frame and pass the existing
phrase loader contract and observed-hand checks. They contain landmarks only, not RGB.

O5S5 cores contain only exact matched, annotated positive intervals, normalized from
actual timestamped Apple observations. Four very short training events have fewer than
two observations in the existing cache and remain excluded from this representation.
They are not mislabeled or deleted. No unannotated gap or unresolved sign becomes blank/OTHER.

## Exclusions and evaluation

The exclusions ledger records 36 candidates: one previously rejected incomplete Citizen
SLEEP clip; six planned SemLex validation files absent from the feature cache; 25 SemLex
validation copies whose raw bytes exactly duplicate retained training videos; four O5S5
intervals too short for this cached core representation. No samples were rejected because
of missing or overlapping signer IDs. Existing roles are preserved.

951 retained SemLex validation records share a source-local signer ID with training;
2 do not. Report SemLex results as familiar-signer diagnostics. This does not prevent
training on the admitted training records. Do not infer cross-corpus identity from ID strings.
SemLex's old candidate manifest has a historical `training_eligible=false` flag; its
explicit approval and exact manifest hash are established by the frozen Stage1 training
provenance. That original file remains unchanged.

## Verification and next step

`verification.json` records independent manifest/evidence/raw-video/cache hash checks,
exact duplicate checks, label-index agreement, valid features, and a frozen Stage1
MPS forward over every admitted window. No optimizer steps or accuracy evaluation.
There are no exact raw-video hash matches against the 494 baseline. ASLLRP segmented
clips can share source utterances/events with baseline subspans: shared-parent metadata
is recorded by the verifier: 139 supplemental records share a baseline parent, all in
the same role. This should inform sampling; these are not independent videos.

Run:

```
venv/bin/python -m unittest test.test_finalize_supplements_v17 -v
venv/bin/python scripts/verify_finalized_supplements_v17.py
```

The data is structurally and annotation-contract usable; successful learning/generalization
must be measured by the combined experiment. Build the combined manifest from these five
pinned lists plus `active/v17/approved_phrase_manifest_20260921_v2.json`, preserving roles
and explicitly handling the three feature representations. Include the baseline local
phrases. Keep mixed-signer metrics honestly labeled. A generic directory scan must not
reintroduce exclusions, stale RGB, or count validation records as training.

No combined training manifest or training run was created here. Flores/NCSLGR unresolved
lexical mappings, local isolated auxiliary data, ASLLVD, and motion-only datasets retain
their separate statuses in the broader reconciliation report; they are not silently
admitted by this finalization. Signer uncertainty alone is not an exclusion gate.
