# Existing-data tail recovery and OOV exposure audit — 2026-09-21

Recovered all 159 incomplete phrase caches using existing, hash-verified recordings.
Extraction finished in 64.5 seconds. No acquisition, training, protected-test access,
checkpoint selection or runtime change. Original archives and videos are unchanged.

## Recovery

New root: `data/local/phrase_tail_recovery_v17_20260921/`, containing the same three
source-root names and train/validation layouts. It contains 159 rebuilt landmark-only
archives and 1,699 symlinks to unchanged originals. This is a local versioned view,
not a portable copy; the original roots must remain available. Active training defaults
still point to the original roots and therefore still reject their incomplete archives.

Each affected archive lacked 1–3 frames (315 frames in total). A new tiny tail alone
would fail the extractor's four-frame minimum. Recovery instead divides the last full
32-frame window plus its tail into two 16–18-frame windows, then uses the existing
Apple Vision extractor and its 32-step output representation. This preserves every
source-frame interval without lowering detection thresholds, inventing tail observations,
or changing earlier windows. Original orientation and zero-lip policy are preserved.
Window normalization/resampling is recomputed for the two affected windows, so this is
new evidence and must not be described as numerically equivalent to the old cache.

Rebuilt files explicitly contain landmarks, ranges and targets only; RGB data is not
regenerated. The old RGB container metadata is replaced by the actual landmark schema.
Each rebuilt archive records source hash, old ranges and recovery policy. The script
refuses an existing destination, and verifies original video hashes and decoded frame
counts before recovery.

Verification (`verification.json`):

- All 1,858 archives pass the shared schema/vocabulary/full-coverage validator and
  CTC feasibility for the existing stride-4, window-8 protocol.
- All 125 NCSLGR alignments match the available timeline.
- Every source archive hash is unchanged. All repaired targets, roles, signer IDs,
  video hashes and source identities are unchanged. Unaffected windows are bitwise equal.
- One of 318 rebuilt windows had insufficient hand detections and retains explicit
  zero/missing evidence and a diagnostic flag. It belongs to local validation archive
  `2cdf041902b3.grounded_streaming_v17.npz`; this is not an annotation of rest or blank.
  Keep that quality flag visible during admission/evaluation review.
- 37 focused tests pass, including a new test of complete, nonoverlapping recovery
  ranges and rejection of unsupported cache layouts. `git diff --check` passes.

`inventory.json` records all source/output paths and hashes; `summary.json` records
counts and extraction duration. No labels were repaired or semantically certified by
this operation. Historical identity/core reports still refer to the original caches.

## Exposure audit

Evidence: `../annotation_identity_audit_v17_20260921/exposure_audit.json`.
The audit traces the four previously evaluated heads to their recorded Stage 1 base
and initial heads, verifies their hashes and the base's Citizen/SemLex manifest hashes,
and examines the admitted 88 NCSLGR and 141 Flores training archives. Stage 1 provenance
records locked-vocabulary Citizen/SemLex supervised training; initial heads contain
state/config only, so their standalone files do not prove a full training lineage.

Of 276 development OOV cores across 97 identities, 259 have documented ASLLRP-training
identity exposure. The remaining 17 cores cover 13 identities and remain unresolved.
Flores contains exact raw-label candidates for EYES, NEAR and BOTH (four of those cores),
but string equality is not an established cross-corpus identity link. These candidates
apply to the with-Flores arm only. No exact-string match cannot establish absence of an
identity, given aliases, unresolved annotations, incidental signs and blank examples.

**Zero globally unseen identities are certified.** All cores also remain reused
validation material from one signer. This audit completes the available-evidence check;
it does not complete an independent unseen-OOV benchmark, and no guessed mapping is
introduced to make one appear complete.

## Recommendation

The missing-tail blocker is resolved in the new cache version. Do not start step 5 yet:
semantic annotation admission, genuinely unseen OOV evaluation, and rest/mixed-stream
truth remain unresolved. Next prepare supervision from only the already established
known/OOV event identities, retain unresolved events as excluded supervision rather than
OTHER, and explicitly keep seen-OOV development metrics separate from any future
identity-disjoint evaluation. A fresh training comparison must use the new caches in
both arms and cannot reuse old numerical results as a matched baseline.

Reproduce with `venv/bin/python scripts/recover_phrase_tails_v17.py` only into a fresh
versioned destination; it intentionally refuses the completed output. Exposure audit:
`venv/bin/python scripts/audit_oov_exposure_v17.py`.
