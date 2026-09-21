# Existing-data contract repair — 2026-09-21

User approved steps1–2 only: timing/padding repair and consistent loader validation.
All acquisition remains stopped. No training, raw-data/cache rewrite or test-set access.

## Changes

- `active/v17/train_unified_streaming_aligned_grounded_v17.py`: NCSLGR alignment
  endpoints now use actual loaded source-frame counts. Reject invalid frame counts
  or events outside the available evidence. The collator requires alignment length
  to equal each sample's evidence length and leaves all padding at ignore index−100.
- `active/v17/train_unified_streaming_ctc_v17.py`: shared phrase loading validates
  embedded Apple landmark fingerprint, tensor shape/finiteness, integral contiguous
  non-overlapping ranges starting at zero, complete source-frame coverage, integral
  targets and stored target-name/index consistency with the frozen100mapping. Both
  training entry points pass checkpoint labels for exact frozen-vocabulary comparison.
  CTC feasibility includes adjacent-repeat blank requirements. Frame-level encoding
  is checked against actual encoded steps rather than assuming one step per window.
- `test/test_phrase_data_contract_v17.py`: runnable regression checks for malformed
  caches, cut events, actual source-frame alignment, checkpoint vocabulary mismatch,
  correct multimodal fingerprint interpretation, CTC feasibility, and zero auxiliary
  loss gradient on padding. Existing tests retained.

This repair does not normalize raw gloss aliases, prove OOV semantics, change the
extractor or expand supervision. NCSLGR's existing unsupported names remain mapped
as stored; semantic mapping review belongs to step3. Entirely zero feature windows
remain legitimate masked data structurally, not automatic labels for rest/OTHER.

## Fresh validation

Each of1,858existing phrase archives was passed through the actual shared loader via
a temporary per-file symlink. Original files were not changed. Machine evidence and
every blocked path are in `validation.json`.

| Source | Train passing / checked | Validation passing / checked |
|---|---:|---:|
| Corrected local | 208 / 232 | 179 / 200 |
| ASLLRP contiguous | 32 / 44 | 12 / 12 |
| ASLLRP OTHER | 807 / 879 | 207 / 225 |
| NCSLGR | 80 / 88 | 33 / 37 |
| Flores OTHER | 141 / 141 | — |

Total1,699pass;159fail solely for incomplete coverage, matching the audit's147explicit
tail omissions plus12NCSLGR unrecorded omissions. These are admission failures, not
159proven bad transcripts. All113complete NCSLGR inputs produce valid alignments.
The synthetic96-frame/98-frame-metadata regression proves alignment uses available
evidence; an event extending beyond available evidence is rejected. The collator
regression proves padded logits have zero auxiliary gradient and mismatched alignment
lengths fail explicitly.

34focused tests passed across phrase contracts, unified CTC, Flores mapping, multimodal
extraction, ASLLRP preparation, streaming prefix and Stage1 emission tests. The new
regressions first exposed10acceptance failures plus the missing source-count API before
implementation. Final tests include a real small Stage1 encoder check for frame/window
CTC feasibility. No model training or accuracy evaluation ran. `git diff --check` passed.

## Operational consequence and recommendation

Existing training commands against these unchanged mixed roots now stop with a
path-specific error at the first incomplete cache. They do not silently train on a
smaller subset. Reproducing historical experiments requires their historical code;
do not mix repaired and original arms under the same experiment identity.

Proceed to step3: separate established known/OOV identities from unresolved mappings
using existing provenance. With no reviewer, unresolved full-sequence annotations
stay out of supervised CTC; do not simply delete their tokens while retaining signing.
Recover the159affected caches from already-local video into a new cache version where
possible. Otherwise explicitly document exclusions and recheck coverage/signer splits.
Do not rewrite metadata to falsely declare an incomplete sequence complete.

Proceed to step4 after that mapping inventory identifies enough confirmed OOV examples
already on disk. Freeze identity/signer-disjoint roles and audit exposure across prior
training sources; if an identity cannot be established as unseen, label that evaluation
as seen-OOV. Measure OOV recall/false-known emissions and known false rejection separately.
No new downloads are needed or authorized for this next step.

Do not start step5 yet. First resolve admission failures, freeze a consistent dataset
and establish the OOV evaluation. Then run a matched repair-only comparison before
changing supervision. Passing structural checks is not evidence of improved accuracy.
