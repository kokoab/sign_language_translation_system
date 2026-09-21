# Annotation identities and OTHER development diagnostic — 2026-09-21

Continued steps3–4 using existing local data only. No acquisition, training, checkpoint
selection/promotion, source-label rewrite or protected-test access occurred.

## Step3: separate lexical evidence from unmatched text

The new audit script traces ASLLRP source occurrences into all1,104existing target+OTHER
crop archives. Each occurrence retains original variant, occurrence label, source
annotation ID, signer, inclusive-to-exclusive crop timing and official ASL-LEX identity.
The ledger contains9,104crop-associated occurrences, not9,104independent source signs.

Classification requires the existing exact occurrence convention and an unambiguous
official SignBankAnnotationID→ASL-LEX code link. Locked identities are known. OOV
requires a lexical sign with a documented code and lemma outside the locked vocabulary.
Unsupported aliases, ambiguous/missing links and unverified nonlexical categories are
unresolved, not negative training truth. Known-class variant families are excluded
from confirmed OOV. These are provenance decisions, not a new visual review.

| Existing ASLLRP crop occurrences | Train | Validation |
|---|---:|---:|
| Confirmed known identity | 1,126 | 271 |
| Confirmed OOV identity | 1,546 | 298 |
| Unresolved mapping/category | 4,594 | 1,269 |

Of5,863unresolved occurrences,5,228lack an unambiguous official link,604differ from
the supported occurrence convention, and31have nonlexical categories not admitted
as lexical OOV. Missing official coverage does not prove bad annotation or that the
sign is actually known. With no fluent reviewer, unresolved evidence is preserved
without converting uncertainty into OTHER or blank labels.

`clip_admission.json` records all1,858current phrase archives. Full-sequence eligibility
requires complete caches plus established mappings; ASLLRP OTHER also requires every
intersecting event to be complete/nonoverlapping and reconstruct the stored targets.
This leaves240known local/contiguous training archives and191validation archives, plus
only6ASLLRP OTHER training sequences under the stricter whole-sequence rule. The small
count is an admission constraint, not a recommendation to train on this reduced set.
Trusted event cores can remain usable inside otherwise unresolved sequences.

Corrected local labels retain the user's existing review provenance. ASLLRP contiguous
retains its established exact-variant contract. Flores and NCSLGR have no newly established
cross-corpus identity link here and are unresolved for this strict admission proposal.
No tokens were deleted from a transcript while retaining their corresponding video.
The sidecar decisions are not installed into active training manifests.

## Step4: a useful diagnostic, not an independent unseen-OOV benchmark

Prepared524unique presegmented validation cores from the existing JONATHAN signer:
248known and276OOV across97OOV identities. Each comes from a complete cache and complete,
nonoverlapping annotation with at least2hand-observed frames and80%hand-frame presence.
Core intervals preserve source timing; no annotation-gap background labels are created.
The presence checks are quality screens, not semantic or boundary re-annotation.

259OOV cores have identities present in ASLLRP training. The remaining17lack ASLLRP-train
exposure under the checked identity mapping, but exposure through other training sources
and checkpoint ancestry is unresolved. None is certified globally unseen. Existing
validation has been reused for development, so none is a fresh test. Only one validation
signer is represented. Verified rest and mixed continuous-stream evaluation remain
separate missing parts of a complete UNKNOWN benchmark.

The evaluator ran the four existing Flores comparison checkpoints with their verified
shared Stage1 ancestor, identical stride4/window8 core inputs, CPU greedy CTC collapse,
and no decoder tuning. Blank is removed; OTHER remains visible for scoring.

| Existing checkpoint | Known exact /248 | OOV exactly OTHER /276 | OOV blank-only /276 | OOV emits any known sign /276 |
|---|---:|---:|---:|---:|
| without Flores,17321 | 84 | 40 | 211 | 25 |
| with Flores,17321 | 85 | 32 | 216 | 28 |
| without Flores,17322 | 82 | 1 | 253 | 22 |
| with Flores,17322 | 72 | 34 | 226 | 16 |

Any OTHER emission on known cores is30,18,0,17respectively; OTHER without the expected
known sign is30,17,0,17. An OOV core emitting OTHER plus a known sign is not exact OTHER;
blank-only output is neither a correct sign nor a correct UNKNOWN classification.
Per-core predictions, source-cache hashes, checkpoint hashes and manifest hash are in
`core_evaluation.json`. Counts were recomputed from saved predictions.

These short, presegmented inputs differ from the full connected sequences used to train
the temporal heads. The results characterize this core protocol, not live OOV recall,
whole-stream boundary detection, frozen Stage1 top1 or general unseen-sign performance.
They do not establish that adding Flores consistently improves OTHER recognition:
the directions differ across seeds, and blank-only responses dominate all four runs.

## Recommendation

Do not start step5 yet. Recover the159incomplete caches from existing raw video into a
new cache version, preserving annotations and time. Build trusted known/OOV core
supervision from the ledger rather than discarding whole corpora or labeling uncertain
gaps as blank. Retain unresolved full-sequence rows separately until established
mapping evidence exists; no new reviewer or external dataset is assumed.

The prepared manifest/evaluator make OTHER measurable now as a reused-development core
diagnostic. A truly independent unseen-OOV evaluation still needs verified non-exposure,
broader held-out signer coverage, and rest/mixed-stream truth. Do not rename this
diagnostic to imply those properties. Freeze those evaluation contracts before any
new model run; the current result does not justify architecture changes or training.

## Reproduction and verification

- `venv/bin/python scripts/audit_annotation_identities_v17.py`
- `venv/bin/python scripts/evaluate_annotation_cores_v17.py`
- `venv/bin/python -m unittest test.test_annotation_identity_audit_v17 test.test_annotation_core_metrics_v17 -v`

36focused tests passed including previous contract/extraction/CTC checks. New identity
and metric tests were first run before the modules existed, then passed after
implementation. Verified524unique parent-annotation pairs, four complete prediction
sets, recomputed metric counts and allowed token range. `git diff --check` passed.

`event_ledger.json` is over1MB: query selected fields/aggregates with jq; do not print
it whole. Its entry is recorded in `artifacts/LARGE_FILES.md`. Raw datasets, training
manifests and all checkpoint files remain unchanged.
