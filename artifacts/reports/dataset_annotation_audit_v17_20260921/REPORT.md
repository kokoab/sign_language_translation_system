# v17 dataset annotation and training-path audit — 2026-09-21

The available data support bounded recognition experiments, but do not yet establish
reliable recognition of arbitrary connected signing or genuinely unseen OTHER signs.
The highest-priority findings concern supervision and evaluation contracts, not globally
broken source videos. No pipeline, dataset, checkpoint or experiment was changed.

## Scope and evidence

Fresh checks covered all 1,858 train/validation phrase archives in the corrected grounded
root, ASLLRP OTHER root and Flores OTHER root. See `evidence.json`. Inspected the current
Flores runner, aligned trainer, shared loader/collator, extraction and preparation paths.
Other local corpora below are assessed from their existing acquisition/admission reports
and relevant ground-truth entries; they were not all re-extracted or visually reviewed.
No protected test data, raw-video semantic re-review, training or download was performed.
The running Flores experiment was not polled; this audit makes no claim about its result.

All checked arrays were finite and correctly shaped, target indices were in range,
stored target names matched indices, and CTC lengths were feasible. No source-item ID
or exact stored video-hash overlap crossed train/validation in these three roots.
This is not a global identity, near-duplicate, semantic-label or timestamp certification.
All embedded landmark fingerprints matched `b872fa3dcc16aab5`; differing enclosing
multimodal fingerprints reflect different containers/configurations, not Apple/MediaPipe mixing.

## Findings

1. **OTHER detection is not established by the current evaluation.**
   `scripts/train_flores_other_v17.py:165` evaluates OTHER emissions on known Citizen
   and SemLex isolated signs. That measures erroneous rejection of known signs, not
   recall/precision on genuinely out-of-vocabulary signing. Shared evaluation removes
   OTHER before known-WER calculation; full-sequence exact accuracy does retain OTHER,
   but the checkpoint-selection score uses known WER/blank false emissions and has no
   explicit unseen-OOV detection gate. A missing OTHER between two correctly recognized
   known signs can therefore leave known WER unchanged. None of the checked validation
   sequences is OTHER-only; training has only one OTHER-only Flores sequence.
   Needed: known signs, independently held-out OOV identities, mixed sequences and
   verified rest/nonsign examples, with OTHER precision/recall and known false rejection
   reported separately. Blank/no signing and OOV signing are different truths.

2. **NCSLGR alignment can supervise padding. Confirmed current-path defect.**
   Twelve NCSLGR archives (8 train, 4 validation) omit 1–3 frames relative to
   `sampled_source_frames`, without a dropped-tail field. Alignments use the larger
   manifest frame count, giving one more target step than the actual rolling evidence.
   `collate_aligned` bounds copied labels by batch width, not each sample's length;
   with a longer batch peer, the eight training cases receive one blank CE target on
   padded input. The auxiliary CE loss does not mask that position. Reproduced against
   the actual collator. No annotated event extends beyond these cached ends, and this
   audit does not attribute overall model failure to this small defect. CTC itself uses
   actual input lengths. Align evidence and annotation clocks and mask auxiliary labels
   to sample length before the next changed-data experiment.

3. **Completeness admission differs across corpora.**
   Flores preparation excludes 14/155 clips with missing final 1–3 frames. Current
   non-Flores archives retain 147 clips with explicitly recorded dropped tails:
   12 ASLLRP contiguous train; 24 local train; 21 local validation; 72 ASLLRP OTHER train;
   18 ASLLRP OTHER validation (293 omitted frames total). The 12 NCSLGR cases above
   are additional, bringing the observed incomplete-cache count to 159/1,858.
   These omissions follow the extractor's minimum-four-frame-tail rule; they are not
   proof of corrupt raw videos or cut-off lexical signs. Evaluate whether each tail
   intersects annotated signing before deciding to recover or exclude it. Apply one
   explicit admission policy; do not automatically discard all affected clips.

4. **Exact text matching is not the same as verified lexical identity.**
   ASLLRP uses the frozen ASL-LEX/SignBank variant mapping. Flores uses uppercased
   whitespace tokens with outer punctuation stripped; NCSLGR uses exact canonical
   strings. Both are conservative text mappings, but a known sign under an alternate
   notation can become OTHER and an identical label can still refer to a different
   visual variant. Neither possibility is a newly proven semantic error in a specific
   clip. Review mappings against source conventions and pixels before claiming that
   every OTHER target is genuinely outside the 100. Preserve raw glosses and distinguish
   confirmed OOV, unresolved mapping, and unannotated spans in audit metadata.

5. **The shared phrase loader trusts preparation too much.**
   `phrase_sequences` checks role but does not enforce landmark fingerprints, vocabulary
   identity, full coverage, range continuity or overlap compatibility at load time.
   A disposable wrong-fingerprint archive was accepted. The actual shared restoration
   function concatenates ranges: overlapping [0,32), [8,40) become 64 frames rather
   than 40. All 1,858 current archives have contiguous non-overlapping ranges and a
   compatible embedded schema, so this is a demonstrated admission weakness, not
   evidence that current inputs mix extractors. Reject incompatible inputs before
   consuming future caches; do not build an overlapping-cache conversion unless needed.

6. **Structural validity does not mean every frame contains usable signing evidence.**
   Completely zero windows occur in local train (9/438), local validation (61/853),
   and Flores train (32/2,471). They may be real rest or failed detection; no semantic
   conclusion is possible from zeros alone. The shared loader uses the feature masks
   but does not explicitly consume the saved window-validity summaries. Review their
   position against signing before any exclusion. No all-zero windows were found in
   the checked ASLLRP/NCSLGR caches.

7. **Coverage and evaluation remain narrow despite large window counts.**
   Corrected local training covers 13 glosses; validation covers 15. Its two training
   signers and one validation signer support a useful held-signer check, not 100-sign
   continuous generalization. No GOOD_MORNING phrase training examples remain, while
   20 validation examples remain. Multiple windows/crops of a recording are not new
   signers or independent examples. Flores signer IDs are not global identities;
   cross-corpus signer separation is unresolved. Repeatedly reused development sets
   are not a fresh test. The official Citizen test remains consumed and sealed.

## Current admitted phrase inputs — freshly counted

| Source | Train clips | Validation clips | Known classes train / validation | OTHER spans train / validation |
|---|---:|---:|---:|---:|
| Corrected local phrases | 232 | 200 | 13 / 15 | 0 / 0 |
| ASLLRP contiguous | 44 | 12 | 35 / 12 | 0 / 0 |
| ASLLRP target + OTHER | 879 | 225 | 53 / 36 | 1705 / 398 |
| NCSLGR strict | 88 | 37 | 14 / 7 | 157 / 71 |
| Flores OTHER supplement | 141 | 0 | 93 / — | 462 / — |

ASLLRP contiguous and OTHER rows can derive from shared parent material; this table
counts archives, not independent source videos. It excludes isolated replay and blank
samples. Flores has 476 known-token occurrences; consecutive unsupported glosses are
collapsed to one OTHER span. That teaches an OOV region, not one output per unknown sign.
Known adjacent repeats remain distinct CTC targets.

## Other available local sources — suitability from existing evidence

| Source | Useful role | Constraint |
|---|---|---|
| Citizen and SemLex | Isolated identity learning/retention | Not natural connected-sentence supervision or an OOV benchmark by themselves |
| ASL STEM Wiki | 111 reviewed bounded spans, 31 glosses, 18 participants | Approved cores are useful; the 268-video candidate pool is not wholly admitted; only 18 glosses have at least three approved participants |
| O5S5 | 256 exact mapped positives, 53 classes, six signers; LG held out | Positive cores only: annotation completeness is unproven, so gaps are not blank and full-narrative WER is invalid |
| MoLo | Two signer crops with 1,517 timed hand annotations | Partial transcript and exact-variant review; deduplicate simultaneous hand tiers; not full-sequence negative supervision |
| Cokely | 31 prepared spans, four signers, 97 tokens | All remain training-ineligible pending visual-variant admission; DIFFERENT is 50/97 tokens |
| RWTH-BOSTON-104 | Small low-resolution sequence auxiliary | Official splits reuse three signers; cannot establish signer-disjoint performance unchanged |
| RIT Homework sample | One 6.5-second paired example | Sanity check only; orientation question remains in acquisition report; full corpus access is restricted |
| Daily Moth sample | Selective first-person-reference annotations | Not complete gloss transcripts |
| CLERC Épée | 1,200 timed MediaPipe sequences, six source IDs | No raw video; not compatible with frozen Apple features; separate temporal research only |
| How2Sign / OpenASL / YouTube-ASL | Caption-paired data or motion pretraining | Captions are not verified gloss sequences/boundaries; YouTube pilot retained 1,411 files, 220 frame-count disagreements quarantined; do not resume paused acquisition |
| Synthetic / generated phrases | Training-only research | No validation/model-selection truth; rejected synthetic motion remains excluded |
| PopSign / SoMe | Existing restricted-purpose holdings | PopSign is one-handed; SoMe is excluded by user review; neither fills general two-handed sentence supervision |

Sources: current `PROJECT_GROUND_TRUTH.md`; data-sources and stage2-ctc logs; reports
`local_phrases_fixed_audit_20260921`, `stage2_data_path_audit_20260921`,
`continuous_asl_acquisition_20260912`, `open_asl_alternatives_20260913`,
`o5s5_citizen100_v17`, `continuous_asl_gap_search_20260921`, and the Flores preparation audit.
Old reports can contain superseded counts; the fresh table above governs this snapshot.

## External-source check and next discussion

A focused public-source search did not verify a new, immediately admissible corpus beyond
the sources already in the logs. Microsoft's [STEM page](https://www.microsoft.com/en-us/research/project/asl-stem-wiki/)
provides a direct dataset download; [Apple's annotation publication](https://machinelearning.apple.com/research/sign-language-annotations)
distinguishes professional human annotations from model-generated pseudo-annotations.
Neither announcement admits every downloaded clip into the locked vocabulary.
This was a bounded availability check, not an exhaustive search or download attempt.

Discuss the intended recognition contract before another experiment: retain known signs,
represent genuine unsupported signing internally as OTHER, and evaluate it independently
from blank/rest and from uncertain annotation mapping. Existing runtime hides OTHER after
CTC collapse; displaying UNKNOWN to the user would be a separate output-policy decision.
First resolve the confirmed timing/padding inconsistency, define a consistent completeness
policy, and construct a held-out OOV evaluation with auditable lexical identities. Do not
repeat native-rate extraction wholesale, relabel all ambiguous glosses, or infer that a
lower known-WER proves reliable UNKNOWN classification.

## Verification

Fresh archive checks and disposable loader/collator counterexamples are in `evidence.json`.
Two Flores mapping/device tests and thirteen extraction/ASLLRP preparation tests passed.
The dropped-tail behavior is explicitly covered by the existing extraction test: passing
that test confirms current behavior, not its suitability for every aligned training task.
