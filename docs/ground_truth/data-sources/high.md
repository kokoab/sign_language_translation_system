# data-sources — high

Dated evidence for constraints that still bind. The constraints themselves are
stated in `PROJECT_GROUND_TRUTH.md`; these are the receipts behind them.

10 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## 2026-09-21 — canonical phrase version2 retains exclusion and training gates

Use active/v17/approved_phrase_manifest_20260921_v2.json (494clips) as current phrase
manifest. Version1 is historical evidence, not the default. One contiguous established
WHEN/OTHER subspan admitted from an excluded parent does not re-admit the full parent.
No unlabeled gap or retiming was used to force salvage. Training-ready remainsfalse;
future recipe must recognize asllrp_verified_span and retain all prior blockers.
Evidence: artifacts/reports/verified_phrase_salvage_v17_20260921/REPORT.md.

## 2026-09-21 — canonical phrase admission requires enforced versioned manifest

New v17 phrase training must honor active/v17/approved_phrase_manifest_20260921.json
and its verifier, not historical broad directory roots. 493 whole clips pass established
identity and recovery/quality rules;1,365 excluded sequences remain preserved. Never
turn unresolved identity into OTHER or delete a target without its video. Current manifest
is not training-ready; source-dependent recipe metrics, auxiliary/rest supervision and
unseen-OOV evaluation remain unresolved. New training needs a reviewed new manifest and
compatible recipe; do not toggle training_ready alone or bypass through a legacy trainer.
Evidence: artifacts/reports/approved_phrase_manifest_v17_20260921/REPORT.md.

## 2026-09-14 — O5S5 exact direct positives; SoMe excluded; RWTH auxiliary-only

User visual review excludes SoMe ASL from all training because its predominantly
one-handed material is unsuitable for the intended recognizer. Preserve the files;
do not silently re-admit them. O5S5 exact ID-gloss equality through official ASL-LEX
`SignBankAnnotationID` admits direct positive contextual supervision: 256 deduplicated
occurrences across 53 locked classes, with all 256 containing Apple Vision hands.
Keep LG validation-only and the other five O5S5 signers training-only. O5S5 gaps are
never background because annotation completeness is unproven. RWTH-BOSTON-104 remains
training-only auxiliary data: it is low-resolution and its split reuses all three
signers. Neither source proves portrait-iPhone generalization. Evidence:
`artifacts/reports/o5s5_citizen100_v17/README.md`.

## 2026-09-13 — acquired continuous sources do not bypass schema or lexical gates

Épée's acquired 1,200 sequences contain MediaPipe poses without raw video; they
cannot be silently mapped into the Apple Vision v17 checkpoint. Its 68/100 exact
raw-label matches do not establish visual-variant equivalence. Newly acquired
MoLo raw video and RIT sample require alignment, completeness and variant review;
OOV or unannotated video is not automatically background. Preserve source licenses
and signer IDs. Acquisition does not establish phone generalization or accuracy.
Evidence: `artifacts/reports/continuous_asl_acquisition_20260912/README.md`.

## 2026-09-07 17:36 PHT — locked vocabulary confirmed; discussion continues

User explicitly retains the exact locked 100 glosses for continuous transcription
and requests further discussion. Vocabulary expansion is out of scope. The desired
behavior remains uninterrupted signing across combinations of those glosses without
waiting for per-sign confirmation. No additional 30-phrase collection is expected.
Web-data selection must preserve exact lexical variants and genuine contiguous
in-vocabulary spans; removing an intervening out-of-vocabulary gloss from a target
while retaining its video is not valid supervision. Annotation/coverage acquisition
and runtime implementation remain proposed, not started. Only this handoff changed.
Next discussion should clarify provisional live text versus final text and whether
Finish is acceptable once per utterance; these are separate from per-sign waiting.

## 2026-08-17 00:59 PST — 2M-Flores multimodal frozen features complete; training unblocked

All 155 long-sentence hand-RGB archives were encoded with MobileCLIP2-S0 in 20 short
sequential workers. The run covered 2,718 windows and 122,783 valid crop views with
zero failures. Peak MPS driver allocation was flat at 65,748,992 bytes across workers.
The hand audit matches all 155 expected archives with no missing, unexpected,
non-finite, non-normalized, mask, box, source, or schema errors. The encoding and audit
report SHA-256 values are respectively
`d6d66f0a6c8ac6ec5a1c498a31d1e7cbe54a30beeff78010cd18c208c2d5806b` and
`2253af42d8e519fe9f16ef8467944e480865d440463c957519445b5d1e8caca6`.

The frozen selected Stage 1 temporal cache then completed all 155 sentences in 19.7
seconds with zero failures. Its Stage 1 checkpoint remains pinned to SHA-256
`1caeadf4b3ca620aa9fef00b35c012b39d7c093f67da1ee2f6987d2c2297906b`;
peak MPS current/driver allocations were 48,528,640/165,232,640 bytes. The independent
frozen-feature audit covers all 155 archives, 2,718 windows, 2,810 full-order target
tokens, feature dimension 612, and all 448 auxiliary classes with no failures. Cache
and audit report SHA-256 values are
`8cff0eb14f71b274686c07d586eaeb2448ddebafc8b49d1dbf27979a2926d152` and
`41b7116adde6c178f84ff00287fc3552f12e08fabbf06674d0241eec60e16743`.
Dual-head CTC training is now unblocked. No evaluation split was accessed during
preprocessing.

## 2026-08-17 00:07 PST — full-gloss 2M-Flores manifests locked; long-video extraction proven

The success gate remains validation-only Stage 2 WER below 20%. The current selected
v2 checkpoint is not yet sufficient because its 254-row signer-held-out ASLLRP
contextual diagnostic is 21.2598% WER. The new 2M-Flores training design therefore
keeps the deployment 100-sign CTC head separate and uses a full-gloss auxiliary CTC
head to teach shared temporal structure without adding hundreds of rare distractor
classes to the deployment decoder.

All 155 acquired `dev` sentences now have a hash-verified training manifest. The
locked 100 labels remain indices 0--99. Five explicit annotation-category tokens and
343 recurring expanded lexical glosses produce a 448-token auxiliary vocabulary.
One-off lexical items map to an explicit unknown token rather than being deleted, so
the full ordered target timing is preserved; 579 token occurrences use that unknown
category. The selected videos require as many as 1,264 source frames, 40 nonoverlapping
32-frame windows, and 37 target tokens. The old eight-window cap must not truncate
this corpus.

The auxiliary vocabulary SHA-256 is
`227c462df9c5e2689645590b22a9e321838425fe9ed4db19902cbc6c9e7f2e44`.
The training manifest SHA-256 is
`eb197bd56c1b6601b75d69c2ff6dad35e0097320e29ef17f1976c794344831f9`.
The Apple Vision/RGB extractor now accepts an explicit maximum-source-frame contract
and fails closed rather than silently truncating declared longer videos. A real
1,264-frame-cap smoke run completed row 3 in 26.0 seconds with 13 windows, zero
failures, and Stage 2 schema fingerprint `277d70d19c5cbb42`. Six focused preparation
and extraction tests, JSON validation, and Python compilation pass. 2M-Flores
`devtest` and all project evaluation splits remain untouched.

## 2026-08-13 17:20 PST — exact ASLLVD supplement selected; failed archive route closed

The initially downloaded Rutgers ASLLRP distribution archives passed transport and
CRC validation but do not contain the newer recording IDs referenced by the current
Data-Sharing Project metadata. The exact-variant preparation therefore failed closed
at zero clips; those archives are not training data. The official BU ASLLVD workbook
provides a usable alternative with direct per-consultant movie URLs. An exact join
from each frozen Citizen class's ASL-LEX code through ASL-LEX 2.0 nonempty
`SignBankAnnotationID` to ASLLVD `Gloss Variant`, removing only the documented `+`
repetition suffix, selects 175 clips across 52 classes and six named consultants.
Selection permits at most one recording per consultant and five per class.

`scripts/prepare_asllvd_asllex_supplement_v17.py` downloads the official movies,
fully decodes them, hashes them, and records exact raw/feature provenance. The
download is resumable and active. The clips remain research-only/noncommercial and
will not be redistributed. The trainer loader independently enforces the exact
variant, signer, eligibility, tier, and false Citizen/SemLex test-access contracts.
`scripts/finalize_asllvd_asllex_supplement_v17.py` now retains extraction failures for
provenance while admitting only schema-valid v17 archives. No frozen test data was
accessed.

The current augmentation-only orientation winner has also been exported as a fallback
FP16 ML Program at `artifacts/coreml/Stage1OrientationV17.mlpackage`. It is 13.29 MiB
with package-tree SHA-256
`1cfd5e97cb8ebb29b424b1391ceb85ed9d62e5b7e25841b86254d414ccd0fb5e`.
Exhaustive parity over all 378 Citizen validation archives has zero top-1 mismatches;
the maximum logit difference is 0.006403. The measured 16.31 ms median and 21.67 ms
p90 are Mac timings only and are not phone evidence. The package and a manifest that
pins checkpoint SHA-256
`a7490409b3dfd76ba1ff432d2392b5e27df33f12e1b088cd36609fe03c082366`
are wired into the orientation benchmark Xcode project. The bundle will be replaced
and parity rerun only if the independent ASLLVD challenger wins the declared
development gates.

## 2026-08-10 10:23 PST — PopSign preview audit completed; training use remains blocked

Two provenance-locked scripts and focused tests were added for a PopSign/Citizen100
variant audit: `scripts/download_popsign_citizen100_previews.py`,
`scripts/evaluate_popsign_citizen100_variant_audit.py`, and their two test modules.
The downloader found 43 exact frozen-label overlaps using only exact names plus the
declared GOODBYE->bye, MOTHER->mom, and FATHER->dad aliases. It acquired three
distinct official train-participant previews per class, 129 clips total, under
`data/local/popsign_citizen100_variant_audit/raw/`. The provenance manifest marks every
preview `training_eligible: false`: PopSign's website previews are deliberately
downsampled and speed-normalized audit media, not the original recordings. The
original PopSign v1 clips are mostly 1944x2592, so the preview quality does not imply
that the source corpus itself is low resolution.

Apple Vision v17 extraction succeeded on all 129 previews with zero no-hand or failed
clips. Frozen Citizen100 model agreement was 33.33% top-1 and 65.12% top-5. The
non-benchmark triage classified 9 labels as model-consistent, 23 as ambiguous, and 11
as high-risk; it approved no class for training because exact lexical equivalence
still requires ASL-fluent/Deaf review. The report is
`artifacts/reports/popsign_citizen100_variant_audit/REPORT.md`. All three focused
preview/audit tests pass. PopSign v2.1 pages currently claim 562 signs, but tested
official archive/preview resource URLs returned HTTP 404, so no v2.1 data was acquired
or treated as available.

## 2026-08-10 10:56 PST — RIT v17 extraction and frozen-model triage completed

Apple Vision v17 extraction succeeded on all 292 selected RIT clips with zero no-hand
and zero failed cases. `audit_v17.py` passed all 292 archives with zero schema errors.
Extraction quality is strong: median observed-hand-frame coverage is 97.59%, median
hand-node presence is 86.91%, median face presence is 93.75%, and median body presence
is 93.75%. These measurements support the capture quality of the source but do not
establish lexical equivalence.

The frozen Citizen100 checkpoint was used only for mismatch triage, never as an
accuracy benchmark. Clip agreement is 38.36% top-1 and 60.62% top-5. Of 60 candidate
classes, 16 are model-consistent, 22 ambiguous, and 22 high-risk; zero classes were
automatically approved. The detailed immutable outputs are under
`artifacts/reports/rit_citizen100_variant_audit/`. ASL-fluent/Deaf review of the exact
ASLLRP entry/variant against each pinned Citizen ASL-LEX code remains required before
any RIT clip can enter a train-only supplement.

## 2026-08-10 14:38 PST — SemLex train extracted and v17-audited; val/test policy clarified

The user manually downloaded the official SemLex `train.tar.gz` to
`/Users/frnzlo/Downloads/train.tar.gz`. It has the exact expected size
(23,673,462,199 bytes) and passed a full `gzip -t` integrity check. The original
Downloads archive was not modified or removed. `--skip-download` support was added to
the bounded SemLex extractor so this local-archive run made no Drive request.

The full 1,624-member exact-ASL-LEX acquisition plan was scanned against the archive.
1,499 clips across all 98 matched classes and 32 SemLex train signers decoded and were
retained (564,485,964 bytes). The other 125 are explicitly rejected: 115 metadata video
IDs are absent from the official train archive, and ten VP9 WebM members fail full-frame
decode validation. Decode-broken members are quarantined under `rejected_raw/`; missing
members have provenance but no fabricated file. The immutable retained/rejected record
is `data/local/semlex_citizen100_train_audit/download_provenance.json`.

SemLex WebM support was added to the v17 batch inventory with a regression test. Apple
Vision v17 extracted 1,499/1,499 retained clips with zero no-hand and zero failed cases;
`audit_v17.py` passed all 1,499 archives with zero schema/invariant errors. Extraction
quality medians are 86.21% observed hand frames, 50.59% hand-node presence, 78.12% face
presence, and 43.75% body presence.

Frozen Citizen model agreement is used only as a cross-domain mismatch diagnostic:
72.85% top-1 and 89.86% top-5 across 1,499 clips. Class triage is 76 model-consistent,
21 cross-domain ambiguous, and one mismatch-review priority (`TAKE`, 2/10 top-1 and
3/10 top-5). Because SemLex is expert-aligned through the exact pinned ASL-LEX entry,
ambiguous classes are not discarded merely for Citizen-model disagreement. `TAKE` is
withheld pending review. A quality-ranked maximum-12-per-class first supplement now
contains 1,058 clips across 97 classes and all 32 retained train signers, materialized
as symlinks under `balanced_raw/` and `balanced_landmarks_v17/`; provenance is
`balanced_train_candidates.json`. It remains `training_eligible:false` until the final
review/approval decision.

SemLex validation/test archives are not required to maintain a numeric split ratio.
The first controlled augmented run should train on Citizen official train plus the
balanced SemLex train-only supplement and continue selecting checkpoints on Citizen's
fixed official validation split. SemLex validation could later be acquired as a
secondary unseen-SemLex-signer diagnostic, not merged into the primary selection
metric. SemLex test should remain untouched until a final frozen model warrants a
one-time independent SemLex-domain evaluation. Never rerun the already-consumed
Citizen official test during development.

Focused validation passes 19/19 tests: 12 v17 extractor tests (including WebM
inventory), five SemLex download/planning tests, one frozen-triage aggregation test,
and one balanced-selector test. The new/modified scripts compile and `git diff --check`
passes. Current SemLex audit storage is about 567 MiB; the
separate original 22.05 GiB archive remains in Downloads.

## 2026-08-09 18:47 PST — Citizen100 exact-variant manifest frozen

The user approved ASL Citizen as the sole primary dataset and confirmed the project is
personal/noncommercial. `active/v17/citizen100_seed.json` replaces the unavailable
standalone Citizen labels LOOK and SAY with EAT and WATER; SEE/TALK/TELL remain. The
reproducible builder `scripts/build_citizen100_v17.py` froze
`active/v17/citizen100_manifest.json` with 100 unique canonical labels and 100 unique
raw-gloss/ASL-LEX variants. No numeric or lexical variants are merged.

Coverage is 1,475 train, 378 validation, and 1,247 test videos (3,100 total). Per-class
signer ranges are 11–16 train, 3–5 validation, and 10–11 test. The manifest status is
`metadata_frozen_pending_asl_review`; exact ASL variant review remains necessary before
a final accuracy claim. Report: `artifacts/reports/CITIZEN100_V17_MANIFEST.md`.

`scripts/download_citizen100_v17.py` now reads only selected ZIP-member byte ranges,
decodes raw ZIP deflate, enforces official size and CRC, writes atomically, retains
official splits, and records SHA-256 provenance. Its dry-run planned 1.47 GiB transfer
and 1.59 GiB output with a 5 GiB reserve. Three focused downloader/manifest tests pass.
The 3,100-video selective download started at approximately 18:46 PST with four workers;
at this timestamp it is active with no observed failures.

The counts in this entry were superseded at 19:03 PST after replacing the accidentally
selected fingerspelling class with lexical `WHAT1`. The corrected total is 3,102.

## 2026-09-22 — user-authorized supplemental admission and signer policy

Use five hash-pinned manifests in artifacts/reports/supplement_finalization_v17_20260922/
plus the approved494phrase manifest for the next combined dataset. Shared/missing signer
IDs are descriptive, not disqualifying; preserve existing roles. Exact duplicate videos
must not appear on both sides of evaluation. Keep verified labels, features, source
boundaries, and exclusions enforced. Include local phrases; do not inflate independent
phrase counts with isolated signs or shared-parent segments. See REPORT.md and verification.json.

Combined entrypoint (2026-09-22): `data/local/combined_dataset_v17_20260922/manifest.json`
now materializes the six pinned inputs:6421records(4547train/1874validation). Future combined
trainers must explicitly consume and verify it; legacy trainers were not switched.
