# Public dataset search — 2026-09-21

User constraints: only public downloads without accounts; no ASL-fluent reviewer,
so use established lexical mappings. This is research and a proposal, not training approval.

## New verified access route: DePaul's NCSLGR ELAN conversion

- Publisher: https://asl.cs.depaul.edu/corpus/index.html
- Annotations: https://asl.cs.depaul.edu/corpus/elanBUcorpus.zip
- Video: https://asl.cs.depaul.edu/corpus/video.zip

Actual unauthenticated annotation GET succeeded: 5,335,358 bytes, ZIP CRC passed,
870 EAF files all parse as XML. Every file has main-gloss and non-dominant-hand-gloss
tiers. No missing/reversed times among alignable annotations were found; this does
not prove media synchronization or semantic completeness.

There are 850 EAFs in the ncslgr10-series, including all166 files in the already-local
10a–10d collections, leaving684 additional EAFs in other10-series collections. These
are annotation files, not684 approved locked-vocabulary clips or independent signers.
Twenty other EAFs describe narrative material. Video ZIP answered a four-byte Range
request with206, valid PK ZIP magic and total size4,108,493,785bytes. No full video
archive or new video clip was downloaded/decoded in this search.

This corrects the earlier blanket implication that all expansion annotations require
a DAI account. The public conversion is an alternative route, not necessarily the
same version as current DAI annotations. Publisher flags: ali has mismatched video
lengths/synchronization; DSP Ski Trip has damaged video; roadtrip2 has an offset and
missing annotation content. Exclude these from any initial admission and validate
all retained timing against actual selected media. EAF media URLs are stale local
paths and participant tier attributes inspected were absent, so reconcile with the
official catalog and existing signer roles. Do not invent signer IDs or gloss aliases.
No new row is marked training eligible. This is existing NCSLGR material via a newly
verified public access route, not a newly discovered independent corpus.

Verified archive retained at
`data/local/dataset_metadata/ncslgr_depaul_public_20260921/elanBUcorpus.zip`;
SHA256 and counts in `evidence.json`.

## Strongest immediate UNKNOWN source: unused Citizen identities

Official public page: https://www.microsoft.com/en-us/research/project/asl-citizen/
Official ZIP: https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip

Fresh local metadata count, using exact `ASL-LEX Code` exclusion against the100frozen
codes:2,623nonlocked codes,38,678training rows across35signers and9,926validation
rows across6signers. These are candidate rows, not a claim that those videos are
already local or admitted. No official test metadata/video was read in this search.

Proposed use: a bounded OOV subset with the original signer-disjoint roles, distinct
OOV lexical identities for training and held-out evaluation, and existing pinned
class mappings. Audit aliases/variant families and all training-source/checkpoint
exposure before describing any identity as unseen. Different code alone must not
override a documented mapping equivalence. Do not include unrelated ambiguous Flores
tokens in the clean OOV benchmark. Isolated OOV results do not establish connected
OOV detection; add trusted timed ASLLRP intervals for that separate evaluation.

## Other leads

- ASL STEM Wiki: public raw media and Apple's human annotation supplement already
  exist locally. Preserve111previously approved bounded spans; the larger candidate
  pool is not automatically admissible without established variant evidence.
  https://www.microsoft.com/en-us/research/project/asl-stem-wiki/
  https://machinelearning.apple.com/research/sign-language-annotations
- O5S5: exact SignBank-linked positive cores already local; retain positive-only use.
- ASL-Homework-RGBD: excluded from acquisition under the user's no-account constraint.
- How2Sign/OpenASL/YouTube-ASL/FLEURS pseudo labels: do not substitute captions or
  generated annotations for verified gloss supervision. No paused acquisition resumed.

No new broad corpus was verified to satisfy all of public access, raw video, strong
signer coverage and established locked100mappings without further admission work.
The concrete new result is public access to additional NCSLGR annotations, alongside
a quantified Citizen OOV candidate pool. Existing trusted data remain the starting point.

## Repair proposal for discussion

1. Fix current NCSLGR evidence/alignment length disagreement and mask auxiliary CE
   labels beyond each sample length. Keep one regression showing padded batch peers
   cannot create supervised steps.
2. Validate schema, frozen label mapping, source ranges and target feasibility at the
   shared loader. Reject overlapping ranges in this non-overlapping path.
3. Apply one explicit completeness policy: preserve valid short tails in future
   extraction; recover affected caches into a new version where possible. Preserve
   timestamps/missingness. Do not silently interpolate a failed detection into an
   observed sign, drop time, or treat all-zero features as verified rest.
4. Separate confirmed known, confirmed OOV and unresolved annotation status. Unresolved
   is admission metadata, not a new output class. With no reviewer, use trusted timed
   cores or exclude affected sequences; never delete a token while keeping its signing
   frames in a full-sequence CTC target. Keep source annotations immutable.
5. Build the independent OOV evaluation before further training. Report unseen-OOV
   recall, false-known emissions, known false rejection, mixed-sequence accuracy,
   known WER/exact and existing hold/repeat/rest checks. Preserve blank+100+OTHER.
6. Run a matched repair-only comparison first, then a separately controlled OOV-data
   addition and finally any newly admitted sentence data. Freeze evaluation identities,
   preserve official test sealing, and require known-sign retention. No model redesign
   or bulk re-extraction is justified by this audit alone.

These actions remain proposed; no training code, runtime defaults or training manifests changed.
