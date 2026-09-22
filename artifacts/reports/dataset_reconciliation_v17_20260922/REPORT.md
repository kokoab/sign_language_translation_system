# Existing v17 datasets: reconciled inventory — 2026-09-22

**The 494-clip baseline was not an inventory of all usable project data.** It mixed
local and ASLLRP clips in each shuffled epoch, trained one common CTC head per seed,
and omitted other established supervision. The two seeds were independent repetitions,
not sequential training of separate datasets. No dataset was deleted by these audits.

This inventory restores prior approvals without converting weaker evidence into stronger
claims. “Usable” always names a role: complete sequence, positive sign interval, isolated
replay or motion-only. Counts of windows/caches are not independent source-video counts.
The listed previous approvals are historical source admissions; a combined-run manifest
must still verify global split compatibility and duplicate/overlapping source intervals.
This report does not silently authorize all files in a directory for training.

## Established supervised sources

| Source / representation | Existing train | Existing validation | Current supported use and limitation |
|---|---:|---:|---|
| Reviewed local whole phrases |232|200|**232train +199validation admitted**; one validation quality flag excluded. Retained whole-phrase caches432; source reviewed pool685videos is larger than this split. |
| ASLLRP contiguous whole phrases |44|12|All56 admitted after tail recovery. |
| ASLLRP OTHER whole crops |879|225|1,104exist; six full training crops pass conservative whole-sequence admission, plus one separately re-extracted training subspan. Other clips can still contain established positive intervals. |
| ASLLRP segmented exact signs |1,116|254|1,370established sign clips. Fresh shared-loader check passes1,115train+254validation; one training cache has incomplete tail. These are additional representations of contextual signs, not1,370new phrases. |
| O5S5 verified positive intervals |199|57|256officially linked positives across53lockedclasses from6narratives. All6Apple-observation files present; exact-positive counts rechecked. LG remains validation-only; five other signers train-only. No O5S5gap/OTHER/background admission. |
| ASL STEM Wiki reviewed intervals |90|21|111final human-reviewed approved spans across31classes. Fresh current-cache check passes88train+21validation; two training tails incomplete. v1/v2feature trees are representations of the same111, not222examples. |
| ASL Citizen locked100 isolated |1,476|378|1,854current train/val archives. Frozen-base provenance records1,475training clips; the extra current file remains unresolved. Pin/reconcile before new replay; official test stays sealed. |
| SemLex locked100 isolated |1,388|978|2,366current archives with existing train/validation audit provenance. Usable as isolated replay/retention once pinned into the new recipe; not full sentences. |
| Local deep-clean isolated auxiliary |13,381|2,896|16,277archives across94classes; prior project-owner vocabulary approval and extraction audits exist. Familiar-signer split is explicitly NOT signer-disjoint. Auxiliary use only under its historical restrictions; do not claim generalization or merge blindly with held-out local signers. |
| ASLLVD official exact-variant supplement |175|0|175previously approved train-only identities/archives across52classes. Recorded feature schema differs from current phrase landmark schema; source/feature and global signer compatibility need explicit checking before combining with current CTC. Not a rejected corpus. |

The local count in the first row is exactly232+199approved; no training local clip was
removed by the single validation quality flag.

**Totals that can be stated without pretending everything is one kind of example:**

- Current approved continuous/phrase view: **494sequence examples,283train/211validation**.
  It includes493whole clips and1derived subspan; this view is preserved.
- Established ASLLRP-segmented/O5S5/STEM supplemental supervision: **1,737sign clips or
  intervals exist**. Excluding three incomplete caches leaves **1,734candidate-ready
  supervision records:1,402train/332validation**, with O5S5 using its existing positive-core
  loader. Cross-source occurrence overlap still needs deduplication before sampling.
- Citizen+SemLex current isolated train/val inventory: **4,220archives**. One Citizen
  training archive remains outside the historical frozen-base count; do not equate
  structural presence with an immutable new-run admission.
- Additional restricted auxiliary inventories: **16,277local isolated clips and175ASLLVD
  clips**, under the limitations above.

Do not add these into a claimed unique-video total. ASLLRP segmented signs, phrase crops,
context windows and derived event cores can represent the same source event; O5S5has256
positive intervals inside6videos. RGB, frozen features, recovered copies and symlink
views are alternate representations, not new data. Validation rows stay validation.

## Existing sequences with unresolved or separate-purpose supervision

| Source | Existing amount | Proper status |
|---|---:|---|
| 2M-Flores ASL |155training-dev video/caches;141full-coverage OTHER caches|Previously used in experiments.14original caches failed tail coverage. Cross-corpus exact identity links remain unresolved for strict locked100/OOV truth. Preserve all155; not declared bad or automatically blank/OTHER. No devtest access. |
| NCSLGR |166local utterances;125with strict targets (88train/37validation)|125recovered phrase timelines pass structural checks. Original exact-text supervision lists174locked-string occurrences across14labels; lexical identity equivalence remains unresolved under the stricter audit. Do not conflate repaired timing with verified lexical mapping. Older motion manifest calls all166train; supervised participant-disjoint roles win for supervised work. |
| How2Sign |1,027training video records|All1,027raw paths present; motion/self-supervised input only. Sentence text is not per-sign truth. Validation/test remain excluded. |
| YouTube-ASL raw-video transition subset |128videos:104train/24validation|All128raw paths present. Motion-only; channel split is a signer proxy, not verified identity. |
| YouTube-ASL keypoint subset |1,411files;1,191count-consistent before quality filters|Separate46-node2D/MediaPipe-style adapter branch, not direct Apple-v17landmarks.220count mismatches quarantined. Acquisition paused. Do not add to128without source overlap checks. |
| OpenASL transition subset |3videos:1train/2validation|All3raw paths present; motion-only domain reference. |
| RWTH-BOSTON-104 |201videos|Existing low-resolution auxiliary corpus; official split reuses3signers. Older project prose described auxiliary use, but its acquisition audit says exact Citizen mapping/signer-role approval incomplete. Final registry keeps it conditional, not approved locked100supervision. |
| Épée |1,200timed sequences|Existing MediaPipe-only data without raw video; no direct frozen-Apple compatibility. Not approved locked100training merely from matching text. |
| MoLo |one acquired17.29-minute video;1,517hand annotations|Prior acquisition record; exact variant/annotation coverage remains unadmitted. |
| RIT fluent-coded public sample |one6.5-second sample|Candidate source; not admitted sign supervision. Other consumed/reserved RIT evaluation is not training. |
| Local phone development |5records:3train/2validation|All5raw paths present. Existing development set, not independent iPhone test evidence; do not merge blindly with local signer holdout. |
| SoMe ASL |Files retained|Explicitly excluded by user visual review. No re-admission. |
| PopSign |Paused incomplete archive|No extracted video in the prior storage audit. No usable approved training count; acquisition stays stopped. |
| Legacy/synthetic/cache variants |Retained|Not new independent data; legacy Apple/MediaPipe formats or synthetic sequences do not become current-v17truth by directory count. |

## What is verified now versus inherited

Fresh checks: canonical494manifest verification; actual archive counts for Citizen,
SemLex, local-deep-clean, ASLLVD and segmented/STEM roots; current-schema/full-coverage/
target checks on all1,370ASLLRPsegmented and111STEMarchives; sixO5S5Apple-cache paths
and256exact-positive interval records; raw-path existence for1,027How2Sign,128YouTube,
3OpenASL,166NCSLGR and5phone records. This does not re-label or visually re-review sources.

Prior evidence retained: official variant admissions, prior structural audits,
human-reviewed STEM approvals, license/role constraints and candidate acquisition counts.
Conflicting or weaker evidence is explicit, not silently elevated. Protected test
archives were not loaded or counted into available training/validation totals.
`evidence.json` pins relevant source documents and records the three blocking cache paths.

## Correct next training policy

The intended broader recognizer should use a **single shared model with a controlled mix
of compatible approved supervision**, not one isolated model per corpus or an unweighted
concatenation of every directory. Whole phrases get their valid CTC targets; established
single-sign intervals/isolated clips provide replay or bounded positive supervision.
Unlabeled motion, incomplete gaps and unresolved OOV text cannot be assigned those targets.
Mixing weights must stop the much larger isolated pools from overwhelming phrase learning.
Keep validation signers/roles and source-event deduplication across that mixture.

Next concrete integration work: pin the supplemental ASLLRP/O5S5/STEM and isolated inputs,
resolve the three incomplete caches and the extra Citizen file, check parent-event/global
signer overlaps, then prepare a matched all-admitted-supervision experiment. Preserve the
494baseline as its comparison. Do not replace this reconciliation with another tiny
whole-phrase-only audit. No new training was launched during this inventory reconciliation.
