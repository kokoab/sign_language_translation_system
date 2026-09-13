# stage2-ctc — log

Measured results, rejected approaches, and progress snapshots. Not read start-to-end —
`rg` this file before re-running an experiment to see if it already failed.

17 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## 2026-09-09 — matched transition-adaptation experiment failed retention gates

The fixed MPS comparison completed normally for both arms and both seeds; training
result SHA-256 is
`badd541dbb156a4f57b8cbffc67fb8fdca36ac9a7c9c21e73954ff494b125e1e`.
All four checkpoints exceeded the required ASLLRP target improvement, reducing the
accepted selector's 603/284 locked-target edits/tokens to 213--224/284. None qualified:

- no-STEM seed 1701 (epoch 19): target 213, local/exact/contextual edits 12/16/57,
  Citizen 331/378;
- no-STEM seed 1702 (epoch 15): target 217, retention 14/14/51, Citizen 327/378;
- with-STEM seed 1701 (epoch 18): target 221, retention 15/15/50, Citizen 322/378,
  held-participant STEM 4/21;
- with-STEM seed 1702 (epoch 11): target 224, retention 14/15/53, Citizen 324/378,
  held-participant STEM 3/21.

The retention ceilings are 6/9/43 and the Citizen floor is 328/378. Thus every seed
failed retention, three also failed Citizen, and neither arm can pass the mandatory
two-seed gate. The plan's failure branch is now binding: produce the versioned
selection/failure evidence, keep the accepted selector/Reel runtime unchanged, and do
not launch another sweep or integrate/export any candidate. No protected test data was
accessed.

The deterministic selector independently recomputed all 20 seed/gate outcomes and
confirmed `selected_arm: null`, `integration_allowed: false`, and
`runtime_action: retain_current_selector`. The versioned failure deliverables are
`selection.json`, `failure.csv`, `per_example_predictions.jsonl` (7,272 rows covering
the baseline, common initialization, and all four seed checkpoints across seven
domains), and the experiment `README.md` under
`artifacts/reports/stage2_v17_transition_adapt_v1/`. History and checkpoint hashes are
pinned; per-example repeat counts reconcile with the aggregate target and
OTHER-inclusive summaries. Independent review passed after those checks were added.

Because no arm qualified, Task 4's export and Reel integration were intentionally not
executed. This is the plan's required rollback outcome, not a missing deployment step:
the accepted general selector and old packages remain the runtime, no candidate package
was created, and candidate Core ML parity/replay/latency checks are not applicable.
The reproducible rollback launch is
`venv/bin/python scripts/live_reel_continuous_v17.py --camera 0`.

Final verification passed 60 focused tests across preparation, extraction, trainer,
selection, legacy CTC decoding, and continuous-Reel lifecycle behavior. These cover
blank/repeat collapse, emission positions, unchanged legacy defaults, preserved frames,
Finish draining, reset, stale-result rejection, and review-only transcript ownership.
All four generated 102-logit checkpoints reload with finite weights. Existing five-video
development replay evidence remains 5 total edits in both baseline and continuous lanes
with zero stale camera frames; no candidate comparison was run because selection is
null. `py_compile` and repository-wide `git diff --check` passed. The authoritative
verification record is
`artifacts/reports/stage2_v17_transition_adapt_v1/verification.json`. No protected test
data was accessed, and no independent iPhone/general continuous-ASL claim is made.

## 2026-09-08 16:13 PHT — real continuous-data inventory supports training before more collection

The non-overlapping, immediately relevant real Stage-2 inventory is 2,022 target-bearing
or strict-vocabulary units: 1,104 exact-variant ASLLRP `OTHER` spans, 111 newly approved
ASL STEM Wiki bounds, 132 target-bearing NCSLGR utterances, 155 acquired 2M-Flores dev
sentences and 520 strict locked-vocabulary local phrase recordings. Counting all 166
NCSLGR utterances, including 34 without a locked target, gives 2,056 usable real units.
These sources contain approximately 3,662 locked-target occurrences: 1,483 ASLLRP, 111
STEM Wiki, 198 NCSLGR, 490 selected 2M-Flores matches and 1,380 strict local phrase
tokens.

Quality tiers must remain explicit. ASLLRP plus human-reviewed STEM Wiki provide the
strongest exact-ASL-LEX subset: 1,215 bounded spans, 1,594 target tokens and 64/100 exact
variants across their union. NCSLGR and 2M-Flores add genuine sentence context but lexical
labels do not prove the pinned Citizen variant; all selected 2M-Flores rows use one local
signer ID. The 520 local clips cover only six fixed phrase templates and three audited
recording identities, so they remain familiar-domain training/diagnostic material.

Decision: do not gather another broad dataset before the next bounded adaptation. First
materialize the 111 reviewed spans and measure whether they improve signer-disjoint
continuous boundaries while preserving existing phrase and isolated gates. More data is
still needed for a final general recognizer: 36/100 glosses lack exact-variant continuous
coverage, and the newly reviewed STEM pool gives only 18 glosses the minimum three-person
coverage. A subsequent search/collection should target those measured gaps rather than
add more unfiltered phrases.

## 2026-09-08 14:27 PHT — Citizen-linked continuous phrase pool expanded to 268 videos

Official ASL Citizen documentation confirms it is an isolated-sign dataset and explicitly
cautions against using concatenated Citizen clips as continuous signing. The closest
released phrase source tied to its dictionary is the professionally glossed ASL STEM Wiki
supplement. A full manual-annotation audit found 268 unambiguous target-bearing videos
covering 44/100 exact locked raw-gloss strings and 506 target occurrences across 19
participant IDs.

Extended `scripts/acquire_asl_stem_wiki_manual_v17.py` with participant and output-manifest
arguments, then selectively range-downloaded 170 additional official videos while reusing
the 98 existing files. The separate manifest is
`data/local/asl_stem_wiki_bootstrap_v1/manual_candidate_expansion_manifest.json`. All
268 local files independently reproduce their recorded SHA-256; their decoded size is
1,040.74 MiB. The completed 71-row human review queue was not modified. This expansion
remains quarantined: participant quality, exact Citizen ASL-LEX variant and sign boundaries
are not yet verified, and no row is training eligible.

FLEURS-ASL is the next public lead: 1,749 genuine sentences from five Certified Deaf
Interpreters. Its released model-generated annotations show useful locked-vocabulary
overlap, but are not proper expert gloss labels and therefore rank below this manual ASL
STEM Wiki pool. ASLLRP remains the strongest already-acquired exact-ASL-LEX continuous
source. No protected Citizen, SemLex, local, RIT or other evaluation split was accessed.
Full findings: `artifacts/reports/citizen100_phrase_source_expansion_v17/README.md`.

## 2026-09-08 06:40 PHT — Cokely public continuous-source acquisition underway

Acquisition and strict clip preparation completed. Five public640x360/29.97fps HLS
sources total218097458bytes and cover all strict candidates; durations72.685-468.214s
exceed referenced EAF endpoints. Generated31non-overlapping clips (3829618bytes),
97target tokens,23distinct sequences,23locked classes,5recordings and4identified
signers: David Hamilton,Patrick Graybill,Mark Morales,MJ Bienvenu. The prepared set is
large enough for the next compatibility/feature-extraction step, but it does not
replace the originally planned30phrasesx13signers generalization evidence. It is
imbalanced: DIFFERENT accounts for50/97tokens. All source/clip hashes, annotation IDs,
times, signer provenance and probe results are in
`artifacts/reports/continuous_reel_v17_cokely_v1/manifest.json`; findings and license
terms are in its README. Representative frames for all4signers were visually checked:
active centered signing aligns with selected times.

Preparation initially failed because annotation6 has no candidate and no downloaded
video, then because strict splitting produced31rather than exactly30clips. Fixed the
root contract to skip no-candidate recordings and require at least30clips. Fresh run
succeeds. All31 clips fully decode; syntax check, git diff --check, and6 combined
Cokely/MoLo focused tests pass. Every row remains `training_eligible:false` because
Cokely's project-specific ID-gloss equality does not establish the exact Citizen
ASL-LEX visual variant. Next: frozen Stage-1 compatibility screen plus visual review,
then v17 feature extraction for compatible train-only clips. Do not touch existing
held-out ASLLRP/local evaluation sets or claim signer-disjoint continuous accuracy.

User authorized gathering enough public continuous video independently. Identified
the Cokely Parallel Corpus: six CC BY-NC-SA4.0 ASL translations with timed project
ID-gloss EAFs and downloadable/streaming video. Downloaded and XML-validated all6 EAFs
(1032898bytes) into `artifacts/reports/continuous_reel_v17_cokely_v1/annotations`.
The publisher CGI initially returned403; its normal page-cookie flow allowed the
public EAF downloads. The 640x360 public HLS streams for recordings1-5 are currently
downloading to `data/local/continuous_phrase_sources_v17/cokely/source_videos`;
recording6 contains no exact frozen raw-label runs and is not being acquired.

Strict case-sensitive Citizen raw-label matching, with unknown labels and unresolved
timing as hard boundaries, finds27maximal multi-sign runs. Splitting only runs longer
than6signs at annotation boundaries yields exactly30non-overlapping clips across
recordings1-5 and4publisher-identified signers. Added a focused red/green test and
`scripts/prepare_cokely_continuous_v17.py`;2 tests pass. This is not yet training
admission: identical text labels in Cokely's project lexicon do not prove the exact
Citizen ASL-LEX visual variant. Generated rows remain `training_eligible:false` until
video/timestamp integrity and cross-corpus visual variant review complete.

## 2026-09-08 — Stage 2 transition-adaptation plan preflight started

Implementation of the retention-gated Stage 2 CTC comparison is active. Preflight
verified the frozen manual STEM snapshot at 111 eligible intervals across 18
participants, the 879/225 ASLLRP known-plus-OTHER frozen sequences, encoder checkpoint
SHA-256 `1caeadf4b3ca620aa9fef00b35c012b39d7c093f67da1ee2f6987d2c2297906b`, the plain
Stage 2 warm start, accepted general selector, FP32 Stage 2 export path, and current
Reel runtime. Work is proceeding in the existing dirty checkout because the required
v17 code, reports, and data are untracked there; unrelated changes remain out of scope.

The currently modified `artifacts/reports/stage2_v17_frozen_features/cache.json`
describes a separate Reel encoder (`278a9933...`) and is not authoritative for this
experiment. Frozen feature inputs must instead fail closed against each archive's
embedded encoder provenance; sampled existing ASLLRP/local archives record the required
`1caeadf4...` checkpoint. The official Citizen test remains closed.

Task-1 implementation produced a provisional 111-row manifest (90 train / 21
validation) and a dedicated timestamp-bounded extractor, but independent review found
it unready for data generation: generated rows omitted `source_group`, audit coverage
was not yet experiment-wide, vocabulary/source-role provenance was under-pinned, and
the 30-fps endpoint rule needed correction for non-30-fps sources. The initial sandboxed
Vision Code 9 was environmental; an approved unsandboxed one-row rerun initialized
Apple Vision and then reproduced the predicted `source_group` `KeyError`. No Stage 2
transition-adaptation archive has been accepted and no full extraction has started.

The accepted general selector's exact baseline gates were measured on the frozen
development/validation inputs: 603/284 ASLLRP known-target edits/tokens over 225 spans,
6/259 familiar local-phrase edits/tokens, 9/24 exact ASLLRP phrase edits/tokens, 43/254
ASLLRP contextual-sign edits/tokens, and 331/378 (87.5661%) Citizen official-validation
single-sign sequence accuracy. The 10% relative ASLLRP gate therefore requires at most
542 target edits; retention ceilings are 6, 9, and 43 edits, and Citizen accuracy must
remain at least 86.5661% (328/378 by integer count). These are validation/development
baselines; no official Citizen test data was accessed.

## 2026-09-07 17:30 PHT — continuous-signing verification and web-data discussion

User clarified the requirement: the signer must continue signing without waiting for
each sign to lock. The previously anticipated 30 phrases from 13 native signers will
NOT arrive; all earlier recommendations depending on that collection are superseded.
Current request is investigation and discussion before implementation or acquisition.

Read `model_stage2_v17.py`, both live Stage-2/Reel paths, relevant acquisition code,
and saved selector/data/replay reports. The direct `live_stage2_ctc_v17.py` loop queues
elapsed 32/30-second windows asynchronously without a per-sign acceptance lock. Its
612-D evidence is 256 landmark features + 256 hand features + 100 fusion scores, not
Squeezeformer alone. The four-layer Transformer uses padding attention masking, not
causal masking; recent hypotheses can change within eight rolling windows (~8.53s),
and emissions leaving that context are frozen. Speech has separate prefix stability.
Finish drains pending windows and invokes the naturalizer; acquisition of observations
pauses during the Finish drain. Reel retains its Stage-1 stability/commit locks, with
Stage 2 disabled by default and used as an optional finished-sequence arbiter.

Saved validation JSON confirms local 7/259 -> 6/259 edits (92/97 exact), ASLLRP
11/24 -> 9/24 (4/12 exact), contextual signs 44/254 -> 43/254. Local 97-clip validation
was not signer-disjoint; 5/5 elapsed-time replays are familiar development recordings.
These establish neither robust unseen-signer continuous accuracy nor bounded live lag.
Existing seven `test.test_live_stage2_ctc_v17` tests pass; no model evaluation rerun.

Web sources checked: full NCSLGR official download page lists 1,887 utterances and
11,854 sign tokens, mostly four native signers, with timed annotations/video index:
https://www.bu.edu/asllrp/ncslgr-for-download/download-info.html . Current acquisition
script covers only ncslgr10a-d (166 utterances); expansion must deduplicate existing
data, preserve participant roles, and verify documented half-speed video variants.
Modern ASLLRP remains a timed-gloss source with login-gated downloads:
https://signstream.cs.rutgers.edu/dai/s/signbank . Existing exact-variant audit found
1,237 target-bearing utterances but only one complete multi-token utterance entirely
inside the locked vocabulary, plus 70 contiguous target-only spans; these are already
partitioned and cannot be counted as new data or move reserved RIT into training.

New candidate MoLo: https://ida.gallaudet.edu/molo/ documents ~46h/46 participants,
natural conversations, CC BY-NC-SA videos and developing ID-gloss ELAN transcripts at
https://osf.io/9uevh (videos https://osf.io/wma3e/). Research browser received 403 on
OSF, so transcript completeness, downloadable payloads and exact-vocabulary overlap
are unverified. How2Sign https://how2sign.github.io/ and OpenASL
https://github.com/chevalierNoir/OpenASL offer video/English-translation resources;
no verified ready ordered 100-gloss supervision was found there. Next proposed step
is metadata/annotation coverage audit of full NCSLGR, modern ASLLRP and MoLo before
selective video acquisition. Preserve transitions and legitimate sign holds when
designing idle gating. No runtime/model changes, bulk downloads, training, protected
split access, or implementation approval in this discussion turn.

## 2026-09-06 21:18 PST — opt-in motion proposals connected to continuous runtime

Previous turn classified progress: new source-comparison clips and six-video results.
Added --context-proposals (requires matchingcontextcheckpoint) to experimental live
script and context_proposals flag to ContinuousRecognizer. Runtime retains aligned
CTC logprobabilities alongside motion evidence, supplies exact CTC scores for proposed
sequences at utterance end, resets both buffers, and caps both at4501observations
(the existing overflow sentinel). Default inference remains unchanged. New actual
checkpoint regression verifies5inputframes produce2observations, repeatedfinish adds
none, resetclearsboth. RED missingarg→GREEN; all8context tests pass.

Recomputed guarded proposal policy from existing joint_context_v2 candidates, no
new model selection: local184/200exact,3.52%WER; Citizen355/378;SemLex752/978 with
6correct→wrong. ASLLRP16/24edits versus14raw; NCSLGR83/121unchanged. Empty/OTHER
leadinghypotheses retainraw; no originallycorrect background sample is changed to
nonempty. Saved guarded_joint_context_v2.json. These harms prevent general promotion.
Actual omittedSCHOOL clip is running with the flag in session75721/output
continuous_rebuild_v17_v1/video_proposals_v2_school. Do not claim recovery yet.

## 2026-09-06 10:29 PST — continuous baseline trained; motion-context and runtime checks started

`continuous_evidence_v17_v1` finished 25 epochs, selecting epoch 20. Development
greedy decoding: local 27.50% exact / 34.07% WER (200 clips), ASLLRP 0/12 exact /
66.67% known WER, NCSLGR 2/37 exact / 84% known WER, rolling isolated 79.65% exact,
and genuine-gap any-token false emission 5.22%. These are development measurements,
not a usable conversational translator or proof that synthetic generation replaces
real continuous data. Official Citizen test was not accessed.

Added `stage3_motion_context_v17.py` and `train_stage3_motion_context_v17.py`: a
motion cross-attention sequence scorer can rank supplied CTC alternatives including
OTHER and end-of-sequence. Two focused tests passed after failing before implementation:
padding invariance, motion dependence, and autoregressive candidate causality. Training
is now running against the frozen continuous checkpoint/cache, with matched and shuffled
motion correction reports. Any selected nonzero weight remains experimental despite
the initial report's `promoted` field; independent gains and harmful corrections must
be assessed before runtime default use. An eight-clip incremental replay smoke is also
running. Logs: `continuous_rebuild_v17_v1/stage3_motion_v1.log` and `runtime_smoke.log`.

Avatar v3 visual inspection still finds hands entering the torso. The input log-scale
channel cannot supply real depth; the next correction is an explicit avatar-space
body-clearance constraint, followed by visual review. It must remain distinct from
training landmarks and must not be presented as recovered source depth.

## 2026-09-06 10:24 PST — clean encoder selected; continuous training started

Clean-lineage v2 completed 20 epochs in 524.19 seconds and selected epoch 18 under
the unchanged retention gate: Citizen 95.50%, SemLex 87.42%, local weak phrase cores
76.30%, ASLLRP manual cores 79.17%, NCSLGR strict cores 22%. This remains an isolated
encoder experiment, not a continuous result. Its checkpoint is now being used by
`python -m active.v17.train_continuous_evidence_v17 --base
artifacts/models/stage1_v17_grounded_clean_lineage_v2/best_model.pth`; log
`artifacts/reports/continuous_rebuild_v17_v1/continuous_v1.log`.

Execution checklist for the approved full design (completion remains unproven):
- [x] Trace inherited exposure and restart encoder from isolated-only provenance.
- [x] Implement true-gap supervision, matched rolling observations, causal scored
  alternatives and revisable-prefix mechanics; focused tests pass.
- [ ] Train/evaluate the continuous recognizer on real development signers, then
  verify incremental runtime equivalence and end-to-end video timing.
- [ ] Produce visually acceptable mesh animation and improved transition trajectories;
  retain exact observation/animation distinction and obtain fluent review.
- [ ] Train motion-conditioned contextual sequence scoring, connect it to existing
  English naturalization, measure useful vs harmful corrections and motion ablations.
- [ ] Compare real-only against reviewed synthetic-augmented recognition with locked
  real development evaluation, compositional coverage and explicit unknown behavior.
- [ ] Complete continuous live UX without per-sign input locks and assess full goal
  against measured accuracy, latency, avatar review, and corpus limitations.

No additional user approval is needed for the already accepted design. Newly available
process skills are applied to focused red/green tests and evidence review without
restarting the approved requirements discussion. The avatar v3 replacement render is
running after mesh fixes; decorative untracked face drawings were removed, and clothing
now follows bind-pose membership instead of changing as hands cross the torso.

## 2026-09-04 — local phrase signer audit and continuous-supervision decision

The 780 videos under `data/raw_videos/PHRASES` were anonymously audited with central-face
YuNet detection, SFace embeddings, and average-linkage cosine clustering. The reusable
script is `scripts/cluster_local_phrase_signers_v17.py`; face embeddings are never
persisted. A naive forced seven-cluster run produced three large groups (298, 298, 177)
and four 1-3-video fragments, including a false split of numbered takes. After fixing
the bystander error by selecting the central rather than largest face, the evidence
supports three recording identities in this phrase root: 298, 300, and 182 assignments,
with 778/780 confident and a 0.737 cosine silhouette. Every main group covers all nine
phrases, and all 180 numbered takes belong to one group. The two uncertain clips must
remain training-only or excluded. The user's seven local signers therefore appear to
refer to the wider local holdings; this specific nine-phrase root cannot honestly be
split into seven phrase signers.

Use anonymous local signers 01 and 03 for training and signer 02 for validation in the
next experiment. Never reuse the previous random train/validation folder assignment as
evidence of signer generalization. NCSLGR will add genuine continuous supervision with
strict raw-gloss matching only; How2Sign, OpenASL, 2M-Flores, and other unaligned videos
may supply self-supervised motion/background exposure but not invented gloss targets.
The official Citizen test remains unopened.

## 2026-09-04 — grounded continuous experiment completed

The statement above applies only to the 780 recordings in `data/raw_videos/PHRASES`,
not to every person visible anywhere in the wider local holdings. The user is correct
that the wider local files contain seven or more people. Some of those files are
isolated-sign holdings (including a class literally named `SENTENCE`), not annotated
phrase sequences, so they cannot be counted as extra continuous phrase signers.

The phrase signer audit is now automatic. `cluster_local_phrase_signers_v17.py` compares
2 through 10 average-linkage cosine solutions and selects the highest silhouette unless
an explicit count is supplied. On the nine phrase folders it selects three clusters:
300, 298, and 182 videos, silhouette 0.737114. A forced-seven audit yields
298/278/180/20/2/1/1 with silhouette 0.554527. Contact-sheet review confirms that the
20/2/singleton fragments are lighting, framing, and expression splits of the same three
people rather than four additional phrase signers. The final anonymous audit is
`artifacts/reports/local_phrase_signer_audit_v17_v4_auto/`. Its output stores no face
embeddings.

The auto-audited split was rebuilt without overwriting the original as
`data/local/stage2_v17_grounded_signer_split_v2_auto/`: 287 local phrase archives from
signers 01/03 train and 200 from signer 02 validate; ASLLRP contributes 44/12 exact
phrases and strict NCSLGR 88/37 participant-disjoint utterances. All 668 NPZ archives
are byte-identical to the already trained `stage2_v17_grounded_signer_split` version,
so rerunning the models would be redundant. No official test or reserved external
evaluation split was accessed.

`prepare_ncslgr_supervision_v17.py` produced
`active/v17/ncslgr_supervised_manifest_v17.json` from published SignStream intervals.
The strict overlap contains 125 utterances and 174 exact 100-gloss occurrences across
14 classes; other lexical events are explicitly `OTHER`, never guessed by normalized
aliases. `prepare_grounded_streaming_data_v17.py` makes the combined leakage-safe
landmark-only archives.

The retention-gated Stage-1 adaptation in
`artifacts/models/stage1_v17_grounded_adapt_v1/` selected epoch 6. Relative to its base,
top-1 changed Citizen validation 94.18% -> 93.39%, the other isolated validation corpus
85.79% -> 84.36%, held-out local phrase cores 91.48% -> 92.59%, ASLLRP cores 83.33% ->
87.50%, and NCSLGR cores 10% -> 24% (NCSLGR top-5 24% -> 58%). It passed the declared
retention gates, but plugging it into the streaming decoder was worse; do not promote
it to the accepted live path.

The new causal experiment is `train_unified_streaming_aligned_grounded_v17.py`. It
uses an 8-frame rolling window, stride 4, CTC blank, the locked 100 glosses, explicit
OTHER, isolated replay, and true NCSLGR frame intervals. It never waits for future
frames and has no phrase grammar. Three matched seeds (17081/17082/17083) compared CTC
alone against CTC plus 0.12 timed-alignment loss. Alignment improved the composite
selection score in all three seeds. Mean results:

| Validation metric | CTC only | + timed alignment |
| --- | ---: | ---: |
| composite score (lower is better) | 0.40582 | **0.39216** |
| exact phrase accuracy | 19.95% | **20.75%** |
| all-phrase WER | 43.59% | **42.67%** |
| local held-out-signer exact | 21.50% | **23.00%** |
| local held-out-signer WER | 39.44% | **39.07%** |
| NCSLGR WER | 86.67% | **80.00%** |
| isolated exact | 81.27% | **82.47%** |
| transition false emission | **13.64%** | 15.44% |

The selected research checkpoint is seed 17081 at
`artifacts/models/unified_streaming_aligned_grounded_v17_v1/best_model.pth`. Its local
held-out-signer result is 25.5% exact / 37.04% WER; ASLLRP 41.67% WER; NCSLGR 78% WER;
isolated exact 82.37%; transition false emission 11.36%. A measured MPS pass was about
11.34 ms median / 20.02 ms p95 for Stage 1 plus the causal head, with an emission step
available every 4 frames (about 133 ms at 30 fps). Compute is not the blocker; boundary
and coarticulation accuracy are.

The full report and machine-readable means are in
`artifacts/reports/grounded_continuous_v17_experiment_v1/`. The conclusion is negative
but useful: exact timing supervision is directionally correct, while existing local,
ASLLRP, and two-participant NCSLGR data are too narrow for a stable continuous demo.
The incoming 30 natural phrases from 13 native signers remain the recommended next
dataset. Keep 10/2/1 signers train/validation/sealed, all synchronized views from one
performance in one split, and record normal conversational speed. Automatically propose
gloss boundaries, then manually correct them. Do not manufacture targets for unaligned
How2Sign/OpenASL videos.

Validation after these additions: all relevant scripts pass `py_compile`; the four
unified causal/shape tests in `test.test_unified_streaming_ctc_v17` pass; and
`git diff --check` passes. Accepted live inference files were not overwritten.

## 2026-09-01 13:08 PST — matched public-data Stage-2 ablation favors conservative 2M + ASLLRP transfer

All currently usable recommended data were already acquired, so no duplicate corpus
download was performed: 2M-Flores has 155 selected `dev` videos and frozen features;
ASLLRP has 1,104 `OTHER`-CTC spans (879 train / 225 signer-held-out validation) and
frozen features; NCSLGR has 166 utterances from two native signers; and the How2Sign
train-only subset has 1,027 acquired clips, 1,026 usable, across six source signer IDs.
NCSLGR and How2Sign remain transition/self-supervision sources because the current
artifacts do not provide compatible exact-variant ordered CTC targets. DSP video is
not downloadable under the published BU terms, Apple's annotation bundle was not
located, and ASL Homework remains access-gated. No large unlabeled corpus was added.

`train_stage_2_other_ctc_v17.py` now accepts the existing
`slt_stage2_temporal_pretrain_v17` checkpoint as a safe initialization. It verifies the
original teacher hash and matching configuration, proves that temporal pretraining did
not change the CTC head, optionally interpolates only the non-head state through
`--temporal-mix`, and keeps the original 100-class checkpoint as the distillation
teacher. The checkpoint/report record complete temporal provenance. Four focused tests
pass, including a 25% interpolation and original-teacher separation test; Python
compilation and `git diff --check` pass.

Full-strength temporal initialization followed by the original high-rate adaptation
was rejected: every trained epoch violated an old-data guard, and its selected epoch 8
had 12/259 local, 13/24 older-ASLLRP, and 365/682 new-ASLLRP full-sequence edits.
Initialization screens at 25%, 50%, and 75% showed that 50% is the strongest mix that
preserves both legacy gates before training. The original training recipe at 50% was
also rejected because epoch 0 remained selected and all trained epochs forgot old
phrases.

A predeclared conservative matched A/B then froze the backbone for all ten epochs,
used a `1e-4` head learning rate, raised replay distillation to 1.0, reduced new ASLLRP
sampling mass to 0.25, and used identical data, seed 1701, and selection gates in both
runs. ASLLRP-only selected epoch 5 with 617/682 new-ASLLRP full edits (90.47% WER),
450/284 target-only edits, 0/225 full exact sequences, 11/24 older-ASLLRP edits, and
7/259 local edits. The 50%-2M + ASLLRP combination selected epoch 6 with 542/682 full
edits (79.47% WER), 365/284 target-only edits, 4/225 full exact sequences, the same
11/24 older-ASLLRP edits, and an improved 6/259 local result. This is 75 fewer full
edits (12.2% relative) and 85 fewer target-only edits (18.9% relative) than the matched
ASLLRP-only run. Both saved artifacts reproduced their metrics exactly after cold CPU
reload.

The combined result is retained only as a research candidate, not promoted for app
deployment or autonomous generation: 79.47% full WER and 1.78% full exact-sequence
accuracy remain weak. It demonstrates useful cross-corpus temporal transfer and shows
that optimization/forgetting is part of the Stage-2 failure, but genuine signer and
transition supervision remain inadequate. How2Sign/NCSLGR-generated combinations must
be exported only as provenance-tracked `synthetic` review candidates; native approval
is required before training use, and generated samples remain prohibited from
validation/test. The detailed comparison is in
`artifacts/reports/stage2_v17_multidata_ablation_v1/EXPERIMENT.md`. No Citizen, SemLex,
local sealed, RIT, 2M-Flores `devtest`, How2Sign validation, or How2Sign test split was
accessed.

## 2026-09-01 06:53 PST — unsafe phone selector withdrawn; full ASLLRP OTHER-CTC rebuild underway

The 02:30 two-head general selector promotion was invalidated by the subsequently
available physical-phone diagnostic recordings. On the five saved `I NEED HELP`
attempts it introduced empty/`WHERE`-biased behavior despite its validation gains.
It has been removed from the mobile target. The app is restored to
`stage2_v17_full_activity_routed_hybrid_v2`: multi-sign input uses the pinned bare
`Stage2PhraseV17FP32` head, while a single-sign output uses the repaired full-motion
Stage-1 plus isolated-correction route. The restored manifest pins checkpoint
`b15b06ff...` and Core ML package tree `0b83fc...`. Flutter tests, the native
activity-alignment test, signed Release build, installation, and launch on the
connected iPhone 13 pass. This supersedes the mobile-promotion claim below; it is not
a new physical-phone accuracy claim.

Two further validation-only shortcuts were implemented and rejected. Reclassifying
CTC-emission segments through Stage 1 collapses to 24/24 ASLLRP and 264/259 local
edits because CTC emission locations are not sign boundaries. Frame-level Stage-1
identity-logit fusion selects zero identity weight on training and leaves ASLLRP at
11/24 while worsening local validation to 8/259. Neither is a deployment candidate.

The highest-leverage Stage-2 data defect is now fixed at the preparation boundary.
The latest ASLLRP sentence metadata contains 17,519 valid annotations from 2,130
utterances, but the old preparation retained only 44 training clips because every
out-of-vocabulary annotation broke the continuous span. The new exact-variant,
signer-disjoint plan keeps full target-bearing natural spans and maps intervening
annotations to an explicit CTC `OTHER` token. It contains 1,104 bounded spans (879
train from BEN/CORY/RACHEL and 225 validation from JONATHAN), 1,483 locked-vocabulary
target tokens, 2,103 collapsed OTHER tokens, and 1,090 unique parent utterances. Crops
are split only between manual annotations and never exceed 256 frames/eight mobile
windows. The plan is at
`artifacts/reports/stage2_v17_asllrp_other_ctc_plan/plan.json`; targeted acquisition
is active under `data/local/asllrp_other_ctc_v17/` rather than downloading an entire
unrelated corpus.

The 101-class Stage-2 contract and warm-start trainer are implemented. A critical
index-boundary test now guarantees acquisition CTC IDs 1..101 are converted exactly
once into archive class IDs 0..100 before the existing loader reapplies the blank
offset. The selected 100-gloss head is extended with a neutral OTHER row while all
old logits remain bit-identical at epoch zero; replay distillation and source-balanced
sampling protect the existing local and sparse-ASLLRP performance during adaptation.
The natural ASLLRP target subset has no exact locked `I`, `NEED`, `HELP`, or `HELLO`
occurrences but has 51 training `WHERE` occurrences, so natural ASLLRP is capped at
40% rather than allowed to dominate adaptation. The other 60% replays local real
phrases, sparse target-only ASLLRP, and the selected train-only multivoice synthetic
pool. That pool covers all 100 classes and includes 194 `NEED` compositions. The
explicit phone-development gate replays the five saved `I NEED HELP` tensor captures;
the current bare head gets 0/5 exact and is the pinned pre-training baseline.
Nineteen focused model/data tests and Python compilation pass. No Citizen, SemLex,
local, RIT, or other test split was accessed.

Integrity clarification: a subsequent over-broad text search for candidate gloss
spellings across `data/local/dataset_metadata/asllrp_signbank/*.csv` printed rows from
the already local RIT external metadata file as well as the intended ASLLRP file. No
RIT video, feature tensor, model prediction, metric, or checkpoint selection was run,
and no printed RIT row is used by the OTHER-CTC preparation or training. The consumed
RIT external evaluation was not rerun. Future variant inspection is pinned explicitly
to `asllrp_sentence_signs_2025_06_28.csv`; the preceding sentence should therefore be
read as “no test evaluation or test media access,” not as “no metadata filename was
ever printed.”

The targeted ASLLRP acquisition and Apple Vision/RGB extraction are now complete.
Acquisition verified 1,090/1,090 parent utterances and 1,104/1,104 bounded spans with
zero failures. The finalized manifest hash is
`35bef0af546fb0ed7a600614a0828f33209219191a2be789813bfd018584b881` and converts
acquisition CTC IDs to archive class IDs exactly once. The fail-closed extraction audit
reports 1,104 expected/actual/audited archives, zero missing/unexpected/failing
archives, 4,505 total windows, zero invalid landmark windows, and 96.9138% mean valid
hand-view coverage. The audit is
`artifacts/reports/stage2_v17_asllrp_other_ctc/extraction_audit.json`.

Hand embedding remains in progress under the prior crash-safe resource contract. Test
batches 64 and 48 were rejected by the 8% MPS watermark at approximately 1.3 GiB
allocated; the worker exited before system pressure or corruption, and the watermark
was not disabled or raised. Batch 32 is stable with short subprocess lifetimes. The
fixed 256px MobileCLIP2 transform was also simplified from per-image
NumPy→PIL→no-op resize→tensor into vectorized NumPy→tensor. A direct parity check
against the pinned transform measured max absolute input difference 0.0, and a focused
unit test passes. Existing completed embeddings are hash/schema-audited and reused.

## 2026-09-01 01:19 PST — Stage-2 phrase evidence is coverage-limited, not proof of corrupt ASLLRP video

The real continuous-training manifest was audited at the exact-sequence, token, and
signer levels after the physical-phone Stage-2 failures. ASLLRP contributes only 44
training clips, 38 distinct two-gloss sequences, 92 target tokens, and three training
signers. Thirty-four of the 38 training sequences occur exactly once; only `HOME WHEN`
and `WHEN FRIEND` occur three times and `HAVE TIME` and `SCHOOL TOMORROW` occur twice.
The 12-clip ASLLRP validation set contains 24 tokens from the held-out signer JONATHAN,
but five of its eight exact sequences were never seen in training. `BAD` is not present
as a training target token at all. The bare phrase CTC result of 11/24 edits (45.8333%
WER) and 2/12 exact sequences is therefore evidence that the current Stage-2 corpus is
too sparse for reliable held-out-signer sequence learning; it is not, by itself,
evidence that the underlying videos or annotations are corrupt.

The local phrase corpus has the opposite limitation. It contributes 390 training and
97 validation clips but only six exact phrases. Every validation phrase and every
validation token occurs in training, with heavy repetition: `HELLO HOW YOU` and
`PLEASE HELP I` each have 107 training examples, while the other four phrases have
32--48. The validation split uses the same local signer pool, as previously permitted
for data expansion. Its 7/259 edits (2.7027% WER) and 91/97 exact sequences (93.8144%)
mainly measure interpolation over repeated phrase templates and recording conditions;
they are not evidence of unseen-signer or novel-phrase generalization. The ASLLRP and
local scores therefore must not be compared as though they were equally difficult
accuracy tests.

Stage 2 is a CTC gloss-sequence recognizer rather than a phrase-ID classifier, so one
example of every exact sentence is not inherently required. It does, however, require
repeated gloss evidence across varied temporal contexts, transitions, durations, and
signers. The current ASLLRP subset is also deficient on those axes, while the local set
has repetitions but almost no phrase/context or signer diversity. ASLLRP remains useful
as a small hard supplemental/held-out diagnostic and should not be discarded, but it
cannot serve as the main training corpus or the sole model-selection gate.

The high isolated-sign validation results across Citizen, SemLex, and local data show
that Stage 1 learned substantial signer/style/camera invariance for the 100 pinned
classes; they do not prove that arbitrary lexical variants are interchangeable.
Project mappings still pin exact raw gloss/ASL-LEX variants and do not merge numeric
variants. A true variant mismatch can remain mislabeled even when a model handles
ordinary signer accent/style variation. The next high-leverage action is genuine
continuous data collection for a bounded functional phrase inventory, with repeated
transitions from multiple signers and a signer-held-out validation partition. No test
split was accessed during this audit.

## 2026-08-24 14:49 PST — exact MobileCLIP2 hand-crop tower exported for mobile Stage 2

The exact frozen MobileCLIP2-S0 visual tower used by the retained 100-gloss teacher is
now exported as an FP32 Core ML image model at
`artifacts/coreml/MobileCLIP2S0ImageEncoderV17FP32.mlpackage`, tree SHA-256
`9309548dd69a5c8e899ea00ee4f0bbe88505ed803d12520a60e5d954ff370974`, size
45,580,737 bytes. It consumes a 256x256 RGB image with pixel scale 1/255 and emits the
same normalized 512-D hand embedding used in training. The source MobileCLIP2-S0
checkpoint remains pinned at SHA-256
`ab91a1a0c4330d6b1913e24d5035dfdea15423316aaec649610c6b1c6ddd0e95`.

Full class-spanning parity covered one decoded crop from every one of the 378 Citizen
validation clips. Core ML versus the unreparameterized PyTorch tower has maximum
absolute error `1.49012e-06` and minimum cosine `0.999999881`; versus the historical
float16 embedding cache it has maximum absolute error `0.000226140` and minimum cosine
`0.999999821`. The 14.91 ms median and 18.74 ms p90 are Mac-host Core ML timings only,
not iPhone, ANE, thermal, or complete-pipeline evidence. The detailed report is
`artifacts/reports/mobile_100gloss_v17/mobileclip2_image_fp32.json`. No Citizen,
SemLex, or local test split was accessed.

This closes the missing crop-to-embedding model export only. Validation of freshly
regenerated RGB embeddings through both Stage-2 Core ML packages, app integration,
orientation-safe simulator execution, and the Stage-2-to-Stage-3 contract remain open.

## 2026-08-16 13:57 PST — new real-gloss Stage 2 source found and audited

The new continuous-ASL dataset search found one immediately actionable supervised
source: Meta's 2M-Flores-ASL. Its videos have human-created sentence glosses plus an
additional expert harmonization pass. A metadata-only audit read all 999 rows of the
official `dev` split and deliberately did not access `devtest`. Of those rows, 811
contain at least one locked Citizen-100 lexical label and collectively cover 95/100
labels; the missing labels are `GOODBYE`, `PLEASE`, `SORRY`, `SAD`, and `TOMORROW`.
The split contains 4,388 normalized gloss tokens, so it must be trained with an
expanded Stage 2 vocabulary. Deleting out-of-vocabulary tokens to manufacture
Citizen-100-only transcripts is forbidden because it would corrupt sequence order and
CTC timing. The dataset's signer field is only a local ID (`0` on 997 rows and `1` on
two), not a global identity, so this source cannot establish a new signer-disjoint
claim.

The complete row-level audit is
`data/local/dataset_metadata/2m_flores_asl/dev_locked100_audit.json`, SHA-256
`4dcb426bab947fdd455a364ede8c7039c10518316bd031f709712f2ee18d7130`.
The ranked evidence is recorded in
`artifacts/reports/stage2_v17_new_dataset_search/NEW_STAGE2_DATASET_SHORTLIST.md`
(SHA-256 `b4a65a67bc68a119936c599124fc7b8442321a323bf76f8a759c6d965294f937`)
and `shortlist.json` (SHA-256
`b4bad7843e2d371467142db1d1d42ca81376a00f9ad3e6a6a9832b7033f0fd97`).

ASL-Homework-RGBD is the second-ranked source: 935 continuous videos from 45 signers
(24 fluent and 21 learners) with ELAN gloss/nonmanual/error annotations, but the full
volume requires authorized Databrary access. Apple's newly reported ASL STEM Wiki
annotations rank third: nearly 500 professionally annotated videos and 8,655 sign
annotations, but no downloadable official annotation bundle was located. How2Sign,
the base ASL STEM Wiki/FLEURS-ASL releases, OpenASL, YouTube-SL-25, and the ASL portion
of SignNet-1M do not currently expose verified ordered ASL gloss targets suitable for
this Stage 2 CTC trainer. Their RGB may later support translation, representation
learning, or separately governed weak supervision, but they are not substitutes for
real gloss sequences.

Raw 2M-Flores is approximately 326 GB total and its `dev` videos are approximately
156 GB, so no bulk download was started during discovery. The next safe action is a
resumable one-file-at-a-time `dev` acquisition that records source hashes, performs an
aspect-preserving 720p30 transcode, verifies duration/decode and derived hashes, then
releases each temporary source MOV. The compressed videos must remain available for
RGB crops. `active/v17/stage2_data_sources_v17.json` is updated to version 2 with this
acquisition order. Two audit tests, JSON validation, and Python compilation pass; the
Python compilation, generated JSON validation, both focused audit tests, and
`git diff --check` pass. Citizen, SemLex, local, and 2M-Flores sealed test splits were
not accessed.

## 2026-08-14 14:12 PST — Stage 2 source audit complete; first real external subset acquired

Stage 2 data gathering is now pinned in
`active/v17/stage2_data_sources_v17.json` (SHA-256
`b5f2d285d6f2aeb6bcc730426f531c7ba7676ec90739098587665b00c67d3b47`). The local
`data/raw_videos/PHRASES` corpus contains 780/780 valid, SHA-unique 640x480 clips
across nine fixed phrases and totals 2,226.700004 seconds (0.6185 hours). It has no
usable signer metadata. Under the frozen 100-class vocabulary, 520 clips are strictly
eligible after the safe `ME -> I` normalization. A further 60 clips require an
explicit semantic decision before treating `FOOD` as `EAT`; this mapping is not
silently approved. The remaining 200 contain out-of-vocabulary `LATE`, `TEACHER`, or
`MEET`. Existing phrase and 15,000-sequence synthetic arrays use legacy 61-node
v16-era schemas and must not be used by v17; synthetic sequences must be regenerated
from hash-pinned, train-only v17 isolated archives.

The public NCSLGR/SignStream static subset has been acquired to
`data/local/ncslgr_continuous_v17_source`: 166/166 compressed frontal videos and
166/166 frame-aligned SignStream annotation files, with all sizes and SHA-256 hashes
verified. The videos total 34,631,570 bytes and 8.8591 minutes, are 324x312, and cover
two normalized participant IDs. Of the 166 utterances, 132 contain at least one target
gloss, totaling 198 target-gloss occurrences across 17 of the 100 classes. This is
real supervised continuous-sign data but is low-resolution, narrow-coverage
supplemental evidence, not the primary Stage 2 corpus. Its manifest SHA-256 is
`c03ff8a5b13c7fa7da9b7ac211ab15b66719768a152c4b0d66b96529b6660744`.

The modern ASLLRP DAI public catalog was queried against all 100 frozen labels across
all 47 BU collections, four RIT sources, and all exposed participant records. Exact
normalized coverage is 76/100 labels with 2,313 occurrences and zero query failures.
This is the best identified broad, exact-gloss supervised source, but bulk video/XML
acquisition requires an authenticated ASLLRP DAI account; access control will not be
bypassed. Official manually realigned How2Sign train/validation metadata was also
acquired and hash-pinned: 32,906 English-aligned sentences. Its official release does
not supply the ordered gloss targets required by the current CTC design, so English
word hits are only weak-label evidence and the approximately 33 GB RGB download is
deferred pending a separately locked self-supervised or translation objective.

The exhaustive local/source report is
`artifacts/reports/stage2_v17_data_audit/audit.json` (SHA-256
`09c1dd1d191584184ff877b5862969ae4ee6b8ce96dde7210ebf1711b910992c`). Acquisition
and audit tooling lives in `scripts/acquire_ncslgr_stage2_v17.py` and
`scripts/audit_stage2_phrase_sources_v17.py`. Four focused parser/fail-closed
vocabulary tests pass; both scripts compile; all new JSON files validate. No Citizen,
SemLex, or local sealed test split was accessed. The next safe actions are full-length
v17 extraction and near-duplicate grouping of the local phrases, a v17 extraction
quality gate on NCSLGR, authenticated modern ASLLRP acquisition, and regeneration of
v17 synthetic sequences.
