# signing-voice — log

Measured results, rejected approaches, and progress snapshots. Not read start-to-end —
`rg` this file before re-running an experiment to see if it already failed.

17 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## 2026-09-01 14:31 PST — source-balanced all-real transition adaptation improves local motion without guard regression

The remaining downloaded domain reference is now compatible too. The three retained
OpenASL train-split clips were converted by
`scripts/prepare_openasl_transition_manifest_v17.py` and the standard extractor with
zero failures. One channel-proxy clip is train-role and two are validation-role; all
three preserve their original acquisition split, and OpenASL validation/test video was
not accessed beyond those already retained train-split derivatives.

`TransitionWindowDataset` now optionally reads either continuous-transition archives
or the existing Stage-2 multimodal v17 archives, filters roles and sources before
preloading, and accepts sources without signer metadata. Existing callers retain their
old narrow glob and behavior. `active/v17/train_transition_all_real_v17.py` reuses the
frozen 2.02M-parameter transition inpainter and adapts it with equal probability over
six genuine corpus families: all-nine local phrases (1,781 train windows), ASLLRP old
+ `OTHER` (3,761), 2M-Flores (2,686), How2Sign (4,938), one NCSLGR training signer
(424), and YouTube/OpenASL web motion (810). No synthetic trajectory participates.

Model selection uses 445 familiar-source local validation windows spanning all nine
phrases, 801 signer-held-out ASLLRP `OTHER` windows, and 137 windows from the held-out
NCSLGR signer. The full 4,938 How2Sign and 802 train-side YouTube windows are replay
guards; a candidate is rejected if any selection/guard domain regresses more than
0.005 relative reconstruction improvement from the frozen initialization. The 15
OpenASL validation windows are reference-only because that sample is too small for
selection.

The bounded run stopped after eight epochs by patience and selected epoch 4. Relative
masked-reconstruction improvement over endpoint interpolation changed from 16.07% to
20.74% on local (+4.67 points), 23.08% to 23.50% on ASLLRP, 20.41% to 21.27% on
held-out NCSLGR, 25.87% to 25.41% on the How2Sign guard (within the predeclared
0.5-point tolerance), 10.15% to 10.74% on the web guard, and 0.43% to 1.79% on the
tiny OpenASL reference. The selected checkpoint is
`artifacts/models/transition_all_real_v17_v1/model.pth`, SHA-256
`0faaa4f8289257fed02ba9f17c1fc1b5fd8035f921ff500292644efd2eb44123`;
training took 71.0 seconds. This is materially better genuine local-motion
reconstruction and demonstrates that using all sources with explicit balance works.
It is still a masked 4--12-frame motion model, not gloss-conditioned full-phrase
generation or a human-naturalness pass. Four focused audit/dataset tests pass. No
project or public test split was accessed.

## 2026-08-22 17:32 PST — first complete content-gated AI signing voice delivered

The promoted signing voice is **not** the unconstrained neural residual decoder.  That
route was rejected because its emitted-style AUC improved only by worsening held-out
geometry 5–8%.  Its code and diagnostic artifacts remain research-only.  The final
system is a compact, interpretable 16-D profile latent learned from 63 train-only
voices.  It factors each real trajectory against its class medoid, learns robust
per-node XYZ signing-space offsets from other glosses, mixes at least three voice
latents to create a new voice, restores mixed signer duration, and uses the frozen
transition timing/inpainting stack for phrase boundaries.  At inference, the frozen
Stage-1 landmark branch selects the strongest style strength in
`[1.0, 0.75, 0.50, 0.40, 0.25, 0.0]` that retains the requested class.  Thus style
cannot silently overwrite content.

The frozen 16-D/content-gated design was evaluated unchanged across three disjoint
folds: 21 entire held-out train-only identities and 2,673 clips.  Aggregate generated
content is 93.9394% versus 93.6775% for the unstyled medoids; spatial reconstruction
improves 15.1171%; velocity and acceleration are unchanged to floating-point noise;
and the strict same-content signer verification AUC is 0.5900.  That AUC ranks the
target signer's other-gloss profile against other signers while holding the exact
requested gloss/prototype fixed.  The hash-pinned summary is
`artifacts/reports/signing_voice_profile_signer_disjoint_summary_v17.json`, SHA-256
`535a3abc5bd70979a24507a9731879b459e7cd05d12754d4b91e28ccf7585d33`.
This is credible content/style generalization evidence, not a fluent Deaf-signer
naturalness or linguistic-correctness judgment.

The fixed-design train-all profile uses all 63 eligible identities and 3,974 examples;
no model selection was performed on it.  It is
`artifacts/models/signing_voice_profile_v17_allvoices_final/model.pth`, SHA-256
`eec0ab97a7dea26fa2c53ec4d2afd3e27f530e8395db6e6115fd838859efbcae`.
Its result report SHA-256 is
`a1cb006e7b4dc350399cce2f7c7e91ac246059dbb13b94c1a25ea8406e42ab3f`.

The requested visual evidence is a complete generated `GOOD MORNING FRIEND` phrase in
three novel voices (Aster, Cobalt, and Juniper), not a hidden interval inside a human
trajectory.  Every voice uses a distinct convex mix of three learned voices; maximum
cosine between the three decoded profiles is 0.1277 and maximum cosine to any training
profile is 0.4315.  All nine generated isolated tokens retain full style strength and
the frozen Stage-1 branch predicts all nine correctly.  Phrase lengths are 79, 81, and
88 frames, and learned boundary spans differ across voices.  The 1920x900, 30-fps,
266-frame H.264 video is `artifacts/reports/signing_voice_phrase_v17.mp4`, SHA-256
`01f083a9fba3e7480e24c7b701c3bebbfbbe0daf709a36149a1dba8847822c4d`.
The preview is `artifacts/reports/signing_voice_phrase_v17_preview.png`, SHA-256
`6c0ad7a22dc6f3e7154d41bf190dbad2771a12e7da835f427fb15d17256251ba`;
the provenance report SHA-256 is
`e2bd22040ba1cf25fd189447e3747bea48ffe0e736b596cc888a1a855ea82d01`.
This is an abstract articulated avatar driven by generated landmarks, not photorealistic
RGB synthesis.

Cold reload reproduces all three raw phrase trajectories within float16 tolerance,
repeats all nine correct content predictions, verifies every checkpoint/report/media
hash, and decodes all 266 video frames.  The cold-reload report is
`artifacts/reports/signing_voice_profile_package_cold_reload_v17.json`, SHA-256
`b6fceebcca79c83af9e741d29109ebf4d841ffbad2e85d4654396e06d9d71cf3`.
Thirty-one affected signing-voice, transition, and Stage-2 tests pass; all changed
entry points compile, final JSON parses, full video decode passes, and
`git diff --check` passes.  No Citizen, SemLex, local, How2Sign, or project validation/
test split was accessed.

The earlier transition inpainter remains separately recorded as a promising
extractor/Stage-1 hard-gloss augmentation: hide a difficult real interval, reconstruct
it from genuine context, and retain the sample only when the class label and integrity
checks remain unchanged.  It is not part of the 16-D voice profile itself and has not
yet been promoted into Stage-1 training.

## 2026-08-22 16:57 PST — direct emitted-style supervision added

The first strict content-matched fold-0 diagnostic was interrupted after epoch 7: it
improved reconstruction but emitted-style AUC remained near chance (0.526–0.532), so
it is correctly rejected as a signing voice.  The missing loss is now explicit and
mirrors the evaluation contract.  A generated gloss is pulled toward its real target
from the same signer and exact gloss; only same-gloss/different-signer batch rows act
as negatives.  The original cross-gloss real-reference loss remains responsible for
learning a content-independent conditioning style.  The new emitted-style loss has
weight 0.25 in both held-out and final train-all scripts.  Nine focused tests pass,
and fold 0 has restarted from scratch.  No interrupted checkpoint is eligible and no
sealed or project validation/test split was accessed.

## 2026-08-22 16:54 PST — emitted-style metric is now content-matched

The first three-epoch rerun was also interrupted and is ineligible after detecting a
class-frequency confound: different held-out identities cover different gloss subsets,
so comparing generated and real embeddings across arbitrary classes lets content help
identify the signer.  The final verification contract now compares each generated
gloss only against real targets of the **exact same gloss**: its aligned same-signer
target is positive and all available same-gloss/different-signer targets are negatives.
Rows without a same-gloss cross-signer negative are excluded from both sides.  This
simultaneously verifies generator output, signer style, and content control.  Eight
focused tests pass, and fold 0 has restarted from scratch.  No checkpoint from either
interrupted diagnostic run is eligible; no sealed or project validation/test split was
accessed.

## 2026-08-22 16:38 PST — signer-aware signing-voice metric correction

The first contrastive fold-0 run was deliberately interrupted after epoch 8 because
its validation negative used the next row in a batch, which was not guaranteed to be
a different signer.  No checkpoint from that interrupted run is eligible.  Style
verification now compares every aligned same-signer/different-gloss reference-target
pair against exhaustive cross-signer pairs and records both pair counts.  A new
contract test verifies that same-signer off-diagonal pairs never enter the negative
set; all five signing-voice tests pass.  Future checkpoints also store the 100
train-only class-median observed durations required for phrase-level timing.  No
sealed or project validation/test split was accessed.

The phrase-level implementation now has a fail-closed composition contract.  It
creates every requested 32-frame gloss from the fixed class prototype plus one shared
continuous style latent, restores train-only class/signer timing, predicts each
boundary span from the generated endpoints, synthesizes the entirely missing
coarticulation interval with the frozen train-all transition inpainter, and concatenates
the result into one complete trajectory.  A “novel voice” must be a normalized convex
mix of at least three unique learned voice centroids, with no source weight above
0.60.  Seven focused tests now pass, including full-phrase boundary/timeline and novel
style-mixture contracts.  The train-all script and a three-avatar same-phrase renderer
are implemented but must not be presented as final evidence until the corrected
signer-disjoint folds, fixed-epoch train-all run, cold reload, and video validation
finish.

## 2026-08-22 16:15 PST — true content/style signing-voice experiment started

The earlier transition inpainter is explicitly retained as a separate candidate for
continuous-trajectory augmentation, including future hard-gloss Stage-1 experiments.
It must not be renamed a signing voice: it reconstructs only a missing interval inside
an existing human trajectory.

The new signing-voice task generates a complete 32-frame isolated gloss from two
separate inputs: a requested class prototype supplies content, while a reference clip
of a **different gloss from the same signer** supplies style.  The style encoder never
receives the reference label.  This makes content copying through the reference a
fail-closed data-contract violation and establishes the required content/style split
for later full-phrase composition and novel latent-style interpolation.

The existing 67-identity train-only feature pool has now been materialized in raw v17
landmark space with exact index alignment: 3,978 trajectories, all 100 classes, 1,475
Citizen official-train items, 1,115 contextual ASLLRP train items, and 1,388 SemLex
official-train items.  Sixty-three dataset-local identities have at least two distinct
classes and are eligible as style voices.  The pool is
`data/local/signing_voice_v17/train_only_landmark_pool.npz`, SHA-256
`0764c4295524d48417f2fd89058c7037c65ccb1d477223104a2a84be61435702`;
its report is `artifacts/reports/signing_voice_v17/landmark_pool.json`.

`active/v17/model_signing_voice_v17.py` implements a continuous 32-dimensional style
encoder and a content-prototype residual Transformer.  Presence/confidence and missing
nodes remain tied to the class prototype; only XYZ style motion is generated.  A
frozen selected Stage-1 landmark branch enforces content preservation.  The first
three contract tests pass.  A two-epoch smoke selected epoch zero, correctly rejecting
premature style perturbation; the predeclared fold-0 signer-disjoint pilot is now
running with seven entire identities held out (three Citizen, three SemLex, and
RACHEL).  At epoch 8 it first exceeded the medoid baseline while retaining 100%
generated-content recognition.  No sealed split or held-out project validation signer
was accessed.

## 2026-08-22 15:46 PST — 133-source/proxy transition-voice package complete

The held-out-fold settings are now being converted into train-all deployment
artifacts without new model selection.  The final deterministic mean used all six
How2Sign train-shard signers plus all 127 usable YouTube-ASL channel-level voice
proxies, with the frozen 90%/10% source balance, LR `5e-5`, and median 63 selected
epochs.  It trained on 4,938 How2Sign and 994 web windows and is
`artifacts/models/transition_inpainter_multicorpus_v17_allvoices_final/model.pth`,
SHA-256 `eba3bbd5086c04f099e4466ac211a7b69322a0dd1f29d680a3134982a4cc7e2a`.
This train-all checkpoint inherits the three-fold held-out evidence; its own training
loss is not an independent accuracy score.

The final 10-epoch stochastic residual layer used the same 133 sources/proxies and
source-balanced sampler.  Its residual normalization is the root of the exact
90%/10% weighted source second moments, avoiding a hidden corpus-size bias.  It is
`artifacts/models/transition_residual_diffusion_multicorpus_v17_allvoices_final/model.pth`,
SHA-256 `0618f7d6976219a6d3180eb24dffdd2570e79bef4437107e23fc348561f1f5d5`,
and pins the deterministic checkpoint hash exactly.  Recommended temperatures remain
0.10 and 0.20 from prior LOSO selection.

The all-voice timing predictor then completed the frozen 39-epoch train-all schedule
over 53,388 balanced span examples.  It is
`artifacts/models/transition_span_multicorpus_v17_allvoices_final/model.pth`, SHA-256
`b752ecf1bebbb6e82ccc803e86beee40adf771fd56750da3811363b0f3c1c555`.
It inherits the three-fold elapsed-span evidence; its own training loss is not an
independent semantic-timing score.

An audit of a suspected timing-feature leak confirmed there is none: v17 channel 5 is
per-frame landmark confidence, not motion derived from an earlier hidden frame.  The
timing model receives only XYZ, presence, and confidence from the visible eight-frame
context on each side.  Twenty affected Stage-2/transition tests pass.  The completed
runs used `num_workers=0` and the 10% MPS process cap; observed RSS stayed below
0.5 GB and system free memory was 45–52%.

All three components cold-reload together from disk on a real extracted How2Sign
trajectory.  The mean is finite and preserves every visible value exactly; stochastic
outputs at 0.10/0.20 are finite, nonzero, bounded, and preserve visible context; and
the timing checkpoint emits finite nine-class logits and a valid 4–12 frame span.
The cold-reload report is
`artifacts/reports/transition_multivoice_package_cold_reload_v17.json`, SHA-256
`734576a216e69d4737c8effa01053a8d1e40aafafe2d260f9f1a468b7b84aed3`.
The canonical hash-pinned package/evidence manifest is
`artifacts/reports/transition_multivoice_package_v17.json`, SHA-256
`5e99828f31390cfb89fc18352e1051592e7e30379bb8e72a26cd963eb8422c4b`.
It records 6 controlled How2Sign signers plus 127 channel-level voice proxies, not 133
identity-verified unique people.  Model sizes are 7.7 MB mean, 8.3 MB diffusion, and
1.7 MB timing.

All 20 affected Stage-2/transition tests pass; all changed entry points compile, all
final JSON parses, all stored hashes/linkages match the bytes on disk, and
`git diff --check` passes.  This completes a generalizing landmark-space experiment,
not a human-naturalness claim: semantic prosody, RGB rendering realism, and blinded
Deaf-signer preference remain unmeasured.  No Citizen, SemLex, local, How2Sign
validation/test, 2M-Flores `devtest`, or consumed RIT test row was accessed.

## 2026-08-22 15:25 PST — signer-context timing passes three held-out folds

Natural transition timing is now measured separately from motion inpainting.  The
new task removes a genuine 4–12 frame interval from a real continuous 32-frame
trajectory and presents only eight visible frames on each side.  The model must
recover elapsed span length without receiving the mask width.  Every source window
contributes all nine balanced target lengths.  The same two-layer, signer-ID-free
context transformer, 10% web balance, seed, and 40-epoch schedule were frozen on
signer 8 and applied unchanged to signers 3 and 5.

Across 42,174 held-out How2Sign examples (the same 4,686 unseen-signer windows), exact
span accuracy is 92.3887%, macro F1 is 0.924508, within-one-frame accuracy is
96.1849%, and MAE is 0.1721 frames.  Across 5,184 channel-held-out web evaluations,
exact accuracy is 87.1528%, macro F1 is 0.876557, within-one-frame accuracy is
91.6860%, and MAE is 0.3258 frames.  Fixed eight-frame timing is 11.1111% accuracy
and 2.2222-frame MAE.  A direct boundary-distance/observed-speed rule is about 12%
accuracy and 3.63–3.73-frame MAE.  Crucially, repeating only the two endpoint poses
while removing local temporal style collapses the trained model to about 11%
accuracy and 2.85–2.88-frame MAE.  Thus genuine local rhythm, not endpoints alone,
drives the result.

The fold checkpoints are
`artifacts/models/transition_span_multicorpus_v17_h3_w010/best_model.pth`
(`fdca6a12e7e4dd64b9234e779b2ea0af9083d4cc6a82b343743ab422ec996257`),
`...h5...` (`085733a44f5219ad05e8b479d216c148e43f911407ef12f48a85c725ccedf9b4`),
and `...h8...` (`1bd5016279272449a1e94e43a39d6410e93cbd276d4e49f1cd7868f54a52ccc2`).
The aggregate report is `artifacts/reports/transition_span_loso_summary_v17.json`,
SHA-256 `e2a71e66575a55d939dd2989d9008b485e92c91b84fcd23c66d64f316fcb780b`.
Eleven focused transition/timing tests pass.  This is strong self-supervised elapsed-
span evidence; it is not semantic prosody or a human-perceptual naturalness result.
No sealed split was accessed.

## 2026-08-22 14:52 PST — 127-channel transition adaptation passes three-fold reconstruction gates

The bounded YouTube-ASL acquisition is complete without downloading the 984-hour
corpus.  It retained 128 hash-pinned 30–38 second derivatives from 128 distinct
public channels (104 train, 24 channel-disjoint internal validation), rejected 11
uncuttable candidates, occupies 72 MB, and preserves each source aspect ratio.  Two
spaced visual contact sheets showed active signing in all 24/24 sampled clips.
`active/v17/youtube_asl_transition_manifest_v17.json` has SHA-256
`4865a62dc44a4b546c130bfa21ed27763538897ad6b86ff312c9d4b8ce4f09f3`;
all 128 file hashes match and there is no role overlap.  Channels are explicitly
voice proxies, not verified one-to-one signer identities.

The fixed Apple Vision v17 extractor completed in 349 seconds with stable sub-1-GB
RSS.  It produced 127/128 usable voice-proxy archives: 103 train voices/802 valid
windows and all 24 validation voices/192 valid windows.  The sole failed train clip
had no valid hand window.  The landmark tree is 9.7 MB, SHA-256
`c2b3ce5a1963e24cdbd7189bb5c9200155dfcd4233097d850c53eae37d2e0ce8`,
and passes the enforced 80-train/16-validation breadth floor.  Extraction and audit
reports have SHA-256 `77ef1023be0ea1e8807ed441950e4e66475ef7a2deb3aac11f16ef3ca99dee59`
and `24bf947e5949e40fdb095e68055c7097209953bbf5c4e8e0d449bf66c2c0b6e5`.

Multi-corpus adaptation warm-started each exact How2Sign-only fold checkpoint and
used source-balanced sampling with a zero-regression held-out How2Sign selection
floor.  Web probabilities 10%, 25%, and 50% were tested on signer 8; 15% and 20%
refined the only promising interval.  Although 50% maximized reconstruction, it
regressed the How2Sign grouped discriminator.  Ten percent was frozen because it was
the only selection candidate improving reconstruction and both discriminator
statistics on both signer-8 domains.  That exact 10%, LR `5e-5`, seed, architecture,
and schedule were then applied unchanged to signer 3 and signer 5.

All six fold/domain reconstruction comparisons improve.  Across 4,686 held-out
How2Sign windows/971 clips, relative reconstruction improves from 18.1259% to
20.2012%, the improved-window fraction rises 68.9714% to 69.8250%, discriminator
balanced accuracy falls 61.4810% to 61.1289%, and AUC falls 0.653906 to 0.648197.
Across the same 24 held-out web channels evaluated by the three independent fold
models, reconstruction improves 8.7223% to 10.9495%, improved-window rate rises
68.2292% to 71.7014%, and AUC falls 0.580539 to 0.570041.  Web balanced accuracy is
the one mixed metric: 55.2083% to 55.3819% (+0.1736 point), so it must not be reported
as an across-the-board distribution win.  The pinned aggregate is
`artifacts/reports/transition_inpainter_multicorpus_loso_summary_v17.json`, SHA-256
`a1df378f741fe758298a57c00329c356c52d29cdb1bfe09f32d029a55a659084`.
This is strong multi-voice landmark evidence, not human-perceptual naturalness.
No sealed split was accessed.

## 2026-08-22 13:28 PST — genuine web-voice expansion is in progress

The six-voice How2Sign artifact is no longer being treated as the endpoint.  A
source audit tested OpenASL first.  Its pinned public train TSV and signer boxes are
SHA-256
`e7e1559bcef5ac77d2c14c2ccfc9db54516768e296235f266b3f4f96de459a40`
and `a79b5327956670db0988bacd96aebb9229ecd5ea948b8de887cb530669152a44`.
All 2,007 eligible 4–12 second train-source videos were queried.  They resolve to
only three public channels (`Sign1News`, `The Daily Moth`, and `nad1880`).  This
matches the literature warning that OpenASL's approximately 220 signer identities
come from finer-grained source metadata that is not present in the public TSV.
Therefore OpenASL channel IDs cannot honestly prove a large voice count.  Three
visually valid clips were retained for possible domain work, but OpenASL was rejected
as the primary voice-expansion evidence.

The replacement source is the official, human-filtered YouTube-ASL video-ID release.
Its generation-pinned list contains 11,096 unique IDs and has SHA-256
`ca5622737279afc33b1f9ddfa585b5e8bd284f1be458c461e10887b03e519191`.
Unlike OpenASL, 195 distinct public channels were found after only 336 deterministic
ID probes.  Acquisition targets 128 channel-level voice proxies: 104 internal train
and 24 channel-disjoint internal validation.  One clip per channel is retained.  The
initial 2–10 second smoke sample exposed an opening title without hands; the contract
was corrected to 30–38 seconds and the same source then visibly contained continuous
signing with face and both hands.  Every derivative preserves source aspect ratio,
uses a maximum 720-pixel side, and is normalized to 30 fps.  At this timestamp 22
channel clips are complete; acquisition is resumable at
`data/local/youtube_asl_transition_subset_v17/acquisition_state.json`.

New reproducible entry points are
`scripts/acquire_openasl_transition_voices_v17.py`,
`scripts/acquire_youtube_asl_transition_voices_v17.py`,
`scripts/prepare_youtube_asl_transition_manifest_v17.py`, and the role-preserving
continuous extractor `scripts/extract_how2sign_transition_landmarks_v17.py`.
`active/v17/train_transition_inpainter_multicorpus_v17.py` implements a
signer-ID-free, source-balanced sampler and joint model selection on one unseen
How2Sign signer plus channel-disjoint YouTube-ASL voices.  It is not allowed to
replace the six-voice model unless both domains improve.  Its sampler and split
invariants are covered by two new focused tests; all eight transition tests pass.
No project sealed split was accessed.

## 2026-08-22 12:42 PST — multi-voice transition layer packaged after stochastic LOSO

The bounded stochastic residual experiment is complete across the same three
How2Sign leave-one-signer-out folds as the retained deterministic inpainter.  The
diffusion model predicts only XYZ residual motion around the frozen deterministic
mean, only inside a contiguous transition mask.  It receives visible motion context
but no signer ID.  All six domain-matched How2Sign train-shard voices participate
across the folds; the two NCSLGR voices remain a separate domain because direct equal
pooling had already regressed unseen-How2Sign evidence.

The two operating temperatures were frozen on signer 8 and applied unchanged to
signers 3 and 5.  Across 4,686 unseen-signer windows/971 source clips, deterministic
mean motion scores 61.4810% balanced accuracy / 0.652976 macro ROC AUC under the
grouped genuine-vs-generated discriminator, versus 67.9684% / 0.730314 for linear
interpolation.  Temperature 0.10 retains a 15.9245% weighted reconstruction
improvement over linear, improves 57.5758% of individual windows, is 2.6963% worse
than the deterministic mean, and scores 61.2356% / 0.644220.  Temperature 0.20 trades
more reconstruction fidelity for diversity: 12.0351% better than linear, 46.6069% of
windows improved, 7.4755% worse than the mean, and 60.3073% / 0.631665.  The pinned
fold report is
`artifacts/reports/transition_residual_diffusion_loso_summary_v17.json`, SHA-256
`f619d94f1d1977bcc5ca9f428af8192cc4a8e5bf48fdbc89320e643cbc5f0e3a`.
Temperature 0.10 is the accuracy/diversity mode; 0.20 is the stronger-diversity mode.

Fixed train-all artifacts were then fit on 4,938 windows from all six How2Sign
train-shard voices, without model selection or independent scoring on those same
rows.  The deterministic mean is
`artifacts/models/transition_inpainter_v17_all6_final/model.pth` (7.7 MB), SHA-256
`7e99aa7b3d8723d47c610230c9ddb87931809cd8f3b3ab7e888013ef5ca2a1bd`;
the 10-epoch stochastic residual layer is
`artifacts/models/transition_residual_diffusion_v17_all6_final/model.pth` (8.3 MB),
SHA-256
`05f8b7aedc69db94a6abe3b32c1d556749c4d1e627da7a6ab8b2dcd7dcd818d9`.
Both used `num_workers=0` and a 10% MPS process cap.  Peak observed RSS was about
0.49 GB for the mean fit and 0.82 GB for diffusion, with system free-memory pressure
at 45% or better; neither run leaked or approached red pressure.

A cold reload from disk on a real extracted trajectory passed at both temperatures:
outputs were finite, every visible context value was preserved exactly, and masked
XYZ deviations were nonzero and inside the hard residual bound.  The verification is
`artifacts/reports/transition_voice_artifact_cold_reload_v17.json`, SHA-256
`71fbd77fcb059cdd50e1b747b34f8a585449b4c3545c3264a09baf0c0703c0e1`.

All 15 affected Stage-2/transition unit tests pass.  Every new Python entry point
compiles, all final model/result/history and aggregate-report JSON files parse, stored
checkpoint hashes and mean-to-diffusion linkage match the bytes on disk, and
`git diff --check` passes.

This establishes a generalizing, context-conditioned **landmark transition layer**—a
useful component of the requested signing “voice.”  It is not text/gloss-conditioned,
does not render RGB, and does not establish human-perceptual naturalness.  A fluent
Deaf-signer blinded preference study on rendered held-out-signers remains mandatory
before calling it genuine human-natural signing.  It also does not change the Stage 2
recognizer's WER.  No Citizen, SemLex, local, How2Sign validation/test, 2M-Flores
`devtest`, consumed RIT test, or JONATHAN data was accessed.

## 2026-08-22 12:07 PST — full leave-one-signer-out transition evidence

Full extraction is complete.  How2Sign yielded 1,026/1,027 usable train-shard clips;
the sole exclusion, `0HfN3Ts0FxQ_18`, was retried and visually inspected and contains
no usable Apple Vision hand window (the signer is resting with hands below/in the lap).
The already acquired NCSLGR continuous subset added 166/166 usable clips from two more
human signers.  The combined tree has 1,192 archives, eight signer identities, 5,499
valid 32-frame windows, seven invalid windows excluded by saved masks, and SHA-256
`79cc83c2c2ff711f505b9af1ceca5271386287b676a98737e65e74e07c487806`.
The complete corpus audit is
`artifacts/reports/transition_inpainter_full_corpus_v17.json`.

Directly pooling the two cross-corpus NCSLGR voices was rejected for the How2Sign
held-out-signer task.  On signer 8, the eight-voice model improved reconstruction by
15.9577% and scored 62.7939% balanced accuracy / 0.666003 ROC AUC under the grouped
genuine-vs-generated discriminator.  The otherwise identical How2Sign-only model
improved by 17.6497%, improved 72.1725% of windows, and reduced discrimination to
61.8141% / 0.655865.  NCSLGR remains useful as a separate genuine domain; it must not be
pooled at equal weight without source balancing or domain-aware adaptation.

The retained residual design was then evaluated leave-one-signer-out across all three
large How2Sign voice pools, always training on the other five How2Sign signer IDs and
never using learned signer-ID embeddings.  The same seed and hyperparameters improved
all three unseen signers:

- signer 3: 840 windows/172 clips, 15.4908% reconstruction improvement (95% bootstrap
  CI 12.6802–18.3631%), discriminator 59.9405% balanced accuracy / 0.648756 AUC versus
  linear 65.2381% / 0.705816;
- signer 5: 2,060 windows/399 clips, 19.6131% (CI 17.7347–21.5042%), discriminator
  61.8204% / 0.654308 versus linear 69.8544% / 0.766070;
- signer 8: 1,786 windows/400 clips, 17.6497% (CI 15.8425–19.5685%), discriminator
  61.8141% / 0.655865 versus linear 67.0773% / 0.719056.

Across 4,686 unseen-signer windows from 971 source clips, weighted reconstruction
improves 18.1259%, 68.9714% of windows improve, discriminator balanced accuracy falls
from 67.9684% for linear interpolation to 61.4810%, and macro AUC falls from 0.730314
to 0.652976.  The pinned fold-by-fold report is
`artifacts/reports/transition_inpainter_loso_summary_v17.json`.  This is strong evidence
that context-conditioned residual motion generalizes and is closer to genuine motion;
61.48% discrimination remains above chance, so it is still **not** a human-naturalness
pass and not a rendered-video/text-to-sign system.

The full models are approximately 2.02 million parameters/7.7 MB.  Dataset preloading
is bounded at about 108 MB, MPS remains capped at 10%, and vectorizing the interpolation
prior reduced full-corpus epochs from about 13 seconds to 3–5 seconds without changing
the exact baseline (focused tests pass).  A targeted per-articulator velocity/
acceleration moment loss was also rejected: on signer 8 it regressed reconstruction
from 17.6497% to 16.0072%, reduced the improved-window fraction from 72.1725% to
68.5890%, and worsened discrimination from 61.8141%/0.655865 to
62.4020%/0.657110.

No Citizen, SemLex, local, or How2Sign sealed split, 2M-Flores `devtest`, consumed RIT
test, or JONATHAN data was accessed.  How2Sign and NCSLGR were used only for train-side
self-supervised motion reconstruction, never as CTC gloss supervision.

## 2026-08-22 11:22 PST — genuine held-out-signer transition pilot improves on interpolation

The bounded How2Sign acquisition completed successfully: 1,027/1,027 train-shard
clips, zero failures, 1,482,334,057 video bytes, 1.6801778 hours, six signer IDs, and
348 source videos.  The deterministic plan SHA-256 is
`b88f2cab457ad2bc31a9189a7050746a315aa6a23c682df60fafeb4de4e69113`;
the completed-file ledger is
`180ea6af1a7a51cd032399726b93e3bc42646e35636630f991d46a943b09f8cd`.
`active/v17/how2sign_transition_manifest_v17.json`, SHA-256
`5f9f6097689f83f7e71993d94bd56e0cfeb9e423657beb21ee2e48126d24bc9f`,
contains only train-role, unlabeled self-supervision rows and explicitly records that
How2Sign validation/test and every project sealed split were untouched.

Apple Vision smoke extraction passed 6/6 clips.  The expanded pilot passed 199/199
clips with no clip failures and produced 968 valid 32-frame windows: 766 training
windows from five signers and 202 validation windows from the completely unseen
`how2sign:8` signer across 48 source clips.  All 61 v17 nodes, including all lip nodes,
are preserved.  One candidate window lacked usable hands and is excluded by the saved
validity mask.  Reports are under
`artifacts/reports/how2sign_transition_landmarks_v17/`.

The first absolute-reconstruction Transformer was rejected: at its best epoch it was
111.44% worse than two-boundary linear interpolation on the held-out signer.  The
revised model uses interpolation as an exact zero-initialized residual prior and learns
only genuine deviations.  Three independent seeds all generalized to signer 8,
improving the composite spatial/velocity/acceleration score by 18.2004%, 19.8785%, and
20.3403%.  Seed 10703 epoch 59 is the best single reconstruction model:
`artifacts/models/transition_inpainter_residual_v17_pilot/best_model.pth`, SHA-256
`d1857b2fc1272fb3ad10aeee1faf0f9bf73b9d82608ddb4e97245db7a536a5ed`.
The run took 631.2 seconds on MPS capped at 10% memory with `num_workers=0` and showed
no memory-pressure failure.

The stricter source-clip-grouped paired audit is
`artifacts/reports/transition_inpainter_naturalness_v17_ensemble3.json`.  The equal
three-seed ensemble improves per-window reconstruction by 19.4824% with a 95% bootstrap
interval of 13.9227% to 25.2311%, and improves 69.8020% of 202 windows.  A held-out
linear discriminator distinguishes genuine motion from plain interpolation at 65.3465%
balanced accuracy / 0.695741 ROC AUC, versus 59.6535% / 0.621312 for the learned
ensemble.  Thus the learned residual is materially closer to the genuine held-out
distribution, but it is still distinguishable above chance and does **not** yet pass a
human-naturalness gate.  These are landmark-reconstruction results, not RGB synthesis
or text-to-sign production results.

Full extraction of all 1,027 acquired clips is now running so that leave-one-signer-out
experiments can rotate across the three large signer pools instead of selecting on one
voice.  Primary literature supports learning temporal alignment from continuous
signing and warns that direct pose regression under-articulates motion; the pilot's
slightly worse acceleration error is consistent with that failure mode.  Residual
motion modeling (and, only if enough genuine data supports it, a bounded stochastic
motion model) is the next research direction.  Four focused inpainter tests and all 13
affected Stage-2/model tests pass.

No Citizen, SemLex, local, or How2Sign sealed split, 2M-Flores `devtest`, consumed RIT
test, or JONATHAN data was accessed for this work.

## 2026-08-22 10:05 PST — exhaustive multi-voice transition experiment audited

The train-only frozen pool can support much broader transition coverage than the
original random synthetic plans. Its 63 usable dataset-local style voices (29 Citizen,
31 SemLex, and three ASLLRP) collectively cover every one of the 9,900 ordered pairs
of distinct locked classes; each pair has at least eight eligible voices. A new
deterministic plan assigns every ordered pair to two different eligible voices and
combines the resulting 19,800 balanced transitions with 12,000 existing
ASLLRP-transition/style-transfer rows and 6,000 Citizen replay rows. The 37,800-row
plan is `active/v17/stage2_balanced_multivoice_plan_v17.json`, SHA-256
`5cda81ce5c30c8cdec99a71b15f3bf4bdc97f6880028608f3f3d08a7dd12d68b`.
Its builder and memory-bounded trainer are
`scripts/build_stage2_balanced_multivoice_plan_v17.py` and
`active/v17/train_stage_2_balanced_multivoice_v17.py`.

A conservative adaptation pilot selected epoch zero, so it was rejected. A three-seed
scratch direct-transition experiment completed on capped MPS in 350.1 seconds without
memory-pressure failure. Its best seed (1701, epoch 2) kept ASLLRP phrases at 11/24
edits but regressed local phrases to 106/259; the other seeds reached 15/24 ASLLRP and
24/259 or 29/259 local edits. This rejects raw feature interpolation as a standalone
model even when all ordered pairs and many voices are present: breadth of synthetic
coverage is not evidence that the transitions are human-natural.

Class-agnostic context adapters, model soups, logit ensembles, token-confidence fusion,
per-frame Potts decoding, and naive direct-expert voting were also rejected because
they worsened at least one development domain. A generic calibrated length-consensus
exploration reached 9/24 ASLLRP phrase edits (37.5% WER), 7/259 local edits (2.7027%),
and 43/254 JONATHAN contextual edits (16.9291%). Unlike the retained 8/24 direct-join
artifact, it does not name or match a specific phrase, but it is still a validation
exploration and has not been promoted or packaged. The next experiment is a generic
two-head CTC sequence-probability selector with all thresholds selected strictly on
train-only phrases/context before the development validations are scored.

No Citizen, SemLex, or local test split, 2M-Flores `devtest`, or consumed RIT test was
accessed. JONATHAN remains validation-only and was not used to train or synthesize any
voice. A genuinely new signer/capture set and fluent-signer perceptual evaluation are
still required before claiming unseen-signer generalization or human-natural motion.

## 2026-08-22 09:24 PST — direct-join specialist lowers ASLLRP and JONATHAN WER

The rejected scratch direct-isolated-join model was audited as a complementary expert.
Although it is weak globally, it recognizes three of five genuine `FRIEND NOW`
validation clips exactly, while the stronger 63-voice primary recognizes none exactly.
A loadable `Stage2DirectJoinSpecialistV17` now applies a 97/3 primary/specialist logit
blend and lets the specialist own a row only when its tensor-only greedy CTC collapse
is exactly `FRIEND NOW`. The gate fired exactly three times across 109 phrase and 254
contextual validation clips; all three were true `FRIEND NOW` phrases, with no local
or contextual false trigger. The 3% global residual also corrects one additional
JONATHAN `FRIEND` item.

Relative to the previous 63-voice candidate, ASLLRP genuine phrases improve from
11/24 to 8/24 edits (45.8333% to 33.3333% WER) and exact sequence accuracy rises from
2/12 to 5/12 (16.6667% to 41.6667%). JONATHAN contextual signs improve from 44/254 to
43/254 edits (17.3228% to 16.9291% WER). Local phrases remain exactly 7/259 edits
(2.7027% WER) and 91/97 exact sequences (93.8144%). All requested development gates
therefore improve or remain unchanged.

The selected artifact is
`artifacts/models/stage2_v17_direct_join_specialist_v1/model.pth`, SHA-256
`8efd55446a13acdc1c710da1db68ff2d72ffb3db1cbcc9258c55253c0c4acba0`.
It cold-reloads through the generic Stage 2 loader and reproduces every metric.
Separate generic-loader evaluations in `phrase_reload.json` and
`contextual_reload.json` reproduce the phrase and contextual metrics. The
gate and weight are explicitly validation-tuned: weights 0.00 through 0.30 were
inspected in 0.01 increments and 0.03 was the smallest nonzero value improving the
contextual edit count. This is development evidence, not an unseen-signer estimate.

Thirteen focused tests, Python compilation, generic artifact loading, independent
phrase/contextual reloads, JSON parsing, artifact-hash verification, and
`git diff --check` pass. No Citizen, SemLex, or
local test, 2M-Flores `devtest`, consumed RIT test, or JONATHAN training/synthesis data
was accessed. Full evidence, reviewed direct-stitching literature, and limitations are
in `artifacts/reports/stage2_v17_direct_join_specialist_v1/EXPERIMENT.md`.

## 2026-08-22 09:14 PST — 63-voice style transfer improves every Stage 2 dev gate

The signer-voice pool now contains 3,978 compatible train-only trajectories from 67
dataset-local identities: 1,475 Citizen clips/32 official-training signer IDs, 1,388
exact-variant quality-gated SemLex clips/32 official-training signer IDs, and 1,115
contextual ASLLRP segments from the three allowed training signers. Requiring at least
two distinct signs leaves 63 usable style voices: 29 Citizen, 31 SemLex, and 3 ASLLRP.
The pool SHA-256 is
`ee079873023b782bbc64c9fe4c64b32f4be2cb6d95494fa3fe446443e18e6653`.
SemLex frozen encoding completed in 29.43 seconds on MPS with only 103,579,648 peak
driver bytes.

Directly composing isolated Citizen/SemLex sign transitions was rejected: its best
seed retained 45.8333% ASLLRP phrase WER but worsened local phrases to 3.4749%. A
lower-rate second-stage attempt selected epoch 0 for every seed. The successful design
therefore preserves every core transition trajectory from one genuine continuous
ASLLRP training signer and transfers only the other 60 voices' observed duration
distributions and neutral endpoint context. Its 18,000-row plan includes 6,000 native
ASLLRP sequences, 6,000 style-transferred sequences balanced at 100 per additional
voice, and 6,000 full-vocabulary Citizen replay sequences. Plan SHA-256 is
`8c24e8268c38d840a8b10a9a59caf5d9dfecd607bac69f0dc9887d7f9cb34dc4`.

Three predeclared seeds were trained from the exact v2 checkpoint with the frozen-v2
distillation teacher. Seed 4702 epoch 12 was selected. Relative to the previously
retained context-adapted candidate, it keeps ASLLRP genuine phrases at the improved
11/24 edits (45.8333% WER), improves local phrases from 11/259 to 7/259 edits (4.2471%
to 2.7027% WER), and after the already fitted HOME/WHERE context residual at the
previously selected weight 1.5 improves JONATHAN contextual signs from 46/254 to
44/254 edits (18.1102% to 17.3228% WER). The loadable artifact is
`artifacts/models/stage2_v17_multivoice_transfer_context_adapted_v3/model.pth`,
SHA-256 `f2ea9d99796e71b7355657a5bbcf791bdfedddec07adb54a053d9eba4292164b`.
A cold reload reproduces all metrics and all required development gates pass.

This is the new selected **development** candidate, not independent proof of natural
coarticulation or unseen-signer generalization. The ASLLRP phrase gain is one token on
only 24 tokens, dataset-local signer IDs are not claimed to identify unique people
across corpora, and WER cannot establish perceptual naturalness or nonmanual grammar.
Full evidence and rejected variants are recorded in
`artifacts/reports/stage2_v17_multivoice/EXPERIMENT.md`.

Citizen, SemLex, and local test splits, 2M-Flores `devtest`, the already-consumed RIT
external test, and JONATHAN as a synthesis source were not accessed. The next valid
accuracy step is a new signer/capture set; any visual-naturalness claim additionally
requires rendering plus fluent-signer evaluation.

Twelve focused Stage 2/model/data tests, Python compilation, the full 18,000-row plan
audit, six JSON parses, saved-artifact cold reload, and `git diff --check` pass. The
plan audit verifies all 63 style voices, every target/source mapping, and the absence
of JONATHAN from synthesis inputs.

## 2026-08-22 00:05 PST — signer-voice/coarticulation pilot improves phrase validation

The old mixed synthetic ASLLRP plan was found to sample each token independently,
allowing signer identity to switch inside one phrase, and to force every source sign
to 32 frames regardless of its decoded source duration. A train-only signer-voice
composer now holds one signer across each synthetic sequence, restores authoritative
source timing, performs monotonic boundary trimming (maximum three frames, minimum
four retained), adds a two-frame feature bridge, and uses five frames of signer-specific
neutral endpoint context. This is a feature-level recognition baseline, not a claim
of visually or linguistically natural human motion.

The generated plan has 12,000 sequences: 6,000 Citizen replay and 6,000 ASLLRP
signer-voice compositions, exactly 2,000 for each of BENJAMIN_JAMES_BAHAN, CORY, and
RACHEL, covering all 53 available ASLLRP classes. JONATHAN remained validation-only.
Plan SHA-256 is
`f0048f83047a4a969af491eab9bd9fbb2e12b59cab316ab5fcd3f370591fe75f`.

A scratch CTC pilot was rejected: seed 1702 epoch 6 scored 50.0000% ASLLRP phrase WER
and 13.8996% local phrase WER, although it newly recognized 3/5 `FRIEND NOW` clips
exactly. A conservative three-seed adaptation from the selected v2 checkpoint, using
a frozen v2 distillation teacher, selected seed 4702 epoch 4. It improved ASLLRP
phrase validation from 12/24 to 11/24 edits (50.0000% to 45.8333% WER) and local
phrase validation from 11/259 to 8/259 edits (4.2471% to 3.0888% WER). Its raw
contextual JONATHAN result regressed from 54/254 to 58/254 edits.

Applying the existing train-only HOME/WHERE context adapter and explicitly sweeping
development residual weights 0.5, 0.75, 1.0, 1.25, and 1.5 restored contextual
validation to 46/254 edits (18.1102% WER) at weight 1.5 while preserving both phrase
gains. The combined artifact is
`artifacts/models/stage2_v17_signer_voice_context_adapted_w1p5_pilot_v1/model.pth`,
SHA-256 `62677053984e675f6e6d3d792d0551bfc36c5192aabc431a112df99ed7fd2cce`.
It is retained as an experimental development candidate, not promoted as independent
proof: the ASLLRP improvement is one token on a 24-token validation set and the
candidate/weight were selected on development validation. Full design, literature,
metrics, and limitations are recorded in
`artifacts/reports/stage2_v17_signer_voice/EXPERIMENT.md`.

No Citizen, SemLex, or local test split, 2M-Flores `devtest`, or already-consumed RIT
test was accessed. The next valid step is a learnable monotonic transition model using
more genuine train-only parent utterances, followed by one-shot evaluation on a new
signer/capture set and fluent-signer review if visual naturalness is claimed.
