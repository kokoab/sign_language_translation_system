# Pretrained pose boundary transfer — 2026-09-22

## Actual pretrained weights found and verified

Author repository: https://github.com/sign-language-processing/segmentation
Pinned commit: `22ca3a6f63b6f031bfb1c0d717fcb259143ba7db`.
Downloaded its shipped safetensors/config only, plus relevant source; no foreign dataset.
Weights: `artifacts/models/pose_boundary_dgs_2026/` (11,485,662bytes).
147tensors load strictly,5,734,141trainable parameters. CPU/MPS forward maximum difference
3.815e-5; synthetic MPS backward has finite nonzero gradients,zero optimizer steps.
The test harness reuses upstream architecture definitions, substituting nn.Module for
Lightning bookkeeping; tensor operations and state keys are preserved. Initial synthetic
backward hit inference-created tensors after a device move; fresh normal model tensors
fixed the harness and the check passes. No optimizer or source weights modified.

The source model is trained on DGS, not ASL; it takes50joints×6(position/velocity)
from MediaPipe Holistic with the upstream pose-anonymization mean/std normalization.
It uses full-sequence attention and symmetric temporal convolutions. Its normalization
also computes whole-clip shoulder statistics. Synthetic future changes alter early output;
this is decisively not a causal live model.

## Frozen real-video transfer

Used all12previously approved ASLLRP development videos /24known signs. Actual native-fps,
native-resolution MediaPipe0.10.14 Holistic complexity1,aspect preserved,one worker,
new tracking instance per clip. Upstream pose-format0.15.0 /pose-anonymization0.0.1
preprocessing,upstream argmax decoding,min3frames,no gap merging. Frozen Reel classifies
predicted intervals with100mscontext and no wrist trim; existing acceptance rules retained.
No fitting,threshold sweep,protected test use or default runtime change.

| Arm | Correct | S/D/I | WER | Retained baseline correct | Exact videos |
| --- | ---: | --- | ---: | ---: | ---: |
| Actual Reel baseline |4/24|1/19/0|83.33%|4/4|0/12|
| Prior learned hand-geometry boundary, two seeds |7–8/24|1/16/0 or0/16/0|66.67–70.83%|3/4|0/12|
| Frozen DGS pose boundary + Reel |11/24|3/10/0|54.17%|4/4|2/12|

22predicted segments. This is conditional OFFLINE composition using future video context,
not an asynchronous live accuracy comparison or independent generalization evidence.
It is promising transfer evidence,not a successful recognition system. Three substitutions
remain; zero insertions does not establish transition rejection on unannotated OOV spans.
12.68s source video required17.52sfrontend and3.25sboundary forward,including cold starts;
Reel compute is additional. Sustained real-time/iPhone performance is not established.
Upstream decoder groups adjacent B/I predictions; do not silently replace it or tune it
on these12videos. Existing repeated-sign/low-motion coverage remains absent.

## Downstream decision diagnostic

`../asl_temporal_boundary_v17_20260922/diagnose.py` audits saved outputs without rerunning
models. Reviewed-boundary verifier gets16/24identities right,but four correct verifiers
are blocked by proposal acceptance. A fixed verifier-authority .45counterfactual yields
14correct,2S/8D/0I,retains4/4,versus12original correct oracle commits. It changes both the
proposal veto AND score authority: some original proposal-boosted commits are removed.
Learned arms become6/7/7/7correct; geometryseed17621 adds a substitution andseed17622
loses one net correct output. This is not a safe universal gate fix; runtime unchanged.

## Fine-tuning preparation now launched, training not started

User requested continued implementation and pretrained adaptation. Reuse exactly the
1,121source records of the reviewed boundary manifest,with source video hashes,original
role/signer/parent/interval contracts. Preparation keeps50-joint raw poses and confidence
before whole-clip normalization so a later bounded-window contract remains possible.
No gaps become background labels. This is feature preparation,not dataset acquisition.
Single-source extraction and serialized pose roundtrip passed. Detached caffeinatePID69981
launched2026-09-22T04:06:11Z,with completion/failure notification. Do not poll; inspect
`preparation_completion.json` and `prepared_manifest.json` on the next authorized check.
The prepared manifest remains training_ready=false. Generic phrase gate stays unchanged.

Next: verify completed feature membership/hashes,then specify bounded-context normalization
and start/end transfer supervision using approved intervals with unknown gaps masked.
Fine-tuning must compare pretrained initialization against the existing trained boundary
under the same latency contract and report retained correct signs as well as false outputs.
Do not train ordinary BIO background targets from uncertified gaps,or feed Apple61×5 into
these weights. No retraining from scratch,foreign blank-head veto or backbone search is
needed for this next candidate. PLAN.md records the bounded scope.


## Parallel extraction update

User requested multiple video workers. Interrupted serial PID69981 with SIGINT after
its850-record checkpoint; the resulting KeyboardInterrupt receipt is preserved as
preparation_completion_serial.json and is an intentional restart,not corrupt input.
Serial code and manifest snapshots preserved. Native per-video extraction is AST-identical.
Three spawned worker processes each keep one sequential tracker per video. Parent alone
writes atomic manifests,now after every completed video. Resume verifies source membership,
video/pose hashes,shape,clock and finite values; completed entries retain their original
source role and extraction provenance. Uncheckpointed tail outputs are re-extracted.

Three-video parallel smoke passed; resume accepted a valid cached record and rejected
changed pose hash,video hash and source identity. git diff --check passed.
Detached PID38560 launched2026-09-22T05:10:00Z with --workers3;850verified-checkpoint
records are eligible for reuse and271remaining videos need extraction. See
prepare_parallel.log and preparation_launch.json. Full completion not yet checked.
No change to frames,resolution,tracking,model,labels or training gate. No measured
threefold speed claim. Training still needs a representative step benchmark; cached
poses eliminate repeated RGB extraction but epoch count determines total runtime.
