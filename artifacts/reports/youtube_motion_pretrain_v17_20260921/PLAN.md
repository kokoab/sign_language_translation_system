# Connected-motion pretraining pilot

User-approved objective: improve connected-sign recognition while preserving isolated
recognition. Use only the frozen 1,411 acquired clips; do not acquire more data.

## Execution

1. Audit all manifest members for frame ordering/count, finite coordinates, hand gaps,
   handedness evidence, coordinate geometry, timestamps and channel semantics. Save
   per-clip evidence and an aggregate compatibility report. Validate the audit on
   synthetic missing/malformed inputs before launching the full scan.
2. Admit training only with a defensible bridge to the existing temporal stack. A
   matching tensor shape is insufficient. Never fabricate confidence, depth, frame
   rate, signer IDs, or gloss boundaries. If this gate fails, publish a blocked
   experiment report, with training and behavioral metrics explicitly unmeasured.
3. If a bridge can be validated, compare identical supervised recipes from identical
   checkpoints, paired seeds, fixed decoder and selection rules, differing only in
   masked-motion pretraining. Mask observed spans; exclude genuinely missing targets.
   Compare reconstruction against interpolation. Keep captions out of supervision.
4. Primary measure: held-out-signer connected WER with substitution/deletion/insertion
   counts, exact sequences, held-sign duplicates, intentional-repeat recall and delay.
   Isolated validation is a retention gate; official Citizen test remains sealed.
   Synthetic stress probes are diagnostics only, never selection truth. Do not promote
   runtime automatically. Report missing measurements instead of inventing them.
5. Keep audit, feature preparation and real-device preflight in the active session.
   Only launch training detached. Write status and reports; notify on exit. The attempted
   CLI resume failed because the live session owns the thread writer; do not retry it.
   No polling, sleeps,
   automatic downloads, or repeat experiment loops.

## Existing code inspected

- `active/v17/schema_v17.py` and `model_v17.py`: five-channel Apple contract and derived
  temporal features.
- `active/v17/pretrain_masked_pose_v17.py`: partwise encoder, not a drop-in trainer for
  the current global encoder.
- `active/v17/pretrain_stage2_temporal_v17.py`: earlier frozen-Apple-feature objective.
- `active/v17/train_unified_streaming_aligned_grounded_v17.py`: existing native-rate
  causal CTC comparison, freezes Stage 1 and trains an evidence head.

The earlier 2M-Flores pretraining had mixed results and failed promotion. New data are
not evidence that this transfer works. Existing unrelated changes stay intact.

## Frozen pilot recipe after foreground audit

- All1,411 JSONs are structurally valid. Quarantine220 manifest/count mismatches;
  preserve their files and report exact count deltas. Start with1,191 count-consistent
  clips, then omit windows with less than75% hand-visible frames or an adjacent wrist
  jump above1.5 shoulder widths.
- Separate46-node XY+presence input (42 hand,4 upper-body). Assign hand detections by
  nearest pose wrist, omit ambiguous assignments. Torso-centered coordinates divided
  by one sequence-median shoulder width; flip Y. No face, depth, confidence or invented
  frame rate. Source-video hash fold0 is SSL validation, never called signer-disjoint.
- Pretrain the **three128-wide causal CTC temporal blocks**,8 fixed epochs of masked
  six-frame spans inside32-frame windows. Discard source adapter and reconstruction
  head. Transfer only `blocks.*`; all non-temporal initial head weights stay identical.
- Keep the Apple Stage1 checkpoint completely frozen. Use the pre-local-adaptation
  orientation-robust checkpoint to avoid inheriting the old local signer leakage.
- Two paired seeds17321/17322; each runs baseline and pretrained CTC for18 epochs
  using the existing native-rate aligned trainer and identical selection/decoder.
- Measure real-sequence substitution/deletion/insertion counts, repeated-reference
  recovery and first-emission delay relative to ASLLRP sign end. Synthetic hold/repeat
  probes are diagnostic only. Real held-sign annotations remain unavailable.
- Report per-seed gates: local WER strictly improves, ASLLRP/NCSLGR WER do not regress,
  isolated CTC exact loses at most1 percentage point. No automatic promotion/acquisition.
