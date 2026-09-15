# Per-frame CTC normalization: completed training result

The fixed12-epoch continuation (13–24) changes only CTC normalization from per-target
to per-valid-output-frame. It retains the anchor epoch12 model and optimizer, all
positive/pooled/anchor/background objectives, learning rates, data and decoder.

Final full training recovers1,117/1,291known signs (86.52%), versus330/1,291before
continuation. Deletions fall817→25. Isolated CTC reaches1,891/1,901 (99.47%). However,
2,336insertions produce194.42%training WER: this is emission recovery, not accurate
sequence recognition. All12epochs and full-coverage schedules are preserved in
`training_summary.json`; checkpoint payloads include optimizer state and provenance.

`training_emission_locations.json` diagnoses64training recordings at epoch16.
The32ASLLRP OTHER recordings emit129known signs for39references:52emissions occur
inside OTHER intervals,42in gaps,35inside known intervals. Annotation-relative
positions are diagnostic and may differ from the CTC alignment. No decoder was changed.

The current supervision provides one positive sequence anchor per event and blank
on separate background windows. The next refinement in
`../joint_ctc_frames_v17_20260915/` adds direct verified-frame CE in the actual sequence
context, including unambiguous OTHER intervals and guarded interior gaps. No held-out
outputs informed that decision. No Citizen test access, deployment or promotion ran.
