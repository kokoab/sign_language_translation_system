# Supervise verified intervals on the actual sequence path

The frame-normalized continuation repairs emission suppression, but still produces
many extra known signs. Its completed epoch24 recovers1,117/1,291known training signs with
2,336insertions and194.42%WER; isolated CTC is1,891/1,901 (99.47%). A64-training-sequence epoch16 location probe shows ASLLRP OTHER
recordings emit129known signs for39references:52emissions are within OTHER intervals
and42within gaps. Timing locations are diagnostic, not exact event alignments.

The existing trainer teaches only one sequence anchor per event and blank on separate
background windows. Short guarded gaps cannot become standalone0.53s windows, and
OTHER intervals remain unsupervised except at their midpoint. Use the existing complete
annotations to train these states in their actual sequence context.

After the preceding fixed run finishes, continue its epoch24 model/optimizer for
12epochs(25–36), with the same architecture, per-frame CTC, anchors, pooled CE/KL,
positive CTC, learning rates, seed17111+epoch and110-update full-coverage schedules.
Add only frame CE on every unambiguous annotated known/OTHER token and tokens inside
verified interior gaps after the existing0.10s guard on both boundaries. Ignore clip
edges, overlapping annotations, unguarded transition regions and padding. Incomplete
O5S5 narratives cannot provide any sequence labels. Existing exact-core supervision
continues unchanged. Normalize new CE by each source's full admitted frame population,
with equal epoch-level source weight, independent of batch partition.

Record training metrics each epoch. Evaluate the final fixed candidate on all334
development recordings,1,356isolated validation clips and57LG positive cores. Verify
all input/checkpoint/code hashes, coverage, decoder metrics, CPU/MPS consistency and
focused tests. Keep the official Citizen test sealed; preserve prior artifacts and
do not promote a candidate that fails the existing gates.
