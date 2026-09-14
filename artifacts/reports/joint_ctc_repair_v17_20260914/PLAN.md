# Repair CTC blank collapse

User authorized fixing the observed collapse and returning measured results.
Prior recipe and artifacts remain immutable. Reuse its exact prepared data, feature
contract, Stage-1 initialization, CTC head, per-epoch schedules and evaluation gates.

1. Measure the missing positive gradient route: original pooled isolated/core CE
   cannot update the CTC head; verified background does update it. Inspect loss/gradient
   behavior on training inputs. Add a regression test for direct positive CTC gradients.
2. Add single-token CTC on each original isolated replay clip and O5S5 positive core,
   retaining the existing pooled CE/KL and complete-sequence CTC. No decoder bias sweep,
   blank suppression, new architecture or unverified negative labels.
3. Run training-only checks, then a bounded corrected joint run with the original
   full-coverage schedule/learning rates. Save optimizer state for exact recovery.
   Measure full training CTC recognition, rather than relying on loss or tiny-fit alone.
4. Evaluate the fixed candidate on all334 frozen development recordings and1,356
   isolated development clips; retain original promotion gates. Report CTC isolated
   recognition separately from pooled accuracy, event-level LG and real latency limits.
5. Verify code/data/checkpoint provenance and focused tests, record results/current
   state, and report what was fixed and what still fails.

First correction adds one equally weighted positive-CTC term: mean(single-token CTC
on replay) + mean(single-token CTC on O5S5 cores), each with the same epoch balancing
as its existing pooled loss. Training remains seed17111,12epochs,110updates/epoch,
base LR1e-5, head LR3e-4, AdamW/clip5/weight decay1e-4. No validation-based adjustment.
If substantial full-training emission fitting remains absent, diagnose on training
only before another bounded correction; do not claim that emitting one token fixes
the problem or select a different objective from held-out errors.
