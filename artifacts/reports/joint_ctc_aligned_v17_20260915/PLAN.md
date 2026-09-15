# Finish alignment after supervised nonblank fitting

Training-only evidence shows that per-frame CTC normalization removes the blank
local minimum, but underweights sequence errors once signs are recognized. The
verified-frame phase improves known recovery further (epoch28:1,186/1,291), while
2,387insertions remain. The problem has shifted from learning sign evidence to
constraining the decoded sequence.

A controlled experiment with the original target-normalized CTC plus positive CE
converges to p(target)=0.0627 from a blank start, but retains p(target)=0.99952 from
the nonblank warm start. See the preceding curriculum_diagnosis.json. This motivates
a short final alignment phase, not a decoder change or validation-based loss sweep.

After the fixed verified-frame run finishes, resume its epoch36 model and optimizer
for six epochs(37–42), restoring original target-normalized CTC for complete sequences
and isolated/core positives. Keep all learned verified-frame CE, anchors, pooled CE/KL,
background CE, learning rates, full data coverage and seed17111+epoch. Reuse the exact
reviewed training loop; only the two CTC loss functions change. Save optimizer state
and measure full training recognition at every epoch.

Evaluate the final epoch42 once on the frozen334development recordings,1,356isolated
validation clips and57LG cores. Verify provenance, coverage, actual CTC outputs,
CPU/MPS consistency, focused tests and report all promotion failures. Official Citizen
test remains sealed. No further changes based on held-out results; no deployment or
promotion without existing gates. This is a training-objective repair using existing
data, not new data acquisition or a new architecture.
