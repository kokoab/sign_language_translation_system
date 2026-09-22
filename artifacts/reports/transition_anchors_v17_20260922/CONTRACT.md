# Bounded temporal-anchor refinement

Authorized scope: user requests faster continuation and explicit transition learning.
Question: does timed sign-core/gap supervision improve the CTC head beyond equal additional
CTC-only updates, while keeping the selected adapted Stage1 weights fixed?

Use only the completed combined6421-record manifest, roles and normalized windows. Two
initializations are selected adapted checkpoints for17421/17422, independently pinned.
For each, compare control versus anchored head-only refinement, three epochs, same head
initialization, RNG reset and paired minibatches (32single records,4phrases;4264single and
536phrase visits per epoch). Cache rich encoder evidence once per seed to avoid repeated
encoder work. No data acquisition, re-extraction, synthetic phrases, live changes or test.

Objective: same target-normalized phrase CTC plus weighted single-sign CTC. The previous
isolated base CE is constant under freezing and is omitted equally in both arms. Anchored
arm additionally applies0.25times interval CE on matching phrase records. Average within
each interval, then within core/gap groups and equally across present groups, avoiding a
long gap overwhelming short sign cores. Head AdamW lr0.0001, weight_decay0.0001, clip5;
MPS forward/backward, CPU CTC only. Stage1 eval/frozen in both arms throughout.

Anchors derive from existing source annotations intersected with approved ASLLRPcontiguous
membership ONLY. Central50% of eligible known intervals use canonical label+1. Internal
gaps are between consecutive eligible known events, trimmed0.05s at each edge, excluding
any overlap with all annotated events including OTHER. No padding or untimed local labels.
After mapping to cached positions:59train cores/27train gaps,17validation cores/3validation
gaps. Training uses only train anchors. Old annotation target IDs are zero-based; they
are never treated directly as CTC targets. Mapping follows cached32-frame linear resampling
within source ranges and is approximate at subframe boundaries. Source counts refer to
correlated intervals, not new independent clips or general100-sign transition coverage.

The blank-gap term is an explicit, bounded alignment hypothesis based on the existing
annotation coverage; it is not a claim that all unannotated movement is physical rest.
Unlike old standalone blank clips, the loss acts within the same connected-sequence forward
pass alongside positive timed sign cores. Prior aligned recipes used different/excluded
sources and inputs; this experiment claims only its own matched control comparison.

Evaluate full1874validation examples at epoch0 and each epoch, preserving source-specific
S/D/I/exact and isolated retention. Same selector: mean local-phrase and ASLLRPcontiguous
knownWER; earliest tie, include initializationepoch0. Afterselection evaluate fulltrain.
Do not promote from a small transition metric, change gates after seeing results, or call
a shifted CTC spike a false sign. Save per-epoch interval losses separately when available.

Preparation must verify source/code/checkpoint/anchor hashes, labels/roles/indexbounds,
finite rich evidence on all7367windows per seed, finite head gradients, absent base
gradients, zerooptimizersteps. Dedicated recipe and matching preflight permit only the new
runner. Existing generic blocked manifests and completed comparison recipes remain intact.
Training detached with caffeinate and completion/failure notification; no polling.
