# Frozen segment-first coherent decoding

Evaluation-only; the checkpoint was not changed. Overlapping rolling-window state evidence is averaged on source timestamps, decoded with OUTSIDE→START→SIGNING→END and END→START repetition, and limited to the v16-style hand-presence range. Edge bias was selected on train sources; validation was evaluated once. Citizen test remained sealed.

Train-selected edge bias: 0.00 (train F1 ±100 ms 26.30%).

Validation boundary F1: ±100 ms 18.79%; ±200 ms 40.46%.

Visible locked100 WER: 103.65% (121 deletions, 33 insertions, 45 substitutions; 192 references).

Held/repeat probe: 2/20 held once; 0/20 repeats exactly twice (synthetic time-warp probes).
