# Finish-time bounded CTC precheck

This is a fixed research experiment, not a runtime change. It tests whether complete-
utterance context is a better final gloss authority than the current causal/Reel path.

The frozen Stage-1 Squeezeformer produces frame evidence. One bidirectional GRU and one
CTC projection emit 102 internal states: blank, the locked 100 glosses, and UNKNOWN.
Only indices 1–100 can reach the visible gloss transcript. Stage 1 is not fine-tuned.

All four admitted training sources use the same CTC objective:

- 923 complete continuous training sequences;
- 1,901 isolated training clips as one-gloss sequences;
- 199 verified positive cores as one-gloss sequences;
- 494 verified background clips as empty sequences.

Training is fixed at seed 17201, 20 complete epochs and final-epoch evaluation. It uses
MPS and deterministic full coverage. Development data is evaluated only after training;
the consumed Citizen test remains sealed. The comparator is the repaired causal CTC on
the identical 334-recording development suite.

Nine focused tests pass, including repeated labels, empty CTC targets, hidden UNKNOWN,
future context and the existing timestamp/chunk contracts. A real MPS packed-GRU
forward/backward check also passes. The detached worker writes `REPORT.md` and
`completion.json` on success or `FAILURE.md` and `completion.json` on failure, then
sends one macOS notification. It is not polled.
