# Reel rejection replay

No model inference, retraining, threshold sweep or live changes. Existing59clip/150sign calibration; one policy declared before scoring.

| Policy | WER | Correct /150 | Substitutions | Deletions | Insertions | Retained /82 |
|---|---:|---:|---:|---:|---:|---:|
| Current policy | 49.33% | 82 | 16 | 52 | 6 | 82 |
| Verifier-aware rescue | 48.67% | 86 | 16 | 48 | 9 | 82 |
| Unfiltered verifier (diagnostic only) | 56.67% | 102 | 28 | 20 | 37 | 82 |

Rescued intervals: 7. Calibration gate passed: False.
First blocking stage counts: {'verifier_rejected_after_proposal': 18, 'proposal_rejected_first': 42, 'commit_score_after_both_accepted': 12}.
Rescue uses final verifier score>=0.8 and final margin>=0.08, bypassing only low_score/low_margin rejection. Original commits and physical/duration/learned-no-emit rejection safeguards remain. A high score is not a calibrated probability of correctness.
Original decisions and all baseline edit counts reproduce exactly; functional policy guards and source/code hashes pass.
Local phrase intervals lack per-sign truth. Recovered correct signs and errors are transcript-aligned; individual rescued intervals cannot all be assigned ground-truth labels. Blocking counts are not counts of recoverable correct signs.
Unfiltered verifier is not an upper bound or deployable policy. It exposes the error trade-off of removing acceptance wholesale.
One fresh commit lock per interval with commit_hits=1; instant threshold does not affect original decisions after score acceptance. This replay does not establish asynchronous live stability/repeat behavior.
Familiar-signers/sixphrase reused development; no unseen-signer or continuous transition-negative guarantee.

Candidate failed calibration; confirmation predictions were not evaluated. No live change recommended from this replay.

## Changed cases

| Clip | Reference | Before | After | Rescued label / score |
|---|---|---|---|---|
| local:HELLO_HOW_YOU:f6a616f4 | HELLO HOW YOU | ∅ | HOW | HOW / 0.908 |
| local:TOMORROW_SCHOOL_GO:9f56f398 | TOMORROW SCHOOL GO | FRIEND | TOMORROW FRIEND | TOMORROW / 0.803 |
| local:HELLO_HOW_YOU:6446e671 | HELLO HOW YOU | HELLO HOW YOU | HELLO HOW HOW HOW YOU | HOW / 0.966 |
| local:HELLO_HOW_YOU:6446e671 | HELLO HOW YOU | HELLO HOW YOU | HELLO HOW HOW HOW YOU | HOW / 0.978 |
| local:HELLO_HOW_YOU:00424e11 | HELLO HOW YOU | HELLO HOW YOU | HELLO HOW HOW YOU | HOW / 0.879 |
| local:HELLO_HOW_YOU:dcec9c83 | HELLO HOW YOU | HELLO YOU | HELLO HOW YOU | HOW / 0.924 |
| local:HELLO_HOW_YOU:HELLO_HOW_YOU_6 | HELLO HOW YOU | HELLO YOU | HELLO HOW YOU | HOW / 0.982 |

Decision: retain current live policy. Rejection causes some correct-sign losses, but the tested override is not a clean fix: four more transcript-correct signs and three more insertions. Removing acceptance wholesale further worsens WER. This does not rule out every possible gate policy; it argues against blanket threshold relaxation and supports targeting continuous-sign identity discrimination next.
