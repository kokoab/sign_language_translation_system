# Reel Stage-1 external ASLLRP phrase check

Date: 2026-09-02

This development-only check replays 12 real ASLLRP contiguous two-gloss clips through
the default phrase/activity-adapted Reel Stage-1 path. These clips were not used by the
local-phrase adapter, but they remain `train_candidate` material rather than a sealed or
independent test set. Stage 2, speech, live lips, and the naturalizer were disabled for
recognition. No protected Citizen or local test split was accessed.

## Live Reel result

| Expected | Committed | Provisional candidates |
| --- | --- | --- |
| NIGHT TIME | — | SAME |
| WATER COLD | — | WATER, SLEEP |
| LIKE READ | — | LIKE, MAKE |
| FRIEND NOW | — | FRIEND |
| NOW BAD | NOW | NOW |
| TIME FRIEND | — | FRIEND |
| FRIEND MAYBE | WRITE | FRIEND, WRITE, SAME |
| FRIEND NOW | — | FRIEND, WHEN |
| FRIEND NOW | — | FRIEND, WHEN |
| FRIEND NOW | FRIEND | FRIEND |
| FRIEND NOW | FRIEND | FRIEND |
| WORK WHERE | WORK | WORK |

- Exact sequences: 0/12
- Reference glosses: 24
- Levenshtein edits: 20
- Aggregate WER: 83.33%
- Emitted glosses: 5 (four aligned correctly, one wrong `WRITE`)
- Dominant failure: deletion/under-emission; most videos yielded only 2-4 probes
- Output histories and low-resolution replays are the timestamped child directories
  beside this README.

## Model-only equal-segment comparison

To separate classification from the Reel stability policy, each ASLLRP clip was divided
into its two target-ordered equal segments, matching the phrase adapter's weak boundary
method. There were 24 segment examples.

| Classifier | Original | Adapted |
| --- | ---: | ---: |
| Landmark proposal top-1 | 10/24 (41.67%) | 8/24 (33.33%) |
| Full multimodal verifier top-1 | 13/24 (54.17%) | 12/24 (50.00%) |
| Full multimodal verifier top-5 | 18/24 (75.00%) | 18/24 (75.00%) |

The local phrase/activity adaptation therefore did not improve this small ASLLRP domain.
The live failure is a combination of signer/domain classification mismatch and an emission
policy that is too conservative for these short, fast clips. This does not contradict the
good user webcam session or the local phrase gains; it limits their generalization claim.

## Validation

- `venv/bin/python -m unittest test.test_live_reel_stage1_v17 -v`: 16/16 passed
- `git diff --check`: passed before recording this report

