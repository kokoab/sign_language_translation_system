# v17 low-motion boundary experiment

This path preserves `live_isolated_v17.py` and changes only the endpoint rule: a sign
starts after visible wrist motion and closes after roughly 0.16 seconds at or below
0.008 normalized wrist motion at the signer's current hand position. The hands do not
need to return to a neutral/rest
pose. A short 0.08-second cooldown prevents the same hold from immediately reopening.

```bash
venv/bin/python scripts/live_motion_valley_v17.py --camera 0
```

The same Stage-1 classifier, extraction contract, confidence gates, RESET/FINISH
controls, local tiny Stage 3, speech, JSON history, and low-resolution video remain in
use. This is still a heuristic isolated-sign boundary, not learned continuous decoding.
Signs with an internal hold can split early, while transitions with no motion valley can
merge neighboring glosses.
