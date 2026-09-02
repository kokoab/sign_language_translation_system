# v17 no-pause streaming-window experiment

This is a separate experimental path. It does not change or replace
`scripts/live_isolated_v17.py`.

Run:

```bash
venv/bin/python scripts/live_streaming_v17.py --camera 0
```

The script continuously extracts Apple Vision landmarks and classifies the newest
1.2-second overlapping window. A new window is submitted at most every 0.2 seconds,
and two consecutive accepted predictions must agree before a gloss enters the buffer.
Classification never queues old windows: if the classifier is busy, the next request
uses the latest available window. No neutral pose or pause is required.

`CASCADE` is the default because it can use the fast landmark model when its evidence
is strong and fall back to the unified model when uncertain. The on-screen mode buttons
still switch between `DEFAULT` and `CASCADE`. `RESET` clears the visible buffer without
deleting logs; `FINISH` sends the buffer to the local tiny Stage-3 model and speaks the
sentence. Sessions, all window predictions, emitted glosses, and low-resolution video
are written under this directory.

This is overlapping-window Stage 1, not a learned continuous decoder. It can reduce
pause dependence, but windows may contain transition motion or parts of neighboring
signs. Consecutive repetitions of the same gloss are intentionally emitted only once
until another label or two rejected windows rearm it. A trained streaming/CTC model is
needed to solve that ambiguity reliably.

An end-to-end smoke on the quarantined Citizen training clip
`6226330398612929-W.H.A.T.mp4` processed six windows, emitted one stabilized gloss, and
wrote history/video successfully. Fallback classification took 0.34–0.41 seconds per
window and the latest-window policy prevented backlog. It stabilized as `TELL`, not the
folder label `WHAT`; this is execution evidence only and demonstrates why real natural
continuous validation is required before making an accuracy claim.
