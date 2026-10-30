# Opening demo replay — 30 October 2026

Input: `data/local/phone_session_pull_v17_20261009c/live_reel_videos/20260930_134459_04.mp4` (full 12.67-second saved phone take; Downloads holds a shorter copy).

Ran the current desktop Live segmental pipeline, with live captions enabled by default, real-time video playback, and speech enabled. No model or app source edits.

Recognized sequence: **HELLO → GOOD → MORNING → HOW → YOU**.

Final caption: **Hello, good morning, how are you?**

| Video time | Observed event |
|---:|---|
| 1.67 s | Caption: Hello |
| 3.18 s | Hello settled; speech queued |
| 3.71 s | Caption: Hello, good |
| 3.88 s | Caption: Hello, good morning |
| 4.58 s | Caption: Hello, good morning, how |
| 6.08 s | Draft: Hello, good morning, how do you |
| 7.62 s | Settled: Hello, good morning, how are you?; remaining sentence speech queued |
| 12.63 s | Finish at end of input; final caption unchanged |

These are events from this desktop replay, not new phone performance measurements. Caption model: `live_captions_v17_20261014`. The exact session JSON is in the timestamped subfolder beside this report.

Command:

```sh
venv/bin/python scripts/live_segmental_v17.py --video data/local/phone_session_pull_v17_20261009c/live_reel_videos/20260930_134459_04.mp4 --realtime-video --output-root artifacts/reports/magsaysay_demo_replay_20261030
```

For the video narration, describe the current demo as live English captions updating while signing. The deployed caption tagger is distinct from the T5 research evaluation retained in the manuscript.
