# Android runtime takeover — 2026-10-06

Completed one-shot diagnostics without status polling. Device benchmarks waited for streamed
BENCH done/FAILED events; Python Stage3 comparison waited for process exit.

Stage3: 89/89 device sentences identical to Python using saved 2026-10-05 tune fixtures.
Existing recognition parity remains89/89 videos, zero commitment mismatches over5,161 frames.
These fixtures do not validate the camera/MediaPipe frontend or independent signer accuracy.

Landmark recognizer batch8 (30 samples per app run, two runs per setting):

| Runtime | Threads / priority | First / second median ms |
|---|---|---|
| App |4 / display(-4)|333.5 /279.6|
| App |2 / display(-4)|349.2 /402.8|
| App |1 / display(-4)|667.6 /732.5|
| App |4 / urgent display(-8)|254.6 /245.7|

Standalone benchmark_model averages:1 thread pinned to one big core383.957ms;
2 threads pinned to big cores198.418ms;4 threads108.799ms.
App/standalone priority and affinity were not matched, and tests ran sequentially.
Higher app priority helped in these samples, but did not close the gap; no production
priority change is justified yet. App single-thread cost is also substantially higher
than standalone this session, so parallel scaling alone is not an established cause.
No app/model code changed. Thread-pool interference remains an untested hypothesis.

Completion callback attempted codex exec resume of the same session; it failed with
thread-store conflict because the desktop session still had an active writer. A macOS
notification was attempted, but automatic conversational continuation was not achieved.
Do not present this callback as a working self-resume mechanism. No FINAL_RESULTS.md
was produced by that callback. Results were reviewed after the user's follow-up.

Next safe action: isolate the recognizer from the boundary model's thread pool in the
existing debug activity and compare under matched foreground/affinity conditions.
Then validate any selected runtime change on the saved tune fixtures and live camera,
including sustained complete-pipeline timing. No retraining or held-out gate use needed.
