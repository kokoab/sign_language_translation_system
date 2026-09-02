# Phrase-calibrated motion-valley diagnostic

A motion-only sweep over the nine development/reference videos selected a single
threshold of motion ≤0.008 for 0.16 seconds. It matched the expected sign count in all
five fully vocabulary-covered phrases. Because those same recordings selected the
threshold, this is a fitted development diagnostic, not held-out accuracy evidence.

| Phrase | Target | Emitted buffer |
| --- | --- | --- |
| GOOD_MORNING | GOOD MORNING | EAT MORNING |
| HELLO_HOW_YOU | HELLO HOW YOU | HOW NEED |
| MY_NAME | MY NAME | MY NAME |
| THANKYOU_FRIEND | THANKYOU FRIEND | NAME |
| TOMORROW_SCHOOL_GO | TOMORROW SCHOOL GO | TOMORROW GO |

Exact sequence accuracy remained 1/5. Twelve clips were formed, nine passed Stage-1
gates, median classification latency was 360.94 ms, and p90 was 490.77 ms. Better
boundary counts did not correct the local continuous-domain classifier errors.

Use this threshold only as the default experimental starting point. Do not replace the
neutral/pause-delimited path without a new naturally signed validation set. Per-clip
evidence is retained under `runs/`.
