# Live Stage-2 elapsed-time-window diagnostic

This diagnostic replays five genuine local development/reference phrases while
extracting only 15 observations per second, close to the 14.47 FPS measured in the
first webcam session. Each 1.067-second motion interval is resampled to the trained
32-frame tensor instead of waiting for 32 successful live extractions.

| Phrase | Final CTC hypothesis | Exact |
| --- | --- | ---: |
| GOOD MORNING | `GOOD MORNING` | yes |
| HELLO HOW YOU | `HELLO HOW YOU` | yes |
| MY NAME | `MY NAME` | yes |
| THANKYOU FRIEND | `THANKYOU FRIEND` | yes |
| TOMORROW SCHOOL GO | `TOMORROW SCHOOL GO` | yes |

The 16 accepted updates used mostly 16 observations for each 1.067-second window,
then resampled them to 32. All five sequences remained exact. This directly confirms
that the trained timebase can survive the laptop's observed extraction throughput and
removes the prior two-times-slow motion distortion.

Post-landmark inference was still variable: 488.70 ms median, 949.41 ms p90, and
2045.51 ms maximum. Hand-image embedding dominates those outliers. This validates the
timebase correction but also reinforces the need for a separately trained and gated
landmark-first Stage-2 cascade if near-instant display is required.

These are familiar-domain development/reference recordings, not independent accuracy
or generalization evidence. No sealed or test split was accessed.
