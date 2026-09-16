# Held-sign replay report

Completed 40 phase runs over 10 pinned recordings.

Boundary-state correction changed 0 final outputs; corrected live and offline outputs differ in 3 runs.

The same model logits feed both legacy and corrected collapse. Differences between those two isolate bookkeeping; offline comparison can additionally change visual context.

**Limits:** Saved videos are re-extracted. Original camera features cannot be reconstructed exactly. Webcam/partial O5S5 references are not scored as ground truth. Phases are sensitivity probes, not independent samples. No training or model promotion occurred.

| Recording | Phase | Legacy live | Corrected live | Offline |
|---|---:|---|---|---|
| webcam_i | 0.0 | I | I | I |
| webcam_i | 0.25 | I | I | I |
| webcam_i | 0.5 | I | I | I |
| webcam_i | 0.75 | I | I | I |
| webcam_hello | 0.0 | HELLO HELLO | HELLO HELLO | HELLO HELLO |
| webcam_hello | 0.25 | KNOW HEAR | KNOW HEAR | KNOW HEAR |
| webcam_hello | 0.5 | HELLO HELLO | HELLO HELLO | HELLO HELLO |
| webcam_hello | 0.75 | HELLO HELLO | HELLO HELLO | HELLO HELLO |
| webcam_eat | 0.0 | I EAT EAT | I EAT EAT | I EAT EAT |
| webcam_eat | 0.25 | I EAT | I EAT | I EAT |
| webcam_eat | 0.5 | I EAT | I EAT | I EAT |
| webcam_eat | 0.75 | EAT EAT | EAT EAT | EAT EAT |
| asllrp_night_time | 0.0 | SCHOOL | SCHOOL | SCHOOL |
| asllrp_night_time | 0.25 | TIME | TIME | TIME |
| asllrp_night_time | 0.5 | LEARN FRIEND | LEARN FRIEND | LEARN FRIEND |
| asllrp_night_time | 0.75 | SCHOOL | SCHOOL | SCHOOL |
| asllrp_time_friend | 0.0 | FRIEND | FRIEND | FRIEND |
| asllrp_time_friend | 0.25 | COLD FRIEND | COLD FRIEND | COLD FRIEND |
| asllrp_time_friend | 0.5 | TIME FRIEND | TIME FRIEND | TIME FRIEND |
| asllrp_time_friend | 0.75 | FRIEND | FRIEND | FRIEND |
| asllrp_morning | 0.0 | DAY TIME TOMORROW FAMILY TOMORROW | DAY TIME TOMORROW FAMILY TOMORROW | DAY TIME TOMORROW FAMILY TOMORROW |
| asllrp_morning | 0.25 | I MORNING MORNING FAMILY | I MORNING MORNING FAMILY | I MORNING MORNING FAMILY |
| asllrp_morning | 0.5 | WHERE TOMORROW STOP HOW SCHOOL | WHERE TOMORROW STOP HOW SCHOOL | WHERE TOMORROW STOP HOW SCHOOL |
| asllrp_morning | 0.75 | DAY HOSPITAL STOP MORE FRIEND | DAY HOSPITAL STOP MORE FRIEND | DAY HOSPITAL STOP MORE FRIEND |
| local_hello_how_you | 0.0 | KNOW HOW MAKE | KNOW HOW MAKE | KNOW HOW MAKE |
| local_hello_how_you | 0.25 | HELLO HOW MOTHER | HELLO HOW MOTHER | HELLO HOW MOTHER |
| local_hello_how_you | 0.5 | HELLO HOW YOU | HELLO HOW YOU | HELLO HOW YOU |
| local_hello_how_you | 0.75 | HELLO HELLO HOW YOU | HELLO HELLO HOW YOU | HELLO HELLO HOW YOU |
| o5s5_lg_hello | 0.0 | WHY NIGHT | WHY NIGHT | WHY NIGHT |
| o5s5_lg_hello | 0.25 | HELLO LEARN | HELLO LEARN | HELLO LEARN |
| o5s5_lg_hello | 0.5 | HELLO HELLO STOP | HELLO HELLO STOP | HELLO HELLO STOP |
| o5s5_lg_hello | 0.75 | MORNING HELLO LESS | MORNING HELLO LESS | MORNING HELLO LESS |
| o5s5_lg_when | 0.0 | WHEN WHEN | WHEN WHEN | WHEN WHEN |
| o5s5_lg_when | 0.25 | WHEN WHEN WHEN | WHEN WHEN WHEN | WHEN WHEN WHEN |
| o5s5_lg_when | 0.5 | WHEN WHEN WHEN | WHEN WHEN WHEN | WHEN WHEN WHEN |
| o5s5_lg_when | 0.75 | WHEN WHEN WHEN | WHEN WHEN WHEN | WHEN WHEN WHEN |
| webcam_long_context | 0.0 | HELLO GOODBYE HOW YOU | HELLO GOODBYE HOW YOU | HELLO HOW HOW YOU |
| webcam_long_context | 0.25 | WHERE HELLO HOW YOU HELLO | WHERE HELLO HOW YOU HELLO | WHERE HELLO YOU YOU HOW HELLO |
| webcam_long_context | 0.5 | WHERE HELLO HOW YOU YOU | WHERE HELLO HOW YOU YOU | WHERE HELLO HOW YOU YOU |
| webcam_long_context | 0.75 | WHERE HELLO HOW TELL HOW | WHERE HELLO HOW TELL HOW | WHERE HELLO HOW YOU YOU HOW |

## Interpretation and next gate

A repeated sign within one short context cannot be fixed by rollover bookkeeping. Inspect its saved logits and window origins before selecting a training intervention. A changed output under a phase shift demonstrates sensitivity, not which phase is linguistically correct.

The rollover regression is covered by focused tests including blank/OTHER-separated repeats and the actual Reel event loop. This does not establish long-hold accuracy on real unseen signers.

Missing: expert-labelled normal/held/twice performances across independent signers, rest and OTHER coverage. Existing webcam clips remain development diagnostics. Do not train on these evaluations or invent their intended labels.

Training is deferred pending a justified objective and separate training examples. Uni-Sign remains a separately scoped gloss-free comparison; no external model or dataset was downloaded by this replay.

Artifacts: manifest.json, provenance.json, results.json, summary.json, and per-phase compressed emission archives. Each accepted-window archive contains logits, its frozen visual features, exact observed source timestamps, and the carried boundary token. CTC steps are model emission positions, not exact sign boundary timestamps.
