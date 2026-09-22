# Live familiar candidate versus Reel: session diagnosis

Reviewed candidate events/report from20260922_074639_809805 and20260922_074748_682276
under artifacts/reports/continuous_live_v17, and Reel history from
artifacts/app_sessions/20260922_074918_286973. These are separate signing runs, not
paired ground-truth accuracy measurements. No WER can be computed without reference
annotations. Candidate runs saved no video/features/logits; only decoded events.

## Measured history

Reel committed I at27.1678s and41.4404s, GO at29.5415s and42.9476s. Full verifier
scores respectively0.915/0.962(I),0.771/0.768(GO); scores are not calibrated accuracy.
Reel also emitted DOCTOR, SCHOOL, MORNING, WHAT, EAT, WHEN, FRIEND, MY, NAME, among others.
It also had rejected proposals and potential false insertions; no claim Reel is perfect.

Candidate run1 emitted HELLO HELLO YOU HOW, then OTHER TOMORROW; run2 emitted HELLO
OTHER OTHER HELLO HOW YOU, then OTHER TOMORROW TOMORROW. Neither emitted I orGO.
Greedy blank won408/418 and511/526loggedsteps respectively. Sparse CTC blanks are normal;
these counts alone do not diagnose why a particular intended sign was omitted.
Observed frame rates13.34/16.90 per wallsecond;1700/2136 featureticks versus801/1249
observations. The30Hzruntime repeats prior observed features. This and causal normalization
are input-domain concerns, not proven root causes.

## Training and model differences

Familiar training localphrases371: HELLO HOW YOU116,PLEASE HELP I115,MY NAME54,
THANKYOU FRIEND39,TOMORROW SCHOOL GO33,GOOD MORNING14. Across all admitted training
phrases I occurs only in PLEASE HELP I; GO only in TOMORROW SCHOOL GO. Zero I GO pairs.
There are28I and23GO isolated training records; they are not absent classes.
FINE is not a locked vocabulary label; no literal FINE emission appears in reviewed logs.

Actual frozen candidate encoder differs from Reel: stage1_v17_asllrp_core_adapt_v1
versus Reel's landmark phrase-adapt proposal and unified multimodal verifier. The19.14%
local score was not obtained by simply adding CTC to the current Reel encoder.

Selected candidate on cached isolated validation: I15/17headexact,16/17base meanlogit
argmax correct; GO9/16headexact,10/16base meanlogitargmax correct. Blank-only0I/1GO.
On training: I28/28headexact,GO22/23. These are separate single-sign diagnostics;
base pooled-window mean logits are not a continuous recognition metric.

## Current interpretation

User live feedback reveals the six-template validation result does not establish free
sign composition. Context-restricted training and visual-window/domain mismatch are
plausible; exact causal attribution requires head/base comparison on the same observations.
No language prior is enabled in the live candidate, so experimental bigram rescoring
cannot explain these failures. No threshold change or additional phrase-biased training
is justified from these logs alone. Bounded replay was attempted but the Reel MP4 lacks its moov atom and cannot be decoded.
No candidate-head replay scores were produced. The app process was not running when checked;
the file is unfinalized/unreadable, but the reason it was not finalized is not established.
