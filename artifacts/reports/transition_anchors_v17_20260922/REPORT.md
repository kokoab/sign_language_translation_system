# Timed-anchor experiment launch

User asked for faster work and actual transition learning. Independent review cleared
anchor prep/contract and runner0cdb5b345116b24ef949b2df17fa83dc12f14963dd1848ebda97fa1445212bad.
Focused anchor-loss test passed, including blank0/known1..100, ignored tokens and finite
nonzero gradients. ActualMPSpreflight passed6421records/7367richwindows perseed; cached
versus direct logits maxdifference0.0both; each gradientprobe included4known/2blank regions,
allheadgradientsfinite/nonzero, basegradientsabsent, zerooptimizersteps. gitdiffcheckclean.

Launched detachedcaffeinate PID40294 at2026-09-21T22:57:08.955694UTC (Sep22PHT).
Two selected adapted initializations17421/17422; control and anchored arms;3epochs each.
Stage1frozen, richfeaturescachedonceperseed. Anchored loss adds0.25intervalCE to unchanged
CTCloss, central signcores plus bounded internalgapblank hypothesis. Genericgateunchanged.
Recipe SHA03b2fac300e3b8fb45a5be62fa8147c20b406867007c530e21649f55e503fe74;
preflightSHA8cb563bc79287a142b3508c4d88c4751776dba87fffe5b696ccfffb00bf8b8a6.
Newfiles: scripts/train_transition_anchors_v17.py, test/test_transition_anchors_v17.py,
scripts/prepare_transition_anchors_v17.py, active/v17/transition_anchors_manifest_20260922.json;
reports under artifacts/reports/transition_anchors_v17_20260922. Earlier matched diagnostic
script/report and sourceinterval audit retained. No trainingpolling or completionclaim.

Nextsafeaction: readnewstatus/results nextsession, compare anchored vscontrol acrossboth
seeds includingepoch0, phraseWER/exact, single-source retention and selectedtrainfit.
No livepromotion based solely on auxiliary loss falling; step5stilldepends on quality,
holds/repeats and latency. Currentapp/livecode/model defaults untouched; no protectedtest,
acquisition or blanket extraction. Allfive recovery steps retained.

