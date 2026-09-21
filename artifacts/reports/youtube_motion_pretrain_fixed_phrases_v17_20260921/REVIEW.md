# Corrected-phrase motion comparison review

## Completion review

All four checkpoints exist; all saved behavior edit counts independently recomputed.
Each arm used the corrected232local training clips and identical4554total training /
2182validation samples. Local evaluation includes200clips/540reference glosses.

Pretraining improves local WER in both seeds (48.52→43.15%;50.19→44.44%), mean
49.35→43.80% (5.56 percentage points). This is more consistent local evidence than the
original mixed-label experiment, but it is not a clean overall win. ASLLRP regresses
54.17→66.67% in seed17321; only1/2 predefined gates pass. Local phrase exact matches
42→46/200 and43→42/200. In seed17322, insertions90→31 but deletions52→130, so the
lower WER includes a substantial shift toward omitted signs. Duplicate output pairs
62→66 and75→38 do not establish held-sign correctness. Synthetic hold exact10→6/20
and6→6/20; repeat exact7→6/20 and6→5/20. No real adjacent-repeat references exist.
Conditional matched-token ASLLRP median delay changes0→−33ms and0→33ms on only8–11
matched events; this is neither an all-token latency score nor hardware latency.

Conclusion: promising local motion transfer, insufficient overall behavior evidence;
no promotion and no additional YouTube acquisition. Existing development sets were
reused; no protected test was accessed. Completion notification returned0. The separate
Flores experiment is unaffected and was not polled during this review.

