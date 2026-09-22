# Local familiar-signer controlled adaptation

User explicitly authorizes relaxing signer separation for local phrases, parameter
adjustment and research on transition/coarticulation augmentation. This trial isolates
signer familiarity before mixing a second augmentation intervention.

Starting pipeline: the older aligned-grounded causal CTC checkpoint, freshly evaluated
on current211approvedphrases at39.04%WER (local38.73%). Reuse its pinnedStage1 and original
source-frame reconstruction, rolling8frame windows/stride4, pooled rich evidence. No
new model architecture, source acquisition, test access, or automatic deployment.

Source membership is the same6421records. The versioned split keeps originaltraining
records and other-source roles; of199local signer02clips,139move to experimentaltraining
and60remainvalidation, stratified by the6target sequences using seed17521hash ordering.
Whole-video hashes disjoint; canonical manifests and source metadata unchanged. Citizen
and other official splits unchanged. This intentionally measures familiar-signer held-out
clips from a reused development pool, not unseen-signer generalization. Original211clip
figures are historical after training on139ofthoseclips; never relabel them as independent.

Two seeds17521/17522. Control trains on original4547records; familiar arm adds139local
records (4686total). BOTH evaluate the same60local+12ASLLRP+1663single validation records.
Stage1frozen/eval, cache its evidence once. Same initialhead and RNG/dropout reset perarm,
6epochs,AdamWheadlr0.0001,weight_decay0.001,gradclip5. Each epoch4264single visits in
batches32 paired with4phrases/update (536phrasevisits). Phrase groups283control/422familiar
cycle/shuffle uniformly; addedlocal recordings change phrase composition as intended.
Loss target-normalized phraseCTC plus correlation-weighted singleCTC (shared-parent0.5).
Blank0,known1–100,OTHER101; collapse BEFORE OTHER removal in knownWER.

Select by local60validation WER; earliesttie, includeepoch0. Retain ASLLRP/single-source
S/D/I/exact and baseline/selectedretention, not just selector. Epoch0samevalidation must
match betweenarms. Evaluate selectedcheckpoints on their corresponding fulltraining set.
No single-score automaticpromotion. This is a bounded familiar-signer experiment, not a
claim that changing the split itself improves recognition of strangers.

Preparation verifies allsource,role,feature,raw,checkpoint,split andcodepins; requires
MPSfiniteevidence on everyrecord, finiteheadCTCgradients, absentbasegradients, zerooptimizer
steps. Dedicatedrecipe/preflight authorizes only this runner. Oldgeneral gate staysfalse.
Train detachedwithcaffeinate, completion/failurenotification; no trainingpolling.
