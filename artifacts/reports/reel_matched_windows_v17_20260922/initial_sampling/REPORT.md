# Matched Reel window comparison — 2026-09-22

Executed current Reel landmark proposal and full visual verifier on identical extracted
observations from12approved, previously-used ASLLRP validation videos. Raw video hashes
match combined manifest. No training or protected test. Same observed frame pool reused
for annotation-delimited core and ±100/250ms crops; existing wrist-motion trimming stays
active in all arms, isolating crop context under current inference. Source fps determines
timestamps;20Hzprocessing and normal sparse body/face schedule use shared live observer.

## Strictly paired results

Six of17eligible historicalcores contained fewer than4fresh20Hzobservations and were
excluded from the paired table. This is a real fast-sign sampling limitation, not a
failure prediction counted as correct. Eleven identicalcoreidentities remain perarm.

| Window | Proposal correct | Verifier correct | Correct conditional commits | Wrong conditional commits |
| --- | ---: | ---: | ---: | ---: |
| Annotated core |6/11|8/11|6/11|0/11|
| Core plus100ms eachside |7/11|7/11|5/11|0/11|
| Core plus250ms eachside |7/11|6/11|2/11|2/11|

These are small matched-window diagnostics, NOT fullstreamWER or unseenbenchmarkaccuracy.
Surroundingcontext can include a neighboringsign; a differentprediction isn't automatically
an insertion. Example coreFRIEND in841314 becomesWHEN in250msexpansion; coreTIME in4236779
becomesFRIEND. Both expandedwindows pass the checked commit conditions.

## Explicit non-overlapping gap test

Three gaps between known annotations had >=4observations, after20msedgeguards, with no
annotatedevent overlap includingOTHER. In842935,event1→nextgap,0.5205–0.8475s:
proposalWORK0.632, verifierWORK0.615, bothaccepted andconditionalcommitpasses. Other gaps
produced NIGHT/WORK andMAYBE/NOW but failed checkedcommitconditions. Gapcase shows a
false sign can be supported by both models, not just a threshold rejecting correctfeatures.
Annotations, not a fresh humanvideoaudit, define these gaps.

Conditionalcommit bypasses candidateactivation/proposalstability/history and starts a
freshVerifiedCommitLock perwindow. It checks currentacceptance/agreement/commitscore,
not whether a fulllive scheduler actually emitted. Rawwindowoutputs retained inresults.json.
All11corewindows met a within-windowwriststartcondition; this subset cannot diagnose
stationary-wrist omissions. Two correct verifiercoretop1 predictions failedconditional
commit checks (WATER/FRIEND), so evidence/gating issues coexist with modelconfusion.

## Recording blocker and interpretation

Five newest appMP4files, including081431, cannot decode (missingmoovatom). Latestcandidate
runs have no video. Thus this comparison uses existing approved rawsources and cannot
attribute the user's exactHELLO→MY orI GO attempt. CurrentReel is not perfect even on
annotatedcores; surroundingmotion further changes evidence, and a checkedgap can produce
a confidentfalse sign. Datasetcorruption oruniversalmodelincapacity is not established.

Next action: repairrecordingfinalization, use this matched diagnostic as a regressioncase,
then test transition-discriminating supervision alongside genuineWORK/GOODBYE and
low-motion positives. Don't loosen thresholds or switchencoder basedon these11samples.
No inferenceweights orcommitthresholds changed for this comparison.
