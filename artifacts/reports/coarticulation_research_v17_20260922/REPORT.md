# Coarticulation research and current supervision audit

## What the current records actually teach

The combined training pool has4264single-sign records. Citizen1475andSemLex1388supply
clip-level identities. ASLLRPsegmented1116,O5S5positivecores195andreviewedSTEM90retain
interval/crop provenance, but the recent loader presents them as one-target sign examples.
They do not uniformly supply entry/sign/exit state labels or named previous/next signs.
Canonical ASL-LEX identity checking is not temporal-phase annotation. Defaultisolated
v17extraction enables hand-activity trimming; phraseextraction disablesit. Raw parent
context is therefore a separate resource, not something to assume the cached positive
core still contains. No newdata acquisition is required to audit/use existing contexts.

## Primary research read

1. Zuo et al., EMNLP2024, Towards Online Continuous Sign Language Recognition and Translation:
https://arxiv.org/html/2401.05336v2
The model trains on sign-centered crops from continuous videos that include varying
surrounding context, plus background and saliency supervision. Their ablation reports
48.4→24.4%testWER after adding signaugmentation. This supports traininginput/livewindow
matching; their RGB+keypoint model and benchmark do not predict our ASL100performance.

2. Villaplana et al., IberSPEECH2021, Generation of Synthetic Sign Language Sentences:
https://www.isca-archive.org/iberspeech_2021/villaplana21_iberspeech.pdf
Their Spanish-sign/LeapMotion/HMM study removes isolatedrest, interpolates between signs
and adds noise. Real-sentenceWER improves fromabout52%to34–37%withsyntheticdata; training
onreal sentences remainsbetter at16.4%. Noiseonlyoninterpolation points had littleeffect;
they expandednoise towholevectors. This is supporting evidence for controlledaugmentation,
not a prescription tocopy theirnoiselevels or treat syntheticmotion as realcoarticulation.

3. Min et al., ICCV2021, Visual Alignment Constraint:
https://arxiv.org/html/2104.02330v2
They diagnose CTC alignment-module overfitting and weakfeedback tovisualfeatures, adding
visual enhancement and alignment losses. Implication for us: highisolatedaccuracy and
lowtrainingCTCloss do not establish good contextualvisualrepresentations. Frozen-head-only
failure does not rule out improving contextualencodertraining.

## Implementation inference, not a claimed paper result

Prefer actual annotated parent context: temporally jitter crop start/end around known
signs, varyduration/contextamount, preserve handshapeandholds, and never use lowwristmotion
alone as rest. Use exactpositiveintervals as anchors; do not label all surrounding signing
blank. Contextchanges can extendinto sign onset/offset, so coarticulation is not only a gap.

Syntheticbridges can supplement realclips with bounded smoothvariations in duration/path,
compatiblecoordinateframes and same-source/signer pairing whereavailable. Missing masks,
handidentity and actualsigncores must remainvalid. Use wholeknown signsequence targets,
not invented manualboundarytruth. Test against a real-onlycontrol; no importingv16arrays
into v17. Keep this separate from local splitrelaxation so results are attributable.

## Work executed now

Oldercheckpoint evaluated oncurrentapproved211phrases:39.04%overallWER,38.73%local,
45.83%ASLLRP. Neweradaptedmodels were50.98/53.65%overallphrases. This is a fullpipeline
comparison with differentweights/windows/evidence, not proof pooling alone is responsible.

Userauthorized familiar-signer localexperimentprepared: keep232original localtrainclips,
move139signer02clips totraining, retain60otherwholeclips forvalidation. Allother roles
unchanged; rawvideohashesdisjoint. Originalsplit/manifests preserved. These resultswillbe
familiar-signer developmentresults; old211validation cannot remainanindependentbenchmark
once139ofitsclips havebeenusedfortraining. See local_familiar_signer_v17_20260922.
