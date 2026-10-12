# Paper metric lineage audit — 2026-10-13

Scope: verify saved measurements and their development context before paper edits.
No new training, protected-test evaluation, production change or manuscript/review edit.

## Findings

The WER values do not form one continuous, controlled improvement curve. The training
selector rescored cached candidate intervals using frozen DGS BIO evidence and its
experiment decoder. Runtime evaluation used the AV boundary model, streaming candidate
generation, recognition gates, different scoring parameters and commitment/flush rules.
The run_v2 checkpoint appears in both its training history (29.65%) and the runtime
configuration (17.70% baseline). Thus a large part of the apparent jump crosses evaluator
boundaries; its individual causes are not isolated by the current records. No claim that
transition fine-tuning caused31.42→12.83 is supported.

Training lineage: original selected chain31.42%, variant-cleaned chain30.09% (report),
restored-ASK/reviewed-input v2 chain29.65%. Data and training changes are bundled; these
are not one-factor causal ablations. Training-set accuracy is not what these WERs measure.

Runtime saved hypotheses were independently rescored with edit distance. All six audited
runtime variants have identical89identities and reference transcripts,226referenceglosses:

| Configuration | Errors | WER |
|---|---:|---:|
| v2 runtime baseline |40|17.70%|
| v2 with dip bridging/commit delay |42|18.58%|
| fragment-trained recognizer with those settings |35|15.49%|
| shorter fragments plus rejection, inherited settings |36|15.93%|
| matched auxiliary export/configuration and acceptance threshold0.4 |34|15.04%|
| subsequent ordered commitment/contact handling |29|12.83%|

Dip bridging reduced observed BIG/LANGUAGE fragmentation but worsened tuningWER.
Fragment training reduced insertions in that runtime comparison; shortened-fragment/
rejection training alone did not improve this WER. Do not omit these counterexamples.
Final five-error reduction: two NAME MY→MY NAME corrections (two edits each), and one
extra HELP removed (one edit). This is a decoder-output correction, not evidence that
isolated classification improved by the same amount. MY-transition fine-tuning onOct12
retained12.83%; it did not produce this reduction.

## Other paper numbers and provenance

95.50% current primary-head isolated validation: previous fresh verification3779/3957,
same correct count inPyTorch/FP16, one different prediction. Auxiliary head95.63→95.58%
after conversion is separate; neither is whole live-runtime accuracy. Evidence:
docs/capstone_papers/review_assets/20261013_verification/verification.json.
95.70→95.53% direct interval adaptation: different earlier chain,4069→4062/4252.
Checkpoint-swap diagnostic reports21lost/14gained and changes to both encoders andhead;
selection prioritizes WER subject to retentionfloors, not maximum isolatedaccuracy.
31.14ms/frame: measured camera-paced iPhone13 replay computation, not clipinference or
publicationlatency;26.56MB: complete currentFP16recognizer package. No causal speedup
claim can compare that protocol with historical23.69ms or classifier-only timings.
Basearchitecture95.50/95.77/96.83 remain component-screening results oncommoninputs,
not finalrecognizer accuracy or proof of universal architecture superiority.

## Wording boundary

Use simple functional wording, e.g. “The recognizer was fine-tuned on transition
examples.” Do not state “one head”: deployed code preserves original and adapted
outputs and applies adapted evidence selectively. Omitting a head-count discussion
is different from asserting a false count. No wording decision silently applied here.

## Remaining verification

A matched cross-evaluator ablation is needed to quantify the29.65→17.70 difference by
boundary model, interval candidates, score parameters and commitment policy separately.
No new causal percentages should be inserted until that comparison exists. Keep the
Transformer downstream experiment separate from this saved-evidence audit.

Sources: verification.json; canonical_recognition_comparison_v17_20261007/downstream_recipe/
chain_9683/span_recognizer/history.json; citizen_variant_local_filter_v17_20261008/REPORT.md,
run_v2/span_recognizer/history.json anduser_take_replay/config_v2_installed.json;
go_split_root_cause_v17_20261009/eval_variants.py and selectedresults files;
have_root_fix_v17_20261010/REPORT.md andguarded_regression_pair_val_tune_citizen_val.json.

## Verified WER data inventory

Manifest boundary_local_calibration_v17_20260922/manifest.json contains89local_phrases
recordings:59calibration+30confirmation;3familiar signers(30/36/23clips),226reference
glosses,6templates/15distinctglosses. Templates:HELLO HOW YOU18,MY NAME18,PLEASE HELP I18,
THANKYOU FRIEND17,TOMORROW SCHOOL GO12,GOOD MORNING6. Source policy originally reserved
confirmation from tuning; laterlab.rows_for('tune') combinesboth and intervaltrainer
selectscheckpoints onthatpool. Do notcall30confirmationclips independentconfirmation
of those later selections. No session/near-duplicate certification; exacthashesunique.
Canonical scoped phrase training179approvedtrain recordings;139approvedvalidation
recordings separately;3370derivedtrain candidate spans are not3370independentvideos.
Isolated replay usesCitizen/SemLex/localisolatedsigns, distinctfromtheWERdenominator.
TargetedOct12transitiontraining uses223windows from3physicalpairs and375development
windows from5pairs ofoneknownvideo; it retainedthe89clipWER, notcaused12.83%.
