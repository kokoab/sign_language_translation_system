# Authorized bounded Reel recognition adaptation

User approved implementation, polling only if training takes under five minutes.

Ruling: start with existing proposal classifier and verifier fusion heads. Freeze all
encoders, including MobileCLIP and temporal encoders, to retain existing representation
and make exact runtime feature caching possible. Last temporal block unfreezing is
not silently added if this run fails. This is a bounded first phase, not a promised
final accurate model. Existing contextual adaptation attempts are not claimed as new.

Use deduplicated, reviewed complete known-sign intervals from established ASLLRP.
O5S5 remains positive-only/exact-core supervision and is outside this context-crop run;
its source_crop_complete is unspecified, so it is not silently treated as complete. Preserve original source parent/signer roles.
Context 0/100/200ms stays outside neighbouring annotated signs; overlapping annotations
are excluded. No weak equal-partition local phrase labels, no inferred gap negatives.
Citizen1475 and SemLex1388 exact existing training cache membership provide isolated
replay; Citizen378/SemLex978 validation guard against forgetting. Protected test stays
sealed. STEM and other sources remain preserved, outside this bounded recipe.

Execute actual Reel preprocessing/hand encoder, capture existing Core ML providers,
check checkpoint/Core ML logit parity, then cache frozen encoder outputs. Verify existing
isolated caches against exact source tensor weights and hashes. Reconstruct all59
whole-video calibration decisions from the original frozen BIO intervals; require exact
baseline hypothesis parity before any fitting. Keep89 local calibration/confirmation
videos outside fitting. Confirmation features can be prepared, but scores are consulted
only after candidate selection. Existing familiar-signer/reused-development limitations
remain; the inherited Reel model is not a clean unseen-local-signers baseline.

One seed, maximum80epochs, patience10epochs without best trained whole-video WER
improvement; no forced40epoch floor. Half reviewed context, half isolated replay;
label smoothing and KL to original logits. Select lower calibration WER with no
correct-count loss, no additional insertions, >=98% baseline retained positions, and
<=1percentage-point isolated accuracy loss for each model/domain. Always retain frozen
fallback and actual best trained checkpoint separately. Save candidate standard weights;
no automatic live/default change or distillation. CPU cached-head training; preparation
and actual first-epoch runtime reported separately. Long run detached with notification.
