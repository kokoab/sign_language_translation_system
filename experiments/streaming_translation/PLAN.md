# Streaming English replay experiment — 2026-10-13 PHT

Authorized: existing recordings first; local pretrained models; optional translator
adaptation capped at 30 minutes; separate live demo only if evidence justifies it.
No writes to ATLAS, no deployment, no acquisition of sign datasets, no protected
test access. All experiment files live here. Existing artifacts are read-only.

Freeze before inference: compare Finish, 1.5-second no-commit timeout, revisable
full-context captions, and full-context captions with conservative sentence commitment.
Use the same inputs and model for every policy. The timeout is a deliberately weak
baseline: no-commit is not established hand rest. Adaptive commitment requires an
unchanged completed sentence across two distinct prefixes; it retains source context.
Unsupported final tails stay provisional rather than being forced into speech.
No unseen future text, annotations or final reference may enter generation or policy.

Data: all 26 reviewed September29 phone utterances, their original word events and
Finish-only words. Missing Finish timestamps preclude actual user-wait savings.
Only event-complete utterances enter causal timing replay. All26 enter full-input
translation checks. These are historical recognition logs, not fresh visual recognition
or unseen-signer evidence. Include explicit text diagnostics for roles, time, negation,
greetings, partial inputs and delayed continuations; synthetic timing stays separate.
References assess faithfulness to input glosses, not independently verified ASL meaning.

Models: existing tiny counted-sentence T5 and cached google/flan-t5-base. Fixed
instruction prompt, deterministic greedy decoding, CPU FP32. Record source hashes,
raw predictions, fallbacks, model time and policy events. No fine-tuning initially.
Cross-check tiny whole-input output against saved native Swift results; distinguish
weight replay from exact application parity if predictions differ.

Metrics: diagnostic reference agreement (normalization only, not general translation
accuracy), individually reviewed semantic errors, first-caption delay from first
recognized word, tail-finalization coverage, frozen-text conflicts, changed final
output relative to full input, revision erasure and measured serial decoder backlog.
Report lexical agreement limitations. Desktop inference time is not iPhone latency.
Phone source timestamps represent recognition arrival, not camera-to-recognition cost.

## Recorded amendment after first model runs

All26 tiny full-input predictions exactly matched saved native Swift outputs. The
zero-shot FLAN prompt mostly copied glosses and dropped NO in explicit negation cases.
Before further inference, added a fixed four-example prompt condition (examples are
not exact evaluation inputs) and Qwen2.5-0.5B-Instruct with the same four examples.
This is transparent development screening, not an untouched model-selection test.
Qwen weights download to this directory; generation remains entirely local. No training.

Actual prepared set:26phone+24declared semantic diagnostics+2synthetic gap stress cases.
20phone utterances have complete original word events. Six require Finish-only words
and are excluded from all causal timing comparisons, while retained for full-input checks.
Finish baseline uses oracle end at final word as a lower-bound proxy; historical
Finish timestamps are absent. No measured human waiting-time reduction is claimed.

Qwen few-shot screening sometimes explains the demonstrations or answers the user
instead of translating. Added one final fixed prompt condition before its run:
system-message translation instruction and gloss-only user input, no demonstrations.
No more prompt search on these cases. Both prompt conditions remain reported; neither
is a held-out evaluation. No training budget is consumed by prompting.

Live-demo gate: no known critical semantic regressions introduced by commitment,
useful captions without persistent backlog, and sufficient correctly finalized tails.
Record failures rather than tuning thresholds repeatedly on this diagnostic set.
