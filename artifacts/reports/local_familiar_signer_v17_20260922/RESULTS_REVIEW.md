# Familiar-signer experiment: completed review

All four runs completed six epochs. Checkpoint SHA256 values match recorded hashes;
earliest-best selection including epoch zero verified; paired initial validation metrics
match. Every epoch visited all4264single records once and536phrase examples. No live
promotion or new training was performed in this review.

| Arm | Selected epoch | Local60 WER | Local exact | Local training WER |
| --- | ---: | ---: | ---: | ---: |
| Control17521 | 0 | 35.80% | 20/60 | 0.16% |
| Familiar17521 | 6 | 19.14% | 33/60 | 4.87% |
| Control17522 | 0 | 35.80% | 20/60 | 0.16% |
| Familiar17522 | 5 | 22.22% | 27/60 | 6.26% |

Local errors S/D/I: control14/23/21, familiar17521 6/16/9, familiar17522 8/18/10,
162 reference tokens. Familiar adaptation reduced all three error types. Both controls
selected unchanged initialization: their further training did not improve local60WER.
This supports the added-local-data recipe, not proof signer identity alone explains the
gain: data quantity/composition also changed. Stage1 stayed frozen, so it does not prove
new coarticulation representations were learned.

Both familiar heads achieved33.33%known-WER and6/12exact on the small ASLLRP validation
subset, versus41.67% and4/12 at initialization. Their Citizen CTC single-sign WER was
6.35/6.08%, SemLex17.84/17.63%, segmentedASLLRP24.80/25.98%, STEM14.29/14.29%,
O5S5 77.19/78.95%. These are CTC-output metrics, not new Stage1 classifier accuracies.
O5S5 remains weak. Do not claim broad all-dataset success.

Metric distinction: this trainer removes OTHER101 after CTC collapse for known-WER;
the previous standalone211clip evaluator counted OTHER. Local examples are unaffected,
but ASLLRP/pooled numbers must not be compared across these scoring conventions.

Scope:60 held-out recordings, six local phrase patterns, familiar signer, reused development
pool and checkpoint selection set. Not unseen-signer/unseen-phrase generalization. The
19–22%range is measured here, not a prediction of arbitrary live signing. Original211clip
benchmark cannot be reused for independent evaluation after139clips entered training.

## Recommended next work

1. Use familiar17521 as the leading candidate,17522 as replication. Verify live input-window,
checkpoint packaging and streaming state equivalence before offering a candidate live mode.
Then evaluate single signs, low-motion signs, repetitions, interrupted phrases and unseen
combinations. Do not replace the live default from local60WER alone.
2. Add a bounded offline decoder comparison on fixed logits: greedy, CTC prefix beam without
language prior, and identical beam with a weak smoothed gloss bigram prior. Train prior only
on experiment-training transcripts; no validation transcript fitting. Include zero prior
weight and preserve all100known signs with smoothing/backoff. Use gloss order, not English
sentence grammar. Prefix decoding may rank visually supported alternatives but must not
append missing signs to complete a phrase. Measure WER S/D/I, exactness, changed-token
errors, single/repeated-sign retention and latency. Six local templates cannot establish
success: include existing independently held-out phrase combinations and report limited
coverage honestly. Do not tune weights repeatedly on the same reported set.
3. Keep actual continuous-context crop jitter as the next separate visual training test;
use known parent intervals and preserve holds. Language rescoring does not teach motion.
Do not combine new visual augmentation and language priors in the first comparison.

## Relevant primary paper

Sign Spotting Disambiguation using Large Language Models (2025):
https://arxiv.org/html/2507.03703v1
Combines visual candidate scores with conditional gloss probabilities in beam search.
Table6 reports LateFusion top1WER47.24%before rescoring versus44.38%(Phi3) or44.73%(Gemma2).
The34.81%top5 figure is a multiple-hypothesis metric, not single-output WER. Different
sign-spotting system/data; no numerical prediction for this project. Its candidate-based
contextual disambiguation supports the proposed experiment; choosing a small smoothed
bigram instead of an LLM is our speed/data-size design decision, not the paper's result.

Language-prior experiment is proposed and recorded, not implemented or launched this turn.
