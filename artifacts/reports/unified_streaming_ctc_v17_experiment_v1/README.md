# Unified streaming CTC experiment (v17)

## Decision

The 32-frame rolling-window CTC head is the only promising variant, but it is **not
ready to replace** `live_reel_stage1_v17.py`. It shows that overlapping windows are
useful when one causal decoder handles blank/background and duplicate collapse. It
also shows that the present offline phrases do not generalize sufficiently to the
held-out signer/domain.

No accepted model, live file, or protected test split was changed or accessed.

## What was trained

The frozen phrase-adapted Stage-1 model supplies, for each 32-frame window:

- its 256-dimensional pooled landmark embedding;
- its 100 original gloss logits.

A separate 111,406-parameter causal temporal head produces 102 outputs: CTC blank,
the locked 100 glosses, and `OTHER`. There is no phrase-template lookup, expected
phrase list, or language-model prior. Greedy decoding collapses repetitions and drops
blank and `OTHER` from the displayed known-gloss sequence.

Training used all compatible local material:

- 434 exact-sequence phrase clips;
- 879 ASLLRP spans with explicit known-gloss/`OTHER` sequences;
- 2,864 isolated clips as one-gloss replay;
- 5,636 incomplete-prefix and inter-sign transition samples as blank supervision.

How2Sign and 2M-Flores were not assigned 100-gloss targets: their local archives do
not provide equally pinned exact labels for this vocabulary. Calling their unknown
signing “blank” would train the recognizer to suppress real signs.

## Matched validation results

| Variant | Exact phrases | Phrase WER | Isolated exact | Boundary false emission |
| --- | ---: | ---: | ---: | ---: |
| Pre-pooling frame evidence, cached windows | 46.79% | 26.15% | 58.04% | 14.21% |
| Pooled evidence, cached non-overlap windows | 55.05% | 37.46% | 67.18% | 16.58% |
| **Pooled evidence, rolling stride 4** | **76.15%** | **10.60%** | **66.00%** | **12.64%** |
| Rolling stride 4, gloss correction frozen | 26.61% | 45.23% | 59.66% | 11.08% |

The selected rolling model's aggregate hides a large domain split:

| Validation source | Clips | Exact | Known-gloss WER |
| --- | ---: | ---: | ---: |
| Local phrases | 97 | **85.57%** | **5.79%** |
| Signer-disjoint ASLLRP contiguous phrases | 12 | **0.00%** | **62.50%** |
| ASLLRP known-plus-`OTHER` spans | 225 | 17.33% | 75.00% |
| Citizen isolated | 378 | 73.28% | 26.72% |
| Other isolated validation | 978 | 63.19% | 36.81% |

The accepted phrase-adapted Stage-1 checkpoint is 95.77% on the same 378 Citizen
validation clips. Therefore the new CTC path currently sacrifices 22.49 points of
isolated accuracy and cannot be promoted.

## Does it assume every input is a phrase?

Architecturally, no. It can emit nothing for any window and it has a distinct
`OTHER` output for real signing outside the known vocabulary. On held-out local
phrase crops it emitted no known gloss for 97.68% of incomplete prefixes and 97.53%
of inter-sign transitions. Across all boundary negatives, including much harder
isolated prefixes, the no-false-emission rate was 87.36%.

This is not yet proof against head scratching or arbitrary daily motion. The current
negative set contains prefixes and transitions, not enough natural non-sign activity.
Those real negatives remain necessary.

## Failure audit

All 12 held-out ASLLRP contiguous clips failed exact sequence match, usually by
under-emitting the second gloss. Examples include `FRIEND NOW -> FRIEND`,
`WORK WHERE -> WORK`, and `LIKE READ -> LIKE`. The local failures similarly include
`HELLO HOW YOU -> HELLO HOW` and `MY NAME -> MY`. The remaining errors include the
known `GOOD`/`THANKYOU` confusion and occasional repeated insertions.

This means the model has learned the nine repeated local phrase templates much more
strongly than general continuous recognition. The weekend signer-disjoint phrases
are still useful: they should supply broader signer, transition, and sequence variety,
not simply more copies of the same templates.

## Reproduction

```bash
venv/bin/python active/v17/train_unified_streaming_ctc_v17.py \
  --evidence-level window \
  --rolling-stride 4 \
  --output-dir artifacts/models/unified_streaming_rolling_ctc_v17_experiment_v2
```

The existing `v1` checkpoint and complete metrics are in
`artifacts/models/unified_streaming_rolling_ctc_v17_experiment_v1/`.

## Recommendation

Keep the stable Reel path for current live demonstrations. After the 30-phrase
collection arrives, retrain this rolling head with the new signer split and evaluate
three gates independently: continuous exact/WER, isolated retention, and ordinary
non-sign false activations. Only build the separate live entry point after the model
passes those gates; a camera UI around the current checkpoint would make a local
template learner look more general than it is.
