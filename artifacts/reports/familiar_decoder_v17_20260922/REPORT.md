# Familiar decoder comparison

Fixed, offline comparison: greedy CTC, prefix beam width 8, the same beam with a fixed 0.1 uniform prior control, and a fixed 0.1 add-one-smoothed bigram extension score. The bigram prior is trained only on distinct familiar-training phrase transcripts; it has no EOS term or explicit completion rule, though unsupported insertions can still occur.

Known-WER collapses CTC then removes OTHER (101); exactness preserves OTHER. This is a reused development evaluation and does not establish arbitrary live performance.

## Familiar 17521

| Decoder | Local60 WER | ASLLRP12 WER | All WER | Local exact | Changed / improved / worsened | CPU ms/sample |
|---|---:|---:|---:|---:|---:|---:|
| Greedy | 19.14% | 33.33% | 18.55% | 33/60 | 0 / 0 / 0 | 0.00 |
| Beam 8 | 17.90% | 33.33% | 18.44% | 35/60 | 5 / 3 / 1 | 2.77 |
| Beam 8 + uniform control | 16.67% | 33.33% | 18.33% | 36/60 | 6 / 5 / 1 | 2.86 |
| Beam 8 + weak bigram | 16.05% | 33.33% | 18.28% | 37/60 | 5 / 5 / 0 | 2.90 |

## Familiar 17522

| Decoder | Local60 WER | ASLLRP12 WER | All WER | Local exact | Changed / improved / worsened | CPU ms/sample |
|---|---:|---:|---:|---:|---:|---:|
| Greedy | 22.22% | 33.33% | 18.88% | 27/60 | 0 / 0 / 0 | 0.00 |
| Beam 8 | 22.84% | 33.33% | 18.93% | 27/60 | 13 / 4 / 5 | 2.89 |
| Beam 8 + uniform control | 21.60% | 33.33% | 18.82% | 30/60 | 13 / 5 / 4 | 2.86 |
| Beam 8 + weak bigram | 20.99% | 33.33% | 18.77% | 31/60 | 14 / 6 / 4 | 2.92 |

## Coverage

```json
{
  "local_phrases": {
    "samples": 60,
    "distinct_target_sequences": 6,
    "sequence_seen": 60,
    "sequence_novel": 0,
    "distinct_target_bigrams": 9,
    "bigram_seen": 9,
    "bigram_novel": 0
  },
  "asllrp_contiguous": {
    "samples": 12,
    "distinct_target_sequences": 8,
    "sequence_seen": 3,
    "sequence_novel": 9,
    "distinct_target_bigrams": 8,
    "bigram_seen": 3,
    "bigram_novel": 5
  }
}
```

The word-pair prior is deliberately weak and was not weight-tuned. The uniform control separates its constant per-token log penalty from learned pair preference. Both priors only score candidates generated from visual CTC extensions; neither adds an EOS completion rule.
