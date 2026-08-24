# Stage 2 v17 Core ML export

The retained compact Stage 2 graph is now represented by two FP32 Core ML packages:

- `Stage2FrozenEncoderV17FP32.mlpackage` converts v17 landmarks plus precomputed
  MobileCLIP2 hand embeddings into up to eight 32-frame window embeddings. Package
  tree SHA-256: `1146b539800e6f09a743f4a8ee882c9b2cd2b01503ff3f362a11e97e8c827bb9`.
- `Stage2CompactContextV17FP32.mlpackage` applies the exact packaged context adapter
  and CTC head. Package tree SHA-256:
  `e92ba7d8b7c61c52bc776840e953c73abb6b012637991d01582d4fd64067760a`.

Cold combined validation covered 363 samples and 574 windows with zero Core ML versus
PyTorch decode mismatches. It reproduced the retained compact metrics exactly:

| Gate | Edits/tokens | WER | Exact sequences |
| --- | ---: | ---: | ---: |
| ASLLRP contiguous phrases | 11/24 | 45.83% | 2/12 |
| Local phrases | 7/259 | 2.70% | 90/97 |
| ASLLRP segmented contextual | 43/254 | 16.93% | 213/254 |

The measured 12.42 ms median and 12.94 ms p90 are Mac-host Core ML timings only. They
are not physical-iPhone latency, ANE, thermal, or sustained-performance evidence.
The packages consume MobileCLIP2 hand embeddings; the crop-to-embedding MobileCLIP2
network is not yet in the iOS app. Therefore this is a validated Stage 2 Core ML graph,
not a complete camera-to-gloss mobile deployment.

Machine-readable evidence is in `frozen_encoder_fp32.json`, `compact_fp32.json`, and
`pipeline_validation.json`. No project test split or 2M-Flores `devtest` was accessed.
