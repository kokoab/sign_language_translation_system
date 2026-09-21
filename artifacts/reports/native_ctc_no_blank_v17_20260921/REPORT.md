# Native-rate CTC without standalone blank clips

## Decision

The candidate **does not pass the matched improvement gate**. This is a matched ablation of the existing source-rate,
8-frame causal-window CTC experiment. It removes only standalone transition clips that
were labeled blank. Full phrase CTC, explicit `OTHER`, timed NCSLGR alignment, locked-100
isolated replay, the Stage-1 initialization, seed, and validation splits remain fixed.

| Validation set | Baseline exact | Baseline WER | Candidate exact | Candidate WER |
| --- | ---: | ---: | ---: | ---: |
| Local held-out signer | 25.50% | 37.04% | 16.00% | 45.19% |
| ASLLRP contiguous | 33.33% | 41.67% | 33.33% | 41.67% |
| NCSLGR held-out signer | 5.41% | 78.00% | 0.00% | 84.00% |

Isolated exact is 82.37% before and
83.19% after. The selected candidate epoch is
8; training took 106.0 seconds
on the configured device.

Because this ablation intentionally contains no standalone blank validation samples,
it does not claim a measured ordinary-motion false-activation rate. The regression on
two independent held-out signer sets is sufficient to reject it.

No runtime was changed. The official Citizen test and external reserved evaluation were
not accessed. Machine-readable result: `artifacts/models/native_ctc_no_blank_v17_20260921/result.json`.
