# v17 d=256 versus d=384 mobile-cost ablation

**Decision:** retain d=256. The wider model did not provide a reliable accuracy gain
and costs materially more storage and runtime.

| Metric | d=256 | d=384 | d=384 / d=256 |
| --- | ---: | ---: | ---: |
| Parameters | 6,470,885 | 14,338,853 | 2.22x |
| Citizen validation top-1 | 95.77% | 95.24% | -0.53 points |
| Citizen validation top-5 | 100.00% | 99.74% | -0.26 points |
| SemLex validation top-1 | 85.89% | 86.71% | +0.82 points |
| SemLex validation top-5 | 96.11% | 96.01% | -0.10 points |
| FP16 Core ML package | 12.58 MiB | 27.59 MiB | 2.19x |
| Mac CPU-only warm median | 0.564 ms | 1.007 ms | 1.79x |
| Mac CPU+Neural Engine warm median | 0.495 ms | 0.770 ms | 1.56x |
| Mac CPU+GPU warm median | 4.778 ms | 6.084 ms | 1.27x |

Citizen paired outcomes are four d=384 corrections and six regressions (exact McNemar
`p=0.754`). SemLex paired outcomes are 40 corrections and 32 regressions (`p=0.410`).
Neither supports a reliable d=384 accuracy advantage.

The Core ML models are fixed `[1,32,61,5]` iOS 15 FP16 ML Programs. Both matched the
PyTorch top-1 result on the export parity sample. Timings are warm batch-one runs on the
current Apple Silicon Mac, not sustained measurements on a low-end or medium iPhone.
Real-device package load, memory, cold latency, sustained latency, and thermals remain
required before claiming mobile readiness.
