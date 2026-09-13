# Ground truth archive — file map

Every dated entry from `PROJECT_GROUND_TRUTH.md`, split by topic. Nothing was deleted.

**How to use this:** you almost never read these files. `rg <term> docs/ground_truth/`
finds anything regardless of which folder it landed in. Open `high.md` when you need the
evidence behind a constraint; `rg` `log.md` before re-running an experiment.

| Topic | high | log | Span | Size | Covers |
|---|---:|---:|---|---:|---|
| [`stage1-architecture/`](stage1-architecture/) | 19 | 94 | 2026-08-09 → 2026-09-10 | 248 KB | Stage-1 encoder, extractor bakeoffs, and the component ablation ladder. |
| [`text-to-sign/`](text-to-sign/) | 6 | 67 | 2026-08-09 → 2026-09-07 | 125 KB | Reverse direction: SignWriting, avatar rendering, motion generation. |
| [`live-streaming/`](live-streaming/) | 10 | 45 | 2026-08-10 → 2026-09-10 | 137 KB | Live camera runtime: Reel path, streaming CTC, commitment and latency. |
| [`data-sources/`](data-sources/) | 8 | 33 | 2026-08-09 → 2026-09-08 | 93 KB | Corpus acquisition, licensing, admission audits, and split policy. |
| [`stage2-ctc/`](stage2-ctc/) | 9 | 17 | 2026-08-14 → 2026-09-10 | 69 KB | Stage-2 continuous recognition, CTC contracts, selectors, adaptation. |
| [`signing-voice/`](signing-voice/) | 2 | 17 | 2026-08-22 → 2026-09-01 | 48 KB | Signer style transfer, coarticulation, and transition synthesis. |
| [`mobile-deployment/`](mobile-deployment/) | 7 | 11 | 2026-08-12 → 2026-09-07 | 57 KB | Core ML export, orientation contract, iPhone and Flutter integration. |
| [`stage3-translation/`](stage3-translation/) | 2 | 3 | 2026-08-24 → 2026-09-06 | 16 KB | Gloss-to-English translation models and their gates. |
| [`capstone-paper/`](capstone-paper/) | 0 | 4 | 2026-08-24 → 2026-09-03 | 7 KB | Paper revisions, benchmarks for publication, repository hygiene. |
| **total** | **63** | **291** | 2026-08-09 → 2026-09-10 | **805 KB** | |

## Adding new entries

Append to the relevant `<topic>/log.md`, newest first. Promote to `high.md` only when the
entry establishes something that binds future work — a frozen gate, an accepted architecture,
a licensing decision. If it changes what a future session must *do*, also update the matching
section of `PROJECT_GROUND_TRUTH.md`.

