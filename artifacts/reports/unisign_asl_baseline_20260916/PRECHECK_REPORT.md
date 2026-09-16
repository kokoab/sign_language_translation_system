# Uni-Sign ASL comparison precheck

Prepared an inference-only comparison; **model quality is not measured yet**.

- Official Uni-Sign revision: `eed438bcb49e30405cd6ccdfcccca330c134e830`.
- Released How2Sign pose-only checkpoint SHA-256: `1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683fb00d`.
- 12 English-paired validation utterances, four source videos, filename signer IDs 1 and 2. Selection uses names and duration only, before predictions.
- Nine earlier difficult recordings, qualitative only. Previous CTC outputs will appear alongside new translations.
- Native lightweight Wholebody preprocessing, native regional normalization and seeded frame sampling. CPU pose extraction; MPS/CPU float32 translation; beam width 4.
- Original sentence boundaries differ from realigned annotations. The worker cuts raw validation video using corrected times rather than trusting the old sentence clips.
- Self-check passed native adapter equivalence, shapes, confidence masking and exclusion of English references from model input. Compilation and whitespace checks passed.

The worker requires strict checkpoint loading, preserves predictions and input hashes, reports BLEU/chrF only for paired references, and records desktop time and memory. This small slice cannot establish benchmark accuracy, signer-disjoint generalization, semantic correctness or iPhone readiness.

`run_baseline.py --launch` starts detached work. Completion writes `REPORT.md`, `results.json`, `summary.json`, `provenance.json` and `completion.json`; failure writes `FAILURE.md` and `completion.json`. One macOS notification announces exit. No training, automatic promotion or polling.
