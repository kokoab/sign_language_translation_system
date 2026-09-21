# Flores OTHER controlled experiment

User approved running this comparison alongside the corrected YouTube comparison.
Question: does adding genuine Flores sentences, with unsupported signing represented
by OTHER, improve the existing 100-sign causal recognizer?

- Keep corrected local cache, frozen Stage1 ancestor, 18 epochs, decoder, selection
  score and paired seeds 17321/17322 fixed. Start each arm from the exact saved
  baseline initialization; no YouTube pretraining in either arm.
- Run independent without-Flores and with-Flores arms; do not depend on the other job.
- Retain original raw glosses; match only uppercase exact locked labels after outer
  sentence punctuation removal. No normalized aliases or numeric-variant merging.
  Consecutive OTHER tokens become one unknown signing span; known repeats remain.
- Reuse matching-schema landmarks, never old model embeddings. Admit only complete
  source-frame coverage, finite arrays, valid hashes, dev-only roles, and feasible
  CTC sequences. Report exclusions. No protected test/devtest access.
- Preserve original source weights; added Flores total sample-weight mass is 10%
  of the original corpus. Uncertain cross-dataset signer identity is a limitation,
  not a signer-disjoint Flores generalization claim. Check exact video hash overlap.
- Evaluate the same held-out development sequences: WER, isolated accuracy,
  hold/repeat diagnostics, conditional emission delay, and OTHER emissions on
  isolated known-sign sequences. This last measure is a rejection diagnostic,
  not frame-grounded attribution of phrase deletions to OTHER.
- Audit/checks run in-session. Only training detaches after its first optimizer step.
  Two CPU threads, MPS memory fraction .12, no polling; macOS completion/failure notice.

Implementation: add optional supplemental training archives and fixed weight mass to
existing aligned trainer; one script for mapping/audit/preflight/paired run/report.
Checks: exact mapping, aliases/variants rejected, unknown-span collapse, known repeats,
CTC length feasibility, source weight preservation, all archive coverage, strict initial
checkpoint loads, real Flores CTC backward, and behavior report smoke. No promotion.
