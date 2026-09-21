# Flores OTHER audit

155 acquired dev videos were hash-verified against the manifest. All landmark archives
match the existing Apple schema and have finite arrays and contiguous source ranges.
141 complete clips admitted;14 caches omit1–3 tail frames and were excluded.
Admitted supervision:476 known occurrences covering93 of100 labels,462 OTHER spans.
Raw transcripts remain recorded. Exact label matching only; consecutive unknowns form
one span, known repeats preserved. All CTC targets fit their available temporal steps.
No exact video hash overlaps with the existing phrase train/validation caches.

Existing Flores frozen embeddings are incompatible with this experiment's frozen
Stage1 checkpoint and are not reused; landmarks are re-encoded. Source-local signer
IDs cannot establish global identity, so cross-corpus signer overlap is unknown.
The existing development evaluations have been reused; results are not a fresh test.
The 10% supplement cap concerns total sample-weight mass, not a guaranteed fraction
of loss or gradients. Official test/devtest data remain sealed.

Real longest-sequence CTC backward, finite gradients, optimizer step and strict saved
initialization loading passed. The shared behavior input pipeline has289 sequences.
See audit.json for every admitted mapping/hash and excluded clip.
