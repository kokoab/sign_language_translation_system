# Data findings versus runtime failure

The original Flores OTHER job has no completed paired result. Its failure was the
2.13GiB MPS process allocation cap, not a failed input audit or a recognition gate.
One without-Flores arm completed; the with-Flores arm stopped after its first epoch.

| Finding | Evidence | Action |
|---|---|---|
| Flores cached tails missing |14/155 archives omit1–3final frames | Excluded; full-video re-extraction could recover them in a later fixed-data experiment |
| Flores old embeddings use a different encoder | Recorded frozen-feature checkpoint differs from this experiment's ancestor | Already avoided: re-encode compatible Apple landmarks with the pinned ancestor |
| Flores glosses extend beyond100labels |141admitted clips map to476known occurrences/93labels and462OTHER spans | Preserve raw transcripts; exact mapping and OTHER spans are already implemented; semantic equivalence across corpora still warrants human review |
| Flores global signer identity unavailable | Metadata supplies dataset-local IDs | Cannot claim cross-dataset signer-disjointness; identity evidence is needed, not relabeling IDs |
| Local phrase labels were mixed | User-reviewed fixed subset removes55admitted training clips | Already corrected;232train/200validation, disjoint signer roles |
| Earlier ASLLRP short-sign/context preparation dropped useful supervision | Prior data-path report documents observation-floor rejection and premature context endings | Audit which paths consume those targets; do not repeat the native-rate cache extraction already completed |

No new proof of corrupt Flores video, incompatible admitted landmark schema, invalid
CTC targets, or incorrect retained local labels was found. Poor WER alone does not
identify a data-label defect. MPS retry holds data fixed so resource repair is not
confounded with another dataset change. Protected test/devtest data remain untouched.
