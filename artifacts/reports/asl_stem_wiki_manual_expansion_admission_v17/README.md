# ASL STEM Wiki manual admission audit

The downloaded pool is **not training eligible**. Source notes explicitly map only P13 to Subject 2 and P28 to Subject 7. The appendix classifies Subject 7 as possible L2 signing, so P28 is excluded. The other downloaded participant-to-subject mappings are unresolved, and none of the manual gloss strings proves the frozen ASL-LEX visual variant.

- Fully decoded videos: 267/267
- Signer/gloss review pairs: 206
- Pairs with a pseudo-position proposal: 103
- Queue statuses: {'l2_excluded': 10, 'source_review_unclassified': 11, 'subject_mapping_unresolved': 185}
- Eligible verified spans: 0
- Training or protected evaluation run: no

Complete `expert_review_queue.csv` by entering signer quality, exact variant, and verified start/end frames. Pseudo positions are review aids only. A later admission step may build short participant-disjoint clips only from rows with all three approvals.

Run `venv/bin/python scripts/review_asl_stem_wiki_manual_v17.py` for the local
side-by-side video review UI. It writes approved decisions back to the queue.

## Carried reviews and automatic proposals

After this audit, 63 completed reviews from the original queue were carried by exact
filename, video SHA-256 and ASL-LEX-code identity. They preserve 33 eligible spans.
The remaining rows are still quarantined. Launch the reviewer with the expanded queue
and annotation paths shown in
`artifacts/reports/asl_stem_wiki_manual_expansion_auto_annotation_v17/README.md`.

## Final review result

The review is complete. `expert_review_queue.final.csv` is the frozen snapshot with
SHA-256 `38437d0afd506d050b0b89a452ae0a9d769d72e28496d9ab3b6c92235d64650a`.
It contains 111 eligible spans across 31 glosses and 18 participants. Eighteen glosses
have at least three approved participants and clear the minimum 2-train/1-validation
coverage floor. See `final_review_summary.json` for exact per-gloss coverage.
