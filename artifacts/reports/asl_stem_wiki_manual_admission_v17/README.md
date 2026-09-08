# ASL STEM Wiki manual admission audit

The downloaded pool is **not training eligible**. Source notes explicitly map only P13 to Subject 2 and P28 to Subject 7. The appendix classifies Subject 7 as possible L2 signing, so P28 is excluded. The other downloaded participant-to-subject mappings are unresolved, and none of the manual gloss strings proves the frozen ASL-LEX visual variant.

- Fully decoded videos: 98/98
- Signer/gloss review pairs: 71
- Pairs with a pseudo-position proposal: 33
- Queue statuses: {'l2_excluded': 10, 'source_review_unclassified': 11, 'subject_mapping_unresolved': 50}
- Eligible verified spans: 0
- Training or protected evaluation run: no

Complete `expert_review_queue.csv` by entering signer quality, exact variant, and verified start/end frames. Pseudo positions are review aids only. A later admission step may build short participant-disjoint clips only from rows with all three approvals.

## Review UI

Run from the repository root:

```bash
venv/bin/python scripts/review_asl_stem_wiki_manual_v17.py
```

The local page shows the expected raw gloss and ASL-LEX code, the source sentence,
and matching Citizen training-split references side by side. Use the timeline or frame
buttons to set inclusive start/end bounds, record signer and variant decisions, then
choose **Save** or **Save & next**. Saves update `expert_review_queue.csv` atomically.
Rows become training eligible only after signer, variant, and valid boundary approval;
source-excluded L2 rows remain ineligible.

When `artifacts/reports/asl_stem_wiki_auto_annotation_v17/annotations.json` exists,
the page also shows the accepted Reel/Stage-2 model's proposed span and agreement
evidence. The automatic filters expose its 40 review proposals and 10 abstentions.
Copying a proposed span does not save it or approve signer/variant decisions.
