# Confident-only continuous supervision

This derived manifest leaves all source data untouched. It accepts only complete, non-overlapping annotations that passed the earlier raw-clock and both-edge checks, then requires at least six raw target observations, at least 80% target-frame hand visibility, no target timestamp gap above 80 ms, and duration no longer than 1.20 seconds.

Model predictions were not used for filtering. Difficult valid signs therefore remain. Annotation gaps are not targets. O5S5 provides positive event cores and edges only; its unannotated surrounding regions cannot provide negative/background supervision.

Accepted **5,331/11,936** annotations: **1,017 known** and **4,314 unknown**. Continuous training covers **61/100 classes**; 17 still have one continuous training signer.

| Split/source/type | Accepted |
| --- | ---: |
| train / asllrp_contiguous / known | 47 |
| train / asllrp_other_ctc / known | 668 |
| train / asllrp_other_ctc / unknown | 2,770 |
| train / o5s5 / known | 76 |
| train / o5s5 / unknown | 742 |
| validation / asllrp_contiguous / known | 11 |
| validation / asllrp_other_ctc / known | 181 |
| validation / asllrp_other_ctc / unknown | 577 |
| validation / o5s5 / known | 34 |
| validation / o5s5 / unknown | 225 |

`confident_supervision.json` contains every accepted row and a decision with exclusion reasons for every source annotation. It is the only continuous supervision manifest recommended for the next controlled experiment.

This is mechanical confidence in crop, timing, sampling and visibility. It is not a claim that every gloss or linguistic boundary was independently re-annotated by an ASL expert. Citizen test remained sealed.
