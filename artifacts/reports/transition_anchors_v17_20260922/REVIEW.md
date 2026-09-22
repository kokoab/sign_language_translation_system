# Transition-anchor comparison: completed, no promotion

All four runs completed three epochs. Verified checkpoint hashes, earliest-best selection
including epoch0, saved validation metrics and per-epoch4264single/536phrase exposure.

| Seed | CTC-only phrase WER | Anchored phrase WER | Exact phrases: control / anchored |
|---|---:|---:|---:|
|17421|50.98%|50.98%|29 /29 of211|
|17422|52.41%|52.41%|24 /22 of211|

Seed17421 selected the unchanged epoch0 checkpoint in both arms. Seed17422 selected control
epoch2 versus anchoredepoch1; anchored has worse source-balanced selection score and fewer
exact phrases despite equal aggregate phraseWER. All-source validationWER: control27.56/
31.25%, anchored27.56/31.47%. These include1663single-sign examples and are not continuous-only.

Conclusion: this bounded frozen-encoder, three-epoch,0.25interval-loss trial did not improve
recognition over equal-update CTC-only controls. It does not show that transition learning
is impossible, or identify whether strength/coverage/representation limits are responsible.
No live promotion or automatic extra epochs. Preserve the failed result; next decision
must reconsider the intervention and existing richer sequence supervision, not claim success
from a lower auxiliary loss. No new training was launched for this review.
