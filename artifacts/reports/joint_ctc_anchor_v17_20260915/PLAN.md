# Verified-anchor CTC repair

The preceding direct-positive CTC run completed 12 epochs: 1,780/1,901 isolated
training clips decode correctly, but only 183/1,291 continuous training signs match.
It escapes isolated blank collapse but does not fit continuous alignment adequately.
No development evaluation of that correction has informed this recipe.

Use the same original initialization, seed 17111, 12 epochs, 110 updates, data,
coverage, AdamW learning rates and architecture. Retain all prior positive CTC,
pooled CE/KL, complete-sequence CTC and verified-background losses.
Add CE on one output token nearest each verified event midpoint, only inside its
interval and outside overlaps with any event. Skip events with no unambiguous token.
No unverified frame gets a CE label. Known and OTHER event CE sums are divided by their full training populations within
each source, giving equal epoch-level means independent of minibatch composition;
complete-sequence source groups retain their existing equal weights. Also add one
midpoint positive CE per isolated replay / exact core, with existing group weights.
Each new anchor term has weight 1 before the same source/epoch normalization.

This directly addresses the measured alignment basin: marginalized CTC likelihood
can improve while every frame still prefers blank. Timed annotations already exist;
use them to supervise emission locations. Preserve greedy decoding and all 102 outputs.
Measure complete training recognition every epoch, then evaluate the final candidate
on the frozen development suite, pooled/CTC isolated, LG cores and gap emissions.
Keep the official Citizen test sealed. No promotion without existing gates.
Verify provenance, focused tests, and CPU/MPS outputs; record measured limitations.
