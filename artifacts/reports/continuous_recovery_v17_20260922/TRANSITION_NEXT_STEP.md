# Transition intervention decision — 2026-09-22

User requests faster progress and explicit transition learning. First complete the bounded
matched diagnostic using current checkpoints and approved cached inputs, without extraction,
training or live changes. This is already authorized, not a request for another approval.

The combined run used full-sequence CTC and isolated identity CE, but no explicit timed
transition term. Prior frame-aligned and standalone-blank experiments already existed;
adding generic blank samples again is not a new hypothesis. The candidate intervention
must target the failure observed with these current weights and this cleaned data.

Decision rule: distinguish contextual sign identity evidence from decoded event output.
If identities survive but CTC misses/adds events, evaluate an explicit timed core/transition
objective against an equal-update CTC-only control. If identities themselves fail, adding
blank pressure alone is unsupported; correct the temporal representation/identity objective.
A CTC spike outside a sign interval is not automatically a transition hallucination.

Candidate intervals: transition_gap_candidates.json has28training and5validation internal
gaps between consecutive eligible known annotations, over0.1s, with no intervening OTHER
annotation. These are candidates, not yet approved framewise blank supervision. Unknown
signs, clip padding, ambiguous boundaries and local recordings without timings stay ignored.
No historical excluded source may be reintroduced through this diagnostic.

Any next training must pin a new recipe/input contract, preserve current Stage1 retention
measurements and compare to an equal-update control; old completed-run pins stay unchanged.
No full retraining, threshold sweep or live promotion follows merely from this document.
