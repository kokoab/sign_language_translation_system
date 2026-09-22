# Research-informed Reel direction

User asks whether to keep Reel after familiar CTC failure; reports HELLO→MY transition
classified GOODBYE and low-wrist-motion misses. Research/code review only: no training,
thresholds or runtime changed. Exact HELLO→MY event is user-reported, not replay-verified.

## Findings

Reel's classifier consumes richer landmarks, but candidate activation uses wrist_motion
(hand.xy[0] displacement) and classify() trims by this same observation motion. These
are distinct layers: rich classifier input cannot recover a sign the candidate scheduler
never submitted. Current session's learned no-emit is disabled. Class probabilities are
renormalized over known labels, so high confidence does not establish a complete lexical
sign. Repeated overlapping classifications can be consistently wrong on a transition.

Primary research: Zuo etal EMNLP2024 online CSLR uses contextual sign crops, background,
saliency and sliding-window decoding. Table4 reports22.2%devWER,320msalgorithmic and29ms
window processing latency on V100/Phoenix14T. These are not end-to-end ASL/iPhone guarantees;
postprocessing also uses voting. https://arxiv.org/html/2401.05336v2
Hollain etal2023 examine handshape/orientation/location/movement features for spotting;
their results are mixed by property, not a universal replacement for learned landmarks.
https://aclanthology.org/2023.at4ssl-1.1.pdf

Research supports streaming recognition as a legitimate target, not an assertion that
natural-speed arbitrary-sign ASL transcription is solved. Video throughput, algorithmic
lookahead and user-visible stable-commit delay must be reported separately.

## Avoid repeating failed work

Project already tried contextual/background window training. Sep12firstcompletewindow
model:316.20%connectedWER,770insertions and16/16transitionfalseemissions; isolatedretention
alsofailedlaterepochs. Sep14O5S5contextual followup included6819positives/494backgrounds;
prior reportrejectedretention/deletiontradeoff. Recent timed-anchorCE alsofailed to improve
phraseWER. Thus neither 'background class' nor 'temporalhead' is a novel sufficient fix.

## Recommendation (hypothesis to test, not a promised solution)

Keep app shell/Reel UX and current good sign-recognition reference. Develop one controlled
continuous spotting candidate, not another unrelated encoder/decoder swap. Candidate
submission should continue while hands are observable without requiring wrist movement;
preserve holds and finger changes instead of wrist-only trimming. Hand presence schedules
computation, not word emission. Removing the motion gate alone risks more false positives.

Use complete timed signs, partial signs with context, and reviewed real non-lexical
transitions from existing sources. Hard negatives must include the windows actually
confused for known signs, alongside genuine examples of the confused sign (GOODBYE) and
low-motion positives. Never mask all gaps as blank: timing can include sign onset/offset
or unknown lexical content; exclude uncertainty. Use existing raw parent context before
random-noise bridges. Review clip membership/schema/roles for any new recipe.

A learned emission decision conditioned on a short visual history can withhold transition
candidates; allow small bounded delay without requiring signer pauses. Preserve real sign
repetitions; don't suppress all repeated labels. Initial comparison keeps trusted encoder
frozen, then unfreezes only if diagnostics prove its evidence insufficient. Because head-only
work has failed before, validate this on a small audited transition/hold set before training
at scale. Test known combinations withheld from fitting, single signs, fast transitions,
repeats, genuineGOODBYE and stationary-wrist signs. Count extra words and misses separately;
measure end-of-sign-to-stable-output delay, not only isolatedaccuracy or repeatedphraseWER.

First necessary work is replayable recording and per-window base/head traces in the app,
then a same-input failure study. Current candidate logs lack these, and latest Reel video
was unreadable. No request for broad new acquisition; use existing data plus normal live
sessions. Do not add language-prior completion as a fix for false transition signs.
