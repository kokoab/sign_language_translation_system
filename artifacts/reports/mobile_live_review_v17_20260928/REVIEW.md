# iPhone live review — 2026-09-28

Discussion-only inspection; no app, decoder, model, or deployment change.
Copied the single saved Live session from connected iPhone 13 (bundle
com.kokoab.sltMobileApp), Documents/live_reel_sessions/20260928_224024.json.
Source code inspected in /Volumes/secret/SLT/mobile_app/slt_mobile_app.

## Saved evidence

- Session starts 22:40:24 PHT, duration 76.38 s. Eleven output events, three resets,
  no Finish/Stop/sentence events. Outputs: fs-CB HOW YOU / fs-CB HOW TAKE /
  PLEASE MY MY PLEASE HELLO. Slashes separate presentation groups, not annotated phrases.
- Both CB outputs contain FS_C and FS_B, with aggregate scores .967 and .889;
  they are false spelling according to the user's HELLO report. Scores are not
  calibrated accuracy. Later HELLO scores .990. No video, observations, rejected
  candidates or raw logits are saved, so exact model attribution is unproven.
- Both spelling runs are emitted with HOW. They appear 2.20 and 1.80 s after
  their stored span ends. The stored commit_seconds predates actual release.
- Non-spelling span frame/time differences imply about 6.3–18.4 processed frames/s
  locally (mostly 6–11), not a whole-session FPS measurement. Ordinary outputs
  arrive .47–1.57 s after stored span ends, before adding current-call compute.
  These are decoder-estimated boundaries, not human-annotated sign ends.
- Accumulated model-stage timing: hand crops/encode 30.42 s, recognizer 21.67 s,
  boundary 7.03 s, span inputs .27 s. Timing counters survive resets and the engine
  is shared across visits, so these are not certified session-only measurements.

## Code findings

- ViewController captureOutput waits for engine.process (Vision + models + both
  decoders) before dispatching skeleton display. Camera preview advances separately.
  Core.LiveVision detects body/face every eight processed frames; ViewController
  retains last valid body/face without expiry. This explains stepped/stale overlay.
- LiveReelViewController explicitly locks landscape. Its skeleton transform assumes
  mirrored upright frames and aspect-fill; portrait requires interface/layout and
  camera/overlay transform checks, not just a plist change.
- Decoder mode .both scales word probabilities by letter NONE probability when
  a letter passes threshold. A confident false letter can suppress HELLO. The
  second letter decoder also emits letters; this session does not log provenance.
- SpellingBuffer already flushes before the next committed non-letter word.
  Otherwise it waits >2.5 s after last letter commit. Letters cannot early-commit.
  Thus faster letter detection and faster spelling-run release are separate changes.
- Runtime flushes/resets on an observation gap >.26 s. Phone stalls could fragment
  signing; this session lacks gap logs to prove which resets were caused by that.
- No finish gesture in active mobile controller. Desktop classic Reel has a
  two-open-palm hold/latch implementation, currently .4 s default; requested behavior
  is 1.0 s, Live only, one finish per hold, excluding gesture from recognized text.
- History shows only the last sentence or one truncated gloss line, has no detail
  view, joins attempts across Reset, saves only on leaving Live, and labels every
  saved session complete even without Finish. Empty attempts and Practice are not
  saved. Leaving Live resets the engine without flushing pending recognition.

## Proposed next action, pending discussion

First separate display tracking from slow recognition and validate rotation/mirroring
in portrait and landscape on-device. Then diagnose word-versus-letter competition
with timestamped candidate/provenance/gap logs and reviewed HELLO/spelling examples.
Keep a visible provisional spelling run; finalize before the next confirmed sign,
on explicit Finish, or an agreed idle fallback. Restore a one-second open-palm
Finish with progress and latch, and improve history attempt boundaries and timing.
Do not infer overall recognition accuracy from this unannotated session.
