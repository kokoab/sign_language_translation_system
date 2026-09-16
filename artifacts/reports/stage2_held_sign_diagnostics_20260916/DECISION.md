# Decision after the held-sign replays

2026-09-16. No additional model training or runtime changes in this follow-up.

## Evidence sufficient to choose the next comparison

The 40 replay runs completed. The boundary-state correction changed no final
outputs on this selected set. The current model is sensitive to the origin of
the approximately one-second windows: the same EAT clip changes between I EAT EAT,
I EAT, and EAT EAT. The original webcam I I was not reproduced from saved video.

One remaining inexpensive explanation was checked: greedy path collapse versus
sequence-level CTC beam decoding. `check_beam.py` applies the existing decoder
once, with its existing width 8/top-k 12, to all 40 saved offline emission streams.
No length penalty, language model, threshold sweep or training was added.

- Beam changed only 2/40 final outputs.
- EAT and WHEN repetitions were unchanged at every window origin.
- On the 16 labelled phase runs, total edit errors increased from 32 to 34 over
  32 reference tokens. These are four correlated offsets of four selected clips,
  not a generalization benchmark.
- For the phase-zero WHEN example, exact CTC sequence log probability is -6.460
  for WHEN and -0.00621 for WHEN WHEN. The model itself strongly prefers the
  repeated sequence. This is model probability, not calibrated confidence in
  what the signer intended.

The saved-emission greedy outputs were asserted equal to the previous report
before comparing beam. Detailed rows and input hashes are in `beam_control.json`.
This new control targets the newly recorded repetition examples; an earlier
broader beam control had also failed to justify replacing greedy decoding.

## Stage 3's appropriate role

Stage 3 can make an English sentence more natural, including deciding how to
express repetition in context. A duplicate gloss string alone does not tell it
whether there were one or two intended signing events. Globally deleting equal
adjacent labels destroys that distinction. Fluent output also cannot repair the
recognizer choosing different sign identities under a window shift.

Allow contextual wording improvements only as translation behavior, preserve the
raw recognition output for diagnosis, and evaluate fidelity of repeated meaning.
Do not advertise string cleanup as a recognition fix. No repeat-removal rule was
implemented in this follow-up.

## Chosen next step

Stop adding repeat heuristics, beam-search tuning and multi-phase training recipes
to the current windowed selector. The next model experiment is **inference with
one released pose-only Uni-Sign ASL checkpoint**, using its native preprocessing.
It is a separately fingerprinted challenger to the combined Stage-2/Stage-3 path,
not an unvalidated replacement of the locked Apple Vision extractor.

Start with completed utterances and explicit Finish. Evaluate on the current
diagnostic videos and a small fixed development set with paired ASL/English
references. Do not fabricate reference translations from English-looking gloss
strings or use diagnostic webcam observations as new training truth. Existing
ASL checkpoint inference comes before fine-tuning or architecture implementation.

The decision gate is whether that existing model recognizes/translates useful
meaning across signers and is less sensitive to clip boundaries, while retaining
intended repetition. It must justify its additional compute and preprocessing.
If it fails the bounded comparison, do not enlarge or train it automatically.
Human-labelled independent holds/repeats are still required for a reliability
claim; their absence need not block a preliminary architecture comparison.

This comparison has not been launched by this follow-up. The completed work here
is the decision and fixed-decoder control; there is no new background training job.
