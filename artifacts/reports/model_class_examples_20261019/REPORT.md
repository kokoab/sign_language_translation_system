# One known-training video per ATLAS class — 2026-10-19 PHT

**Later primary re-audit:** All100actual copied videos were freshly decoded and
directly classified by both the deployed primary output and pinned PyTorch primary:
100/100expected classes in each backend. ASK/YOU distinction and visual-review limits:
`../model_class_examples_reaudit_20261019/REPORT.md`. Original streaming evidence below
is preserved; these are different verification procedures.

Created `data/local/model_class_examples/`: **100 original MP4 videos, one per locked
class, no subfolders or extra files, 64,874,681 bytes (64.9 MB)**. Names are canonical
class labels, such as `I.mp4`, `HELLO.mp4`, and `THANKYOU.mp4`. Original encoding,
resolution, duration and pixels are preserved. Every destination SHA256 equals its
training source SHA256; source recordings were copied, not moved.

All 100 selections are ASL Citizen **official train** clips. Citizen covered every
class, so no ASL-LEX/local fallback or new acquisition was needed. ASL-LEX codes are
identity pins, not a claim that these videos came from ASL-LEX. Candidate admission
checked exact canonical label, class index, raw gloss, ASL-LEX code, provenance hash,
training path, finite extraction diagnostics and the existing rejection list. All100
indices/labels match the bundled recognizer. Initial eligible pool:1,475videos.

## Recognition evidence

Each delivered video gave exactly its expected single word using the current bundled
ATLAS FP16 ConfidenceDual recognizer, FP16 hand encoder and boundary models, native
Swift/Apple Vision on macOS, compute units `.all`, and EOF flushing. Finish-gesture
detection was off. Standalone verification resets Vision extraction state for each
independent input file. **94 exact outputs are in `final_03_replay.json`; the six
replacements passed in `repair_03_replay.json`.** `independent_verification.json` maps
all100delivered source files to their exact output and protocol/harness hash. No
recognition gaps remain under this standalone protocol.

Earlier shared-engine checks passed90/100, then10/10alternatives, but subsequent
combined checks gave95/100,97/100and97/100with changing failures. The canonical harness
retains Vision's eight-frame auxiliary cadence across resets; those shared-stream
failures are retained, not overwritten. To evaluate standalone files consistently,
`native_independent/` contains report-only copies of the engine and harness. The only
behavioral changes are making the Vision binding mutable and constructing a new
`LiveVision` after each input-file reset. The extractor implementation, model weights,
decoder, thresholds and canonical application sources are unchanged. Existing Swift
unused-value compiler warnings persisted. CoreML needed approved access outside the
sandbox; stale harness-resource links were avoided by using canonical app model paths.

These are selected known-training demonstrations, **not independent accuracy or a
promise of identical outputs for every live-camera/playback order**. No official test
or validation video was selected or decoded. No training, checkpoint selection,
threshold change, dataset deletion, acquisition or phone installation occurred.

## Files and verification

- `manifest.csv` and `manifest.json`: filename, source path/hash, signer, class identity
  and per-video recognition evidence; metadata stays outside the flat video folder.
- `candidate_pools.json`, `selection.json`, `attempts.json`: selection provenance;
  125candidate attempts, with failures preserved, plus complete-set diagnostic replays.
- `source_model_hashes.json`:34canonical source/model/provenance/configuration files;
  unchanged after completion. Protocol files pin the independent harness binary.
- `select_examples.py`: candidate admission, native replay, selection, copying and
  SHA256 assertions. `finish` requires exact independent evidence for every selected
  source before copying and refuses to overwrite a populated destination.

Final checks: exactly100regular MP4 files, zero subfolders,100unique class indices,
100exact standalone hypotheses, all source/destination hashes equal, all34canonical
source/model/configuration/provenance hashes unchanged. No model-accuracy test suite
was rerun for this data-copy task.

Next safe action: use these files as familiar examples. Fresh phone-camera playback
or signer generalization would require separate verification; retain training/test gates.
