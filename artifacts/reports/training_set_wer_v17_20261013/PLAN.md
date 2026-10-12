# Matched training-set WER — 2026-10-13

User requested direct evaluation on training examples. No new training or tuning.
179approved scoped training recordings,495referenceglosses. Fixed historical DGS boundary
and candidate-decoder configuration across preinterval,postinterval,v2andpartialsign
recognizers. Existing checkpoints only; no current-runtime latency claim or held-out result.
Code:evaluate.py. Outputs:results.json (state/completed_recordings),evaluate.log.
No production/manuscript changes. Separate from parallel Transformerexperiment.
Reuse localtrain recordings intersected with approved scoped training video hashes;
assert no overlap with tune/test membership before loading any inputs. Modelweights,
manifest and usedcache hashes recorded. Complete data, not selected successfulclips.
Latercheckpoint comparisons bundle trainingdata/recipe changes; sameevaluation doesnot
make upstream training differences causal one-factor ablations. Transition selective
runtime policy is not implemented by this fixed historical decoder and is not claimed.

Correction: existing training memos lack full decoder candidate coverage. Initial
partial results are invalid and archived. evaluate.py now prepares missing windows
into report-local complete_spans caches before inference, with coverage assertions.
Six checkpoints include transition-start and transition-fine-tuned controls; this is
not the deployed selective transition runtime. Active log: evaluate_complete.log.
