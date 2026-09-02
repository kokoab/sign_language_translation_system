# v17 live isolated-sign prototype

This laptop diagnostic recognizes the locked 100-gloss vocabulary with Apple Vision.
It is pause-delimited Stage 1, not continuous Stage 2.

## Run it

Hybrid mode is the default. It runs the fast unified Stage-1 model first and only reads
real mouth/lower-face pixels when GOOD and THANKYOU are the top-two candidates:

```bash
venv/bin/python scripts/live_isolated_v17.py --camera 0
```

The live window has two clickable modes. `DEFAULT` is the unified-first hybrid above.
`CASCADE` runs the landmark-only orientation model first; when its softmax score is
below 0.70, it falls back to the unified landmark+hand model. The existing targeted
GOOD/THANKYOU visual and YOU/NEED landmark rerankers remain available. The selected
mode is snapshotted when a clip is submitted, so a click while classification is
running changes the next sign. Start directly in the experimental mode with:

```bash
venv/bin/python scripts/live_isolated_v17.py --camera 0 --mode cascade
```

To disable that visual tie-break and use only the unified landmark+hand model:

```bash
venv/bin/python scripts/live_isolated_v17.py --camera 0 --mode fast
```

Test one saved isolated video without opening the webcam:

```bash
venv/bin/python scripts/live_isolated_v17.py \
  --video path/to/GLOSS/video.mp4 --no-display --no-speech
```

Model warm-up happens before the camera opens and can take several seconds. Then:

1. Hold a neutral upper-body rest pose briefly.
2. Sign one vocabulary item at normal speed.
3. Return to approximately the same rest pose for about 0.2 seconds.

The window shows hand/face detection quality, wrist motion, automatic-boundary
progress, top-three model scores, the current utterance's gloss buffer at the bottom,
and the active mode. Click `DEFAULT` or `CASCADE` to switch. Click `RESET` (or press
`R`) to clear the on-screen utterance without deleting its logs. Click `FINISH` (or
press `F`) to wait for an in-flight sign, rephrase the accepted gloss buffer with the
local fine-tuned 15.6M-parameter T5 checkpoint, and speak the finished sentence. Space only closes the current
isolated-sign buffer; `Q` quits. Accepted glosses are also spoken individually with one
queued native macOS speech synthesizer immediately after the matching HUD update unless
`--no-speech` is passed. Low-score,
near-tie, overlong, and insufficient-hand clips are shown as `UNKNOWN` and are not
spoken.

The tiny model is loaded on a separate worker and defaults to CPU so it does not contend
with the live Stage-1 MPS workload. Exact reviewed phrases bypass generation, and model
errors fall back to literal gloss-preserving English. Use `--naturalizer literal` for
deterministic rendering only, or `--naturalizer ollama` to compare the former
`llama3.2:1b` path. RESET invalidates an older pending result so it is logged but cannot
reappear or speak after the display has been cleared.

Every session creates `history.json` and a 640-pixel-wide, 15 FPS reference MP4 under
`artifacts/reports/live_isolated_v17/<timestamp>/`. Scores are uncalibrated model
scores, not probabilities of correctness. Current development defaults require a 0.25
top score, a 0.08 margin, and a clip no longer than 2.5 seconds. These are protective
prototype gates, not a calibrated open-set recognizer.

`history.json` retains predictions across RESET and records reset/finish events, raw
gloss buffers, naturalizer decisions, fallback decisions, model latency,
the finished sentence, and speech queue/start timestamps. FINISH consumes the current
buffer, so the next accepted sign starts a new utterance.

Saved `--video` inputs are treated as one isolated clip and classified once at EOF.
Automatic motion/neutral segmentation is used only for the live camera path.

Live extraction targets 30 processed FPS. Hands and face are requested on every
processed frame so lip movement remains continuously visible as landmarks; body is
requested every eight frames. To preserve the Stage-1 training contract, only every
eighth detected face is inserted into the classifier's landmark tensor. The intervening
face observations are for live display, tracking quality, and optional targeted visual
evidence—not extra training-distribution inputs. Crop frames remain 1280 pixels and are
motion-trimmed before the fixed 32-frame resample. Vision detection uses a same-aspect
720-pixel copy for lower latency; normalized coordinates are applied to the crop frame
without distortion. Vision pauses while the post-clip Core ML classifier runs so the
two workloads do not contend for the same hardware. Hand/body bones are black and every
visible landmark point, including the face/lip points, is white.

On one 90-frame Mac development replay, hands+face every frame with body every eight
frames measured 18.44 ms median and 26.70 ms p90 extraction, below the 33.33 ms
per-frame budget for 30 FPS. This is not an iPhone performance claim. The full measured
result and the validation-only fast-cascade study are recorded in
`artifacts/reports/live_isolated_v17_fast_cascade_validation_v1/report.json`.
