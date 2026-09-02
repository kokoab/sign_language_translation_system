# Live v17 Stage-2 CTC local phrase diagnostic

The separate `scripts/live_stage2_ctc_v17.py` path reuses the accepted v17 general
CTC selector through its parity-validated Core ML frozen encoder, primary head, and
specialist head. It consumes non-overlapping 32-frame continuous windows and updates
one growing greedy CTC sequence; it does not use the failed overlapping isolated-sign
window heuristic.

| Familiar development phrase | Final hypothesis | Exact |
| --- | --- | ---: |
| GOOD MORNING | `GOOD MORNING` | yes |
| HELLO HOW YOU | `HELLO HOW YOU` | yes |
| MY NAME | `MY NAME` | yes |
| THANKYOU FRIEND | `THANKYOU FRIEND` | yes |
| TOMORROW SCHOOL GO | `TOMORROW SCHOOL GO` | yes |

Across 16 accepted sequence updates, post-landmark inference latency was 235.77 ms
median, 291.89 ms p90, and 351.34 ms maximum. This includes hand-image embedding,
the frozen multimodal encoder, and both CTC heads, but not the Apple Vision work
amortized over incoming camera frames. The CTC selector itself took roughly 3-5 ms;
hand-image embedding remained the dominant per-update cost.

`GOOD MORNING` demonstrates why the UI distinguishes a live hypothesis from spoken
stable output: its first window predicted `THANKYOU`, the second revised to `GOOD`,
and the final tail produced `GOOD MORNING`. Stable-prefix speech withheld the unstable
first prediction.

The saved-video FINISH smoke used `MY NAME`, produced one logged utterance, and rendered
`My name.` through the deterministic literal path. Webcam FINISH defaults to the
promoted tiny Stage-3 model and native local speech.

These five recordings are familiar local development/reference data and were already
inside the Stage-2 development domain. The 5/5 result proves execution and a material
improvement over the failed live overlap heuristic; it is not independent signer or
novel-phrase accuracy. No sealed test, Citizen test, SemLex test, local test,
2M-Flores `devtest`, or consumed RIT example was accessed.

Run the webcam experiment with:

```bash
venv/bin/python scripts/live_stage2_ctc_v17.py
```

Press **RESET** to clear only the visible current utterance and **FINISH** to close the
current CTC tail, naturalize the gloss sequence, and speak the finished sentence.
