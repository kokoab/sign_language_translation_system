# Streaming-window evaluation on nine genuine local phrases

The separate `live_streaming_v17.py` path was run unchanged on the nine representative
project-owned local phrase recordings selected by the genuine-motion reference audit.
This is a development/train-source diagnostic, not an independent signer-disjoint test.

## Result

| Phrase | Target | OOV glosses | Emitted streaming buffer |
| --- | --- | --- | --- |
| GOOD_MORNING | GOOD MORNING | — | EAT MORNING |
| HELLO_HOW_YOU | HELLO HOW YOU | — | HELLO HOW |
| I_WANT_FOOD | I WANT FOOD | FOOD | UNDERSTAND WANT |
| MY_NAME | MY NAME | — | NAME |
| PLEASE_HELP_ME | PLEASE HELP ME | ME | STOP HELP I |
| SORRY_I_LATE | SORRY I LATE | LATE | SORRY |
| THANKYOU_FRIEND | THANKYOU FRIEND | — | BAD NAME DOCTOR |
| TOMORROW_SCHOOL_GO | TOMORROW SCHOOL GO | — | TOMORROW LESS |
| YESTERDAY_TEACHER_MEET | YESTERDAY TEACHER MEET | TEACHER, MEET | ANSWER |

- Exact sequences: 0/9 overall.
- Exact sequences among the five fully vocabulary-covered phrases: 0/5.
- Windows processed/accepted: 97/69.
- Median window-classification latency: 548.88 ms; p90: 1,334.46 ms.
- No Citizen validation/test, SemLex test, or local test partition was accessed.

The mechanics work: every video completed, histories and low-resolution videos were
written, windows did not queue, and stabilization suppressed repeated overlap outputs.
The accuracy gate fails. The isolated Stage-1 classifier sometimes finds correct
components, but fixed 1.2-second windows include partial signs and transitions and often
miss short signs before two-window agreement. Four phrases also contain unavailable
classes, so full recovery is impossible under the current locked-100 vocabulary.

Do not replace the pause-delimited live path with this configuration. The next
experiment should use a learned framewise/boundary model or Stage-2 sequence decoder,
not merely lower the stabilization threshold; accepting every window would increase
transition hallucinations and duplicates.

Per-window scores, timings, model provenance, and videos are retained under `runs/`.
