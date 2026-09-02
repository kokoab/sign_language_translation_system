# Initial motion-valley evaluation on nine local phrases

The first low-motion endpoint configuration used motion ≤0.006 for 0.12 seconds,
without requiring a return to neutral. It was evaluated on the same nine project-owned
development/reference recordings as the overlap path.

| Phrase | Emitted buffer |
| --- | --- |
| GOOD_MORNING | MORNING |
| HELLO_HOW_YOU | YOU |
| I_WANT_FOOD | WANT |
| MY_NAME | MY NAME |
| PLEASE_HELP_ME | *(empty)* |
| SORRY_I_LATE | SORRY |
| THANKYOU_FRIEND | *(empty)* |
| TOMORROW_SCHOOL_GO | TOMORROW GO |
| YESTERDAY_TEACHER_MEET | ANSWER |

It scored 1/9 exact overall and 1/5 among fully vocabulary-covered phrases. Fifteen
clips were formed, nine passed Stage-1 gates, median classification latency was 448.01
ms, and p90 was 529.23 ms. This is cleaner than the overlap output but still fails the
accuracy gate. Per-clip evidence is retained under `runs/`.

No Citizen validation/test, SemLex test, or local test partition was accessed.
