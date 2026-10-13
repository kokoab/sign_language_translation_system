# Streaming translation experiment

This directory is independent of the ATLAS source tree. It replays recorded
recognition events through local English models. No app, recognizer, datasets,
checkpoint or deployment configuration is modified. No training has run.

Read `PLAN.md` for the frozen comparison and transparent amendments. Results and
interpretation are in `REPORT.md`; `outputs.csv` and `traces.csv` expose predictions
and the complete caption timeline.

From repository root:

```sh
venv/bin/python experiments/streaming_translation/prepare.py
venv/bin/python experiments/streaming_translation/replay.py selfcheck
venv/bin/python experiments/streaming_translation/replay.py tiny
venv/bin/python experiments/streaming_translation/replay.py flan
venv/bin/python experiments/streaming_translation/replay.py flan_fewshot
venv/bin/python experiments/streaming_translation/fetch_qwen.py
venv/bin/python experiments/streaming_translation/replay.py qwen
venv/bin/python experiments/streaming_translation/replay.py qwen_direct
venv/bin/python experiments/streaming_translation/analyze.py
venv/bin/python experiments/streaming_translation/verify.py
```

`fetch_qwen.py` is the only network step. It fetches a pinned public checkpoint;
all translation uses local weights with `HF_HUB_OFFLINE=1`. The two T5 weights
must already exist at the paths named in `replay.py`. Model downloads are ignored
by git. Existing generations are cached per model condition, input and confidence
buckets: cache replay reproduces stored timing, not a fresh benchmark. For a fresh
timing run, move that condition's JSONL cache aside and rerun. If changing the prompt,
weights or decoding configuration, use a new condition/cache name.

52 cases:26 historical phone utterances,24 declared text diagnostics,2 synthetic
gap stress cases. Only20 phone utterances have complete causal word-event traces.
The six Finish-dependent spelling cases remain in full-input evaluation but cannot
support a claim of automatic live spelling from these records. The historical phone
logs cover six sessions within a previously reviewed12-session inventory.

Reference matching ignores case and punctuation, and normalizes don't/do not.
It is a restricted reproducibility statistic, not a semantic accuracy score.
Some listed acceptable translations are not exhaustive. Explicit role-form
diagnostics declare a statement interpretation; the gloss stream alone cannot
determine all natural ASL question/statement distinctions. Ambiguous cases have
no unique gold output. No artificial example is represented as a recorded video.

Policy timing is computed from original recognition arrival timestamps with measured
single-request desktop generation costs added on a serial queue. Duplicate inputs
reuse measured generation cost. No camera workload, mobile export, coalescing or
real speech synthesis is included. The Finish baseline assumes an oracle endpoint
at the last arriving word; actual Finish times were not logged. Neither visible
draft latency nor agreement with full-input output demonstrates semantic correctness.

Primary model documentation:
- https://huggingface.co/google/flan-t5-base
- https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
