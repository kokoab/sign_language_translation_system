# Live gloss-to-English caption engine

Experiment code and evidence. The desktop app uses the packaged runtime
`scripts/live_captions_v17.py` with models in `artifacts/models/live_captions_v17_20261014/`
(rebuild with `export_runtime.py`). Nothing in `ATLAS/` imports it or is changed by it.
The runtime uses context 0 (open clause only, chosen on validation by
`context_select.py`); pass `context=0` to `LiveCaptioner` here to match it.
Results: `REPORT_V2.md` (fixed engine), `REPORT.md` (first tagger), `RESEARCH.md`.

```python
from live_caption import LiveCaptioner
eng = LiveCaptioner(end_s=3.5, draft_mode='no_post', context=0,
                    live_path=['tagger_k1.pt', 'tagger_k1b.pt'])
r = eng.run(['I', 'GO', 'HOSPITAL', 'TOMORROW', 'MORNING'], [0.9] * 5,
            [0.0, 1.0, 2.0, 3.0, 4.0])
for e in r['events']:
    print(e['at'], e['kind'], e['text'])
```

For a live integration, keep one `state = {'frozen': [], 'open': 0}` per
conversation. Call `eng.step(state, glosses, scores, False)` on every recognized
item, and `eng.step(state, glosses, scores, True, tail_period=False)` once no sign
has arrived for `end_s` seconds. Text in `state['frozen']` never changes.

Behavior:
- The newest sign appears at once, as a draft without guessed punctuation.
- A sign is fixed when the next sign arrives.
- When a clause closes, it is rewritten once, and only to fix meaning: moving a WH
  sign or NO, restoring negation, or a tense that contradicts its time sign.
- Signs are never dropped when recognized with confidence ≥ 0.6, and words for
  unsigned glosses are never added.
- Pronouns and here/there are inserted only when the model is at least 50% sure.

Rebuild from the corpus (read-only input):

```sh
P=../../../venv/bin/python
$P build_data.py && $P tagger.py prepare
./run_training.sh            # k=1, k=0, k=2 (full-context: $P tagger.py full 3)
$P add_boundaries.py && $P tagger.py 1 3 b
$P finalizer.py build && $P finalizer.py mover 4 && $P finalizer.py labeler 4
$P evaluate_v2.py && $P verify.py
```

`data*.json`, `finalizer_data.json` and `*.pt` are git-ignored. Hashes are in
`integrity.json`.
