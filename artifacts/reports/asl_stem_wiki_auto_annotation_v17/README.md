# ASL STEM Wiki automatic boundary proposals

The accepted v17 Reel landmark classifier and Stage 2 selector scanned a five-second
radius around each expected token position. The run completed all 71 queue rows across
59 source videos. It wrote proposals only to `annotations.json`; it did not change the
human-review CSV or make any row automatically training eligible.

- Human-reviewed calibration rows: 21
- Strict boundary-and-variant successes: 5 (23.81%)
- Remaining untouched rows: 50
- Review proposals: 40
- Abstentions: 10
- High-confidence automatic approvals: 0

The model often recognizes the target gloss while selecting a rejected Citizen variant
or imprecise boundary. A safe high-confidence threshold could not retain at least three
reviewed successes while excluding every reviewed failure. Treat the displayed score as
uncalibrated evidence and verify the Citizen reference and inclusive bounds manually.

Run the reviewer from the repository root:

```bash
venv/bin/python scripts/review_asl_stem_wiki_manual_v17.py
```

Use **Automatic: needs review** for the 40 usable proposals and **Automatic: abstained**
for the 10 weak cases. **Use automatic bounds** copies a proposal into the editable
fields; nothing is persisted until **Save** or **Save & next** is pressed.
