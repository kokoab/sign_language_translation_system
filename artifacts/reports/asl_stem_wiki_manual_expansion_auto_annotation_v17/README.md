# Expanded ASL STEM Wiki automatic review

The accepted v17 Reel classifier and Stage 2 CTC selector scanned every review pair
within five seconds of its expected sentence position.

- Source videos fully scanned: 154/154
- Participant/gloss review pairs: 206
- Previous human reviews carried by exact identity: 63
- Existing training-eligible spans preserved: 33
- Untouched rows at scan time: 143
- High-evidence proposals: 8
- Review proposals: 112
- Abstentions: 23
- Automatically training-eligible rows: 0

The high-evidence threshold was calibrated against the carried human boundaries and
variant decisions, but it does not replace Citizen-reference review. The UI exposes
separate high-evidence, review and abstention filters. Copying automatic bounds remains
editable and does not persist until the reviewer saves.

Run from the repository root:

```bash
venv/bin/python scripts/review_asl_stem_wiki_manual_v17.py \
  --queue artifacts/reports/asl_stem_wiki_manual_expansion_admission_v17/expert_review_queue.csv \
  --auto-annotations artifacts/reports/asl_stem_wiki_manual_expansion_auto_annotation_v17/annotations.json
```
