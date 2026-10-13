# Manuscript Chapter 4 reorganization — 2026-10-13

Applied only to Manuscript tab `t.d1buvx4fb1j1` in the ATLAS Google Doc.

- Original 4.1 and 4.2 requirements, SDLC, Activity List, GANTT, PERT and design blocks compare exactly after ignoring shifted indexes.
- Other tab bodies compare exactly, including Page Check.
- Participant provisions preserved verbatim and relocated to 4.5.5.
- Existing development methods follow the design discussion in 4.2.
- 4.3: prototype; 4.4: implementation plan; 4.5: implementation results.
- Original tables and seven figure assets carried over; captions, references and contents lists renumbered.
- Native deployment resource and risk matrices inserted; resource matrix split for page width.
- Missing budgets, module durations, evaluator assignments and software-quality findings remain pending.
- No model measurements, datasets, training, production app or device installation changed.

Backups: before.json and original figure_*.png. PDF layout checks: checked.pdf, checked99.png, checked100.png; contents page numbers subsequently synchronized. See page_map.json.

A Docs request initially failed because pageBreakBefore cannot be applied across table paragraphs; removed that field from the broad style request. Contents-range replacement initially included a paragraph newline before a section break; corrected the range. Both failed requests were atomic and made no changes. Removed four internal placement notes caught during layout QA, corrected inherited indentation and fitted tables to page margins. Resource/risk captions now start with their tables.

Next safe action: user review of Manuscript, then supply missing implementation resource and evaluation records. Do not infer these from model metrics.

Final read-back: all seven original tables and all planned prose present; contents pages resolved. Final artifact: final_verified.pdf. Numeric page references verified against final PDF. Structured checks: verification.json.
