# Approved post-defense revisions applied — 2026-10-28 PHT

Existing Google Doc: https://docs.google.com/document/d/1i-sVr5ny6vmf6QDb_Dbr1e_CA3_xH7y7s68-wVqaSzc
Edited only Revised Manuscript (`t.npvgh2mbwuhs`). No new document created.
Source: `docs/capstone_papers/POST_DEFENSE_BATIANCELA_REVIEW.md`, version 3.

## Applied scope

- P-17: Objective 2 in 1.3; P-01: Finish gesture in 3.2.
- P-02: 4.5 Flowchart, navigation caption renamed and actual pipeline image inserted as Figure 9.
- P-04/P-05: Evaluation Metrics and Parameter Selection before 4.6, native Table 11 and metric definitions.
- P-08/P-09: 4.7.2 precision procedure and device timing method.
- P-11: 4.8 opening; P-19: development-implementation wording in 4.8.2 and Figure 18 caption.
- P-13/P-21: new 4.8.3, native Table 18, WER-gap paragraph immediately after table, followed by approved results summary.
- P-14: old 4.8.4 replaced, now 4.8.5, native Table 20; old 31.14 ms passage removed.
- P-15: new 4.8.6 calibration section and actual reliability image as Figure 19.
- P-16: 5.1 results paragraph; P-20: 5.2 timing recommendation.
- P-18: Guo, Naeini and Zuo references inserted alphabetically with italic venue text and links.
- Renumbered original captions and references, contents, figure/table lists, and shifted 4.8 subsections. New quoted wording kept at final numbers.

## Verification

`verification.json`: every approved quoted paragraph appears exactly once; native table cell values match source exactly; captions are Figures 1–19 and Tables 1–28; no 31.14 ms occurrence remains. Other tabs compare identically after excluding refreshed temporary image contentUri fields. All 125 body-target navigation page references match the final PDF. Three front-matter references retain ii/iv/v.

Exported the existing all-tab document through authenticated streaming download (about 15 MB, beyond files.export limit). Final export is `artifacts/generated/post_defense_doc_qa_20261027/final.pdf`; read only relevant rendered pages. Pipeline image resized proportionally to fit one page; calibration image/caption rendered; native tables have header rows and repeating headers, peer typography and horizontal rules. Table 18 kept together. Condensed contents paragraph spacing to keep its entries on two pages. Source text and images preserved.

## Corrections and transient failures

A temporary duplicate P-19 paragraph introduced by quote-block parsing was removed before final verification. Google rejected two formatting batches atomically: pageBreakBefore cannot be changed inside tables; tabStops could not be set through updateParagraphStyle. Retried with valid scoped fields. One final Docs read hit a shared project read quota (429); later retry succeeded. Initial automatic page matching found prose/table-cell mentions of short headings; replaced with line-anchored matching in native document order, corrected eight affected entries and verified zero mismatches. No outstanding edit or image placeholder.

Connector tools became available in this extension during this session after earlier absence; do not infer that the extension categorically cannot use them. No training, model, dataset or app changes.
